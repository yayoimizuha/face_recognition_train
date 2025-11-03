# Analysis Report: Loss Not Decreasing with Increased Batch Size

## Date
2025-11-03

## Environment
- Command: `torchrun --nproc_per_node=8 train_v2_restart.py configs/glint360k_mbv4_hybrid_med.py`
- Number of GPUs: 8
- Config file: `configs/glint360k_mbv4_hybrid_med.py`

## Problem Statement
When increasing batch size in distributed training, the loss fails to decrease properly and accuracy does not improve.

## Root Cause Analysis

### 1. Incorrect Gradient Scaling (Primary Issue)

**Location**: `partial_fc_v2.py`, `DistCrossEntropyFunc.backward()` method (line 227)

```python
def backward(ctx, loss_gradient):
    ...
    batch_size = logits.size(0)
    ...
    logits[index] -= one_hot
    logits.div_(batch_size)  # ← Issue: dividing by local batch size
    return logits * loss_gradient.item(), None
```

**Detailed Explanation**:
- Current implementation divides gradients by **local batch size** (batch size per GPU)
- However, `AllGatherFunc.backward()` (line 260) multiplies gradients by `world_size`:
  ```python
  grad_out *= len(grad_list)  # multiply by world_size
  ```
- This results in an effective gradient scale of:
  ```
  Effective gradient scale = (1 / local_batch_size) × world_size
                           = world_size / local_batch_size
  ```

**Concrete Impact**:

| Configuration | GPUs | Local BS | Global BS | Effective Scale | Expected Scale | Error |
|---------------|------|----------|-----------|-----------------|----------------|-------|
| Single GPU | 1 | 128 | 128 | 1/128 | 1/128 | Correct |
| 8 GPUs | 8 | 128 | 1024 | 8/128 = 1/16 | 1/1024 | **64x too large** |
| 8 GPUs (larger) | 8 | 256 | 2048 | 8/256 = 1/32 | 1/2048 | **64x too large** |

**Conclusion**: Even when increasing batch size, the effective gradient scale is not properly adjusted, preventing proper learning.

### 2. Learning Rate Scaling Issue

**Current Configuration**:
```python
# config
config.batch_size = 128  # local batch size
config.lr = 0.1

# train_v2_restart.py (lines 187-189)
cfg.total_batch_size = cfg.batch_size * world_size  # = 128 * 8 = 1024
```

**Problems**:
1. **Linear Scaling Rule** not applied
   - Theory: When batch size increases by N×, learning rate should also increase by N×
   - Current: Batch size 8× (128→1024) but learning rate fixed at 0.1
   
2. **AdamW optimizer considerations**
   ```python
   config.optimizer = "adamw"
   config.adam_betas = (0.9, 0.99)
   ```
   - AdamW uses adaptive learning rates, making it more sensitive to batch size changes
   - Larger batch sizes require more careful learning rate tuning

### 3. Warmup Epoch Too Small

```python
config.warmup_epoch = 0.04  # Only 0.04 epochs
```

**Issue**:
- With global batch size 1024, warmup_step ≈ 667 steps
- For large-scale dataset (17M images), warmup period is extremely short
- Can cause instability during early training

## Recommended Solutions

### Priority 1: Fix Gradient Scaling (Critical)

Modify `partial_fc_v2.py`, `DistCrossEntropyFunc.backward()`:

```python
@staticmethod
def backward(ctx, loss_gradient):
    (index, logits, label,) = ctx.saved_tensors
    batch_size = logits.size(0)
    world_size = distributed.get_world_size()
    
    one_hot = torch.zeros(
        size=[index.size(0), logits.size(1)], 
        device=logits.device, 
        dtype=logits.dtype
    )
    one_hot.scatter_(1, label[index], 1.0)
    logits[index] -= one_hot
    
    # Fix: divide by global batch size
    global_batch_size = batch_size * world_size
    logits.div_(global_batch_size)
    
    return logits * loss_gradient.item(), None
```

**Important**: This fix, combined with `AllGatherFunc.backward()`'s `grad_out *= len(grad_list)`, produces correct scaling:
```
Effective gradient scale = (1 / (local_batch_size × world_size)) × world_size
                         = 1 / (local_batch_size × world_size)
                         = 1 / global_batch_size  ✓ Correct
```

### Priority 2: Apply Linear Scaling Rule for Learning Rate

Modify `configs/glint360k_mbv4_hybrid_med.py`:

```python
# Base learning rate (assuming reference batch size 256)
base_lr = 0.1
reference_batch_size = 256
config.batch_size = 128

# Linear Scaling Rule: lr = base_lr * (actual_batch_size / reference_batch_size)
# With 8 GPUs: global_batch_size = 128 * 8 = 1024
# scaled_lr = 0.1 * (1024 / 256) = 0.4
config.lr = base_lr * (config.batch_size * 8) / reference_batch_size
```

### Priority 3: Increase Warmup Epochs

```python
config.warmup_epoch = 1.0  # Increase from 0.04 to 1.0
```

**Rationale**:
- More stable training initialization
- Larger batch sizes benefit from longer warmup periods

### Priority 4: Adjust AdamW Parameters

```python
# Use more conservative beta2
config.adam_betas = (0.9, 0.95)  # Was: (0.9, 0.99)

# Or adjust weight_decay
config.weight_decay = 5e-5  # Was: 1e-4
```

## Verification Steps

### 1. Confirm Fix Effectiveness

```bash
# Baseline (before fix)
torchrun --nproc_per_node=8 train_v2_restart.py configs/glint360k_mbv4_hybrid_med.py

# After fix
# Apply changes to partial_fc_v2.py, then run same command
```

### 2. Compare Learning Curves

- Check if loss curve decreases properly
- Verify validation accuracy improves
- Assess stability during warmup period

### 3. Experiments with Different Batch Sizes

| Config | GPUs | Local BS | Global BS | LR | Expected Result |
|--------|------|----------|-----------|-----|-----------------|
| Small | 8 | 64 | 512 | 0.2 | Baseline |
| Medium | 8 | 128 | 1024 | 0.4 | Equivalent performance |
| Large | 8 | 256 | 2048 | 0.8 | Equivalent performance |

## Summary

### Core Issues
1. **Most Critical**: Gradient scaling uses local batch size instead of global batch size
2. **Important**: Learning rate not scaled according to batch size
3. **Supporting**: Warmup period too short, AdamW parameters not optimized for large batches

### Solution Priority
1. **Priority 1 (Critical)**: Fix gradient scaling in `partial_fc_v2.py`
2. **Priority 2 (Highly Recommended)**: Apply Linear Scaling Rule for learning rate
3. **Priority 3 (Recommended)**: Increase warmup epochs
4. **Priority 4 (Situational)**: Adjust AdamW parameters
5. **Priority 5 (Optional)**: Utilize gradient accumulation

### Expected Benefits
With these fixes:
- Training proceeds properly even with increased batch size
- Loss curves decrease normally
- Validation accuracy improves
- Multi-GPU training becomes stable

## References

1. Goyal, P., et al. (2017). "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour." arXiv:1706.02677
   - Theoretical foundation for Linear Scaling Rule

2. You, Y., et al. (2019). "Large Batch Optimization for Deep Learning: Training BERT in 76 minutes." arXiv:1904.00962
   - Best practices for large batch training

3. InsightFace PartialFC paper: "Partial FC: Training 10 Million Identities on a Single Machine"
   - Foundation architecture for this implementation

---

**Created**: 2025-11-03  
**Author**: GitHub Copilot Analysis  
**Version**: 1.0
