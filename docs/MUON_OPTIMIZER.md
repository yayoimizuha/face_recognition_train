# Muon Optimizer Documentation

## Overview

**Muon** (MomentUm Orthogonalized by Newton-schulz) is a novel optimization algorithm designed for training deep neural networks. It combines momentum-based gradient descent with Newton-Schulz matrix orthogonalization, providing improved convergence properties and training stability.

## What is Muon?

Muon is particularly effective for optimizing 2D parameters (weight matrices) in neural networks. The key innovation is the application of Newton-Schulz iterations to orthogonalize the gradient updates, which helps maintain better optimization geometry throughout training.

### Key Features

- **Newton-Schulz Orthogonalization**: Applies iterative orthogonalization to gradient updates
- **Momentum-Based Updates**: Combines with momentum (including Nesterov) for acceleration
- **2D Parameter Optimization**: Specifically designed for weight matrices in Linear and Conv layers
- **Implicit Regularization**: The orthogonalization acts as implicit regularization
- **Stable Training**: More stable training dynamics compared to standard optimizers

## Mathematical Background

### Newton-Schulz Iteration

The Newton-Schulz method computes the orthogonalization of a matrix G through iterative refinement:

```
X₀ = G / ||G||
Xₙ₊₁ = a·Xₙ + b·A·Xₙ + c·A²·Xₙ
where A = Xₙ·Xₙᵀ
```

The coefficients (a=3.4445, b=-4.7750, c=2.0315) are tuned for optimal convergence.

### Update Rule

For parameter θ with gradient g:

1. **Momentum Update**: `m = β·m + g`
2. **Nesterov (optional)**: `g' = g + β·m`
3. **Orthogonalization**: `g_orth = NewtonSchulz(g')`
4. **Parameter Update**: `θ = θ - η·g_orth`

## Installation

Muon is included in this repository. No additional installation is required beyond the standard dependencies.

## Usage

### Basic Usage

```python
from muon import Muon
import torch.nn as nn

# Create your model
model = nn.Sequential(
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# Initialize Muon optimizer
optimizer = Muon(
    model.parameters(),
    lr=0.02,           # Learning rate
    momentum=0.95,     # Momentum coefficient
    nesterov=True      # Use Nesterov momentum
)

# Training loop
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
```

### Configuration File Usage

In your config file (e.g., `configs/my_config.py`):

```python
from easydict import EasyDict as edict

config = edict()

# Model configuration
config.network = "r50"
config.embedding_size = 512

# Muon optimizer configuration
config.optimizer = "muon"
config.lr = 0.02          # Recommended: 0.01-0.05
config.momentum = 0.95    # Recommended: 0.90-0.95
config.nesterov = True    # Recommended: True

# Other training parameters
config.batch_size = 512
config.num_epoch = 100
```

Then run training:

```bash
torchrun --nproc_per_node=4 train_v2.py configs/my_config.py
```

## Hyperparameter Guidelines

### Learning Rate

- **Recommended range**: 0.01 - 0.05
- **Typical value**: 0.02
- **Note**: Muon typically uses higher learning rates than AdamW

Muon's orthogonalization allows for more aggressive learning rates without instability.

### Momentum

- **Recommended range**: 0.90 - 0.95
- **Typical value**: 0.95
- **Note**: Higher momentum works well with Muon

The orthogonalization step benefits from high momentum, as it helps maintain direction consistency.

### Nesterov Momentum

- **Recommended**: True
- **Benefit**: Improves convergence speed and final performance

Nesterov momentum provides look-ahead updates that work synergistically with orthogonalization.

### Newton-Schulz Steps

- **Default**: 5 steps
- **Range**: 3-7 steps
- **Note**: More steps = more accurate orthogonalization but slower

## Performance Characteristics

### When to Use Muon

**Best suited for:**
- Convolutional Neural Networks (CNNs)
- ResNet architectures
- Face recognition models
- Vision Transformers
- Models with many 2D weight matrices

**Consider alternatives when:**
- Training very large language models (>1B parameters)
- Models with predominantly 1D parameters
- Require memory-efficient optimization

### Comparison with Other Optimizers

| Aspect | Muon | SGD | AdamW |
|--------|------|-----|-------|
| Convergence Speed | Fast | Moderate | Fast |
| Memory Overhead | Low | Low | High |
| Hyperparameter Sensitivity | Low | High | Moderate |
| Best for 2D params | ✓ | ✗ | ✗ |
| Best for 1D params | ✗ | ✓ | ✓ |
| Learning rate range | 0.01-0.05 | 0.01-0.1 | 0.0001-0.001 |

## Implementation Details

### Parameter Handling

Muon treats different parameter types differently:

- **2D Parameters** (e.g., Linear, Conv weights): Full Muon update with orthogonalization
- **Non-2D Parameters** (e.g., biases, BatchNorm): Standard momentum update without orthogonalization

This hybrid approach ensures optimal performance across all parameter types.

### Memory Considerations

Muon has similar memory requirements to SGD with momentum:
- Stores momentum buffer for each parameter
- No additional state like Adam's second moment
- Orthogonalization is computed on-the-fly

### Computational Cost

Per-step cost:
1. **Momentum update**: O(n) - same as SGD
2. **Newton-Schulz iterations**: O(d³) where d = min(rows, cols) for 2D params
3. **Parameter update**: O(n) - same as SGD

For typical CNNs, the Newton-Schulz cost is negligible compared to forward/backward passes.

## Advanced Topics

### Mixed Precision Training

Muon works well with automatic mixed precision (AMP):

```python
from torch.amp import autocast, GradScaler

scaler = GradScaler()
optimizer = Muon(model.parameters(), lr=0.02)

for data, target in train_loader:
    optimizer.zero_grad()
    
    with autocast(device_type='cuda', dtype=torch.float16):
        output = model(data)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Distributed Training

Muon is fully compatible with PyTorch's DistributedDataParallel:

```python
import torch.distributed as dist

# Initialize process group
dist.init_process_group(backend='nccl')

# Wrap model with DDP
model = torch.nn.parallel.DistributedDataParallel(model)

# Use Muon optimizer as usual
optimizer = Muon(model.parameters(), lr=0.02)
```

### Learning Rate Scheduling

Muon works well with standard learning rate schedulers:

```python
from lr_scheduler import PolynomialLRWarmup

optimizer = Muon(model.parameters(), lr=0.02)
scheduler = PolynomialLRWarmup(
    optimizer=optimizer,
    warmup_iters=1000,
    total_iters=100000
)

for epoch in range(num_epochs):
    for batch in train_loader:
        # ... training step ...
        scheduler.step()
```

## Troubleshooting

### Common Issues

**Issue**: Training is unstable / loss diverges
- **Solution**: Reduce learning rate (try 0.01 instead of 0.02)
- **Solution**: Ensure gradients are properly clipped if needed

**Issue**: Training is slower than expected
- **Solution**: This is normal for the first few steps due to compilation overhead
- **Solution**: Ensure you're using GPU for training

**Issue**: Out of memory errors
- **Solution**: Reduce batch size
- **Solution**: Use gradient accumulation

### Performance Tips

1. **Use torch.compile**: Can speed up Newton-Schulz iterations
   ```python
   model = torch.compile(model)
   ```

2. **Profile your training**: Identify bottlenecks
   ```python
   with torch.profiler.profile() as prof:
       # training step
   ```

3. **Adjust NS steps**: Fewer steps = faster but less accurate orthogonalization

## References

- [Original Muon Implementation](https://github.com/KellerJordan/cifar10-airbench)
- [Newton-Schulz Iteration](https://en.wikipedia.org/wiki/Newton%27s_method)
- Research papers on matrix orthogonalization in optimization

## Examples

See `configs/muon_example.py` for a complete configuration example.

## Support

For issues or questions about the Muon optimizer:
1. Check this documentation
2. Review the example configuration in `configs/muon_example.py`
3. Examine the test file `test_muon.py` for usage examples
4. Open an issue on the repository

## License

This implementation is part of the face_recognition_train repository and follows the same license terms.
