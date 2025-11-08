# Training on Google Cloud TPU v5e-8

This guide explains how to train face recognition models on Google Cloud TPU v5e-8 using PyTorch/XLA.

> **Quick Start**: See [TPU_QUICKSTART.md](TPU_QUICKSTART.md) for a condensed reference guide.

## Prerequisites

1. **Google Cloud TPU VM**: You need access to a TPU v5e-8 VM instance
2. **Python 3.12**: The repository requires Python 3.12 or higher
3. **PyTorch 2.8**: Compatible with PyTorch/XLA 2.8

## TPU v5e-8 Specifications

- **Number of cores**: 8
- **Memory**: 16GB HBM per core (128GB total)
- **Compute**: Optimized for ML workloads
- **Native precision**: bfloat16 (recommended for training)

## Installation

### Quick Installation (Recommended)

```bash
# Clone the repository
git clone https://github.com/yayoimizuha/face_recognition_train.git
cd face_recognition_train

# Run the automated setup script
./setup_tpu.sh
```

### Manual Installation

#### 1. Set up the repository

```bash
# Clone the repository
git clone https://github.com/yayoimizuha/face_recognition_train.git
cd face_recognition_train

# Install dependencies using uv (preferred)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync -U
uv sync --extra tpu

# OR install using pip
pip install -r requirements-tpu.txt
```

#### 2. Install PyTorch/XLA for TPU

```bash
pip install torch~=2.8.0 torchvision~=0.23.0
pip install https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.8.0-cp312-cp312-linux_x86_64.whl
```

## Configuration

TPU training uses a modified configuration file. The key differences from GPU training are:

1. **No DALI support**: DALI is CUDA-specific and not available on TPU
2. **Batch size per core**: TPU v5e-8 has 8 cores, so total batch size = `batch_size * 8`
3. **bfloat16 precision**: TPU v5e natively supports bfloat16 (`config.amp = torch.bfloat16`)
4. **Dataset type**: Use `imagefolder` or `webdataset`, not MXNet rec files with DALI

### Example Configuration

See `configs/edgeface_s_gamma_05_tpu.py` for a complete example:

```python
import torch
from easydict import EasyDict as edict

config = edict()

# Model settings
config.network = "edgeface_s_gamma_05"
config.embedding_size = 512

# TPU-specific settings
config.amp = torch.bfloat16  # Use bfloat16 for TPU
config.batch_size = 128      # Per core (total: 128 * 8 = 1024)
config.dali = False          # DALI not supported on TPU

# Optimizer
config.optimizer = "adamw"
config.lr = 6e-3
config.weight_decay = 0.05

# Dataset
config.rec = "data/webface12m"
config.num_classes = 617970
config.num_image = 12720066
config.num_epoch = 100
```

## Training

### Using the provided script

```bash
./run_tpu.sh configs/edgeface_s_gamma_05_tpu.py
```

### Manual execution

```bash
# Set environment variables
export XLA_USE_BF16=1      # Enable bfloat16
export PJRT_DEVICE=TPU     # Use TPU runtime

# Run training
python3 train_v2_tpu.py configs/edgeface_s_gamma_05_tpu.py
```

The `train_v2_tpu.py` script will automatically:
- Detect all 8 TPU cores
- Spawn 8 training processes (one per core)
- Use PyTorch/XLA's parallel loader for efficient data loading
- Apply gradient synchronization across cores

## Key Differences from GPU Training

### 1. Script Usage

- **GPU**: Use `train_v2.py` with `torchrun`
- **TPU**: Use `train_v2_tpu.py` (no `torchrun` needed)

### 2. Data Loading

- **GPU**: Supports DALI for fast data loading
- **TPU**: Uses standard PyTorch DataLoader with XLA ParallelLoader

### 3. Mixed Precision

- **GPU**: Can use `torch.float16` or `torch.bfloat16`
- **TPU**: Recommended to use `torch.bfloat16` (natively supported)

### 4. Distributed Training

- **GPU**: Uses `DistributedDataParallel` with NCCL backend
- **TPU**: Uses PyTorch/XLA's built-in distributed primitives

### 5. Optimizer Step

- **GPU**: Standard `optimizer.step()`
- **TPU**: `xm.optimizer_step(optimizer)` for XLA graph compilation

### 6. Graph Compilation

- **TPU**: Requires `xm.mark_step()` after optimizer step for efficient graph compilation

## Performance Tips

### 1. Batch Size

Start with a per-core batch size of 128-256 and adjust based on:
- Model size
- Available memory (16GB HBM per core)
- Training stability

Total effective batch size = `batch_size * 8`

### 2. Number of Workers

Set `config.num_workers` to 4-8 based on your TPU VM's CPU cores.

### 3. Gradient Accumulation

For very large models, use gradient accumulation:
```python
config.gradient_acc = 2  # Effective batch size = batch_size * 8 * 2
```

### 4. Mixed Precision

Always use bfloat16 on TPU for optimal performance:
```python
config.amp = torch.bfloat16
```

### 5. Data Pipeline

- Use `webdataset` for large-scale datasets
- Pre-fetch data to minimize I/O bottlenecks
- Store data on persistent disk or Cloud Storage

## Monitoring

### Logs

Training logs are saved to `<output_dir>/logs/`

### TensorBoard

```bash
tensorboard --logdir=<output_dir>/tensorboard
```

### WandB (optional)

Enable in config:
```python
config.using_wandb = True
config.wandb_key = "your-api-key"
config.wandb_project = "face-recognition-tpu"
```

## Checkpointing

Checkpoints are saved using XLA-specific save method:
- Per-core checkpoints: `checkpoint_tpu_{rank}.pt`
- Model only: `model_{epoch}.pt`

Load checkpoints in the config:
```python
config.resume = True
```

## Troubleshooting

### Out of Memory

- Reduce `config.batch_size`
- Enable gradient checkpointing (if implemented in model)
- Use gradient accumulation

### Slow Training

- Check data loading (increase `num_workers`)
- Verify `xm.mark_step()` is called appropriately
- Monitor TPU utilization with Cloud Monitoring

### NaN Loss

- Reduce learning rate
- Use gradient clipping (already enabled)
- Check data preprocessing

## References

- [PyTorch/XLA Documentation](https://pytorch.org/xla/release/r2.8/index.html)
- [Google Cloud TPU Documentation](https://cloud.google.com/tpu/docs)
- [PyTorch/XLA Performance Guide](https://github.com/pytorch/xla/blob/master/TROUBLESHOOTING.md)

## Example: Training EdgeFace-S on TPU v5e-8

```bash
# 1. Prepare your dataset
# Ensure data is in ImageFolder format or WebDataset format

# 2. Update config
# Edit configs/edgeface_s_gamma_05_tpu.py with your dataset path

# 3. Run training
./run_tpu.sh configs/edgeface_s_gamma_05_tpu.py

# 4. Monitor progress
tensorboard --logdir=edgeface_s_gamma_05_tpu/tensorboard
```

## Cost Optimization

- Use preemptible TPU VMs for lower cost
- Implement checkpointing to resume from interruptions
- Monitor training progress to avoid over-training
- Consider using TPU Pods for multi-node training (beyond single v5e-8)
