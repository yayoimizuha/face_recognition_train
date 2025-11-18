# Clean-upped Face Recognition model training Repository 
## Installation Instructions

### Use uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync -U
```
[Glint-360k](https://academictorrents.com/details/e5f46ee502b9e76da8cc3a0e4f7c17e4000c7b1e)
[VGGFace2](https://academictorrents.com/details/535113b8395832f09121bc53ac85d7bc8ef6fa5b)

### Step 5: Training

#### EdgeFace-S(𝛾=0.5)

Launch the following command after setting the root path and output path in the config files:

```bash
torchrun --nproc_per_node=8 train_v2.py configs/edgeface_s_gamma_05.py
```
After finishing this step, launch:

```bash
torchrun --nproc_per_node=8 train_v2_restart.py configs/edgeface_s_gamma_05_restart.py
```

#### EdgeFace-XS(𝛾=0.6)

Launch the following command after setting the root path and output path in the config files:

```bash
torchrun --nproc_per_node=4 train_v2.py configs/edgeface_xs_gamma_06.py
```
After finishing this step, launch:

```bash
torchrun --nproc_per_node=4 train_v2_restart.py configs/edgeface_xs_gamma_06_restart.py
```

### Using WebDataset for training

This repo now supports WebDataset shards for training. Set `config.rec` to one of:

- A directory containing `.tar` shards (auto-expanded as `<dir>/*.tar`)
- A single `.tar` file path
- A brace/glob pattern like `/data/shards/{000000..000999}.tar` or `/data/shards/*.tar`
- A text file listing shard URLs/paths (one per line), e.g. `shards.txt`

Samples must include an image (`jpg/jpeg/png`) and an integer class label in the `cls` key.
Example config snippet:

```python
config.rec = "/data/webdataset/shards/{000000..000127}.tar"
```

## Optimizer Options

This repository supports multiple optimizers. You can select an optimizer by setting `config.optimizer` in your configuration file:

### SGD (Stochastic Gradient Descent)
```python
config.optimizer = "sgd"
config.lr = 0.1
config.momentum = 0.9
config.weight_decay = 5e-4
```

### AdamW
```python
config.optimizer = "adamw"
config.lr = 0.001
config.weight_decay = 0.1
config.adam_betas = (0.9, 0.999)
```

### LAMB (Layer-wise Adaptive Moments optimizer for Batch training)
LAMB is designed for efficient large batch training, making it ideal for training small models with very large batch sizes.

```python
config.optimizer = "lamb"
config.lr = 6e-3
config.weight_decay = 0.05
config.adam_betas = (0.9, 0.999)
config.batch_size = 2048  # Large batch sizes work well with LAMB
```

**Example usage:**
```bash
torchrun --nproc_per_node=8 train_v2.py configs/edgeface_s_lamb_test.py
```

### RAdamScheduleFree
Schedule-free RAdam optimizer that doesn't require a separate learning rate scheduler.

```python
config.optimizer = "radam_schedulefree"
config.lr = 0.001
config.weight_decay = 0.1
config.adam_betas = (0.9, 0.999)
```


# EdgeFace Models via `torch.hub`

## Available Models

- `edgeface_base`
- `edgeface_xs_gamma_06`
- `edgeface_xs_q`
- `edgeface_xxs`
- `edgeface_xxs_q`
- `edgeface_s_gamma_05`
- `resnet50k`

## AMP Support Matrix (autocast / GradScaler)

This repository expects you to set `config.amp` to a torch dtype directly (e.g., `torch.float16`, `torch.bfloat16`). AMP is not disabled on CPU, and no automatic fallback is performed when a dtype is unsupported by your environment. Please verify support on your setup (PyTorch/device/driver/BLAS).

| Device | autocast | GradScaler | Recommended dtype | Notes |
|---|---|---|---|---|
| NVIDIA CUDA GPU | Supported (fp16, bf16) | Supported | bf16 or fp16 (depends on HW/model) | Use the dtype that matches your GPU capability and model stability. |
| AMD ROCm GPU | Supported | Supported | bf16 (RDNA3+) or fp16 | bf16 availability may depend on ROCm version and hardware. |
| Intel GPU (XPU / oneAPI) | Supported | Supported | bf16 | Ensure oneAPI/XPU stack is properly set up. |
| CPU | Enabled (not disabled in this repo) | Enabled (torch.amp) | bf16 (if ISA supports), otherwise per-environment | No fallback is implemented; specifying an unsupported dtype will raise an error. |

Quick usage
- In your config file, set the dtype directly:
	- `config.amp = torch.float16` or `config.amp = torch.bfloat16`
- This repo enables autocast/GradScaler even on CPU. There is no automatic fallback on unsupported dtypes.
