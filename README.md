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

### Using Muon Optimizer

This repository now supports the **Muon optimizer** (MomentUm Orthogonalized by Newton-schulz), a novel optimizer that combines momentum-based updates with Newton-Schulz orthogonalization.

🚀 **[Quick Start Guide](docs/MUON_QUICKSTART.md)** | 📚 **[Full Documentation (EN)](docs/MUON_OPTIMIZER.md)** | 🇯🇵 **[日本語ドキュメント](docs/MUON_OPTIMIZER_JA.md)**

#### What is Muon?

Muon is particularly effective for training deep neural networks with 2D parameters (weight matrices). It provides:

- **Better convergence**: Often converges faster than SGD/Adam on certain architectures
- **Implicit regularization**: Newton-Schulz orthogonalization acts as regularization
- **Stability**: More stable training dynamics
- **Effectiveness for CNNs and Transformers**: Particularly strong for convolutional architectures

#### Key Features

- Uses Newton-Schulz iterations for matrix orthogonalization
- Combines momentum-based gradient descent (with optional Nesterov)
- Designed for 2D parameters (Linear, Conv layers)
- Falls back to standard momentum for non-2D parameters (biases, BatchNorm)

#### Usage Example

```bash
torchrun --nproc_per_node=4 train_v2.py configs/muon_example.py
```

#### Configuration

In your config file:

```python
config.optimizer = "muon"
config.lr = 0.02          # Typically higher than AdamW (0.01-0.05)
config.momentum = 0.95    # High momentum recommended (0.90-0.95)
config.nesterov = True    # Nesterov momentum improves convergence
```

See `configs/muon_example.py` for a complete example configuration.

#### Recommended Hyperparameters

| Parameter | Recommended Range | Default | Notes |
|-----------|------------------|---------|-------|
| `lr` | 0.01 - 0.05 | 0.02 | Higher than AdamW |
| `momentum` | 0.90 - 0.95 | 0.95 | High momentum works well |
| `nesterov` | True/False | True | Recommended to enable |

#### When to Use Muon

- **Good for**: CNNs, ResNets, Vision Transformers, face recognition models
- **Better than AdamW**: When you have well-structured 2D parameters
- **Consider AdamW**: For models with many non-2D parameters or when training very large models

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
