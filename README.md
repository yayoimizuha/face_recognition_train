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

#### Training on Google Cloud TPU v5e-8

For training on TPU, use the TPU-specific script and configuration:

```bash
./run_tpu.sh configs/edgeface_s_gamma_05_tpu.py
```

See [docs/TPU_TRAINING.md](docs/TPU_TRAINING.md) for detailed instructions on TPU training setup, configuration, and optimization.

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
| Google Cloud TPU | Supported (bf16) | N/A (XLA handles scaling) | bf16 | Native bfloat16 support. Use `train_v2_tpu.py` for TPU training. |
| CPU | Enabled (not disabled in this repo) | Enabled (torch.amp) | bf16 (if ISA supports), otherwise per-environment | No fallback is implemented; specifying an unsupported dtype will raise an error. |

Quick usage
- In your config file, set the dtype directly:
	- `config.amp = torch.float16` or `config.amp = torch.bfloat16`
- This repo enables autocast/GradScaler even on CPU. There is no automatic fallback on unsupported dtypes.
