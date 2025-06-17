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


# EdgeFace Models via `torch.hub`

## Available Models

- `edgeface_base`
- `edgeface_xs_gamma_06`
- `edgeface_xs_q`
- `edgeface_xxs`
- `edgeface_xxs_q`
- `edgeface_s_gamma_05`
- `resnet50k`