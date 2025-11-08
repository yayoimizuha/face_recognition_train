# Quick Reference: TPU v5e-8 Training

## Quick Start

```bash
# 1. Install dependencies
uv sync --extra tpu

# 2. Install PyTorch/XLA on TPU VM
pip install https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.8.0-cp312-cp312-linux_x86_64.whl

# 3. Run training
./run_tpu.sh configs/edgeface_s_gamma_05_tpu.py
```

## Key Differences vs GPU Training

| Aspect | GPU (`train_v2.py`) | TPU (`train_v2_tpu.py`) |
|--------|---------------------|-------------------------|
| Launch | `torchrun --nproc_per_node=8` | `python3` (xmp.spawn handles multi-process) |
| Backend | NCCL/Gloo | PyTorch/XLA |
| Data Loading | DALI supported | Standard DataLoader + ParallelLoader |
| Mixed Precision | fp16/bf16 | bf16 (native) |
| Optimizer Step | `optimizer.step()` | `xm.optimizer_step(opt)` |
| Checkpointing | `torch.save()` | `xm.save()` |
| Graph Compilation | Automatic | Requires `xm.mark_step()` |

## Environment Variables

```bash
export XLA_USE_BF16=1      # Enable bfloat16
export PJRT_DEVICE=TPU     # Use TPU runtime
```

## Configuration Checklist

- [ ] Set `config.amp = torch.bfloat16`
- [ ] Set `config.dali = False`
- [ ] Adjust `config.batch_size` (per core, total = batch_size * 8)
- [ ] Set `config.dataset_type = "imagefolder"` or `"webdataset"`
- [ ] Update `config.rec` to dataset path
- [ ] Set `config.num_workers = 4-8`

## Performance Tips

1. **Batch Size**: Start with 128 per core (1024 total)
2. **Data Loading**: Use 4-8 workers
3. **Mixed Precision**: Always use bfloat16
4. **Dataset Format**: WebDataset for large-scale datasets
5. **Mark Steps**: `xm.mark_step()` is called after optimizer step

## Troubleshooting

**OOM**: Reduce `config.batch_size`  
**Slow**: Check `num_workers`, verify `xm.mark_step()` usage  
**NaN Loss**: Lower learning rate, verify data preprocessing

See [docs/TPU_TRAINING.md](docs/TPU_TRAINING.md) for detailed guide.
