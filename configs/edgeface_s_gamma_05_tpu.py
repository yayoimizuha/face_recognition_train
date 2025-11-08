from easydict import EasyDict as edict
import torch

# TPU v5e-8 configuration for EdgeFace-S training
# Reference: https://docs.pytorch.org/xla/release/r2.8/index.html

config = edict()

# Model configuration
config.margin_list = (1.0, 0.0, 0.4)
config.network = "edgeface_s_gamma_05"
config.resume = False
config.output = 'edgeface_s_gamma_05_tpu/'
config.embedding_size = 512
config.sample_rate = 0.3

# TPU-specific settings
config.device_type = "tpu"  # Not used in TPU training script, but kept for consistency
config.dist_backend = "gloo"  # TPU uses XLA's distributed backend, this is ignored

# Mixed precision training
# TPU v5e supports bfloat16 natively
config.fp16 = False
config.amp = torch.bfloat16  # Use bfloat16 for TPU

# Optimizer settings
config.weight_decay = 0.05
config.optimizer = "adamw"
config.lr = 6e-3
config.adam_betas = (0.9, 0.999)

# Batch size - TPU v5e-8 has 8 cores
# Adjust batch size per core based on available memory
config.batch_size = 128  # Per core batch size (total: 128 * 8 = 1024)

# Training settings
config.verbose = 2000
config.frequent = 10

# DALI is not supported on TPU
config.dali = False
config.dali_aug = False

# Data loader settings
config.num_workers = 4  # Adjust based on TPU VM CPU cores

# Dataset configuration
config.rec = "data/webface12m"
config.num_classes = 617970
config.num_image = 12720066
config.num_epoch = 100
config.warmup_epoch = 2
config.val_targets = []
config.dataset_type = "imagefolder"

# Gradient accumulation
config.gradient_acc = 1

# Random seed
config.seed = 2048

# Logging and checkpointing
config.save_all_states = True
config.interclass_filtering_threshold = 0

# WandB configuration (optional)
import os
config.using_wandb = False  # Set to True if using WandB
config.wandb_key = os.environ.get("WANDB_API_KEY", "")
config.suffix_run_name = "tpu_v5e8"
config.wandb_entity = os.environ.get("WANDB_ENTITY", "")
config.wandb_project = os.environ.get("WANDB_PROJECT", "face-recognition-tpu")
config.wandb_log_all = False
config.save_artifacts = False
config.wandb_resume = False
config.notes = "Training EdgeFace-S on Google Cloud TPU v5e-8 with PyTorch/XLA"
