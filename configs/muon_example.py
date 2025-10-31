"""
Example configuration for using Muon optimizer with face recognition training.

Muon (MomentUm Orthogonalized by Newton-schulz) is a novel optimizer that combines
momentum-based updates with Newton-Schulz orthogonalization. It's particularly
effective for training deep neural networks with 2D parameters.

Key characteristics of Muon:
- Better convergence properties than standard SGD/Adam for certain architectures
- Particularly effective for convolutional and transformer networks
- Uses matrix orthogonalization to improve optimization landscape
- Requires fewer hyperparameter tuning compared to AdamW

Recommended hyperparameters:
- Learning rate: 0.01 - 0.05 (typically higher than AdamW)
- Momentum: 0.90 - 0.95
- Nesterov: True (recommended)
"""

from easydict import EasyDict as edict

config = edict()

# Model architecture
config.network = "r50"
config.embedding_size = 512

# Dataset configuration
config.rec = "data/webface12m"  # Update with your dataset path
config.num_classes = 617970
config.num_image = 12720066
config.num_epoch = 100
config.warmup_epoch = 2

# Training configuration
config.batch_size = 512
config.verbose = 2000
config.frequent = 10
config.num_workers = 4

# Muon optimizer configuration
config.optimizer = "muon"
config.lr = 0.02  # Muon typically uses higher learning rates than AdamW
config.momentum = 0.95  # High momentum recommended
config.nesterov = True  # Nesterov momentum improves convergence

# Note: Muon doesn't use weight_decay in the traditional sense
# The Newton-Schulz orthogonalization acts as implicit regularization

# Margin loss configuration
config.margin_list = (1.0, 0.0, 0.4)
config.sample_rate = 0.3
config.interclass_filtering_threshold = 0

# AMP configuration
config.fp16 = False
config.amp = None  # Set to torch.float16 or torch.bfloat16 if needed

# Device configuration
config.device_type = "cuda"

# Checkpointing
config.resume = False
config.save_all_states = True
config.output = 'output_muon_r50/'

# Validation
config.val_targets = []

# Data augmentation (if using DALI)
config.dali = False
config.dali_aug = False
config.dataset_type = "imagefolder"

# Gradient accumulation
config.gradient_acc = 1

# Random seed
config.seed = 2048

# WandB logging (optional)
# config.using_wandb = False
# config.wandb_key = ""
# config.wandb_entity = ""
# config.wandb_project = ""
