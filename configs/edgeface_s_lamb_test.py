from easydict import EasyDict as edict
import torch

# Test configuration for LAMB optimizer
# LAMB is designed for large batch training
# Adjust data paths and num_classes according to your dataset

config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.network = "edgeface_s_gamma_05"
config.resume = False
config.output = 'edgeface_s_lamb_test/'
config.embedding_size = 512
config.sample_rate = 0.3
config.fp16 = False
config.amp = torch.bfloat16  # Use bfloat16 for mixed precision training
config.device_type = "cuda"
config.dist_backend = "nccl"
config.weight_decay = 0.05
config.batch_size = 2048  # Large batch size for LAMB optimizer
config.optimizer = "lamb"
config.lr = 6e-3
config.adam_betas = (0.9, 0.999)
config.verbose = 2000
config.dali = True 
config.dali_aug = True

config.num_workers = 6

# Update these paths according to your dataset
config.rec = "data/webface12m"
config.num_classes = 617970
config.num_image = 12720066
config.num_epoch = 100
config.warmup_epoch = 2
config.val_targets = []
