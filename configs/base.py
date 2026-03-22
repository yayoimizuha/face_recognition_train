import os

from dotenv import load_dotenv, find_dotenv
from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

_ = load_dotenv(find_dotenv())

config = edict()

# Margin Base Softmax
config.margin_list = (1.0, 0.5, 0.0)
config.network = "r50"
config.resume = False
config.save_all_states = True
config.output = "ms1mv3_arcface_r50"

config.embedding_size = 512

# Device selection: "cuda" | "xpu" | "cpu"
config.device_type = "cuda"

# Apply Global Depthwise Convolution head instead of default pooling/classifier
# When True, timm-based backbones will replace their head with GDConv that
# computes a depthwise HxW conv followed by 1x1 conv to produce `embedding_size`.
config.apply_gdconv = False

# Partial FC
config.sample_rate = 1
config.interclass_filtering_threshold = 0

# Legacy flag (deprecated): use config.amp instead. If True, it maps to torch.float16.
config.fp16 = False
# AMP dtype selection: set torch dtype directly (e.g., torch.bfloat16, torch.float16) or None to disable
config.amp = None
config.batch_size = 128

# For SGD 
config.optimizer = "sgd"
config.lr = 0.1
config.momentum = 0.9
config.weight_decay = 5e-4

# For AdamW
# config.optimizer = "adamw"
# config.lr = 0.001
# config.weight_decay = 0.1
# AdamW beta hyperparameters (Beta1, Beta2)
config.adam_betas = (0.9, 0.999)

config.verbose = 2000
config.frequent = 10

# For Large Sacle Dataset, such as WebFace42M
config.dali = False
config.dali_aug = False
config.dataset_type = "imagefolder"  # choose from: "imagefolder", "webdataset"

# Gradient ACC
config.gradient_acc = 1

# setup seed
config.seed = 2048

# dataload numworkers
config.num_workers = 2

# WandB Logger
config.wandb_key = os.environ["WANDB_API_KEY"]
config.suffix_run_name = None
config.using_wandb = True
config.wandb_entity = os.environ["WANDB_ENTITY"]
config.wandb_project = os.environ["WANDB_PROJECT"]
config.wandb_log_all = True
config.save_artifacts = False
config.wandb_resume = True  # resume wandb run: Only if the you wand t resume the last run that it was interrupted
config.notes = "Training various FaceNets with new codebase and dataset and algorithms."


config.rec = "/mnt/nvme/Glint360k_WebDataset"
config.val_dir = "/mnt/nvme/data1"
