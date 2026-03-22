from easydict import EasyDict as edict
import torch

config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.network = "resnet50.a1_in1k"
config.resume = True
# 出力先（None だと train_v2.py 内の os.path.join でエラーになるため明示）
config.output = "./work_dirs/glint360k_resnet50"
config.embedding_size = 512
config.sample_rate = 1.0
config.device_type = "cuda"
config.dist_backend = "nccl"  # 明示指定（"gloo"|"ccl"|"nccl"）
config.amp = torch.bfloat16  # set torch dtype directly
config.batch_size = 128
config.verbose = 1000
config.dali = False

config.apply_gdconv = False

config.optimizer = "sgd"
config.adam_betas = (0.9, 0.999)
config.lr = 0.2
config.weight_decay = 0.0001

# DataLoader 用
config.num_workers = 20
# DALI 増強は無効（HF の DataLoader を使用）
config.dali_aug = False
config.dataset_type = "webdataset"

# Glint360k のクラス数と画像枚数（スケジューラ計算用）
config.num_classes = 360232
config.num_image = 17091657
config.num_epoch = 70
config.warmup_epoch = 5
config.val_targets = ["lfw", "cfp_fp", "agedb_30"]
