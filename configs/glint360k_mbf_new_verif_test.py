from easydict import EasyDict as edict
import torch

config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.network = "mbf"
config.resume = False
# 出力先（None だと train_v2.py 内の os.path.join でエラーになるため明示）
config.output = "./work_dirs/glint360k_mbf_verif_test"
config.embedding_size = 512
config.sample_rate = 1.0
config.device_type = "cuda"
config.dist_backend = "nccl"  # 明示指定（"gloo"|"ccl"|"nccl"）
# config.fp16 = True  # legacy flag; kept for backward compatibility
config.amp = torch.bfloat16  # set torch dtype directly
config.momentum = 0.9
config.weight_decay = 1e-4
config.batch_size = 128
config.lr = 0.1
config.verbose = 200
config.dali = False
# DataLoader 用
config.num_workers = 5
# DALI 増強は無効（HF の DataLoader を使用）
config.dali_aug = False
config.dataset_type = "webdataset"
config.rec = "/mnt/nvme/Glint360k_WebDataset/"

# Glint360k のクラス数と画像枚数（スケジューラ計算用）
config.num_classes = 360232
config.num_image = 17091657
config.num_epoch = 4
config.warmup_epoch = 0
config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]
config.val_dir = "/mnt/nvme/data1/"