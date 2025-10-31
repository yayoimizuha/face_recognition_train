import argparse
import logging
import os
from datetime import datetime

import numpy as np
import torch
from backbones import get_model
from dataset import get_dataloader
from losses import CombinedMarginLoss
from lr_scheduler import PolynomialLRWarmup
from muon import Muon
from partial_fc_v2 import PartialFC_V2
from torch import distributed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.utils_callbacks import CallBackLogging, CallBackVerification
from utils.utils_config import get_config
from utils.utils_distributed_sampler import setup_seed
from utils.utils_logging import AverageMeter, init_logging
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook
from torch.amp.grad_scaler import GradScaler
import os
import sys
assert torch.__version__ >= "1.12.0", "In order to enjoy the features of the new torch, \
we have upgraded the torch to 1.12.0. torch before than 1.12.0 may not work in the future."

# detect device type and choose dist backend accordingly
def _detect_device_type() -> str:
    # Prefer Intel XPU if available
    if torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main(args):

    # get config
    cfg = get_config(args.config)
    # global control random seed
    setup_seed(seed=cfg.seed, cuda_deterministic=False)

    # decide device from config or auto-detect
    cfg_device = getattr(cfg, "device_type", None)
    requested_device = (cfg_device or "").lower() if isinstance(cfg_device, str) else None
    detected_device = _detect_device_type()
    device_type = requested_device or detected_device
    # validate availability; if requested but unavailable, fall back
    if requested_device == "cuda" and not torch.cuda.is_available():
        logging.warning("Requested device 'cuda' is unavailable. Falling back to detected '%s'", detected_device)
        device_type = detected_device
    if requested_device == "xpu" and not torch.xpu.is_available():
        logging.warning("Requested device 'xpu' is unavailable. Falling back to detected '%s'", detected_device)
        device_type = detected_device

    # Select backend by device
    if device_type == "xpu":
        dist_backend = "ccl"  # oneCCL for Intel XPU
    elif device_type == "cuda":
        dist_backend = "nccl"  # CUDA/ROCm
    else:
        dist_backend = "gloo"  # CPU fallback

    # init process group
    try:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        distributed.init_process_group(dist_backend)
    except KeyError:
        rank = 0
        local_rank = 0
        world_size = 1
        distributed.init_process_group(
            backend=dist_backend,
            init_method="tcp://127.0.0.1:12584",
            rank=rank,
            world_size=world_size,
        )

    # set device
    if device_type == "cuda":
        torch.cuda.set_device(local_rank)
    elif device_type == "xpu":
        torch.xpu.set_device(local_rank)

    device = torch.device(device_type)

    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output)

    summary_writer = (
        SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))
        if rank == 0
        else None
    )
    
    wandb_logger = None
    if cfg.using_wandb and rank == 0:
        import wandb
        # Sign in to wandb
        try:
            wandb.login(key=cfg.wandb_key)
        except Exception as e:
            print("WandB Key must be provided in config file (base.py).")
            print(f"Config Error: {e}")
        # Initialize wandb
        run_name = datetime.now().strftime("%y%m%d_%H%M") + f"_GPU{rank}"
        run_name = run_name if cfg.suffix_run_name is None else run_name + f"_{cfg.suffix_run_name}"
        try:
            wandb_logger = wandb.init(
                entity = cfg.wandb_entity, 
                project = cfg.wandb_project, 
                sync_tensorboard = True,
                resume=cfg.wandb_resume,
                name = run_name, 
                notes = cfg.notes) if rank == 0 or cfg.wandb_log_all else None
            if wandb_logger:
                wandb_logger.config.update(cfg)
        except Exception as e:
            print("WandB Data (Entity and Project name) must be provided in config file (base.py).")
            print(f"Config Error: {e}")
    # DALI は CUDA 前提のため、非 CUDA 環境では自動で無効化
    dali_enabled = bool(getattr(cfg, "dali", False) and device.type == "cuda")
    train_loader = get_dataloader(
        cfg.rec,
        local_rank,
        cfg.batch_size,
        dali_enabled,
        cfg.dali_aug,
        cfg.seed,
        cfg.num_workers,
        getattr(cfg, "dataset_type", "imagefolder"),
        device_type=device_type,
    )

    backbone = get_model(
        cfg.network, dropout=0.0, amp=cfg.amp, num_features=cfg.embedding_size).to(device)

    ddp_device_ids = [local_rank] if device.type != "cpu" else None
    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=ddp_device_ids, bucket_cap_mb=16,
        find_unused_parameters=True)
    # NCCL 環境のみ fp16 圧縮フックを使用（他 backend では未対応/非推奨）
    if dist_backend == "nccl":
        backbone.register_comm_hook(None, fp16_compress_hook)

    backbone.train()
    # FIXME using gradient checkpoint if there are some unused parameters will cause error
    backbone._set_static_graph()

    margin_loss = CombinedMarginLoss(
        64,
        cfg.margin_list[0],
        cfg.margin_list[1],
        cfg.margin_list[2],
        cfg.interclass_filtering_threshold
    )

    if cfg.optimizer == "sgd":
        module_partial_fc = PartialFC_V2(
            margin_loss, cfg.embedding_size, cfg.num_classes,
            cfg.sample_rate, False, amp=cfg.amp)
        module_partial_fc.train().to(device)
        # TODO the params of partial fc must be last in the params list
        opt = torch.optim.SGD(
            params=[{"params": backbone.parameters()}, {"params": module_partial_fc.parameters()}],
            lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)

    elif cfg.optimizer == "adamw":
        module_partial_fc = PartialFC_V2(
            margin_loss, cfg.embedding_size, cfg.num_classes,
            cfg.sample_rate, False, amp=cfg.amp)
        module_partial_fc.train().to(device)
        opt = torch.optim.AdamW(
            params=[{"params": backbone.parameters()}, {"params": module_partial_fc.parameters()}],
            lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    elif cfg.optimizer == "muon":
        module_partial_fc = PartialFC_V2(
            margin_loss, cfg.embedding_size, cfg.num_classes,
            cfg.sample_rate, False, amp=cfg.amp)
        module_partial_fc.train().to(device)
        # Muon optimizer with configurable parameters
        momentum = getattr(cfg, "momentum", 0.95)
        nesterov = getattr(cfg, "nesterov", True)
        opt = Muon(
            params=[{"params": backbone.parameters()}, {"params": module_partial_fc.parameters()}],
            lr=cfg.lr, momentum=momentum, nesterov=nesterov)
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")

    cfg.total_batch_size = cfg.batch_size * world_size
    cfg.warmup_step = cfg.num_image // cfg.total_batch_size * cfg.warmup_epoch
    cfg.total_step = cfg.num_image // cfg.total_batch_size * cfg.num_epoch

    lr_scheduler = PolynomialLRWarmup(
        optimizer=opt,
        warmup_iters=cfg.warmup_step,
        total_iters=cfg.total_step)

    start_epoch = 0
    global_step = 0
    if cfg.resume:
        dict_checkpoint = torch.load(os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))
        start_epoch = dict_checkpoint["epoch"]
        global_step = dict_checkpoint["global_step"]
        backbone.module.load_state_dict(dict_checkpoint["state_dict_backbone"])
        module_partial_fc.load_state_dict(dict_checkpoint["state_dict_softmax_fc"])
        # restore optimizer and lr scheduler to continue training seamlessly
        try:
            opt.load_state_dict(dict_checkpoint["state_optimizer"])
        except Exception:
            logging.warning("state_optimizer not found or incompatible; continuing without optimizer state")
        try:
            lr_scheduler.load_state_dict(dict_checkpoint["state_lr_scheduler"])
        except Exception:
            logging.warning("state_lr_scheduler not found or incompatible; continuing without scheduler state")
        del dict_checkpoint

    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))

    ver_prefix = getattr(cfg, "val_dir", None) or cfg.rec
    callback_verification = CallBackVerification(
        val_targets=cfg.val_targets, rec_prefix=ver_prefix, 
        summary_writer=summary_writer, wandb_logger = wandb_logger
    )
    callback_logging = CallBackLogging(
        frequent=cfg.frequent,
        total_step=cfg.total_step,
        batch_size=cfg.batch_size,
        start_step = global_step,
        writer=summary_writer
    )

    loss_am = AverageMeter()
    # Enable GradScaler when AMP dtype is set (also on CPU)
    amp = GradScaler(device=device_type, enabled=(cfg.amp is not None), growth_interval=100)

    for epoch in range(start_epoch, cfg.num_epoch):

        if isinstance(train_loader, DataLoader):
            sampler = getattr(train_loader, "sampler", None)
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
        for _, (img, local_labels) in enumerate(train_loader):
            global_step += 1
            # Ensure tensors are on the right device
            try:
                img = img.to(device, non_blocking=True)
            except Exception:
                img = img.to(device)
            try:
                local_labels = local_labels.to(device, non_blocking=True)
            except Exception:
                local_labels = local_labels.to(device)
            local_embeddings = backbone(img)
            loss: torch.Tensor = module_partial_fc(local_embeddings, local_labels)

            if amp.is_enabled():
                amp.scale(loss).backward()
                if global_step % cfg.gradient_acc == 0:
                    amp.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    amp.step(opt)
                    amp.update()
                    opt.zero_grad()
            else:
                loss.backward()
                if global_step % cfg.gradient_acc == 0:
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    opt.step()
                    opt.zero_grad()
            lr_scheduler.step()

            with torch.no_grad():
                if wandb_logger:
                    wandb_logger.log({
                        'Loss/Step Loss': loss.item(),
                        'Loss/Train Loss': loss_am.avg,
                        'Process/Step': global_step,
                        'Process/Epoch': epoch
                    })
                
                loss_am.update(loss.item(), 1)
                callback_logging(global_step, loss_am, epoch, amp.is_enabled(), lr_scheduler.get_last_lr()[0], amp)

                if global_step % cfg.verbose == 0 and global_step > 0:
                    callback_verification(global_step, backbone)

        if cfg.save_all_states:
            checkpoint = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "state_dict_backbone": backbone.module.state_dict(),
                "state_dict_softmax_fc": module_partial_fc.state_dict(),
                "state_optimizer": opt.state_dict(),
                "state_lr_scheduler": lr_scheduler.state_dict()
            }
            torch.save(checkpoint, os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))

        if rank == 0:
            path_module = os.path.join(cfg.output, f"model_{epoch}.pt")
            torch.save(backbone.module.state_dict(), path_module)

            if wandb_logger and cfg.save_artifacts:
                artifact_name = f"{run_name}_E{epoch}"
                model = wandb.Artifact(artifact_name, type='model')
                model.add_file(path_module)
                wandb_logger.log_artifact(model)
                
        if dali_enabled:
            reset_fn = getattr(train_loader, "reset", None)
            if callable(reset_fn):
                reset_fn()

    if rank == 0:
        path_module = os.path.join(cfg.output, "model.pt")
        torch.save(backbone.module.state_dict(), path_module)
        
        if wandb_logger and cfg.save_artifacts:
            artifact_name = f"{run_name}_Final"
            model = wandb.Artifact(artifact_name, type='model')
            model.add_file(path_module)
            wandb_logger.log_artifact(model)



if __name__ == "__main__":
    # Enable cudnn benchmark only when available (CUDA)
    if hasattr(torch.backends, "cudnn") and torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="Distributed Arcface Training in Pytorch")
    parser.add_argument("config", type=str, help="py config file")
    main(parser.parse_args())
