import argparse
import logging
import os
from datetime import datetime

import numpy as np
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
from backbones import get_model
from dataset import get_dataloader
from losses import CombinedMarginLoss
from lr_scheduler import PolynomialLRWarmup
from partial_fc_v2 import PartialFC_V2
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.utils_callbacks import CallBackLogging, CallBackVerification
from utils.utils_config import get_config
from utils.utils_distributed_sampler import setup_seed
from utils.utils_logging import AverageMeter, init_logging
from timm.layers.norm_act import convert_sync_batchnorm

assert (
    torch.__version__ >= "1.12.0"
), "In order to enjoy the features of the new torch, \
we have upgraded the torch to 1.12.0. torch before than 1.12.0 may not work in the future."


def _mp_fn(index, args):
    """Training function for each TPU core."""
    # Get XLA device for this process
    device = xm.xla_device()
    
    # get config
    cfg = get_config(args.config)
    
    # Get world size and rank from XLA
    world_size = xm.xrt_world_size()
    rank = xm.get_ordinal()
    
    # global control random seed
    setup_seed(seed=cfg.seed, cuda_deterministic=False)

    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output)

    summary_writer = SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard")) if rank == 0 else None

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
        run_name = datetime.now().strftime("%y%m%d_%H%M") + f"_TPU{rank}"
        run_name = run_name if cfg.suffix_run_name is None else run_name + f"_{cfg.suffix_run_name}"
        try:
            wandb_logger = (
                wandb.init(
                    entity=cfg.wandb_entity,
                    project=cfg.wandb_project,
                    sync_tensorboard=True,
                    resume=cfg.wandb_resume,
                    name=run_name,
                    notes=cfg.notes,
                )
                if rank == 0 or cfg.wandb_log_all
                else None
            )
            if wandb_logger:
                wandb_logger.config.update(cfg)
        except Exception as e:
            print("WandB Data (Entity and Project name) must be provided in config file (base.py).")
            print(f"Config Error: {e}")

    # TPU doesn't support DALI
    dali_enabled = False
    train_loader = get_dataloader(
        cfg.rec,
        rank,
        cfg.batch_size,
        dali_enabled,
        False,  # dali_aug
        cfg.seed,
        cfg.num_workers,
        getattr(cfg, "dataset_type", "imagefolder"),
        device_type="cpu",  # Load data on CPU first, will move to TPU
    )

    backbone = get_model(cfg.network, dropout=0.0, amp=cfg.amp, num_features=cfg.embedding_size).to(device)

    # Convert BatchNorm layers to SyncBatchNorm for proper distributed training
    backbone = convert_sync_batchnorm(backbone)

    backbone.train()

    margin_loss = CombinedMarginLoss(
        64, cfg.margin_list[0], cfg.margin_list[1], cfg.margin_list[2], cfg.interclass_filtering_threshold
    )

    if cfg.optimizer == "sgd":
        module_partial_fc = PartialFC_V2(
            margin_loss, cfg.embedding_size, cfg.num_classes, cfg.sample_rate, False, amp=cfg.amp
        )
        module_partial_fc.train().to(device)
        opt = torch.optim.SGD(
            params=[{"params": backbone.parameters()}, {"params": module_partial_fc.parameters()}],
            lr=cfg.lr,
            momentum=0.9,
            weight_decay=cfg.weight_decay,
        )

    elif cfg.optimizer == "adamw":
        module_partial_fc = PartialFC_V2(
            margin_loss, cfg.embedding_size, cfg.num_classes, cfg.sample_rate, False, amp=cfg.amp
        )
        module_partial_fc.train().to(device)
        betas = tuple(getattr(cfg, "adam_betas", (0.9, 0.999)))
        opt = torch.optim.AdamW(
            params=[{"params": backbone.parameters()}, {"params": module_partial_fc.parameters()}],
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=betas,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")

    cfg.total_batch_size = cfg.batch_size * world_size
    cfg.warmup_step = cfg.num_image // cfg.total_batch_size * cfg.warmup_epoch
    cfg.total_step = cfg.num_image // cfg.total_batch_size * cfg.num_epoch

    lr_scheduler = PolynomialLRWarmup(optimizer=opt, warmup_iters=cfg.warmup_step, total_iters=cfg.total_step)

    start_epoch = 0
    global_step = 0
    if cfg.resume:
        dict_checkpoint = torch.load(os.path.join(cfg.output, f"checkpoint_tpu_{rank}.pt"))
        start_epoch = dict_checkpoint["epoch"]
        global_step = dict_checkpoint["global_step"]
        backbone.load_state_dict(dict_checkpoint["state_dict_backbone"])
        module_partial_fc.load_state_dict(dict_checkpoint["state_dict_softmax_fc"])
        opt.load_state_dict(dict_checkpoint["state_optimizer"])
        lr_scheduler.load_state_dict(dict_checkpoint["state_lr_scheduler"])
        del dict_checkpoint

    if rank == 0:
        for key, value in cfg.items():
            num_space = 25 - len(key)
            logging.info(": " + key + " " * num_space + str(value))

    ver_prefix = getattr(cfg, "val_dir", None) or cfg.rec
    callback_verification = CallBackVerification(
        val_targets=cfg.val_targets, rec_prefix=ver_prefix, summary_writer=summary_writer, wandb_logger=wandb_logger
    )
    callback_logging = CallBackLogging(
        frequent=cfg.frequent,
        total_step=cfg.total_step,
        batch_size=cfg.batch_size,
        start_step=global_step,
        writer=summary_writer,
    )

    loss_am = AverageMeter()

    # Create ParallelLoader for TPU
    para_loader = pl.ParallelLoader(train_loader, [device])

    for epoch in range(start_epoch, cfg.num_epoch):
        if isinstance(train_loader, DataLoader):
            sampler = getattr(train_loader, "sampler", None)
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

        for _, (img, local_labels) in enumerate(para_loader.per_device_loader(device)):
            global_step += 1
            
            # Data is already on TPU device via ParallelLoader
            local_embeddings = backbone(img)
            loss: torch.Tensor = module_partial_fc(local_embeddings, local_labels)

            # Check for NaN/Inf in loss
            if not torch.isfinite(loss):
                print(f"Loss is NaN/Inf at step {global_step} (epoch {epoch}).")

            # Use automatic mixed precision if configured
            if cfg.amp is not None:
                with torch.cuda.amp.autocast(dtype=cfg.amp):
                    loss.backward()
            else:
                loss.backward()

            if global_step % cfg.gradient_acc == 0:
                # Check for NaN/Inf in gradients
                if rank == 0:
                    for p in backbone.parameters():
                        if p.grad is not None and not torch.isfinite(p.grad).all():
                            print(f"Gradient is NaN/Inf at step {global_step} (epoch {epoch}).")

                torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                
                # Use XLA optimizer step
                xm.optimizer_step(opt)
                opt.zero_grad()

            lr_scheduler.step()
            
            # Mark step for XLA graph compilation
            xm.mark_step()

            with torch.no_grad():
                if wandb_logger:
                    wandb_logger.log(
                        {
                            "Loss/Step Loss": loss.item(),
                            "Loss/Train Loss": loss_am.avg,
                            "Process/Step": global_step,
                            "Process/Epoch": epoch,
                        }
                    )

                if loss.item() > 0:
                    loss_am.update(loss.item(), 1)
                callback_logging(
                    global_step, loss_am, epoch, cfg.amp is not None, lr_scheduler.get_last_lr()[0], None
                )

                if global_step % cfg.verbose == 0 and global_step > 0:
                    callback_verification(global_step, backbone)

        if cfg.save_all_states:
            checkpoint = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "state_dict_backbone": backbone.state_dict(),
                "state_dict_softmax_fc": module_partial_fc.state_dict(),
                "state_optimizer": opt.state_dict(),
                "state_lr_scheduler": lr_scheduler.state_dict(),
            }
            xm.save(checkpoint, os.path.join(cfg.output, f"checkpoint_tpu_{rank}.pt"))

        if rank == 0:
            path_module = os.path.join(cfg.output, f"model_{epoch}.pt")
            xm.save(backbone.state_dict(), path_module)

            if wandb_logger and cfg.save_artifacts:
                artifact_name = f"{run_name}_E{epoch}"
                model = wandb.Artifact(artifact_name, type="model")
                model.add_file(path_module)
                wandb_logger.log_artifact(model)

    if rank == 0:
        path_module = os.path.join(cfg.output, "model.pt")
        xm.save(backbone.state_dict(), path_module)

        if wandb_logger and cfg.save_artifacts:
            artifact_name = f"{run_name}_Final"
            model = wandb.Artifact(artifact_name, type="model")
            model.add_file(path_module)
            wandb_logger.log_artifact(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed Arcface Training on TPU with PyTorch/XLA")
    parser.add_argument("config", type=str, help="py config file")
    args = parser.parse_args()
    
    # Launch training on TPU cores
    # For TPU v5e-8, this will spawn 8 processes (one per core)
    xmp.spawn(_mp_fn, args=(args,), nprocs=None)  # nprocs=None uses all available TPU cores
