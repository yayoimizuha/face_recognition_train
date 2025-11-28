import argparse
import logging
import os
from datetime import datetime

import numpy as np
import torch
from backbones import get_model
from dataset import get_dataloader
from losses import CombinedMarginLoss
from lr_scheduler import PolynomialLRWarmup, DummyScheduler
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
from timm.layers.norm_act import convert_sync_batchnorm
from schedulefree import RAdamScheduleFree
from torch_optimizer import Lamb
import os
import sys

assert (
    torch.__version__ >= "1.12.0"
), "In order to enjoy the features of the new torch, \
we have upgraded the torch to 1.12.0. torch before than 1.12.0 may not work in the future."


# detect device type and choose dist backend accordingly
def _detect_device_type() -> str:
    # Prefer Intel XPU if available
    if torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# Intel MPI/PMI/OMPI 環境変数を PyTorch の env:// 用に補完・マッピング
def _setup_mpi_env_for_torch() -> None:
    env = os.environ
    # RANK
    if "RANK" not in env:
        for k in ("PMI_RANK", "OMPI_COMM_WORLD_RANK", "MPI_RANKID"):
            if k in env:
                env["RANK"] = env[k]
                break
    # WORLD_SIZE
    if "WORLD_SIZE" not in env:
        for k in ("PMI_SIZE", "OMPI_COMM_WORLD_SIZE"):
            if k in env:
                env["WORLD_SIZE"] = env[k]
                break
    # LOCAL_RANK
    if "LOCAL_RANK" not in env:
        for k in ("MPI_LOCALRANKID", "OMPI_COMM_WORLD_LOCAL_RANK"):
            if k in env:
                env["LOCAL_RANK"] = env[k]
                break
    # MASTER_PORT は未設定ならデフォルトを補完
    if (
        "WORLD_SIZE" in env
        and env["WORLD_SIZE"].isdigit()
        and int(env["WORLD_SIZE"]) > 1
    ):
        env.setdefault("MASTER_PORT", "29500")
        env.setdefault("MASTER_ADDR", "127.0.0.1")


def main(args):
    # get config
    cfg = get_config(args.config)
    # global control random seed
    setup_seed(seed=cfg.seed, cuda_deterministic=False)

    # decide device from config or auto-detect（getattr廃止）
    requested_device = cfg.device_type.lower()
    detected_device = _detect_device_type()
    device_type = requested_device
    # validate availability; if requested but unavailable, fall back
    if requested_device == "cuda" and not torch.cuda.is_available():
        logging.warning(
            "Requested device 'cuda' is unavailable. Falling back to detected '%s'",
            detected_device,
        )
        device_type = detected_device
    if requested_device == "xpu" and not torch.xpu.is_available():
        logging.warning(
            "Requested device 'xpu' is unavailable. Falling back to detected '%s'",
            detected_device,
        )
        device_type = detected_device

    # Select backend from config（auto廃止・未知は即クラッシュ）
    rb = cfg.dist_backend.lower()
    if rb not in ("gloo", "nccl", "ccl"):
        raise ValueError(f"Unsupported dist_backend: {cfg.dist_backend}")
    dist_backend = rb

    # backend 可用性の検証とフォールバック
    if dist_backend == "nccl" and not torch.cuda.is_available():
        logging.warning(
            "NCCL requested/selected but CUDA is unavailable. Falling back to 'gloo'."
        )
        dist_backend = "gloo"
    if dist_backend == "ccl":
        try:
            import oneccl_bindings_for_pytorch  # noqa: F401
        except Exception as e:
            logging.warning(
                "oneCCL backend 'ccl' is not available (%s). Falling back to 'gloo'.", e
            )
            dist_backend = "gloo"

    # init process group
    _setup_mpi_env_for_torch()
    try:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        distributed.init_process_group(backend=dist_backend, init_method="env://")
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

    # Try to prefetch W&B run id/name from checkpoint for seamless resume
    resume_wandb_id = None
    resume_wandb_name = None
    if cfg.resume:
        _ckpt_probe = os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt")
        if os.path.exists(_ckpt_probe):
            try:
                _d = torch.load(_ckpt_probe, map_location="cpu", weights_only=False)
                resume_wandb_id = _d.get("wandb_run_id")
                resume_wandb_name = _d.get("wandb_run_name")
            except Exception as e:
                logging.warning("Could not read checkpoint for wandb metadata: %s", e)

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
        run_name = (
            resume_wandb_name
            if resume_wandb_name is not None
            else datetime.now().strftime("%y%m%d_%H%M") + f"_GPU{rank}"
        )
        if cfg.suffix_run_name is not None and resume_wandb_name is None:
            run_name = run_name + f"_{cfg.suffix_run_name}"
        try:
            init_kwargs = dict(
                entity=cfg.wandb_entity,
                project=cfg.wandb_project,
                sync_tensorboard=True,
                name=run_name,
                notes=cfg.notes,
            )
            # If we have a stored run id, resume into that exact run
            if resume_wandb_id is not None:
                init_kwargs.update({"id": resume_wandb_id, "resume": "allow"})
            else:
                init_kwargs.update({"resume": cfg.wandb_resume})

            wandb_logger = wandb.init(**init_kwargs) if (rank == 0 or cfg.wandb_log_all) else None
            if wandb_logger:
                wandb_logger.config.update(cfg)
                # Persist run id alongside outputs for future restarts
                try:
                    with open(os.path.join(cfg.output, "wandb_run.id"), "w") as f:
                        f.write(wandb_logger.id)
                except Exception as e:
                    logging.warning("Failed to write wandb_run.id: %s", e)
        except Exception as e:
            print(
                "WandB Data (Entity and Project name) must be provided in config file (base.py)."
            )
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
        cfg.dataset_type,
        device_type=device_type,
    )

    backbone = get_model(
        cfg.network,
        dropout=0.0,
        amp=cfg.amp,
        num_features=cfg.embedding_size,
        apply_gdconv=cfg.apply_gdconv,
    ).to(device)

    # Convert BatchNorm layers to SyncBatchNorm for proper distributed training
    backbone = convert_sync_batchnorm(backbone)

    ddp_device_ids = [local_rank] if device.type != "cpu" else None
    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone,
        broadcast_buffers=False,
        device_ids=ddp_device_ids,
        bucket_cap_mb=16,
        find_unused_parameters=False,  # _set_static_graph()で未使用パラメータを自動処理するためFalseに
    )
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
        cfg.interclass_filtering_threshold,
    )

    if cfg.optimizer == "sgd":
        module_partial_fc = PartialFC_V2(
            margin_loss,
            cfg.embedding_size,
            cfg.num_classes,
            cfg.sample_rate,
            False,
            amp=cfg.amp,
        )
        module_partial_fc.train().to(device)
        # TODO the params of partial fc must be last in the params list
        opt = torch.optim.SGD(
            params=[
                {"params": backbone.parameters()},
                {"params": module_partial_fc.parameters()},
            ],
            lr=cfg.lr,
            momentum=0.9,
            weight_decay=cfg.weight_decay,
        )

    elif cfg.optimizer == "adamw":
        module_partial_fc = PartialFC_V2(
            margin_loss,
            cfg.embedding_size,
            cfg.num_classes,
            cfg.sample_rate,
            False,
            amp=cfg.amp,
        )
        module_partial_fc.train().to(device)
        betas = tuple(getattr(cfg, "adam_betas", (0.9, 0.999)))
        opt = torch.optim.AdamW(
            params=[
                {"params": backbone.parameters()},
                {"params": module_partial_fc.parameters()},
            ],
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=betas,
        )
    elif cfg.optimizer == "radam_schedulefree":
        module_partial_fc = PartialFC_V2(
            margin_loss,
            cfg.embedding_size,
            cfg.num_classes,
            cfg.sample_rate,
            False,
            amp=cfg.amp,
        )
        module_partial_fc.train().to(device)
        betas = tuple(getattr(cfg, "adam_betas", (0.9, 0.999)))
        opt = RAdamScheduleFree(
            params=[
                {"params": backbone.parameters()},
                {"params": module_partial_fc.parameters()},
            ],
            lr=cfg.lr,
            betas=betas,
            weight_decay=cfg.weight_decay,
        )
    elif cfg.optimizer == "lamb":
        module_partial_fc = PartialFC_V2(
            margin_loss,
            cfg.embedding_size,
            cfg.num_classes,
            cfg.sample_rate,
            False,
            amp=cfg.amp,
        )
        module_partial_fc.train().to(device)
        betas = tuple(getattr(cfg, "adam_betas", (0.9, 0.999)))
        opt = Lamb(
            params=[
                {"params": backbone.parameters()},
                {"params": module_partial_fc.parameters()},
            ],
            lr=cfg.lr,
            betas=betas,
            weight_decay=cfg.weight_decay,
        )
    else:
        raise

    cfg.total_batch_size = cfg.batch_size * world_size
    cfg.steps_per_epoch = cfg.num_image // cfg.total_batch_size
    if cfg.steps_per_epoch == 0:
        raise ValueError(
            "steps_per_epoch computed to 0. Dataset is too small for the configured batch_size * world_size; reduce the total batch size or adjust num_image."
        )
    cfg.warmup_step = cfg.steps_per_epoch * cfg.warmup_epoch
    cfg.total_step = cfg.steps_per_epoch * cfg.num_epoch

    # RAdamScheduleFree doesn't need a separate learning rate scheduler
    if cfg.optimizer == "radam_schedulefree":
        lr_scheduler = DummyScheduler(optimizer=opt)
        opt.train()  # Initialize optimizer in train mode
    else:
        lr_scheduler = PolynomialLRWarmup(
            optimizer=opt, warmup_iters=cfg.warmup_step, total_iters=cfg.total_step
        )

    start_epoch = 0
    global_step = 0
    # Prepare GradScaler early so we can restore its state if present
    amp = GradScaler(
        device=device_type, enabled=(cfg.amp is not None), growth_interval=100
    )
    if cfg.resume:
        checkpoint_path = os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt")
        if os.path.exists(checkpoint_path):
            logging.info("Resuming from checkpoint: %s", checkpoint_path)
            dict_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            start_epoch = dict_checkpoint.get("epoch", 0)
            global_step = dict_checkpoint.get("global_step", 0)
            # Model / module states
            try:
                backbone.module.load_state_dict(dict_checkpoint["state_dict_backbone"])
            except Exception as e:
                logging.error("Failed to load backbone state_dict: %s", e)
                raise
            try:
                module_partial_fc.load_state_dict(dict_checkpoint["state_dict_softmax_fc"])
            except Exception as e:
                logging.error("Failed to load partial FC state_dict: %s", e)
                raise
            # Optimizer / LR scheduler
            state_opt = dict_checkpoint.get("state_optimizer")
            if state_opt is not None:
                try:
                    opt.load_state_dict(state_opt)
                except Exception as e:
                    logging.warning("Could not restore optimizer state: %s", e)
            else:
                logging.warning("Optimizer state not found in checkpoint.")
            state_sched = dict_checkpoint.get("state_lr_scheduler")
            if state_sched is not None:
                try:
                    lr_scheduler.load_state_dict(state_sched)
                except Exception as e:
                    logging.warning("Could not restore lr_scheduler state: %s", e)
            else:
                logging.warning("LR scheduler state not found in checkpoint.")
            # GradScaler
            amp_state = dict_checkpoint.get("amp_state")
            if amp_state and amp.is_enabled():
                try:
                    amp.load_state_dict(amp_state)
                except Exception as e:
                    logging.warning("Could not restore GradScaler state: %s", e)
            elif amp.is_enabled():
                logging.warning("GradScaler state absent; starting fresh scale.")
            # RNG states (ensure deterministic continuation of sampling order where possible)
            try:
                rng_state_torch = dict_checkpoint.get("rng_state_torch")
                if rng_state_torch is not None:
                    torch.random.set_rng_state(rng_state_torch)
                rng_state_cuda = dict_checkpoint.get("rng_state_cuda")
                if rng_state_cuda is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(rng_state_cuda)
                rng_state_numpy = dict_checkpoint.get("rng_state_numpy")
                if rng_state_numpy is not None:
                    np.random.set_state(rng_state_numpy)
            except Exception as e:
                logging.warning("Failed to restore RNG states: %s", e)
            del dict_checkpoint
            logging.info(
                "Resumed from epoch %d, global_step %d (AMP scale=%s)",
                start_epoch,
                global_step,
                (amp._get_scale_async() if amp.is_enabled() else "N/A"),
            )
        else:
            logging.info("Checkpoint not found at %s, starting from scratch", checkpoint_path)

    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))

    ver_prefix = getattr(cfg, "val_dir", None) or cfg.rec
    callback_verification = CallBackVerification(
        val_targets=cfg.val_targets,
        rec_prefix=ver_prefix,
        summary_writer=summary_writer,
        wandb_logger=wandb_logger,
    )
    callback_logging = CallBackLogging(
        frequent=cfg.frequent,
        total_step=cfg.total_step,
        batch_size=cfg.batch_size,
        start_step=global_step,
        writer=summary_writer,
    )

    loss_am = AverageMeter()
    # (GradScaler already instantiated before resume; keep reference name 'amp')

    for epoch in range(start_epoch, cfg.num_epoch):

        if isinstance(train_loader, DataLoader):
            sampler = train_loader.sampler
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
        steps_this_epoch = 0
        data_iter = iter(train_loader)
        while steps_this_epoch < cfg.steps_per_epoch:
            try:
                batch = next(data_iter)
            except StopIteration:
                if cfg.dataset_type != "webdataset":
                    raise RuntimeError(
                        f"DataLoader exhausted after {steps_this_epoch} steps, but cfg.steps_per_epoch={cfg.steps_per_epoch}. "
                        "Adjust num_image/steps_per_epoch or ensure the dataset has enough samples."
                    )
                logging.warning(
                    "Rank %d WebDataset iterator exhausted at epoch %d step %d; retrying new iterator.",
                    rank,
                    epoch,
                    steps_this_epoch,
                )
                data_iter = iter(train_loader)
                continue

            img, local_labels = batch
            steps_this_epoch += 1
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

            # 1) 損失のNaN/Inf簡易チェック（シンプルな停止処理）
            if not torch.isfinite(loss):
                print(f"Loss is NaN/Inf at step {global_step} (epoch {epoch}).")

            if amp.is_enabled():
                amp.scale(loss).backward()
                if global_step % cfg.gradient_acc == 0:
                    amp.unscale_(opt)
                    # 2) 勾配のNaN/Inf簡易チェック（unscale後に検査）
                    if distributed.get_rank() == 0:
                        for p in backbone.parameters():
                            if p.grad is not None and not torch.isfinite(p.grad).all():
                                print(
                                    f"Gradient is NaN/Inf at step {global_step} (epoch {epoch})."
                                )

                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    amp.step(opt)
                    amp.update()
                    opt.zero_grad()
            else:
                loss.backward()
                if global_step % cfg.gradient_acc == 0:
                    # 2) 勾配のNaN/Inf簡易チェック（FP32経路）
                    if distributed.get_rank() == 0:
                        for p in backbone.parameters():
                            if p.grad is not None and not torch.isfinite(p.grad).all():
                                print(
                                    f"Gradient is NaN/Inf at step {global_step} (epoch {epoch})."
                                )

                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    opt.step()
                    opt.zero_grad()
            lr_scheduler.step()

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
                    global_step,
                    loss_am,
                    epoch,
                    amp.is_enabled(),
                    lr_scheduler.get_last_lr()[0],
                    amp,
                )

                if global_step % cfg.verbose == 0 and global_step > 0:
                    # Put optimizer in eval mode for verification when using RAdamScheduleFree
                    if cfg.optimizer == "radam_schedulefree":
                        opt.eval()
                    callback_verification(global_step, backbone)
                    # Put optimizer back in train mode
                    if cfg.optimizer == "radam_schedulefree":
                        opt.train()

        if rank == 0:
            logging.info(
                "Epoch %d finished with %d/%d steps (global_step=%d)",
                epoch,
                steps_this_epoch,
                cfg.steps_per_epoch,
                global_step,
            )

        if cfg.save_all_states:
            # Put optimizer in eval mode for checkpoint saving when using RAdamScheduleFree
            if cfg.optimizer == "radam_schedulefree":
                opt.eval()

            checkpoint = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "state_dict_backbone": backbone.module.state_dict(),
                "state_dict_softmax_fc": module_partial_fc.state_dict(),
                "state_optimizer": opt.state_dict(),
                "state_lr_scheduler": lr_scheduler.state_dict(),
                "amp_state": amp.state_dict() if amp.is_enabled() else None,
                "rng_state_torch": torch.random.get_rng_state(),
                "rng_state_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "rng_state_numpy": np.random.get_state(),
                "wandb_run_id": (wandb_logger.id if wandb_logger else None),
                "wandb_run_name": (run_name if 'run_name' in locals() else None),
            }
            torch.save(
                checkpoint, os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt")
            )

            # Put optimizer back in train mode
            if cfg.optimizer == "radam_schedulefree":
                opt.train()

        if rank == 0:
            path_module = os.path.join(cfg.output, f"model_{epoch}.pt")
            torch.save(backbone.module.state_dict(), path_module)

            if wandb_logger and cfg.save_artifacts:
                artifact_name = f"{run_name}_E{epoch}"
                model = wandb.Artifact(artifact_name, type="model")
                model.add_file(path_module)
                wandb_logger.log_artifact(model)

        if dali_enabled:
            reset_fn = getattr(train_loader, "reset", None)
            if callable(reset_fn):
                reset_fn()

    if rank == 0:
        path_module = os.path.join(cfg.output, "model.pt")
        torch.save(backbone.module.state_dict(), path_module)
        logging.info("Training finished successfully. Final model saved: %s", path_module)

        if wandb_logger and cfg.save_artifacts:
            artifact_name = f"{run_name}_Final"
            model = wandb.Artifact(artifact_name, type="model")
            model.add_file(path_module)
            wandb_logger.log_artifact(model)

    # Cleanup to avoid warnings on exit
    del train_loader
    del backbone
    del module_partial_fc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    distributed.destroy_process_group()


if __name__ == "__main__":
    # Enable cudnn benchmark only when available (CUDA)
    if hasattr(torch.backends, "cudnn") and torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="Distributed Arcface Training in Pytorch"
    )
    parser.add_argument("config", type=str, help="py config file")
    main(parser.parse_args())
