import argparse
import os
import sys
import time

import torch

# プロジェクトルートをPYTHONPATHに追加
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.utils_config import get_config
from dataset import get_dataloader


def _init_dist_for_single_process(backend: str, local_rank: int = 0):
    import torch.distributed as dist
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method="tcp://127.0.0.1:29645",
            rank=0,
            world_size=1,
        )
    # デバイス設定
    if backend == "nccl" and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return dist


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Check DALI dataloader")
    parser.add_argument("config", type=str, help="path to config .py")
    parser.add_argument("--num-batches", type=int, default=5, help="batches to iterate")
    parser.add_argument("--save-dir", type=str, default="", help="optional dir to save sample images")
    parser.add_argument("--samples-per-batch-to-save", type=int, default=4, help="images to save per batch")
    args = parser.parse_args()

    cfg = get_config(args.config)

    if not torch.cuda.is_available():
        raise RuntimeError("DALI requires CUDA. Enable CUDA to run this check.")

    backend = "nccl"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    _ = _init_dist_for_single_process(backend=backend, local_rank=local_rank)
    device = torch.device("cuda", local_rank)

    # DataLoader 準備（DALI前提）
    dataset_type = getattr(cfg, "dataset_type", "webdataset")
    batch_size = getattr(cfg, "batch_size", 128)
    dali_aug = getattr(cfg, "dali_aug", False)
    num_workers = getattr(cfg, "num_workers", 2)
    rec = cfg.rec

    print(f"[INFO] Building DALI loader: dataset_type={dataset_type}, batch_size={batch_size}, rec={rec}")
    loader = get_dataloader(
        root_dir=rec,
        local_rank=local_rank,
        batch_size=batch_size,
        dali=True,
        dali_aug=dali_aug,
        seed=getattr(cfg, "seed", 2048),
        num_workers=num_workers,
        dataset_type=dataset_type,
        device_type="cuda",
    )

    # 保存ディレクトリ
    save_dir = args.save_dir
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    total_images = 0
    t0 = time.time()
    try:
        for bi, (imgs, labels) in enumerate(loader):
            if bi >= args.num_batches:
                break
            # imgs: [N, C, H, W], labels: [N]
            n, c, h, w = imgs.shape
            imgs_device = str(imgs.device)
            labels_device = str(labels.device)
            finite = torch.isfinite(imgs).all().item()
            lmin = int(labels.min().item())
            lmax = int(labels.max().item())
            nunique = int(labels.unique().numel())
            total_images += n

            print(
                f"[Batch {bi}] data={n}x{c}x{h}x{w} {imgs.dtype} {imgs_device}, "
                f"label={labels.shape} {labels.dtype} {labels_device}, "
                f"finite={finite}, label[min,max]=[{lmin},{lmax}], unique={nunique}"
            )

            # 任意保存
            if save_dir and args.samples_per_batch_to_save > 0:
                k = min(args.samples_per_batch_to_save, n)
                # [-1,1] -> [0,1]
                vis = (imgs[:k].float().clamp(-1, 1) + 1.0) * 0.5
                # save_imageはCPU/GPU両対応
                try:
                    from torchvision.utils import save_image
                    for i in range(k):
                        out_path = os.path.join(save_dir, f"batch{bi:03d}_idx{i:02d}_label{int(labels[i].item())}.png")
                        save_image(vis[i], out_path)
                except Exception as e:
                    print(f"[WARN] saving images skipped: {e}")

            torch.cuda.synchronize(device)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        # DALI iterator は明示resetが必要
        reset_fn = getattr(loader, "reset", None)
        if callable(reset_fn):
            reset_fn()

    dt = time.time() - t0
    ips = total_images / max(dt, 1e-6)
    print(f"[DONE] Batches={min(args.num_batches, bi+1)}, Images={total_images}, Time={dt:.3f}s, IPS={ips:.1f}/s")


if __name__ == "__main__":
    # cuDNNの自動最適化
    if hasattr(torch.backends, "cudnn") and torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True
    main()
