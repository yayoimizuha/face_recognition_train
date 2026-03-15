"""
Fine-tuning script for face recognition using the helloproject-face-webdatasets dataset.
Single GPU, ArcFace loss, iResNet-50 backbone.

Usage:
    python hp_finetune/finetune_facenet.py
"""

import os
import sys
from typing import cast

# backbones など親ディレクトリのモジュールを参照できるようにする
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from datasets import load_dataset
from backbones import get_model
import math
from datetime import datetime
import numpy as np
import matplotlib
import matplotlib_fontja

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import KFold
from tqdm import tqdm
import wandb

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
BACKBONE = "timm/resnet50.a1_in1k"
EMB_SIZE = 512
NUM_EPOCHS = 200
BATCH_SIZE = 128
LR = 2e-3
WEIGHT_DECAY = 5e-4
ARC_S = 64.0
ARC_M = 0.5
NUM_WORKERS = 16
K_FOLDS = 5
USE_AMP = True  # torch.autocast (混合精度学習)
USE_COMPILE = True  # torch.compile (PyTorch 2.0+, 初回 epoch にコンパイルコスト発生)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "work_dirs")
WANDB_PROJECT = "face-recognition-finetune"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
class HuggingFaceFaceDataset(Dataset):
    def __init__(self, hf_dataset, transform):
        self.data = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = item["image"].convert("RGB")
        label = item["label"]
        return self.transform(img), label


transform = transforms.Compose(
    [
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

transform_eval = transforms.Compose(
    [
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


# ──────────────────────────────────────────────
# ArcFace head (single GPU 版)
# ──────────────────────────────────────────────
class ArcFaceHead(nn.Module):
    def __init__(self, embedding_size, num_classes, s=64.0, m=0.5):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)  # cos(π - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embeddings, labels):
        # normalize
        embeddings = F.normalize(embeddings, dim=1)
        weight = F.normalize(self.weight, dim=1)

        # fp16 autocast 下では cos_theta が [-1,1] をわずかに超えることがあるため clamp
        cos_theta = F.linear(embeddings, weight).clamp(
            -1.0 + 1e-7, 1.0 - 1e-7
        )  # (B, C)
        sin_theta = (1.0 - cos_theta**2).clamp(0, 1).sqrt()

        # cos(θ + m)
        phi = cos_theta * self.cos_m - sin_theta * self.sin_m
        # easy-margin 相当: θ > π-m の場合は cos_theta - mm を使う
        phi = torch.where(cos_theta > self.th, phi, cos_theta - self.mm)

        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        logits = one_hot * phi + (1.0 - one_hot) * cos_theta
        logits *= self.s
        return logits


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    # 固定入力サイズ (112×112) に最適な cuDNN カーネルを自動選択
    torch.backends.cudnn.benchmark = True

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_DIR, run_timestamp)
    os.makedirs(run_dir, exist_ok=True)

    # 実行スクリプトのコピーを保存
    shutil.copy2(__file__, os.path.join(run_dir, os.path.basename(__file__)))

    wandb.init(
        project=WANDB_PROJECT,
        name=f"run_{run_timestamp}",
        config={
            "backbone": BACKBONE,
            "emb_size": EMB_SIZE,
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "arc_s": ARC_S,
            "arc_m": ARC_M,
            "k_folds": K_FOLDS,
            "use_amp": USE_AMP,
        },
    )

    # Load dataset
    print("Loading dataset...")
    raw = load_dataset("yayoimizuha/helloproject-face-webdatasets")
    train_data = raw["train"]

    num_classes = max(train_data["label"]) + 1
    print(f"num_classes={num_classes}, num_images={len(train_data)}")

    # クラス名（存在すれば使用、なければ str(id) にフォールバック）
    label_feature = train_data.features.get("label")
    if hasattr(label_feature, "names"):
        class_names = label_feature.names
    else:
        class_names = [str(i) for i in range(num_classes)]

    full_dataset = HuggingFaceFaceDataset(train_data, transform)
    eval_dataset_full = HuggingFaceFaceDataset(train_data, transform_eval)

    # K-Fold の index を事前生成（epoch % K_FOLDS 番目の fold を val に使う）
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    fold_indices = list(kf.split(range(len(full_dataset))))

    # Model
    backbone = get_model(BACKBONE, dropout=0.0, amp=None, num_features=EMB_SIZE).to(
        DEVICE
    )
    head = ArcFaceHead(EMB_SIZE, num_classes, s=ARC_S, m=ARC_M).to(DEVICE)

    # torch.compile で TorchDynamo + Inductor による最適化 (PyTorch 2.0+)
    # torch.compile の返り値は nn.Module と互換だが型推論が FunctionType になるため cast する
    if USE_COMPILE:
        backbone = cast(nn.Module, torch.compile(backbone))
        head = cast(nn.Module, torch.compile(head))

    optimizer = torch.optim.SGD(
        [{"params": backbone.parameters()}, {"params": head.parameters()}],
        lr=LR,
        momentum=0.9,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[
            int(NUM_EPOCHS * 0.6),
            int(NUM_EPOCHS * 0.75),
            int(NUM_EPOCHS * 0.9),
        ],
        gamma=0.7,
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(DEVICE.type, enabled=USE_AMP, init_scale=2048.0)

    # Training loop（epoch ごとに使う fold を切り替え）
    epoch_bar = tqdm(range(NUM_EPOCHS), desc="Epochs")
    for epoch in epoch_bar:
        fold_idx = epoch % K_FOLDS
        train_indices, val_indices = fold_indices[fold_idx]

        train_loader = DataLoader(
            Subset(full_dataset, train_indices),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            Subset(eval_dataset_full, val_indices),
            batch_size=BATCH_SIZE * 2,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        # --- train ---
        backbone.train()
        head.train()

        total_loss = 0.0
        total_steps = 0
        train_bar = tqdm(train_loader, desc=f"Train e{epoch} f{fold_idx}", leave=False)
        for step, (imgs, labels) in enumerate(train_bar):
            imgs = imgs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
                embeddings = backbone(imgs)
                logits = head(embeddings, labels)
                loss = criterion(logits, labels)

            loss_val = loss.item()
            if math.isnan(loss_val) or math.isinf(loss_val):
                tqdm.write(
                    f"[WARN] epoch={epoch} fold={fold_idx} step={step} loss={loss_val} — skipped"
                )
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(backbone.parameters()) + list(head.parameters()), 5.0
            )
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss_val
            total_steps += 1
            train_bar.set_postfix(loss=f"{loss_val:.4f}")
            if step % 100 == 0:
                wandb.log(
                    {"train/loss": loss_val, "fold": fold_idx, "epoch": epoch},
                    step=epoch * len(train_loader) + step,
                )

        scheduler.step()
        avg_loss = total_loss / max(total_steps, 1)

        # --- val ---
        backbone.eval()
        head.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        # val では labels を ArcFace head に渡さず、cos similarity で分類する
        weight_norm = F.normalize(head.weight, dim=1)
        with torch.no_grad():
            for imgs, labels in tqdm(
                val_loader, desc=f"Val   e{epoch} f{fold_idx}", leave=False
            ):
                imgs = imgs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
                    embeddings = backbone(imgs)
                    emb_norm = F.normalize(embeddings, dim=1)
                    logits = F.linear(emb_norm, weight_norm) * ARC_S
                    val_loss += criterion(logits, labels).item()
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total

        epoch_bar.set_postfix(
            fold=fold_idx,
            train_loss=f"{avg_loss:.4f}",
            val_loss=f"{avg_val_loss:.4f}",
            val_acc=f"{val_acc:.4f}",
            lr=f"{scheduler.get_last_lr()[0]:.6f}",
        )
        wandb.log(
            {
                "train/avg_loss": avg_loss,
                "val/loss": avg_val_loss,
                "val/acc": val_acc,
                "train/lr": scheduler.get_last_lr()[0],
                "fold": fold_idx,
                "epoch": epoch,
            },
            step=(epoch + 1) * len(train_loader),
        )

        torch.save(
            backbone.state_dict(), os.path.join(run_dir, f"backbone_epoch{epoch}.pt")
        )

    torch.save(backbone.state_dict(), os.path.join(run_dir, "backbone_final.pt"))
    print("Done. Model saved to", run_dir)

    # ──────────────────────────────────────────────
    # Confusion matrix (train データで推論)
    # ──────────────────────────────────────────────
    print("Generating confusion matrix...")
    eval_dataset = HuggingFaceFaceDataset(train_data, transform_eval)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    backbone.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in eval_loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            embeddings = backbone(imgs)
            # ArcFace head の weight を nearest-neighbor 分類器として流用
            weight = F.normalize(head.weight, dim=1)
            emb = F.normalize(embeddings, dim=1)
            logits = F.linear(emb, weight)  # cos similarity → argmax で予測
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

    # 図サイズはクラス数に合わせてスケール（最低 8、最大 60 インチ）
    fig_size = max(8, min(60, num_classes // 3))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(
        ax=ax, xticks_rotation=90, colorbar=False, values_format="d", cmap="Blues"
    )
    ax.set_title("Confusion Matrix (train set)")
    plt.tight_layout()

    cm_path = os.path.join(run_dir, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=100)
    plt.close(fig)
    print(f"Confusion matrix saved to {cm_path}")

    wandb.log({"confusion_matrix": wandb.Image(cm_path)})

    wandb.finish()


if __name__ == "__main__":
    main()
