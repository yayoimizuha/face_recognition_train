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
import kornia.augmentation as K
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
EMB_DROPOUT = 0.4  # Embedding → ArcFace Head 間の Dropout 率
RE_P = 0.5  # Random Erasing の適用確率
RE_SCALE = (0.02, 0.33)  # Random Erasing で消去する面積の割合
RE_RATIO = (0.3, 3.3)  # Random Erasing の縦横比の範囲
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


# Kornia Random Erasing: テンソル (C, H, W) を受け取り (C, H, W) を返すラッパー
_kornia_random_erasing = K.RandomErasing(
    p=RE_P,
    scale=RE_SCALE,
    ratio=RE_RATIO,
    same_on_batch=False,
)


def _apply_random_erasing(tensor: torch.Tensor) -> torch.Tensor:
    """(C, H, W) → unsqueeze → Kornia RandomErasing → squeeze して返す"""
    return _kornia_random_erasing(tensor.unsqueeze(0)).squeeze(0)


transform_train = transforms.Compose(
    [
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        transforms.Lambda(_apply_random_erasing),
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
# Face Recognition Model
# backbone → Dropout → ArcFace Head を 1 クラスに集約
# ──────────────────────────────────────────────
class FaceRecognitionModel(nn.Module):
    """
    backbone (iResNet 等) + Embedding Dropout + ArcFace Head を一体化したモデル。

    - train forward : embeddings → dropout → arcface logits を返す（loss 計算用）
    - eval  forward : embeddings（dropout なし）を返す（cos 類似度での推論用）
    - self.arc_weight プロパティで ArcFace の重みテンソルにアクセス可能
    """

    def __init__(
        self,
        backbone_name: str,
        emb_size: int,
        num_classes: int,
        dropout_p: float = 0.4,
        arc_s: float = 64.0,
        arc_m: float = 0.5,
    ):
        super().__init__()

        # ── backbone ──────────────────────────────
        self.backbone = get_model(
            backbone_name, dropout=0.0, amp=None, num_features=emb_size
        )

        # ── Embedding Dropout ─────────────────────
        self.dropout = nn.Dropout(p=dropout_p)

        # ── ArcFace Head ──────────────────────────
        self.arc_s = arc_s
        self.arc_m = arc_m
        self.arc_weight = nn.Parameter(torch.FloatTensor(num_classes, emb_size))
        nn.init.xavier_uniform_(self.arc_weight)

        self._cos_m = math.cos(arc_m)
        self._sin_m = math.sin(arc_m)
        self._th = math.cos(math.pi - arc_m)  # cos(π - m)
        self._mm = math.sin(math.pi - arc_m) * arc_m

    # ------------------------------------------------------------------
    # embedding: backbone + dropout（train 時のみ dropout が有効）
    # ------------------------------------------------------------------
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """画像テンソル → L2 正規化前 embedding（dropout 込み）"""
        return self.dropout(self.backbone(x))

    # ------------------------------------------------------------------
    # ArcFace logits（train 用）
    # ------------------------------------------------------------------
    def arcface_logits(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        emb = F.normalize(embeddings, dim=1)
        w = F.normalize(self.arc_weight, dim=1)

        # fp16 autocast 下では cos_theta が [-1,1] をわずかに超えるため clamp
        cos_theta = F.linear(emb, w).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        sin_theta = (1.0 - cos_theta**2).clamp(0, 1).sqrt()

        phi = cos_theta * self._cos_m - sin_theta * self._sin_m
        phi = torch.where(cos_theta > self._th, phi, cos_theta - self._mm)

        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        logits = (one_hot * phi + (1.0 - one_hot) * cos_theta) * self.arc_s
        return logits

    # ------------------------------------------------------------------
    # cos 類似度スコア（val / 推論用、margin なし）
    # ------------------------------------------------------------------
    def cos_logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        emb = F.normalize(embeddings, dim=1)
        w = F.normalize(self.arc_weight, dim=1)
        return F.linear(emb, w) * self.arc_s

    # ------------------------------------------------------------------
    # forward: train → arcface logits, eval → normalized embedding
    # ------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor, labels: torch.Tensor | None = None
    ) -> torch.Tensor:
        emb = self.embed(x)
        if self.training:
            assert labels is not None, "train モードでは labels が必要です"
            return self.arcface_logits(emb, labels)
        else:
            return self.cos_logits(emb)


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
            "emb_dropout": EMB_DROPOUT,
            "random_erasing_p": RE_P,
            "random_erasing_scale": RE_SCALE,
            "random_erasing_ratio": RE_RATIO,
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

    full_dataset = HuggingFaceFaceDataset(train_data, transform_train)
    eval_dataset_full = HuggingFaceFaceDataset(train_data, transform_eval)

    # K-Fold の index を事前生成（epoch % K_FOLDS 番目の fold を val に使う）
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    fold_indices = list(kf.split(range(len(full_dataset))))

    # Model
    model = FaceRecognitionModel(
        backbone_name=BACKBONE,
        emb_size=EMB_SIZE,
        num_classes=num_classes,
        dropout_p=EMB_DROPOUT,
        arc_s=ARC_S,
        arc_m=ARC_M,
    ).to(DEVICE)

    # torch.compile で TorchDynamo + Inductor による最適化 (PyTorch 2.0+)
    # torch.compile の返り値は nn.Module と互換だが型推論が FunctionType になるため cast する
    if USE_COMPILE:
        model = cast(nn.Module, torch.compile(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
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
        model.train()

        total_loss = 0.0
        total_steps = 0
        train_bar = tqdm(train_loader, desc=f"Train e{epoch} f{fold_idx}", leave=False)
        for step, (imgs, labels) in enumerate(train_bar):
            imgs = imgs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
                logits = model(imgs, labels)
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
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
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in tqdm(
                val_loader, desc=f"Val   e{epoch} f{fold_idx}", leave=False
            ):
                imgs = imgs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
                    # eval モードでは model.forward が cos_logits を返す
                    logits = model(imgs)
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

        torch.save(model.state_dict(), os.path.join(run_dir, f"model_epoch{epoch}.pt"))

    torch.save(model.state_dict(), os.path.join(run_dir, "model_final.pt"))
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

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in eval_loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            # eval モードでは model.forward が cos_logits を返す
            logits = model(imgs)
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
