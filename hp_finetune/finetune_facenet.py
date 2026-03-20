"""
Fine-tuning script for face recognition using the helloproject-face-webdatasets dataset.
Single GPU, ArcFace loss, MobileNetV4-Hybrid-Medium backbone with GWAP.

Architecture:
    backbone (timm mobilenetv4_hybrid_medium features) → GWAP → Linear+BN+SiLU+Dropout → Linear → embedding

Usage:
    python hp_finetune/finetune_facenet.py
"""

import argparse
import os
import sys

# backbones など親ディレクトリのモジュールを参照できるようにする
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import shutil
from datetime import datetime
from typing import cast

import matplotlib
import matplotlib_fontja  # noqa: F401 — フォント登録の副作用 import

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from datasets import load_dataset
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from tqdm import tqdm

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
BACKBONE = "timm/mobilenetv4_hybrid_medium.e500_r224_in1k"
BACKBONE_DIM = 960  # mobilenetv4_hybrid_medium の forward_features 出力チャンネル数
HIDDEN_DIM = 1024  # embedding head の中間層次元
EMB_SIZE = 512
INPUT_SIZE = 224  # 事前学習時と同じ解像度
DROPOUT = 0.3  # embedding head の Dropout 率
NUM_EPOCHS = 200
BATCH_SIZE = 128
LR = 2e-3  # head / GWAP / ArcFace 用
LR_BACKBONE = 2e-4  # 事前学習済み backbone 用（head の 1/10）
WEIGHT_DECAY = 5e-4
ARC_S = 30.0
ARC_M = 0.5
NUM_WORKERS = 16
VAL_RATIO = 0.2  # 学習データの 20% を検証に使用
USE_AMP = True  # torch.autocast (混合精度学習)
USE_COMPILE = True  # torch.compile (PyTorch 2.0+, 初回 epoch にコンパイルコスト発生)
SAVE_INTERVAL = 10  # この epoch 間隔でもチェックポイントを保存
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "work_dirs")
WANDB_PROJECT = "face-recognition-finetune"

# ImageNet 正規化統計量（事前学習済みモデルに合わせる）
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


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


transform_train = transforms.Compose(
    [
        transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.87, 1.0), ratio=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomAffine(degrees=20, translate=(0.2, 0.2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        # RandomErasing: 顔の局所特徴を部分的に隠して正則化する
        # p=0.3  : 30% の確率で適用
        # scale  : 消去面積を画像の 2〜10% に制限
        # value='random': ランダムノイズで埋める
        transforms.RandomErasing(
            p=0.3, scale=(0.02, 0.10), ratio=(0.3, 3.3), value="random"
        ),
    ]
)

transform_eval = transforms.Compose(
    [
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


# ──────────────────────────────────────────────
# Global Weighted Average Pooling (GWAP)
# Qiu, "Global Weighted Average Pooling Bridges
#  Pixel-level Localization and Image-level Classification"
# class-agnostic 版 (Section 3.2, Eq.5–8)
# ──────────────────────────────────────────────
class GWAP(nn.Module):
    """Class-agnostic Global Weighted Average Pooling.

    特徴マップ F ∈ (B, C, H, W) に対して、空間位置ごとの重要度スコアを
    学習可能な 1×1 conv → sigmoid → exp で生成し、空間方向に正規化した
    重みで加重平均を取る。出力は (B, C)。

    数式:
        M(x,y) = exp(σ(w · F(x,y) + b))       ... Eq.5
        α(x,y) = M(x,y) / Σ_{x,y} M(x,y)     ... Eq.6
        f = Σ_{x,y} α(x,y) · F(x,y)           ... Eq.8
    """

    def __init__(self, in_channels: int):
        super().__init__()
        # 1×1 conv: (B, C, H, W) → (B, 1, H, W)
        self.score_conv = nn.Conv2d(in_channels, 1, kernel_size=1, bias=True)
        nn.init.xavier_uniform_(self.score_conv.weight)
        nn.init.zeros_(self.score_conv.bias)  # type: ignore[arg-type]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) → (B, C)"""
        # score: (B, 1, H, W)
        score = self.score_conv(x)
        # M(x,y) = exp(sigmoid(score))
        m = torch.exp(torch.sigmoid(score))  # (B, 1, H, W), 値域 [e^0, e^1] = [1, e]
        # α = M / Σ M (空間方向の正規化)
        alpha = m / m.sum(dim=(2, 3), keepdim=True)  # (B, 1, H, W)
        # 加重平均: (B, C, H, W) * (B, 1, H, W) → sum → (B, C)
        out = (x * alpha).sum(dim=(2, 3))
        return out


# ──────────────────────────────────────────────
# Face Recognition Model
# backbone (timm MobileNetV4) + GWAP + Head + ArcFace
# ──────────────────────────────────────────────
class FaceRecognitionModel(nn.Module):
    """
    backbone (timm MobileNetV4-Hybrid-Medium features) + GWAP + embedding head
    + ArcFace Head を一体化したモデル。

    構造:
        backbone.forward_features(x) → (B, 960, H, W)
        GWAP                         → (B, 960)
        Linear(960, 1024)            → BN(1024) → SiLU → Dropout
        Linear(1024, 512)            → embedding (B, 512)

    公開メソッド:
        embed(x)           → embedding (backbone + GWAP + head)
        arcface_logits(e,l) → ArcFace margin 付き logits (loss 計算用)
        cos_logits(e)      → margin なしの cosine logits (推論・val 用)

    forward の挙動:
        train モード : arcface_logits を返す（margin 付き、labels 必須）
        eval  モード : cos_logits を返す（margin なし）

    NOTE: train loss と val loss は margin の有無により非対称になる。
          これは ArcFace の仕様上意図的な挙動であり、両者の絶対値を直接比較
          すべきではない。val loss はあくまで epoch 間の相対的な改善度指標として使う。
    """

    def __init__(
        self,
        backbone_name: str,
        backbone_dim: int,
        hidden_dim: int,
        emb_size: int,
        num_classes: int,
        dropout: float = 0.4,
        arc_s: float = 64.0,
        arc_m: float = 0.5,
    ):
        super().__init__()

        # ── backbone ──────────────────────────────
        # timm の分類ヘッド (conv_head, norm_head, classifier) を除去し、
        # forward_features のみ使う (出力: B, backbone_dim, H, W)
        timm_name = backbone_name.removeprefix("timm/")
        self.backbone = timm.create_model(timm_name, pretrained=True, num_classes=0)
        # num_classes=0 にすると classifier は Identity になるが、
        # MobileNetV3 系は conv_head/norm_head がまだ残る。
        # forward_features() を使えば conv_head の前で止まる。

        # ── GWAP ──────────────────────────────────
        self.gwap = GWAP(backbone_dim)

        # ── Embedding Head ────────────────────────
        self.head = nn.Sequential(
            nn.Linear(backbone_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_size),
        )

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
    # embedding: backbone + GWAP + head
    # ------------------------------------------------------------------
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """画像テンソル → embedding"""
        feat = self.backbone.forward_features(x)  # (B, 960, H, W)
        pooled = self.gwap(feat)  # (B, 960)
        return self.head(pooled)  # (B, emb_size)

    # ------------------------------------------------------------------
    # ArcFace logits（train 用 — margin 付き）
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
    # forward: train → arcface logits, eval → cosine logits (margin なし)
    # ------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor, labels: torch.Tensor | None = None
    ) -> torch.Tensor:
        emb = self.embed(x)
        if self.training:
            if labels is None:
                raise ValueError("train モードでは labels が必須です")
            return self.arcface_logits(emb, labels)
        else:
            return self.cos_logits(emb)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dump-inputs",
        action="store_true",
        help="学習せず、train DataLoader の入力画像を 4x4 グリッドでダンプして終了する",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 固定入力サイズ (224x224) に最適な cuDNN カーネルを自動選択
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
            "backbone_dim": BACKBONE_DIM,
            "hidden_dim": HIDDEN_DIM,
            "emb_size": EMB_SIZE,
            "input_size": INPUT_SIZE,
            "dropout": DROPOUT,
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "lr_backbone": LR_BACKBONE,
            "weight_decay": WEIGHT_DECAY,
            "arc_s": ARC_S,
            "arc_m": ARC_M,
            "val_ratio": VAL_RATIO,
            "use_amp": USE_AMP,
            "pooling": "GWAP",
            "scheduler": "CosineAnnealingLR",
        },
    )

    # Load dataset
    print("Loading dataset...")
    raw = load_dataset("yayoimizuha/helloproject-face-webdatasets")
    train_data = raw["train"]

    # ClassLabel feature があればその names を信頼する。
    # max(label)+1 方式はラベル欠番時に不正確になる。
    label_feature = train_data.features.get("label")
    if hasattr(label_feature, "names"):
        class_names = label_feature.names
        num_classes = len(class_names)
    else:
        num_classes = max(train_data["label"]) + 1
        class_names = [str(i) for i in range(num_classes)]
    print(f"num_classes={num_classes}, num_images={len(train_data)}")

    full_dataset = HuggingFaceFaceDataset(train_data, transform_train)
    eval_dataset = HuggingFaceFaceDataset(train_data, transform_eval)

    # StratifiedShuffleSplit でクラス比率を保ちながら train/val を固定分割する
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=VAL_RATIO, random_state=42)
    labels_array = train_data["label"]
    train_indices, val_indices = next(
        splitter.split(range(len(full_dataset)), labels_array)
    )

    train_loader = DataLoader(
        Subset(full_dataset, train_indices),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        Subset(eval_dataset, val_indices),
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )

    # ── DEBUG: --dump-inputs 時のみ、train DataLoader の入力をダンプして終了 ──
    if args.dump_inputs:
        N_DUMP_BATCHES = 3  # 出力する画像枚数（バッチ数）
        GRID_N = 4  # 4x4 グリッド = 16 サンプル / 画像

        debug_iter = iter(train_loader)
        for batch_idx in range(N_DUMP_BATCHES):
            imgs, labels = next(debug_iter)
            imgs = imgs[: GRID_N * GRID_N]
            labels = labels[: GRID_N * GRID_N]

            fig, axes = plt.subplots(GRID_N, GRID_N, figsize=(16, 16))
            for i, (img, label) in enumerate(zip(imgs, labels)):
                ax = axes[i // GRID_N][i % GRID_N]
                # 逆 Normalize: ImageNet stats を元に戻す
                mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
                std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
                img_np = ((img * std + mean).permute(1, 2, 0).numpy()).clip(0.0, 1.0)
                ax.imshow(img_np)
                ax.set_title(class_names[label.item()], fontsize=8)
                ax.axis("off")

            fig.suptitle(
                f"Train Input Batch {batch_idx} (with RandomErasing)", fontsize=12
            )
            plt.tight_layout()
            save_path = os.path.join(run_dir, f"debug_input_batch{batch_idx}.png")
            fig.savefig(save_path, dpi=100)
            plt.close(fig)
            print(f"Saved: {save_path}")

        print("Debug dump complete. Exiting without training.")
        sys.exit(0)
    # ── DEBUG END ──────────────────────────────────────────────────────────────

    # Model
    # torch.compile 後は state_dict のキーに _orig_mod. プレフィックスが付く
    # ことがあるため、compile 前のモデル参照を保持して保存に使う。
    model = FaceRecognitionModel(
        backbone_name=BACKBONE,
        backbone_dim=BACKBONE_DIM,
        hidden_dim=HIDDEN_DIM,
        emb_size=EMB_SIZE,
        num_classes=num_classes,
        dropout=DROPOUT,
        arc_s=ARC_S,
        arc_m=ARC_M,
    ).to(device)
    model_to_save = model  # compile 前の参照を保持

    if USE_COMPILE:
        model = cast(nn.Module, torch.compile(model))

    # ── Optimizer: backbone と head/GWAP/ArcFace で学習率を分ける ──
    # NOTE: model_to_save (compile 前) からパラメータを取得する。
    # torch.compile 後の model.parameters() は同じテンソルを返すが、
    # 一貫性のため compile 前の参照を使う。
    backbone_params = list(model_to_save.backbone.parameters())
    backbone_param_ids = {id(p) for p in backbone_params}
    head_params = [
        p for p in model_to_save.parameters() if id(p) not in backbone_param_ids
    ]

    optimizer = torch.optim.SGD(
        [
            {"params": backbone_params, "lr": LR_BACKBONE},
            {"params": head_params, "lr": LR},
        ],
        momentum=0.9,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
        eta_min=1e-6,
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(device.type, enabled=USE_AMP, init_scale=2048.0)

    best_val_acc = 0.0
    global_step = 0

    # Training loop
    epoch_bar = tqdm(range(NUM_EPOCHS), desc="Epochs")
    for epoch in epoch_bar:
        # --- train ---
        model.train()

        total_loss = 0.0
        total_steps = 0
        train_bar = tqdm(train_loader, desc=f"Train e{epoch}", leave=False)
        for imgs, labels in train_bar:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=USE_AMP):
                logits = model(imgs, labels)
                # ArcFace スケール付き logits は fp16 だと exp() がオーバーフロー
                # するため、CrossEntropy の入力は fp32 にキャストする。
                loss = criterion(logits.float(), labels)

            # nan/inf 時は zero_grad して continue するだけ。
            # scaler.update() を backward() なしで呼ぶと内部状態が壊れる。
            if not torch.isfinite(loss):
                tqdm.write(
                    f"[WARN] epoch={epoch} step={total_steps} "
                    f"loss={loss.item()} — skipped"
                )
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            loss_val = loss.item()
            total_loss += loss_val
            total_steps += 1
            train_bar.set_postfix(loss=f"{loss_val:.4f}")

            if global_step % 100 == 0:
                wandb.log(
                    {"train/loss": loss_val, "epoch": epoch},
                    step=global_step,
                )
            global_step += 1

        scheduler.step()
        avg_loss = total_loss / total_steps if total_steps > 0 else float("nan")

        # --- val ---
        # NOTE: val では margin なしの cos_logits で CrossEntropy を計算する。
        # ArcFace の margin がないため train loss より低くなるのが正常。
        # 絶対値を比較するのではなく、epoch 間の val loss/acc の推移で判断する。
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Val   e{epoch}", leave=False):
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, enabled=USE_AMP):
                    logits = model(imgs)
                # ArcFace スケール (arc_s) 付きの logits は値域が大きく
                # fp16 のまま CrossEntropy に渡すと exp() がオーバーフローする。
                # fp32 にキャストしてから loss を計算する。
                loss = criterion(logits.float(), labels)
                val_loss_sum += loss.item()
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        avg_val_loss = val_loss_sum / len(val_loader)
        val_acc = val_correct / val_total

        epoch_bar.set_postfix(
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
                "epoch": epoch,
            },
            step=global_step,
        )

        # ベストモデルの保存 + 定期チェックポイント
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                model_to_save.state_dict(),
                os.path.join(run_dir, "model_best.pt"),
            )

        if (epoch + 1) % SAVE_INTERVAL == 0:
            torch.save(
                model_to_save.state_dict(),
                os.path.join(run_dir, f"model_epoch{epoch}.pt"),
            )

    torch.save(model_to_save.state_dict(), os.path.join(run_dir, "model_final.pt"))
    print(f"Done. Best val_acc={best_val_acc:.4f}. Model saved to {run_dir}")

    # ──────────────────────────────────────────────
    # Confusion matrix (train データ全体で推論)
    # ──────────────────────────────────────────────
    print("Generating confusion matrix...")
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
            imgs = imgs.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=USE_AMP):
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
