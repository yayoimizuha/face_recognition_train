"""
Fine-tuning script for face recognition using the helloproject-face-dataset dataset.
Single GPU, ArcFace loss, FastViT-S12 backbone with GWAP.

Architecture:
    backbone (timm fastvit_s12 features) → GWAP → Linear+BN+SiLU+Dropout → Linear → embedding

After ArcFace training, a lightweight binary anomaly classifier is fitted:
    - Input feature: GWAP output (1024-dim backbone features, before the embedding head)
    - Positive examples: yayoimizuha/helloproject-face-dataset (all samples)
    - Negative examples: yayoimizuha/helloproject-face-errors
                       + tonyassi/celebrity-1000 (CELEBRITY_NEG_SAMPLES samples)
    - Classifier (threshold) is stored as nn.Buffer inside the model.
    - In eval mode, forward() returns (cos_logits, anomaly_score) tuple.

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
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from tqdm import tqdm

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
BACKBONE = "timm/fastvit_s12.apple_in1k"
BACKBONE_DIM = 1024  # fastvit_s12 の forward_features 出力チャンネル数
HIDDEN_DIM = 1024  # embedding head の中間層次元
EMB_SIZE = 512
INPUT_SIZE = 224  # 事前学習時と同じ解像度
DROPOUT = 0.3  # embedding head の Dropout 率
NUM_EPOCHS = 150
BATCH_SIZE = 128
LR = 3e-3  # head / GWAP / ArcFace 用
LR_BACKBONE = 3e-4  # 事前学習済み backbone 用（head の 1/10）
WEIGHT_DECAY = 5e-4
ARC_S = 30.0
ARC_M = 0.5
NUM_WORKERS = 16
VAL_RATIO = 0.2  # 学習データの 20% を検証に使用
USE_AMP = True  # torch.autocast (混合精度学習)
AMP_DTYPE = "bf16"  # "bf16" (推奨: Ampere+ GPU) or "fp16"
# bf16: 指数部が fp32 と同じ 8bit → Inf/NaN によるBN running_stats 汚染が発生しない
# fp16: 指数部 5bit (max=65504) → backbone activation が大きいと Inf → BN 汚染の原因
# GradScaler は bf16 では不要 (オーバーフローしないため)
_AMP_TORCH_DTYPE = torch.bfloat16 if AMP_DTYPE == "bf16" else torch.float16
USE_COMPILE = True  # torch.compile (PyTorch 2.0+, 初回 epoch にコンパイルコスト発生)
SAVE_INTERVAL = 10  # この epoch 間隔でもチェックポイントを保存
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "work_dirs")
WANDB_PROJECT = "face-recognition-finetune"

# 異常検知分類器設定
ANOMALY_HIDDEN_DIM = 256  # AnomalyClassifier 中間層次元
ANOMALY_DROPOUT = 0.3  # AnomalyClassifier Dropout 率
ANOMALY_LR = 1e-3  # AnomalyClassifier 学習率
ANOMALY_EPOCHS = 20  # AnomalyClassifier 学習エポック数
CELEBRITY_NEG_SAMPLES = 3000  # tonyassi/celebrity-1000 から使う負例サンプル数

# ImageNet 正規化統計量（事前学習済みモデルに合わせる）
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
class HuggingFaceFaceDataset(Dataset):
    """HuggingFace dataset ラッパー。

    Args:
        hf_dataset: HuggingFace dataset オブジェクト。各アイテムは "image" キーを持つ。
        transform:  画像変換。
        labeled:    True (default) の場合は (img, label) を返す。
                    False の場合は label が存在しないデータ（負例など）に使用し、
                    label=-1 を返す。
    """

    def __init__(self, hf_dataset, transform, labeled: bool = True):
        self.data = hf_dataset
        self.transform = transform
        self.labeled = labeled

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = self.transform(item["image"].convert("RGB"))
        label = int(item["label"]) if self.labeled else -1
        return img, label


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
# Anomaly Classifier
# ──────────────────────────────────────────────
class AnomalyClassifier(nn.Module):
    """軽量二値分類器による異常検知モジュール。

    GWAP 出力（backbone の生特徴、1024次元）を入力として、正例（既知顔）/
    負例（未知顔・ドメイン外）を分類する。

    設計:
        - 入力: GWAP 出力 (B, gwap_dim) — backbone の生特徴（ArcFace head 変換前）
        - 構造: Linear(gwap_dim, hidden) → BN(hidden) → ReLU → Dropout → Linear(hidden, 1)
        - 出力: logit scalar (B,)  sigmoid 後に 0=正常, 1=異常
        - threshold: nn.Buffer として保持（Youden's J で設定）

    GWAP 出力を使う理由:
        embedding head の Linear+BN による変換は「クラス識別に有利な方向」へ
        特徴を圧縮するため、画質・テクスチャ等の品質情報が失われやすい。
        GWAP 出力はその変換前であり、低品質画像・ドメイン外顔の検出に
        より豊富な情報が残っている。

    統計量と閾値は nn.Buffer として保持されるため、state_dict に含まれ、
    .pt ファイルへの保存・復元、および ONNX エクスポートに対応する。
    """

    def __init__(
        self,
        gwap_dim: int,
        hidden_dim: int = ANOMALY_HIDDEN_DIM,
        dropout: float = ANOMALY_DROPOUT,
    ):
        super().__init__()
        self.fc1 = nn.Linear(gwap_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)

        # バッファ: 勾配不要、state_dict に含まれる
        self.register_buffer("threshold", torch.tensor(float("inf")))

    def logit(self, x: torch.Tensor) -> torch.Tensor:
        """sigmoid 前の raw logit を返す（BCEWithLogitsLoss 用）。

        Args:
            x: (B, gwap_dim) — GWAP 出力（backbone 生特徴）
        Returns:
            logit: (B,) — raw logit（sigmoid 適用前）
        """
        h = torch.relu(self.bn1(self.fc1(x)))
        h = self.drop(h)
        return self.fc2(h).squeeze(1)  # (B,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, gwap_dim) — GWAP 出力（backbone 生特徴）
        Returns:
            score: (B,) — 異常スコア（大きいほど異常）。sigmoid(logit) 値。
        """
        return torch.sigmoid(self.logit(x))

    def is_fitted(self) -> bool:
        """閾値が有限値であれば fit 済みとみなす。"""
        return self.threshold.item() != float("inf")


# ──────────────────────────────────────────────
# Face Recognition Model
# backbone (timm FastViT-S12) + GWAP + Head + ArcFace + AnomalyClassifier
# ──────────────────────────────────────────────
class FaceRecognitionModel(nn.Module):
    """
    backbone (timm FastViT-S12 features) + GWAP + embedding head
    + ArcFace Head + AnomalyClassifier を一体化したモデル。

    構造:
        backbone.forward_features(x) → (B, 1024, H, W)
        GWAP                         → (B, 1024)   ← gwap_out (anomaly 入力)
        Linear(1024, 1024)           → BN(1024) → SiLU → Dropout
        Linear(1024, 512)            → embedding (B, 512)  ← 生の embedding

    公開メソッド:
        embed(x)                → 生の embedding (backbone + GWAP + head)
        embed_gwap(x)           → GWAP 出力 (backbone + GWAP のみ、anomaly 用)
        arcface_logits(e, l)    → ArcFace margin 付き logits (loss 計算用)
        cos_logits(e)           → margin なしの cosine logits (推論・val 用)
        anomaly_score(g)        → 異常スコア sigmoid(classifier(gwap_out))

    forward の挙動:
        train モード : arcface_logits を返す（margin 付き、labels 必須）
        eval  モード : (cos_logits, anomaly_score) タプルを返す

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
            nn.BatchNorm1d(
                hidden_dim, eps=1e-3
            ),  # eps: 1e-5→1e-3（fp16精度に対して1e-5は消えうる）
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

        # ── Anomaly Classifier ────────────────────
        # GWAP 出力 (backbone_dim 次元) を入力とする軽量二値分類器
        self.anomaly = AnomalyClassifier(backbone_dim)

    # ------------------------------------------------------------------
    # embed_gwap: backbone + GWAP のみ（anomaly 分類器の入力用）
    # ------------------------------------------------------------------
    def embed_gwap(self, x: torch.Tensor) -> torch.Tensor:
        """画像テンソル → GWAP 出力（backbone 生特徴、anomaly 分類器入力）"""
        feat = self.backbone.forward_features(x)  # (B, backbone_dim, H, W)
        return self.gwap(feat).float()  # (B, backbone_dim)

    # ------------------------------------------------------------------
    # embedding: backbone + GWAP + head (生の embedding、正規化前)
    # ------------------------------------------------------------------
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """画像テンソル → 生の embedding（正規化前）"""
        feat = self.backbone.forward_features(x)  # (B, 1024, H, W)
        pooled = self.gwap(feat)  # (B, 1024)
        # head (BatchNorm1d を含む) は fp32 で実行する。
        # autocast (bf16/fp16) のまま BatchNorm1d に通すと
        # running_var が汚染されて NaN を生むリスクがある。
        return self.head(pooled.float())  # (B, emb_size)

    # ------------------------------------------------------------------
    # ArcFace logits（train 用 — margin 付き）
    # ------------------------------------------------------------------
    @torch.autocast(device_type="cuda", enabled=False)
    def arcface_logits(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        # ArcFace のマージン演算は fp32 で行う。
        # torch.compile + AMP の組み合わせで Inductor が backward の mm に
        # float/half 混在コードを生成する問題を回避する。
        embeddings = embeddings.float()
        emb = F.normalize(embeddings, dim=1)
        w = F.normalize(self.arc_weight.float(), dim=1)

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
    @torch.autocast(device_type="cuda", enabled=False)
    def cos_logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        embeddings = embeddings.float()
        emb = F.normalize(embeddings, dim=1)
        w = F.normalize(self.arc_weight.float(), dim=1)
        return F.linear(emb, w) * self.arc_s

    # ------------------------------------------------------------------
    # anomaly_score: 異常スコア（GWAP 出力を入力とする二値分類器）
    # NOTE: GWAP 出力（backbone 生特徴）を入力として使用する
    # ------------------------------------------------------------------
    def anomaly_score(self, gwap_out: torch.Tensor) -> torch.Tensor:
        """GWAP 出力 → 異常スコア（大きいほど異常、0〜1）"""
        return self.anomaly(gwap_out)

    # ------------------------------------------------------------------
    # forward: train → arcface logits, eval → (cos_logits, anomaly_score)
    # ------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor, labels: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.training:
            emb = self.embed(x)
            if labels is None:
                raise ValueError("train モードでは labels が必須です")
            return self.arcface_logits(emb, labels)
        else:
            # eval モード: GWAP 出力と embedding を両方計算
            feat = self.backbone.forward_features(x)
            gwap_out = self.gwap(feat).float()  # (B, backbone_dim)
            emb = self.head(gwap_out)  # (B, emb_size)
            logits = self.cos_logits(emb)
            score = self.anomaly(gwap_out)  # (B,)
            return logits, score


# ──────────────────────────────────────────────
# Anomaly Classifier Fitting
# ──────────────────────────────────────────────
def fit_anomaly_classifier(
    model: FaceRecognitionModel,
    pos_loader: DataLoader,
    neg_loaders: list[DataLoader],
    device: torch.device,
    epochs: int = ANOMALY_EPOCHS,
    lr: float = ANOMALY_LR,
) -> dict:
    """ArcFace 学習後に軽量二値分類器（AnomalyClassifier）を学習しモデルに書き込む。

    設計:
        - 入力特徴量: GWAP 出力（backbone 生特徴、1024次元）
          embedding head 変換前であり、画質・テクスチャ情報が保存されている。
        - 正例: face-dataset 全体（ラベル=0）
        - 負例: face-errors + celebrity-1000 サンプル（ラベル=1）
        - 損失: BCEWithLogitsLoss(pos_weight = N_neg / N_pos) でクラス不均衡補正
        - ArcFace backbone/head は凍結し、anomaly.fc1, anomaly.bn1, anomaly.fc2 のみ学習

    手順:
        1. 正例・負例の GWAP 特徴量を収集（backbone は frozen, no_grad）
        2. BCEWithLogitsLoss + Adam で AnomalyClassifier を学習
        3. 学習後のスコアで AUC を計算
        4. Youden's J 統計量で最適閾値を決定し threshold バッファを更新

    Args:
        model:       FaceRecognitionModel（eval モードに切り替えて使用）
        pos_loader:  正例 DataLoader（ラベル付き、labeled=True）
        neg_loaders: 負例 DataLoader のリスト（各 DataLoader は labeled=False）
        device:      計算デバイス
        epochs:      AnomalyClassifier 学習エポック数
        lr:          AnomalyClassifier 学習率

    Returns:
        dict with keys: auc, threshold, pos_mean_score, neg_mean_score
    """
    model.eval()

    # ── Step 1: 正例の GWAP 特徴量を収集 ──────────────────────────────
    print("  [AnomalyClassifier] Collecting positive GWAP features...")
    pos_feats = []
    with torch.no_grad():
        for batch in tqdm(pos_loader, desc="  pos gwap", leave=False):
            imgs = batch[0].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type, enabled=USE_AMP, dtype=_AMP_TORCH_DTYPE
            ):
                gwap = model.embed_gwap(imgs)
            pos_feats.append(gwap.float().cpu())
    pos_feats_t = torch.cat(pos_feats, dim=0)  # (N_pos, backbone_dim)
    print(f"  [AnomalyClassifier] Positive features: {pos_feats_t.shape}")

    # ── Step 2: 負例の GWAP 特徴量を収集 ──────────────────────────────
    print("  [AnomalyClassifier] Collecting negative GWAP features...")
    neg_feats = []
    for neg_loader in neg_loaders:
        with torch.no_grad():
            for batch in tqdm(neg_loader, desc="  neg gwap", leave=False):
                imgs = batch[0].to(device, non_blocking=True)
                with torch.autocast(
                    device_type=device.type, enabled=USE_AMP, dtype=_AMP_TORCH_DTYPE
                ):
                    gwap = model.embed_gwap(imgs)
                neg_feats.append(gwap.float().cpu())
    neg_feats_t = torch.cat(neg_feats, dim=0)  # (N_neg, backbone_dim)
    print(f"  [AnomalyClassifier] Negative features: {neg_feats_t.shape}")

    n_pos = pos_feats_t.shape[0]
    n_neg = neg_feats_t.shape[0]

    # ── Step 3: AnomalyClassifier を学習 ──────────────────────────────
    # backbone/head/arc_weight は凍結し anomaly のみ学習する
    print(f"  [AnomalyClassifier] Training classifier ({epochs} epochs)...")
    model.anomaly.train()

    optimizer = torch.optim.Adam(model.anomaly.parameters(), lr=lr)
    # pos_weight = N_neg / N_pos: 正例（ラベル=0）が多いためクラス不均衡を補正
    # BCEWithLogitsLoss の target: 正例=0, 負例=1
    pos_weight = torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 全データをシャッフルして DataLoader を作る
    all_feats = torch.cat([pos_feats_t, neg_feats_t], dim=0)
    all_targets = torch.cat(
        [
            torch.zeros(n_pos, dtype=torch.float32),
            torch.ones(n_neg, dtype=torch.float32),
        ],
        dim=0,
    )

    from torch.utils.data import TensorDataset

    clf_dataset = TensorDataset(all_feats, all_targets)
    clf_loader = DataLoader(
        clf_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=True,
        num_workers=0,  # テンソルデータなので worker 不要
    )

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_steps = 0
        for feat_batch, tgt_batch in clf_loader:
            feat_batch = feat_batch.to(device)
            tgt_batch = tgt_batch.to(device)

            # sigmoid 前の logit を直接計算（BCEWithLogitsLoss に渡す）
            logit = model.anomaly.logit(feat_batch)

            loss = criterion(logit, tgt_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_steps += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"  [AnomalyClassifier] epoch {epoch + 1}/{epochs}  loss={epoch_loss / max(n_steps, 1):.4f}"
            )

    model.anomaly.eval()

    # ── Step 4: AUC 計算と Youden's J で最適閾値を決定 ──────────────────
    print("  [AnomalyClassifier] Computing AUC and optimal threshold (Youden's J)...")
    model.eval()
    with torch.no_grad():
        pos_scores = model.anomaly(pos_feats_t.to(device)).cpu().numpy()
        neg_scores = model.anomaly(neg_feats_t.to(device)).cpu().numpy()

    all_scores = np.concatenate([pos_scores, neg_scores])
    all_labels = np.concatenate(
        [
            np.zeros(len(pos_scores), dtype=np.int32),
            np.ones(len(neg_scores), dtype=np.int32),
        ]
    )

    from sklearn.metrics import roc_curve

    auc = roc_auc_score(all_labels, all_scores)
    fpr, tpr, roc_thresholds = roc_curve(all_labels, all_scores)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    optimal_threshold = float(roc_thresholds[best_idx])

    pos_mean_score = float(pos_scores.mean())
    neg_mean_score = float(neg_scores.mean())
    print(f"  [AnomalyClassifier] AUC={auc:.4f}, threshold={optimal_threshold:.4f}")
    print(
        f"  [AnomalyClassifier] pos_mean_score={pos_mean_score:.4f}, "
        f"neg_mean_score={neg_mean_score:.4f}"
    )

    # ── Step 5: threshold バッファに書き込んでフィット完了 ────────────
    model.anomaly.threshold.copy_(
        torch.tensor(optimal_threshold, dtype=torch.float32, device=device)
    )
    print("  [AnomalyClassifier] Model buffers updated.")

    return {
        "auc": auc,
        "threshold": optimal_threshold,
        "pos_mean_score": pos_mean_score,
        "neg_mean_score": neg_mean_score,
    }


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
    # TF32: Ampere以降のGPUでfp32 matmulをTF32近似で高速化する
    # bf16 autocast内の演算には影響しないが、arcface_logitsなどの
    # fp32強制箇所でわずかに高速化される。精度への影響は実質なし。
    torch.set_float32_matmul_precision("high")

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
            "anomaly_hidden_dim": ANOMALY_HIDDEN_DIM,
            "anomaly_epochs": ANOMALY_EPOCHS,
            "celebrity_neg_samples": CELEBRITY_NEG_SAMPLES,
            "amp_dtype": AMP_DTYPE,
        },
    )

    # Load dataset
    print("Loading dataset...")
    raw = load_dataset("yayoimizuha/helloproject-face-dataset")
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

    full_dataset = HuggingFaceFaceDataset(train_data, transform_train, labeled=True)
    eval_dataset = HuggingFaceFaceDataset(train_data, transform_eval, labeled=True)

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
    anomaly_param_ids = {id(p) for p in model_to_save.anomaly.parameters()}
    head_params = [
        p
        for p in model_to_save.parameters()
        if id(p) not in backbone_param_ids and id(p) not in anomaly_param_ids
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
    # bf16 は値域が fp32 と同等なので GradScaler は不要。
    # fp16 フォールバック時のみ有効化する。
    _use_scaler = USE_AMP and AMP_DTYPE == "fp16"
    scaler = torch.amp.GradScaler(device.type, enabled=_use_scaler, init_scale=2048.0)

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

            with torch.autocast(
                device_type=device.type, enabled=USE_AMP, dtype=_AMP_TORCH_DTYPE
            ):
                logits = model(imgs, labels)
                # ArcFace スケール付き logits は fp16 だと exp() がオーバーフロー
                # するため、CrossEntropy の入力は fp32 にキャストする。
                loss = criterion(logits.float(), labels)

            # nan/inf 時は backward をスキップするが、scaler.update() は呼んで
            # GradScaler の内部カウンタ（scale 昇降の判定）を正常に進める。
            if not torch.isfinite(loss):
                tqdm.write(
                    f"[WARN] epoch={epoch} step={total_steps} "
                    f"loss={loss.item()} — skipped"
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
        # eval モードでは forward() が (cos_logits, anomaly_score) タプルを返す。
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Val   e{epoch}", leave=False):
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.autocast(
                    device_type=device.type, enabled=USE_AMP, dtype=_AMP_TORCH_DTYPE
                ):
                    logits, _anomaly = model(imgs)  # タプルを展開
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
            with torch.autocast(
                device_type=device.type, enabled=USE_AMP, dtype=_AMP_TORCH_DTYPE
            ):
                logits, _anomaly = model(imgs)  # タプルを展開
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

    # ──────────────────────────────────────────────
    # AnomalyClassifier の学習
    # best model の重みをロードしてから学習する
    # ──────────────────────────────────────────────
    print("Fitting AnomalyClassifier (loading model_best.pt)...")
    best_ckpt_path = os.path.join(run_dir, "model_best.pt")
    model_to_save.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    model_to_save.eval()

    # 正例 DataLoader: face-dataset 全体 (transform_eval)
    pos_loader_anomaly = DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # 負例 DataLoader 1: helloproject-face-errors
    print("Loading helloproject-face-errors dataset...")
    error_raw = load_dataset("yayoimizuha/helloproject-face-errors")
    # split 名を動的に解決（通常は "train"）
    error_split_name = list(error_raw.keys())[0]
    error_hf = error_raw[error_split_name]
    print(f"  face-errors split='{error_split_name}', num_images={len(error_hf)}")

    error_dataset = HuggingFaceFaceDataset(error_hf, transform_eval, labeled=False)
    neg_loader_errors = DataLoader(
        error_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # 負例 DataLoader 2: tonyassi/celebrity-1000 からサンプリング
    print(f"Loading tonyassi/celebrity-1000 ({CELEBRITY_NEG_SAMPLES} samples)...")
    celebrity_raw = load_dataset("tonyassi/celebrity-1000", split="train")
    # ランダムサンプリング（シードを固定して再現性を確保）
    rng_indices = (
        np.random.default_rng(42)
        .choice(
            len(celebrity_raw),
            size=min(CELEBRITY_NEG_SAMPLES, len(celebrity_raw)),
            replace=False,
        )
        .tolist()
    )
    celebrity_subset = celebrity_raw.select(rng_indices)
    print(f"  celebrity-1000 sampled: {len(celebrity_subset)} images")

    celebrity_dataset = HuggingFaceFaceDataset(
        celebrity_subset, transform_eval, labeled=False
    )
    neg_loader_celebrity = DataLoader(
        celebrity_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    anomaly_stats = fit_anomaly_classifier(
        model=model_to_save,
        pos_loader=pos_loader_anomaly,
        neg_loaders=[neg_loader_errors, neg_loader_celebrity],
        device=device,
    )

    wandb.log(
        {
            "anomaly/auc": anomaly_stats["auc"],
            "anomaly/threshold": anomaly_stats["threshold"],
            "anomaly/pos_mean_score": anomaly_stats["pos_mean_score"],
            "anomaly/neg_mean_score": anomaly_stats["neg_mean_score"],
        }
    )

    # AnomalyClassifier 込みのモデルを保存
    anomaly_model_path = os.path.join(run_dir, "model_best_with_anomaly.pt")
    torch.save(model_to_save.state_dict(), anomaly_model_path)
    print(f"Model with AnomalyClassifier saved to {anomaly_model_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
