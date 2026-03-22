"""
Fine-tuning script for face recognition using the helloproject-face-dataset dataset.
Single GPU, ArcFace loss, MobileNetV4-Hybrid-Medium backbone with GWAP.

Architecture:
    backbone (timm mobilenetv4_hybrid_medium features) → GWAP → Linear+BN+SiLU+Dropout → Linear → embedding

After ArcFace training, Mahalanobis distance-based anomaly detection is fitted:
    - Positive examples: yayoimizuha/helloproject-face-dataset (all samples)
    - Negative examples: yayoimizuha/helloproject-face-errors (for threshold calibration)
    - Statistics (mean, precision matrix, threshold) are stored as nn.Buffers inside the model.
    - In eval mode, forward() returns (cos_logits, anomaly_score) tuple.
    - Mahalanobis distance is computed on raw (non-normalized) embeddings for maximum precision.

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
BACKBONE = "timm/mobilenetv4_hybrid_medium.e500_r224_in1k"
BACKBONE_DIM = 960  # mobilenetv4_hybrid_medium の forward_features 出力チャンネル数
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

# Mahalanobis 異常検知設定
MAHAL_REG_LAMBDA = 1e-5  # 共分散行列の正則化項 (特異行列回避)

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
# Mahalanobis Distance Layer
# ──────────────────────────────────────────────
class MahalanobisLayer(nn.Module):
    """Multi-class マハラノビス距離計算モジュール（Tied covariance）。

    fit() でクラスごとの embedding 平均と Pooled within-class 精度行列を設定し、
    forward() で各 embedding の「最近クラスへのマハラノビス距離」を返す。

    設計:
        - class_means: (num_classes, D) — クラスごとの embedding 平均
        - precision:   (D, D)          — 全クラス共通の Pooled within-class 精度行列
        - threshold:   scalar          — 異常検知閾値（Youden's J で設定）

    推論:
        d(x) = min_c sqrt((x - μ_c)^T Σ_w^{-1} (x - μ_c))
        スコアが大きいほど異常（全クラスから外れた入力）

    統計量と閾値は nn.Buffer として保持されるため、state_dict に含まれ、
    .pt ファイルへの保存・復元、および ONNX エクスポートに対応する。

    NOTE: 精度を重視して生の（正規化前の）embedding を入力として使用する。
          ArcFace の embedding head 出力（emb_size 次元）をそのまま渡す。
    """

    def __init__(self, emb_size: int, num_classes: int = 1):
        super().__init__()
        self.emb_size = emb_size
        self.num_classes = num_classes
        # バッファ: 勾配不要、state_dict に含まれる
        # class_means: (num_classes, D) — クラスごとの平均ベクトル
        self.register_buffer("class_means", torch.zeros(num_classes, emb_size))
        self.register_buffer("precision", torch.eye(emb_size))
        self.register_buffer("threshold", torch.tensor(float("inf")))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, emb_size) — 生の embedding（正規化前）
        Returns:
            dist: (B,) — 最近クラスへのマハラノビス距離（大きいほど異常）
        """
        # diff: (B, 1, D) - (1, C, D) = (B, C, D)
        diff = x.unsqueeze(1) - self.class_means.unsqueeze(0)
        # (B, C, D) @ (D, D) = (B, C, D)
        left = diff @ self.precision
        dist_sq = (left * diff).sum(dim=2)  # (B, C)
        # 各サンプルについて全クラスの中で最小二乗距離を取る
        min_dist_sq = dist_sq.clamp(min=0.0).min(dim=1).values  # (B,)
        dist = min_dist_sq.sqrt()  # (B,)
        # NaN ガード: 数値的不安定時に 0.0 で置換（距離が算出不能 → 正常扱い）
        # posinf は inf のまま保持して downstream で異常判定させる
        return torch.nan_to_num(dist, nan=0.0, neginf=0.0)

    def is_fitted(self) -> bool:
        """閾値が有限値であれば fit 済みとみなす。"""
        return self.threshold.item() != float("inf")


# ──────────────────────────────────────────────
# Face Recognition Model
# backbone (timm ConvNeXt V2 Large) + GWAP + Head + ArcFace + MahalanobisLayer
# ──────────────────────────────────────────────
class FaceRecognitionModel(nn.Module):
    """
    backbone (timm ConvNeXt V2 Large features) + GWAP + embedding head
    + ArcFace Head + MahalanobisLayer を一体化したモデル。

    構造:
        backbone.forward_features(x) → (B, 1536, H, W)
        GWAP                         → (B, 1536)
        Linear(1536, 1024)           → BN(1024) → SiLU → Dropout
        Linear(1024, 512)            → embedding (B, 512)  ← 生の embedding

    公開メソッド:
        embed(x)                → 生の embedding (backbone + GWAP + head)
        arcface_logits(e, l)    → ArcFace margin 付き logits (loss 計算用)
        cos_logits(e)           → margin なしの cosine logits (推論・val 用)
        mahalanobis_score(e)    → min-class マハラノビス距離 (異常スコア)

    forward の挙動:
        train モード : arcface_logits を返す（margin 付き、labels 必須）
        eval  モード : (cos_logits, anomaly_score) タプルを返す
                       anomaly_score = min_c dist(embed, μ_c)（生の embedding を使用）

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

        # ── Mahalanobis 異常検知 ──────────────────
        self.mahal = MahalanobisLayer(emb_size, num_classes)

    # ------------------------------------------------------------------
    # embedding: backbone + GWAP + head (生の embedding、正規化前)
    # ------------------------------------------------------------------
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """画像テンソル → 生の embedding（正規化前）"""
        feat = self.backbone.forward_features(x)  # (B, 1536, H, W)
        pooled = self.gwap(feat)  # (B, 1536)
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
    # マハラノビス距離（異常スコア）
    # NOTE: 生の embedding（正規化前）を入力として使用する
    # ------------------------------------------------------------------
    def mahalanobis_score(self, embeddings: torch.Tensor) -> torch.Tensor:
        """生の embedding → マハラノビス距離（大きいほど異常）"""
        return self.mahal(embeddings)

    # ------------------------------------------------------------------
    # forward: train → arcface logits, eval → (cos_logits, anomaly_score)
    # ------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor, labels: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        emb = self.embed(x)
        if self.training:
            if labels is None:
                raise ValueError("train モードでは labels が必須です")
            return self.arcface_logits(emb, labels)
        else:
            # eval モード: (cos_logits, anomaly_score) タプルを返す
            logits = self.cos_logits(emb)
            anomaly = self.mahal(emb)  # 生の embedding でマハラノビス距離を計算
            return logits, anomaly


# ──────────────────────────────────────────────
# Mahalanobis Fitting
# ──────────────────────────────────────────────
def fit_mahalanobis(
    model: FaceRecognitionModel,
    pos_loader: DataLoader,
    neg_loader: DataLoader,
    device: torch.device,
    reg_lambda: float = MAHAL_REG_LAMBDA,
) -> dict:
    """ArcFace 学習後に Multi-class Mahalanobis 異常検知の統計量を計算してモデルに書き込む。

    設計: クラスごとの平均ベクトル + Pooled within-class 精度行列（Tied covariance）
        - d(x) = min_c sqrt((x - μ_c)^T Σ_w^{-1} (x - μ_c))
        - ArcFace の特徴空間はクラスが球面上に分散するため、全体平均を使う
          Global Mahalanobis は機能しない。各クラスの重心との距離を使うことで、
          正例（学習済みクラスの顔）は小さく、負例（未知顔・非顔）は大きくなる。

    手順:
        1. 正例 (face-dataset 全体) の生 embedding とラベルを収集
        2. クラスごとの平均 μ_c を計算
        3. Pooled within-class 共分散行列 Σ_w を計算し、精度行列 Σ_w^{-1} を求める
        4. モデルの class_means / precision バッファを更新
        5. MahalanobisLayer 経由で正例・負例それぞれの min-distance を計算
        6. Youden's J 統計量で最適閾値を決定し、threshold バッファを更新

    Args:
        model:      FaceRecognitionModel（eval モードに切り替えて使用）
        pos_loader: 正例 DataLoader（ラベル付き、labeled=True）
        neg_loader: 負例 DataLoader（ラベルなし、labeled=False）
        device:     計算デバイス
        reg_lambda: 共分散行列の対角正則化係数（相対スケーリングのベースライン）

    Returns:
        dict with keys: auc, threshold, pos_mean_dist, neg_mean_dist
    """
    model.eval()

    num_classes = model.mahal.num_classes
    emb_size = model.mahal.emb_size

    # ── Step 1: 正例 embedding とラベルを収集 ──────────────────────────
    print("  [Mahalanobis] Collecting positive embeddings with labels...")
    pos_embeddings = []
    pos_labels_list = []
    with torch.no_grad():
        for batch in tqdm(pos_loader, desc="  pos embed", leave=False):
            imgs, lbls = batch[0].to(device, non_blocking=True), batch[1]
            with torch.autocast(
                device_type=device.type, enabled=USE_AMP, dtype=_AMP_TORCH_DTYPE
            ):
                emb = model.embed(imgs)
            pos_embeddings.append(emb.float().cpu())
            pos_labels_list.append(lbls)
    pos_emb = torch.cat(pos_embeddings, dim=0)  # (N_pos, D)
    pos_labels = torch.cat(pos_labels_list, dim=0)  # (N_pos,)
    print(
        f"  [Mahalanobis] Positive embeddings: {pos_emb.shape}, classes: {num_classes}"
    )

    # ── Step 2: クラスごとの平均を計算 ──────────────────────────────────
    # NOTE: float64 で計算することで大規模な累積丸め誤差を抑制する。
    print("  [Mahalanobis] Computing per-class means (float64)...")
    pos_emb64 = pos_emb.double()  # float32 → float64
    class_means64 = torch.zeros(num_classes, emb_size, dtype=torch.float64)
    class_counts = torch.zeros(num_classes, dtype=torch.long)

    for c in range(num_classes):
        mask = pos_labels == c
        if mask.sum() > 0:
            class_means64[c] = pos_emb64[mask].mean(dim=0)
            class_counts[c] = mask.sum()

    n_empty = int((class_counts == 0).sum())
    if n_empty > 0:
        print(
            f"  [Mahalanobis] WARNING: {n_empty} classes have no samples; "
            f"their means remain zero."
        )

    # ── Step 3: Pooled within-class 共分散行列を計算 ────────────────────
    # S_w = Σ_c Σ_{x in c} (x - μ_c)(x - μ_c)^T / (N - C)
    # クラスごとに中心化して累積することで、クラス間の分散（クラスセントロイド間の分散）
    # を除去する。これにより ArcFace 特徴空間でも正常なクラス内分散が得られる。
    print("  [Mahalanobis] Computing pooled within-class covariance (float64)...")
    n_total = pos_emb64.shape[0]
    n_nonempty_classes = int((class_counts > 0).sum())
    dof = max(n_total - n_nonempty_classes, 1)  # 自由度 (N - C)

    # 各サンプルをそのクラス平均で中心化する
    class_mean_per_sample = class_means64[pos_labels.long()]  # (N_pos, D)
    centered = pos_emb64 - class_mean_per_sample  # (N_pos, D)
    cov_w = (centered.T @ centered) / dof  # (D, D) float64

    # 正則化: λ = trace(Σ_w)/D × 1e-3 (「平均クラス内分散の 0.1%」を下限保証)
    trace_mean = cov_w.diagonal().mean().item()
    adaptive_lambda = max(trace_mean * 1e-3, reg_lambda)
    print(
        f"  [Mahalanobis] within-class trace/D={trace_mean:.4g}, "
        f"adaptive λ={adaptive_lambda:.4g}"
    )
    cov_w += adaptive_lambda * torch.eye(emb_size, dtype=torch.float64)

    # 精度行列 Σ_w^{-1}: float64 で計算し float32 に変換して保存
    precision64 = torch.linalg.inv(cov_w)
    if not torch.isfinite(precision64).all():
        print(
            "  [Mahalanobis] linalg.inv returned non-finite values, falling back to pinv"
        )
        precision64 = torch.linalg.pinv(cov_w)
    precision = precision64.float()  # float32 に戻してバッファへ書き込む

    # ── Step 4: バッファを更新して MahalanobisLayer を使えるようにする ──
    # class_means バッファのサイズが変化しないよう、同じ num_classes で初期化済み。
    model.mahal.class_means.copy_(class_means64.float().to(device))
    model.mahal.precision.copy_(precision.to(device))
    print(
        "  [Mahalanobis] Buffers pre-loaded; computing distances via MahalanobisLayer..."
    )

    # 正例距離（min-distance to nearest class）
    pos_dists = []
    with torch.no_grad():
        for batch in tqdm(pos_loader, desc="  pos dist", leave=False):
            imgs = batch[0].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type, enabled=USE_AMP, dtype=_AMP_TORCH_DTYPE
            ):
                emb = model.embed(imgs).float()
            pos_dists.append(model.mahal(emb).cpu())
    pos_dists = torch.cat(pos_dists, dim=0).numpy()  # (N_pos,)

    # ── Step 5: 負例のマハラノビス距離を計算 ────────────────────────────
    # neg_loader は labeled=False の HuggingFaceFaceDataset を使用しているため、
    # batch は常に (imgs, -1) のタプルを返す。
    print("  [Mahalanobis] Collecting negative embeddings and computing distances...")
    neg_dists = []
    with torch.no_grad():
        for batch in tqdm(neg_loader, desc="  neg dist", leave=False):
            imgs = batch[0].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type, enabled=USE_AMP, dtype=_AMP_TORCH_DTYPE
            ):
                emb = model.embed(imgs).float()
            neg_dists.append(model.mahal(emb).cpu())
    neg_dists = torch.cat(neg_dists, dim=0).numpy()  # (N_neg,)
    print(f"  [Mahalanobis] Negative distances computed: {len(neg_dists)} samples")

    # ── Step 6: AUC 計算と Youden's J で最適閾値を決定 ──────────────────
    print("  [Mahalanobis] Computing AUC and optimal threshold (Youden's J)...")
    # ラベル: 正例=0 (normal), 負例=1 (anomaly)
    all_dists = np.concatenate([pos_dists, neg_dists])
    all_labels = np.concatenate(
        [
            np.zeros(len(pos_dists), dtype=np.int32),
            np.ones(len(neg_dists), dtype=np.int32),
        ]
    )

    # NaN/Inf チェック: 残存する場合はサンプルを除外して警告を出す
    nan_mask = ~np.isfinite(all_dists)
    if nan_mask.any():
        n_nan = int(nan_mask.sum())
        print(
            f"  [Mahalanobis] WARNING: {n_nan}/{len(all_dists)} distances are NaN/Inf — excluding from AUC computation"
        )
        valid = ~nan_mask
        all_dists = all_dists[valid]
        all_labels = all_labels[valid]
        pos_dists = pos_dists[np.isfinite(pos_dists)]
        neg_dists = neg_dists[np.isfinite(neg_dists)]

    auc = roc_auc_score(all_labels, all_dists)

    # Youden's J: J = Sensitivity + Specificity - 1 を最大化する閾値
    from sklearn.metrics import roc_curve

    fpr, tpr, roc_thresholds = roc_curve(all_labels, all_dists)
    j_scores = tpr - fpr  # Youden's J = TPR - FPR
    best_idx = int(np.argmax(j_scores))
    optimal_threshold = float(roc_thresholds[best_idx])

    pos_mean_dist = float(pos_dists.mean())
    neg_mean_dist = float(neg_dists.mean())
    print(f"  [Mahalanobis] AUC={auc:.4f}, threshold={optimal_threshold:.4f}")
    print(
        f"  [Mahalanobis] pos_mean_dist={pos_mean_dist:.4f}, neg_mean_dist={neg_mean_dist:.4f}"
    )

    # ── Step 7: バッファに閾値を書き込んでフィット完了 ──────────────────
    # class_means / precision は Step 4 で書き込み済み。閾値のみ更新する。
    model.mahal.threshold.copy_(
        torch.tensor(optimal_threshold, dtype=torch.float32, device=device)
    )
    print("  [Mahalanobis] Model buffers updated.")

    return {
        "auc": auc,
        "threshold": optimal_threshold,
        "pos_mean_dist": pos_mean_dist,
        "neg_mean_dist": neg_mean_dist,
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
            "mahal_reg_lambda": MAHAL_REG_LAMBDA,
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
    # Mahalanobis 異常検知の学習
    # best model の重みをロードしてから統計量を計算する
    # ──────────────────────────────────────────────
    print("Fitting Mahalanobis anomaly detector (loading model_best.pt)...")
    best_ckpt_path = os.path.join(run_dir, "model_best.pt")
    model_to_save.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    model_to_save.eval()

    # 正例 DataLoader: face-dataset 全体 (transform_eval)
    pos_loader_mahal = DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # 負例 DataLoader: helloproject-face-errors
    print("Loading helloproject-face-errors dataset...")
    error_raw = load_dataset("yayoimizuha/helloproject-face-errors")
    # split 名を動的に解決（通常は "train"）
    error_split_name = list(error_raw.keys())[0]
    error_hf = error_raw[error_split_name]
    print(f"  face-errors split='{error_split_name}', num_images={len(error_hf)}")

    error_dataset = HuggingFaceFaceDataset(error_hf, transform_eval, labeled=False)
    neg_loader_mahal = DataLoader(
        error_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    mahal_stats = fit_mahalanobis(
        model=model_to_save,
        pos_loader=pos_loader_mahal,
        neg_loader=neg_loader_mahal,
        device=device,
    )

    wandb.log(
        {
            "mahal/auc": mahal_stats["auc"],
            "mahal/threshold": mahal_stats["threshold"],
            "mahal/pos_mean_dist": mahal_stats["pos_mean_dist"],
            "mahal/neg_mean_dist": mahal_stats["neg_mean_dist"],
        }
    )

    # マハラノビス統計量込みのモデルを保存
    mahal_model_path = os.path.join(run_dir, "model_best_with_mahal.pt")
    torch.save(model_to_save.state_dict(), mahal_model_path)
    print(f"Model with Mahalanobis detector saved to {mahal_model_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
