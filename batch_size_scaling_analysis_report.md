# バッチサイズ拡大時にlossが下がらない・精度が向上しない問題の分析レポート

## 実行日時
2025-11-03

## 実行環境
- コマンド: `torchrun --nproc_per_node=8 train_v2_restart.py configs/glint360k_mbv4_hybrid_med.py`
- GPU数: 8
- 設定ファイル: `configs/glint360k_mbv4_hybrid_med.py`

## 問題の概要
分散学習環境でバッチサイズを拡大した際に、損失が適切に減少せず、精度が向上しない現象が発生している。

## 根本原因の分析

### 1. 勾配スケーリングの不一致（主要原因）

**問題箇所**: `partial_fc_v2.py` の `DistCrossEntropyFunc.backward()` メソッド（227行目）

```python
def backward(ctx, loss_gradient):
    ...
    batch_size = logits.size(0)
    ...
    logits[index] -= one_hot
    logits.div_(batch_size)  # ←問題: ローカルバッチサイズで割っている
    return logits * loss_gradient.item(), None
```

**詳細説明**:
- 現在の実装では、勾配を**ローカルバッチサイズ**（各GPU上のバッチサイズ）で除算している
- しかし、`AllGatherFunc.backward()`（260行目）で勾配に`world_size`を乗算している：
  ```python
  grad_out *= len(grad_list)  # world_size倍する
  ```
- この結果、実効的な勾配スケールは以下のようになる：
  ```
  実効勾配スケール = (1 / local_batch_size) × world_size
                   = world_size / local_batch_size
  ```

**具体的な影響**:

| 設定 | GPU数 | ローカルBS | グローバルBS | 実効スケール | 期待スケール | 誤差 |
|------|------|-----------|-------------|-------------|-------------|------|
| 単一GPU | 1 | 128 | 128 | 1/128 | 1/128 | 正常 |
| 8GPU | 8 | 128 | 1024 | 8/128 = 1/16 | 1/1024 | **64倍大きい** |
| 8GPU（BS拡大） | 8 | 256 | 2048 | 8/256 = 1/32 | 1/2048 | **64倍大きい** |

**結論**: バッチサイズを増やしても、実効的な勾配スケールが正しく調整されないため、学習が適切に進まない。

### 2. 学習率スケーリングの問題

**問題箇所**: `configs/glint360k_mbv4_hybrid_med.py` と `train_v2_restart.py`

現在の設定:
```python
# config
config.batch_size = 128  # ローカルバッチサイズ
config.lr = 0.1

# train_v2_restart.py (187-189行目)
cfg.total_batch_size = cfg.batch_size * world_size  # = 128 * 8 = 1024
cfg.warmup_step = cfg.num_image // cfg.total_batch_size * cfg.warmup_epoch
cfg.total_step = cfg.num_image // cfg.total_batch_size * cfg.num_epoch
```

**問題点**:
1. **Linear Scaling Rule**が適用されていない
   - 理論: バッチサイズをN倍にする場合、学習率もN倍にすべき
   - 現状: バッチサイズが8倍（128→1024）でも学習率は固定（0.1）
   
2. **AdamWオプティマイザとの組み合わせ**
   ```python
   config.optimizer = "adamw"
   config.adam_betas = (0.9, 0.99)
   ```
   - AdamWは適応的学習率を使用するため、バッチサイズ拡大の影響を受けやすい
   - 大きなバッチサイズでは、より慎重な学習率調整が必要

### 3. 勾配累積の設定

**現在の設定**: `configs/base.py`より
```python
config.gradient_acc = 1  # 勾配累積なし
```

**問題点**:
- `configs/glint360k_mbv4_hybrid_med.py`には`gradient_acc`の設定がないため、デフォルト値1が使用される
- 勾配累積が有効に活用されていない

### 4. Warmupエポック数が非常に小さい

```python
config.warmup_epoch = 0.04  # わずか0.04エポック
```

**問題点**:
- グローバルバッチサイズ1024の場合、warmup_stepは約667ステップ
- 大規模データセット（17M画像）に対して、warmup期間が極めて短い
- 学習初期の不安定性を招く可能性がある

### 5. SyncBatchNormの統計量

**コード**: `train_v2_restart.py` 143行目
```python
backbone = convert_sync_batchnorm(backbone)
```

**考慮点**:
- SyncBatchNormは全GPUで統計量を同期するため、大きなバッチサイズで有利
- しかし、バッチサイズが小さい場合（GPU単位で128）、統計量の精度が影響を受ける可能性がある

## 推奨される対策

### 優先度1: 勾配スケーリングの修正（必須）

**partial_fc_v2.py** の `DistCrossEntropyFunc.backward()` を修正：

```python
@staticmethod
def backward(ctx, loss_gradient):
    (index, logits, label,) = ctx.saved_tensors
    batch_size = logits.size(0)
    world_size = distributed.get_world_size()
    
    one_hot = torch.zeros(
        size=[index.size(0), logits.size(1)], 
        device=logits.device, 
        dtype=logits.dtype
    )
    one_hot.scatter_(1, label[index], 1.0)
    logits[index] -= one_hot
    
    # 修正: グローバルバッチサイズで除算
    global_batch_size = batch_size * world_size
    logits.div_(global_batch_size)
    
    return logits * loss_gradient.item(), None
```

**重要**: この修正により、AllGatherFunc.backward()の`grad_out *= len(grad_list)`と組み合わせて、正しいスケールになる：
```
実効勾配スケール = (1 / (local_batch_size × world_size)) × world_size
                = 1 / (local_batch_size × world_size)
                = 1 / global_batch_size  ✓ 正しい
```

### 優先度2: 学習率のLinear Scaling適用

**configs/glint360k_mbv4_hybrid_med.py** を修正：

```python
# ベース学習率（参照バッチサイズ256を想定）
base_lr = 0.1
reference_batch_size = 256
config.batch_size = 128

# Linear Scaling Rule: lr = base_lr * (actual_batch_size / reference_batch_size)
# GPUs=8の場合: global_batch_size = 128 * 8 = 1024
# scaled_lr = 0.1 * (1024 / 256) = 0.4
config.lr = base_lr * (config.batch_size * 8) / reference_batch_size

# または、より一般的な書き方として設定ファイルに記載：
# config.lr = 0.1  # for batch_size=128, 1GPU
# 実際の学習率は train_v2_restart.py で自動スケーリング
```

**train_v2_restart.py** での自動スケーリング実装例（171-173行目付近に追加）：

```python
# Linear Scaling Rule適用（オプション）
if hasattr(cfg, 'lr_scaling') and cfg.lr_scaling:
    reference_bs = getattr(cfg, 'reference_batch_size', 256)
    cfg.lr = cfg.lr * (cfg.total_batch_size / reference_bs)
    logging.info(f"Learning rate scaled: {cfg.lr} (total_batch_size={cfg.total_batch_size})")
```

### 優先度3: Warmupエポックの調整

```python
config.warmup_epoch = 1.0  # 0.04 → 1.0 に増加
```

**理由**:
- より安定した学習開始
- 大きなバッチサイズでは、より長いwarmup期間が推奨される

### 優先度4: AdamW最適化パラメータの調整

```python
# より保守的なbeta2を使用
config.adam_betas = (0.9, 0.95)  # 元: (0.9, 0.99)

# または、weight_decayを調整
config.weight_decay = 5e-5  # 元: 1e-4
```

### 優先度5: 勾配累積の活用（オプション）

大きなバッチサイズが必要な場合の代替手段：

```python
# メモリが不足する場合の設定例
config.batch_size = 64  # 物理バッチサイズを小さく
config.gradient_acc = 4  # 4ステップ累積で実効BS=256/GPU
# 実効グローバルBS = 64 * 8 * 4 = 2048
```

## 検証手順

### 1. 修正効果の確認

```bash
# 修正前のベースライン確認
torchrun --nproc_per_node=8 train_v2_restart.py configs/glint360k_mbv4_hybrid_med.py

# 修正後の確認
# partial_fc_v2.py を修正後、同じコマンドで実行
```

### 2. 学習曲線の比較

- Loss曲線が適切に減少しているか
- 検証精度が向上しているか
- Warmup期間中の安定性

### 3. 異なるバッチサイズでの実験

| 設定 | GPU数 | ローカルBS | グローバルBS | 学習率 | 期待される結果 |
|------|------|-----------|-------------|--------|--------------|
| 小 | 8 | 64 | 512 | 0.2 | ベースライン |
| 中 | 8 | 128 | 1024 | 0.4 | 同等の性能 |
| 大 | 8 | 256 | 2048 | 0.8 | 同等の性能 |

## 追加の調査項目

### 1. 数値安定性の確認

現在のコードには簡易的なNaN/Infチェックがある（257-268行目）：
```python
if not torch.isfinite(loss):
    print(f"Loss is NaN/Inf at step {global_step} (epoch {epoch}).")
```

**推奨**: より詳細なロギングを追加：
- 勾配ノルムの記録
- Loss値の詳細な履歴
- 学習率の変化

### 2. Gradient Clippingの効果確認

現在のコード（270行目、283行目）:
```python
torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
```

**推奨**: クリップ閾値の調整実験：
- 大きなバッチサイズでは、より小さい値（例：1.0）を試す
- クリップが発動する頻度をログに記録

### 3. Learning Rate Schedulerの動作確認

`PolynomialLRWarmup`の動作をバッチサイズ変更時に確認：
```python
# lr_scheduler.py のテスト実行
python lr_scheduler.py
```

## まとめ

### 問題の本質
1. **最重要**: 勾配スケーリングがローカルバッチサイズベースで、グローバルバッチサイズに対応していない
2. **重要**: 学習率がバッチサイズに応じてスケールされていない
3. **補助的**: Warmup期間が短すぎる、AdamWパラメータが大バッチに最適化されていない

### 解決の優先順位
1. **優先度1（必須）**: `partial_fc_v2.py`の勾配スケーリング修正
2. **優先度2（強く推奨）**: 学習率のLinear Scaling適用
3. **優先度3（推奨）**: Warmupエポック数の増加
4. **優先度4（状況に応じて）**: AdamWパラメータの調整
5. **優先度5（オプション）**: 勾配累積の活用

### 期待される効果
これらの修正により：
- バッチサイズを増やしても適切に学習が進む
- Loss曲線が正常に減少する
- 検証精度が向上する
- 複数GPU環境での学習が安定する

## 参考文献

1. Goyal, P., et al. (2017). "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour." arXiv:1706.02677
   - Linear Scaling Ruleの理論的根拠

2. You, Y., et al. (2019). "Large Batch Optimization for Deep Learning: Training BERT in 76 minutes." arXiv:1904.00962
   - 大規模バッチ学習のベストプラクティス

3. InsightFace PartialFC論文: "Partial FC: Training 10 Million Identities on a Single Machine" 
   - 本実装の基礎となるアーキテクチャ

## 付録: コード変更の詳細

### A. 修正前後の比較

**修正前（partial_fc_v2.py:227）**:
```python
logits.div_(batch_size)  # batch_size = local batch size
```

**修正後**:
```python
world_size = distributed.get_world_size()
global_batch_size = batch_size * world_size
logits.div_(global_batch_size)
```

### B. 設定ファイルのテンプレート

```python
from easydict import EasyDict as edict
import torch

config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.network = "mobilenetv4_hybrid_medium.ix_e550_r384_in1k"
config.resume = False
config.output = "./work_dirs/glint360k_mbv4_hybrid_med"
config.embedding_size = 512
config.sample_rate = 1.0
config.device_type = "cuda"
config.amp = torch.bfloat16
config.momentum = 0.9
config.weight_decay = 1e-4

# バッチサイズと学習率の設定
config.batch_size = 128  # ローカルバッチサイズ（GPU単位）
config.reference_batch_size = 256  # 参照バッチサイズ
config.base_lr = 0.1  # 参照バッチサイズでの学習率

# Linear Scaling Rule適用
# 8GPU使用時: total_batch_size = 128 * 8 = 1024
# scaled_lr = 0.1 * (1024 / 256) = 0.4
config.lr = config.base_lr * (config.batch_size * 8) / config.reference_batch_size

config.verbose = 2000
config.dali = False
config.optimizer = "adamw"
config.adam_betas = (0.9, 0.95)  # より保守的なbeta2

config.num_workers = 5
config.dali_aug = False
config.dataset_type = "webdataset"
config.rec = "/mnt/nvme/Glint360k_WebDataset/"

config.num_classes = 360232
config.num_image = 17091657
config.num_epoch = 50
config.warmup_epoch = 1.0  # 増加: 0.04 → 1.0
config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]
config.val_dir = "/mnt/nvme/data1"
```

---

**作成日**: 2025-11-03  
**作成者**: GitHub Copilot Analysis  
**バージョン**: 1.0
