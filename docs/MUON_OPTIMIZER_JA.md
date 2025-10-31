# Muon オプティマイザー ドキュメント

## 概要

**Muon**（MomentUm Orthogonalized by Newton-schulz）は、ディープニューラルネットワークの学習のために設計された新しい最適化アルゴリズムです。モーメンタムベースの勾配降下法とNewton-Schulz行列直交化を組み合わせ、改善された収束特性と学習の安定性を提供します。

## Muonとは？

Muonは、ニューラルネットワークの2次元パラメータ（重み行列）の最適化に特に効果的です。主要な革新は、Newton-Schulz反復を適用して勾配更新を直交化することで、学習全体を通じてより良い最適化幾何学を維持するのに役立ちます。

### 主な特徴

- **Newton-Schulz直交化**: 勾配更新に反復的な直交化を適用
- **モーメンタムベース更新**: 加速のためにモーメンタム（Nesterovを含む）と組み合わせ
- **2次元パラメータ最適化**: LinearおよびConvレイヤーの重み行列専用に設計
- **暗黙的正則化**: 直交化が暗黙的な正則化として機能
- **安定した学習**: 標準的なオプティマイザーと比較してより安定した学習ダイナミクス

## 数学的背景

### Newton-Schulz反復

Newton-Schulz法は、反復的な改良を通じて行列Gの直交化を計算します：

```
X₀ = G / ||G||
Xₙ₊₁ = a·Xₙ + b·A·Xₙ + c·A²·Xₙ
ここで A = Xₙ·Xₙᵀ
```

係数（a=3.4445, b=-4.7750, c=2.0315）は最適な収束のために調整されています。

### 更新ルール

パラメータθと勾配gに対して：

1. **モーメンタム更新**: `m = β·m + g`
2. **Nesterov（オプション）**: `g' = g + β·m`
3. **直交化**: `g_orth = NewtonSchulz(g')`
4. **パラメータ更新**: `θ = θ - η·g_orth`

## インストール

Muonはこのリポジトリに含まれています。標準的な依存関係以外の追加インストールは不要です。

## 使用方法

### 基本的な使用方法

```python
from muon import Muon
import torch.nn as nn

# モデルを作成
model = nn.Sequential(
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# Muonオプティマイザーを初期化
optimizer = Muon(
    model.parameters(),
    lr=0.02,           # 学習率
    momentum=0.95,     # モーメンタム係数
    nesterov=True      # Nesterovモーメンタムを使用
)

# 学習ループ
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
```

### 設定ファイルでの使用

設定ファイル（例：`configs/my_config.py`）内で：

```python
from easydict import EasyDict as edict

config = edict()

# モデル設定
config.network = "r50"
config.embedding_size = 512

# Muonオプティマイザー設定
config.optimizer = "muon"
config.lr = 0.02          # 推奨: 0.01-0.05
config.momentum = 0.95    # 推奨: 0.90-0.95
config.nesterov = True    # 推奨: True

# その他の学習パラメータ
config.batch_size = 512
config.num_epoch = 100
```

学習を実行：

```bash
torchrun --nproc_per_node=4 train_v2.py configs/my_config.py
```

## ハイパーパラメータガイドライン

### 学習率

- **推奨範囲**: 0.01 - 0.05
- **典型的な値**: 0.02
- **注意**: MuonはAdamWよりも高い学習率を使用します

Muonの直交化により、不安定性なしでより積極的な学習率を使用できます。

### モーメンタム

- **推奨範囲**: 0.90 - 0.95
- **典型的な値**: 0.95
- **注意**: 高いモーメンタムがMuonで良好に機能します

直交化ステップは高いモーメンタムから恩恵を受けます。これは方向の一貫性を維持するのに役立ちます。

### Nesterovモーメンタム

- **推奨**: True
- **利点**: 収束速度と最終的なパフォーマンスを改善

Nesterovモーメンタムは、直交化と相乗的に機能する先読み更新を提供します。

### Newton-Schulzステップ

- **デフォルト**: 5ステップ
- **範囲**: 3-7ステップ
- **注意**: ステップが多い = より正確な直交化だが遅い

## パフォーマンス特性

### Muonを使用すべき時

**最適な用途：**
- 畳み込みニューラルネットワーク（CNN）
- ResNetアーキテクチャ
- 顔認識モデル
- Vision Transformer
- 多くの2次元重み行列を持つモデル

**代替を検討する場合：**
- 非常に大きな言語モデル（>1Bパラメータ）の学習
- 主に1次元パラメータを持つモデル
- メモリ効率的な最適化が必要な場合

### 他のオプティマイザーとの比較

| 側面 | Muon | SGD | AdamW |
|------|------|-----|-------|
| 収束速度 | 速い | 中程度 | 速い |
| メモリオーバーヘッド | 低い | 低い | 高い |
| ハイパーパラメータ感度 | 低い | 高い | 中程度 |
| 2次元パラメータに最適 | ✓ | ✗ | ✗ |
| 1次元パラメータに最適 | ✗ | ✓ | ✓ |
| 学習率範囲 | 0.01-0.05 | 0.01-0.1 | 0.0001-0.001 |

## 実装の詳細

### パラメータ処理

Muonは異なるパラメータタイプを異なる方法で処理します：

- **2次元パラメータ**（例：Linear、Conv重み）: 直交化を伴う完全なMuon更新
- **非2次元パラメータ**（例：バイアス、BatchNorm）: 直交化なしの標準モーメンタム更新

このハイブリッドアプローチにより、すべてのパラメータタイプで最適なパフォーマンスが保証されます。

### メモリに関する考慮事項

Muonはモーメンタム付きSGDと同様のメモリ要件を持ちます：
- 各パラメータにモーメンタムバッファを保存
- Adamの二次モーメントのような追加状態はなし
- 直交化はその場で計算

### 計算コスト

ステップあたりのコスト：
1. **モーメンタム更新**: O(n) - SGDと同じ
2. **Newton-Schulz反復**: O(d³) ここで d = 2次元パラメータのmin(行, 列)
3. **パラメータ更新**: O(n) - SGDと同じ

典型的なCNNでは、Newton-Schulzのコストは順伝播/逆伝播と比較して無視できます。

## 高度なトピック

### 混合精度学習

Muonは自動混合精度（AMP）で良好に機能します：

```python
from torch.amp import autocast, GradScaler

scaler = GradScaler()
optimizer = Muon(model.parameters(), lr=0.02)

for data, target in train_loader:
    optimizer.zero_grad()
    
    with autocast(device_type='cuda', dtype=torch.float16):
        output = model(data)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 分散学習

MuonはPyTorchのDistributedDataParallelと完全に互換性があります：

```python
import torch.distributed as dist

# プロセスグループを初期化
dist.init_process_group(backend='nccl')

# DDPでモデルをラップ
model = torch.nn.parallel.DistributedDataParallel(model)

# 通常通りMuonオプティマイザーを使用
optimizer = Muon(model.parameters(), lr=0.02)
```

## トラブルシューティング

### よくある問題

**問題**: 学習が不安定 / 損失が発散
- **解決策**: 学習率を下げる（0.02の代わりに0.01を試す）
- **解決策**: 必要に応じて勾配が適切にクリッピングされていることを確認

**問題**: 学習が予想より遅い
- **解決策**: コンパイルのオーバーヘッドのため、最初の数ステップでは正常です
- **解決策**: 学習にGPUを使用していることを確認

**問題**: メモリ不足エラー
- **解決策**: バッチサイズを減らす
- **解決策**: 勾配累積を使用

### パフォーマンスのヒント

1. **torch.compileを使用**: Newton-Schulz反復を高速化できます
   ```python
   model = torch.compile(model)
   ```

2. **学習をプロファイル**: ボトルネックを特定
   ```python
   with torch.profiler.profile() as prof:
       # 学習ステップ
   ```

3. **NSステップを調整**: ステップが少ない = より速いがより不正確な直交化

## 参考文献

- [オリジナルMuon実装](https://github.com/KellerJordan/cifar10-airbench)
- [Newton-Schulz反復](https://en.wikipedia.org/wiki/Newton%27s_method)
- 最適化における行列直交化に関する研究論文

## 例

完全な設定例については、`configs/muon_example.py`を参照してください。

## サポート

Muonオプティマイザーに関する問題や質問については：
1. このドキュメントを確認
2. `configs/muon_example.py`の例示設定を確認
3. 使用例については`test_muon.py`テストファイルを確認
4. リポジトリにissueを開く

## ライセンス

この実装はface_recognition_trainリポジトリの一部であり、同じライセンス条項に従います。
