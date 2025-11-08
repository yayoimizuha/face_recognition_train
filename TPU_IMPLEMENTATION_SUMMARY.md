# Google Cloud TPU v5e-8 Training Support - Implementation Summary

## 概要 (Overview)

このPRは、Google Cloud TPU v5e-8でのface recognition modelの訓練をサポートするための変更を実装しました。PyTorch/XLA 2.8を使用し、8コアのTPUで効率的な訓練が可能になります。

This PR implements support for training face recognition models on Google Cloud TPU v5e-8 using PyTorch/XLA 2.8, enabling efficient training across 8 TPU cores.

## 主な変更点 (Key Changes)

### 1. 訓練スクリプト (Training Scripts)

#### train_v2_tpu.py
- PyTorch/XLAを使用したTPU専用訓練スクリプト
- `xmp.spawn()`による8コアのマルチプロセス訓練
- XLA ParallelLoaderによる効率的なデータロード
- `xm.optimizer_step()`によるXLA最適化された勾配更新
- `xm.mark_step()`によるグラフコンパイル
- `xm.save()`によるチェックポイント保存

Key features:
- Multi-process training via `xmp.spawn()` for 8 TPU cores
- XLA ParallelLoader for efficient data distribution
- XLA-optimized gradient updates with `xm.optimizer_step()`
- Graph compilation with `xm.mark_step()`
- XLA-compatible checkpointing with `xm.save()`

#### run_tpu.sh
- TPU訓練用の起動スクリプト
- XLA環境変数の自動設定 (`XLA_USE_BF16=1`, `PJRT_DEVICE=TPU`)
- torch_xlaの自動インストール確認

Convenience script for launching TPU training with automatic environment setup.

### 2. 設定ファイル (Configuration)

#### configs/edgeface_s_gamma_05_tpu.py
- TPU v5e-8向けに最適化された設定
- bfloat16混合精度訓練（TPUネイティブサポート）
- コアあたりバッチサイズ128（合計1024）
- DALI無効化（CUDA専用のためTPUでは非サポート）

TPU-optimized configuration:
- Native bfloat16 mixed precision (`config.amp = torch.bfloat16`)
- Batch size: 128 per core (1024 total across 8 cores)
- DALI disabled (CUDA-only, not supported on TPU)

### 3. セットアップツール (Setup Tools)

#### setup_tpu.sh
- TPU VM用の自動セットアップスクリプト
- 依存関係の自動インストール
- PyTorch/XLAのインストール
- TPUデバイスの検証

Automated setup script for TPU VMs:
- Installs all dependencies
- Installs PyTorch/XLA
- Verifies TPU device availability

#### requirements-tpu.txt
- TPU訓練に必要な依存関係のリスト
- torch_xlaを含む

TPU-specific dependency list including torch_xla.

### 4. ドキュメント (Documentation)

#### docs/TPU_TRAINING.md (268 lines)
包括的なガイド:
- インストール手順
- TPU v5e-8の仕様
- 設定方法
- パフォーマンスチューニング
- トラブルシューティング

Comprehensive guide covering:
- Installation procedures
- TPU v5e-8 specifications
- Configuration guidelines
- Performance optimization tips
- Troubleshooting guide

#### docs/TPU_QUICKSTART.md (58 lines)
- 経験豊富なユーザー向けのクイックリファレンス
- GPUとTPUの違いの比較表
- 環境変数の設定

Quick reference guide for experienced users with comparison tables.

#### README.md
- TPU訓練セクションの追加
- AMPサポート表にTPUを追加

Updated with TPU training section and AMP support matrix.

### 5. 依存関係 (Dependencies)

#### pyproject.toml
- `torch_xla~=2.8`をオプション依存関係として追加
- TPU VM用のPyTorch/XLA wheelインデックスを追加

Added:
- `torch_xla~=2.8` as optional dependency (`[project.optional-dependencies.tpu]`)
- PyTorch/XLA wheel index for TPU VM

## 使用方法 (Usage)

### クイックスタート (Quick Start)

```bash
# 1. リポジトリのクローン
git clone https://github.com/yayoimizuha/face_recognition_train.git
cd face_recognition_train

# 2. セットアップ（TPU VMで実行）
./setup_tpu.sh

# 3. 訓練の実行
./run_tpu.sh configs/edgeface_s_gamma_05_tpu.py
```

### 詳細な手順 (Detailed Steps)

詳細は以下のドキュメントを参照してください:
- [TPU_TRAINING.md](docs/TPU_TRAINING.md) - 完全なガイド
- [TPU_QUICKSTART.md](docs/TPU_QUICKSTART.md) - クイックリファレンス

See documentation for detailed instructions:
- [TPU_TRAINING.md](docs/TPU_TRAINING.md) - Complete guide
- [TPU_QUICKSTART.md](docs/TPU_QUICKSTART.md) - Quick reference

## GPUとの主な違い (Key Differences from GPU Training)

| 項目 | GPU (`train_v2.py`) | TPU (`train_v2_tpu.py`) |
|------|---------------------|-------------------------|
| 起動方法 | `torchrun --nproc_per_node=8` | `python3` (xmp.spawnが処理) |
| バックエンド | NCCL/Gloo | PyTorch/XLA |
| データローディング | DALI対応 | DataLoader + ParallelLoader |
| 混合精度 | fp16/bf16 | bf16（ネイティブサポート） |
| オプティマイザステップ | `optimizer.step()` | `xm.optimizer_step(opt)` |
| チェックポイント | `torch.save()` | `xm.save()` |
| グラフコンパイル | 自動 | `xm.mark_step()`が必要 |

## パフォーマンス推奨事項 (Performance Recommendations)

1. **バッチサイズ**: コアあたり128から開始（合計1024）
2. **ワーカー数**: 4-8ワーカーに設定
3. **混合精度**: 常にbfloat16を使用
4. **データセット形式**: 大規模データセットにはWebDatasetを使用
5. **環境変数**: `XLA_USE_BF16=1`を設定

Performance tips:
1. **Batch Size**: Start with 128 per core (1024 total)
2. **Workers**: Set to 4-8 workers
3. **Mixed Precision**: Always use bfloat16
4. **Dataset Format**: Use WebDataset for large-scale datasets
5. **Environment**: Set `XLA_USE_BF16=1`

## テスト済み環境 (Tested Environment)

- **TPU**: Google Cloud TPU v5e-8
- **Python**: 3.12
- **PyTorch**: 2.8.0
- **PyTorch/XLA**: 2.8.0
- **OS**: Linux (TPU VM)

## 参考文献 (References)

- [PyTorch/XLA Documentation (v2.8)](https://docs.pytorch.org/xla/release/r2.8/index.html)
- [Google Cloud TPU Documentation](https://cloud.google.com/tpu/docs)
- [PyTorch/XLA GitHub](https://github.com/pytorch/xla)

## ファイル一覧 (Files Added/Modified)

新規ファイル (New files):
- `train_v2_tpu.py` (276 lines)
- `configs/edgeface_s_gamma_05_tpu.py` (76 lines)
- `run_tpu.sh` (29 lines)
- `setup_tpu.sh` (70 lines)
- `requirements-tpu.txt` (38 lines)
- `docs/TPU_TRAINING.md` (268 lines)
- `docs/TPU_QUICKSTART.md` (58 lines)

変更ファイル (Modified files):
- `pyproject.toml` (+13 lines)
- `README.md` (+11 lines)

合計 (Total): **839 lines added**

## 次のステップ (Next Steps)

ユーザーは以下を実行できます:
1. TPU VM上でセットアップスクリプトを実行
2. データセットをImageFolderまたはWebDataset形式で準備
3. 設定ファイルをデータセットに合わせて更新
4. run_tpu.shで訓練を開始

Users can now:
1. Run the setup script on a TPU VM
2. Prepare datasets in ImageFolder or WebDataset format
3. Update configuration files for their datasets
4. Start training with run_tpu.sh
