# hp_finetune 使用説明書

## 概要

| スクリプト | 役割 |
|---|---|
| `finetune_facenet.py` | データセットを使ってモデルをファインチューニングする |
| `export_onnx.py` | 学習済み `.pt` チェックポイントを ONNX に変換する |
| `infer_onnx.py` | ONNX モデルでデータセット全体を推論し、誤分類を分析する |

---

## finetune_facenet.py — ファインチューニング

HuggingFace データセット (`yayoimizuha/helloproject-face-dataset`) を使い、
FastViT-S12 + GWAP + ArcFace でファインチューニングを行う。
ArcFace 学習後、`yayoimizuha/helloproject-face-errors` と `tonyassi/celebrity-1000` を
負例として `AnomalyClassifier`（二値分類器）を追加学習する。

### 基本的な使い方

```bash
python hp_finetune/finetune_facenet.py
```

### オプション

| オプション | 説明 |
|---|---|
| `--dump-inputs` | 学習を行わず、DataLoader の入力画像を 4×4 グリッドで保存して終了する（データ確認用） |

### 出力

`hp_finetune/work_dirs/<タイムスタンプ>/` 以下に保存される。

| ファイル | 内容 |
|---|---|
| `model_best.pt` | 検証精度が最も高かったエポックの重み |
| `model_best_with_anomaly.pt` | `model_best.pt` に `AnomalyClassifier`（二値分類器）を追加学習した最終モデル |
| `model_final.pt` | 最終エポックの重み |
| `model_epochN.pt` | `SAVE_INTERVAL` ごとの定期チェックポイント |
| `confusion_matrix.png` | 学習データ全体での混同行列 |
| `finetune_facenet.py` | 再現性のためにコピーされた実行スクリプト |

### 主な設定値（スクリプト冒頭の定数）

| 定数 | デフォルト値 | 説明 |
|---|---|---|
| `NUM_EPOCHS` | 150 | ArcFace 学習エポック数 |
| `BATCH_SIZE` | 128 | バッチサイズ |
| `LR` | 3e-3 | head / GWAP / ArcFace の学習率 |
| `LR_BACKBONE` | 3e-4 | backbone の学習率 |
| `EMB_SIZE` | 512 | 埋め込みベクトルの次元数 |
| `VAL_RATIO` | 0.2 | 検証データの割合 |
| `USE_AMP` | `True` | 混合精度学習の有効化 |
| `AMP_DTYPE` | `"bf16"` | AMP の dtype（`"bf16"` または `"fp16"`） |
| `ANOMALY_HIDDEN_DIM` | 256 | AnomalyClassifier 中間層の次元数 |
| `ANOMALY_DROPOUT` | 0.3 | AnomalyClassifier の Dropout 率 |
| `ANOMALY_LR` | 1e-3 | AnomalyClassifier の学習率 |
| `ANOMALY_EPOCHS` | 20 | AnomalyClassifier の学習エポック数 |
| `CELEBRITY_NEG_SAMPLES` | 3000 | 負例として使う `tonyassi/celebrity-1000` のサンプル数 |

#### `AMP_DTYPE` の選択基準

| 値 | 指数部 | 最大値 | GradScaler | 推奨環境 |
|---|---|---|---|---|
| `"bf16"` | 8 bit（fp32 と同等） | 3.4e38 | **不要**（自動無効化） | Ampere 以降（A100 / H100 / H200 など）← **推奨** |
| `"fp16"` | 5 bit | 65504 | 必要 | Volta / Turing など bf16 非対応 GPU |

> **注意:** `"fp16"` では backbone の activation が 65504 を超えると Inf が発生し、
> `BatchNorm1d` の `running_mean` / `running_var` が NaN に汚染される。
> 汚染が起きると eval 時に `embed()` が NaN を返し、AnomalyClassifier の異常スコア計算が失敗する。
> Ampere 以降の GPU では必ず `"bf16"` を使用すること。

---

## export_onnx.py — ONNX エクスポート

学習済みの `.pt` チェックポイントから **7 種類** の ONNX モデルを生成する。
Block-FP8 スタイルの per-block 量子化を採用し、ウェイトノードを
「量子化済みウェイト × スケーリング係数」のサブグラフに付け替える。
DequantizeLinear を使わず標準 ONNX ノード（Cast, Mul, Reshape, Slice）のみで
構成するため、TensorRT を含む全てのランタイムで互換性がある。

量子化するとモデル精度が著しく劣化するウェイトノードは 2 段階の感度検出
（ウェイト再構成誤差 → 出力差検証）で自動的に除外される。

### 基本的な使い方

```bash
python hp_finetune/export_onnx.py \
    --checkpoint hp_finetune/work_dirs/<タイムスタンプ>/model_best_with_anomaly.pt
```

### オプション

| オプション | デフォルト | 説明 |
|---|---|---|
| `--checkpoint` | (必須) | `.pt` チェックポイントのパス |
| `--num-classes` | 自動検出 | クラス数（省略時はチェックポイントから自動取得） |
| `--opset` | 19 | ONNX opset バージョン（FP8 には 19 以上が必要） |
| `--eval-samples` | 50 | 各バリアントの品質評価サンプル数 |
| `--block-size` | 32 | per-block 量子化のブロックサイズ（要素数） |
| `--fp8-format` | `e4m3fn` | FP8 のエンコーディング形式（`e4m3fn` / `e5m2`） |
| `--nrmse-threshold` | 0.02 | 感度検出 Stage 1: ウェイト再構成 NRMSE しきい値 |
| `--output-diff-threshold` | 0.5 | 感度検出 Stage 2: 出力最大絶対差しきい値 |
| `--sensitivity-samples` | 8 | 感度検出 Stage 2: 検証用サンプル数 |

### 出力

チェックポイントと同じディレクトリに以下が生成される。

| ファイル名 | ウェイト | 計算 dtype | 説明 |
|---|---|---|---|
| `<stem>.onnx` | fp32 | fp32 | 基本モデル（他の全バリアントのベース） |
| `<stem>_bf16.onnx` | bf16 | bf16 | グラフ全体 bf16 化（AnomalyClassifier は fp32 維持） |
| `<stem>_fp16.onnx` | fp16 | fp16 | グラフ全体 fp16 化（AnomalyClassifier は fp32 維持） |
| `<stem>_bf16int8.onnx` | INT8 per-block | bf16 | INT8 量子化ウェイト + bf16 スケール |
| `<stem>_bf16fp8.onnx` | FP8 per-block | bf16 | FP8 量子化ウェイト + bf16 スケール |
| `<stem>_fp16int8.onnx` | INT8 per-block | fp16 | INT8 量子化ウェイト + fp16 スケール |
| `<stem>_fp16fp8.onnx` | FP8 per-block | fp16 | FP8 量子化ウェイト + fp16 スケール |

> **注意:** fp16 / bf16 系モデルは GPU 環境（Ampere 以降）での推論を想定している。
> CPU では fp32 と速度差がほとんどない。

### Per-block 量子化の仕組み

ウェイトテンソルをブロック単位（デフォルト 32 要素）に分割し、各ブロック内で
absmax スケーリングにより量子化する。ONNX グラフ上では標準ノードのみで復元する:

```
weight_quantized [int8/fp8, (num_blocks, block_size)]
  → Cast(to=bf16/fp16)
  → Mul(× scale [(num_blocks, 1)])
  → Reshape → [Slice] → Reshape
  → 元の Conv/Gemm/MatMul ノードに接続
```

### 感度検出（2 段階ハイブリッド）

1. **Stage 1（ウェイト再構成誤差）**: 各ウェイトを量子化→復元し、NRMSE が
   しきい値を超えるノードを「疑い」リストに追加。推論不要で高速。
2. **Stage 2（出力差検証）**: 「疑い」ノードのみ、1 ノードずつ量子化した
   グラフで推論を実行し、fp32 出力との最大絶対差で最終判定。

感度検出で除外されたノードは BF16/FP16 のまま維持される。

### モデル出力について

**分類のみ (`model_best.pt` から生成した場合)**

- 出力は **分類ロジット** `(batch_size, num_classes)`
- 確率が必要な場合は消費側で softmax を適用する

**分類 + 異常検知 (`model_best_with_anomaly.pt` から生成した場合)**

- 出力テンソルが **2つ** になる
  - `logits`        : `(batch_size, num_classes)` — 分類ロジット（上と同じ）
  - `anomaly_score` : `(batch_size,)` — 異常スコア（sigmoid 出力、0〜1。大きいほど異常）
- `anomaly_threshold` がメタデータに埋め込まれており、`anomaly_score > threshold` で異常判定できる
- AnomalyClassifier ノード・イニシャライザは常に fp32 で維持される

バッチサイズは動的（任意のバッチサイズで推論可能）

---

## infer_onnx.py — ONNX 推論・誤分類分析

エクスポートした ONNX モデルでデータセット全体を推論し、
誤分類のランキングや誤り画像を保存する。

### 基本的な使い方

```bash
# CPU（デフォルト）
python hp_finetune/infer_onnx.py \
    --onnx hp_finetune/work_dirs/<タイムスタンプ>/model_best.onnx

# CUDA
python hp_finetune/infer_onnx.py \
    --onnx hp_finetune/work_dirs/<タイムスタンプ>/model_best.onnx \
    --provider cuda --device-id 0

# TensorRT（FP16 + エンジンキャッシュ）
python hp_finetune/infer_onnx.py \
    --onnx hp_finetune/work_dirs/<タイムスタンプ>/model_best_fp16int8.onnx \
    --provider tensorrt --trt-fp16 --trt-cache-dir /tmp/trt_cache
```

### オプション

| オプション | デフォルト | 説明 |
|---|---|---|
| `--onnx` | (必須) | ONNX モデルのパス |
| `--output-dir` | `<onnx_dir>/infer_results/<stem>` | 結果の保存先ディレクトリ |
| `--batch-size` | 32 | ONNX 推論のバッチサイズ |
| `--max-samples` | 全サンプル | 学習データセットから処理するサンプル数の上限 |
| `--top-k` | 20 | 混同上位ペアの表示件数 |
| `--top-class-k` | 0 | サマリに表示するクラス別精度の行数（精度昇順・worst first）。0 = 全クラス表示 |
| `--provider` | `auto` | 実行プロバイダー（`auto` / `cpu` / `cuda` / `tensorrt`） |
| `--device-id` | 0 | GPU デバイス番号（CUDA / TensorRT 使用時） |
| `--trt-fp16` | false | TensorRT エンジンで FP16 精度を有効にする |
| `--trt-int8` | false | TensorRT エンジンで INT8 精度を有効にする |
| `--trt-cache-dir` | `<output_dir>/trt_cache` | TensorRT エンジンキャッシュの保存先 |

**`--provider` の選択優先順位（`auto`）:** TensorRT → CUDA → CPU

### 出力

`--output-dir` 以下に以下が保存される。

| ファイル／ディレクトリ | 内容 |
|---|---|
| `wrong_predictions.csv` | 誤分類サンプルの一覧（index、真ラベル、予測ラベル、確信度） |
| `confusion_ranking.csv` | 誤分類ペア `(true_class, pred_class, count)` の件数降順ランキング |
| `class_accuracy.csv` | クラスごとの正解率（昇順ソート、最も混同されやすいクラスが先頭） |
| `errors_predictions.csv` | `helloproject-face-errors` データセットの推論結果（正解ラベルなし） |
| `distribution_anomaly_score.png` | 異常スコアの分布ヒストグラム（学習データ正例 vs errors 負例）

---

## 典型的なワークフロー

```
1. finetune_facenet.py  →  work_dirs/<timestamp>/model_best.pt
                                                  model_best_with_anomaly.pt
2. export_onnx.py       →  work_dirs/<timestamp>/model_best_with_anomaly.onnx（他 6 バリアント）
3. infer_onnx.py        →  work_dirs/<timestamp>/infer_results/<stem>/
```

---

## 変更履歴

### 2026-03-21 (3)

#### `export_onnx.py` — 全面再実装

- **出力を 5 種類 → 7 種類に拡張**: FP8 バリアント（`bf16fp8` / `fp16fp8`）を追加
- **量子化方式を ORT `quantize_static` → per-block absmax 量子化に変更**
  - Block-FP8 スタイル: ウェイトテンソルをブロック単位（デフォルト 32 要素）に分割し、各ブロック内で absmax スケーリングで量子化
  - DequantizeLinear を使わず標準ノード（Cast, Mul, Reshape, Slice）のみで復元サブグラフを構築
  - TensorRT の zero_point ≠ 0 非サポート問題と Int32 DQ ノード問題を根本的に解消
- **感度検出を 2 段階ハイブリッドに刷新**
  - Stage 1: ウェイト再構成誤差（NRMSE）で高速フィルタリング
  - Stage 2: 出力差検証で「疑い」ノードのみ精密検査
  - 精度劣化の大きいノードは自動的に量子化対象から除外され BF16/FP16 のまま維持
- **opset デフォルトを 18 → 19 に変更**（FP8E4M3FN に opset 19 が必要）
- **キャリブレーションデータが不要に**: weight-only 量子化のため、データセットからのキャリブレーション収集が不要。感度検出 Stage 2 のみ少数（デフォルト 8 枚）のプローブサンプルを使用

#### `weight_conversion.py` — 新規作成

- BF16/FP16 全体変換（outdated 版の機能を移植）
- per-block INT8/FP8 量子化エンジン（`quantize_weight_per_block`, `dequantize_weight_per_block`）
- ONNX 復元サブグラフ構築（`build_block_dequant_subgraph`）
- 2 段階感度検出（`find_sensitive_initializers`）

#### `onnx_graph_utils.py` — 新規作成

- outdated 版のグラフ操作ユーティリティを移植・改良
- BF16/FP16 全グラフ変換（`convert_graph_to_bf16`, `convert_graph_to_fp16`）
- AnomalyClassifier ノード特定・イニシャライザ復元機能

#### `ARCHITECTURE.md` — 新規作成

- コード構造・処理フロー・設計判断の根拠をドキュメント化

#### `finetune_facenet.py`

- **AMP dtype を fp16 → bf16 に変更**（`AMP_DTYPE = "bf16"`）
  - fp16 では backbone activation が 65504 を超えると Inf が発生し、`BatchNorm1d` の `running_mean` / `running_var` が NaN に汚染される問題があった
  - bf16 は指数部が fp32 と同じ 8 bit（最大値 3.4e38）のため、この Inf オーバーフローが発生しない
  - H200（Hopper）では bf16 の Tensor Core 速度は fp16 と同等のため速度低下なし
- **`GradScaler` を bf16 時は自動無効化**（`_use_scaler = USE_AMP and AMP_DTYPE == "fp16"`）
  - bf16 はオーバーフローしないため gradient scaling が不要

### 2026-03-21

#### `export_onnx.py`
- モデル再構築時に `DROPOUT` / `ARC_S` / `ARC_M` を saved script から読んだ値 (`run_cfg`) で正しく渡すよう修正（コンストラクタのデフォルト値との不一致を解消）
- fp32 ONNX エクスポート時に `input_size` / `imagenet_mean` / `imagenet_std` を `metadata_props` に書き込むよう追加（`infer_onnx.py` がハードコードデフォルトにフォールバックしなくなる）
- `export_fp16_onnx` / `export_bf16_onnx` / `export_int8_onnx` に `has_anomaly` パラメータを追加し、`verify_dynamic_batch` に伝搬
- NaN 判定を `x != x` イディオムから `math.isnan(x)` に変更

#### `verification.py`
- `verify_dynamic_batch` に `has_anomaly: bool = False` パラメータを追加。`True` のとき `outputs[1]` (anomaly_score) の shape `(bs,)` も検証する
- `DEFAULT_EVAL_SAMPLES` の重複定義を削除し、`data_utils.py` からインポートに統一

#### `infer_onnx.py`
- `_load_image_config_from_onnx` の `input_size` 解決順序を修正：メタデータキー `input_size` → ONNX 入力 shape → デフォルト `224` の順に優先。旧実装は ONNX shape が存在するとメタデータ値を上書きしていた
