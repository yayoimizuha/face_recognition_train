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
MobileNetV4-Hybrid-Medium + GWAP + ArcFace でファインチューニングを行う。

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
| `model_final.pt` | 最終エポックの重み |
| `model_epochN.pt` | `SAVE_INTERVAL` ごとの定期チェックポイント |
| `confusion_matrix.png` | 学習データ全体での混同行列 |
| `finetune_facenet.py` | 再現性のためにコピーされた実行スクリプト |

### 主な設定値（スクリプト冒頭の定数）

| 定数 | デフォルト値 | 説明 |
|---|---|---|
| `NUM_EPOCHS` | 200 | 学習エポック数 |
| `BATCH_SIZE` | 128 | バッチサイズ |
| `LR` | 2e-3 | head / GWAP / ArcFace の学習率 |
| `LR_BACKBONE` | 2e-4 | backbone の学習率 |
| `EMB_SIZE` | 512 | 埋め込みベクトルの次元数 |
| `VAL_RATIO` | 0.2 | 検証データの割合 |

---

## export_onnx.py — ONNX エクスポート

学習済みの `.pt` チェックポイントから 5 種類の ONNX モデルを生成する。
クラスラベルや入力サイズなどのメタデータが ONNX ファイルに埋め込まれる。

### 基本的な使い方

```bash
python hp_finetune/export_onnx.py \
    --checkpoint hp_finetune/work_dirs/<タイムスタンプ>/model_best.pt
```

### オプション

| オプション | デフォルト | 説明 |
|---|---|---|
| `--checkpoint` | (必須) | `.pt` チェックポイントのパス |
| `--num-classes` | 自動検出 | クラス数（省略時はチェックポイントから自動取得） |
| `--calib-samples` | 64 | INT8 量子化のキャリブレーションサンプル数 |
| `--eval-samples` | 32 | 各バリアントの品質評価サンプル数 |
| `--opset` | 18 | ONNX opset バージョン |
| `--calib-method` | `minmax` | INT8 キャリブレーション手法（`minmax` / `entropy` / `percentile`） |
| `--percentile` | 99.999 | `--calib-method percentile` 使用時のクリッピングパーセンタイル値 |

#### `--calib-method` の選択基準

| 手法 | 計算コスト | 精度 | 説明 |
|---|---|---|---|
| `minmax` | 低 | 中 | 観測した最小・最大値をそのままスケールに使う（デフォルト） |
| `entropy` | 中〜高 | 高 | fp32 と INT8 の出力分布の KL ダイバージェンスを最小化する（TensorRT 相当） |
| `percentile` | 低 | 中〜高 | `--percentile` で指定したパーセンタイル値でクリッピングし外れ値の影響を排除する |

### 出力

チェックポイントと同じディレクトリに以下が生成される。

| ファイル名 | 内容 |
|---|---|
| `<stem>.onnx` | fp32（基本モデル） |
| `<stem>_fp16.onnx` | fp16（GPU Ampere 以降で高速） |
| `<stem>_bf16.onnx` | bf16（GPU bf16 対応環境向け） |
| `<stem>_fp16int8.onnx` | INT8 量子化 + fp16 残余重み |
| `<stem>_bf16int8.onnx` | INT8 量子化 + bf16 残余重み |

> **注意:** fp16 / bf16 モデルは GPU 環境（Ampere 以降）での推論を想定している。
> CPU では fp32 と速度差がほとんどない。

### モデル出力について

- 出力は **分類ロジット** `(batch_size, num_classes)`
- 確率が必要な場合は消費側で softmax を適用する
- バッチサイズは動的（任意のバッチサイズで推論可能）

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
| `--top-k` | 20 | 混同上位ペアの表示件数 |
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
| `wrong_predictions.csv` | 誤分類サンプルの一覧（index、真ラベル、予測ラベル、確信度、画像パス） |
| `confusion_ranking.csv` | 誤分類ペア `(true_class, pred_class, count)` の件数降順ランキング |
| `class_accuracy.csv` | クラスごとの正解率（昇順ソート、最も混同されやすいクラスが先頭） |
| `wrong_images/<pred_class>/<true_class>_NNN.jpg` | 誤分類された元画像 |

---

## 典型的なワークフロー

```
1. finetune_facenet.py  →  work_dirs/<timestamp>/model_best.pt
2. export_onnx.py       →  work_dirs/<timestamp>/model_best.onnx（他 4 バリアント）
3. infer_onnx.py        →  work_dirs/<timestamp>/infer_results/<stem>/
```
