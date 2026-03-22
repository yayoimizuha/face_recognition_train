# ONNX Export Architecture

## 概要

学習済み `.pt` チェックポイントから 7 種類の ONNX モデルを生成するエクスポートパイプライン。
ウェイトノードを「量子化済みウェイト × スケーリング係数」のサブグラフに付け替える
Block 量子化方式を採用し、DequantizeLinear を使わず標準 ONNX ノード（Cast, Mul, Reshape）
のみで構成することで TensorRT 互換性を確保する。

## 出力モデル一覧

| モデル名    | ウェイト格納      | 計算 dtype | AnomalyClassifier | サフィックス      |
|-------------|-------------------|-----------|-------------------|-------------------|
| FP32        | fp32              | fp32      | fp32              | `.onnx`           |
| BF16        | bf16              | bf16      | fp32              | `_bf16.onnx`      |
| FP16        | fp16              | fp16      | fp32              | `_fp16.onnx`      |
| BF16+INT8   | int8 per-block    | bf16      | fp32              | `_bf16int8.onnx`  |
| BF16+FP8    | fp8e4m3 per-block | bf16      | fp32              | `_bf16fp8.onnx`   |
| FP16+INT8   | int8 per-block    | fp16      | fp32              | `_fp16int8.onnx`  |
| FP16+FP8    | fp8e4m3 per-block | fp16      | fp32              | `_fp16fp8.onnx`   |

## ファイル構成

```
hp_finetune/
├── export_onnx.py          # メインエクスポートスクリプト (CLI + フロー制御)
├── weight_conversion.py    # ウェイト変換エンジン
│                           #   - BF16/FP16 全体変換 (Cast 挿入)
│                           #   - Per-block INT8/FP8 量子化 + 復元サブグラフ構築
├── onnx_graph_utils.py     # ONNX グラフ操作ユーティリティ
│                           #   - バッチ次元動的化
│                           #   - Reshape 修正
│                           #   - AnomalyClassifier ノード特定
│                           #   - メタデータ埋め込み
│                           #   - Shape inference (TensorRT 用)
├── config_loader.py        # チェックポイント横の finetune_facenet.py から定数を読み取る
├── data_utils.py           # データセットアクセス・画像前処理・キャリブレーション
├── verification.py         # 動的バッチ検証・出力比較・品質評価
├── finetune_facenet.py     # 学習スクリプト (モデル定義含む)
├── infer_onnx.py           # ONNX 推論・誤分類分析
├── ARCHITECTURE.md         # ← このファイル
└── USAGE.md                # 使い方ドキュメント
```

## 処理フロー

```
                    checkpoint (.pt)
                         │
                    ┌────▼────┐
                    │ Step 1  │  PyTorch モデル構築
                    │ load    │  (ClassificationModel / ClassificationWithAnomalyModel)
                    └────┬────┘
                         │
                    ┌────▼────┐
                    │ Step 2  │  torch.onnx.export → FP32 ONNX
                    │ fp32    │  + Reshape 修正 + バッチ動的化 + メタデータ
                    └────┬────┘
                         │
              ┌──────────┼──────────┐
              │          │          │
         ┌────▼────┐┌───▼────┐     │
         │ Step 3  ││ Step 4 │     │
         │ bf16    ││ fp16   │     │
         │ 全体変換 ││ 全体変換│     │
         └────┬────┘└───┬────┘     │
              │         │          │
      ┌───────┼───┐ ┌───┼───┐      │
      │       │   │ │   │   │      │
   ┌──▼──┐┌──▼──┐│┌▼──┐│┌──▼──┐   │
   │bf16 ││bf16 │││fp16│││fp16 │   │
   │+int8││+fp8 │││+i8 │││+fp8 │   │
   └─────┘└─────┘│└────┘│└─────┘   │
                 │      │          │
                 └──────┘          │
                                   │
         ┌─────────────────────────┘
         │
    ┌────▼────┐
    │ Step 5  │  感度検出 (2段階ハイブリッド)
    │ detect  │  → 量子化OK / 除外 の判定
    └────┬────┘
         │
    ┌────▼────┐
    │ Step 6  │  全モデル品質評価 + ファイルサイズ報告
    │ verify  │
    └─────────┘
```

## Per-Block 量子化の仕組み

### 概念

Block-FP8 スタイルの量子化を INT8 / FP8 の両方に適用する。
ウェイトテンソルをブロック単位（デフォルト 32 要素）に分割し、
各ブロック内で absmax スケーリングを行って量子化する。

### 量子化パラメータ

| dtype    | 最大表現値 | scale 計算式                     |
|----------|-----------|----------------------------------|
| INT8     | 127       | `scale = absmax(block) / 127.0`  |
| FP8E4M3  | 448       | `scale = absmax(block) / 448.0`  |

### ONNX グラフ上の復元サブグラフ

元のグラフ:
```
weight_fp32 ──► Conv/Gemm/MatMul
```

量子化後のグラフ (標準ノードのみ、DequantizeLinear 不使用):
```
weight_quantized [int8/fp8, shape=(num_blocks, block_size)]
  │
  ├─► Cast(to=FLOAT/FLOAT16/BFLOAT16)
  │     │
  │     ▼
  │   Mul(× scale [shape=(num_blocks, 1)])   ← broadcast
  │     │
  │     ▼
  │   Reshape(to=padded_flat_shape)
  │     │
  │     ▼
  │   Slice(start=0, end=original_numel)     ← パディング除去
  │     │
  │     ▼
  │   Reshape(to=original_weight_shape)
  │     │
  └───► (元の Consumer ノードの input に接続)
```

### 感度検出 (2段階ハイブリッド)

量子化するとモデル出力が著しく劣化するウェイトノードを自動検出し、
そのノードは量子化せず BF16/FP16 のまま維持する。

**Stage 1: ウェイト再構成誤差 (高速、推論不要)**

各ウェイトテンソルについて:
1. per-block 量子化 → 復元
2. NRMSE = sqrt(mean((orig - restored)²)) / sqrt(mean(orig²)) を計算
3. NRMSE > weight_nrmse_threshold → 「疑い」リストに追加
4. NRMSE ≤ threshold → 量子化 OK (Stage 2 スキップ)

**Stage 2: 出力差検証 (「疑い」ノードのみ、推論実行)**

「疑い」ノードそれぞれについて:
1. そのノードだけを量子化したグラフを構築
2. 少数のサンプルで fp32 と量子化グラフの両方を推論
3. max_abs_diff > output_diff_threshold → 除外確定
4. max_abs_diff ≤ threshold → 量子化 OK

## モジュール間の依存関係

```
export_onnx.py
  ├── weight_conversion.py     (ウェイト変換・量子化)
  ├── onnx_graph_utils.py      (グラフ操作)
  ├── config_loader.py         (RunConfig 読み込み)
  ├── data_utils.py            (画像データ・キャリブレーション)
  ├── verification.py          (品質検証)
  └── finetune_facenet.py      (FaceRecognitionModel 定義)

weight_conversion.py
  └── (numpy, torch, onnx のみ — 外部モジュール依存なし)

onnx_graph_utils.py
  ├── (numpy, onnx, onnxruntime)
  └── weight_conversion.py  (convert_initializers / TargetDtype — bf16 変換時)

verification.py
  └── data_utils.py

data_utils.py
  └── (datasets, torchvision)
```

## 設計判断の根拠

### DequantizeLinear を使わない理由

TensorRT は DequantizeLinear の zero_point ≠ 0 をサポートしていない。
また INT32 の DequantizeLinear 入力も拒否される。
標準の Cast + Mul + Reshape で構成すれば、どのランタイムでも問題なく動作する。

### Per-block 量子化を採用する理由

Per-tensor (テンソル全体で 1 つの scale) は精度が低い。
Per-channel (出力チャネルごとに 1 つの scale) は Conv には適するが Gemm/MatMul
には適用しにくい。Per-block (固定ブロック単位) は粒度と精度のバランスが良く、
Block-FP8 の業界標準に近い。

### AnomalyClassifier ノードを fp32 で維持する理由

`AnomalyClassifier` は `fc1`（Linear 1024→256）、`BN`（BatchNorm1d）、`fc2`（Linear 256→1）で構成される。
最終的な sigmoid 出力は 0〜1 の確率値であり、閾値判定に用いる。
fp32 維持のサイズコストは軽微だが、BF16/FP16 では BN の `running_mean` / `running_var` に
量子化誤差が入り、小さなスコア差での閾値判定が不安定になる恐れがあるため fp32 を維持する。

### 感度検出を 2 段階にする理由

Stage 1 (ウェイト再構成誤差) は推論不要で高速だが、ウェイト誤差が小さくても
グラフ後段で誤差が増幅されるケースを見逃す。Stage 2 (出力差検証) は正確だが
全ノードに実行すると遅い。Stage 1 で大部分を通過させ、疑わしいノードのみ
Stage 2 で精密検査するハイブリッド方式が最もバランスが良い。
