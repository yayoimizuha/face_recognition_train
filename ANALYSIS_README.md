# バッチサイズ拡大時の問題分析レポート / Batch Size Scaling Issue Analysis

## レポートファイル / Report Files

1. **`batch_size_scaling_analysis_report.md`** (日本語 / Japanese)
   - 詳細な問題分析と対策案を日本語で記載
   - Detailed analysis and solutions in Japanese

2. **`batch_size_scaling_analysis_report_EN.md`** (English)
   - English version of the analysis report
   - 分析レポートの英語版

## 問題の要約 / Summary

### 日本語

**問題**: 
`torchrun --nproc_per_node=8 train_v2_restart.py configs/glint360k_mbv4_hybrid_med.py` を使用した分散学習環境で、バッチサイズを拡大してもlossが適切に減少せず、精度が向上しない。

**根本原因**:
1. **最重要**: `partial_fc_v2.py`の`DistCrossEntropyFunc.backward()`（227行目）で勾配がローカルバッチサイズで除算されているが、グローバルバッチサイズで除算すべき
   - 結果: 8GPU環境で64倍の勾配スケール誤差が発生
2. **重要**: 学習率がLinear Scaling Ruleに従ってスケールされていない
3. **補助的**: Warmup期間が極端に短い（0.04エポック）

**推奨対策の優先順位**:
1. 優先度1（必須）: 勾配スケーリングの修正
2. 優先度2（強く推奨）: 学習率のLinear Scaling適用  
3. 優先度3（推奨）: Warmupエポック数の増加
4. 優先度4-5（状況に応じて）: AdamWパラメータ調整と勾配累積の活用

詳細は `batch_size_scaling_analysis_report.md` を参照してください。

### English

**Problem**: 
When using `torchrun --nproc_per_node=8 train_v2_restart.py configs/glint360k_mbv4_hybrid_med.py` in distributed training, loss does not decrease properly and accuracy does not improve even when batch size is increased.

**Root Causes**:
1. **Most Critical**: In `partial_fc_v2.py`, `DistCrossEntropyFunc.backward()` (line 227) divides gradients by local batch size, but should divide by global batch size
   - Result: 64× gradient scaling error in 8-GPU environment
2. **Important**: Learning rate not scaled according to Linear Scaling Rule
3. **Supporting**: Warmup period extremely short (0.04 epochs)

**Recommended Solutions (Priority Order)**:
1. Priority 1 (Critical): Fix gradient scaling
2. Priority 2 (Highly Recommended): Apply Linear Scaling Rule for learning rate
3. Priority 3 (Recommended): Increase warmup epochs
4. Priority 4-5 (Situational): Adjust AdamW parameters and utilize gradient accumulation

See `batch_size_scaling_analysis_report_EN.md` for details.

## 実装されていない理由 / Why Not Implemented

ユーザーのリクエストに従い、**リポジトリの編集は行わず、レポートのみを作成しました**。

As requested by the user, **no repository modifications were made - only analysis reports were created**.

## 次のステップ / Next Steps

### レポートを確認した後の推奨アクション / Recommended Actions After Reviewing Reports

1. **最優先**: `partial_fc_v2.py`の勾配スケーリング修正を適用
   - Apply gradient scaling fix in `partial_fc_v2.py` (Priority 1)

2. **学習率調整**: Linear Scaling Ruleに基づいた学習率設定
   - Adjust learning rate based on Linear Scaling Rule (Priority 2)

3. **実験**: 修正後、異なるバッチサイズで学習曲線を比較
   - After fixes, compare learning curves with different batch sizes (Priority 3)

各レポートには具体的なコード例と実装ガイドが含まれています。
Each report includes specific code examples and implementation guides.

## 技術的な詳細 / Technical Details

### 勾配スケーリングの問題 / Gradient Scaling Issue

**現在の実装 / Current Implementation**:
```python
# partial_fc_v2.py:227
logits.div_(batch_size)  # batch_size = ローカルバッチサイズ / local batch size
```

**結果 / Result**:
- 実効勾配スケール / Effective gradient scale = `world_size / local_batch_size`
- 8GPU、BS=128の場合 / With 8 GPUs, BS=128: `8/128 = 1/16`
- 期待値 / Expected: `1/(128*8) = 1/1024`
- 誤差 / Error: **64倍 / 64×**

**修正案 / Proposed Fix**:
```python
# partial_fc_v2.py:227付近
world_size = distributed.get_world_size()
global_batch_size = batch_size * world_size
logits.div_(global_batch_size)
```

### 学習率スケーリング / Learning Rate Scaling

**Linear Scaling Rule**:
```
新しい学習率 / New LR = 基本学習率 / Base LR × (現在のバッチサイズ / 参照バッチサイズ)
                      = Base LR × (Current Batch Size / Reference Batch Size)
```

**例 / Example**:
- 参照バッチサイズ / Reference BS: 256
- 基本学習率 / Base LR: 0.1
- 現在のバッチサイズ / Current BS: 1024 (8 GPUs × 128)
- 推奨学習率 / Recommended LR: 0.1 × (1024 / 256) = **0.4**

## 参考資料 / References

各レポートに詳細な参考文献リストが含まれています。主要な論文：
Each report includes a detailed reference list. Key papers:

1. Goyal et al. (2017) - "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour"
2. You et al. (2019) - "Large Batch Optimization for Deep Learning: Training BERT in 76 minutes"
3. InsightFace PartialFC - "Partial FC: Training 10 Million Identities on a Single Machine"

## サポート / Support

質問や追加の分析が必要な場合は、Issueを開いてください。
For questions or additional analysis needs, please open an issue.

---

**作成日 / Created**: 2025-11-03  
**バージョン / Version**: 1.0
