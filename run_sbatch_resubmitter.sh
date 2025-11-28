#!/bin/bash
# 学習完了までジョブを繰り返し提出するラッパースクリプト
#
# 使い方:
#   ./run_sbatch_resubmitter.sh configs/glint360k_hgnetv2_b1.py
#   ./run_sbatch_resubmitter.sh configs/glint360k_hgnetv2_b1.py 100 30
#
# 引数:
#   $1: configファイル（必須）
#   $2: 最大試行回数（デフォルト: 100）
#   $3: ポーリング間隔秒（デフォルト: 30）
#   $4: ジョブスクリプト（デフォルト: run_sbatch_restart_worker.sh）

# 共通関数の読み込み
cd /home/apacsc14/face_recognition_train || exit 1
source scripts/training_common.sh

CONFIG_FILE="${1:-}"
MAX_ATTEMPTS=${2:-100}
POLL_INTERVAL=${3:-30}
JOB_SCRIPT=${4:-run_sbatch_restart_worker.sh}

# 引数検証
if [ -z "$CONFIG_FILE" ]; then
  echo "Usage: $0 <config_file> [max_attempts] [poll_interval] [job_script]" >&2
  echo "Example: $0 configs/glint360k_hgnetv2_b1.py 100 30" >&2
  exit 2
fi

validate_config "$CONFIG_FILE" || exit 2

if [ ! -f "$JOB_SCRIPT" ]; then
  echo "Job script $JOB_SCRIPT not found." >&2
  exit 2
fi

# configファイルからoutputディレクトリを取得
OUTPUT_DIR=$(get_output_dir "$CONFIG_FILE") || { echo "Failed to extract output directory" >&2; exit 2; }

echo "Config file: $CONFIG_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Max attempts: $MAX_ATTEMPTS"
echo "Poll interval: ${POLL_INTERVAL}s"

attempt=0
while [ $attempt -lt $MAX_ATTEMPTS ]; do
  # 完了チェック
  if is_training_done "$OUTPUT_DIR"; then
    echo "Training completed!"
    exit 0
  fi
  
  attempt=$((attempt+1))
  echo "[attempt $attempt/$MAX_ATTEMPTS] Submitting job: $JOB_SCRIPT"
  
  # CONFIG_FILEを環境変数として渡す
  jid=$(sbatch --parsable --export=ALL,CONFIG_FILE="$CONFIG_FILE" "$JOB_SCRIPT") || { echo "sbatch failed"; sleep $POLL_INTERVAL; continue; }
  echo "Submitted job $jid"

  # ジョブがキューを離れるまで待機
  while squeue -h -j "$jid" >/dev/null 2>&1; do
    sleep $POLL_INTERVAL
    if is_training_done "$OUTPUT_DIR"; then
      echo "Training completed during job; cancelling $jid"
      scancel "$jid" 2>/dev/null || true
      exit 0
    fi
  done

  # ジョブ終了後のチェック
  if is_training_done "$OUTPUT_DIR"; then
    echo "Training completed after job $jid."
    exit 0
  fi

  echo "Job $jid finished but training not complete. Resubmitting..."
done

echo "Reached max attempts ($MAX_ATTEMPTS). Training not confirmed complete. Exiting with code 1."
exit 1
