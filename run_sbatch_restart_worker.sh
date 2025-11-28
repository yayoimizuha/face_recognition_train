#!/bin/bash
#SBATCH --job-name=facenet_trainer_restart
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
##SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=01:00:00

# 短時間スロットを繰り返して学習を完了させるためのジョブスクリプト
#
# 使い方:
#   sbatch run_sbatch_restart_worker.sh configs/glint360k_hgnetv2_b1.py
#   または環境変数で指定:
#   CONFIG_FILE=configs/glint360k_hgnetv2_b1.py sbatch run_sbatch_restart_worker.sh

# 共通関数の読み込み
cd /home/apacsc14/face_recognition_train || exit 1
source scripts/training_common.sh

# configファイルの指定（引数 > 環境変数、指定なしはエラー）
CONFIG_FILE="${1:-${CONFIG_FILE:-}}"

if [ -z "$CONFIG_FILE" ]; then
  log_error "Config file not specified"
  echo "Usage: sbatch $0 <config_file>" >&2
  echo "   or: CONFIG_FILE=<config_file> sbatch $0" >&2
  exit 1
fi

validate_config "$CONFIG_FILE" || exit 1

log_info "Using config: $CONFIG_FILE"

# configファイルからoutputディレクトリを取得
OUTPUT_DIR=$(get_output_dir "$CONFIG_FILE") || { log_error "Failed to extract output directory"; exit 1; }

log_info "Output directory: $OUTPUT_DIR"

# データの準備
export DATA_TMP_DIR=/tmp/facenet_dataset_tempdir
mkdir -p "$DATA_TMP_DIR"/{Glint_360k,validation}

log_info "Preparing dataset temp dir: $DATA_TMP_DIR"

# データのコピー
find ~/Glint360k -type f -mindepth 1 -maxdepth 1 -printf "%f\n" | xargs -IX -t -P 10 cp ~/Glint360k/X "$DATA_TMP_DIR/Glint_360k/X" || true
cp ~/face_valid/*.bin "$DATA_TMP_DIR/validation/" || true

# ログ設定
LOGDIR="$PROJECT_ROOT/logs"
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/train_${SLURM_JOB_ID:-local}.log"

log_info "Start training, log -> $LOGFILE"

# 学習実行
OMP_NUM_THREADS=12 uv run --no-sync torchrun --nproc-per-node 8 train_v2_restart.py "$CONFIG_FILE" 2>&1 | tee "$LOGFILE"

EXIT_CODE=${PIPESTATUS[0]:-0}
log_info "Training process exited with code $EXIT_CODE"

# 完了チェック（model.ptが存在すれば完了）
if is_training_done "$OUTPUT_DIR"; then
  log_info "Training completed! Final model found: $OUTPUT_DIR/model.pt"
  touch "$OUTPUT_DIR/TRAINING_DONE"
else
  log_info "Training not yet complete (model.pt not found). Will be resubmitted."
fi

# 一時データのクリーンアップ
log_info "Cleaning temp dir: $DATA_TMP_DIR"
find "$DATA_TMP_DIR" -type f | xargs -n 1 -P 10 truncate -s 0 || true
rm -rf "$DATA_TMP_DIR" || true

log_info "Job finished"
exit 0
