#!/bin/bash
#SBATCH --job-name=facenet_trainer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
##SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=01:00:00

export DATA_TMP_DIR=/tmp/facenet_dataset_tempdir
mkdir -p /tmp/facenet_dataset_tempdir
mkdir $DATA_TMP_DIR/Glint_360k
mkdir $DATA_TMP_DIR/validation

find ~/Glint360k -type f -mindepth 1 -maxdepth 1 -printf "%f\n" | xargs -IX -t -P 10 cp ~/Glint360k/X $DATA_TMP_DIR/Glint_360k/X
cp ~/face_valid/*.bin $DATA_TMP_DIR/validation/ 

cd /home/apacsc14/face_recognition_train
OMP_NUM_THREADS=12 uv run --no-sync torchrun --nproc-per-node 8 train_v2.py configs/glint360k_hgnetv2_b1.py

find $DATA_TMP_DIR -type f | xargs -n 1 -P 10 truncate -s 0 
rm -rf $DATA_TMP_DIR
