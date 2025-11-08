#!/bin/bash
# Script to run training on Google Cloud TPU v5e-8
# Reference: https://docs.pytorch.org/xla/release/r2.8/index.html

# Check if config file is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <config_file>"
    echo "Example: $0 configs/edgeface_s_gamma_05_tpu.py"
    exit 1
fi

CONFIG=$1

# Install torch_xla if not already installed
# This assumes you're on a TPU VM with Python 3.12
echo "Checking PyTorch/XLA installation..."
python3 -c "import torch_xla" 2>/dev/null || {
    echo "Installing PyTorch/XLA for TPU..."
    pip install torch~=2.8.0 torchvision~=0.23.0
    pip install https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.8.0-cp312-cp312-linux_x86_64.whl
}

# Set environment variables for TPU
export XLA_USE_BF16=1  # Enable bfloat16 on TPU
export PJRT_DEVICE=TPU  # Use TPU runtime

# Run training on TPU
# The train_v2_tpu.py script will automatically use all 8 cores of TPU v5e-8
python3 train_v2_tpu.py $CONFIG
