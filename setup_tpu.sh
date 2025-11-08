#!/bin/bash
# Setup script for Google Cloud TPU v5e-8 training environment
# Run this on your TPU VM after cloning the repository

set -e  # Exit on error

echo "========================================="
echo "TPU Training Environment Setup"
echo "========================================="

# Check Python version
PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

if [[ ! $PYTHON_VERSION == 3.12* ]]; then
    echo "Warning: This repository requires Python 3.12. Current version: $PYTHON_VERSION"
    echo "Continuing anyway, but you may encounter compatibility issues."
fi

# Check if we're on a TPU VM
if [ ! -f /etc/tpu_metadata.json ]; then
    echo "Warning: This doesn't appear to be a TPU VM."
    echo "The setup will continue, but TPU-specific features may not work."
fi

echo ""
echo "Step 1: Installing PyTorch and torchvision..."
pip install -q torch~=2.8.0 torchvision~=0.23.0

echo ""
echo "Step 2: Installing PyTorch/XLA for TPU..."
pip install -q https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.8.0-cp312-cp312-linux_x86_64.whl

echo ""
echo "Step 3: Installing other dependencies..."
pip install -q -r requirements-tpu.txt

echo ""
echo "Step 4: Verifying installation..."

# Verify PyTorch
python3 -c "import torch; print(f'✓ PyTorch {torch.__version__} installed')" || {
    echo "✗ PyTorch installation failed"
    exit 1
}

# Verify PyTorch/XLA
python3 -c "import torch_xla; print('✓ PyTorch/XLA installed')" || {
    echo "✗ PyTorch/XLA installation failed"
    exit 1
}

# Verify TPU devices
python3 -c "import torch_xla.core.xla_model as xm; print(f'✓ TPU devices available: {xm.xrt_world_size()}')" || {
    echo "✗ TPU devices not detected"
    echo "  Make sure you're on a TPU VM and the TPU runtime is available"
    exit 1
}

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Prepare your dataset in ImageFolder or WebDataset format"
echo "2. Update the config file: configs/edgeface_s_gamma_05_tpu.py"
echo "3. Run training: ./run_tpu.sh configs/edgeface_s_gamma_05_tpu.py"
echo ""
echo "For more information, see docs/TPU_TRAINING.md"
