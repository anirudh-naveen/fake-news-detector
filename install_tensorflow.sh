#!/bin/bash
# Script to install TensorFlow based on your macOS architecture

echo "Detecting your system architecture..."

ARCH=$(uname -m)
echo "Architecture: $ARCH"

if [ "$ARCH" = "arm64" ]; then
    echo "Detected Apple Silicon (M1/M2/M3). Installing tensorflow-macos and tensorflow-metal..."
    pip uninstall -y tensorflow tensorflow-macos tensorflow-metal 2>/dev/null
    pip install tensorflow-macos tensorflow-metal
    echo "✓ TensorFlow for Apple Silicon installed!"
elif [ "$ARCH" = "x86_64" ]; then
    echo "Detected Intel Mac. Installing standard TensorFlow..."
    pip uninstall -y tensorflow tensorflow-macos tensorflow-metal 2>/dev/null
    pip install tensorflow
    echo "✓ TensorFlow for Intel Mac installed!"
else
    echo "Unknown architecture. Installing standard TensorFlow..."
    pip install tensorflow
fi

echo ""
echo "Testing TensorFlow import..."
python -c "import tensorflow as tf; print(f'✓ TensorFlow {tf.__version__} imported successfully!')" || echo "✗ TensorFlow import failed"

