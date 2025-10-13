#!/bin/bash
echo "Cleaning up install: venv, torch_extensions, and .so files..."
rm -rf .venv
rm -rf ~/.cache/torch_extensions
rm -rf ~/.cache/uv/builds-v0
# Note: Keeping ccache to speed up rebuilds
# To clear ccache, run: ccache -C
# pycache recursively
find . -name "__pycache__" -type d -exec rm -rf {} +
find . -name "*.so*" -type f -delete
# Clean build artifacts
rm -rf build
rm -rf dist
rm -rf *.egg-info

echo "Done cleaning."
echo ""
echo "Reinstall the package to rebuild CUDA extensions:"
echo "     uv venv && source .venv/bin/activate"
echo "     uv pip install -e '.[dev]'"
echo ""
echo "To check if kernels are available, run:"
echo "     python -c 'import spear._btp; print(spear._btp)'"
echo ""
echo "If you want to clean the ccache too, run:"
echo "     ccache -C"