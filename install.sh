#!/bin/bash

# Casting Solidification Simulator - Complete Setup Script
# For M3 MacBook Air

echo ""
echo "================================================================"
echo "    Casting Solidification Simulator - Installation"
echo "    Optimized for Apple M3 MacBook Air"
echo "================================================================"
echo ""

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "⚠️  Warning: This script is optimized for macOS (M3 chip)"
    echo "   It may still work on other systems but without M3 optimizations"
    echo ""
fi

# Check Python version
echo "Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    echo "✓ Python $PYTHON_VERSION found"
else
    echo "❌ Python 3 not found. Please install Python 3.9 or higher"
    echo "   Download from: https://www.python.org/downloads/"
    exit 1
fi

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 not found. Installing pip..."
    python3 -m ensurepip --upgrade
fi

echo ""
echo "================================================================"
echo "Installing Dependencies"
echo "================================================================"
echo ""
echo "This will install:"
echo "  • PyQt5 (GUI framework)"
echo "  • VTK (3D visualization)"
echo "  • NumPy, SciPy (numerical computing)"
echo "  • Numba (M3 optimization)"
echo "  • Trimesh (mesh processing)"
echo "  • PyOpenCL (GPU acceleration)"
echo "  • And more..."
echo ""
echo "⏱️  Estimated time: 5-10 minutes"
echo ""

read -p "Continue with installation? (y/n): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Installation cancelled."
    exit 0
fi

echo ""
echo "Installing packages..."
pip3 install -r requirements.txt

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Installation failed. Please check errors above."
    exit 1
fi

echo ""
echo "================================================================"
echo "✓ Installation Complete!"
echo "================================================================"
echo ""

# Test hardware detection
# config.hardware_config prints its detection summary on import. This replaces a
# call to a test_hardware.py that does not exist in the repo, which made the
# check always fail and always take the "Setup Incomplete" branch below.
echo "Testing hardware detection..."
python3 -c "import config.hardware_config"

if [ $? -eq 0 ]; then
    echo ""
    echo "================================================================"
    echo "Setup Complete - Ready to Use!"
    echo "================================================================"
    echo ""
    echo "Next steps:"
    echo ""
    echo "1. Generate sample STL files (optional):"
    echo "   python3 generate_samples.py"
    echo ""
    echo "2. Run the simulator:"
    echo "   python3 main.py"
    echo ""
    echo "3. Quick start guide:"
    echo "   cat QUICKSTART.md"
    echo ""
    echo "================================================================"
    echo ""
    
    read -p "Launch simulator now? (y/n): " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "Launching Casting Solidification Simulator..."
        echo ""
        python3 main.py
    fi
else
    echo ""
    echo "⚠️  Hardware test completed with warnings"
    echo "   The simulator will still work but may not be fully optimized"
    echo ""
fi
