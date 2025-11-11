#!/bin/bash
# Installation script for censim package

set -e  # Exit on error

echo ""
echo "============================================================"
echo "Installing censim with C k-mer library"
echo "============================================================"
echo ""

# Build C library
echo "Step 1/2: Building C k-mer library..."
cd src/censim/kmer_c
make shared
cd ../../..

# Install Python package
echo ""
echo "Step 2/2: Installing Python package..."
pip install -e .

echo ""
echo "============================================================"
echo "✓ Installation complete!"
echo "============================================================"
echo ""
echo "Test the installation with:"
echo "  python3 -c \"from censim.kmer_wrapper import analyze_repeat_sequence; print('Success!')\""
echo ""
