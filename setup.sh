#!/bin/bash

# ──────────────────────────────────────────────
#  ASL Translator — One-click setup script
#  Run this once to build the app on your Mac
# ──────────────────────────────────────────────

echo ""
echo "╔══════════════════════════════════════╗"
echo "║      ASL Translator Setup Script     ║"
echo "╚══════════════════════════════════════╝"
echo ""

# ── Step 1: Check Python ──────────────────────
echo "▶ Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install it from https://python.org"
    exit 1
fi
PYTHON_VERSION=$(python3 -c "import sys; print(sys.version_info.minor)")
if [ "$PYTHON_VERSION" -lt 9 ]; then
    echo "❌ Python 3.9+ required. Please update Python."
    exit 1
fi
echo "✅ Python3 found"

# ── Step 2: Dataset info ─────────────────────
echo ""
echo "▶ Dataset path is configured via the app's Settings page on first launch."
echo "   Make sure your dataSet/wlasl-complete folder is accessible on this machine."
echo "✅ Skipping dataset check (handled by config.json at runtime)"

# ── Step 3: Create virtual environment ────────
echo ""
echo "▶ Creating virtual environment..."
if [ ! -d "myenv" ]; then
    python3 -m venv myenv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# ── Step 4: Activate and install packages ─────
echo ""
echo "▶ Installing required packages (this may take a few minutes)..."
source myenv/bin/activate

pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

if [ $? -ne 0 ]; then
    echo "❌ Package installation failed. Check requirements.txt"
    exit 1
fi
echo "✅ Packages installed"

# ── Step 5: Build the app ─────────────────────
echo ""
echo "▶ Building ASL-Translator.app (this takes 1-3 minutes)..."
MEDIAPIPE_DIR=$(python3 -c "import mediapipe, os; print(os.path.dirname(mediapipe.__file__))")
echo "   MediaPipe found at: $MEDIAPIPE_DIR"

pyinstaller --onedir --windowed \
    --name "ASL-Translator" \
    --add-data "kily_module:kily_module" \
    --add-data "drews_module:drews_module" \
    --add-data "$MEDIAPIPE_DIR:mediapipe" \
    --hidden-import cv2 \
    --hidden-import numpy \
    --hidden-import mediapipe \
    --collect-all cv2 \
    --collect-all mediapipe \
    launcher.py \
    --noconfirm \
    --log-level WARN

if [ $? -ne 0 ]; then
    echo "❌ Build failed. Check the output above for errors."
    exit 1
fi
echo "✅ App built successfully"

# ── Step 6: Move to Applications ──────────────
echo ""
echo "▶ Installing to /Applications..."
cp -r dist/ASL-Translator.app /Applications/
chmod +x /Applications/ASL-Translator.app

if [ $? -eq 0 ]; then
    echo "✅ App installed to /Applications/ASL-Translator.app"
else
    echo "⚠️  Could not copy to /Applications (permission issue)"
    echo "   You can manually drag dist/ASL-Translator.app to Applications"
fi

# ── Done ──────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════╗"
echo "║           Setup Complete! 🎉         ║"
echo "╠══════════════════════════════════════╣"
echo "║                                      ║"
echo "║  App location:                       ║"
echo "║  /Applications/ASL-Translator.app    ║"
echo "║                                      ║"
echo "║  OR double-click:                    ║"
echo "║  dist/ASL-Translator.app             ║"
echo "║                                      ║"
echo "║  Set dataset path in Settings        ║"
echo "║  on first launch of the app          ║"
echo "║                                      ║"
echo "╚══════════════════════════════════════╝"
echo ""