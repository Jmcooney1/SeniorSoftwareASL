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
echo "▶ Dataset path is configured via config.json (created automatically)."
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
echo "▶ Building ASL-Translator distribution folder..."
MEDIAPIPE_DIR=$(python3 -c "import mediapipe, os; print(os.path.dirname(mediapipe.__file__))")
echo "   MediaPipe found at: $MEDIAPIPE_DIR"

# Resolve absolute paths so PyInstaller can always find the data folders
KILY_DIR="$(pwd)/kily_module"
DREWS_DIR="$(pwd)/drews_module"

pyinstaller --onedir --windowed \
    --name "ASL-Translator" \
    --distpath "$(pwd)/dist_temp" \
    --workpath "$(pwd)/build_temp" \
    --add-data "$KILY_DIR:kily_module" \
    --add-data "$DREWS_DIR:drews_module" \
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
echo "✅ App built"

# ── Step 6: Assemble the final folder layout ───
echo ""
echo "▶ Creating ASL-Translator distribution layout..."

DIST="$(pwd)/dist/ASL-Translator"
rm -rf "$DIST"
mkdir -p "$DIST/dataset"
mkdir -p "$DIST/savedVideoPoints"

# Move the built bundle into ASL_app/
mv "$(pwd)/dist_temp/ASL-Translator" "$DIST/ASL_app"

# Clean up temp build artifacts
rm -rf "$(pwd)/dist_temp"
rm -rf "$(pwd)/build_temp"
rm -f  "$(pwd)/ASL-Translator.spec"

if [ $? -eq 0 ]; then
    echo "✅ Layout created"
else
    echo "❌ Failed to assemble layout"
    exit 1
fi

# ── Step 7: Create config.json ────────────────
echo ""
echo "▶ Creating config.json..."

cat > "$DIST/config.json" << EOF
{
    "dataset_path": "$DIST/dataset",
    "save_dir": "$DIST/savedVideoPoints"
}
EOF

if [ $? -eq 0 ]; then
    echo "✅ config.json created"
else
    echo "❌ Failed to create config.json"
    exit 1
fi

# ── Done ──────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════╗"
echo "║           Setup Complete! 🎉         ║"
echo "╠══════════════════════════════════════╣"
echo "║                                      ║"
echo "║  Distribution folder:                ║"
echo "║  dist/ASL-Translator/                ║"
echo "║  ├── ASL_app/   ← launch from here   ║"
echo "║  ├── dataset/   ← put your data here ║"
echo "║  ├── savedVideoPoints/               ║"
echo "║  └── config.json                     ║"
echo "║                                      ║"
echo "║  Drop your dataset here:             ║"
echo "║  dataset/kily_dataset/wlasl-complete/║"
echo "║  ├── wlasl_class_list.txt            ║"
echo "║  └── videos/                         ║"
echo "║                                      ║"
echo "╚══════════════════════════════════════╝"
echo ""