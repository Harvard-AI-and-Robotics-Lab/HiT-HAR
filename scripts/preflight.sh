#!/usr/bin/env bash
# HiT-HAR Preflight Check
# Verifies environment, data files, and import chain before training.
set -e

echo "=== HiT-HAR Preflight Check ==="
echo ""

PYTHON_BIN="${PYTHON:-python}"

# 1. Python version
echo "[1/5] Python..."
"$PYTHON_BIN" --version || { echo "FAIL: Python not found: $PYTHON_BIN"; exit 1; }

# 2. PyTorch
echo "[2/5] PyTorch..."
"$PYTHON_BIN" -c "import torch; print(f'  torch {torch.__version__}')" || { echo "FAIL: torch not importable"; exit 1; }

# 3. CUDA
echo "[3/5] CUDA..."
"$PYTHON_BIN" -c "
import torch
if torch.cuda.is_available():
    print(f'  CUDA available: {torch.cuda.get_device_name(0)}')
else:
    print('  CUDA not available (will use CPU)')
"

# 4. Data files
echo "[4/5] Data files..."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

check_file() {
    if [ -f "$1" ]; then
        echo "  OK: $1"
    else
        echo "  MISSING: $1"
    fi
}

check_file "$ROOT_DIR/data/gold/har_gold_unified.csv"
check_file "$ROOT_DIR/data/labels/scenario_labels.csv"
check_file "$ROOT_DIR/configs/baseline_5class.yaml"

# 5. Import chain
echo "[5/5] Import chain..."
cd "$ROOT_DIR"
"$PYTHON_BIN" -c "
import sys; sys.path.insert(0, '.')
from src.config import load_config
from src.models.hit_har import build_model
from src.data.paired_dataset import PairedDataset, ACTION_MAP, SCENARIO_MAP
from src.training.losses_hit_har import HiTHARLoss, FocalLoss
from src.training.trainer import HiTHARTrainer
from src.evaluation.evaluate import evaluate

assert 'Task Operation' in ACTION_MAP, 'ACTION_MAP not updated!'
assert 'Essential Operation' not in ACTION_MAP, 'Old label still present!'

print('  All imports OK')
print(f'  ACTION_MAP: {ACTION_MAP}')
"

echo ""
echo "=== Preflight PASSED ==="
