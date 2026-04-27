#!/usr/bin/env bash
# =============================================================================
# run_tests.sh — Run nobias_pipeline.py on each CIF in test_cifs/
#
# Usage (from inside nobias_cairo/):
#   bash test/run_tests.sh
#
# Or from the repo root:
#   bash classification/nobias_cairo/test/run_tests.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NOBIAS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CIF_DIR="$NOBIAS_DIR/test_cifs"
PIPELINE="$NOBIAS_DIR/nobias_pipeline.py"

# --- Locate model files (search relative to nobias_cairo and classification/) ---
CLASSIFICATION_DIR="$(cd "$NOBIAS_DIR/.." && pwd)"

# XGBoost model: prefer _v2_high_recall, fall back to the v1 in classification/
XGB_MODEL=""
for candidate in \
    "$NOBIAS_DIR/binary_interaction_classifier_xgb_v2_high_recall.json" \
    "$CLASSIFICATION_DIR/binary_interaction_classifier_xgb_v2_high_recall.json" \
    "$NOBIAS_DIR/binary_interaction_classifier_xgb.json" \
    "$CLASSIFICATION_DIR/binary_interaction_classifier_xgb.json"; do
    if [ -f "$candidate" ]; then
        XGB_MODEL="$candidate"
        break
    fi
done

# ResNet model: prefer _high_recall, fall back to stable
RESNET_MODEL=""
for candidate in \
    "$NOBIAS_DIR/resnet18_classifier_stable_high_recall.pth" \
    "$CLASSIFICATION_DIR/resnet18_classifier_stable_high_recall.pth" \
    "$NOBIAS_DIR/resnet18_classifier_stable.pth" \
    "$CLASSIFICATION_DIR/resnet18_classifier_stable.pth"; do
    if [ -f "$candidate" ]; then
        RESNET_MODEL="$candidate"
        break
    fi
done

echo "============================================="
echo "  NoBIAS Pipeline — CIF Test Runner"
echo "  Pipeline   : $PIPELINE"
echo "  CIF dir    : $CIF_DIR"
echo "  XGB model  : ${XGB_MODEL:-NOT FOUND}"
echo "  ResNet     : ${RESNET_MODEL:-NOT FOUND (ResNet stage will fail)}"
echo "============================================="
echo ""

if [ -z "$XGB_MODEL" ]; then
    echo "  ERROR: No XGBoost model file found. Run 'python download.py' in classification/ first."
    exit 1
fi

PASS=0
FAIL=0
SKIP=0

for cif in "$CIF_DIR"/*.cif; do
    name="$(basename "$cif")"

    if [ ! -f "$cif" ]; then
        echo "  [SKIP] $name — file not found"
        (( SKIP++ )) || true
        continue
    fi

    echo "-------------------------------------------"
    echo "  Running: $name"
    echo "-------------------------------------------"

    EXTRA_ARGS="--xgb-model $XGB_MODEL"
    if [ -n "$RESNET_MODEL" ]; then
        EXTRA_ARGS="$EXTRA_ARGS --resnet-model $RESNET_MODEL"
    fi

    if python "$PIPELINE" "$cif" $EXTRA_ARGS; then
        echo "  [PASS] $name"
        (( PASS++ )) || true
    else
        echo "  [FAIL] $name — pipeline exited with error"
        (( FAIL++ )) || true
    fi

    echo ""
done

echo "============================================="
echo "  Results: $PASS passed  |  $FAIL failed  |  $SKIP skipped"
echo "============================================="

[ "$FAIL" -eq 0 ]
