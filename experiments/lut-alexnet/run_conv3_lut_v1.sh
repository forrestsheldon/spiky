#!/usr/bin/env bash
# First LUT head (agreed 2026-10-20): CIFAR-10, conv3 tap of the BN AlexNet (1536-d), heads with W=1024.
# LUT trunk = ProjectionMHL(ConfidenceLUT read_top_n=2, 8 heads x 64 tables x 2**8 cells, d=48), library defaults
# (no cell TV, no table dropout, zero-init decompress), weight decay 0.05 as for the dense heads (tables included).
# Dense W=1024 and W=4096 rerun on the same widened lr grid so both sides get the same tuning; 3 seeds each.
# Launch detached from the repo root:
#   nohup caffeinate -i bash experiments/lut-alexnet/run_conv3_lut_v1.sh > experiments/lut-alexnet/stdout.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
COMMON=(--exp exp_conv3_dense_vs_lut_v1 --backbone alexnet_cifar_bn1 --datasets cifar10 --inputs conv3
        --lrs 1e-4 3e-4 1e-3 3e-3 1e-2 --seeds 0 1 2)

$PY run_heads.py "${COMMON[@]}" --trunk lut --widths 1024 --device cpu   # the reference cartridge: CPU beats MPS
$PY run_heads.py "${COMMON[@]}" --trunk mlp --widths 1024 4096
$PY make_results.py
