#!/usr/bin/env bash
# Compute/bandwidth-matched LUT (agreed 2026-10-21): CIFAR-10 conv3, LUT W=1024 with h=16 heads of d=64, so the
# compress (1536 -> 1024) and decompress (1024 -> 1024) projections have exactly the shapes of dense W=1024's fc1
# and fc2: same multiply-adds per input (2.63M), +5% values read for the table rows (2.76M vs 2.63M), 19.4M params.
# Pairs with the dense runs in exp_conv3_dense_vs_lut_v1; same protocol as run_conv3_lut_v1.sh.
# The parameter-matched LUT (tph=16) is in run_lut_round2.sh; this waits for that script so the CPU is free.
# Launch detached from the repo root:
#   nohup caffeinate -i bash experiments/lut-alexnet/run_lut_matched_v1.sh > experiments/lut-alexnet/stdout_matched.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
while pgrep -f "run_lut_round2.sh" > /dev/null; do sleep 30; done

$PY run_heads.py --backbone alexnet_cifar_bn1 --datasets cifar10 --inputs conv3 --lrs 1e-4 3e-4 1e-3 3e-3 1e-2 \
    --seeds 0 1 2 --exp exp_conv3_dense_vs_lut_v1 --trunk lut --widths 1024 --device cpu --lut-h 16 --lut-d 64
$PY make_results.py
