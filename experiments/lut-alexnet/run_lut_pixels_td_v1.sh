#!/usr/bin/env bash
# Table dropout on the raw-pixel LUT (agreed 2026-10-21). The pixel LUT without regularisation fits train to
# 99.2% but tests at 50.2% (dense W=1024: 61.5% / 54.3%): it memorises. Same LUT (ConfidenceLUT n=2, 8x64 tables,
# 2**8 cells, d=48, W=1024) with whole-table dropout 0.2 (the lutorch_ex README's example) and 0.5 (the dense heads'
# dropout); same protocol (lr 1e-4..1e-2 on seed 0, seeds 1-2 at the val-selected lr, weight decay 0.05), CPU like
# the unregularised LUT. Pairs with the dense pixel runs in exp_pixels_dense_vs_lut_v1.
# Waits for run_lut_matched_v1.sh so the CPU is free. Launch detached from the repo root:
#   nohup caffeinate -i bash experiments/lut-alexnet/run_lut_pixels_td_v1.sh > experiments/lut-alexnet/stdout_td.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
while pgrep -f "run_lut_matched_v1.sh" > /dev/null; do sleep 30; done

for td in 0.2 0.5; do
  $PY run_heads.py --backbone alexnet_cifar_bn1 --datasets cifar10 --inputs pixels --lrs 1e-4 3e-4 1e-3 3e-3 1e-2 \
      --seeds 0 1 2 --exp exp_pixels_dense_vs_lut_v1 --trunk lut --widths 1024 --device cpu --table-dropout "$td"
done
$PY make_results.py
