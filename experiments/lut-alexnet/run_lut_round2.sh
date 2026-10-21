#!/usr/bin/env bash
# LUT round 2 (agreed 2026-10-21), all CIFAR-10, same protocol as run_conv3_lut_v1.sh (lr 1e-4..1e-2 swept on
# seed 0, seeds 1-2 at the val-selected lr, weight decay 0.05 everywhere, LUT = ConfidenceLUT n=2, h=8, nap=8, d=48):
#   1. raw pixels (3072-d): LUT W=1024 (tph=64) vs dense W=1024 / 4096            -> exp_pixels_dense_vs_lut_v1
#   2. conv3, parameter-matched LUT: tph=16 (2.57M params vs dense W=1024's 2.63M) -> exp_conv3_dense_vs_lut_v1
#      (pairs with the dense runs already there)
#   3. final (4096-d): LUT W=1024 (tph=64) vs dense W=1024 / 4096                  -> exp_final_dense_vs_lut_v1
# Two queues run at once, LUT heads on the CPU and dense heads on MPS; ExpDir's lock makes the shared folders
# safe. The queues compete for the CPU, so the recorded wall-clock times (train_s, epoch_s) are not clean timings.
# Launch detached from the repo root:
#   nohup caffeinate -i bash experiments/lut-alexnet/run_lut_round2.sh > experiments/lut-alexnet/stdout.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
COMMON=(--backbone alexnet_cifar_bn1 --datasets cifar10 --lrs 1e-4 3e-4 1e-3 3e-3 1e-2 --seeds 0 1 2)
LUT=(--trunk lut --widths 1024 --device cpu)
DENSE=(--trunk mlp --widths 1024 4096)

(
  $PY run_heads.py "${COMMON[@]}" "${LUT[@]}" --exp exp_pixels_dense_vs_lut_v1 --inputs pixels
  $PY run_heads.py "${COMMON[@]}" "${LUT[@]}" --exp exp_conv3_dense_vs_lut_v1 --inputs conv3 --lut-tph 16
  $PY run_heads.py "${COMMON[@]}" "${LUT[@]}" --exp exp_final_dense_vs_lut_v1 --inputs final
) &
cpu_queue=$!
(
  $PY run_heads.py "${COMMON[@]}" "${DENSE[@]}" --exp exp_final_dense_vs_lut_v1 --inputs final
  $PY run_heads.py "${COMMON[@]}" "${DENSE[@]}" --exp exp_pixels_dense_vs_lut_v1 --inputs pixels
) &
gpu_queue=$!
status=0
wait $cpu_queue || status=1
wait $gpu_queue || status=1
$PY make_results.py
exit $status
