#!/usr/bin/env bash
# The 2026-10-20 bootstrap runs, in order. Launch detached from the repo root:
#   nohup caffeinate -i bash experiments/lut-alexnet/run_all.sh > experiments/lut-alexnet/stdout.log 2>&1 &
# Every stage is resumable, so rerunning this after an interruption continues where it stopped.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
CIFAR_TAR=${LUT_ALEXNET_CACHE:-_cache}/data/cifar-10-python.tar.gz
CIFAR_MD5=c58f30108f718f92721af3b95e74349a

# 1. BN backbones + feature caches, then dense heads, for the 1-channel datasets
$PY run_backbones.py --exp exp_backbones_v1 --bn 1 --datasets mnist fmnist
$PY run_heads.py --exp exp_dense_heads_v1 --backbone alexnet_cifar_bn1 --datasets mnist fmnist
$PY make_results.py

# 2. CIFAR-10 (waits for a complete download if one is still in progress)
until [ -f "$CIFAR_TAR" ] && [ "$(md5 -q "$CIFAR_TAR" 2>/dev/null || md5sum "$CIFAR_TAR" | cut -d' ' -f1)" = "$CIFAR_MD5" ]; do
  echo "waiting for $CIFAR_TAR"; sleep 60
done
$PY run_backbones.py --exp exp_backbones_v1 --bn 1 --datasets cifar10
$PY run_heads.py --exp exp_dense_heads_v1 --backbone alexnet_cifar_bn1 --datasets cifar10
$PY make_results.py

# 3. plain (no BatchNorm) backbones: end-to-end baseline only, no feature cache
$PY run_backbones.py --exp exp_backbones_v1 --bn 0 --no-features
$PY make_results.py
