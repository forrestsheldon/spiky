# lut-alexnet bootstrap results

Accuracies in %. Head rows: mean ± SE over seeds (n in the last column); lr chosen per row on val
accuracy (each section states its rule), test reported for that lr only. Val = 5k images held out
of train before backbone training. Backbones are a single seed, so their rows carry no SE.

## Backbones, end to end (dense W=4096 head)

| dataset | BatchNorm | epochs | train | val | test | params (backbone + head) | time |
|---|---|---|---|---|---|---|---|
| mnist | yes | 10 | 99.96 | 99.50 | 99.63 | 2.25M + 33.60M | 12 min |
| mnist | no | 10 | 99.99 | 99.32 | 99.54 | 2.25M + 33.60M | 12 min |
| fmnist | yes | 30 | 97.43 | 94.56 | 94.29 | 2.25M + 33.60M | 62 min |
| fmnist | no | 30 | 97.51 | 94.46 | 93.95 | 2.25M + 33.60M | 35 min |
| cifar10 | yes | 30 | 97.73 | 89.68 | 89.89 | 2.25M + 33.60M | 27 min |
| cifar10 | no | 30 | 98.75 | 87.72 | 87.99 | 2.25M + 33.60M | 28 min |

## Heads on cached features: `exp_dense_heads_v1`

Features from `alexnet_cifar_bn1`; 10 epochs. Trunks: `mlp` (dropout 0.5). `pixels` = raw-pixel MLP reference (no backbone). MACs = multiply-adds per input in the head's linear layers (a LUT trunk also adds table rows, no multiplies).
Selection: lr per (dataset, input, trunk, width) by final val_acc of seed 0; the other seeds run at that lr only; test reported, never used for selection.

| dataset | input | in dim | trunk | W | lr | train | val | test | params | MACs | n |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mnist | pixels | 1024 | mlp | 4096 | 0.001 | 99.83 | 98.41 ± 0.07 | 98.66 ± 0.05 | 21.02M | 21.01M | 3 |
| mnist | pixels | 1024 | mlp | 1024 | 0.001 | 99.63 | 98.23 ± 0.07 | 98.55 ± 0.03 | 2.11M | 2.11M | 3 |
| mnist | conv3 | 1536 | mlp | 4096 | 0.0003 | 99.72 | 99.33 ± 0.01 | 99.26 ± 0.01 | 23.12M | 23.11M | 3 |
| mnist | conv3 | 1536 | mlp | 1024 | 0.001 | 99.67 | 99.27 ± 0.01 | 99.26 ± 0.01 | 2.63M | 2.63M | 3 |
| mnist | conv4 | 1024 | mlp | 4096 | 0.0003 | 99.91 | 99.44 ± 0.00 | 99.50 ± 0.00 | 21.02M | 21.01M | 3 |
| mnist | conv4 | 1024 | mlp | 1024 | 0.001 | 99.89 | 99.42 ± 0.01 | 99.49 ± 0.02 | 2.11M | 2.11M | 3 |
| mnist | final | 4096 | mlp | 4096 | 0.001 | 99.97 | 99.41 ± 0.01 | 99.54 ± 0.02 | 33.60M | 33.60M | 3 |
| mnist | final | 4096 | mlp | 1024 | 0.0003 | 99.98 | 99.42 ± 0.02 | 99.58 ± 0.01 | 5.26M | 5.25M | 3 |
| fmnist | pixels | 1024 | mlp | 4096 | 0.0003 | 92.21 | 89.66 ± 0.06 | 89.20 ± 0.03 | 21.02M | 21.01M | 3 |
| fmnist | pixels | 1024 | mlp | 1024 | 0.001 | 91.05 | 89.37 ± 0.04 | 88.75 ± 0.04 | 2.11M | 2.11M | 3 |
| fmnist | conv3 | 1536 | mlp | 4096 | 0.0003 | 94.59 | 93.59 ± 0.04 | 92.91 ± 0.01 | 23.12M | 23.11M | 3 |
| fmnist | conv3 | 1536 | mlp | 1024 | 0.001 | 94.39 | 93.44 ± 0.05 | 92.82 ± 0.03 | 2.63M | 2.63M | 3 |
| fmnist | conv4 | 1024 | mlp | 4096 | 0.0003 | 96.35 | 94.13 ± 0.03 | 93.58 ± 0.02 | 21.02M | 21.01M | 3 |
| fmnist | conv4 | 1024 | mlp | 1024 | 0.0003 | 95.84 | 94.27 ± 0.02 | 93.42 ± 0.00 | 2.11M | 2.11M | 3 |
| fmnist | final | 4096 | mlp | 4096 | 0.003 | 98.84 | 94.47 ± 0.01 | 94.00 ± 0.01 | 33.60M | 33.60M | 3 |
| fmnist | final | 4096 | mlp | 1024 | 0.0003 | 98.37 | 94.39 ± 0.04 | 93.98 ± 0.02 | 5.26M | 5.25M | 3 |
| cifar10 | pixels | 3072 | mlp | 4096 | 0.0003 | 68.72 | 55.38 ± 0.16 | 56.22 ± 0.16 | 29.41M | 29.40M | 3 |
| cifar10 | pixels | 3072 | mlp | 1024 | 0.001 | 61.48 | 52.87 ± 0.33 | 54.33 ± 0.06 | 4.21M | 4.20M | 3 |
| cifar10 | conv3 | 1536 | mlp | 4096 | 0.0003 | 89.54 | 83.77 ± 0.06 | 83.12 ± 0.11 | 23.12M | 23.11M | 3 |
| cifar10 | conv3 | 1536 | mlp | 1024 | 0.0003 | 86.88 | 83.05 ± 0.07 | 81.67 ± 0.07 | 2.63M | 2.63M | 3 |
| cifar10 | conv4 | 1024 | mlp | 4096 | 0.001 | 95.50 | 87.35 ± 0.04 | 86.91 ± 0.08 | 21.02M | 21.01M | 3 |
| cifar10 | conv4 | 1024 | mlp | 1024 | 0.001 | 94.85 | 86.95 ± 0.04 | 86.72 ± 0.02 | 2.11M | 2.11M | 3 |
| cifar10 | final | 4096 | mlp | 4096 | 0.001 | 99.85 | 89.27 ± 0.09 | 89.65 ± 0.06 | 33.60M | 33.60M | 3 |
| cifar10 | final | 4096 | mlp | 1024 | 0.003 | 99.12 | 89.40 ± 0.09 | 89.49 ± 0.05 | 5.26M | 5.25M | 3 |

<details><summary>Full lr sweep</summary>

| dataset | input | trunk | W | lr | val | test | n |
|---|---|---|---|---|---|---|---|
| mnist | pixels | mlp | 4096 | 0.0003 | 98.32 ± — | 98.48 ± — | 1 |
| mnist | pixels | mlp | 4096 | 0.001 | 98.41 ± 0.07 | 98.66 ± 0.05 | 3 |
| mnist | pixels | mlp | 4096 | 0.003 | 97.84 ± — | 97.97 ± — | 1 |
| mnist | pixels | mlp | 1024 | 0.0003 | 97.90 ± — | 98.10 ± — | 1 |
| mnist | pixels | mlp | 1024 | 0.001 | 98.23 ± 0.07 | 98.55 ± 0.03 | 3 |
| mnist | pixels | mlp | 1024 | 0.003 | 98.24 ± — | 98.30 ± — | 1 |
| mnist | conv3 | mlp | 4096 | 0.0003 | 99.33 ± 0.01 | 99.26 ± 0.01 | 3 |
| mnist | conv3 | mlp | 4096 | 0.001 | 99.26 ± — | 99.32 ± — | 1 |
| mnist | conv3 | mlp | 4096 | 0.003 | 99.00 ± — | 98.98 ± — | 1 |
| mnist | conv3 | mlp | 1024 | 0.0003 | 99.22 ± — | 99.15 ± — | 1 |
| mnist | conv3 | mlp | 1024 | 0.001 | 99.27 ± 0.01 | 99.26 ± 0.01 | 3 |
| mnist | conv3 | mlp | 1024 | 0.003 | 99.24 ± — | 99.13 ± — | 1 |
| mnist | conv4 | mlp | 4096 | 0.0003 | 99.44 ± 0.00 | 99.50 ± 0.00 | 3 |
| mnist | conv4 | mlp | 4096 | 0.001 | 99.40 ± — | 99.51 ± — | 1 |
| mnist | conv4 | mlp | 4096 | 0.003 | 99.42 ± — | 99.38 ± — | 1 |
| mnist | conv4 | mlp | 1024 | 0.0003 | 99.38 ± — | 99.43 ± — | 1 |
| mnist | conv4 | mlp | 1024 | 0.001 | 99.42 ± 0.01 | 99.49 ± 0.02 | 3 |
| mnist | conv4 | mlp | 1024 | 0.003 | 99.36 ± — | 99.39 ± — | 1 |
| mnist | final | mlp | 4096 | 0.0003 | 99.34 ± — | 99.56 ± — | 1 |
| mnist | final | mlp | 4096 | 0.001 | 99.41 ± 0.01 | 99.54 ± 0.02 | 3 |
| mnist | final | mlp | 4096 | 0.003 | 99.36 ± — | 99.56 ± — | 1 |
| mnist | final | mlp | 1024 | 0.0003 | 99.42 ± 0.02 | 99.58 ± 0.01 | 3 |
| mnist | final | mlp | 1024 | 0.001 | 99.34 ± — | 99.57 ± — | 1 |
| mnist | final | mlp | 1024 | 0.003 | 99.38 ± — | 99.45 ± — | 1 |
| fmnist | pixels | mlp | 4096 | 0.0003 | 89.66 ± 0.06 | 89.20 ± 0.03 | 3 |
| fmnist | pixels | mlp | 4096 | 0.001 | 89.38 ± — | 88.89 ± — | 1 |
| fmnist | pixels | mlp | 4096 | 0.003 | 87.64 ± — | 86.87 ± — | 1 |
| fmnist | pixels | mlp | 1024 | 0.0003 | 88.66 ± — | 88.22 ± — | 1 |
| fmnist | pixels | mlp | 1024 | 0.001 | 89.37 ± 0.04 | 88.75 ± 0.04 | 3 |
| fmnist | pixels | mlp | 1024 | 0.003 | 88.54 ± — | 87.82 ± — | 1 |
| fmnist | conv3 | mlp | 4096 | 0.0003 | 93.59 ± 0.04 | 92.91 ± 0.01 | 3 |
| fmnist | conv3 | mlp | 4096 | 0.001 | 93.46 ± — | 92.86 ± — | 1 |
| fmnist | conv3 | mlp | 4096 | 0.003 | 92.20 ± — | 91.49 ± — | 1 |
| fmnist | conv3 | mlp | 1024 | 0.0003 | 93.34 ± — | 92.67 ± — | 1 |
| fmnist | conv3 | mlp | 1024 | 0.001 | 93.44 ± 0.05 | 92.82 ± 0.03 | 3 |
| fmnist | conv3 | mlp | 1024 | 0.003 | 92.98 ± — | 92.22 ± — | 1 |
| fmnist | conv4 | mlp | 4096 | 0.0003 | 94.13 ± 0.03 | 93.58 ± 0.02 | 3 |
| fmnist | conv4 | mlp | 4096 | 0.001 | 94.06 ± — | 93.57 ± — | 1 |
| fmnist | conv4 | mlp | 4096 | 0.003 | 94.08 ± — | 93.26 ± — | 1 |
| fmnist | conv4 | mlp | 1024 | 0.0003 | 94.27 ± 0.02 | 93.42 ± 0.00 | 3 |
| fmnist | conv4 | mlp | 1024 | 0.001 | 94.26 ± — | 93.52 ± — | 1 |
| fmnist | conv4 | mlp | 1024 | 0.003 | 94.12 ± — | 93.47 ± — | 1 |
| fmnist | final | mlp | 4096 | 0.0003 | 94.30 ± — | 93.89 ± — | 1 |
| fmnist | final | mlp | 4096 | 0.001 | 94.26 ± — | 94.01 ± — | 1 |
| fmnist | final | mlp | 4096 | 0.003 | 94.47 ± 0.01 | 94.00 ± 0.01 | 3 |
| fmnist | final | mlp | 1024 | 0.0003 | 94.39 ± 0.04 | 93.98 ± 0.02 | 3 |
| fmnist | final | mlp | 1024 | 0.001 | 94.22 ± — | 93.92 ± — | 1 |
| fmnist | final | mlp | 1024 | 0.003 | 94.30 ± — | 94.08 ± — | 1 |
| cifar10 | pixels | mlp | 4096 | 0.0003 | 55.38 ± 0.16 | 56.22 ± 0.16 | 3 |
| cifar10 | pixels | mlp | 4096 | 0.001 | 53.04 ± — | 54.27 ± — | 1 |
| cifar10 | pixels | mlp | 4096 | 0.003 | 41.30 ± — | 42.46 ± — | 1 |
| cifar10 | pixels | mlp | 1024 | 0.0003 | 52.76 ± — | 54.30 ± — | 1 |
| cifar10 | pixels | mlp | 1024 | 0.001 | 52.87 ± 0.33 | 54.33 ± 0.06 | 3 |
| cifar10 | pixels | mlp | 1024 | 0.003 | 46.48 ± — | 47.50 ± — | 1 |
| cifar10 | conv3 | mlp | 4096 | 0.0003 | 83.77 ± 0.06 | 83.12 ± 0.11 | 3 |
| cifar10 | conv3 | mlp | 4096 | 0.001 | 83.58 ± — | 82.73 ± — | 1 |
| cifar10 | conv3 | mlp | 4096 | 0.003 | 81.20 ± — | 80.32 ± — | 1 |
| cifar10 | conv3 | mlp | 1024 | 0.0003 | 83.05 ± 0.07 | 81.67 ± 0.07 | 3 |
| cifar10 | conv3 | mlp | 1024 | 0.001 | 82.96 ± — | 82.21 ± — | 1 |
| cifar10 | conv3 | mlp | 1024 | 0.003 | 81.72 ± — | 80.66 ± — | 1 |
| cifar10 | conv4 | mlp | 4096 | 0.0003 | 87.10 ± — | 86.85 ± — | 1 |
| cifar10 | conv4 | mlp | 4096 | 0.001 | 87.35 ± 0.04 | 86.91 ± 0.08 | 3 |
| cifar10 | conv4 | mlp | 4096 | 0.003 | 86.52 ± — | 86.33 ± — | 1 |
| cifar10 | conv4 | mlp | 1024 | 0.0003 | 86.76 ± — | 86.47 ± — | 1 |
| cifar10 | conv4 | mlp | 1024 | 0.001 | 86.95 ± 0.04 | 86.72 ± 0.02 | 3 |
| cifar10 | conv4 | mlp | 1024 | 0.003 | 86.74 ± — | 86.51 ± — | 1 |
| cifar10 | final | mlp | 4096 | 0.0003 | 89.26 ± — | 89.68 ± — | 1 |
| cifar10 | final | mlp | 4096 | 0.001 | 89.27 ± 0.09 | 89.65 ± 0.06 | 3 |
| cifar10 | final | mlp | 4096 | 0.003 | 89.24 ± — | 89.39 ± — | 1 |
| cifar10 | final | mlp | 1024 | 0.0003 | 89.32 ± — | 89.56 ± — | 1 |
| cifar10 | final | mlp | 1024 | 0.001 | 89.40 ± — | 89.81 ± — | 1 |
| cifar10 | final | mlp | 1024 | 0.003 | 89.40 ± 0.09 | 89.49 ± 0.05 | 3 |

</details>

## Heads on cached features: `exp_conv3_dense_vs_lut_v1`

Features from `alexnet_cifar_bn1`; 10 epochs. Trunks: `lut-ConfidenceLUT-n2-h8t64nap8d48`, `mlp` (dropout 0.5). `pixels` = raw-pixel MLP reference (no backbone). MACs = multiply-adds per input in the head's linear layers (a LUT trunk also adds table rows, no multiplies).
Selection: lr per (dataset, input, trunk, width) by final val_acc of seed 0; the other seeds run at that lr only; test reported, never used for selection.

| dataset | input | in dim | trunk | W | lr | train | val | test | params | MACs | n |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cifar10 | conv3 | 1536 | lut-ConfidenceLUT-n2-h8t64nap8d48 | 1024 | 0.001 | 93.14 | 83.81 ± 0.16 | 83.51 ± 0.07 | 7.29M | 0.99M | 3 |
| cifar10 | conv3 | 1536 | mlp | 4096 | 0.0003 | 89.54 | 83.77 ± 0.06 | 83.12 ± 0.11 | 23.12M | 23.11M | 3 |
| cifar10 | conv3 | 1536 | mlp | 1024 | 0.0003 | 86.88 | 83.05 ± 0.07 | 81.67 ± 0.07 | 2.63M | 2.63M | 3 |

Paired on the same test images (selected lr each side). Δ = test acc A − B, mean ± SE over seed pairs; per seed: b/c = images only A / only B gets right, p = McNemar.

| dataset | input | A | B | Δ test | per seed b/c, p |
|---|---|---|---|---|---|
| cifar10 | conv3 | lut-ConfidenceLUT-n2-h8t64nap8d48 W=1024 | mlp W=4096 | +0.39 ± 0.04 | 295/253, p=0.08; 276/245, p=0.19; 274/231, p=0.062 |
| cifar10 | conv3 | lut-ConfidenceLUT-n2-h8t64nap8d48 W=1024 | mlp W=1024 | +1.84 ± 0.09 | 452/272, p=2.9e-11; 464/262, p=8.7e-14; 432/262, p=1.4e-10 |
| cifar10 | conv3 | mlp W=4096 | mlp W=1024 | +1.45 ± 0.13 | 240/102, p=1.3e-13; 251/80, p=9.3e-21; 232/105, p=6.7e-12 |

<details><summary>Full lr sweep</summary>

| dataset | input | trunk | W | lr | val | test | n |
|---|---|---|---|---|---|---|---|
| cifar10 | conv3 | lut-ConfidenceLUT-n2-h8t64nap8d48 | 1024 | 0.0001 | 82.20 ± — | 81.53 ± — | 1 |
| cifar10 | conv3 | lut-ConfidenceLUT-n2-h8t64nap8d48 | 1024 | 0.0003 | 83.22 ± — | 82.90 ± — | 1 |
| cifar10 | conv3 | lut-ConfidenceLUT-n2-h8t64nap8d48 | 1024 | 0.001 | 83.81 ± 0.16 | 83.51 ± 0.07 | 3 |
| cifar10 | conv3 | lut-ConfidenceLUT-n2-h8t64nap8d48 | 1024 | 0.003 | 82.24 ± — | 81.13 ± — | 1 |
| cifar10 | conv3 | lut-ConfidenceLUT-n2-h8t64nap8d48 | 1024 | 0.01 | 80.36 ± — | 79.98 ± — | 1 |
| cifar10 | conv3 | mlp | 4096 | 0.0001 | 82.98 ± — | 81.75 ± — | 1 |
| cifar10 | conv3 | mlp | 4096 | 0.0003 | 83.77 ± 0.06 | 83.12 ± 0.11 | 3 |
| cifar10 | conv3 | mlp | 4096 | 0.001 | 83.58 ± — | 82.73 ± — | 1 |
| cifar10 | conv3 | mlp | 4096 | 0.003 | 81.20 ± — | 80.32 ± — | 1 |
| cifar10 | conv3 | mlp | 4096 | 0.01 | 25.80 ± — | 26.61 ± — | 1 |
| cifar10 | conv3 | mlp | 1024 | 0.0001 | 80.44 ± — | 79.17 ± — | 1 |
| cifar10 | conv3 | mlp | 1024 | 0.0003 | 83.05 ± 0.07 | 81.67 ± 0.07 | 3 |
| cifar10 | conv3 | mlp | 1024 | 0.001 | 82.96 ± — | 82.21 ± — | 1 |
| cifar10 | conv3 | mlp | 1024 | 0.003 | 81.72 ± — | 80.66 ± — | 1 |
| cifar10 | conv3 | mlp | 1024 | 0.01 | 61.50 ± — | 61.60 ± — | 1 |

</details>
