# lut-alexnet: bootstrap harness

**Data → dense backbone → cached features → dense (later LUT) head**, with seeds and a noise floor.
Design agreed 2026-10-20 (deepspike `projects/alexnet.md`). Backbones and head trunks are pluggable;
AlexNet is the first backbone and the plain MLP the first trunk. The LUT trunk (`ProjectionMHL`) lands
2026-10-21 in the slot described in `harness/heads.py`.

## Layout

| path | what |
|---|---|
| `harness/data.py` | MNIST / Fashion-MNIST (zero-padded 28→32, 1 ch) / CIFAR-10 as uint8 tensors; fixed 5k val split; on-device crop+flip (F-MNIST, CIFAR-10 only) |
| `harness/backbones.py` | `AlexNetCIFAR` (3×3 convs 64/192/384/256/256, pools after conv1/2/5 → 4096-d; `batchnorm` flag; taps conv3/conv4 avg-pooled to 2×2) and the `BACKBONES` registry |
| `harness/heads.py` | `Head` = pluggable trunk (`"mlp"`: Linear–ReLU–Dropout ×2, width W) + linear readout; `TRUNKS` registry |
| `harness/train.py` | the one training loop (AdamW + one-cycle, seeded shuffles and augmentation) |
| `harness/features.py` | feature extraction + on-disk cache validated by metadata (checkpoint sha1, taps, split) |
| `harness/runlog.py` | experiment folders: `config.json`, append-only `metrics.csv`, resumable `summary.json`; mean ± SE |
| `run_backbones.py` | stage 1: train a backbone per dataset end to end (dense W=4096 head), cache its taps |
| `run_heads.py` | stage 2: head-only sweeps on cached features; input `pixels` is the raw-pixel MLP reference |
| `make_results.py` | `results.md`: backbone table + head table (lr picked on val, test mean ± SE) |
| `tests/` | shapes, feature caching, fixed-seed determinism (CPU, synthetic data) |
| `exp_*/` | one folder per experiment (committed); `_cache/` holds datasets, checkpoints and features (gitignored) |

## Running

Uses the spiky venv (`torch`, `torchvision`; `spiky` itself only from the LUT trunk on). Data download to
`_cache/data` on first use; override the cache root with `LUT_ALEXNET_CACHE`.

```bash
pytest experiments/lut-alexnet/tests
python experiments/lut-alexnet/run_backbones.py --smoke && python experiments/lut-alexnet/run_heads.py --smoke
python experiments/lut-alexnet/run_backbones.py --exp exp_backbones_v1 --bn 1
python experiments/lut-alexnet/run_heads.py --exp exp_dense_heads_v1 --backbone alexnet_cifar_bn1
python experiments/lut-alexnet/run_backbones.py --exp exp_backbones_v1 --bn 0 --no-features
python experiments/lut-alexnet/make_results.py
```

Both stages skip runs already in their `summary.json`, so an interrupted sweep is resumed by rerunning it.

## Protocol and its caveats

- **Splits.** 5k images of each official train set (fixed permutation, seed 0) are held out *before*
  backbone training. Head lrs are swept on seed 0 and the lr with the best val accuracy is then run on
  seeds 1 and 2 (`--full-grid`: every seed at every lr, picked by mean val). Test is computed once per
  run, at the end, and never used for selection.
- **Budget (M1, MPS).** Backbone ≈ 63 s/epoch on 55k images: 10 epochs MNIST, 30 F-MNIST and CIFAR-10.
  Heads: 10 epochs, ≈ 11–15 s/epoch at W=4096 and 1–3 s at W=1024.
- **Backbone.** One seed per (dataset, BatchNorm). Trained end to end with the dense W=4096 head, so the
  cached features were shaped for that head. Head-only results inherit this (it favours dense heads) and
  carry no backbone-seed variance: their SE is a *head* noise floor, not the whole pipeline's.
- **Head inputs.** Features are of un-augmented images, stored float16. The backbone has seen (and
  largely fit) the train images, so train-feature accuracy is near 100% and head differences show up on
  val/test. `conv3`/`conv4` taps are earlier, pooled, less separable inputs where the head must do more.
- **Determinism.** Same seed + same device + same inputs → bitwise-identical training on CPU (tested).
  A spot check on MPS (M1, torch 2.14.1; head and BN backbone) was bitwise identical too, but that is not
  guaranteed by PyTorch; across devices expect agreement within noise only.
- **Not wired yet:** W&B logging (not installed in this env); the metric legend that would go into the
  W&B notes is in each experiment's `config.json` (`metrics_legend`).
