"""Stage 1: train each backbone once per dataset (end to end with a dense W=4096 head), then cache its taps.

    python experiments/lut-alexnet/run_backbones.py --exp exp_backbones_v1 --datasets mnist fmnist cifar10 --bn 1
    python experiments/lut-alexnet/run_backbones.py --smoke     # MNIST subset, 1 epoch, under _cache/smoke/

Checkpoints go to _cache/backbones/, features to _cache/features/ (both gitignored); the experiment folder
gets config.json, metrics.csv (one row per epoch) and summary.json (one entry per run). Rerunning skips
finished runs and only (re)computes a missing or stale feature cache.
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import data as D  # noqa: E402
from harness.backbones import build_backbone  # noqa: E402
from harness.features import cached_features, extract_all, file_sha1  # noqa: E402
from harness.heads import Classifier, build_head  # noqa: E402
from harness.runlog import ExpDir, git_state  # noqa: E402
from harness.train import OptimConfig, evaluate, fit, n_params, pick_device, seed_everything  # noqa: E402

HERE = Path(__file__).resolve().parent
EPOCHS = {"mnist": 10, "fmnist": 30, "cifar10": 30}
TAPS = ("conv3", "conv4", "final")
BACKBONE = "alexnet_cifar"
HEAD = dict(kind="mlp", width=4096, dropout=0.5)

METRICS_LEGEND = {
    "epoch": "1-based epoch",
    "lr_end": "one-cycle learning rate after the epoch's last step",
    "train_loss": "mean cross-entropy over the epoch's training batches (augmented, dropout on), nats",
    "train_acc": "fraction of training batches' images classified correctly during the epoch (augmented, dropout on)",
    "val_loss / val_acc": "cross-entropy (nats) / accuracy on the 5k held-out val split, eval mode, no augmentation",
    "epoch_s": "wall-clock seconds for the epoch including the val pass",
    "summary final_{train,val,test}_acc": "accuracy of the end-to-end model after the last epoch, eval mode",
}


def subset(data, n):
    for s, k in (("train", n), ("val", n // 4), ("test", n // 4)):
        sp = getattr(data, s)
        setattr(data, s, D.Split(sp.x[:k], sp.y[:k]))
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="exp_backbones_v1")
    ap.add_argument("--datasets", nargs="+", default=list(D.DATASETS))
    ap.add_argument("--bn", nargs="+", type=int, default=[1], help="BatchNorm flags to train (1, 0 or both)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tap-pool", type=int, default=2)
    ap.add_argument("--epochs", type=int, help="override the per-dataset epoch count")
    ap.add_argument("--no-features", action="store_true", help="train only; do not cache features")
    ap.add_argument("--device")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    root = D.CACHE_ROOT
    if args.smoke:
        args.datasets, args.epochs = ["mnist"], 1
        root = D.CACHE_ROOT / "smoke"
        exp_path = root / args.exp
    else:
        exp_path = HERE / args.exp
    device = pick_device(args.device)
    config = {
        "stage": "backbones", "backbone": BACKBONE, "taps": TAPS, "tap_pool": args.tap_pool, "head": HEAD,
        "datasets": args.datasets, "batchnorm": args.bn, "seed": args.seed,
        "optim": {d: OptimConfig(epochs=args.epochs or EPOCHS[d], lr=args.lr).to_dict() for d in args.datasets},
        "split": {"n_val": D.N_VAL, "split_seed": D.SPLIT_SEED}, "augment": {d: D.DATASETS[d]["augment"]
                                                                             for d in args.datasets},
        "device": str(device), "torch": torch.__version__, "git": git_state(), "smoke": args.smoke,
        "metrics_legend": METRICS_LEGEND,
    }
    exp = ExpDir(exp_path, config)

    for ds in args.datasets:
        data = D.load(ds)
        if args.smoke:
            data = subset(data, 2048)
        prepare = D.image_prepare(data, device)
        for bn in args.bn:
            run_id = f"{ds}__{BACKBONE}_bn{bn}"
            ckpt = root / "backbones" / f"{run_id}.pt"
            cfg = OptimConfig(epochs=args.epochs or EPOCHS[ds], lr=args.lr)
            seed_everything(args.seed)
            backbone = build_backbone(BACKBONE, data.channels, batchnorm=bool(bn), tap_pool=args.tap_pool)
            model = Classifier(backbone, build_head(HEAD["kind"], backbone.out_dim, HEAD["width"],
                                                    data.n_classes, dropout=HEAD["dropout"])).to(device)
            if exp.done(run_id) and ckpt.exists():
                print(f"[{run_id}] done; loading checkpoint")
                model.load_state_dict(torch.load(ckpt, weights_only=True))
            else:
                print(f"[{run_id}] training {cfg.epochs} epochs on {device}, {n_params(model) / 1e6:.1f}M params")
                t0 = time.time()
                log = exp.logger(run_id)

                def show(row, log=log):
                    log(row)
                    print(f"  ep {row['epoch']:3d}  loss {row['train_loss']:.4f}  train {row['train_acc']:.4f}  "
                          f"val {row['val_acc']:.4f}  {row['epoch_s']:.1f}s", flush=True)

                fit(model, (data.train.x, data.train.y), prepare, cfg, args.seed,
                    val=(data.val.x, data.val.y), log=show)
                ckpt.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), ckpt)
                res = {s: evaluate(model, getattr(data, s).x, getattr(data, s).y, prepare)["acc"]
                       for s in ("train", "val", "test")}
                exp.record(run_id, {
                    "dataset": ds, "batchnorm": bool(bn), "seed": args.seed, "optim": cfg.to_dict(),
                    "params_backbone": n_params(backbone), "params_head": n_params(model.head),
                    **{f"final_{s}_acc": a for s, a in res.items()},
                    "train_s": time.time() - t0, "checkpoint": str(ckpt.relative_to(D.CACHE_ROOT)),
                    "checkpoint_sha1": file_sha1(ckpt),
                })
                print(f"[{run_id}] train {res['train']:.4f}  val {res['val']:.4f}  test {res['test']:.4f}")
            if args.no_features:
                continue
            meta = {"run_id": run_id, "checkpoint_sha1": file_sha1(ckpt), "taps": list(TAPS),
                    "tap_pool": args.tap_pool, "n_val": D.N_VAL, "split_seed": D.SPLIT_SEED, "dtype": "float16",
                    "n": {s: len(getattr(data, s).y) for s in ("train", "val", "test")}}
            path = root / "features" / f"{run_id}.pt"
            feats = cached_features(path, meta, lambda: extract_all(model.backbone, data, prepare, TAPS))
            print(f"[{run_id}] features {path.name}: " +
                  ", ".join(f"{t} {tuple(feats['train'][t].shape)}" for t in TAPS))


if __name__ == "__main__":
    main()
