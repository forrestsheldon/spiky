"""Stage 2: head-only training on cached backbone features, plus the raw-pixel MLP reference (input "pixels").

    python experiments/lut-alexnet/run_heads.py --exp exp_dense_heads_v1 --backbone alexnet_cifar_bn1
    python experiments/lut-alexnet/run_heads.py --smoke     # MNIST smoke features, 1 epoch, under _cache/smoke/

Every (dataset, input, trunk, width, lr, seed) is one run in the experiment folder's summary.json; per-epoch
rows go to metrics.csv. By default the lr sweep uses the first seed only and the remaining seeds run at the lr
with the best val accuracy (recorded under "selected"); --full-grid runs every seed at every lr. Inputs are
fed as stored (no augmentation: the features are of un-augmented images).
"""
import argparse
import itertools
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import data as D  # noqa: E402
from harness.heads import build_head  # noqa: E402
from harness.runlog import ExpDir, git_state  # noqa: E402
from harness.train import (OptimConfig, cast_prepare, evaluate, fit, n_params, pick_device,  # noqa: E402
                           seed_everything)

HERE = Path(__file__).resolve().parent

METRICS_LEGEND = {
    "epoch": "1-based epoch",
    "lr_end": "one-cycle learning rate after the epoch's last step",
    "train_loss": "mean cross-entropy over the epoch's training batches (dropout on), nats",
    "train_acc": "fraction of training inputs classified correctly during the epoch (dropout on)",
    "val_loss / val_acc": "cross-entropy (nats) / accuracy on the 5k held-out val split, eval mode",
    "epoch_s": "wall-clock seconds for the epoch including the val pass",
    "summary final_{train,val,test}_acc": "accuracy of the head after the last epoch, eval mode; test is "
                                          "computed once, at the end, and never used for selection",
}


def load_inputs(ds, inp, backbone_tag, root):
    if inp == "pixels":
        data = D.load(ds)
        xs = D.pixel_features(data)
        return {s: {"x": xs[s], "y": getattr(data, s).y} for s in xs}, {"source": "pixels"}
    path = root / "features" / f"{ds}__{backbone_tag}.pt"
    blob = torch.load(path, weights_only=True)
    splits = {s: {"x": v[inp], "y": v["y"]} for s, v in blob["splits"].items()}
    return splits, {"source": str(path.relative_to(D.CACHE_ROOT)), "feature_meta": blob["meta"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="exp_dense_heads_v1")
    ap.add_argument("--backbone", default="alexnet_cifar_bn1", help="feature cache tag: <dataset>__<this>.pt")
    ap.add_argument("--datasets", nargs="+", default=list(D.DATASETS))
    ap.add_argument("--inputs", nargs="+", default=["pixels", "final", "conv4", "conv3"])
    ap.add_argument("--trunk", default="mlp")
    ap.add_argument("--widths", nargs="+", type=int, default=[4096, 1024])
    ap.add_argument("--lrs", nargs="+", type=float, default=[3e-4, 1e-3, 3e-3])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--full-grid", action="store_true", help="every seed at every lr (default: sweep lr on seed 0)")
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--device")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    root = D.CACHE_ROOT
    exp_path = HERE / args.exp
    if args.smoke:
        root = D.CACHE_ROOT / "smoke"
        exp_path = root / args.exp
        args.datasets, args.inputs, args.widths, args.lrs, args.seeds, args.epochs = (
            ["mnist"], ["final", "conv3"], [1024], [3e-4, 1e-3], [0, 1], 1)
    device = pick_device(args.device)
    config = {
        "stage": "heads", "backbone_features": args.backbone, "trunk": args.trunk, "dropout": args.dropout,
        "grid": {k: getattr(args, k) for k in ("datasets", "inputs", "widths", "lrs", "seeds")},
        "optim": OptimConfig(epochs=args.epochs, lr=float("nan")).to_dict() | {"lr": "grid"},
        "selection": ("lr per (dataset, input, trunk, width) by mean final val_acc over seeds" if args.full_grid else
                      f"lr per (dataset, input, trunk, width) by final val_acc of seed {args.seeds[0]}; the other "
                      "seeds run at that lr only") + "; test reported, never used for selection",
        "device": str(device), "torch": torch.__version__, "git": git_state(), "smoke": args.smoke,
        "metrics_legend": METRICS_LEGEND,
    }
    exp = ExpDir(exp_path, config)
    prepare = cast_prepare(device)

    for ds, inp in itertools.product(args.datasets, args.inputs):
        state = {}

        def run(width, lr, seed):
            run_id = f"{ds}__{inp}__{args.trunk}_W{width}__lr{lr:g}__s{seed}"
            if exp.done(run_id):
                return exp.summary["runs"][run_id]
            if not state:
                splits, source = load_inputs(ds, inp, args.backbone, root)
                state.update(splits=splits, source=source,
                             train=(splits["train"]["x"].to(device), splits["train"]["y"]),
                             val=(splits["val"]["x"].to(device), splits["val"]["y"]))
            train, val, splits = state["train"], state["val"], state["splits"]
            in_dim, n_classes = train[0].shape[1], D.DATASETS[ds]["n_classes"]
            cfg = OptimConfig(epochs=args.epochs, lr=lr)
            seed_everything(seed)
            head = build_head(args.trunk, in_dim, width, n_classes, dropout=args.dropout).to(device)
            t0 = time.time()
            hist = fit(head, train, prepare, cfg, seed, val=val, log=exp.logger(run_id))
            res = {
                "train": evaluate(head, *train, prepare)["acc"],
                "val": hist[-1]["val_acc"],
                "test": evaluate(head, splits["test"]["x"].to(device), splits["test"]["y"], prepare)["acc"],
            }
            result = {
                "dataset": ds, "input": inp, "in_dim": in_dim, "trunk": args.trunk, "width": width, "lr": lr,
                "seed": seed, "params": n_params(head), **{f"final_{s}_acc": a for s, a in res.items()},
                "train_s": time.time() - t0, **state["source"],
            }
            exp.record(run_id, result)
            print(f"[{run_id}] val {res['val']:.4f}  test {res['test']:.4f}  ({time.time() - t0:.0f}s)", flush=True)
            return result

        for width in args.widths:
            if args.full_grid:
                for lr, seed in itertools.product(args.lrs, args.seeds):
                    run(width, lr, seed)
                continue
            # lr sweep on the first seed; the remaining seeds only at the lr with the best val accuracy
            sweep = {lr: run(width, lr, args.seeds[0])["final_val_acc"] for lr in args.lrs}
            best = max(sweep, key=sweep.get)
            exp.select(f"{ds}__{inp}__{args.trunk}_W{width}", best)
            for seed in args.seeds[1:]:
                run(width, best, seed)

if __name__ == "__main__":
    main()
