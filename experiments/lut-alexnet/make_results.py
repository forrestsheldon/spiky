"""Writes results.md: end-to-end backbone accuracies and head-only accuracies, mean +- SE over seeds.

    python experiments/lut-alexnet/make_results.py --backbones exp_backbones_v1 --heads exp_dense_heads_v1

For each (dataset, input, trunk, width) the learning rate is the one run_heads selected on val (its
summary's "selected"), or else the one with the best mean val accuracy; the test column is that lr's test
accuracy. The full lr sweep follows in a second table.
"""
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness.runlog import mean_se  # noqa: E402

HERE = Path(__file__).resolve().parent
ORDER = {"mnist": 0, "fmnist": 1, "cifar10": 2, "pixels": 0, "conv3": 1, "conv4": 2, "final": 3}


def pct(m, se=None):
    s = f"{100 * m:.2f}"
    if se is None:
        return s
    return f"{s} ± —" if math.isnan(se) else f"{s} ± {100 * se:.2f}"


def head_groups(runs):
    groups = defaultdict(list)
    for r in runs.values():
        groups[(r["dataset"], r["input"], r["trunk"], r["width"], r["lr"])].append(r)
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", default="exp_backbones_v1")
    ap.add_argument("--heads", nargs="+", default=["exp_dense_heads_v1"])
    ap.add_argument("--out", default="results.md")
    args = ap.parse_args()

    lines = ["# lut-alexnet bootstrap results", "",
             "Accuracies in %. Head rows: mean ± SE over seeds (n in the last column); lr chosen per row on val",
             "accuracy (each section states its rule), test reported for that lr only. Val = 5k images held out",
             "of train before backbone training. Backbones are a single seed, so their rows carry no SE.", ""]

    bb_path = HERE / args.backbones / "summary.json"
    if bb_path.exists():
        runs = json.loads(bb_path.read_text())["runs"]
        lines += ["## Backbones, end to end (dense W=4096 head)", "",
                  "| dataset | BatchNorm | epochs | train | val | test | params (backbone + head) | time |",
                  "|---|---|---|---|---|---|---|---|"]
        for rid, r in sorted(runs.items(), key=lambda kv: (ORDER[kv[1]["dataset"]], not kv[1]["batchnorm"])):
            lines.append(f"| {r['dataset']} | {'yes' if r['batchnorm'] else 'no'} | {r['optim']['epochs']} | "
                         f"{pct(r['final_train_acc'])} | {pct(r['final_val_acc'])} | {pct(r['final_test_acc'])} | "
                         f"{r['params_backbone'] / 1e6:.2f}M + {r['params_head'] / 1e6:.2f}M | "
                         f"{r['train_s'] / 60:.0f} min |")
        lines.append("")

    for heads in args.heads:
        path = HERE / heads / "summary.json"
        if not path.exists():
            continue
        cfg = json.loads((HERE / heads / "config.json").read_text())["invocations"][-1]
        summary = json.loads(path.read_text())
        groups = head_groups(summary["runs"])
        best = {}
        for (ds, inp, trunk, w, lr), rs in groups.items():
            key = (ds, inp, trunk, w)
            selected = summary.get("selected", {}).get(f"{ds}__{inp}__{trunk}_W{w}")
            val = mean_se([r["final_val_acc"] for r in rs])[0] if selected is None else float(lr == selected)
            if key not in best or val > best[key][0]:
                best[key] = (val, lr)
        lines += [f"## Heads on cached features: `{heads}`", "",
                  f"Features from `{cfg['backbone_features']}`; {cfg['optim']['epochs']} epochs, "
                  f"dropout {cfg['dropout']}. `pixels` = raw-pixel MLP reference (no backbone).",
                  f"Selection: {cfg['selection']}.", "",
                  "| dataset | input | in dim | trunk | W | lr | train | val | test | params | n |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for key in sorted(best, key=lambda k: (ORDER[k[0]], ORDER[k[1]], k[2], -k[3])):
            rs = groups[(*key, best[key][1])]
            tr, va, te = (mean_se([r[f"final_{s}_acc"] for r in rs]) for s in ("train", "val", "test"))
            lines.append(f"| {key[0]} | {key[1]} | {rs[0]['in_dim']} | {key[2]} | {key[3]} | {best[key][1]:g} | "
                         f"{pct(tr[0])} | {pct(*va)} | {pct(*te)} | {rs[0]['params'] / 1e6:.2f}M | {len(rs)} |")
        lines += ["", "<details><summary>Full lr sweep</summary>", "",
                  "| dataset | input | trunk | W | lr | val | test | n |", "|---|---|---|---|---|---|---|---|"]
        for (ds, inp, trunk, w, lr) in sorted(groups, key=lambda k: (ORDER[k[0]], ORDER[k[1]], k[2], -k[3], k[4])):
            rs = groups[(ds, inp, trunk, w, lr)]
            va, te = (mean_se([r[f"final_{s}_acc"] for r in rs]) for s in ("val", "test"))
            lines.append(f"| {ds} | {inp} | {trunk} | {w} | {lr:g} | {pct(*va)} | {pct(*te)} | {len(rs)} |")
        lines += ["", "</details>", ""]

    (HERE / args.out).write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
