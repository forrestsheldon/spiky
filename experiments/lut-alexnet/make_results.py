"""Writes results.md: end-to-end backbone accuracies and head-only accuracies, mean +- SE over seeds.

    python experiments/lut-alexnet/make_results.py --backbones exp_backbones_v1 --heads exp_dense_heads_v1

For each (dataset, input, trunk, width) the learning rate is the one run_heads selected on val (its
summary's "selected"), or else the one with the best mean val accuracy; the test column is that lr's test
accuracy. The full lr sweep follows in a second table. Where the experiment saved test predictions, a paired
table compares every two head configs on the same inputs, seed by seed, on the same test images.
"""
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness.runlog import load_predictions, mcnemar, mean_se  # noqa: E402

HERE = Path(__file__).resolve().parent
ORDER = {"mnist": 0, "fmnist": 1, "cifar10": 2, "pixels": 0, "conv3": 1, "conv4": 2, "final": 3}


def pct(m, se=None):
    s = f"{100 * m:.2f}"
    if se is None:
        return s
    return f"{s} ± —" if math.isnan(se) else f"{s} ± {100 * se:.2f}"


def macs(r):
    """Multiply-adds per input in the head's linear layers (older runs did not record it: plain MLP only)."""
    if "linear_macs" in r:
        return r["linear_macs"]
    w = r["width"]
    return r["in_dim"] * w + w * w + w * 10


def trunk_line(inv):
    tag = inv.get("trunk_tag", inv["trunk"])
    return f"`{tag}`" + (f" (dropout {inv['dropout']})" if inv["trunk"] == "mlp" else "")


def paired(best, groups, preds):
    """Rows comparing every two head configs of a (dataset, input): accuracy difference per seed pair and the
    McNemar test on the same test images."""
    rows = []
    cells = sorted(best, key=lambda k: (ORDER[k[0]], ORDER[k[1]], k[2], -k[3]))
    for i, a in enumerate(cells):
        for bk in cells[i + 1:]:
            if a[:2] != bk[:2] or f"_labels__{a[0]}" not in preds:
                continue
            labels = preds[f"_labels__{a[0]}"]
            ra = {r["seed"]: r for r in groups[(*a, best[a][1])]}
            rb = {r["seed"]: r for r in groups[(*bk, best[bk][1])]}
            diffs, tests = [], []
            for seed in sorted(set(ra) & set(rb)):
                ida, idb = (f"{k[0]}__{k[1]}__{k[2]}_W{k[3]}__lr{best[k][1]:g}__s{seed}" for k in (a, bk))
                if ida in preds and idb in preds:
                    diffs.append(ra[seed]["final_test_acc"] - rb[seed]["final_test_acc"])
                    tests.append(mcnemar(preds[ida], preds[idb], labels))
            if diffs:
                m, se = mean_se(diffs)
                detail = "; ".join(f"{b}/{c}, p={p:.2g}" for b, c, p in tests)
                rows.append(f"| {a[0]} | {a[1]} | {a[2]} W={a[3]} | {bk[2]} W={bk[3]} | "
                            f"{100 * m:+.2f} ± {'—' if se != se else f'{100 * se:.2f}'} | {detail} |")
    return rows


def head_groups(runs):
    groups = defaultdict(list)
    for r in runs.values():
        groups[(r["dataset"], r["input"], r["trunk"], r["width"], r["lr"])].append(r)
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", default="exp_backbones_v1")
    ap.add_argument("--heads", nargs="+", default=["exp_dense_heads_v1", "exp_conv3_dense_vs_lut_v1",
                                                   "exp_pixels_dense_vs_lut_v1", "exp_final_dense_vs_lut_v1"])
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
        invocations = json.loads((HERE / heads / "config.json").read_text())["invocations"]
        cfg = invocations[-1]
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
                  f"Features from `{cfg['backbone_features']}`; {cfg['optim']['epochs']} epochs. Trunks: "
                  + ", ".join(dict.fromkeys(trunk_line(i) for i in invocations))
                  + ". `pixels` = raw-pixel MLP reference (no backbone). MACs = multiply-adds per input in the"
                  " head's linear layers (a LUT trunk also adds table rows, no multiplies).",
                  f"Selection: {cfg['selection']}.", "",
                  "| dataset | input | in dim | trunk | W | lr | train | val | test | params | MACs | n |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for key in sorted(best, key=lambda k: (ORDER[k[0]], ORDER[k[1]], k[2], -k[3])):
            rs = groups[(*key, best[key][1])]
            tr, va, te = (mean_se([r[f"final_{s}_acc"] for r in rs]) for s in ("train", "val", "test"))
            lines.append(f"| {key[0]} | {key[1]} | {rs[0]['in_dim']} | {key[2]} | {key[3]} | {best[key][1]:g} | "
                         f"{pct(tr[0])} | {pct(*va)} | {pct(*te)} | {rs[0]['params'] / 1e6:.2f}M | "
                         f"{macs(rs[0]) / 1e6:.2f}M | {len(rs)} |")
        rows = paired(best, groups, load_predictions(HERE / heads))
        if rows:
            lines += ["", "Paired on the same test images (selected lr each side). Δ = test acc A − B, mean ± SE "
                      "over seed pairs; per seed: b/c = images only A / only B gets right, p = McNemar.", "",
                      "| dataset | input | A | B | Δ test | per seed b/c, p |", "|---|---|---|---|---|---|", *rows]
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
