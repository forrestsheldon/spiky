"""First lutorch_ex experiment: a LUT layer vs a dense MLP on sklearn's 8x8 digits (CPU, ~1 min).

Run from any env with torch, scikit-learn and spiky installed (`pip install -e .` in the spiky repo):
    python experiments/lut-alexnet/digits_first_experiment.py
"""
import time

import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from spiky.lutorch_ex import (ConfidenceLUT, LUTSpec, ManifestoHardLUT, ManifestoSoftLUT,
                              ProjectionMHL, cell_tv_penalty)

torch.set_num_threads(4)
X, y = load_digits(return_X_y=True)
X = (X / 16.0 - 0.5).astype("float32")
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
Xtr, Xte = torch.tensor(Xtr), torch.tensor(Xte)
ytr, yte = torch.tensor(ytr), torch.tensor(yte)

# 4 heads x 16 tables, 2**6 = 64 cells per table, 16-wide rows
SPEC = LUTSpec(h_in=4, h_out=4, tph=16, nap=6, d_in=16, d_out=16)


def make(name, seed):
    torch.manual_seed(seed)
    if name == "dense MLP (64-256-10)":
        return nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 10))
    cart = {
        "ManifestoHardLUT": lambda: ManifestoHardLUT(SPEC, seed=seed),
        "ManifestoSoftLUT": lambda: ManifestoSoftLUT(SPEC, seed=seed),
        "ConfidenceLUT n=1": lambda: ConfidenceLUT(SPEC, seed=seed, read_top_n=1),
        "ConfidenceLUT n=2": lambda: ConfidenceLUT(SPEC, seed=seed, read_top_n=2),
    }[name]()
    # compress 64 -> 4x16, cartridge, decompress 4x16 -> 10 logits (decompress starts at zero)
    return ProjectionMHL(cart, input_dim=64, output_dim=10)


def run(name, seed, epochs=150, lr=3e-3, tv=0.0):
    model = make(name, seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    t0 = time.time()
    for _ in range(epochs):
        model.train()
        for idx in torch.randperm(len(Xtr), generator=g).split(128):
            loss = nn.functional.cross_entropy(model(Xtr[idx]), ytr[idx])
            if tv and not name.startswith("dense"):
                loss = loss + tv * cell_tv_penalty(model)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        acc = (model(Xte).argmax(1) == yte).float().mean().item()
    n_params = sum(p.numel() for p in model.parameters())
    return acc, n_params, time.time() - t0


if __name__ == "__main__":
    names = ["dense MLP (64-256-10)", "ManifestoHardLUT", "ManifestoSoftLUT",
             "ConfidenceLUT n=1", "ConfidenceLUT n=2"]
    seeds = [0, 1, 2]
    print(f"digits: {len(Xtr)} train / {len(Xte)} test; LUTSpec = {SPEC}")
    print(f"{'model':24s} {'params':>8s} {'test acc (3 seeds)':>26s} {'mean':>6s} {'s/run':>6s}")
    for name in names:
        res = [run(name, s) for s in seeds]
        accs = [r[0] for r in res]
        print(f"{name:24s} {res[0][1]:8d} {'  '.join(f'{a:.3f}' for a in accs):>26s} "
              f"{sum(accs)/len(accs):6.3f} {sum(r[2] for r in res)/len(res):6.1f}")
