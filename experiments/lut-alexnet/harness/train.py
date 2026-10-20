"""One training loop for both stages: AdamW + one-cycle, whole-split tensors, seeded shuffles/augmentation.

``prepare(xb, gen, train)`` turns a stored batch into model input on the device (image normalisation and
augmentation for backbones, a dtype cast for cached features), so the loop is the same for both stages.
"""
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class OptimConfig:
    epochs: int
    lr: float  # one-cycle peak
    batch_size: int = 256
    weight_decay: float = 0.05  # on weights with ndim >= 2 only (no BN / bias decay)
    pct_start: float = 0.25
    label_smoothing: float = 0.0

    def to_dict(self):
        return asdict(self)


def seed_everything(seed):
    """Seeds the global CPU / MPS / CUDA generators (model init and dropout). Data order and augmentation
    use their own CPU generator inside ``fit``."""
    torch.manual_seed(seed)


def pick_device(name=None):
    if name:
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cast_prepare(device):
    """``prepare`` for cached feature tensors (stored float16 or float32)."""
    return lambda xb, gen, train: xb.to(device).float()


def _param_groups(model, weight_decay):
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


@torch.no_grad()
def evaluate(model, x, y, prepare, batch_size=1000):
    model.eval()
    loss, correct = 0.0, 0
    for i in range(0, len(x), batch_size):
        logits = model(prepare(x[i:i + batch_size], None, False))
        yb = y[i:i + batch_size].to(logits.device)
        loss += F.cross_entropy(logits, yb, reduction="sum").item()
        correct += (logits.argmax(1) == yb).sum().item()
    return {"loss": loss / len(x), "acc": correct / len(x)}


def fit(model, train, prepare, cfg: OptimConfig, seed, val=None, log=None):
    """Trains ``model`` in place on ``train = (x, y)``; evaluates on ``val`` after every epoch.

    Returns the per-epoch history; each row is also passed to ``log`` as it is produced. Same seed, same device
    and same inputs give the same run (bitwise on CPU, tested; see the README for MPS).
    """
    x, y = train
    device = next(model.parameters()).device
    gen = torch.Generator().manual_seed(seed + 1)
    opt = torch.optim.AdamW(_param_groups(model, cfg.weight_decay), lr=cfg.lr)
    steps = cfg.epochs * ((len(x) + cfg.batch_size - 1) // cfg.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=steps, pct_start=cfg.pct_start)
    history = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        tot_loss = torch.zeros((), device=device)
        tot_correct = torch.zeros((), device=device, dtype=torch.long)
        for idx in torch.randperm(len(x), generator=gen).split(cfg.batch_size):
            xb = prepare(x[idx], gen, True)
            yb = y[idx].to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, label_smoothing=cfg.label_smoothing)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot_loss += loss.detach() * len(idx)
            tot_correct += (logits.detach().argmax(1) == yb).sum()
        row = {"epoch": epoch, "lr_end": sched.get_last_lr()[0],
               "train_loss": tot_loss.item() / len(x), "train_acc": tot_correct.item() / len(x)}
        if val is not None:
            v = evaluate(model, *val, prepare)
            row.update(val_loss=v["loss"], val_acc=v["acc"])
        row["epoch_s"] = time.time() - t0
        history.append(row)
        if log:
            log(row)
    return history


def n_params(module: nn.Module):
    return sum(p.numel() for p in module.parameters())
