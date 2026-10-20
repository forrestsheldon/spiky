"""Datasets as whole-split uint8 tensors, a fixed train/val split, and batched on-device augmentation.

Every image is stored as uint8 (N, C, 32, 32): MNIST and Fashion-MNIST are zero-padded 28 -> 32 (1 channel),
CIFAR-10 is 3 channels. The last ``N_VAL`` images of a fixed permutation of the official train set are held
out as a validation split, *before* any backbone sees them, so head hyperparameters are chosen on val and the
test set is only reported.
"""
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent.parent
CACHE_ROOT = Path(os.environ.get("LUT_ALEXNET_CACHE", HERE / "_cache"))

N_VAL = 5000
SPLIT_SEED = 0

# augment: random 32x32 crop of the 4-pixel zero-padded image + horizontal flip (training batches only)
DATASETS = {
    "mnist": dict(tv_class="MNIST", channels=1, pad_to_32=2, augment=False, n_classes=10),
    "fmnist": dict(tv_class="FashionMNIST", channels=1, pad_to_32=2, augment=True, n_classes=10),
    "cifar10": dict(tv_class="CIFAR10", channels=3, pad_to_32=0, augment=True, n_classes=10),
}


@dataclass
class Split:
    x: torch.Tensor  # uint8 (N, C, 32, 32)
    y: torch.Tensor  # int64 (N,)


@dataclass
class ImageData:
    name: str
    train: Split
    val: Split
    test: Split
    mean: torch.Tensor  # (C,) per-channel mean of train / 255
    std: torch.Tensor
    augment: bool
    n_classes: int

    @property
    def channels(self):
        return self.train.x.shape[1]


def _to_uint8_nchw(data, pad):
    x = torch.as_tensor(data)
    if x.ndim == 3:  # (N, 28, 28) grayscale
        x = x.unsqueeze(1)
    else:  # CIFAR numpy (N, 32, 32, 3)
        x = x.permute(0, 3, 1, 2)
    if pad:
        x = F.pad(x, (pad, pad, pad, pad))
    return x.contiguous()


def train_val_split(n, n_val=N_VAL, seed=SPLIT_SEED):
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    return perm[:-n_val].sort().values, perm[-n_val:].sort().values


def from_tensors(name, x_train, y_train, x_test, y_test, n_val=N_VAL):
    """Builds ImageData from uint8 (N, C, 32, 32) tensors (also the entry point for synthetic data in tests)."""
    spec = DATASETS[name]
    tr, va = train_val_split(len(x_train), n_val)
    xf = x_train[tr].float() / 255
    return ImageData(
        name=name,
        train=Split(x_train[tr], y_train[tr]),
        val=Split(x_train[va], y_train[va]),
        test=Split(x_test, y_test),
        mean=xf.mean(dim=(0, 2, 3)),
        std=xf.std(dim=(0, 2, 3)),
        augment=spec["augment"],
        n_classes=spec["n_classes"],
    )


def load(name, root=None):
    import torchvision

    spec = DATASETS[name]
    root = Path(root or CACHE_ROOT / "data")
    cls = getattr(torchvision.datasets, spec["tv_class"])
    out = []
    for train in (True, False):
        ds = cls(str(root), train=train, download=True)
        out += [_to_uint8_nchw(ds.data, spec["pad_to_32"]), torch.as_tensor(ds.targets, dtype=torch.long)]
    return from_tensors(name, *out)


def random_crop_flip(x, gen, pad=4, flip=True):
    """Per-image random crop of the zero-padded batch plus horizontal flip; offsets drawn on the CPU ``gen``
    so the result is the same on every device."""
    n, _, h, w = x.shape
    oy = torch.randint(0, 2 * pad + 1, (n,), generator=gen)
    ox = torch.randint(0, 2 * pad + 1, (n,), generator=gen)
    rows = oy[:, None] + torch.arange(h)
    cols = ox[:, None] + torch.arange(w)
    if flip:
        cols = torch.where((torch.rand(n, generator=gen) < 0.5)[:, None], cols.flip(1), cols)
    rows, cols = rows.to(x.device), cols.to(x.device)
    xp = F.pad(x, (pad, pad, pad, pad)).permute(0, 2, 3, 1)  # (N, H+2p, W+2p, C)
    idx = torch.arange(n, device=x.device)[:, None, None]
    return xp[idx, rows[:, :, None], cols[:, None, :]].permute(0, 3, 1, 2).contiguous()


def image_prepare(data: ImageData, device):
    """Returns ``prepare(xb, gen, train)``: uint8 batch -> normalised float on ``device``, augmented when training."""
    mean = data.mean.to(device)[None, :, None, None]
    std = data.std.to(device)[None, :, None, None]

    def prepare(xb, gen, train):
        x = xb.to(device).float() / 255
        if train and data.augment:
            x = random_crop_flip(x, gen)
        return (x - mean) / std

    return prepare


def pixel_features(data: ImageData):
    """Normalised, flattened pixels per split: the input of the raw-pixel MLP reference."""
    mean, std = data.mean[None, :, None, None], data.std[None, :, None, None]
    return {s: ((getattr(data, s).x.float() / 255 - mean) / std).flatten(1) for s in ("train", "val", "test")}
