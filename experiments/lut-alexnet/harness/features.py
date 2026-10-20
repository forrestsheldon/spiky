"""Feature extraction from a trained backbone and an on-disk cache keyed by the run's metadata.

A cache file holds, per split, every tap as a float16 (N, d) tensor plus the int64 labels, and the ``meta`` dict
it was computed from. A load whose ``meta`` differs from the request recomputes and overwrites, so a changed
backbone config or checkpoint never silently reuses stale features.
"""
import hashlib
import json
from pathlib import Path

import torch

SPLITS = ("train", "val", "test")


@torch.no_grad()
def extract(backbone, x, prepare, taps, batch_size=1000, dtype=torch.float16):
    backbone.eval()
    out = {t: [] for t in taps}
    for i in range(0, len(x), batch_size):
        feats = backbone(prepare(x[i:i + batch_size], None, False), taps=taps)
        for t in taps:
            out[t].append(feats[t].to("cpu", dtype))
    return {t: torch.cat(v) for t, v in out.items()}


def file_sha1(path, n=12):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def meta_key(meta, n=10):
    return hashlib.sha1(json.dumps(meta, sort_keys=True).encode()).hexdigest()[:n]


def cached_features(path, meta, compute):
    """Returns ``{split: {tap: tensor, "y": labels}}`` from ``path`` when its meta matches, else from
    ``compute()`` (which must return that structure), saving the result first."""
    path = Path(path)
    if path.exists():
        blob = torch.load(path, weights_only=True)
        if blob["meta"] == meta:
            return blob["splits"]
    splits = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"meta": meta, "splits": splits}, tmp)
    tmp.replace(path)
    return splits


def extract_all(backbone, data, prepare, taps):
    return {s: {**extract(backbone, getattr(data, s).x, prepare, taps), "y": getattr(data, s).y}
            for s in SPLITS}
