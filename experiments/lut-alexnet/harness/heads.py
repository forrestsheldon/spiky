"""Classifier heads: a pluggable *trunk* (in_dim -> width) followed by a linear readout (width -> classes).

The trunk is the part the dense-vs-LUT comparison swaps:

- ``"mlp"``: AlexNet's fc1-fc2, kept plain (Linear-ReLU-Dropout twice).
- ``"lut"``: one lutorch_ex ``ProjectionMHL`` in place of fc1-fc2 (compress -> LUT cartridge -> decompress).

fc3 is the readout and is shared by every trunk. Anything mapping (N, in_dim) -> (N, width) can be registered
in ``TRUNKS``; the training loop and feature cache are unchanged.
"""
import torch.nn as nn


def mlp_trunk(in_dim, width, dropout=0.5):
    return nn.Sequential(
        nn.Linear(in_dim, width), nn.ReLU(inplace=True), nn.Dropout(dropout),
        nn.Linear(width, width), nn.ReLU(inplace=True), nn.Dropout(dropout),
    )


def lut_trunk(in_dim, width, *, cartridge="ConfidenceLUT", h=8, tph=64, nap=8, d=48, seed=0, **cartridge_kwargs):
    """``ProjectionMHL``: compress in_dim -> h x d, a lutorch_ex cartridge (h heads x tph tables of 2**nap cells,
    d-wide rows), decompress -> width. The library zero-initialises decompress, so a fresh trunk outputs zero.
    ``seed`` draws the cartridge's anchors and initial tables; the projections use the global seed."""
    from spiky import lutorch_ex as lx

    spec = lx.LUTSpec(h_in=h, h_out=h, tph=tph, nap=nap, d_in=d, d_out=d)
    cart = getattr(lx, cartridge)(spec, seed=seed, **cartridge_kwargs)
    return lx.ProjectionMHL(cart, input_dim=in_dim, output_dim=width)


TRUNKS = {"mlp": mlp_trunk, "lut": lut_trunk}


def cost(head):
    """Per-input arithmetic: multiply-adds in every nn.Linear, and rows read from LUT tables (each read is a
    d_out-wide add, no multiplies)."""
    linear = sum(m.in_features * m.out_features for m in head.modules() if isinstance(m, nn.Linear))
    reads = 0
    for m in head.modules():
        spec = getattr(m, "spec", None)
        if spec is not None and hasattr(spec, "n_tables"):
            reads += spec.n_tables * getattr(m, "read_top_n", 2)
    return {"linear_macs": linear, "lut_row_reads": reads}


class Head(nn.Module):
    def __init__(self, trunk, width, n_classes):
        super().__init__()
        self.trunk = trunk
        self.readout = nn.Linear(width, n_classes)

    def forward(self, x):
        return self.readout(self.trunk(x))


def build_head(kind, in_dim, width, n_classes, **trunk_kwargs):
    return Head(TRUNKS[kind](in_dim, width, **trunk_kwargs), width, n_classes)


class Classifier(nn.Module):
    """Backbone + head, trained end to end; the backbone's ``final`` tap feeds the head."""

    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        return self.head(self.backbone(x))
