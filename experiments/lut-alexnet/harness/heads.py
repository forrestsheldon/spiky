"""Classifier heads: a pluggable *trunk* (in_dim -> width) followed by a linear readout (width -> classes).

The trunk is the part the dense-vs-LUT comparison swaps. ``"mlp"`` is AlexNet's fc1-fc2, kept plain
(Linear-ReLU-Dropout twice); fc3 is the readout and is shared by every trunk.

Adding the LUT trunk (planned 2026-10-21): register a factory with the same signature, e.g.

    def lut_trunk(in_dim, width, spec, cartridge="ConfidenceLUT", **cartridge_kwargs):
        cart = getattr(spiky.lutorch_ex, cartridge)(spec, **cartridge_kwargs)
        return ProjectionMHL(cart, input_dim=in_dim, output_dim=width)
    TRUNKS["lut"] = lut_trunk

Anything mapping (N, in_dim) -> (N, width) works; the training loop and feature cache are unchanged.
"""
import torch.nn as nn


def mlp_trunk(in_dim, width, dropout=0.5):
    return nn.Sequential(
        nn.Linear(in_dim, width), nn.ReLU(inplace=True), nn.Dropout(dropout),
        nn.Linear(width, width), nn.ReLU(inplace=True), nn.Dropout(dropout),
    )


TRUNKS = {"mlp": mlp_trunk}


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
