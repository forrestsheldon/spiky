"""Pluggable convolutional backbones: (N, C, 32, 32) -> a dict of named feature taps.

A backbone exposes ``tap_dims`` (name -> width) and ``forward(x, taps=None)``: with ``taps=None`` it returns the
final feature vector, otherwise a dict holding the requested taps. ``"final"`` is the vector the classifier head
reads during backbone training; the other taps are earlier, pooled activations cached so heads can be compared on
less linearly separable inputs.
"""
import torch.nn as nn
import torch.nn.functional as F


class AlexNetCIFAR(nn.Module):
    """CIFAR AlexNet: five 3x3 convs (64/192/384/256/256), 2x2 max-pools after conv1, conv2 and conv5,
    so a 32x32 input ends at 256x4x4 = 4096 features. ``batchnorm`` adds BN after every conv.
    Taps conv3 (384x8x8) and conv4 (256x8x8) are post-ReLU, average-pooled to ``tap_pool`` x ``tap_pool``."""

    def __init__(self, in_channels=3, batchnorm=False, widths=(64, 192, 384, 256, 256), tap_pool=2):
        super().__init__()

        def block(cin, cout):
            layers = [nn.Conv2d(cin, cout, 3, padding=1, bias=not batchnorm)]
            if batchnorm:
                layers.append(nn.BatchNorm2d(cout))
            return nn.Sequential(*layers, nn.ReLU(inplace=True))

        chans = (in_channels, *widths)
        self.conv1, self.conv2, self.conv3, self.conv4, self.conv5 = (
            block(chans[i], chans[i + 1]) for i in range(5))
        self.pool = nn.MaxPool2d(2)
        self.tap_pool = tap_pool
        self.tap_dims = {
            "conv3": widths[2] * tap_pool ** 2,
            "conv4": widths[3] * tap_pool ** 2,
            "final": widths[4] * 4 * 4,
        }

    @property
    def out_dim(self):
        return self.tap_dims["final"]

    def forward(self, x, taps=None):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        c3 = self.conv3(x)
        c4 = self.conv4(c3)
        final = self.pool(self.conv5(c4)).flatten(1)
        if taps is None:
            return final
        acts = {"conv3": c3, "conv4": c4}
        return {t: final if t == "final" else F.adaptive_avg_pool2d(acts[t], self.tap_pool).flatten(1)
                for t in taps}


BACKBONES = {"alexnet_cifar": AlexNetCIFAR}


def build_backbone(name, in_channels, **kwargs):
    return BACKBONES[name](in_channels=in_channels, **kwargs)
