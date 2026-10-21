"""Shapes, feature caching and fixed-seed determinism of the bootstrap harness (CPU, synthetic data, no downloads).

    pytest experiments/lut-alexnet/tests
"""
import json
import math

import pytest
import torch

from harness import data as D
from harness.backbones import AlexNetCIFAR, build_backbone
from harness.features import cached_features, extract, extract_all
from harness.heads import Classifier, build_head, cost
from harness.runlog import ExpDir, load_predictions, mcnemar, mean_se
from harness.train import OptimConfig, cast_prepare, evaluate, fit, seed_everything

CPU = torch.device("cpu")
SMALL = (8, 16, 16, 16, 16)  # narrow AlexNet for fast CPU tests; final = 16 * 4 * 4 = 256


def synthetic(name, n_train=96, n_test=32, n_val=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    c = D.DATASETS[name]["channels"]
    x = torch.randint(0, 256, (n_train + n_test, c, 32, 32), generator=g, dtype=torch.uint8)
    y = torch.randint(0, 10, (n_train + n_test,), generator=g)
    return D.from_tensors(name, x[:n_train], y[:n_train], x[n_train:], y[n_train:], n_val=n_val)


# ---- data ----

def test_grayscale_padded_to_32():
    x = D._to_uint8_nchw(torch.full((5, 28, 28), 7, dtype=torch.uint8), pad=2)
    assert x.shape == (5, 1, 32, 32) and x.dtype == torch.uint8
    assert x[:, :, 2:30, 2:30].eq(7).all() and x[:, :, :2].eq(0).all() and x[:, :, :, 30:].eq(0).all()


def test_cifar_layout_to_nchw():
    x = torch.arange(2 * 32 * 32 * 3, dtype=torch.int64).remainder(256).to(torch.uint8).reshape(2, 32, 32, 3)
    out = D._to_uint8_nchw(x.numpy(), pad=0)
    assert out.shape == (2, 3, 32, 32) and torch.equal(out[1, 2, 5, 7], x[1, 5, 7, 2])


def test_train_val_split_is_fixed_and_disjoint():
    tr, va = D.train_val_split(100, n_val=20)
    tr2, va2 = D.train_val_split(100, n_val=20)
    assert torch.equal(tr, tr2) and torch.equal(va, va2)
    assert len(va) == 20 and len(tr) == 80
    assert set(tr.tolist()).isdisjoint(va.tolist()) and set(tr.tolist()) | set(va.tolist()) == set(range(100))


def test_dataset_specs():
    d = synthetic("fmnist")
    assert d.channels == 1 and d.augment and d.mean.shape == (1,)
    d = synthetic("cifar10")
    assert d.channels == 3 and d.augment and d.mean.shape == (3,)
    assert not D.DATASETS["mnist"]["augment"]
    assert len(d.train.y) == 64 and len(d.val.y) == 32 and len(d.test.y) == 32


def test_random_crop_flip_shapes_and_determinism():
    x = torch.rand(6, 3, 32, 32)
    a = D.random_crop_flip(x, torch.Generator().manual_seed(1))
    b = D.random_crop_flip(x, torch.Generator().manual_seed(1))
    c = D.random_crop_flip(x, torch.Generator().manual_seed(2))
    assert a.shape == x.shape and torch.equal(a, b) and not torch.equal(a, c)


def _crop_flip_nhwc_reference(x, gen, pad=4, flip=True):
    """The first implementation (NHWC gather, permuted back); the NCHW gather must reproduce it exactly."""
    import torch.nn.functional as F
    n, _, h, w = x.shape
    oy = torch.randint(0, 2 * pad + 1, (n,), generator=gen)
    ox = torch.randint(0, 2 * pad + 1, (n,), generator=gen)
    rows = oy[:, None] + torch.arange(h)
    cols = ox[:, None] + torch.arange(w)
    if flip:
        cols = torch.where((torch.rand(n, generator=gen) < 0.5)[:, None], cols.flip(1), cols)
    xp = F.pad(x, (pad, pad, pad, pad)).permute(0, 2, 3, 1)
    idx = torch.arange(n)[:, None, None]
    return xp[idx, rows[:, :, None], cols[:, None, :]].permute(0, 3, 1, 2).contiguous()


@pytest.mark.parametrize("channels", [1, 3])
def test_random_crop_flip_matches_reference_with_canonical_strides(channels):
    x = torch.rand(16, channels, 32, 32)
    out = D.random_crop_flip(x, torch.Generator().manual_seed(3))
    assert torch.equal(out, _crop_flip_nhwc_reference(x, torch.Generator().manual_seed(3)))
    assert out.stride() == (channels * 32 * 32, 32 * 32, 32, 1)


def test_random_crop_flip_degenerate_cases():
    x = torch.rand(8, 1, 32, 32)
    assert torch.equal(D.random_crop_flip(x, torch.Generator(), pad=0, flip=False), x)
    out = D.random_crop_flip(x, torch.Generator().manual_seed(0), pad=0, flip=True)
    for o, i in zip(out, x):
        assert torch.equal(o, i) or torch.equal(o, i.flip(-1))


def test_image_prepare_normalises_and_augments_only_in_training():
    d = synthetic("cifar10")
    prep = D.image_prepare(d, CPU)
    xe = prep(d.train.x, None, False)
    assert xe.dtype == torch.float32 and xe.shape == d.train.x.shape
    assert torch.allclose(xe.mean(dim=(0, 2, 3)), torch.zeros(3), atol=1e-4)
    xt = prep(d.train.x, torch.Generator().manual_seed(0), True)
    assert xt.shape == xe.shape and not torch.equal(xt, xe)


def test_pixel_features_shapes():
    d = synthetic("mnist")
    px = D.pixel_features(d)
    assert px["train"].shape == (64, 1024) and px["test"].shape == (32, 1024)


# ---- backbones and heads ----

@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("bn", [False, True])
def test_alexnet_shapes(channels, bn):
    m = AlexNetCIFAR(in_channels=channels, batchnorm=bn).eval()
    assert m.tap_dims == {"conv3": 384 * 4, "conv4": 256 * 4, "final": 4096}
    x = torch.randn(2, channels, 32, 32)
    assert m(x).shape == (2, 4096)
    taps = m(x, taps=("conv3", "conv4", "final"))
    assert {k: v.shape[1] for k, v in taps.items()} == m.tap_dims
    assert torch.equal(taps["final"], m(x))
    assert any(isinstance(l, torch.nn.BatchNorm2d) for l in m.modules()) == bn


def test_alexnet_widths_and_tap_pool_are_configurable():
    m = build_backbone("alexnet_cifar", 3, widths=SMALL, tap_pool=4)
    assert m.tap_dims == {"conv3": 16 * 16, "conv4": 16 * 16, "final": 256}
    assert m(torch.randn(1, 3, 32, 32), taps=("conv3",))["conv3"].shape == (1, 256)


def test_alexnet_param_count_matches_cost_table():
    # projects/alexnet.md cost table: conv1-5 = 0.002 + 0.11 + 0.66 + 0.88 + 0.59 M, fc1 = fc2 = 16.8M
    m = AlexNetCIFAR(in_channels=3)
    assert sum(p.numel() for p in m.parameters()) == pytest.approx(2.25e6, rel=0.01)
    head = build_head("mlp", 4096, 4096, 10)
    assert sum(p.numel() for p in head.parameters()) == 2 * (4096 * 4096 + 4096) + 4096 * 10 + 10


@pytest.mark.parametrize("width", [4096, 1024])
def test_head_shapes(width):
    head = build_head("mlp", 1536, width, 10)
    assert head(torch.randn(3, 1536)).shape == (3, 10)
    assert head.trunk(torch.randn(3, 1536)).shape == (3, width)
    model = Classifier(AlexNetCIFAR(in_channels=1, widths=SMALL), build_head("mlp", 256, width, 10))
    assert model(torch.randn(2, 1, 32, 32)).shape == (2, 10)


# ---- feature caching ----

def test_extract_float16_matches_backbone():
    d = synthetic("cifar10")
    bb = AlexNetCIFAR(widths=SMALL, batchnorm=True).eval()
    prep = D.image_prepare(d, CPU)
    feats = extract(bb, d.test.x, prep, ("conv4", "final"), batch_size=10)
    assert feats["final"].dtype == torch.float16 and feats["final"].shape == (32, 256)
    with torch.no_grad():
        ref = bb(prep(d.test.x, None, False))
    assert torch.allclose(feats["final"].float(), ref, rtol=1e-2, atol=1e-3)


def test_feature_cache_roundtrip_and_invalidation(tmp_path):
    d = synthetic("mnist")
    bb = AlexNetCIFAR(in_channels=1, widths=SMALL).eval()
    prep = D.image_prepare(d, CPU)
    calls = []

    def compute():
        calls.append(1)
        return extract_all(bb, d, prep, ("conv3", "final"))

    path = tmp_path / "feats.pt"
    meta = {"run_id": "x", "checkpoint_sha1": "abc"}
    first = cached_features(path, meta, compute)
    second = cached_features(path, dict(meta), compute)
    assert len(calls) == 1 and path.exists()
    for s in ("train", "val", "test"):
        assert set(second[s]) == {"conv3", "final", "y"}
        for k in second[s]:
            assert torch.equal(first[s][k], second[s][k])
    assert second["train"]["final"].shape == (64, 256) and torch.equal(second["val"]["y"], d.val.y)
    cached_features(path, {**meta, "checkpoint_sha1": "def"}, compute)
    assert len(calls) == 2


# ---- determinism ----

def _train_backbone(seed, name="fmnist"):
    d = synthetic(name)
    seed_everything(seed)
    model = Classifier(build_backbone("alexnet_cifar", d.channels, batchnorm=True, widths=SMALL),
                       build_head("mlp", 256, 32, 10))
    hist = fit(model, (d.train.x, d.train.y), D.image_prepare(d, CPU), OptimConfig(epochs=2, lr=1e-3, batch_size=16),
               seed, val=(d.val.x, d.val.y))
    return model.state_dict(), hist


def _assert_same(sd1, sd2):
    assert sd1.keys() == sd2.keys()
    for k in sd1:
        assert torch.equal(sd1[k], sd2[k]), k


def test_backbone_training_is_deterministic_under_fixed_seed():
    sd1, h1 = _train_backbone(0)
    sd2, h2 = _train_backbone(0)
    _assert_same(sd1, sd2)
    assert [{k: v for k, v in r.items() if k != "epoch_s"} for r in h1] == \
           [{k: v for k, v in r.items() if k != "epoch_s"} for r in h2]
    sd3, _ = _train_backbone(1)
    assert any(not torch.equal(sd1[k], sd3[k]) for k in sd1 if sd1[k].is_floating_point())


def _train_head(seed, feats):
    seed_everything(seed)
    head = build_head("mlp", feats.shape[1], 64, 10)
    y = torch.arange(len(feats)) % 10
    fit(head, (feats, y), cast_prepare(CPU), OptimConfig(epochs=3, lr=3e-3, batch_size=16), seed, val=(feats, y))
    return head.state_dict(), evaluate(head, feats, y, cast_prepare(CPU))


def test_head_training_is_deterministic_under_fixed_seed():
    feats = torch.randn(80, 48, generator=torch.Generator().manual_seed(0)).half()
    (sd1, e1), (sd2, e2), (sd3, _) = _train_head(0, feats), _train_head(0, feats), _train_head(1, feats)
    _assert_same(sd1, sd2)
    assert e1 == e2
    assert any(not torch.equal(sd1[k], sd3[k]) for k in sd1)


def test_head_learns_separable_features():
    g = torch.Generator().manual_seed(0)
    y = torch.arange(200) % 10
    feats = torch.nn.functional.one_hot(y, 10).float() * 3 + 0.3 * torch.randn(200, 10, generator=g)
    seed_everything(0)
    head = build_head("mlp", 10, 64, 10, dropout=0.0)
    fit(head, (feats, y), cast_prepare(CPU), OptimConfig(epochs=10, lr=3e-3, batch_size=32), 0)
    assert evaluate(head, feats, y, cast_prepare(CPU))["acc"] > 0.95


# ---- LUT trunk ----

LUT_SMALL = dict(cartridge="ConfidenceLUT", read_top_n=2, h=2, tph=4, nap=3, d=8)


def test_lut_head_shapes_and_zero_start():
    pytest.importorskip("spiky.lutorch_ex")
    head = build_head("lut", 48, 32, 10, seed=0, **LUT_SMALL)
    x = torch.randn(5, 48)
    assert head.trunk(x).shape == (5, 32) and head(x).shape == (5, 10)
    assert head.trunk(x).eq(0).all()  # decompress starts at zero (library default)
    assert torch.allclose(head(x), head.readout.bias.expand(5, 10))


def _train_lut(seed, cart_seed, feats, y):
    seed_everything(seed)
    head = build_head("lut", feats.shape[1], 32, 10, seed=cart_seed, **LUT_SMALL)
    fit(head, (feats, y), cast_prepare(CPU), OptimConfig(epochs=3, lr=3e-3, batch_size=16), seed)
    return head.state_dict()


def test_lut_head_training_is_deterministic_under_fixed_seed():
    pytest.importorskip("spiky.lutorch_ex")
    feats = torch.randn(64, 48, generator=torch.Generator().manual_seed(0))
    y = torch.arange(64) % 10
    sd1, sd2 = _train_lut(0, 0, feats, y), _train_lut(0, 0, feats, y)
    _assert_same(sd1, sd2)
    sd3 = _train_lut(0, 1, feats, y)  # same global seed, different cartridge seed: different anchors
    assert any(not torch.equal(sd1[k], sd3[k]) for k in sd1)


def test_lut_head_learns_separable_features():
    pytest.importorskip("spiky.lutorch_ex")
    g = torch.Generator().manual_seed(0)
    y = torch.arange(200) % 10
    feats = torch.nn.functional.one_hot(y, 10).float() * 3 + 0.3 * torch.randn(200, 10, generator=g)
    seed_everything(0)
    head = build_head("lut", 10, 32, 10, seed=0, **LUT_SMALL)
    fit(head, (feats, y), cast_prepare(CPU), OptimConfig(epochs=20, lr=3e-3, batch_size=32), 0)
    assert evaluate(head, feats, y, cast_prepare(CPU))["acc"] > 0.9


def test_head_cost():
    assert cost(build_head("mlp", 1536, 1024, 10)) == {"linear_macs": 1536 * 1024 + 1024 * 1024 + 1024 * 10,
                                                        "lut_row_reads": 0}
    pytest.importorskip("spiky.lutorch_ex")
    c = cost(build_head("lut", 48, 32, 10, seed=0, **LUT_SMALL))
    assert c == {"linear_macs": 48 * 16 + 16 * 32 + 32 * 10, "lut_row_reads": 2 * 4 * 2}


# ---- bookkeeping ----

def test_expdir_concurrent_writers_lose_nothing(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    harness_root = str(Path(__file__).resolve().parent.parent)
    script = (
        "import sys; sys.path.insert(0, sys.argv[1]); from harness.runlog import ExpDir\n"
        "w = int(sys.argv[3]); e = ExpDir(sys.argv[2], {'writer': w})\n"
        "for j in range(25):\n"
        "    e.logger(f'w{w}_{j}')({'epoch': 1, 'val_acc': 0.5})\n"
        "    e.save_predictions(f'w{w}_{j}', list(range(3000)))\n"
        "    e.record(f'w{w}_{j}', {'final_val_acc': 0.5})\n"
        "    e.select(f'cell{w}', j)\n"
    )
    exp_path = tmp_path / "exp"
    procs = [subprocess.Popen([sys.executable, "-c", script, harness_root, str(exp_path), str(w)])
             for w in range(4)]
    assert all(p.wait(timeout=60) == 0 for p in procs)
    again = ExpDir(exp_path, {"writer": 0})
    assert len(again.summary["runs"]) == 100 and again.summary["selected"] == {f"cell{w}": 24 for w in range(4)}
    assert len(load_predictions(exp_path)) == 100
    assert len((exp_path / "metrics.csv").read_text().splitlines()) == 101  # header + one row per run
    assert len(json.loads((exp_path / "config.json").read_text())["invocations"]) == 4


def test_predictions_roundtrip_and_mcnemar(tmp_path):
    exp = ExpDir(tmp_path / "exp", {"a": 1})
    exp.save_predictions("_labels__x", torch.tensor([0, 1, 2, 3]))
    exp.save_predictions("r1", torch.tensor([0, 1, 9, 9]))
    exp.save_predictions("r1", torch.tensor([0, 1, 2, 9]))  # a rerun: the later line wins
    preds = load_predictions(tmp_path / "exp")
    assert preds["r1"] == [0, 1, 2, 9] and ExpDir(tmp_path / "exp", {"a": 1}).predicted == {"_labels__x", "r1"}
    assert mcnemar([0, 1, 2, 9], [0, 9, 9, 9], [0, 1, 2, 3])[:2] == (2, 0)
    assert mcnemar([0, 1], [0, 1], [0, 1]) == (0, 0, 1.0)
    b, c, p = mcnemar([1] * 60 + [0] * 40, [0] * 60 + [1] * 40, [1] * 100)  # 60 vs 40 discordant
    assert (b, c) == (60, 40) and p == pytest.approx(0.0574, abs=1e-3)

def test_expdir_resume_and_mean_se(tmp_path):
    exp = ExpDir(tmp_path / "exp", {"a": 1})
    exp.logger("r1")({"epoch": 1, "val_acc": 0.5})
    exp.logger("r1")({"epoch": 2, "val_acc": 0.6})
    exp.record("r1", {"final_val_acc": 0.6})
    again = ExpDir(tmp_path / "exp", {"a": 1})
    assert again.done("r1") and not again.done("r2")
    ExpDir(tmp_path / "exp", {"a": 2})
    assert json.loads((tmp_path / "exp" / "config.json").read_text()) == {"invocations": [{"a": 1}, {"a": 2}]}
    assert (tmp_path / "exp" / "metrics.csv").read_text().splitlines()[0] == "run,epoch,val_acc"
    m, se = mean_se([0.90, 0.92, 0.94])
    assert m == pytest.approx(0.92) and se == pytest.approx(0.02 / math.sqrt(3))
    assert math.isnan(mean_se([0.5])[1])
