"""Pins the throughput meter's arithmetic and the FLOPs probe's non-invasiveness.

The properties that matter: the first window is discarded rather than reported as a very slow
step, rates divide by the measured window rather than an assumed one, and counting a step leaves
neither gradients nor RNG state behind — the probe runs on the batch the first real step will
train on, so anything it perturbed would perturb training.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from engine.mfu import ThroughputMeter, _counted_kernels, measure_step_flops
from layers.common.attention import SelfAttention, counting_kernels
from layers.dinov3.attention import SelfAttention as Dinov3SelfAttention


class _TinyAlgorithm(nn.Module):
    """Stands in for a BaseAlgorithm: called on a batch, returns a metrics dict with a loss."""

    def __init__(self, dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"loss": self.net(batch).square().mean()}


def _no_autocast():  # noqa: ANN202 - matches Trainer._autocast's factory shape
    import contextlib

    return contextlib.nullcontext()


@pytest.mark.unit
def test_first_window_is_discarded() -> None:
    meter = ThroughputMeter(step_flops=10**12, peak_flops=10**14, samples_per_step=4)
    meter.start()
    assert meter.window(10) is None


@pytest.mark.unit
def test_rates_divide_by_the_measured_window(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = iter([100.0, 102.0, 106.0])
    monkeypatch.setattr("engine.mfu.time.perf_counter", lambda: next(clock))

    meter = ThroughputMeter(step_flops=2 * 10**12, peak_flops=10**13, samples_per_step=8)
    meter.start()  # t=100
    first = meter.window(4)  # t=102, discarded
    assert first is None

    second = meter.window(10)  # t=106, so 10 steps in 4s
    assert second is not None
    # 10 steps x 2 TFLOP = 20 TFLOP over 4s = 5 TFLOP/s against a 10 TFLOP/s peak.
    assert second.tflops_per_s == pytest.approx(5.0)
    assert second.mfu == pytest.approx(0.5)
    assert second.samples_per_s == pytest.approx(20.0)


@pytest.mark.unit
def test_mfu_is_omitted_when_the_peak_is_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = iter([0.0, 1.0, 2.0])
    monkeypatch.setattr("engine.mfu.time.perf_counter", lambda: next(clock))

    meter = ThroughputMeter(step_flops=10**12, peak_flops=None, samples_per_step=2)
    meter.start()
    meter.window(1)
    window = meter.window(1)

    assert window is not None
    assert window.mfu is None
    assert "mfu" not in window.as_metrics()
    # Throughput is still reported: it needs no table.
    assert "tflops_per_s" in window.as_metrics()
    assert "samples_per_s" in window.as_metrics()


@pytest.mark.unit
def test_step_count_not_assumed_to_be_uniform(monkeypatch: pytest.MonkeyPatch) -> None:
    """A run resumed mid-cadence produces a short first window; rates must still be right."""
    clock = iter([0.0, 1.0, 3.0])
    monkeypatch.setattr("engine.mfu.time.perf_counter", lambda: next(clock))

    meter = ThroughputMeter(step_flops=10**12, peak_flops=10**12, samples_per_step=1)
    meter.start()
    meter.window(3)
    window = meter.window(5)  # 5 steps in 2s, not the 10 a fixed log_every would assume

    assert window is not None
    assert window.tflops_per_s == pytest.approx(2.5)


@pytest.mark.unit
def test_measuring_a_step_counts_forward_and_backward() -> None:
    algorithm = _TinyAlgorithm()
    batch = torch.randn(4, 32)

    counted = measure_step_flops(algorithm, batch, _no_autocast)

    # Two 32x32 linears over 4 samples: 2*4*32*32 FLOPs each forward. Backward roughly doubles
    # the second layer and the first, so the total must exceed the forward alone rather than
    # equal it -- the point of counting backward instead of assuming a multiplier.
    forward_only = 2 * (2 * 4 * 32 * 32)
    assert counted > forward_only


@pytest.mark.unit
def test_measuring_leaves_no_gradients_behind() -> None:
    """The probe's backward must not contaminate the first optimizer step."""
    algorithm = _TinyAlgorithm()
    measure_step_flops(algorithm, torch.randn(4, 32), _no_autocast)
    assert all(p.grad is None for p in algorithm.parameters())


@pytest.mark.unit
def test_counting_kernels_restores_backend_selection() -> None:
    """The context manager is only for counting; it must not leave the run on a slower kernel."""
    attention = SelfAttention(32, 4, backend="auto")
    before = (attention.backend, attention._flash4_usable)

    with counting_kernels(attention):
        assert attention.backend == "sdpa"
        assert attention._flash4_usable is False

    assert (attention.backend, attention._flash4_usable) == before


@pytest.mark.unit
def test_counting_kernels_reaches_nested_attention() -> None:
    model = nn.Sequential(SelfAttention(32, 4, backend="auto"), SelfAttention(32, 4))
    with counting_kernels(model):
        assert all(m.backend == "sdpa" for m in model if isinstance(m, SelfAttention))
    assert all(m.backend == "auto" for m in model if isinstance(m, SelfAttention))


@pytest.mark.unit
def test_dinov3_attention_is_also_forced_onto_a_countable_kernel() -> None:
    """The DINOv3 port has its own FA4 switch and no base class in common with the other one.

    Regression test for a real miss: `_counted_kernels` originally entered only
    `layers.common.attention.counting_kernels`, which rewires nothing on a DINOv3 model, so the
    measured FLOPs came back short by the entire attention term with no warning.
    """
    layer = Dinov3SelfAttention(32, num_heads=4)
    # `use_fa4=True` cannot be constructed off Hopper, so set it directly: the context manager's
    # job is to flip the flag whatever set it.
    layer.use_fa4 = True

    with _counted_kernels(nn.Sequential(layer)):
        assert layer.use_fa4 is False
    assert layer.use_fa4 is True


@pytest.mark.unit
def test_every_registered_model_is_covered_by_the_counted_kernel_sweep() -> None:
    """No attention module anywhere may keep a custom kernel while a step is being counted.

    Written against the registry rather than a fixed list so that adding a model, or a third
    attention implementation, fails here instead of silently under-reporting MFU. Asserts the
    property that matters -- nothing is left on an uncountable kernel -- rather than naming the
    classes, which is what let the DINOv3 module slip through in the first place.
    """
    import components  # noqa: F401  (populates the registry by importing every model)
    from models.registry import ModelRegistry

    def uncountable(module: nn.Module) -> list[str]:
        found = []
        for name, layer in module.named_modules():
            if isinstance(layer, SelfAttention) and layer.selected_backend(
                torch.bfloat16, on_cuda=True
            ) != "sdpa":
                found.append(f"{name} (common)")
            if isinstance(layer, Dinov3SelfAttention) and layer.use_fa4:
                found.append(f"{name} (dinov3)")
        return found

    # Tiny configs per registered model. Asserted to cover the registry exactly, so a new model
    # cannot be added without someone deciding whether the sweep reaches its attention.
    tiny: dict[str, dict] = {
        "vit3d": dict(
            img_size=(16, 16, 16), patch_size=(8, 8, 8), in_channels=1,
            embed_dim=32, depth=1, num_heads=2,
        ),
        "muvit3d": dict(
            levels=(1, 4), img_size=(16, 16, 16), patch_size=(8, 8, 8),
            embed_dim=32, depth=1, num_heads=2,
        ),
        "dinov3_vit": dict(
            img_size=32, patch_size=8, in_chans=3, embed_dim=64, depth=1, num_heads=4,
            pos_embed_rope_dtype="fp32",
        ),
        "dinov3_vit3d": dict(
            img_size=16, patch_size=8, in_chans=1, embed_dim=96, depth=1, num_heads=4,
            pos_embed_rope_dtype="fp32",
        ),
    }
    assert set(ModelRegistry.available()) == set(tiny), (
        "the registry changed; add the new model here and confirm _counted_kernels reaches it"
    )

    for name, kwargs in tiny.items():
        model = ModelRegistry.build(name, **kwargs)
        # Force both implementations on, so the sweep has something to switch off. This stands in
        # for a Hopper box, where "auto"/use_fa4 would select the custom kernel for real.
        for layer in model.modules():
            if isinstance(layer, SelfAttention):
                layer._flash4_usable = True
            if isinstance(layer, Dinov3SelfAttention):
                layer.use_fa4 = True

        assert uncountable(model), f"{name}: test setup failed to arm any attention layer"
        with _counted_kernels(model):
            assert not uncountable(model), f"{name}: left on an uncountable kernel while counting"
