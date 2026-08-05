"""Unit tests for the swappable-kernel self-attention module."""

from __future__ import annotations

import math
import sys
import types

import pytest
import torch
import torch.nn as nn

from models import attention as attention_module
from models.attention import BACKENDS, SelfAttention, flash4_status


def _installed_flash4_stub() -> types.ModuleType:
    """Stand in for `flash_attn.cute`, so what flash4_status says about a *device* can be tested
    on a machine where the optional package is absent."""
    stub = types.ModuleType("flash_attn.cute")
    stub.flash_attn_func = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    return stub


@pytest.fixture
def flash4_reported_usable(monkeypatch):
    """Make `flash4_status` report FA4 as usable, so kernel choice is testable off Hopper.

    Patches the module-level status function rather than assigning to `_flash4_usable`, because
    how usability turns into a per-module decision is itself part of what these tests check:
    writing the private attribute would bypass the constructor's own derivation, and a test that
    bypasses the code it is meant to pin cannot fail when that code is wrong.
    """
    monkeypatch.setattr(attention_module, "flash4_status", lambda: (True, ""))


def _reference_attention(module: SelfAttention, x: torch.Tensor) -> torch.Tensor:
    """Textbook multi-head attention, computed independently of the module's own code path."""
    batch, tokens, dim = x.shape
    heads, head_dim = module.num_heads, module.head_dim

    qkv = torch.nn.functional.linear(x, module.qkv.weight, module.qkv.bias)
    qkv = qkv.reshape(batch, tokens, 3, heads, head_dim).permute(2, 0, 3, 1, 4)
    query, key, value = qkv[0], qkv[1], qkv[2]  # each (B, H, N, Dh)

    scores = query @ key.transpose(-2, -1) / math.sqrt(head_dim)
    attended = torch.softmax(scores, dim=-1) @ value  # (B, H, N, Dh)
    attended = attended.transpose(1, 2).reshape(batch, tokens, dim)
    return torch.nn.functional.linear(attended, module.proj.weight, module.proj.bias)


@pytest.mark.unit
def test_matches_textbook_attention():
    # The whole point of hand-rolling this: prove it is still attention.
    torch.manual_seed(0)
    module = SelfAttention(64, 4, backend="sdpa").eval()
    x = torch.randn(3, 11, 64)

    with torch.no_grad():
        assert torch.allclose(module(x), _reference_attention(module, x), atol=1e-5)


@pytest.mark.unit
def test_output_shape_matches_input():
    module = SelfAttention(64, 4)
    assert module(torch.randn(2, 7, 64)).shape == (2, 7, 64)


@pytest.mark.unit
def test_accepts_any_token_count():
    # MAE feeds a masked subset, so the sequence length varies between calls.
    module = SelfAttention(64, 4)
    for tokens in (1, 5, 64):
        assert module(torch.randn(2, tokens, 64)).shape == (2, tokens, 64)


@pytest.mark.unit
def test_gradients_reach_both_projections():
    module = SelfAttention(64, 4)
    module(torch.randn(2, 7, 64)).pow(2).mean().backward()
    assert module.qkv.weight.grad is not None
    assert module.proj.weight.grad is not None


@pytest.mark.unit
def test_head_dim_and_scale():
    module = SelfAttention(64, 4)
    assert module.head_dim == 16
    assert module.scale == pytest.approx(16**-0.5)


@pytest.mark.unit
def test_rejects_dim_not_divisible_by_heads():
    with pytest.raises(ValueError, match="divisible"):
        SelfAttention(65, 4)


@pytest.mark.unit
def test_rejects_an_unknown_backend():
    with pytest.raises(ValueError, match="backend must be one of"):
        SelfAttention(64, 4, backend="flashattention5")


@pytest.mark.unit
def test_sdpa_backend_never_selects_flash4(flash4_reported_usable):
    # Needs FA4 reported usable to mean anything: on hardware that cannot run it, "sdpa" is what
    # every backend resolves to anyway, and the test would pass without the code under test.
    module = SelfAttention(64, 4, backend="sdpa")
    assert module.selected_backend(torch.bfloat16, on_cuda=True) == "sdpa"


@pytest.mark.unit
def test_fp32_never_selects_flash4(flash4_reported_usable):
    # FA4's kernels take half precision or narrower, so precision="fp32" must fall back.
    module = SelfAttention(64, 4, backend="auto")
    assert module.selected_backend(torch.float32, on_cuda=True) == "sdpa"


@pytest.mark.unit
def test_cpu_never_selects_flash4(flash4_reported_usable):
    module = SelfAttention(64, 4, backend="auto")
    assert module.selected_backend(torch.bfloat16, on_cuda=False) == "sdpa"


@pytest.mark.unit
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_half_precision_on_cuda_selects_flash4_when_usable(dtype, flash4_reported_usable):
    module = SelfAttention(64, 4, backend="auto")
    assert module.selected_backend(dtype, on_cuda=True) == "flash4"


@pytest.mark.unit
def test_status_reports_a_missing_install(monkeypatch):
    # Every reason flash4_status can give is asserted here, because only one of the three can be
    # reached on any given machine and each is what an explicit flash4 request has to explain.
    monkeypatch.setitem(sys.modules, "flash_attn.cute", None)  # makes the import raise
    assert flash4_status() == (False, "flash-attn-4 is not installed")


@pytest.mark.unit
def test_status_reports_a_pre_hopper_gpu(monkeypatch):
    monkeypatch.setitem(sys.modules, "flash_attn.cute", _installed_flash4_stub())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 0))
    assert flash4_status() == (False, "compute capability 8.0 is below 9.0 (Hopper)")


@pytest.mark.unit
def test_status_accepts_hopper(monkeypatch):
    monkeypatch.setitem(sys.modules, "flash_attn.cute", _installed_flash4_stub())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
    assert flash4_status() == (True, "")


@pytest.mark.unit
def test_status_reports_a_missing_cuda_device(monkeypatch):
    monkeypatch.setitem(sys.modules, "flash_attn.cute", _installed_flash4_stub())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert flash4_status() == (False, "no CUDA device available")


@pytest.mark.unit
def test_auto_backend_works_without_flash4():
    # The common case on any machine without a Hopper GPU: it must simply run.
    usable, reason = flash4_status()
    if not usable:
        assert reason  # a reason is always given, so an explicit request can explain itself
    assert SelfAttention(64, 4, backend="auto")(torch.randn(2, 7, 64)).shape == (2, 7, 64)


@pytest.mark.unit
def test_explicit_flash4_fails_loudly_when_unusable(monkeypatch):
    # Silently downgrading would make a benchmark read as "the kernel did not help". The status
    # is forced rather than read from the machine so this asserts the same thing on every host,
    # and so the reason can be checked for being carried through to the caller.
    monkeypatch.setattr(attention_module, "flash4_status", lambda: (False, "made-up reason"))
    with pytest.raises(ValueError, match="was requested but is unusable: made-up reason"):
        SelfAttention(64, 4, backend="flash4")


@pytest.mark.unit
def test_explicit_flash4_rejects_inputs_it_cannot_run_on(flash4_reported_usable):
    # Reachable on a CPU node only because the status is forced: construction has to succeed
    # before forward can refuse the inputs. On real hardware this is the fp32 case.
    module = SelfAttention(64, 4, backend="flash4")
    with pytest.raises(ValueError, match="cannot run on"):
        module(torch.randn(2, 7, 64))


@pytest.mark.unit
def test_backends_tuple_is_the_accepted_set(flash4_reported_usable):
    # Every entry is checked on every host; without the forced status this silently skipped
    # "flash4" anywhere without a Hopper GPU, i.e. everywhere the suite normally runs.
    for backend in BACKENDS:
        assert SelfAttention(64, 4, backend=backend).backend == backend


@pytest.mark.unit
def test_parameter_shapes_match_a_packed_qkv_attention():
    # Same parameter budget as the nn.MultiheadAttention this replaced, under different names:
    # a checkpoint from before the swap cannot be loaded after it.
    module = SelfAttention(64, 4)
    named = dict(module.named_parameters())
    assert named["qkv.weight"].shape == (192, 64)
    assert named["qkv.bias"].shape == (192,)
    assert named["proj.weight"].shape == (64, 64)
    assert sorted(named) == ["proj.bias", "proj.weight", "qkv.bias", "qkv.weight"]


@pytest.mark.unit
def test_no_module_uses_torch_multihead_attention():
    # A regression guard: the point of this module is that the attention call is ours to swap.
    module = SelfAttention(64, 4)
    assert not any(isinstance(m, nn.MultiheadAttention) for m in module.modules())
