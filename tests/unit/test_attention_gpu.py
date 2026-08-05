from __future__ import annotations

import pytest
import torch

from models.attention import SelfAttention, flash4_status

pytestmark = [
    pytest.mark.gpu_dist,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device"),
]


def _requires_flash4() -> None:
    usable, reason = flash4_status()
    if not usable:
        pytest.skip(f"flash4 unusable here: {reason}")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_flash4_and_sdpa_agree(dtype):
    _requires_flash4()
    torch.manual_seed(0)
    flash = SelfAttention(384, 6, backend="flash4").cuda().to(dtype)
    sdpa = SelfAttention(384, 6, backend="sdpa").cuda().to(dtype)
    sdpa.load_state_dict(flash.state_dict())

    x = torch.randn(2, 128, 384, device="cuda", dtype=dtype)
    with torch.no_grad():
        gap = (flash(x).float() - sdpa(x).float()).abs().max().item()
    # Half precision, and the kernels accumulate in different orders.
    assert gap < 5e-2, f"backends disagree by {gap}"


def test_flash4_produces_gradients():
    _requires_flash4()
    module = SelfAttention(384, 6, backend="flash4").cuda().to(torch.bfloat16)
    x = torch.randn(2, 128, 384, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    module(x).float().pow(2).mean().backward()

    assert x.grad is not None
    assert module.qkv.weight.grad is not None
    assert torch.isfinite(module.qkv.weight.grad).all()


def test_flash4_gradients_match_sdpa():
    _requires_flash4()
    torch.manual_seed(0)
    flash = SelfAttention(384, 6, backend="flash4").cuda().to(torch.bfloat16)
    sdpa = SelfAttention(384, 6, backend="sdpa").cuda().to(torch.bfloat16)
    sdpa.load_state_dict(flash.state_dict())

    x = torch.randn(2, 128, 384, device="cuda", dtype=torch.bfloat16)
    xf, xs = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
    flash(xf).float().pow(2).mean().backward()
    sdpa(xs).float().pow(2).mean().backward()

    assert (xf.grad.float() - xs.grad.float()).abs().max().item() < 5e-2


def test_auto_selects_flash4_under_autocast():
    # Asserts the CHOICE, not just the output: the two kernels can agree to the bit on small
    # inputs, so identical results would not tell us which one ran.
    _requires_flash4()
    module = SelfAttention(384, 6, backend="auto").cuda()
    assert module.selected_backend(torch.bfloat16, on_cuda=True) == "flash4"
    with torch.autocast("cuda", dtype=torch.bfloat16):
        assert module(torch.randn(2, 64, 384, device="cuda")).dtype == torch.bfloat16


def test_auto_falls_back_to_sdpa_for_fp32():
    _requires_flash4()
    module = SelfAttention(384, 6, backend="auto").cuda()
    assert module.selected_backend(torch.float32, on_cuda=True) == "sdpa"
    assert module(torch.randn(2, 64, 384, device="cuda")).dtype == torch.float32


def test_explicit_flash4_rejects_fp32_inputs():
    _requires_flash4()
    module = SelfAttention(384, 6, backend="flash4").cuda()
    with pytest.raises(ValueError, match="cannot run on"):
        module(torch.randn(2, 64, 384, device="cuda"))
