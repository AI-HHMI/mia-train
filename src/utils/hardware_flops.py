"""Peak arithmetic throughput per GPU, the denominator of model FLOPs utilization.

MFU is FLOPs actually executed per second divided by what the device could execute per second.
The numerator is measurable; the denominator is not. CUDA reports clock rates and SM counts but
not tensor-core throughput, and deriving it would mean hardcoding cores-per-SM and ops-per-cycle
per architecture — the same table, one indirection further from the vendor's published number.
So this is a table, keyed on `torch.cuda.get_device_name()`.

Three things about the numbers, because each is a way to be silently wrong by 2x:

- **They are dense, not sparse.** NVIDIA's marketing figures assume 2:4 structured sparsity and
  are exactly double. Nothing here prunes weights, so quoting the sparse number would halve every
  reported MFU.
- **Form factor matters more than the model name.** An H100 SXM is 989 TFLOP/s bf16 and an H100
  PCIe is 756 — a 31% difference behind the same "H100". The entries below are therefore matched
  most-specific-first, and `deploy/lsf/README.md` targets `gpu_h100`/`gpu_h200`, which are 8-GPU
  HGX baseboards, i.e. SXM.
- **fp32 is not one number.** With TF32 matmuls enabled the tensor cores run at half the bf16
  rate; without, fp32 falls back to the CUDA cores at roughly a fifteenth. PyTorch defaults
  `torch.backends.cuda.matmul.allow_tf32` to False, so the honest fp32 peak depends on a runtime
  flag rather than on the hardware alone, and `peak_flops` reads it.

An unknown device returns None and the caller skips the metric. That is deliberate: a plausible
wrong denominator produces a plausible wrong MFU, and an MFU nobody can tell is wrong is worse
than no MFU at all.

Sources: NVIDIA H100, H200, A100, L4, L40S and T4 datasheets, dense (non-sparse) tensor-core
figures at rated boost clocks.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DevicePeak:
    """Peak dense throughput for one GPU, in FLOP/s.

    `fp32` is the CUDA-core rate and `tf32` the tensor-core rate that `allow_tf32` selects
    between; `bf16` covers both bf16 and fp16, which run at the same rate on every device here.
    """

    label: str
    bf16: float
    tf32: float
    fp32: float


_T = 1.0e12

# Matched in order, first hit wins, so a variant must precede the family it belongs to:
# "NVIDIA H100 PCIe" also contains "H100". Match keys are upper-cased substrings of
# `torch.cuda.get_device_name()`.
_DEVICES: tuple[tuple[tuple[str, ...], DevicePeak], ...] = (
    # Blackwell. "B200" before "B100"; both are SXM-only parts.
    (("B200",), DevicePeak("NVIDIA B200", 2250 * _T, 1100 * _T, 80 * _T)),
    # Hopper. H200 shares the H100 SXM compute die — its advantage is 141GB of HBM3e, not FLOPs,
    # which the repo's own benchmark independently found (deploy/lsf/README.md: "H100 and H200
    # are indistinguishable here, within 2% at every size").
    (("H200",), DevicePeak("NVIDIA H200 SXM", 989.4 * _T, 494.7 * _T, 67 * _T)),
    (("H100 NVL",), DevicePeak("NVIDIA H100 NVL", 835.5 * _T, 417.7 * _T, 60 * _T)),
    (("H100 PCIE",), DevicePeak("NVIDIA H100 PCIe", 756.5 * _T, 378.2 * _T, 51 * _T)),
    # "NVIDIA H100 80GB HBM3" is the SXM part; it does not spell out "SXM".
    (("H100",), DevicePeak("NVIDIA H100 SXM", 989.4 * _T, 494.7 * _T, 67 * _T)),
    # Ampere. SXM4 and PCIe A100s differ in power and clocks but are both rated 312 dense bf16.
    (("A100",), DevicePeak("NVIDIA A100", 312 * _T, 156 * _T, 19.5 * _T)),
    # Ada Lovelace.
    (("L40S",), DevicePeak("NVIDIA L40S", 362 * _T, 183 * _T, 91.6 * _T)),
    (("L40",), DevicePeak("NVIDIA L40", 181.05 * _T, 90.5 * _T, 90.5 * _T)),
    (("L4",), DevicePeak("NVIDIA L4", 121 * _T, 60.5 * _T, 30.3 * _T)),
    # Ampere workstation/consumer parts that turn up in dev boxes.
    (("A40",), DevicePeak("NVIDIA A40", 149.7 * _T, 74.8 * _T, 37.4 * _T)),
    (("A6000",), DevicePeak("NVIDIA RTX A6000", 154.8 * _T, 77.4 * _T, 38.7 * _T)),
)

# Turing and Volta predate bf16 tensor cores: T4 and V100 have fp16 tensor cores but no bf16 and
# no TF32. `precision = "bf16"` on one of them runs, slowly and unaccelerated, at a rate no
# datasheet quotes — so they are recorded as known-but-unsupported rather than omitted, and the
# reason says so instead of the lookup silently missing.
_NO_BF16 = (("T4",), ("V100",))


def _normalize(device_name: str) -> str:
    return device_name.upper().replace("-", " ")


def lookup_device(device_name: str) -> DevicePeak | None:
    """Peak throughput for a `torch.cuda.get_device_name()` string, or None if untabulated."""
    normalized = _normalize(device_name)
    for keys, peak in _DEVICES:
        if any(key in normalized for key in keys):
            return peak
    return None


def peak_flops(
    device_name: str,
    precision: str,
    allow_tf32: bool | None = None,
) -> tuple[float | None, str]:
    """Peak FLOP/s for this device at this training precision, and why it is unavailable.

    Returns `(None, reason)` rather than raising or guessing, because a run on an untabulated GPU
    should lose one metric, not fail. `precision` takes the same values as `[trainer].precision`.

    `allow_tf32` defaults to reading `torch.backends.cuda.matmul.allow_tf32`, which is what
    actually decides whether fp32 matmuls reach the tensor cores.
    """
    normalized = _normalize(device_name)
    if precision == "bf16" and any(
        key in normalized for keys in _NO_BF16 for key in keys
    ):
        return None, f"{device_name} has no bf16 tensor cores, so it has no rated bf16 peak"

    peak = lookup_device(device_name)
    if peak is None:
        return None, (
            f"{device_name!r} is not in the peak-FLOPS table "
            f"(src/utils/hardware_flops.py); set [trainer].peak_tflops to enable MFU"
        )

    if precision == "bf16":
        return peak.bf16, ""
    if precision == "fp32":
        if allow_tf32 is None:
            allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        return (peak.tf32 if allow_tf32 else peak.fp32), ""
    return None, f"unknown precision {precision!r}"


def known_devices() -> tuple[str, ...]:
    """Labels of every tabulated device, for error messages and documentation."""
    return tuple(peak.label for _, peak in _DEVICES)
