"""Pins the peak-FLOPS table's lookup behaviour.

The table is the denominator of every reported MFU, and the two ways it can be quietly wrong are
both about matching: a variant falling through to its family (an H100 PCIe read as an SXM is a
31% error) and an untabulated device being guessed at instead of reported. Both are asserted here.
"""

from __future__ import annotations

import pytest

from utils.hardware_flops import known_devices, lookup_device, peak_flops

_T = 1.0e12


@pytest.mark.unit
@pytest.mark.parametrize(
    ("device_name", "expected_label"),
    [
        # The string an H100 SXM actually reports, observed on the gpu_h100 queue. It does not
        # contain "SXM", which is the whole reason the generic H100 entry has to mean SXM.
        ("NVIDIA H100 80GB HBM3", "NVIDIA H100 SXM"),
        ("NVIDIA H100 PCIe", "NVIDIA H100 PCIe"),
        ("NVIDIA H100 NVL", "NVIDIA H100 NVL"),
        ("NVIDIA H200", "NVIDIA H200 SXM"),
        ("NVIDIA A100-SXM4-80GB", "NVIDIA A100"),
        ("NVIDIA A100 80GB PCIe", "NVIDIA A100"),
        ("NVIDIA L4", "NVIDIA L4"),
        ("NVIDIA L40S", "NVIDIA L40S"),
    ],
)
def test_device_names_map_to_the_right_entry(device_name: str, expected_label: str) -> None:
    peak = lookup_device(device_name)
    assert peak is not None
    assert peak.label == expected_label


@pytest.mark.unit
def test_variants_are_matched_before_their_family() -> None:
    """"NVIDIA H100 PCIe" contains "H100", so ordering is what keeps them apart."""
    sxm = peak_flops("NVIDIA H100 80GB HBM3", "bf16")[0]
    pcie = peak_flops("NVIDIA H100 PCIe", "bf16")[0]
    assert sxm == pytest.approx(989.4 * _T)
    assert pcie == pytest.approx(756.5 * _T)
    assert sxm != pcie


@pytest.mark.unit
def test_bf16_peaks_are_dense_not_sparse() -> None:
    """Vendor headline figures assume 2:4 sparsity and are exactly double these."""
    assert peak_flops("NVIDIA A100-SXM4-80GB", "bf16")[0] == pytest.approx(312 * _T)
    assert peak_flops("NVIDIA H200", "bf16")[0] == pytest.approx(989.4 * _T)


@pytest.mark.unit
def test_unknown_device_is_reported_not_guessed() -> None:
    value, reason = peak_flops("NVIDIA GeForce RTX 4090", "bf16")
    assert value is None
    assert "peak_tflops" in reason


@pytest.mark.unit
@pytest.mark.parametrize("device_name", ["Tesla T4", "Tesla V100-SXM2-16GB"])
def test_pre_ampere_devices_have_no_bf16_peak(device_name: str) -> None:
    """Turing and Volta have fp16 tensor cores but no bf16 ones, so no rated bf16 peak exists."""
    value, reason = peak_flops(device_name, "bf16")
    assert value is None
    assert "bf16" in reason


@pytest.mark.unit
def test_fp32_peak_follows_the_tf32_flag() -> None:
    """Whether fp32 matmuls reach the tensor cores is a runtime flag, not a property of the GPU."""
    with_tf32, _ = peak_flops("NVIDIA A100-SXM4-80GB", "fp32", allow_tf32=True)
    without, _ = peak_flops("NVIDIA A100-SXM4-80GB", "fp32", allow_tf32=False)
    assert with_tf32 is not None and without is not None
    assert with_tf32 == pytest.approx(156 * _T)
    assert without == pytest.approx(19.5 * _T)
    assert with_tf32 > without


@pytest.mark.unit
def test_unknown_precision_is_rejected() -> None:
    value, reason = peak_flops("NVIDIA H100 80GB HBM3", "int8")
    assert value is None
    assert "precision" in reason


@pytest.mark.unit
def test_known_devices_is_non_empty_and_labelled() -> None:
    labels = known_devices()
    assert labels
    assert all(label.startswith("NVIDIA") for label in labels)
