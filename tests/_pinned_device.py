"""Pin the CUDA device properties a size gate reads (for the L2-graph tests).

The L2-block graphs size their blocks from ``L2_cache_size`` and the PRISM ramp
fold gates on ``multi_processor_count``, so a test that asserts a block size or
a gate decision is written for one card. The tests were measured on an RTX A6000
(6 MiB L2, 84 SMs); the CI laptop GPU is an RTX 3050 (1 MiB, 16 SMs). Pinning
the two numbers makes every decision the tests assert the same on any CUDA
device, while the kernels and the bitwise comparisons still run on the real one.
"""

from __future__ import annotations

import torch

A6000_L2 = 6 * 2**20
A6000_SMS = 84


class _Props:
    def __init__(self, real, overrides):
        self._real, self._overrides = real, overrides

    def __getattr__(self, name):
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._real, name)


def pin_device_properties(monkeypatch, l2=A6000_L2, n_sm=A6000_SMS):
    """Report ``l2`` bytes of L2 and ``n_sm`` SMs from ``get_device_properties``."""
    real = torch.cuda.get_device_properties
    overrides = {"L2_cache_size": l2, "multi_processor_count": n_sm}
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device=None: _Props(real(device), overrides),
    )
