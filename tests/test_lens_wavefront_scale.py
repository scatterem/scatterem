"""Scale test of the lens wavefront on CUDA (CONTRACT section 5, SPEC 4.F).

The test runs a batch of 64 lens vectors on a 1024 x 1024 float32 grid. It
checks that no call synchronises the host and that the peak memory of the
forward pass stays within the ``(batch + 3 * 12)`` grid budget.
"""

from __future__ import annotations

import pytest
import torch

from scatterem.physics import optics as O

MIB = 2**20


@pytest.mark.slow
@pytest.mark.parametrize("func", [O.lens_wavefront, O.lens_wavefront_slope])
def test_batch_64_on_1024_grid(require_cuda, func) -> None:
    dev = require_cuda
    f = torch.fft.fftfreq(1024, d=0.05, device=dev)
    fy, fx = f[:, None], f[None, :]
    lens = torch.randn(64, 12, device=dev)
    components = 2 if func is O.lens_wavefront_slope else 1
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(dev)
    base = torch.cuda.memory_allocated(dev)
    torch.cuda.set_sync_debug_mode("error")
    try:
        out = func(lens, fy, fx, lambda_a=0.025)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated(dev) - base
    budget = (64 * components + 3 * 12) * 4 * MIB * 2
    assert out.shape[:3] == (64, 1024, 1024)
    assert peak <= budget, f"peak {peak / MIB:.0f} MiB > budget {budget / MIB:.0f} MiB"
