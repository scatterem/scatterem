"""The float64 total of a cube that ``Dataset4DStem`` records as its dose.

``dataset_4dstem.py`` is frozen size debt, so the sum lives here.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["float64_slice_total", "host_total"]


def float64_slice_total(array: Tensor) -> float:
    """Sum ``array`` in float64, one partial sum per outer slice, on the host.

    Each outer slice is summed in float64 on its device, matching the
    ``total_intensity`` property's own precision strategy. The partial sums
    cross to the host in ONE ``tolist()``, then add in slice order. A
    ``float()`` per slice was one host sync per scan row: 306 on a large
    padded cube, 6 ms of a 90 ms ``pad_and_taper_realspace``.
    Same kernels, same partial sums, same host order, so the total is
    bit-identical to the per-slice loop. The ``+=`` loop is deliberate:
    Python 3.12's ``sum()`` adds floats with compensation, a different result.

    Parameters
    ----------
    array : Tensor
        The cube; its first axis is the outer slice.

    Returns
    -------
    float
        The total, ``0.0`` for an empty first axis.
    """
    partial = [array[i].to(torch.float64).sum() for i in range(array.shape[0])]
    return host_total(torch.stack(partial)) if partial else 0.0


def host_total(partial: Tensor) -> float:
    """Add the float64 partial sums on the host, in order, after ONE read.

    Parameters
    ----------
    partial : Tensor
        The ``(n,)`` float64 partial sums, on any device.

    Returns
    -------
    float
        Their total, added left to right with ``+=``.
    """
    total = 0.0
    for value in partial.tolist():
        total += value
    return total
