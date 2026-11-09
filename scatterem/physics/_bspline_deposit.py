"""Cubic B-spline deposit of point weights on interlaced periodic grids.

Literature: the charge assignment with the cardinal cubic B-spline, its
Fourier transform ``sinc^4``, the removal of the assignment smoothing in
Fourier space and the interlaced grid that removes the odd aliases follow
Hockney and Eastwood (1988), *Computer Simulation Using Particles*, ch. 5
and 7. Written for scatterem from this literature.

Units: grid coordinates are in grid steps [dimensionless]; weights carry
the unit of the caller.
"""

from __future__ import annotations

import logging
import math

import torch

_log = logging.getLogger(__name__)


def cubic_weights(t: torch.Tensor):
    """Return the base index and the 4 cubic B-spline weights per point.

    For the grid coordinate ``t`` with ``u = t - floor(t)``, the points
    ``floor(t) - 1 .. floor(t) + 2`` get ``M4(t - n)``::

        (1 - u)^3 / 6, (3u^3 - 6u^2 + 4) / 6, (-3u^3 + 3u^2 + 3u + 1) / 6, u^3 / 6

    The weights add up to 1 and are ``C^2`` in ``t``.
    """
    base = torch.floor(t.detach())
    u = t - base
    u2 = u * u
    u3 = u2 * u
    w = torch.stack(
        [
            (1.0 - u) ** 3 / 6.0,
            (3.0 * u3 - 6.0 * u2 + 4.0) / 6.0,
            (-3.0 * u3 + 3.0 * u2 + 3.0 * u + 1.0) / 6.0,
            u3 / 6.0,
        ],
        dim=-1,
    )
    return base.to(torch.int64) - 1, w


def deposit(ty, tx, weight, points, dtype) -> torch.Tensor:
    """Return the grid ``(Ny, Nx)`` of cubic B-spline deposits.

    Parameters
    ----------
    ty, tx : torch.Tensor
        float64 ``(P,)`` grid coordinates [grid steps]; the grid wraps.
    weight : torch.Tensor
        float64 ``(P,)`` weight per point.
    points : tuple of int
        ``(Ny, Nx)``.
    dtype : torch.dtype
        Real dtype of the grid.

    Notes
    -----
    The sum is deterministic on every device: a stable sort of the flat
    target indices, then a segment sum, then one write per distinct index.
    """
    ny, nx = points
    by, wy = cubic_weights(ty)
    bx, wx = cubic_weights(tx)
    steps = torch.arange(4, device=ty.device)
    iy = torch.remainder(by[:, None] + steps, ny)
    ix = torch.remainder(bx[:, None] + steps, nx)
    flat = (iy[:, :, None] * nx + ix[:, None, :]).reshape(-1)
    values = (weight[:, None, None] * wy[:, :, None] * wx[:, None, :]).reshape(-1)
    values = values.to(dtype)
    order = torch.argsort(flat, stable=True)
    target = flat[order]
    distinct, lengths = torch.unique_consecutive(target, return_counts=True)
    sums = torch.segment_reduce(values[order], "sum", lengths=lengths)
    grid = torch.zeros(ny * nx, dtype=dtype, device=ty.device)
    return grid.index_add(0, distinct, sums).reshape(ny, nx)


def _signed(n: int, count: int, device) -> torch.Tensor:
    j = torch.arange(count, device=device, dtype=torch.float64)
    return torch.where(j >= (n + 1) // 2, j - n, j)


def layout_factors(points, dtype, device):
    """Return the factors ``a`` and ``b`` of ``D = a R_A + b R_B``.

    ``a = 1 / (2 W)`` and ``b = P / (2 W)`` on the half layout, with
    ``W = sinc(j_y / Ny)^4 sinc(j_x / Nx)^4`` and the interlacing phase
    ``P = exp(-i pi (j_y / Ny + j_x / Nx))`` for signed FFT-order ``j``.

    On the Nyquist row and column (even ``N``), ``P`` takes the value that
    gives the Hermitian part of the full-layout spectrum: 0 on the row and
    the column, and ``-1`` at the corner ``(-Ny/2, -Nx/2)``. ``irfft2`` then
    returns exactly ``Re(ifft2(full spectrum))``, the band-limited point
    value of the series over the FFT-order frequencies.
    """
    ny, nx = points
    half = nx // 2 + 1
    jy = _signed(ny, ny, device)
    jx = torch.arange(half, device=device, dtype=torch.float64)
    jx = torch.where(jx >= (nx + 1) // 2, jx - nx, jx)
    wy = torch.sinc(jy / ny) ** 4
    wx = torch.sinc(jx / nx) ** 4
    inv = 1.0 / (2.0 * wy[:, None] * wx[None, :])
    py = torch.polar(torch.ones_like(jy), -math.pi * jy / ny)
    px = torch.polar(torch.ones_like(jx), -math.pi * jx / nx)
    if ny % 2 == 0:
        py[ny // 2] = 0.0
    if nx % 2 == 0:
        px[half - 1] = 0.0
    phase = py[:, None] * px[None, :]
    if ny % 2 == 0 and nx % 2 == 0:
        phase[ny // 2, half - 1] = -1.0
    complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
    return inv.to(dtype), (phase * inv).to(complex_dtype)


def layout_spectrum(ty, tx, weight, points, dtype, factors) -> torch.Tensor:
    """Return the smoothing-free spectrum ``D`` of weighted points.

    ``D(q) = [R_A(q) + P(q) R_B(q)] / (2 W(q))`` on the ``rfft2`` half layout,
    with ``R_A`` the rfft2 of the deposit at grid points ``n h`` and ``R_B``
    of the deposit at ``n h + h / 2`` (both axes shifted). ``D(q)``
    approximates ``sum_p w_p exp(-2 pi i q . r_p)``.
    """
    inv, mixed = factors
    grid_a = deposit(ty, tx, weight, points, dtype)
    spec = torch.fft.rfft2(grid_a) * inv
    del grid_a
    grid_b = deposit(ty - 0.5, tx - 0.5, weight, points, dtype)
    return spec + torch.fft.rfft2(grid_b) * mixed
