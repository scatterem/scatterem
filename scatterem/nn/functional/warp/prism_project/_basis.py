"""The beamlet basis of a PRISM projection, kept as the factors it is made of.

``basis_fft[m, p, qy, qx] = sum_b w[p, b] Psi_m(h_b)`` scattered onto the
``(M1, M2)`` beam grid is ``(Nmodes, Bp, M1, M2)``: for pure PRISM at f = 1 that
is the size of the S-matrix itself (60 GB at 27,069 beams on 528^2), although it
holds ONE non-zero per parent.  The projection only ever needs a block of
parents at a time, so the factors -- the probe coefficient per aperture beam, the
``(Bp, B)`` weights and each beam's pixel -- are what is passed around, and a
dense block is scattered when it is used.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["BeamletBasis", "as_route_basis"]


class BeamletBasis:
    """``(Nmodes, Bp, M1, M2)`` Fourier-space beamlet basis in factored form.

    Parameters
    ----------
    psi : Tensor
        ``(Nmodes, B)`` probe coefficient at each aperture beam.
    weights : Tensor
        ``(Bp, B)`` parent-to-beam weights (the identity for pure PRISM).
    pixel : Tensor
        ``(B,)`` flat index ``qy * M2 + qx`` of each beam on the beam grid.
    grid : (int, int)
        The beam grid ``(M1, M2)``.
    """

    def __init__(self, psi: Tensor, weights: Tensor, pixel: Tensor, grid):
        self.psi, self.weights, self.pixel = psi, weights, pixel
        self.grid = (int(grid[0]), int(grid[1]))

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return (int(self.psi.shape[0]), int(self.weights.shape[0]), *self.grid)

    @property
    def dtype(self) -> torch.dtype:
        return self.psi.dtype

    @property
    def device(self) -> torch.device:
        return self.psi.device

    @property
    def requires_grad(self) -> bool:
        return bool(self.psi.requires_grad or self.weights.requires_grad)

    def element_size(self) -> int:
        return self.psi.element_size()

    def block(self, p0: int, p1: int) -> Tensor:
        """Dense ``(Nmodes, p1 - p0, M1, M2)`` basis of parents ``p0 .. p1``.

        ACCUMULATES: at f > 1 distinct beams round onto the same beam-grid
        pixel and must sum (an assignment with duplicate indices races on CUDA).
        """
        nmodes, _, m1, m2 = self.shape
        out = torch.zeros(
            (nmodes, p1 - p0, m1 * m2), dtype=self.dtype, device=self.device
        )
        weighted = torch.einsum("pb,mb->mpb", self.weights[p0:p1], self.psi)
        out.index_add_(2, self.pixel, weighted)
        return out.view(nmodes, p1 - p0, m1, m2)

    def dense(self) -> Tensor:
        """The whole basis, for the routes that need it as one tensor."""
        return self.block(0, self.shape[1])


def as_route_basis(basis, use_warp: bool, *graph_inputs: Tensor):
    """``basis`` in the form the chosen projection route takes.

    The forward-only torch route scatters a factored basis a block of parents
    at a time; the Warp kernels and a live graph are written for ONE tensor, so
    they get the dense one.  A tensor is returned as it is.
    """
    if isinstance(basis, BeamletBasis) and (
        use_warp
        or (
            torch.is_grad_enabled()
            and (basis.requires_grad or any(t.requires_grad for t in graph_inputs))
        )
    ):
        return basis.dense()
    return basis
