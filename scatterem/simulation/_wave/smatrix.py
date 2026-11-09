"""The plane-wave PRISM scattering matrix, advanced by multislice.

:class:`ScatteringMatrix` holds one multislice-propagated plane wave per beam of
the probe-forming aperture (Ophus, Adv. Struct. Chem. Imag. **3**, 13 (2017);
Brown *et al.*, Phys. Rev. Research **1**, 033186 (2019), Eqs. (10)-(11)), so a
probe at any scan position and depth is a linear combination of its rows.
Generic, not EELS-specific.
Moved verbatim (2026-09-27) from ``simulation/eels/forward/prism_eels.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import pi

import torch
from torch import Tensor

from .exit_walk import _SliceAdvanceMixin, _unit_transmission_slices
from .prism_beams import _checked_factor

__all__ = ["ScatteringMatrix"]


class ScatteringMatrix(_SliceAdvanceMixin):
    """Plane-wave scattering matrix :math:`\\mathcal S` advanced by multislice.

    Each row is the multislice-propagated plane wave for one beam in the
    probe-forming aperture.  Advancing the matrix to a given depth and forming a
    probe by linear combination reproduces the elastic wave at that depth for any
    scan position.

    Parameters
    ----------
    probe_q : Tensor
        Probe-forming aperture in reciprocal space ``(Ny, Nx)`` (complex;
        corner origin).  Non-zero pixels define the beams.
    transmissions : Tensor
        Real-space transmission functions ``(NZ, Ny, Nx)``.
    kernel : Tensor
        Fresnel propagator ``(Ny, Nx)`` for one inter-slice distance.
    interpolation_factor : (int, int), optional
        PRISM interpolation factors ``(fy, fx)`` (Ophus 2017); ``(1, 1)``
        keeps every beam and is exact. See :meth:`__init__` for the full
        anti-aliasing discussion.
    """

    def __init__(
        self,
        probe_q: Tensor,
        transmissions: Tensor,
        kernel: Tensor,
        interpolation_factor: Sequence[int] = (1, 1),
    ):
        """
        Parameters
        ----------
        probe_q : Tensor
            Probe-forming aperture in reciprocal space ``(Ny, Nx)`` (complex;
            corner origin).  Non-zero pixels define the beams.
        transmissions : Tensor
            Real-space transmission functions ``(NZ, Ny, Nx)``.
        kernel : Tensor
            Fresnel propagator ``(Ny, Nx)`` for one inter-slice distance.
        interpolation_factor : (int, int)
            PRISM interpolation factors ``(fy, fx)`` (Ophus 2017).  Only beams
            whose pixel indices are multiples of ``fy`` / ``fx`` are kept, so the
            number of multislice-propagated plane waves -- and the cost of
            advancing the matrix and forming probes -- drops by ``fy*fx``.  The
            synthesised probe is then periodic with period ``(Ny/fy, Nx/fx)`` and
            equals the full probe wherever it fits inside that window (the PRISM
            anti-aliasing condition; the localized EELS transition potential
            samples the probe only at the ionized atom, so it is the probe value
            there that must be alias-free).  ``(1, 1)`` keeps every beam and is
            exact.
        """
        device = transmissions.device
        dtype = transmissions.dtype
        self.transmissions = transmissions
        self.kernel = kernel
        self.ny, self.nx = transmissions.shape[-2:]

        fy, fx = _checked_factor(interpolation_factor, self.ny, self.nx)
        self.interpolation_factor = (fy, fx)

        beams = torch.nonzero(probe_q.abs() > 0, as_tuple=False)  # (Nbeams, 2)
        if fy > 1 or fx > 1:
            # Keep every fy-th / fx-th beam.  On the corner-origin grid this is a
            # valid frequency sublattice because fy | Ny and fx | Nx.
            keep = (beams[:, 0] % fy == 0) & (beams[:, 1] % fx == 0)
            beams = beams[keep]
        self.beam_gy = beams[:, 0].to(device)
        self.beam_gx = beams[:, 1].to(device)
        # Fourier subsampling makes the synthesised probe the periodic sum of the
        # full probe scaled by 1/(fy*fx) (the DFT decimation identity), so restore
        # that factor: the probe in the central N/f window then matches the full
        # f=1 probe in amplitude (and the EELS intensity stays on the same scale).
        self.coeffs0 = (probe_q[self.beam_gy, self.beam_gx] * (fy * fx)).to(
            dtype
        )  # (Nbeams,)

        # Initial scattering matrix: one plane wave per beam (real space).
        nbeams = beams.shape[0]
        S_q = torch.zeros((nbeams, self.ny, self.nx), dtype=dtype, device=device)
        idx = torch.arange(nbeams, device=device)
        S_q[idx, self.beam_gy, self.beam_gx] = 1.0
        # In-place beam-chunked inverse FFT. ifft2 acts per beam (last two dims),
        # so transform the one-hot Fourier S in beam-chunks and write each chunk
        # back into its own slice. A single ifft2 over the whole (Nbeams, Ny, Nx)
        # buffer would allocate a second full-size output -- 27 GiB at the 20 nm /
        # 1 M-atom rung -- which doubles the S-matrix footprint and OOMs an 80 GB
        # A100. The chunked form caps the extra to one (beam_chunk, Ny, Nx) tensor.
        beam_chunk = max(1, int(2.5e8 // max(1, self.ny * self.nx)))
        for b0 in range(0, nbeams, beam_chunk):
            sl = slice(b0, b0 + beam_chunk)
            S_q[sl] = torch.fft.ifft2(S_q[sl], dim=(-2, -1))
        self.S = S_q  # (Nbeams, Ny, Nx), now real-space plane waves
        self._slice = 0
        self._unit_slices = _unit_transmission_slices(transmissions)

    def coeffs_at(self, scan_pixels: Tensor) -> Tensor:
        """Per-(scan, beam) synthesis coefficients ``(P, Nbeams)``.

        The probe at a scan position is ``sum_beam coeffs[p, beam] * S[beam]``;
        because the rest of the propagation (the second scattering matrix) is
        **linear**, these *same* coefficients linearly combine any per-column
        quantity -- in particular the exit waves of the transition-scattered
        columns in the dual-scattering-matrix algorithm
        (:func:`~scatterem.simulation.eels.prism_transition_potential`).

        Parameters
        ----------
        scan_pixels : Tensor
            Scan positions ``(P, 2)`` ``(Ry, Rx)`` in pixels.

        Returns
        -------
        Tensor
            Synthesis coefficients ``(P, Nbeams)``.
        """
        ny, nx = self.ny, self.nx
        # Use SIGNED spatial frequencies (fftfreq * N), not the raw corner-origin
        # indices, so a sub-pixel scan shift matches the Fourier shift theorem
        # used by the conventional path (fourier_shift).  For an INTEGER shift the
        # two differ only by exp(2*pi*i * integer * R) = 1, so integer-scan results
        # are unchanged; for a fractional shift the raw index mis-phases the
        # negative-frequency beams and corrupts the probe.
        gy = (((self.beam_gy + ny // 2) % ny) - ny // 2).to(torch.float64)
        gx = (((self.beam_gx + nx // 2) % nx) - nx // 2).to(torch.float64)
        ry = scan_pixels[:, 0].to(torch.float64)
        rx = scan_pixels[:, 1].to(torch.float64)
        # Phase ramp exp(-2πi g·R/N) for each (scan, beam).
        phase = torch.exp(
            -2j * pi * (gy[None, :] * ry[:, None] / ny + gx[None, :] * rx[:, None] / nx)
        ).to(self.S.dtype)
        return phase * self.coeffs0[None, :]  # (P, Nbeams)

    def probe_at_current_plane(self, scan_pixels: Tensor) -> Tensor:
        """Form probes at the current plane for scan positions (in pixels).

        Parameters
        ----------
        scan_pixels : Tensor
            Scan positions ``(P, 2)`` ``(Ry, Rx)`` in pixels.

        Returns
        -------
        Tensor
            Real-space probes ``(P, Ny, Nx)``.
        """
        return torch.einsum("pb,byx->pyx", self.coeffs_at(scan_pixels), self.S)
