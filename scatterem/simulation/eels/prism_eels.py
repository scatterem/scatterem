"""Scattering-matrix (PRISM) STEM-EELS.

The PRISM algorithm of Brown *et al.*, Phys. Rev. Research **1**, 033186 (2019)
(Eqs. (10)-(11)) accelerates STEM-EELS by precomputing a scattering matrix
:math:`\\mathcal S_1` that propagates the probe-forming plane-wave components to
the plane of an inelastic transition.  Because that matrix is shared by *all*
scan positions, the cost of forming the probe at the ionization plane no longer
scales with the number of probe positions -- the key to the paper's near-linear
scaling.

For an **energy-filtered 4D** output the full exit diffraction pattern is
required, so the second scattering matrix :math:`\\mathcal S_2` cannot be reduced
to a small detector aperture; here the inelastically-scattered wave is propagated
to the exit surface by ordinary multislice.  The :math:`\\mathcal S_1` reuse
across scan positions is the part that makes this cheaper than the conventional
algorithm when many probe positions are scanned.

This module reuses the propagation primitives from
:mod:`scatterem.simulation.eels.multislice_eels` and shares the
:class:`~scatterem.simulation.eels.transition_potentials.TransitionPotentials`
data structure with the conventional path, so both produce identical physics.
"""

from __future__ import annotations

from math import pi
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from .multislice_eels import (
    _fourier_shift,
    _propagate,
    _slice_exit_wave,
    _unit_transmission_slices,
    propagator_kernel,
)
from .transition_potentials import TransitionPotentials

__all__ = [
    "ScatteringMatrix",
    "PartitionedScatteringMatrix",
    "prism_transition_potential",
    "prism_transition_potential_hybrid",
]


def _factor_pair(f) -> tuple:
    """Normalise an interpolation factor to a ``(fy, fx)`` int pair."""
    if isinstance(f, int):
        return (f, f)
    return (int(f[0]), int(f[1]))


class ScatteringMatrix:
    """Plane-wave scattering matrix :math:`\\mathcal S` advanced by multislice.

    Each row is the multislice-propagated plane wave for one beam in the
    probe-forming aperture.  Advancing the matrix to a given depth and forming a
    probe by linear combination reproduces the elastic wave at that depth for any
    scan position.
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

        fy, fx = int(interpolation_factor[0]), int(interpolation_factor[1])
        if fy < 1 or fx < 1:
            raise ValueError(f"interpolation_factor must be >= 1, got ({fy}, {fx})")
        if self.ny % fy or self.nx % fx:
            raise ValueError(
                f"interpolation_factor {(fy, fx)} must divide the grid "
                f"{(self.ny, self.nx)} (subsampling needs a sublattice)"
            )
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

    def advance_to(self, target_slice: int) -> None:
        """Propagate the scattering matrix forward to ``target_slice``.

        A slice whose transmission function is exactly ``1 + 0j`` (see
        :func:`_unit_transmission_slices`) has an identity multiply, so it is
        skipped -- bit-identically, and the multiply is a full pass over the
        ``(Nbeams, Ny, Nx)`` matrix.  The propagation is *not* skipped.
        """
        for j in range(self._slice, target_slice):
            if not self._unit_slices[j]:
                self.S = self.S * self.transmissions[j]
            self.S = _propagate(self.S, self.kernel)
        self._slice = target_slice

    def coeffs_at(self, scan_pixels: Tensor) -> Tensor:
        """Per-(scan, beam) synthesis coefficients ``(P, Nbeams)``.

        The probe at a scan position is ``sum_beam coeffs[p, beam] * S[beam]``;
        because the rest of the propagation (the second scattering matrix) is
        **linear**, these *same* coefficients linearly combine any per-column
        quantity -- in particular the exit waves of the transition-scattered
        columns in the dual-scattering-matrix algorithm
        (:func:`prism_transition_potential`).

        Parameters
        ----------
        scan_pixels : Tensor
            Scan positions ``(P, 2)`` ``(Ry, Rx)`` in pixels.
        """
        ny, nx = self.ny, self.nx
        # Use SIGNED spatial frequencies (fftfreq * N), not the raw corner-origin
        # indices, so a sub-pixel scan shift matches the Fourier shift theorem
        # used by the conventional path (_fourier_shift).  For an INTEGER shift the
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


def _select_parent_beams(
    all_signed: np.ndarray, n_radial: int, n_angular: int
) -> np.ndarray:
    """Hex-ring parent-beam selection (mirrors ``SMeta.make_beamlet_meta``).

    ``all_signed`` are the aperture beams in SIGNED spatial frequency (pixels).
    Samples the aperture on the DC beam plus ``n_radial`` concentric hex rings,
    snaps each sample to the nearest aperture beam, and deduplicates.  Returns
    indices into ``all_signed``.
    """
    radius = float(np.linalg.norm(all_signed, axis=1).max())
    a_off = np.pi / n_angular
    samples = [[0.0, 0.0]]
    for i, r in enumerate(np.linspace(0.0, radius, n_radial + 1)[1:]):
        n_ang = n_angular * (1 + i)
        for a in np.linspace(-np.pi, np.pi, n_ang, endpoint=False):
            samples.append([r * np.sin(a + a_off * i), r * np.cos(a + a_off * i)])
    samples = np.asarray(samples, dtype=np.float64)
    d = np.linalg.norm(samples[:, None, :] - all_signed[None, :, :], axis=2)
    return np.unique(np.argmin(d, axis=1))


_PARTITION_WEIGHT_CACHE: dict = {}

# Complex-element budget per parent chunk in the integer-scan probe synthesis
# (mirrors the 6e8 budget _propagate uses). One chunk holds ~3 live buffers
# (beamlet basis, de-tilted parents, rolled basis), so the synthesis transient
# is bounded regardless of Bp -- the property the tip-scale emission/EELS conv
# forms need (they synthesise ONE centroid probe per ionizing slice on grids
# where even 3-4 full (Bp, Ny, Nx) buffers are tens of GB).
_PROBE_PARENT_CHUNK_ELEMS = int(6e8)


def _fresnel_window_kernel(Wy, Wx, n_slices, dxy, dxx, lam, dz, device, dtype):
    """Fresnel propagator over ``n_slices`` slices on a ``(Wy, Wx)`` window grid.

    Same sign convention as :func:`propagator_kernel` (forward = ``exp(-i pi lam
    (n dz) |q|^2)``); built on the window's own frequency grid (pixel size
    ``dxy``/``dxx``) so a windowed reconstruction can be propagated locally.
    """
    qy = torch.fft.fftfreq(Wy, d=dxy, device=device, dtype=torch.float64).view(Wy, 1)
    qx = torch.fft.fftfreq(Wx, d=dxx, device=device, dtype=torch.float64).view(1, Wx)
    chi = -pi * lam * (n_slices * dz) * (qy**2 + qx**2)
    return torch.exp(1j * chi).to(dtype)


def _pad_window(idx: Tensor, n: int, m: int):
    """Pad a contiguous (step-1, periodic) window ``idx`` by ``m`` on each side.

    Returns ``(padded_idx, inner_slice)`` where ``padded_idx`` is the widened
    window (capped to the full axis) and ``inner_slice`` selects the original
    ``idx`` positions within it. Used to give the windowed forward-Fresnel enough
    margin that its wrap-around stays outside the inner window.
    """
    L = int(idx.shape[0])
    Lp = min(L + 2 * m, n)
    left = (Lp - L) // 2
    start = (int(idx[0].item()) - left) % n
    padded = (start + torch.arange(Lp, device=idx.device)) % n
    return padded, slice(left, left + L)


def _partition_weights(
    beams_np: np.ndarray, all_signed: np.ndarray, n_radial: int, n_angular: int
):
    """Parent indices + natural-neighbor weights for an aperture geometry.

    Cached on the aperture beam set + ``(n_radial, n_angular)``: the geometry (and
    hence the parents and the NNW weights) is independent of the probe phase
    (defocus), the transmissions, and the sample thickness, so the expensive
    Delaunay/NNW computation runs once per aperture and is reused across every
    subsequent :class:`PartitionedScatteringMatrix` with the same aperture.
    Returns ``(parent_indices, w)`` with ``w`` of shape ``(B, Bp)``.
    """
    from ._nnw import natural_neighbor_weights

    key = (beams_np.shape, beams_np.tobytes(), int(n_radial), int(n_angular))
    cached = _PARTITION_WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached
    pidx = _select_parent_beams(all_signed, int(n_radial), int(n_angular))
    # Default is the fast vectorised Delaunay-barycentric weights ("linear").
    w = natural_neighbor_weights(all_signed[pidx], all_signed)  # (B, Bp)
    _PARTITION_WEIGHT_CACHE[key] = (pidx, w)
    return pidx, w


class PartitionedScatteringMatrix:
    """Partitioned-PRISM scattering matrix (Pelz 2021) for STEM-EELS.

    Instead of one multislice-propagated plane wave per aperture pixel, the
    matrix is built on a small set of ``Bp`` *parent* beams (a hex-ring subsample
    of the aperture).  The probe at a scan position is synthesised by natural-
    neighbor interpolation of the *de-tilted* parent columns:

        psi(r, R) = sum_p S_dt[p](r) * basis_p(r - R),
        S_dt[p](r)   = S[p](r) * exp(-2 pi i g_p . r / N),      (de-tilted column)
        basis_p(rho) = sum_b coeffs[b] w[b,p] exp(2 pi i g_b . rho / N),

    where ``w[b,p]`` are the natural-neighbor weights expressing aperture beam
    ``b`` in terms of the parents.  In the full-parent limit (every beam its own
    parent, ``w`` the identity) this is algebraically the exact per-pixel probe,
    so :class:`ScatteringMatrix` is the ``Bp -> B`` limit and serves as the
    oracle.  Presents the ``advance_to`` / ``probe_at_current_plane`` interface of
    :class:`ScatteringMatrix` plus :meth:`coeffs_at` / :meth:`reconstruct_columns`,
    so it can drive either the hybrid path or the dual-scattering-matrix exit (the
    reconstructed columns are scalar-combinable). Accepts ``interpolation_factor``
    so PRISM beam subsampling composes with partitioning. Accuracy is
    regime-dependent (the natural-neighbor reconstruction error).
    """

    def __init__(
        self,
        probe_q: Tensor,
        transmissions: Tensor,
        kernel: Tensor,
        n_radial: int = 4,
        n_angular: int = 6,
        interpolation_factor: Sequence[int] = (1, 1),
        focal_backprop=None,
        mag_preserve: bool = True,
        wavelength: float = None,
        slice_distance: float = None,
        gridsize: Sequence[float] = None,
    ):
        if int(n_radial) < 2:
            raise ValueError(
                f"n_radial must be >= 2 for NNW partitioning, got {n_radial}"
            )
        device = transmissions.device
        dtype = transmissions.dtype
        self.transmissions = transmissions
        self.kernel = kernel
        self._mag_preserve = bool(mag_preserve)
        # Optional spatial reference before interpolation, followed by forward
        # propagation to the ionization plane. None/0 leaves the reference at
        # the current plane; "centroid" uses half the traversed depth (rounded
        # to a slice), and a float specifies that depth fraction. No optimum
        # reference plane is assumed.
        self._focal_backprop = focal_backprop
        # Physical scale for the windowed-Fresnel forward step (only needed when
        # focal_backprop is on AND the reconstruction is windowed).
        self._lam = None if wavelength is None else float(wavelength)
        self._dz = None if slice_distance is None else float(slice_distance)
        self._gridsize = (
            None if gridsize is None else (float(gridsize[0]), float(gridsize[1]))
        )
        self._S_bp = None  # cached full-grid back-propagated parents
        self._S_bp_key = None
        ny, nx = transmissions.shape[-2:]
        self.ny, self.nx = ny, nx
        self._dtype = dtype

        fy, fx = int(interpolation_factor[0]), int(interpolation_factor[1])
        if fy < 1 or fx < 1:
            raise ValueError(f"interpolation_factor must be >= 1, got ({fy}, {fx})")
        if ny % fy or nx % fx:
            raise ValueError(
                f"interpolation_factor {(fy, fx)} must divide the grid {(ny, nx)}"
            )
        self.interpolation_factor = (fy, fx)

        # Aperture beam bookkeeping is done on CPU (small; natural_neighbor_weights
        # and the numpy indexing are CPU-only); only the final S / Bq tensors are
        # moved to the transmissions' device, so this works on GPU too.
        beams = torch.nonzero(probe_q.abs() > 0, as_tuple=False).cpu()  # (B, 2)
        if fy > 1 or fx > 1:
            # PRISM interpolation_factor: keep every fy-th / fx-th beam first, then
            # partition THAT (subsampled) aperture.  The two reductions stack -- f
            # shrinks the exact beam set (and hence the dual-S exit columns), the
            # parents shrink the S1-build basis.
            keep = (beams[:, 0] % fy == 0) & (beams[:, 1] % fx == 0)
            beams = beams[keep]
        by, bx = beams[:, 0], beams[:, 1]
        # Decimation identity: subsampling scales the synthesised probe by
        # 1/(fy*fx), so restore amplitude (matches ScatteringMatrix.coeffs0).
        coeffs = (probe_q.detach().cpu()[by, bx] * (fy * fx)).to(
            torch.complex128
        )  # (B,)
        gy_s = ((by + ny // 2) % ny) - ny // 2
        gx_s = ((bx + nx // 2) % nx) - nx // 2
        all_signed = torch.stack([gy_s, gx_s], 1).numpy().astype(np.float64)

        pidx, w = _partition_weights(
            beams.numpy(), all_signed, int(n_radial), int(n_angular)
        )
        self.n_parents = int(len(pidx))
        pidx_t = torch.as_tensor(np.asarray(pidx), dtype=torch.long)
        parent_by = by[pidx_t]
        parent_bx = bx[pidx_t]
        parent_signed = all_signed[pidx]  # (Bp, 2)

        # Store only the SMALL beamlet weights + beam/parent indices.  The
        # (Bp, Ny, Nx) beamlet spectrum Bq, the de-tilt phase and the real-space
        # basis are built LAZILY in probe_at_current_plane, so the persistent
        # footprint during the (memory-heavy) multislice advance is just
        # S (Bp, Ny, Nx) -- ~B/Bp smaller than the full per-pixel scattering
        # matrix, the win for large-scale simulations.
        w = torch.as_tensor(w, dtype=torch.complex128)
        self._w = w.to(dtype=dtype, device=device)  # (B, Bp) NNW weights
        self.coeffs0 = coeffs.to(dtype=dtype, device=device)  # (B,) aperture coeffs
        self._cw = self.coeffs0[:, None] * self._w  # (B, Bp)
        self._by = by.to(device)
        self._bx = bx.to(device)
        self._parent_signed = parent_signed  # numpy (Bp, 2)
        self._device = device

        # Parent column initial waves (deltas at parent beams -> plane waves).
        Sq = torch.zeros((self.n_parents, ny, nx), dtype=dtype, device=device)
        Sq[
            torch.arange(self.n_parents, device=device),
            parent_by.to(device),
            parent_bx.to(device),
        ] = 1.0
        self.S = torch.fft.ifft2(Sq, dim=(-2, -1))
        self._slice = 0
        self._unit_slices = _unit_transmission_slices(transmissions)

        # Signed-frequency grids (small, 1-D) for the de-tilt and sub-pixel ramp.
        self._fy = (
            torch.fft.fftfreq(ny, device=device, dtype=torch.float64) * ny
        ).view(ny, 1)
        self._fx = (
            torch.fft.fftfreq(nx, device=device, dtype=torch.float64) * nx
        ).view(1, nx)

    def _signed_beam_freqs(self):
        ny, nx = self.ny, self.nx
        gy = ((self._by.to(torch.float64) + ny // 2) % ny) - ny // 2  # (B,)
        gx = ((self._bx.to(torch.float64) + nx // 2) % nx) - nx // 2
        return gy, gx

    def _detilted_S(self, S: Tensor = None) -> Tensor:
        """De-tilted parent columns ``S[p] * exp(-2 pi i g_p . r / N)`` (lazy)."""
        ny, nx = self.ny, self.nx
        if S is None:
            S = self.S
        gp = torch.as_tensor(
            self._parent_signed, dtype=torch.float64, device=self._device
        )
        yy = torch.arange(ny, device=self._device, dtype=torch.float64).view(1, ny, 1)
        xx = torch.arange(nx, device=self._device, dtype=torch.float64).view(1, 1, nx)
        detilt = torch.exp(
            -2j * pi * (gp[:, 0, None, None] * yy / ny + gp[:, 1, None, None] * xx / nx)
        ).to(self._dtype)
        return S * detilt

    def _backprop_n_half(self) -> float:
        """Diagnostic alias for the back-propagation distance in slices."""
        return float(self._backprop_slices())

    def _fresnel_pow(self, n_half: float, sign: float) -> Tensor:
        """Fourier Fresnel kernel for ``n_half`` slices, reusing the per-slice
        propagator phase (sign +1 = forward, -1 = back-propagation)."""
        return torch.exp((sign * n_half) * 1j * torch.angle(self.kernel)).to(
            self._dtype
        )

    def reconstruct_columns(self) -> Tensor:
        """Exact aperture columns reconstructed from the parents (``B, Ny, Nx``).

        ``S_approx[b](r) = exp(2 pi i g_b . r / N) sum_p w[b,p] S_dt[p](r)`` -- the
        natural-neighbor interpolation that expresses each aperture beam in terms
        of the (propagated, de-tilted) parents.  Combined with :meth:`coeffs_at`
        this reproduces :meth:`probe_at_current_plane` exactly, but as a *scalar*
        linear combination of fixed columns -- the form the dual-scattering-matrix
        exit leg needs (so the partitioned S1 still gets the ``P``-independent
        exit).  Materialises ``B`` columns, so the build-time ``Bp`` memory saving
        does not extend to the exit step.
        """
        ny, nx = self.ny, self.nx
        n = self._backprop_slices()
        Sd = self._detilted_S(self._fresnel(self.S, -n) if n else None)  # (Bp, Ny, Nx)
        recon = torch.einsum("bp,pyx->byx", self._w, Sd)  # (B, Ny, Nx)
        # NOTE: pure linear NNW here (no mag_preserve) so this stays algebraically
        # equal to probe_at_current_plane; mag_preserve lives in the windowed path.
        gy, gx = self._signed_beam_freqs()
        yy = torch.arange(ny, device=self._device, dtype=torch.float64).view(1, ny, 1)
        xx = torch.arange(nx, device=self._device, dtype=torch.float64).view(1, 1, nx)
        tilt = torch.exp(
            2j * pi * (gy[:, None, None] * yy / ny + gx[:, None, None] * xx / nx)
        ).to(self._dtype)
        recon = recon * tilt
        return self._fresnel(recon, n) if n else recon  # forward to current plane

    def reconstruct_columns_window(self, iy: Tensor, ix: Tensor) -> Tensor:
        """Reconstruct the exact aperture columns on a crop window only ``(B, wy, wx)``.

        With magnitude replacement and focal referencing disabled, equals
        ``reconstruct_columns()[:, iy][:, :, ix]`` but never
        materialises the full ``(B, Ny, Nx)`` matrix -- it de-tilts the cropped
        ``Bp`` parent columns, NNW-combines them, and re-tilts, all on the window.
        This realises the ``Bp``-parent memory saving (the persistent footprint is
        just the parents) and is also cheaper (window-sized einsum, not full grid).
        ``iy`` / ``ix`` are 1-D pixel-index tensors of the window rows / columns.

        With ``focal_backprop`` the parents are first back-propagated (full grid,
        once per plane) to the scattering-centroid plane; the reconstruction is done
        on a window padded by the Fresnel reach, then forward-propagated locally and
        cropped to the inner window -- so the centroid-plane interpolation accuracy
        is realised without ever forming the full ``(B, Ny, Nx)`` matrix.
        """
        ny, nx = self.ny, self.nx
        n = self._backprop_slices()
        if n:
            return self._reconstruct_columns_window_fb(iy, ix, n)
        Sw = self.S[:, iy][:, :, ix]  # (Bp, wy, wx) parent columns on the window
        return self._nnw_window(Sw, iy, ix)

    def coeffs_at(self, scan_pixels: Tensor) -> Tensor:
        """Per-(scan, beam) synthesis coefficients ``(P, B)`` (cf. ``ScatteringMatrix``).

        ``probe = sum_b coeffs[p, b] * reconstruct_columns()[b]`` reproduces
        :meth:`probe_at_current_plane`.
        """
        ny, nx = self.ny, self.nx
        gy, gx = self._signed_beam_freqs()
        ry = scan_pixels[:, 0].to(torch.float64)
        rx = scan_pixels[:, 1].to(torch.float64)
        phase = torch.exp(
            -2j * pi * (gy[None, :] * ry[:, None] / ny + gx[None, :] * rx[:, None] / nx)
        ).to(self._dtype)
        return phase * self.coeffs0[None, :]  # (P, B)

    def _beamlet_spectrum(self) -> Tensor:
        """Beamlet spectra ``Bq[p, gy, gx] = sum_b coeffs[b] w[b,p]`` (lazy)."""
        Bq = torch.zeros(
            (self.n_parents, self.ny, self.nx), dtype=self._dtype, device=self._device
        )
        Bq[:, self._by, self._bx] = self._cw.t()
        return Bq

    def advance_to(self, target_slice: int) -> None:
        """Propagate the parent columns forward to ``target_slice``.

        Identity (``1 + 0j``) transmission slices are skipped as in
        :meth:`ScatteringMatrix.advance_to`.
        """
        for j in range(self._slice, target_slice):
            if not self._unit_slices[j]:
                self.S = self.S * self.transmissions[j]
            self.S = _propagate(self.S, self.kernel)
        self._slice = target_slice

    def probe_at_current_plane(self, scan_pixels: Tensor) -> Tensor:
        """Synthesise probes ``(P, Ny, Nx)`` at the current plane by NNW
        interpolation of the de-tilted parent columns.

        Two paths:

        * **Integer (Nyquist-step) scan** -- the dominant fast case.  The probe
          shift ``basis_p(r - R)`` for an integer ``R`` is a cyclic ROLL of the
          real-space beamlet basis ``basis_p = ifft2(Bq_p)``, accumulated in
          PARENT CHUNKS (budget ``_PROBE_PARENT_CHUNK_ELEMS``) so the transient
          stays ~3 chunk-sized buffers instead of 4-5 full ``(Bp, Ny, Nx)``
          tensors -- the memory property the tip-scale emission/EELS conv forms
          rely on (one centroid probe per ionizing slice).
        * **Sub-pixel scan** -- the general Fourier-shift path: one iFFT per
          (scan, parent), chunked over scan positions.
        """
        ny, nx = self.ny, self.nx
        norm = float(ny * nx)
        # Focal back-propagation: reference the columns to the scattering centroid
        # of the traversed slices before fusion (improves the partitioned interp);
        # the synthesised probe is forward-propagated back to the current plane at
        # the end so the result is exact in the full-parent limit.
        n_half = self._backprop_n_half()
        ry = scan_pixels[:, 0].to(torch.float64)
        rx = scan_pixels[:, 1].to(torch.float64)
        P = scan_pixels.shape[0]

        ry_i = torch.round(ry)
        rx_i = torch.round(rx)
        if torch.allclose(ry, ry_i) and torch.allclose(rx, rx_i):
            sy = (ry_i.long() % ny).tolist()
            sx = (rx_i.long() % nx).tolist()
            out = torch.zeros((P, ny, nx), dtype=self._dtype, device=self._device)
            fres_b = self._fresnel_pow(n_half, -1.0) if n_half != 0.0 else None
            gp = torch.as_tensor(
                self._parent_signed, dtype=torch.float64, device=self._device
            )  # (Bp, 2)
            yy = torch.arange(ny, device=self._device, dtype=torch.float64).view(
                1, ny, 1
            )
            xx = torch.arange(nx, device=self._device, dtype=torch.float64).view(
                1, 1, nx
            )
            cw_t = self._cw.t()  # (Bp, B)
            # The per-chunk de-tilt + beamlet basis below inline _detilted_S /
            # _beamlet_spectrum on a parent SLICE -- calling those helpers would
            # materialise the full-Bp buffers this rewrite exists to avoid (keep
            # the three conventions in sync if either helper changes).
            pchunk = max(1, int(_PROBE_PARENT_CHUNK_ELEMS // max(1, ny * nx)))
            for p0 in range(0, self.n_parents, pchunk):
                p1 = min(p0 + pchunk, self.n_parents)
                S_c = self.S[p0:p1]
                if fres_b is not None:
                    S_c = torch.fft.ifft2(
                        torch.fft.fft2(S_c, dim=(-2, -1)) * fres_b, dim=(-2, -1)
                    )
                detilt = torch.exp(
                    -2j
                    * pi
                    * (
                        gp[p0:p1, 0, None, None] * yy / ny
                        + gp[p0:p1, 1, None, None] * xx / nx
                    )
                ).to(self._dtype)
                S_dt_c = S_c * detilt
                del detilt
                Bq_c = torch.zeros(
                    (p1 - p0, ny, nx), dtype=self._dtype, device=self._device
                )
                Bq_c[:, self._by, self._bx] = cw_t[p0:p1]
                basis_c = torch.fft.ifft2(Bq_c, dim=(-2, -1)) * norm
                del Bq_c
                # The roll is re-paid per chunk when P > 1 (O(chunks*P)); the
                # dominant inelastic caller passes a single centroid probe
                # (P == 1) and rolls are cheap next to the FFTs, so bounding the
                # transient wins over hoisting the roll out of the chunk loop.
                for k in range(P):
                    shifted = torch.roll(basis_c, shifts=(sy[k], sx[k]), dims=(-2, -1))
                    out[k] += (S_dt_c * shifted).sum(0)
        else:
            S_src = self.S
            if n_half != 0.0:
                S_src = torch.fft.ifft2(
                    torch.fft.fft2(self.S, dim=(-2, -1))
                    * self._fresnel_pow(n_half, -1.0),
                    dim=(-2, -1),
                )
            S_dt = self._detilted_S(S_src)[None]  # (1, Bp, Ny, Nx)
            Bq = self._beamlet_spectrum()  # (Bp, Ny, Nx)
            chunk = max(1, int(5e7 // max(1, self.n_parents * ny * nx)))
            out_chunks = []
            for s in range(0, P, chunk):
                ryc = ry[s : s + chunk].view(-1, 1, 1)
                rxc = rx[s : s + chunk].view(-1, 1, 1)
                ramp = torch.exp(
                    -2j * pi * (self._fy[None] * ryc / ny + self._fx[None] * rxc / nx)
                ).to(
                    self._dtype
                )  # (c, Ny, Nx)
                basis = torch.fft.ifft2(Bq[None] * ramp[:, None], dim=(-2, -1)) * norm
                out_chunks.append((S_dt * basis).sum(1))  # (c, Ny, Nx)
            out = torch.cat(out_chunks, dim=0)

        if n_half != 0.0:
            # Forward-propagate the synthesised probe back to the current plane.
            out = torch.fft.ifft2(
                torch.fft.fft2(out, dim=(-2, -1)) * self._fresnel_pow(n_half, 1.0),
                dim=(-2, -1),
            )
        return out

    def _backprop_slices(self) -> int:
        """Number of slices to Fresnel back-propagate before interpolation.

        ``"centroid"`` => half the propagated depth (the centroid of uniform
        scattering between entrance and the current plane); a float => that
        fraction of the current depth; ``None``/``0`` => no back-propagation.
        """
        # No interpolation error exists when every beam is a parent. Avoid a
        # cropped Fresnel round trip, which would introduce a window error.
        if self.n_parents == self._by.numel():
            return 0
        fb = self._focal_backprop
        if not fb:
            return 0
        if isinstance(fb, str):
            if fb != "centroid":
                raise ValueError(f"focal_backprop str must be 'centroid', got {fb!r}")
            return int(round(self._slice / 2))
        return int(round(float(fb) * self._slice))

    def _fresnel(self, S: Tensor, n_slices: int) -> Tensor:
        """Fresnel-propagate ``S`` forward by ``n_slices`` (negative = backward)."""
        if n_slices == 0:
            return S
        K = self.kernel**n_slices  # |kernel|=1, so negative powers are the conjugate
        return torch.fft.ifft2(torch.fft.fft2(S, dim=(-2, -1)) * K, dim=(-2, -1))

    def _backprop_parents(self, n: int) -> Tensor:
        """Full-grid parents back-propagated by ``n`` slices (cached per plane).

        Cheap (``Bp`` columns) and shared across every atom in the slice, so the
        windowed reconstruction can then work from these centroid-plane parents.
        """
        key = (self._slice, n)
        if self._S_bp_key != key:
            self._S_bp = self._fresnel(self.S, -n)
            self._S_bp_key = key
        return self._S_bp

    def _fb_margin(self, n: int) -> int:
        """Window padding (px) covering the Fresnel spread of the forward step.

        Spread ~ (n*dz) * lam * k_max; padded so the windowed-FFT wrap-around stays
        out of the inner window. Falls back to a safe constant if scale is unknown.
        """
        if not (self._lam and self._dz and self._gridsize):
            return 24
        gy, gx = self._signed_beam_freqs()
        gmax = float(torch.maximum(gy.abs().max(), gx.abs().max()))
        kmax = gmax / min(self._gridsize)  # 1/A
        dx = min(self._gridsize[0] / self.ny, self._gridsize[1] / self.nx)
        spread_px = abs(n) * self._dz * self._lam * kmax / dx
        return int(
            min(max(4, np.ceil(1.5 * spread_px) + 2), min(self.ny, self.nx) // 2)
        )

    def _nnw_window(self, Sw: Tensor, iy: Tensor, ix: Tensor) -> Tensor:
        """De-tilt the parent columns ``Sw`` on the window ``(iy, ix)``, NNW-combine
        (with mag_preserve), and re-tilt -- the per-pixel-local reconstruction."""
        ny, nx = self.ny, self.nx
        gp = torch.as_tensor(
            self._parent_signed, dtype=torch.float64, device=self._device
        )  # (Bp, 2)
        yy = iy.to(torch.float64).view(1, -1, 1)
        xx = ix.to(torch.float64).view(1, 1, -1)
        detilt = torch.exp(
            -2j * pi * (gp[:, 0, None, None] * yy / ny + gp[:, 1, None, None] * xx / nx)
        ).to(self._dtype)
        Sd = Sw * detilt
        recon = torch.einsum("bp,pwv->bwv", self._w, Sd)  # (B, wy, wx)
        if self._mag_preserve:
            # Counter NNW phase-decoherence amplitude loss: keep the complex-sum phase
            # but restore the (interpolated) magnitude the complex average shrinks.
            mag = torch.einsum("bp,pwv->bwv", self._w, Sd.abs().to(self._dtype)).real
            recon = mag * (recon / recon.abs().clamp_min(1e-20))
        gy, gx = self._signed_beam_freqs()
        tilt = torch.exp(
            2j * pi * (gy[:, None, None] * yy / ny + gx[:, None, None] * xx / nx)
        ).to(self._dtype)
        return recon * tilt

    def _reconstruct_columns_window_fb(self, iy: Tensor, ix: Tensor, n: int) -> Tensor:
        """Focal-back-prop windowed reconstruction: combine the parents at the
        centroid plane (where they interpolate accurately), then forward-propagate
        the result to the current plane -- all on a padded window."""
        m = self._fb_margin(n)
        py, iny = _pad_window(iy, self.ny, m)
        px, inx = _pad_window(ix, self.nx, m)
        Sw = self._backprop_parents(n)[:, py][:, :, px]  # (Bp, Py, Px) at centroid
        recon = self._nnw_window(Sw, py, px)  # (B, Py, Px) at centroid
        Py, Px = py.shape[0], px.shape[0]
        if self._lam and self._dz and self._gridsize:
            dxy = self._gridsize[0] / self.ny
            dxx = self._gridsize[1] / self.nx
            K = _fresnel_window_kernel(
                Py, Px, n, dxy, dxx, self._lam, self._dz, self._device, self._dtype
            )
        else:  # fall back to the full-grid single-slice kernel restricted to a square
            raise ValueError(
                "focal_backprop windowed reconstruction needs wavelength, "
                "slice_distance and gridsize"
            )
        recon = torch.fft.ifft2(torch.fft.fft2(recon, dim=(-2, -1)) * K, dim=(-2, -1))
        return recon[:, iny][:, :, inx]  # crop to the inner window


def prism_transition_potential(
    probe_q: Tensor,
    transmissions: Tensor,
    transition_potentials: TransitionPotentials,
    sites: np.ndarray,
    scan_pixels: Tensor,
    *,
    wavelength: float,
    gridsize: Sequence[float],
    slice_distance: float,
    return_real_space: bool = False,
    interpolation_factor=1,
    partition=None,
    scan_chunk: int = 64,
) -> Tensor:
    """Dual-scattering-matrix PRISM STEM-EELS (Brown & Ophus).

    Implements the two-scattering-matrix algorithm of Brown, Ciston & Ophus,
    Phys. Rev. Research **1**, 033186 (2019), Eq. (11):

        S_{n,iz} = S2 . F . H_{n0} . F^{-1} . S1,

    where :math:`\\mathcal S_1` propagates the probe-forming plane waves from the
    entrance surface to the ionization slice ``iz`` and :math:`\\mathcal S_2`
    propagates the inelastically-scattered wave from ``iz`` to the exit surface.

    The key to the full PRISM benefit (vs. :func:`prism_transition_potential_hybrid`)
    is the *order of operations*. Because every operator after :math:`\\mathcal S_1`
    is linear, the exit wave for scan position ``p`` is::

        psi_exit(p) = S2[ H_n0 . (sum_b coeff_p[b] S1_col[b]) ]
                    = sum_b coeff_p[b] * S2[ H_n0 . S1_col[b] ],

    so the transition is applied to the ``Nbeams`` columns of :math:`\\mathcal S_1`
    and **those columns** are propagated to the exit (the action of
    :math:`\\mathcal S_2`), *then* the scan positions are synthesised by the same
    linear combination used to form the probe. The expensive exit-surface
    multislice therefore runs ``Nbeams`` times per (site, transition) -- **independent
    of the number of scan positions** ``P`` -- instead of ``P`` times. For
    ``P >> Nbeams`` (a dense scan through a small aperture, the PRISM regime) this
    is the source of the near-linear scaling.

    With the exact per-pixel scattering matrix (``partition=None``) the result is
    identical (to float rounding) to :func:`prism_transition_potential_hybrid`;
    this routine just reorders the work. Energy-filtered 4D output is preserved --
    ``S2`` is applied as on-the-fly multislice of the column basis (the full exit
    diffraction pattern is kept, so no detector reduction is made).
    Detector-integrating the returned cube over an EELS aperture reproduces the
    Brown & Ophus ``I(x, y)`` image.

    **Shrinking the column count (two compatible levers).**

    * ``interpolation_factor = f`` subsamples the aperture beams to ``Nbeams / f**2``
      columns -- exact within the PRISM aliasing window -- shrinking the exact
      columns of **both** legs (S1 build and exit) by ``f**2``.
    * ``partition`` (e.g. ``{"n_radial": 4}``) builds S1 on ``Bp << Nbeams`` parent
      beams (Pelz/Ophus partitioned PRISM), then reconstructs the exact aperture
      columns from the parents at the ionization plane and runs the same dual-S
      exit. This cheapens the **S1 build + memory** (``Bp`` parent propagations) and
      is *approximate* (natural-neighbor reconstruction error -- accurate in-focus /
      thin / small-aperture, degrading with thickness/aperture); the exit leg still
      propagates the reconstructed ``Nbeams / f**2`` columns, so the dual-S
      ``P``-independence is preserved. Note partitioning does **not** reduce the
      (usually dominant) exit leg -- only the S1 build; ``interpolation_factor`` is
      the lever for the exit. The two compose (``partition`` partitions the
      already-subsampled aperture). Unlike partitioning *S2*, this is correct: the
      reconstruction resolves the position-dependent NNW basis *before* ``H_{n0}``
      is applied (see :meth:`PartitionedScatteringMatrix.reconstruct_columns`).

    Parameters
    ----------
    probe_q, transmissions, transition_potentials, sites, scan_pixels, wavelength,
    gridsize, slice_distance, return_real_space, interpolation_factor, partition
        As in :func:`prism_transition_potential_hybrid`. ``partition`` and
        ``interpolation_factor`` may be combined.
    scan_chunk : int
        Number of scan positions synthesised per batch (caps the
        ``(scan_chunk, Ny, Nx)`` temporary in the linear-combination step).

    Returns
    -------
    Tensor
        Intensity ``(P, Ny, Nx)`` summed over transitions and sites.
    """
    device = transmissions.device
    ny, nx = transmissions.shape[-2:]
    nz = transmissions.shape[0]
    H = transition_potentials.array.to(device=device)  # (n_trans, Ny, Nx)
    n_trans = H.shape[0]
    kernel = propagator_kernel(
        (ny, nx), gridsize, wavelength, slice_distance, device=device, dtype=H.dtype
    )

    sites = np.atleast_2d(np.asarray(sites, dtype=np.float64))
    site_slice = np.clip((sites[:, 2] % 1.0 * nz).astype(int), 0, nz - 1)

    # Slices whose transmission function is exactly ``1 + 0j``: their multiply in
    # the exit leg below is the identity and is skipped bit-identically.  The exit
    # leg is 85 % of a dual-S simulate and one transmit is a full pass over the
    # (Ncols, Ny, Nx) column stack, so this is worth the one batched reduction.
    unit_slices = _unit_transmission_slices(transmissions)

    P = scan_pixels.shape[0]
    out = torch.zeros((P, ny, nx), dtype=torch.float64, device=device)

    # S1: built once, advanced lazily to each ionization plane, reused across all
    # scan positions (and all sites/transitions).  Both matrices expose the same
    # ``columns`` (scalar-combinable basis at the current slice) + ``coeffs_at``
    # interface, so the dual-S exit is identical for either.  With partitioning the
    # S1 build/memory uses only Bp parent columns; the exact aperture columns are
    # reconstructed (NNW) at the ionization plane for the exit -- so partitioning
    # cheapens the S1 leg (approximately, NNW error) while the exit stays
    # P-independent.  ``interpolation_factor`` subsamples the aperture beams (and
    # composes with partitioning), shrinking the exact-column count of BOTH legs.
    partitioned = partition is not None
    if partitioned:
        smatrix = PartitionedScatteringMatrix(
            probe_q.to(transmissions.dtype),
            transmissions,
            kernel,
            interpolation_factor=_factor_pair(interpolation_factor),
            **partition,
        )
    else:
        smatrix = ScatteringMatrix(
            probe_q.to(transmissions.dtype),
            transmissions,
            kernel,
            interpolation_factor=_factor_pair(interpolation_factor),
        )

    for i in range(nz):
        in_slice = np.nonzero(site_slice == i)[0]
        if in_slice.size == 0:
            continue
        smatrix.advance_to(i)
        # Scalar-combinable columns at slice i + the matching synthesis weights.
        cols = smatrix.reconstruct_columns() if partitioned else smatrix.S
        coeffs = smatrix.coeffs_at(scan_pixels)  # (P, Ncols)
        for s in in_slice:
            sy = sites[s, 0] % 1.0 * ny
            sx = sites[s, 1] % 1.0 * nx
            h_site = _fourier_shift(H, (sy, sx))  # (n_trans, Ny, Nx)
            for t in range(n_trans):
                # Apply the transition to each column (H_n0 . S1), then propagate
                # the columns to the exit surface (the action of S2).
                post = h_site[t][None] * cols  # (Ncols, Ny, Nx)
                exit_cols = _slice_exit_wave(
                    post, transmissions, kernel, i, unit_slices
                )
                field = (
                    exit_cols
                    if return_real_space
                    else torch.fft.fft2(exit_cols, dim=(-2, -1))
                )  # (Ncols, Ny, Nx)
                # Synthesise every scan position by the SAME linear combination
                # that forms the probe (chunked over P to cap memory).
                for c0 in range(0, P, scan_chunk):
                    c1 = min(c0 + scan_chunk, P)
                    ex = torch.einsum("cb,byx->cyx", coeffs[c0:c1], field)
                    out[c0:c1] += ex.abs().pow(2).to(torch.float64)

    return out


def prism_transition_potential_hybrid(
    probe_q: Tensor,
    transmissions: Tensor,
    transition_potentials: TransitionPotentials,
    sites: np.ndarray,
    scan_pixels: Tensor,
    *,
    wavelength: float,
    gridsize: Sequence[float],
    slice_distance: float,
    return_real_space: bool = False,
    interpolation_factor=1,
    partition=None,
) -> Tensor:
    """Hybrid PRISM/multislice STEM-EELS (single scattering matrix).

    This is the original implementation: the **first** scattering matrix
    :math:`\\mathcal S_1` (reused across scan positions) forms the probe at the
    ionization plane, but the inelastically-scattered wave is then propagated to
    the exit by ordinary multislice **per scan position** -- so the exit-leg cost
    still scales with the number of probe positions.

    :func:`prism_transition_potential` is the proper dual-scattering-matrix
    algorithm (Brown & Ophus) and is the preferred entry point; this function is
    kept for the partitioned-PRISM path and as the equivalence oracle. The two
    return bit-for-bit-equivalent results (up to float rounding).

    Parameters
    ----------
    probe_q : Tensor
        Probe-forming aperture in reciprocal space ``(Ny, Nx)`` (complex).
    transmissions : Tensor
        Real-space transmission functions ``(NZ, Ny, Nx)``.
    transition_potentials : TransitionPotentials
        Transition potentials for the edge.
    sites : ndarray
        Fractional ``(Nsite, 3)`` ``(y, x, z)`` coordinates of ionizable atoms.
    scan_pixels : Tensor
        Scan positions ``(P, 2)`` in pixels.
    wavelength, gridsize, slice_distance
        As in :func:`..multislice_eels.transition_potential_multislice`.
    interpolation_factor : int or (int, int)
        PRISM interpolation factor ``f`` (or ``(fy, fx)``); see
        :class:`ScatteringMatrix`.  ``1`` (default) is exact; ``f > 1`` subsamples
        the beams to speed up the scattering matrix by ``f**2`` and is accurate
        while the probe fits in the ``1/f`` window.
    partition : dict or None
        If given (e.g. ``{"n_radial": 4}``), use a
        :class:`PartitionedScatteringMatrix` (partitioned PRISM, Pelz 2021)
        instead; ``interpolation_factor`` is then ignored.  ``None`` (default)
        uses the exact per-pixel scattering matrix.

    Returns
    -------
    Tensor
        Intensity ``(P, Ny, Nx)`` summed over transitions and sites.
    """
    device = transmissions.device
    ny, nx = transmissions.shape[-2:]
    nz = transmissions.shape[0]
    H = transition_potentials.array.to(device=device)
    kernel = propagator_kernel(
        (ny, nx), gridsize, wavelength, slice_distance, device=device, dtype=H.dtype
    )

    sites = np.atleast_2d(np.asarray(sites, dtype=np.float64))
    site_slice = np.clip((sites[:, 2] % 1.0 * nz).astype(int), 0, nz - 1)

    P = scan_pixels.shape[0]
    out = torch.zeros((P, ny, nx), dtype=torch.float64, device=device)

    # Slices whose transmission function is exactly ``1 + 0j`` -- their multiply is
    # the identity for every finite input and is skipped, bit-exactly.  Computed
    # once per call (one batched reduction, one device sync); ``transmissions`` is
    # only ever read from here on, never written in place.
    unit_slices = _unit_transmission_slices(transmissions)

    # S1: reused across all scan positions; advanced lazily to each plane.
    if partition is not None:
        smatrix = PartitionedScatteringMatrix(
            probe_q.to(transmissions.dtype),
            transmissions,
            kernel,
            interpolation_factor=_factor_pair(interpolation_factor),
            **partition,
        )
    else:
        smatrix = ScatteringMatrix(
            probe_q.to(transmissions.dtype),
            transmissions,
            kernel,
            interpolation_factor=_factor_pair(interpolation_factor),
        )

    for i in range(nz):
        in_slice = np.nonzero(site_slice == i)[0]
        if in_slice.size == 0:
            continue
        smatrix.advance_to(i)
        probes = smatrix.probe_at_current_plane(scan_pixels)  # (P, Ny, Nx)
        for s in in_slice:
            sy = sites[s, 0] % 1.0 * ny
            sx = sites[s, 1] % 1.0 * nx
            h_site = _fourier_shift(H, (sy, sx))  # (n_trans, Ny, Nx)
            psi_n = h_site[None] * probes[:, None]  # (P, n_trans, Ny, Nx)
            psi_n = _slice_exit_wave(psi_n, transmissions, kernel, i, unit_slices)
            if return_real_space:
                out += psi_n.abs().pow(2).sum(dim=1).to(torch.float64)
            else:
                diff = torch.fft.fft2(psi_n, dim=(-2, -1))
                out += diff.abs().pow(2).sum(dim=1).to(torch.float64)

    return out
