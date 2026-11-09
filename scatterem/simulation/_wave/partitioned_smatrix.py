"""The partitioned PRISM scattering matrix (Bp parent beams, NNW-interpolated).

:class:`PartitionedScatteringMatrix` walks only ``Bp`` parent beams of the
aperture and reconstructs the others by natural-neighbour interpolation of the
de-tilted parents.  Generic, not EELS-specific.  Moved verbatim (2026-09-27)
from ``simulation/eels/forward/prism_eels.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import pi

import numpy as np
import torch
from torch import Tensor

from .exit_walk import _SliceAdvanceMixin, _unit_transmission_slices
from .focal_reference import _focal_slices, _FocalReferenceMixin
from .prism_beams import _checked_factor, _partition_weights

__all__ = ["PartitionedScatteringMatrix"]


# Complex-element budget per parent chunk in the integer-scan probe synthesis
# (mirrors the 6e8 budget propagate uses). One chunk holds ~3 live buffers
# (beamlet basis, de-tilted parents, rolled basis), so the synthesis transient
# is bounded regardless of Bp, on grids where even 3-4 full (Bp, Ny, Nx)
# buffers are tens of GB.
_PROBE_PARENT_CHUNK_ELEMS = int(6e8)


class PartitionedScatteringMatrix(_FocalReferenceMixin, _SliceAdvanceMixin):
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
    so ``ScatteringMatrix`` is the ``Bp -> B`` limit and serves as the
    oracle.  Presents the ``advance_to`` / ``probe_at_current_plane`` interface of
    ``ScatteringMatrix`` plus :meth:`coeffs_at` / :meth:`reconstruct_columns`,
    so it can drive either the hybrid path or the dual-scattering-matrix exit (the
    reconstructed columns are scalar-combinable). Accepts ``interpolation_factor``
    so PRISM beam subsampling composes with partitioning. Accuracy is
    regime-dependent (the natural-neighbor reconstruction error).

    Parameters
    ----------
    probe_q : Tensor
        Probe-forming aperture in reciprocal space ``(Ny, Nx)`` (complex;
        corner origin).  Non-zero pixels define the beams.
    transmissions : Tensor
        Real-space transmission functions ``(NZ, Ny, Nx)``.
    kernel : Tensor
        Fresnel propagator ``(Ny, Nx)`` for one inter-slice distance.
    n_radial : int, optional
        Number of concentric hex rings of parent beams sampled from the
        aperture (must be >= 2).
    n_angular : int, optional
        Number of angular samples on the first hex ring (subsequent rings
        scale with ring index).
    interpolation_factor : (int, int), optional
        PRISM interpolation factors ``(fy, fx)`` (Ophus 2017), applied to the
        aperture before partitioning; ``(1, 1)`` keeps every beam.
    focal_backprop : float, str or None, optional
        Fresnel referencing of the parent columns before natural-neighbor
        fusion (Pelz 2021). A float back-propagates by that fraction of the
        traversed depth, rounded to whole slices; ``"centroid"`` selects half
        the traversed depth. The fused columns are then propagated forward to
        the current plane again. ``None`` (default) or ``0`` disables the
        referencing. It is skipped in the full-parent limit, where the columns
        are exact.
    mag_preserve : bool, optional
        Restore each reconstructed column's magnitude from the same NNW
        interpolation of ``|S|``, keeping the complex sum's phase. Applies to
        :meth:`reconstruct_columns_window`.
    wavelength : float, optional
        Electron wavelength in Angstrom. With ``slice_distance`` and
        ``gridsize`` it sets the windowed Fresnel step that
        :meth:`reconstruct_columns_window` needs when ``focal_backprop`` is on.
    slice_distance : float, optional
        Slice distance in Angstrom (see ``wavelength``).
    gridsize : (float, float), optional
        Real-space grid extent ``(Ly, Lx)`` in Angstrom (see ``wavelength``).
    """

    def __init__(
        self,
        probe_q: Tensor,
        transmissions: Tensor,
        kernel: Tensor,
        n_radial: int = 4,
        n_angular: int = 6,
        interpolation_factor: Sequence[int] = (1, 1),
        focal_backprop: float | str | None = None,
        mag_preserve: bool = True,
        wavelength: float | None = None,
        slice_distance: float | None = None,
        gridsize: Sequence[float] | None = None,
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
        ny, nx = transmissions.shape[-2:]
        self.ny, self.nx = ny, nx
        self._dtype = dtype
        # Optional Fresnel referencing (Pelz 2021) per transition slice; see
        # _backprop_slices. No optimum reference plane is assumed.
        self._init_focal_reference(focal_backprop, wavelength, slice_distance, gridsize)

        fy, fx = _checked_factor(interpolation_factor, ny, nx)
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
        self._device = device
        self._set_geometry(parent_signed)

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

    def _set_geometry(self, parent_signed: np.ndarray) -> None:
        """Store the signed parent and beam frequencies, on the device, once.

        The geometry does not change as the columns advance through the
        specimen, so the reconstruction helpers reuse these small float64
        tensors at every plane instead of converting them again.
        """
        ny, nx = self.ny, self.nx
        self._parent_signed = parent_signed  # numpy (Bp, 2)
        self._parent_signed_device = torch.as_tensor(
            parent_signed, dtype=torch.float64, device=self._device
        )  # (Bp, 2)
        self._beam_signed = (
            ((self._by.to(torch.float64) + ny // 2) % ny) - ny // 2,
            ((self._bx.to(torch.float64) + nx // 2) % nx) - nx // 2,
        )  # (B,), (B,)

    def _signed_beam_freqs(self) -> tuple[Tensor, Tensor]:
        """Signed beam frequencies ``(gy, gx)``, each ``(B,)`` float64 (cached)."""
        return self._beam_signed

    def _detilted_S(self, S: Tensor | None = None) -> Tensor:
        """De-tilted parent columns ``S[p] * exp(-2 pi i g_p . r / N)`` (lazy).

        ``S`` defaults to the current ``self.S``; pass a focal-back-propagated
        copy to de-tilt that instead.
        """
        S = self.S if S is None else S
        ny, nx = self.ny, self.nx
        gp = self._parent_signed_device
        yy = torch.arange(ny, device=self._device, dtype=torch.float64).view(1, ny, 1)
        xx = torch.arange(nx, device=self._device, dtype=torch.float64).view(1, 1, nx)
        detilt = torch.exp(
            -2j * pi * (gp[:, 0, None, None] * yy / ny + gp[:, 1, None, None] * xx / nx)
        ).to(self._dtype)
        return S * detilt

    def _backprop_slices(self) -> int:
        """Whole slices to back-propagate the parents before interpolation.

        Zero in the full-parent limit: there is no interpolation error to
        reduce, and a cropped Fresnel round trip would only add window error.
        """
        if self.n_parents >= int(self._by.numel()):
            return 0
        return _focal_slices(self.focal_backprop, int(self._slice))

    def _backprop_n_half(self) -> float:
        """The back-propagation distance of :meth:`_backprop_slices`, as a float."""
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

        Returns
        -------
        Tensor
            Reconstructed aperture columns ``(B, Ny, Nx)``.
        """
        ny, nx = self.ny, self.nx
        # Mirror probe_at_current_plane's referencing so the dual-S exit columns
        # stay the exact, scalar-combinable reordering of the synthesised probe:
        # back-propagate the parents before NNW fusion, then forward-propagate the
        # reconstructed columns to the current plane. The synthesis is linear in
        # the columns, so this equals forward-propagating the probe. Pure linear
        # NNW here (no mag_preserve), so the identity with the probe path holds.
        n = self._backprop_slices()
        Sd = self._detilted_S(self._fresnel(self.S, -n) if n else None)
        recon = torch.einsum("bp,pyx->byx", self._w, Sd)  # (B, Ny, Nx)
        gy, gx = self._signed_beam_freqs()
        yy = torch.arange(ny, device=self._device, dtype=torch.float64).view(1, ny, 1)
        xx = torch.arange(nx, device=self._device, dtype=torch.float64).view(1, 1, nx)
        tilt = torch.exp(
            2j * pi * (gy[:, None, None] * yy / ny + gx[:, None, None] * xx / nx)
        ).to(self._dtype)
        recon = recon * tilt
        return self._fresnel(recon, n) if n else recon

    def reconstruct_columns_window(self, iy: Tensor, ix: Tensor) -> Tensor:
        """Reconstruct the exact aperture columns on a crop window only ``(B, wy, wx)``.

        With ``mag_preserve`` and ``focal_backprop`` off, this equals
        ``reconstruct_columns()[:, iy][:, :, ix]`` but never materialises the
        full ``(B, Ny, Nx)`` matrix -- it de-tilts the cropped ``Bp`` parent
        columns, NNW-combines them, and re-tilts, all on the window.  This
        realises the ``Bp``-parent memory saving (the persistent footprint is
        just the parents) and is also cheaper (window-sized einsum, not full
        grid).

        With ``focal_backprop`` the parents are first back-propagated on the
        full grid (once per plane, cached). The reconstruction is then done on
        a window padded by the Fresnel reach, propagated forward locally and
        cropped to the inner window. This needs ``wavelength``,
        ``slice_distance`` and ``gridsize`` at construction.

        Parameters
        ----------
        iy : Tensor
            1-D pixel-index tensor of the window rows.
        ix : Tensor
            1-D pixel-index tensor of the window columns.

        Returns
        -------
        Tensor
            Reconstructed aperture columns on the window, ``(B, wy, wx)``.

        Raises
        ------
        ValueError
            If the referencing is on and the physical scale is missing.
        """
        n = self._backprop_slices()
        if n:
            gy, gx = self._signed_beam_freqs()
            gmax = float(torch.maximum(gy.abs().max(), gx.abs().max()))
            # Parents referenced -n slices back; the window returns +n forward.
            return self._focal_window(iy, ix, self._slice, -n, gmax, self._nnw_window)
        Sw = self.S[:, iy][:, :, ix]  # (Bp, wy, wx) parent columns on the window
        return self._nnw_window(Sw, iy, ix)

    def _nnw_window(self, Sw: Tensor, iy: Tensor, ix: Tensor) -> Tensor:
        """De-tilt the window's parents, NNW-combine them, and re-tilt."""
        ny, nx = self.ny, self.nx
        gp = self._parent_signed_device  # (Bp, 2)
        yy = iy.to(torch.float64).view(1, -1, 1)
        xx = ix.to(torch.float64).view(1, 1, -1)
        detilt = torch.exp(
            -2j * pi * (gp[:, 0, None, None] * yy / ny + gp[:, 1, None, None] * xx / nx)
        ).to(self._dtype)
        Sd = Sw * detilt
        recon = torch.einsum("bp,pwv->bwv", self._w, Sd)  # (B, wy, wx)
        if self._mag_preserve:
            # See DetectorExitSMatrix: the complex NNW average shrinks |S| wherever
            # its summands disagree in phase. Keep the phase, restore the magnitude.
            mag = torch.einsum("bp,pwv->bwv", self._w, Sd.abs().to(self._dtype)).real
            recon = mag * (recon / recon.abs().clamp_min(1e-20))
        gy, gx = self._signed_beam_freqs()
        tilt = torch.exp(
            2j * pi * (gy[:, None, None] * yy / ny + gx[:, None, None] * xx / nx)
        ).to(self._dtype)
        return recon * tilt

    def coeffs_at(self, scan_pixels: Tensor) -> Tensor:
        """Per-(scan, beam) synthesis coefficients ``(P, B)`` (cf. ``ScatteringMatrix``).

        ``probe = sum_b coeffs[p, b] * reconstruct_columns()[b]`` reproduces
        :meth:`probe_at_current_plane`.

        Parameters
        ----------
        scan_pixels : Tensor
            Scan positions ``(P, 2)`` ``(Ry, Rx)`` in pixels.

        Returns
        -------
        Tensor
            Synthesis coefficients ``(P, B)`` over the full (unpartitioned)
            aperture beam set.
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

    def probe_at_current_plane(self, scan_pixels: Tensor) -> Tensor:
        """Synthesise probes at the current plane by NNW interpolation.

        Interpolates the de-tilted parent columns. Two paths:

        * **Integer (Nyquist-step) scan** -- the dominant fast case.  The probe
          shift ``basis_p(r - R)`` for an integer ``R`` is a cyclic ROLL of the
          real-space beamlet basis ``basis_p = ifft2(Bq_p)``, accumulated in
          PARENT CHUNKS (budget ``_PROBE_PARENT_CHUNK_ELEMS``) so the transient
          stays ~3 chunk-sized buffers instead of 4-5 full ``(Bp, Ny, Nx)``
          tensors.
        * **Sub-pixel scan** -- the same parent chunks: ``R`` splits into the
          integer roll and a remainder ramp on the chunk's spectrum, one iFFT
          per (position, parent).  The transient is chunk-sized, not
          per-position: at the default budget (one chunk of all parents) and
          four positions it peaks at 6.4 full tensors, against 11.4 for the
          old full-memory branch (64 px test fixture).

        Parameters
        ----------
        scan_pixels : Tensor
            Scan positions ``(P, 2)`` ``(Ry, Rx)`` in pixels.

        Returns
        -------
        Tensor
            Real-space probes ``(P, Ny, Nx)``.
        """
        ny, nx = self.ny, self.nx
        norm = float(ny * nx)
        # Fresnel referencing: columns referenced to an earlier plane before
        # fusion, the probe propagated forward at the end (exact at full Bp).
        n_half = self._backprop_n_half()
        ry = scan_pixels[:, 0].to(torch.float64)
        rx = scan_pixels[:, 1].to(torch.float64)
        P = scan_pixels.shape[0]

        ry_i, rx_i = torch.round(ry), torch.round(rx)
        # R = integer roll + remainder ramp on the spectrum: exact (a roll is the
        # ramp at integer frequencies), and it keeps the parent-chunk bound.
        if torch.allclose(ry, ry_i) and torch.allclose(rx, rx_i):
            fy = fx = torch.zeros_like(ry)
        else:
            fy, fx = ry - ry_i, rx - rx_i
        frac = ((fy != 0) | (fx != 0)).tolist()
        sy = (ry_i.long() % ny).tolist()
        sx = (rx_i.long() % nx).tolist()
        out = torch.zeros((P, ny, nx), dtype=self._dtype, device=self._device)
        fres_b = self._fresnel_pow(n_half, -1.0) if n_half != 0.0 else None
        gp = self._parent_signed_device  # (Bp, 2)
        yy = torch.arange(ny, device=self._device, dtype=torch.float64).view(1, ny, 1)
        xx = torch.arange(nx, device=self._device, dtype=torch.float64).view(1, 1, nx)
        cw_t = self._cw.t()  # (Bp, B)
        # The per-chunk de-tilt + beamlet basis inline _detilted_S /
        # _beamlet_spectrum on a parent SLICE (the helpers would build full-Bp
        # buffers); keep the three conventions in sync.
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
            del detilt, S_c
            Bq_c = torch.zeros(
                (p1 - p0, ny, nx), dtype=self._dtype, device=self._device
            )
            Bq_c[:, self._by, self._bx] = cw_t[p0:p1]
            # Whole-pixel rows roll one shared basis; Bq_c keeps its buffer only
            # for rows with a remainder (an unused name is an alias, not a
            # buffer).  Each position's b / shifted is freed before the next.
            basis_c = Bq_c if all(frac) else torch.fft.ifft2(Bq_c, dim=(-2, -1)) * norm
            if not any(frac):
                Bq_c = basis_c  # drops the spectrum's buffer
            # The roll is re-paid per chunk when P > 1 (O(chunks*P)); the
            # dominant inelastic caller passes a single centroid probe.
            for k in range(P):
                b = basis_c
                if frac[k]:  # sub-pixel remainder: one iFFT per (chunk, position)
                    ph = self._fy * fy[k] / ny + self._fx * fx[k] / nx
                    ramp = torch.exp(-2j * pi * ph).to(self._dtype)
                    b = torch.fft.ifft2(Bq_c * ramp, dim=(-2, -1)).mul_(norm)
                shifted = torch.roll(b, shifts=(sy[k], sx[k]), dims=(-2, -1))
                del b
                out[k] += (S_dt_c * shifted).sum(0)
                del shifted

        if n_half != 0.0:
            # Forward-propagate the synthesised probe back to the current plane.
            out = torch.fft.ifft2(
                torch.fft.fft2(out, dim=(-2, -1)) * self._fresnel_pow(n_half, 1.0),
                dim=(-2, -1),
            )
        return out
