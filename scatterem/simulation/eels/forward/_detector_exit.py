"""The detector-side exit scattering matrix ``S2``, and its crop windows.

Split out of ``prism_eels_image.py``: ``S2`` is built per slice and peeled
backwards, which is a self-contained piece of the double-channeling image --
the contraction body only ever asks it for columns on a crop window.

Code motion out of that module, plus ``mag_preserve``, the ``dechirp``
(on by default) and the optional ``focal_backprop`` referencing of the
partitioned branch (see :class:`DetectorExitSMatrix`), and the corrected
de-tilt sign of that branch. Nothing here reassociates a sum or changes a
tensor's shape or strides, so the bit-identity contracts the contraction body
documents are unaffected; the unpartitioned path is unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import pi

import numpy as np
import torch
from torch import Tensor

from scatterem.physics.propagation import _mul_kernel, _scaled_propagator, propagate
from scatterem.simulation._wave.exit_walk import _unit_transmission_slices
from scatterem.simulation._wave.focal_reference import (
    _focal_slices,
    _FocalReferenceMixin,
)
from scatterem.simulation._wave.prism_beams import _partition_weights


def _conj_propagate(waves: Tensor, kernel: Tensor) -> Tensor:
    """Inverse Fresnel step ``ifft(fft(.) * conj(kernel))`` (kernel is even).

    ``ifft2``'s default ``norm="backward"`` applies its ``1/(MY*MX)`` as a
    SEPARATE full-stack pass after the transform: at the live
    ``r7_eels/large/image`` shape ((37, 256, 256) complex64, 18.50 MiB) the
    inverse costs 0.1757 ms against the forward's 0.1175, and the 0.0582 ms
    difference is 16.3 % of the 0.3562 ms round trip.
    :func:`~scatterem.physics.propagation.propagate` already
    deletes exactly that pass by transforming with ``norm="forward"`` and
    folding the scale into the propagator plane instead -- one pass over a
    single ``(MY, MX)`` plane, and through :func:`_scaled_propagator`'s memo
    only on the first call for a given propagator.  Its docstring records that
    the fold was deliberately kept out of ``propagator_kernel`` *because this
    function normalises its own inverse with the same kernel*; this is that
    fold, made on this side.

    The fold is taken ONLY when ``MY*MX`` is a power of two, because that is
    exactly when it is BIT-EXACT rather than merely equivalent: scaling the
    plane by ``1/(MY*MX)`` is then a lossless binary exponent shift, so
    ``x * (conj(k)/n)`` and ``(x * conj(k))/n`` round identically.  Measured at
    256 px (``n = 65536``): ``torch.equal`` True and the round trip goes
    0.3562 -> 0.2958 ms = **1.2043x**, worth 1.0405x on ``large/image`` where
    this leg is 23 % of the row.  At 192 px (``n = 36864 = 2**12 * 9``) the same
    fold is worth more still (1.3049x) but moves the result by 2.3e-7 of peak,
    which this target's sha256 digest gate would -- correctly -- refuse, so the
    shipped out-of-place spelling is kept there.  It is also kept for the
    autograd path, where ``_mul_kernel``'s in-place write would clobber a value
    the graph needs.
    """
    n = int(waves.shape[-2]) * int(waves.shape[-1])
    if n & (n - 1) == 0 and not (
        kernel.requires_grad or (torch.is_grad_enabled() and waves.requires_grad)
    ):
        return torch.fft.ifft2(
            _mul_kernel(
                torch.fft.fft2(waves, dim=(-2, -1)),
                _scaled_propagator(kernel, n).conj(),
            ),
            dim=(-2, -1),
            norm="forward",
        )
    return torch.fft.ifft2(
        torch.fft.fft2(waves, dim=(-2, -1)) * kernel.conj(), dim=(-2, -1)
    )


class DetectorExitSMatrix(_FocalReferenceMixin):
    """S2: columns are detector-output beams back-propagated to the current slice.

    ``S2_d(r)`` is the adjoint (transpose) multislice of the conjugate detector
    plane wave ``e_d*`` from the exit surface back to the current slice, so that
    for any wave ``w`` at that slice ``sum_r S2_d(r) w(r) == fft(forward_exit(w))[d]``
    -- the amplitude reaching detector beam ``d``. Built once covering all slices
    (slice 0 -> exit) and :meth:`peel_to` advances the start slice forward (removing
    front slices) so it is reused across every atom.

    With ``partition`` the transpose multislice is built on only ``Bp`` *parent*
    detector beams (a hex-ring subsample); the full detector columns are
    NNW-reconstructed on a crop window via :meth:`columns_window`. By reciprocity
    the de-tilted back-propagated columns are locally similar, so the same
    partitioned-PRISM interpolation applies -- shrinking the resident S2 from
    ``ndet`` to ``Bp`` columns.

    Parameters
    ----------
    det_idx : Tensor
        Corner-origin pixel indices of the accepted detector beams, shape
        ``(ndet, 2)``, integer dtype.
    transmissions : Tensor
        Per-slice transmission functions, shape ``(nz, ny, nx)``, complex dtype;
        determines the grid shape and the build device/dtype.
    kernel : Tensor
        Fresnel propagator kernel for one slice step, shape ``(ny, nx)``, complex
        dtype.
    partition : dict or None
        When given, enables partitioned (parent-beam) construction; recognized
        keys are ``"n_radial"`` and ``"n_angular"``, the hex-ring subsample
        resolution passed to the NNW weighting, and ``"dechirp"`` (default
        True), which removes each column's accumulated vacuum propagator phase
        before the NNW combination and restores it per output beam afterwards.
        ``"dechirp": False`` is the ablation without it. ``None`` builds the
        full ``ndet`` columns directly.
    mag_preserve : bool, default True
        Restore the magnitude of the partitioned reconstruction from the same
        NNW interpolation, keeping the complex sum's phase. Ignored when
        ``partition`` is None (the columns are then exact).
    focal_backprop : float, str or None, optional
        Fresnel referencing of the partitioned parents before interpolation:
        the parents are propagated toward the exit by that fraction of the
        remaining path ``nz - start`` (``"centroid"`` is half of it, rounded to
        whole slices), combined on a padded window, and propagated back.
        ``None`` (default) or ``0`` disables it. It is skipped in the
        full-parent limit, where the columns are exact.
    wavelength : float, optional
        Electron wavelength in Angstrom. With ``slice_distance`` and
        ``gridsize`` it sets the windowed Fresnel step that ``focal_backprop``
        needs.
    slice_distance : float, optional
        Slice distance in Angstrom (see ``wavelength``).
    gridsize : (float, float), optional
        Real-space grid extent ``(Ly, Lx)`` in Angstrom (see ``wavelength``).
    """

    def __init__(
        self,
        det_idx: Tensor,
        transmissions: Tensor,
        kernel: Tensor,
        partition: dict | None = None,
        mag_preserve: bool = True,
        focal_backprop: float | str | None = None,
        wavelength: float | None = None,
        slice_distance: float | None = None,
        gridsize: Sequence[float] | None = None,
    ):
        device = transmissions.device
        dtype = transmissions.dtype
        self.transmissions = transmissions
        self.kernel = kernel
        self._mag_preserve = bool(mag_preserve)
        self._init_focal_reference(focal_backprop, wavelength, slice_distance, gridsize)
        nz, ny, nx = transmissions.shape
        self._nz, self.ny, self.nx = nz, ny, nx
        self.det_idx = det_idx = det_idx.to(device=device)  # (ndet, 2) corner origin
        self.ndet = det_idx.shape[0]
        self._dtype, self._device = dtype, device

        dgy = ((det_idx[:, 0] + ny // 2) % ny - ny // 2).to(torch.float64)
        dgx = ((det_idx[:, 1] + nx // 2) % nx - nx // 2).to(torch.float64)
        self._det_signed = torch.stack([dgy, dgx], dim=1)  # (ndet, 2)

        self.partitioned = partition is not None
        if partition is not None:
            # Parent detector beams + NNW weights (geometry on CPU, like S1).
            all_signed = self._det_signed.cpu().numpy().astype(np.float64)
            pidx, w = _partition_weights(
                det_idx.cpu().numpy(),
                all_signed,
                int(partition.get("n_radial", 4)),
                int(partition.get("n_angular", 6)),
            )
            self._w = torch.as_tensor(w, dtype=dtype, device=device)  # (ndet, Bp)
            pidx_t = torch.as_tensor(np.asarray(pidx), device=device)
            build_signed = self._det_signed[pidx_t]
            self._dechirp = bool(partition.get("dechirp", True))
        else:
            build_signed = self._det_signed
            self._dechirp = False
        self._build_signed = build_signed.to(device)

        # Conjugate plane waves e_d*(r) = exp(-2pi i g_d . r / N) for the build beams.
        ry = torch.arange(ny, device=device, dtype=torch.float64).view(1, ny, 1)
        rx = torch.arange(nx, device=device, dtype=torch.float64).view(1, 1, nx)
        gy = self._build_signed[:, 0]
        gx = self._build_signed[:, 1]
        v = torch.exp(
            -2j * pi * (gy[:, None, None] * ry / ny + gx[:, None, None] * rx / nx)
        ).to(dtype)

        # A slice whose transmission function is exactly ``1 + 0j`` has an identity
        # multiply, so BOTH walks of this matrix skip it -- bit-identically, and the
        # multiply is a full out-of-place pass over the ``(ndet, Ny, Nx)`` stack.
        # ``ScatteringMatrix.advance_to`` has done this since it shipped; this is the
        # same mechanism on the one S-matrix in the package that never got it, and it
        # fires twice per slice here because this matrix walks the chain twice (the
        # backward build below and :meth:`peel_to`'s forward re-walk).  The
        # propagation is *not* skipped, for the reason
        # :func:`~scatterem.simulation._wave.exit_walk._slice_exit_wave`
        # records: composing two Fresnel kernels is not the same float as applying
        # them either side of an FFT round trip.
        self._unit_slices = _unit_transmission_slices(transmissions)

        # Transpose multislice from exit back to slice 0 (adjoint of forward_exit:
        # transmit at every slice, propagate between, in reverse slice order).
        u = v
        for j in range(nz - 1, -1, -1):
            if not self._unit_slices[j]:
                u = u * transmissions[j]
            if j > 0:
                u = propagate(u, kernel)
        self.S = u  # (n_build, ny, nx): ndet columns, or Bp parents if partitioned
        self._start = 0

    def peel_to(self, start: int) -> None:
        """Advance the start slice to ``start`` (remove front slices, reused S2).

        Parameters
        ----------
        start : int
            Target starting slice index; must be >= the current start slice, since
            slices already peeled off cannot be restored.
        """
        for j in range(self._start, start):
            w = (
                self.S
                if self._unit_slices[j]
                else self.S * self.transmissions[j].conj()
            )
            self.S = _conj_propagate(w, self.kernel)
        self._start = start

    def _vacuum_phase(self, beams: Tensor, n_steps: int) -> Tensor:
        """Phase a beam's column accumulates over ``n_steps`` vacuum Fresnel steps.

        The propagator is even, so the phase at ``+g`` is the one a column built
        from the conjugate plane wave ``exp(-2 pi i g.r)`` picks up.

        Parameters
        ----------
        beams : Tensor
            Signed beam indices, shape ``(n, 2)``.
        n_steps : int
            Number of propagation steps between the current slice and the exit.

        Returns
        -------
        Tensor
            Unit-modulus phases, shape ``(n,)``, in the matrix's complex dtype.
        """
        by = beams[:, 0].long() % self.ny
        bx = beams[:, 1].long() % self.nx
        angle = torch.angle(self.kernel)[by, bx].to(torch.float64)
        return torch.exp(1j * float(n_steps) * angle).to(self._dtype)

    def _backprop_slices(self) -> int:
        """Whole slices to propagate the parents toward the exit before NNW.

        Zero in the full-parent limit: the columns are already exact, and a
        cropped Fresnel round trip would only add window error.
        """
        if int(self.S.shape[0]) == int(self.ndet):
            return 0
        return _focal_slices(self.focal_backprop, self._nz - self._start)

    def columns_window(self, iy: Tensor, ix: Tensor) -> Tensor:
        """Detector columns on a crop window ``(ndet, wy, wx)``.

        Direct crop when full; NNW reconstruction from the ``Bp`` parents (de-tilt,
        de-chirp, combine, re-chirp, re-tilt -- all on the window, never the full
        grid) when partitioned. With ``focal_backprop`` the parents are combined
        at the referenced plane on a padded window, then propagated back to the
        current plane.

        Parameters
        ----------
        iy : Tensor
            Row pixel indices of the crop window, 1D integer tensor of length wy.
        ix : Tensor
            Column pixel indices of the crop window, 1D integer tensor of length wx.

        Returns
        -------
        Tensor
            Detector columns on the window, shape ``(ndet, wy, wx)``, complex
            dtype.

        Raises
        ------
        ValueError
            If the referencing is on and the physical scale is missing.
        """
        if not self.partitioned:
            return self.S[:, iy][:, :, ix]
        n = self._backprop_slices()
        if not n:
            Sw = self.S[:, iy][:, :, ix]  # (Bp, wy, wx) parent columns on the window
            return self._nnw_window(Sw, iy, ix)
        # Parents referenced n steps toward the exit, combined there (their
        # vacuum phase over those n extra steps is de-chirped too), and the
        # window propagated back to the current plane.
        return self._focal_window(
            iy,
            ix,
            self._start,
            n,
            float(self._det_signed.abs().max()),
            lambda Sw, py, px: self._nnw_window(Sw, py, px, extra_steps=n),
        )

    def _nnw_window(
        self, Sw: Tensor, iy: Tensor, ix: Tensor, extra_steps: int = 0
    ) -> Tensor:
        """De-tilt and de-chirp the window's parents, NNW-combine, re-chirp, re-tilt.

        ``extra_steps`` counts the Fresnel steps of the optional referencing,
        whose vacuum phase is removed together with the path to the exit.
        """
        ny, nx = self.ny, self.nx
        yy = iy.to(torch.float64).view(1, -1, 1)
        xx = ix.to(torch.float64).view(1, 1, -1)
        gp = self._build_signed
        # An S2 column CARRIES exp(-2 pi i g.r): it is built from the conjugate
        # detector plane wave. Removing it therefore needs exp(+2 pi i g.r), the
        # opposite sign to S1's de-tilt. Inverted, the de-tilt doubles the tilt and
        # the interpolation acts on a rapidly oscillating column (relative error
        # 1.22 in vacuum, against 4e-4 corrected).
        detilt = torch.exp(
            2j * pi * (gp[:, 0, None, None] * yy / ny + gp[:, 1, None, None] * xx / nx)
        ).to(self._dtype)
        Sd = Sw * detilt
        n_steps = self._nz - self._start - 1 + extra_steps  # to the exit
        if self._dechirp:
            Sd = Sd / self._vacuum_phase(gp, n_steps)[:, None, None]
        recon = torch.einsum("dp,pwv->dwv", self._w, Sd)  # (ndet, wy, wx)
        if self._mag_preserve:
            # The NNW reconstruction averages parent columns COMPLEX, so wherever
            # the summands disagree in phase the result's magnitude shrinks -- a
            # systematic underestimate of the exit amplitude, not interpolation
            # error. Keep the complex sum's PHASE and restore the magnitude
            # interpolated the same way.
            mag = torch.einsum("dp,pwv->dwv", self._w, Sd.abs().to(self._dtype)).real
            recon = mag * (recon / recon.abs().clamp_min(1e-20))
        gd = self._det_signed
        if self._dechirp:
            recon = recon * self._vacuum_phase(gd, n_steps)[:, None, None]
        tilt = torch.exp(
            -2j * pi * (gd[:, 0, None, None] * yy / ny + gd[:, 1, None, None] * xx / nx)
        ).to(self._dtype)
        return recon * tilt


def _periodic_window(center: int, width: int, n: int, device) -> Tensor:
    """``width`` indices centred on ``center`` with periodic wraparound.

    For an even ``width`` the window holds one more index below ``center``
    than above it, so ``center`` is at position ``width // 2``.
    """
    half = width // 2
    rng = torch.arange(-half, width - half, device=device)
    return (center + rng) % n


def _periodic_window_stack(centers: np.ndarray, width: int, n: int, device) -> Tensor:
    """:func:`_periodic_window` for many centres at once -- ``(len(centers), width)``.

    Same integer arithmetic as the scalar form (``(center + rng) % n`` in int64), so
    row ``j`` is elementwise equal to ``_periodic_window(int(centers[j]), ...)``; the
    point is that the whole call's windows cost one launch instead of two per atom.
    """
    half = width // 2
    rng = torch.arange(-half, width - half, device=device)
    c = torch.as_tensor(np.asarray(centers, dtype=np.int64), device=device).view(-1, 1)
    return (c + rng) % n
