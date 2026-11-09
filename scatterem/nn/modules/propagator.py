from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import Size, Tensor
from torch.nn import Module

from scatterem.physics._frequency_grid import fft_frequency_grid
from scatterem.physics.propagation import (
    angular_spectrum_kernel,
    propagator_kernel,
    propagator_phase,
    tilt_phase,
)
from scatterem.utils.env import env_flag

_BCAST_CMUL = None


def _bcast_cmul_(
    w: Tensor, plane: Tensor, plane_first: bool, scale: float | None = None
) -> bool:
    """In-place broadcast complex multiply ``w[b] *= plane``; ``False`` = use torch.

    Thin lazy-import shim in front of
    :func:`scatterem.nn.functional._bcast_cmul_triton.bcast_cmul_`, so that
    ``import scatterem`` never requires Triton (the CPU torch builds do not ship
    it).  That module holds the mechanism, the gates and -- load-bearing -- the
    bit-exactness contract: it reproduces the torch op it replaces to the bit, or
    it declines.  Callers keep their own torch expression for the ``False`` case.
    """
    # Under torch.compile, decline BEFORE anything else and let the caller's own torch
    # expression be traced.  Dynamo constant-folds this test at trace time, so the
    # whole shim -- the lazy import, the os.environ read and the tensor-property gates
    # below -- vanishes from the graph rather than becoming a graph break plus a set of
    # new guards.  Without it a compiled caller recompiles
    # across steps.  The kernel would decline at that shape anyway.
    if torch.compiler.is_compiling():
        return False
    global _BCAST_CMUL
    if _BCAST_CMUL is None:
        try:
            from scatterem.nn.functional._bcast_cmul_triton import bcast_cmul_

            _BCAST_CMUL = bcast_cmul_
        except Exception:  # no triton / no CUDA build -- torch path only
            _BCAST_CMUL = False
    return bool(_BCAST_CMUL) and _BCAST_CMUL(w, plane, plane_first, scale=scale)


def _sampling_yx(
    dx: float | Sequence[float] | np.ndarray,
) -> tuple[float, float]:
    """Broadcast ``dx`` to a ``(dy, dx)`` pair of floats."""
    dyx = np.broadcast_to(np.asarray(dx, dtype=float), (2,))
    return float(dyx[0]), float(dyx[1])


def _extent(shape: Sequence[int], dx: float | Sequence[float] | np.ndarray) -> tuple:
    """Real-space extent ``(Ly, Lx)`` from pixel size."""
    dy, dx_ = _sampling_yx(dx)
    return (int(shape[-2]) * dy, int(shape[-1]) * dx_)


class Propagator(Module):
    """Fresnel/angular-spectrum single-slice propagator for multislice simulation.

    Precomputes a ``(MY, MX)`` transfer-function phase ramp for propagating a
    wave field by ``slice_thickness`` and applies it in :meth:`forward` via an
    FFT round-trip.  The ramp is built by
    :func:`~scatterem.physics.propagation.propagator_kernel` (paraxial) or
    :func:`~scatterem.physics.propagation.angular_spectrum_kernel` (exact).

    Parameters
    ----------
    slice_thickness : float
        Propagation distance per slice, in the same length units as ``dx``.
    wavelength : float
        Electron wavelength, in the same length units as ``dx``.
    dx : float or Sequence of float or numpy.ndarray
        Real-space pixel size, scalar or per-axis.
    shape : tuple or torch.Size or numpy.ndarray
        Simulation grid shape ``(MY, MX)``.
    device : str or torch.device or None, optional
        Device to build the phase ramp on.
    method : str, optional
        ``"angular_spectrum"``/``"asm"`` (exact, default) or
        ``"fresnel"``/``"paraxial"`` (paraxial approximation).
    tilt : Sequence of float or None, optional
        Global crystal tilt ``(theta_y, theta_x)`` in RADIANS (adaptive
        propagator ptychography, Sha et al. 2022). Multiplies ``prop`` by
        ``exp(+2*pi*i * dz * (qy*tan(theta_y) + qx*tan(theta_x)))``, PtyRAD's
        kernel: the wave is translated by ``-dz*tan(theta)`` per slice, i.e.
        the object's columns advance by ``+dz*tan(theta)`` per slice relative
        to the beam. Only the physical kernel ``prop`` carries it; the
        reference-plane :meth:`fresnel_kernel`/:meth:`fresnel_phase` stay
        untilted. ``None`` or all-zero leaves ``prop`` bit-identical.
    """

    def __init__(
        self,
        slice_thickness: float,
        wavelength: float,
        dx: float | Sequence[float] | np.ndarray,
        shape: tuple | Size | np.ndarray,
        device: str | torch.device | None = "cpu",
        method: str = "angular_spectrum",
        tilt: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        critical_thickness = (
            min(
                np.asarray(shape)[-2:]
                * np.broadcast_to(np.asarray(dx, dtype=float), (2,)) ** 2
            )
            / wavelength
        )
        if abs(slice_thickness) > critical_thickness:
            raise ValueError(
                "Fresnel transfer-function propagation may be undersampled: "
                "slice_thickness should be <= min(Ny*dy^2, Nx*dx^2)/wavelength"
            )
        # _shape kept as an attribute so PrismScanOp can verify the propagator
        # is sized for the simulation grid before building the S-matrix.
        self._shape = tuple(shape)
        self.method = method
        # Physical parameters kept so a kernel can be rebuilt for a DIFFERENT
        # number of slices or a DIFFERENT grid. PrismScanOp needs both: it
        # back-propagates the S-matrix by a fractional slice count on the
        # (SY, SX) grid and must then re-propagate the projected patches
        # forward on the smaller (M1, M2) patch grid. A patch is a crop, so it
        # keeps the same real-space pixel size and only the grid shape changes.
        self._slice_thickness = float(slice_thickness)
        self._wavelength = float(wavelength)
        self._dx = dx
        extent = _extent(self._shape, dx)
        if method in ("angular_spectrum", "asm"):
            prop = angular_spectrum_kernel(
                self._shape[-2:],
                extent,
                wavelength,
                slice_thickness,
                device=device,
            )
        elif method in ("fresnel", "paraxial"):
            prop = propagator_kernel(
                self._shape[-2:],
                extent,
                wavelength,
                slice_thickness,
                device=device,
            )
        else:
            raise ValueError(
                f"Propagator method must be 'angular_spectrum'/'asm' or "
                f"'fresnel'/'paraxial'; got {method!r}"
            )
        q = fft_frequency_grid(shape, _sampling_yx(dx), device=device)
        prop = self._fold_tilt(prop, q, tilt)
        # A registered buffer, so `.to(device)` moves the ramp with the module;
        # as a plain attribute it stayed on the construction device and the
        # forward's `Psi * ramp` then raised on a cross-device multiply.
        # NON-persistent: `prop` is fully determined by the constructor
        # arguments, so it does not belong in `state_dict()` and this does not
        # change what a checkpoint contains.
        self.register_buffer("prop", prop, persistent=False)
        # Optional antialiasing aperture folded into `prop` (see
        # fold_antialias_aperture); None on every reconstruction path.  Kept as
        # a plain attribute rather than a buffer: it is never multiplied here
        # (it is already inside `prop`), only tested for presence and
        # identity-compared against the transmission-side aperture, so it must
        # not be swapped for a copy when the module is moved.
        self.antialias_aperture: Tensor | None = None

    def _fold_tilt(
        self, prop: Tensor, q: Tensor, tilt: Sequence[float] | None
    ) -> Tensor:
        """Record ``tilt`` and multiply its phase ramp into ``prop``.

        Skipped for ``None`` or an all-zero tilt, so ``prop`` stays
        bit-identical to the untilted kernel.
        """
        self.tilt = None if tilt is None else (float(tilt[0]), float(tilt[1]))
        if self.tilt is None or not any(self.tilt):
            return prop
        phase = tilt_phase(q, self._slice_thickness, self.tilt)
        return prop * torch.polar(torch.ones_like(phase), phase)

    def fold_antialias_aperture(self, aperture: Tensor) -> None:
        """Multiply an antialiasing aperture into the propagation kernel.

        Band-limiting the wave on every propagation, not only the object, is
        part of a band-limited multislice.  The aperture is supplied by the
        caller because it must have the same cut radius in 1/Å as the one the
        transmission function was masked with, and only the owner of both grids
        can check that.  Its rim width is set by this propagator's grid, so on a
        grid of another size the taper differs from the transmission's.  Once it is folded in the kernel is no longer unitary:
        it deletes the stop band on every step, so a masked multislice does not
        conserve probe intensity.  Not idempotent -- a second call would square
        the rim taper into a narrower, wrong aperture -- so it is refused;
        rebuild the propagator to change the aperture.

        Parameters
        ----------
        aperture : Tensor
            Real ``(Ny, Nx)`` aperture on this propagator's own grid, corner
            origin (the same FFT layout as ``prop``).

        Raises
        ------
        ValueError
            If *aperture* is not shaped like the kernel, or if an aperture has
            already been folded in.
        """
        if self.antialias_aperture is not None:
            raise ValueError(
                "an antialias aperture is already folded into this propagator's "
                "kernel; folding a second one would square the rim taper into a "
                "narrower, wrong aperture. Build a fresh Propagator instead."
            )
        # Compare FULL shapes, not the trailing two: ``prop`` is exactly rank 2,
        # so a trailing-axes test would admit a ``(1, Ny, Nx)`` aperture and
        # ``prop * aperture`` would then silently broadcast the kernel to rank 3.
        if tuple(aperture.shape) != tuple(self.prop.shape):
            raise ValueError(
                f"antialias aperture shape {tuple(aperture.shape)} != propagator "
                f"kernel shape {tuple(self.prop.shape)}"
            )
        self.antialias_aperture = aperture
        # Assigning a tensor to a registered buffer's name routes back into
        # `_buffers` (and stays non-persistent), so the masked kernel is still
        # the buffer that `.to(device)` moves with the module.
        self.prop = self.prop * aperture.to(device=self.prop.device)

    def fresnel_kernel(
        self,
        n_slices: float,
        shape: tuple | Size | None = None,
        device: str | torch.device | None = None,
    ) -> Tensor:
        """Pure-phase transfer for ``n_slices`` slices, on ``shape`` (default: own).

        ``n_slices`` may be fractional and may be negative (back-propagation).

        This exists because ``torch.angle(self.prop)`` is NOT a usable substitute:
        ``prop`` is a unit-modulus complex exponential, so ``angle`` wraps its
        phase into (-pi, pi]. Raising a wrapped phase to a **non-integer** power
        is wrong wherever the true phase left that interval -- on a realistic
        grid the per-slice phase reaches several radians, so a few percent of
        pixels come out wrong by up to the full 2.0 available to a unit-modulus
        value. Rebuilding from the unwrapped physical phase avoids that entirely.

        It is the REFERENCE-PLANE transfer, and it never carries the antialias
        aperture: its consumers (the PRISM focal back-propagation and its
        undo) move the field to a plane and back, and the two
        transfers must cancel exactly.  A masked pair would leave the rim taper
        squared instead.  The one-step PHYSICAL kernel is ``self.prop``, which
        is where :meth:`fold_antialias_aperture` puts the aperture; on an
        unmasked, untilted propagator ``fresnel_kernel(1.0)`` is bit-identical to
        it.

        Parameters
        ----------
        n_slices : float
            Number of slices to propagate by; may be fractional or negative
            (back-propagation).
        shape : tuple or torch.Size or None, optional
            Grid shape to build the kernel on; defaults to this propagator's
            own ``shape``.
        device : str or torch.device or None, optional
            Device to build the kernel on; defaults to ``self.prop.device``.

        Returns
        -------
        torch.Tensor
            Complex unit-modulus transfer-function phase ramp, shape ``shape``.
        """
        shape_t = self._shape if shape is None else tuple(shape)
        if device is None:
            device = self.prop.device
        dz = self._slice_thickness * float(n_slices)
        return torch.exp(
            1j
            * propagator_phase(
                shape_t[-2:],
                _extent(shape_t, self._dx),
                self._wavelength,
                dz,
                method=self.method,
                device=device,
            )
        ).to(self.prop.dtype)

    def fresnel_phase(
        self,
        shape: tuple | Size | None = None,
        device: str | torch.device | None = None,
    ) -> Tensor:
        """Unwrapped phase of ONE slice of free-space propagation.

        The real ``phi`` with ``fresnel_kernel(n) == exp(1j * n * phi)``, never
        carrying the antialias aperture.  It exists because the alternative,
        ``torch.angle(fresnel_kernel(1.0))``, is wrapped into (-pi, pi] and may
        not be scaled by a fractional slice count (see :meth:`fresnel_kernel`).
        The S-matrix column preparation reads a beam's vacuum phase off it
        (:mod:`scatterem.nn.scan_exit_wave._columns`).

        Parameters
        ----------
        shape : tuple or torch.Size or None, optional
            Grid shape to build the phase on; defaults to this propagator's
            own ``shape``.
        device : str or torch.device or None, optional
            Device to build the phase on; defaults to ``self.prop.device``.

        Returns
        -------
        torch.Tensor
            Real phase in radians, shape ``shape``, corner origin.
        """
        shape_t = self._shape if shape is None else tuple(shape)
        if device is None:
            device = self.prop.device
        return propagator_phase(
            shape_t[-2:],
            _extent(shape_t, self._dx),
            self._wavelength,
            self._slice_thickness,
            method=self.method,
            device=device,
        ).to(dtype=torch.float32)

    def _scaled_ramp(self, ramp: Tensor, n: int) -> Tensor:
        """``ramp * (1.0 / n)``, memoised in one slot on the instance.

        An L2-blocked, graph-replayed slice loop calls this propagator once per
        (block, slice), so rebuilding the ``[MY, MX]`` plane per call was 992
        launches per ``r3_simulator/medium/simulate`` row (4.5 % of its device
        time).  Same multiply on the same operands, so bit-exact.  Keyed on the
        ``ramp`` object (held, so its ``id`` cannot be reused), its version and
        inference mode; skipped under ``torch.compile``, for a ``ramp`` that
        carries a graph, and never populated inside a CUDA-graph capture (a
        capture only reads an entry made eagerly, which lives as long as this
        module).
        """
        if torch.compiler.is_compiling() or ramp.requires_grad:
            return ramp * (1.0 / n)
        key = (ramp._version, n, torch.is_inference_mode_enabled())
        hit = self.__dict__.get("_ramp_memo")
        if hit is not None and hit[0] is ramp and hit[1] == key:
            return hit[2]
        scaled = ramp * (1.0 / n)
        if not (scaled.is_cuda and torch.cuda.is_current_stream_capturing()):
            self.__dict__["_ramp_memo"] = (ramp, key, scaled)
        return scaled

    def _kernel(self) -> Tensor:
        """The one-step kernel :meth:`forward` applies (a hook for subclasses)."""
        return self.prop

    def forward(
        self, waves: Tensor, defer_norm: bool = False
    ) -> Tensor | tuple[Tensor, float]:
        """Propagate a wave field by one slice thickness.

        Parameters
        ----------
        waves : torch.Tensor
            Wave field, shape ``(Nmodes, K, M1, M2)``, complex.
        defer_norm : bool, optional
            When True, return ``(waves, inv_n)`` and do NOT apply the inverse
            transform's ``1/N`` -- the caller has undertaken to apply
            ``inv_n`` itself, fused into the multiply it makes next. ``inv_n``
            is ``1.0`` whenever nothing was deferred (a power-of-two grid
            folds it onto the ramp below for free, and complex32 needs
            "ortho"), so the caller needs no copy of the predicate. Only ever
            taken on the grids where the fold is NOT bit-exact and torch
            therefore runs a separate full-field scaling pass; see
            :mod:`scatterem.nn.functional._bcast_cmul_triton`.

        Returns
        -------
        torch.Tensor or tuple of (torch.Tensor, float)
            Propagated wave field, shape ``(Nmodes, K, M1, M2)``; or, when
            ``defer_norm`` is True, ``(waves, inv_n)`` with the outstanding
            inverse-transform scale factor.
        """
        in_dtype = waves.dtype
        # Run the FFT round-trip in fp32 for an fp16 wave field when (a) forced via
        # SCATTEREM_FP32_FFT (FFTs are the dominant fp16 precision-loss op), or (b)
        # the grid is NOT a power of two on either axis -- cuFFT has no half-
        # precision plan there and raises CUFFT_INVALID_SIZE (e.g. anisotropic-f
        # tips with N=(2944,1536)). The cast is per-call/transient; the stored
        # waves stay complex32. Power-of-2 fp16 grids keep the half FFT (the
        # validated fp16 path is unchanged).
        my, mx = int(waves.shape[-2]), int(waves.shape[-1])
        non_pow2 = (my & (my - 1)) != 0 or (mx & (mx - 1)) != 0
        force_fp32 = env_flag("SCATTEREM_FP32_FFT")
        if in_dtype == torch.complex32 and (force_fp32 or non_pow2):
            waves = waves.to(torch.complex64)
        # The fft2/ifft2 round-trip total scaling is 1/N regardless of `norm`,
        # so the result is identical for bare vs "ortho". Use bare (faster, one
        # fewer scaling pass) for fp32, and "ortho" for complex32 — the bare
        # intermediate after fft2 is unnormalized (x N, overflows fp16) whereas
        # ortho keeps it at 1/sqrt(N). In-place `*=` preserves `waves`' dtype.
        norm = "ortho" if waves.dtype == torch.complex32 else None
        waves = torch.fft.fft2(waves, norm=norm)
        ramp = self._kernel().to(waves.dtype)
        inv_norm = norm
        if norm is None and not non_pow2:
            # Bare inverse: torch's default ("backward") inverse normalization
            # applies the round-trip's 1/N as a SEPARATE full-field scaling pass
            # over `waves`. Ride the 1/N in on the phase ramp that is already
            # being multiplied here and take the UNNORMALIZED inverse instead
            # (`norm="forward"` on an inverse transform = no scaling), which
            # removes that pass entirely. The ramp is
            # [MY,MX], so the extra multiply is negligible next to the
            # [Nmodes,K,MY,MX] pass it replaces.
            #
            # Gated on a power-of-two grid because that is exactly where the fold
            # is BIT-IDENTICAL to scaling after the ifft2 (1/N is then an exact
            # binary scaling, so it commutes through the multiply). Measured: on
            # a power-of-two grid the forward reproduces to the sha256; on 96^2
            # it does not, so those grids deliberately keep
            # the extra scaling pass rather than trade exactness for 1.14x.
            # NOT done for complex32 either: there "ortho" is load-bearing (the
            # bare post-fft2 intermediate is xN and overflows fp16), and 1/N
            # would push the ramp itself into fp16 subnormals.
            ramp = self._scaled_ramp(ramp, waves.shape[-2] * waves.shape[-1])
            inv_norm = "forward"
        # Non-power-of-two grid, complex64: the fold above is unavailable, so the
        # 1/N would cost a whole extra read+write pass over `waves`.  Hand it to the
        # caller instead of paying for it (`defer_norm`), which is bit-exact because
        # `ifft2(norm=None)` IS `ifft2(norm="forward") * (1/N)` bitwise -- verified
        # at 96^2/128^2/192^2, three spellings each.  Restricted to complex64 in AND
        # out: for a complex32 caller the `.to(in_dtype)` below would round an
        # unnormalized (xN) field into fp16 and overflow.
        deferred = 1.0
        if defer_norm and inv_norm is None and in_dtype == torch.complex64:
            inv_norm = "forward"
            deferred = 1.0 / (waves.shape[-2] * waves.shape[-1])
        if not _bcast_cmul_(waves, ramp, False):
            waves *= ramp
        waves = torch.fft.ifft2(waves, norm=inv_norm)
        waves = waves.to(in_dtype)
        return (waves, deferred) if defer_norm else waves
