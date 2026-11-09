"""PRISM scattering-matrix metadata: the beam basis a PRISM forward model
is built on, chosen from an aperture or from an existing probe."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from scatterem.data.s_meta import SMeta
from scatterem.nn.scan_exit_wave._prism_geometry import (
    PROBE_RIM_PX,
    aperture_support,
    derived_prism_grid,
)
from scatterem.physics._frequency_grid import fft_frequency_grid


def _f_lattice_beams(P, f, dx, wavelength, semiconvergence_angle, margin, device):
    """``(MY, MX)`` bool numpy mask of the f-lattice beams (Ophus Eq 6).

    Every f-th corner-origin index where the probe HAS a Fourier coefficient
    (``_geometry.aperture_support``): since ``N = f * P`` is a multiple of ``f``,
    ``index % f == 0`` selects exactly the f-spaced lattice in BOTH the positive
    and aliased-negative halves of the fftfreq ordering, and pixel ``(i, j)`` of
    the ``P`` window is beam ``(f i, f j)`` of ``N``.  A ``margin`` selects the
    legacy disk ``|k| <= margin * alpha / lambda`` instead.
    """
    fy, fx = f
    MY, MX = fy * P[0], fx * P[1]
    on = torch.zeros((MY, MX), dtype=torch.bool, device=device)
    if margin is None:
        on[::fy, ::fx] = aperture_support(
            P, dx, wavelength, semiconvergence_angle, device
        )
    else:
        # CORNER-origin reciprocal coords on the N grid (centered=False).
        qf = fft_frequency_grid((MY, MX), [float(dx[0]), float(dx[1])], device=device)
        disk = torch.linalg.norm(
            qf, dim=0
        ) <= semiconvergence_angle / wavelength * float(margin)
        on[::fy, ::fx] = disk[::fy, ::fx]
    return on.cpu().numpy()


def build_prism_smeta(
    *,
    pixels: tuple[int, int],
    f: int | tuple[int, int],
    dx,
    wavelength: float,
    semiconvergence_angle: float,
    fov_shape: tuple[int, int] | None,
    device,
    n_radial: int | None = None,
    n_angular: int = 6,
    margin: float | None = None,
) -> SMeta:
    """Synthesize a pure- or partitioned-PRISM, **f-lattice** :class:`SMeta` (Ophus 2017).

    Instance-free so the reconstruction path can reuse it for f>1 later (spec
    forward-compat): everything is taken from the explicit arguments, not from a
    simulator instance.

    Parameters
    ----------
    pixels : (int, int)
        Detector window ``(P_y, P_x)``.  The simulation grid is ``N = f * pixels``
        per axis; the detector is ``N / f = pixels`` (dk = f·Δq).
    f : int
        PRISM interpolation factor (``>= 1``).
    dx : array-like (2,)
        Real-space pitch ``Δp = unitcell / pixels`` per axis (Å).  UNCHANGED by
        ``f`` -- the N grid is the larger probe window at the same pitch.
    wavelength : float
        Electron wavelength (Å).
    semiconvergence_angle : float
        Probe semiconvergence angle (rad).
    fov_shape : (int, int) or None
        Extent ``(NY_fov, NX_fov)`` of an object that has to FIT inside the
        simulation grid: a finite specimen, a reconstructed volume, or anything
        scanned with a non-periodic crop.  Raises if ``f * pixels < FOV``.
        ``None`` for a PERIODIC specimen under a periodic crop: the ``N`` grid
        is then the crystal itself and a window leaving it wraps onto the next
        cell, so there is nothing to fit -- which is what lets PRISM at
        ``f = 1`` scan at all.
    device : torch.device or str
        Device the SMeta tensors are built on.
    n_radial : int or None
        Partitioned-PRISM knob.  ``None`` (the default) -> **pure PRISM**: every
        f-lattice beam is its own parent (the behavior below).  When set, after
        the pure full-aperture SMeta is built it is replaced by
        :meth:`SMeta.make_beamlet_meta(n_radial, n_angular) <SMeta.make_beamlet_meta>`,
        which selects ``Bp << B`` parent beams on ``n_radial`` (+ DC) hex rings
        and builds the Voronoi natural-neighbor weights ``(B, Bp)`` + beamlet
        basis ``(Bp, MY, MX)``.
    n_angular : int
        Beamlet hex-ring angular sample count passed through to
        :meth:`SMeta.make_beamlet_meta` (default 6).  Ignored when
        ``n_radial is None``.
    margin : float or None
        ``None`` (the default): the beams are the support of the probe-forming
        aperture (``_geometry.aperture_support``), i.e. every Fourier
        coefficient the probe has.  A number selects the legacy disk
        ``|k| <= margin * alpha / lambda`` instead, which truncates the
        aperture's soft edge when the aperture is small in pixels (it was
        hard-coded at 1.05); kept for the reconstruction model's
        ``prism_spec["aperture"] = "kmax_margin"``.

    Returns
    -------
    SMeta
        When ``n_radial is None``: a pure-PRISM SMeta on the ``N`` grid with an
        f-lattice BF-disk aperture, identity natural-neighbor weights and one-hot
        beamlets.  When ``n_radial`` is set: the partitioned SMeta (parent-beam
        subset ``Bp << B``, Voronoi NNW ``(B, Bp)``, beamlets ``(Bp, MY, MX)``).

    Raises
    ------
    ValueError
        If ``f * pixels < FOV`` (object does not fit) -- the message tells the
        caller to increase ``pixels`` so ``f * pixels >= scan_extent + detector``.
    """
    # Anisotropic interpolation factor (f_y, f_x): a scalar broadcasts to both
    # axes. Long-narrow specimens (tips) want f_y >> f_x so the narrow axis keeps
    # a small N_x while the long axis gets a large N_y, instead of a wasteful
    # square N = max(f)*pixels grid.
    fy, fx = (int(f), int(f)) if isinstance(f, int) else (int(f[0]), int(f[1]))
    P = tuple(int(p) for p in pixels)
    dx = np.asarray(dx, dtype=np.float32)

    MY, MX = derived_prism_grid((P[0], P[1]), (fy, fx))

    # An object that is not the periodic N grid itself must fit inside it.
    if fov_shape is not None and (MY < int(fov_shape[0]) or MX < int(fov_shape[1])):
        raise ValueError(
            f"PRISM simulation grid N=(fy,fx)*pixels=({MY}, {MX}) is smaller than "
            f"the object FOV ({int(fov_shape[0])}, {int(fov_shape[1])}); the object "
            f"does not fit. Increase `pixels`/`f` so that (fy,fx)*pixels >= "
            f"scan_extent + detector (fy={fy}, fx={fx}, pixels={P})."
        )

    kmax = semiconvergence_angle / wavelength
    all_beams = _f_lattice_beams(
        P, (fy, fx), dx, wavelength, semiconvergence_angle, margin, device
    )

    # BF-disk radius in pixels: kmax / dk, dk = 1 / (N * dx) along axis 0.
    dk0 = 1.0 / (MY * float(dx[0]))
    R = int(round(kmax / dk0))

    s_meta = SMeta(
        all_beams=all_beams,
        parent_beams=all_beams,
        numerical_aperture_radius_pixels=R,
        dx=dx,
        M=(MY, MX),
        device=torch.device(device),
    )
    # Pure PRISM: identity natural-neighbor weights + one-hot beamlets.
    s_meta.natural_neighbor_weights = torch.eye(
        s_meta.Bp, dtype=torch.float32, device=device
    )
    # Pure PRISM: the beamlet weights AT the beams are the identity (each beam is
    # its own parent). Store (Bp, B) directly -- not a dense (Bp, MY, MX) one-hot
    # (mostly zeros, 10s of GB on large grids; the consumer only reads the beams).
    s_meta.beamlets = torch.eye(s_meta.Bp, dtype=torch.float32, device=device)

    if n_radial is not None:
        # Partitioned PRISM: replace the pure full-aperture SMeta with a
        # parent-beam subset (Bp << B) + Voronoi natural-neighbor weights +
        # beamlet basis. make_beamlet_meta is CPU/device-agnostic and reuses the
        # f-lattice ``all_beams`` above, so it works for any f. Default
        # (n_radial is None) returns the pure SMeta unchanged (M2d behavior).
        # The rings span the APERTURE -- its radius plus the width of its edge, one
        # pixel of the P window, i.e. f beams -- not the ~4.5 px beyond it in which
        # that edge falls to zero: those beams are in the set so the expansion is
        # complete, and carry no weight.
        s_meta = s_meta.make_beamlet_meta(
            n_radial_samples=int(n_radial),
            n_angular_samples=int(n_angular),
            radius=None if margin is not None else kmax / dk0 + fy * PROBE_RIM_PX,
        )
    return s_meta


def _build_prism_smeta_at_radius(
    *,
    radius: float,
    pixels: tuple[int, int],
    f: int,
    dx,
    fov_shape: tuple[int, int],
    device,
    n_radial: int | None = None,
    n_angular: int = 6,
) -> SMeta:
    """Build the f-lattice BF-disk :class:`SMeta` at an explicit aperture ``radius`` (1/Aa).

    Factored out of :func:`build_prism_smeta` so the power-based sizer can pick
    its own radius while reusing the SAME f-lattice mask, ``R`` and beamlet/NNW
    construction.  ``build_prism_smeta`` keeps its anchored ``kmax*1.05`` radius;
    this helper takes the radius as an argument (the power-shell radius for
    :func:`build_prism_smeta_for_probe`).
    """
    f = int(f)
    P = tuple(int(p) for p in pixels)
    dx = np.asarray(dx, dtype=np.float32)

    MY, MX = f * P[0], f * P[1]
    fov_y, fov_x = int(fov_shape[0]), int(fov_shape[1])
    if MY < fov_y or MX < fov_x:
        raise ValueError(
            f"PRISM simulation grid N=f*pixels=({MY}, {MX}) is smaller than the "
            f"object FOV ({fov_y}, {fov_x}); the object does not fit. Increase "
            f"`pixels` so that f*pixels >= scan_extent + detector "
            f"(f={f}, pixels={P})."
        )

    qf = fft_frequency_grid((MY, MX), [float(dx[0]), float(dx[1])], device=device)
    qmag = torch.linalg.norm(qf, dim=0)
    iy = torch.arange(MY, device=device)[:, None]
    ix = torch.arange(MX, device=device)[None, :]
    flattice = ((iy % f) == 0) & ((ix % f) == 0)
    all_beams = ((qmag <= radius) & flattice).cpu().numpy()

    dk0 = 1.0 / (MY * float(dx[0]))
    R = int(round(radius / dk0))

    s_meta = SMeta(
        all_beams=all_beams,
        parent_beams=all_beams,
        numerical_aperture_radius_pixels=R,
        dx=dx,
        M=(MY, MX),
        device=torch.device(device),
    )
    s_meta.natural_neighbor_weights = torch.eye(
        s_meta.Bp, dtype=torch.float32, device=device
    )
    # Pure PRISM: the beamlet weights AT the beams are the identity (each beam is
    # its own parent). Store (Bp, B) directly -- not a dense (Bp, MY, MX) one-hot
    # (mostly zeros, 10s of GB on large grids; the consumer only reads the beams).
    s_meta.beamlets = torch.eye(s_meta.Bp, dtype=torch.float32, device=device)

    if n_radial is not None:
        s_meta = s_meta.make_beamlet_meta(
            n_radial_samples=int(n_radial), n_angular_samples=int(n_angular)
        )
    return s_meta


def build_prism_smeta_for_probe(
    *,
    probe: Tensor,
    pixels: tuple[int, int],
    f: int,
    dx,
    wavelength: float,
    semiconvergence_angle: float,
    fov_shape: tuple[int, int],
    device,
    power_fraction: float = 0.9999,
    n_radial: int | None = None,
    n_angular: int = 6,
) -> SMeta:
    """Power-based f-lattice-shell PRISM :class:`SMeta` from the probe (faithful PRISM).

    Sizes the BF-disk aperture from the *incoherent* multi-mode probe power
    spectrum instead of the fixed ``kmax*1.05`` margin of
    :func:`build_prism_smeta`.  The f-lattice only has beams at every ``f``-th
    N-grid frequency pixel, so the candidate aperture radii are the f-lattice
    shells ``{0, f*dk, 2f*dk, ...}`` (``dk = 1/(N*dx)``).  The chosen radius is
    the SMALLEST shell whose cumulative power reaches ``power_fraction`` of the
    total, **floored at ``kmax*1.2``** so the aperture is never tighter than
    1.2x the convergence angle.  This captures the soft (tanh) probe-edge tail
    the fixed margin truncates -- the too-tight-aperture footgun for f-lattice
    PRISM (Ophus 2017).

    Self-contained and reusable by both the reconstruction and simulation paths
    (everything is taken from the explicit arguments, like
    :func:`build_prism_smeta`).

    Parameters
    ----------
    probe : Tensor
        Complex real-space probe of shape ``(Nmodes, MY, MX)`` on the
        ``N = f*pixels`` grid.  The incoherent power spectrum
        ``P(k) = sum_m |FFT(probe[m])|^2`` (sum over the mode axis) drives the
        aperture sizing.  A single-mode ``(1, MY, MX)`` probe is fine.
    pixels : (int, int)
        Detector window ``(P_y, P_x)``; the sim grid is ``N = f*pixels``.
    f : int
        PRISM interpolation factor (``>= 1``).
    dx : array-like (2,)
        Real-space pitch ``Delta p`` per axis (Aa), UNCHANGED by ``f``.
    wavelength : float
        Electron wavelength (Aa).
    semiconvergence_angle : float
        Probe semiconvergence angle (rad). ``kmax = semiconvergence_angle /
        wavelength`` sets the ``kmax*1.2`` floor.
    fov_shape : (int, int)
        Object FOV ``(NY_fov, NX_fov)``.  Raises if ``f*pixels < FOV``.
    device : torch.device or str
        Device the SMeta tensors are built on.
    power_fraction : float
        Target cumulative incoherent-power fraction the aperture must enclose
        (default ``0.9999`` -- 99.99%).
    n_radial : int or None
        Partitioned-PRISM knob, forwarded to
        :meth:`SMeta.make_beamlet_meta <SMeta.make_beamlet_meta>` (``None`` ->
        pure PRISM).
    n_angular : int
        Beamlet hex-ring angular sample count (default 6).  Ignored when
        ``n_radial is None``.

    Returns
    -------
    SMeta
        The f-lattice BF-disk SMeta at the power-chosen radius (pure, or
        partitioned when ``n_radial`` is set), enclosing ``>= power_fraction`` of
        the incoherent probe power and never tighter than ``kmax*1.2``.

    Raises
    ------
    ValueError
        If ``f*pixels < FOV`` (object does not fit), or the probe grid does not
        match ``N = f*pixels``.
    """
    f = int(f)
    P: tuple[int, int] = (int(pixels[0]), int(pixels[1]))
    dx = np.asarray(dx, dtype=np.float32)
    MY, MX = f * P[0], f * P[1]

    probe = probe.to(device)
    if probe.ndim != 3 or tuple(probe.shape[-2:]) != (MY, MX):
        raise ValueError(
            f"probe shape {tuple(probe.shape)} must be (Nmodes, {MY}, {MX}) "
            f"matching the N=f*pixels grid (f={f}, pixels={P})."
        )

    # Incoherent multi-mode power spectrum on the N grid (corner origin, like qf).
    P_k = (torch.abs(torch.fft.fft2(probe, dim=(-2, -1))) ** 2).sum(dim=0)
    total = P_k.sum()

    # f-lattice shell radii: the aperture only has beams at every f-th N-grid
    # frequency pixel, so candidate radii are multiples of f*dk (dk = 1/(N*dx)).
    qf = fft_frequency_grid((MY, MX), [float(dx[0]), float(dx[1])], device=device)
    qmag = torch.linalg.norm(qf, dim=0)
    dk = 1.0 / (MY * float(dx[0]))
    kmax = semiconvergence_angle / wavelength
    floor_radius = kmax * 1.2

    # Cumulative power over f-lattice shells; pick the smallest shell radius whose
    # cumulative >= power_fraction*total. Shell radii go up to the Nyquist corner.
    qmax = float(qmag.max())
    n_shells = int(np.ceil(qmax / (f * dk))) + 1
    target = float(power_fraction) * float(total)
    chosen_radius = qmax  # fallback: whole grid (never reached given Nyquist span)
    for s in range(n_shells + 1):
        r = s * f * dk
        captured = float(P_k[qmag <= r].sum())
        if captured >= target:
            chosen_radius = r
            break

    # Floor at kmax*1.2 (never tighter than 1.2x the convergence angle). Snap the
    # floor up to the enclosing f-lattice shell so the radius stays shell-aligned.
    if chosen_radius < floor_radius:
        s_floor = int(np.ceil(floor_radius / (f * dk)))
        chosen_radius = s_floor * f * dk

    return _build_prism_smeta_at_radius(
        radius=chosen_radius,
        pixels=P,
        f=f,
        dx=dx,
        fov_shape=fov_shape,
        device=device,
        n_radial=n_radial,
        n_angular=n_angular,
    )
