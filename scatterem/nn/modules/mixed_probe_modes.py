"""Mixed-state (incoherent multi-mode) probe builders as free functions.

The functions build the Hermite-like mode basis and the power ladder of a
mixed-state probe from one coherent probe or a measured aperture.

Everything below works on NumPy arrays in the ``(..., Ny, Nx)`` layout and
returns complex arrays of the input dtype, except
:func:`probe_stack_from_aperture`, which returns the ``(pmodes, 1, N, N)``
``complex64`` torch stack a ptychography probe model takes.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


def _unit(field: np.ndarray) -> np.ndarray:
    """Scale ``field`` to unit L2 norm over all its pixels."""
    return field / np.sqrt(np.sum(np.abs(field) ** 2))


def hermite_like(fundam: np.ndarray, M: float, N: float) -> np.ndarray:
    """Orthonormal Hermite-like probe modes built on a fundamental probe.

    This is an independent implementation of the mode seeding idea of
    Thibault & Menzel, Nature 494, 68 (2013): the secondary modes of a
    mixed-state probe start as low-order polynomials times the fundamental.

    Let ``u = x - cx`` and ``v = y - cy`` be pixel offsets from the intensity
    centroid ``(cx, cy)`` of ``|fundam|^2``, with pixel coordinates
    ``arange(n) - n / 2`` along each axis. The raw mode of order ``(m, n)`` is
    ``u^m v^n fundam``. Mode 0 is the normalised fundamental. Each later mode is
    damped by a Gaussian whose widths are the intensity variances of
    ``fundam``, normalised, and then made orthogonal to all earlier modes by
    modified Gram-Schmidt: each overlap is taken from the already-updated mode.
    Every mode, mode 0 included, is then normalised once more.

    Parameters
    ----------
    fundam : numpy.ndarray
        Complex fundamental probe, ``(Ny, Nx)`` or ``(..., Ny, Nx)`` with all
        leading axes of size one.
    M : int or float
        Highest power of ``u`` (x direction); cast to int.
    N : int or float
        Highest power of ``v`` (y direction); cast to int.

    Returns
    -------
    numpy.ndarray
        ``((M + 1) * (N + 1), Ny, Nx)`` array of unit-norm, mutually orthogonal
        modes with the dtype of ``fundam``. Mode ``k`` has the y power as its
        slow index and the x power as its fast index. A negative ``M`` or ``N``
        gives an empty ``(0, Ny, Nx)`` array. All modes are computed in
        complex128 and rounded to the dtype of ``fundam`` once, at the end.

    Raises
    ------
    ValueError
        If ``fundam`` has a leading axis whose size is not one.
    """
    n_x_pow, n_y_pow = int(M), int(N)
    ny, nx = fundam.shape[-2:]
    if fundam.size != ny * nx:
        raise ValueError(
            f"fundam must be (Ny, Nx) or have leading axes of size one, "
            f"got shape {fundam.shape}"
        )
    fundam = fundam.reshape(ny, nx)
    if n_x_pow < 0 or n_y_pow < 0:
        return np.zeros((0, ny, nx), dtype=fundam.dtype)
    weight = np.abs(fundam) ** 2
    total = np.sum(weight)
    xs = np.arange(nx) - nx / 2
    ys = (np.arange(ny) - ny / 2)[:, None]
    mean_x = np.sum(xs * weight) / total
    mean_y = np.sum(ys * weight) / total
    dx = xs - mean_x
    dy = ys - mean_y
    var_x = np.sum(dx**2 * weight) / total
    var_y = np.sum(dy**2 * weight) / total
    window = np.exp(-(dx**2) / (2 * var_x) - dy**2 / (2 * var_y))

    basis: list[np.ndarray] = []
    for y_pow in range(n_y_pow + 1):
        for x_pow in range(n_x_pow + 1):
            mode = dx**x_pow * dy**y_pow * fundam
            if basis:
                mode = _unit(mode * window)
                for earlier in basis:
                    mode = mode - np.sum(earlier * np.conj(mode)) * earlier
            else:
                mode = _unit(mode)
            basis.append(_unit(mode))
    return np.stack(basis).astype(fundam.dtype, copy=False)


def mixed_mode_powers(pmodes: int, pmode_init_pows: Sequence[float]) -> np.ndarray:
    """Power fraction of every mode of a mixed-state probe.

    Parameters
    ----------
    pmodes : int
        Number of incoherent probe modes.
    pmode_init_pows : sequence of float
        Powers of the secondary modes: entry ``i`` belongs to mode ``i + 1``.
        When there are more secondary modes than entries, the extra modes get
        the last entry. Not read when ``pmodes == 1``.

    Returns
    -------
    numpy.ndarray
        ``(pmodes,)`` float64 powers with sum one. Mode 0 gets what the
        secondary modes leave, ``1 - sum(secondaries)``.

    Raises
    ------
    ValueError
        If the secondary powers add up to more than one.
    """
    powers = np.ones(pmodes, dtype=np.float64)
    if pmodes == 1:
        return powers
    pows = list(pmode_init_pows)
    secondary = np.array(
        [pows[min(k, len(pows) - 1)] for k in range(pmodes - 1)], dtype=np.float64
    )
    used = np.sum(secondary)
    if used > 1:
        raise ValueError("Modes total power exceeds 1, check pmode_init_pows")
    powers[0] = 1 - used
    powers[1:] = secondary
    return powers


def make_mixed_probe(
    probe: np.ndarray, pmodes: int, pmode_init_pows: Sequence[float]
) -> np.ndarray:
    """Expand one coherent probe into a stack of incoherent probe modes.

    This is an independent implementation of the mixed-state probe seeding of
    Thibault & Menzel, Nature 494, 68 (2013). It takes the first ``pmodes``
    unit-norm :func:`hermite_like` modes of ``probe`` on the smallest
    ``(M + 1) x (N + 1)`` order grid that holds them, and scales mode ``k`` by
    the square root of its power from :func:`mixed_mode_powers`. The power of
    the stack is therefore one, independent of the power of ``probe``.

    Parameters
    ----------
    probe : numpy.ndarray
        Complex ``(Ny, Nx)`` coherent probe.
    pmodes : int
        Number of incoherent probe modes in the stack.
    pmode_init_pows : sequence of float
        Powers of the secondary modes; see :func:`mixed_mode_powers`.

    Returns
    -------
    numpy.ndarray
        Complex ``(pmodes, Ny, Nx)`` mixed-state probe with the dtype of
        ``probe``: a complex64 probe gives a complex64 stack. The power
        scaling is done in float64 and rounded once.

    Raises
    ------
    ValueError
        If the secondary powers add up to more than one.
    """
    logger.info(
        "Start making mixed-state STEM probe with %d incoherent probe modes", pmodes
    )
    powers = mixed_mode_powers(pmodes, pmode_init_pows)
    grid_x = np.ceil(np.sqrt(pmodes)) - 1
    grid_y = np.ceil(pmodes / (grid_x + 1)) - 1
    modes = hermite_like(probe, grid_x, grid_y)[:pmodes]
    mixed = modes * np.sqrt(powers)[:, None, None]
    logger.info("Relative power of probe modes = %s", powers)
    return mixed.astype(modes.dtype, copy=False)


def recenter_aperture(aperture: np.ndarray) -> np.ndarray:
    """Roll a detector-space aperture so its intensity centroid is centred.

    The centroid of ``|aperture|^2`` (pixel indices from 0) is moved to
    ``(n0 / 2, n1 / 2)`` by a circular roll of whole pixels, the shift
    rounded half-to-even.

    Parameters
    ----------
    aperture : numpy.ndarray
        ``(n0, n1)`` real or complex aperture with nonzero total intensity.

    Returns
    -------
    numpy.ndarray
        The rolled aperture, same shape and dtype.
    """
    n0, n1 = aperture.shape
    intensity = np.abs(aperture) ** 2
    total = intensity.sum()
    yy, xx = np.mgrid[0:n0, 0:n1]
    cy = (yy * intensity).sum() / total
    cx = (xx * intensity).sum() / total
    shift = (int(round(n0 / 2 - cy)), int(round(n1 / 2 - cx)))
    return np.roll(aperture, shift, axis=(0, 1))


def _validate_aperture_args(
    aperture: np.ndarray, ndet: int, pmodes: int, mode0_frac: float
) -> None:
    if aperture.ndim != 2 or aperture.shape[0] != aperture.shape[1]:
        raise ValueError(f"aperture must be square 2-D, got shape {aperture.shape}")
    n = aperture.shape[0]
    if ndet < n:
        raise ValueError(f"ndet must be at least the aperture size {n}, got {ndet}")
    if (ndet - n) % 2:
        raise ValueError(
            f"ndet - aperture size must be even for a symmetric pad, got {ndet} - {n}"
        )
    if pmodes < 1:
        raise ValueError(f"pmodes must be >= 1, got {pmodes}")
    if not 0.0 <= mode0_frac <= 1.0:
        raise ValueError(f"mode0_frac must lie in [0, 1], got {mode0_frac}")
    if not np.any(aperture):
        raise ValueError("aperture has zero total intensity")


def probe_stack_from_aperture(
    aperture: np.ndarray | Tensor,
    ndet: int,
    pmodes: int,
    mode0_frac: float = 0.96,
    *,
    device: torch.device | str | None = None,
    keep_position: bool = False,
) -> Tensor:
    """Real-space mixed-state probe stack seeded from a detector-space aperture.

    Steps, all in complex128:

    1. :func:`recenter_aperture` rolls the aperture's intensity centroid to the
       centre pixel ``(n / 2, n / 2)`` of its own grid. ``keep_position=True``
       skips this step: the aperture stays where it is on its grid.
    2. It is zero-padded symmetrically to ``(ndet, ndet)`` (Fourier
       interpolation of the real-space probe: finer real-space sampling at an
       unchanged detector pixel), keeping it on the centre pixel ``ndet / 2``.
    3. The centred inverse FFT ``fftshift(ifft2(fftshift(A)))``
       (``torch.fft`` default ``norm="backward"``) gives the real-space
       primary probe centred on pixel ``(ndet / 2, ndet / 2)``. For even
       sizes ``fftshift`` and ``ifftshift`` coincide, so this is also
       ``fftshift(ifft2(ifftshift(A)))``. The absolute scale of the FFT drops
       out: every mode is renormalized.
    4. For ``pmodes == 1`` the result is that probe with unit power. Otherwise
       it is :func:`make_mixed_probe` with the power ladder ``mode0_frac`` for
       mode 0 and ``(1 - mode0_frac) / (pmodes - 1)`` for each Hermite-like
       secondary mode.

    The stack therefore has total power ``sum |P|^2 = 1``; rescale it to the
    measured intensity if the model needs that.

    Parameters
    ----------
    aperture : numpy.ndarray or Tensor
        ``(n, n)`` real or complex detector-space aperture (the complex pupil
        function, e.g. a recovered aperture), DC near the centre, any offset
        removed by step 1. A tensor is detached and copied to the CPU.
    ndet : int
        Padded detector size; ``ndet - n`` must be even and non-negative.
    pmodes : int
        Number of incoherent probe modes, ``>= 1``.
    mode0_frac : float, optional
        Fraction of the power in mode 0, in ``[0, 1]``; ignored when
        ``pmodes == 1``. By default 0.96.
    device : torch.device or str, optional
        Device of the returned tensor; CPU when omitted.
    keep_position : bool, optional
        Keep the aperture at its position on the grid (no step 1). Use it when
        the aperture is centred on the data disk but its intensity centroid is
        not, for example an aperture with a needle shadow: the roll of step 1
        then moves the seed off the data disk. The modes differ from the
        default ones only by the real-space phase ramp of that roll (the
        Hermite-like seeding commutes with it). By default False.

    Returns
    -------
    Tensor
        ``(pmodes, 1, ndet, ndet)`` complex64 real-space probe stack.

    Raises
    ------
    ValueError
        If the aperture is not square 2-D or has zero intensity, ``ndet`` is
        smaller than the aperture or differs from it by an odd count,
        ``pmodes < 1`` or ``mode0_frac`` lies outside ``[0, 1]``.
    """
    if isinstance(aperture, Tensor):
        aperture = aperture.detach().cpu().numpy()
    aperture = np.asarray(aperture)
    _validate_aperture_args(aperture, ndet, pmodes, mode0_frac)

    ap = aperture.astype(np.complex128)
    if not keep_position:
        ap = recenter_aperture(ap)
    pad = (ndet - ap.shape[0]) // 2
    ap_t = torch.as_tensor(np.pad(ap, ((pad, pad), (pad, pad))), dtype=torch.complex128)
    probe0 = torch.fft.fftshift(torch.fft.ifft2(torch.fft.fftshift(ap_t))).numpy()
    if pmodes == 1:
        stack = probe0[None] / np.sqrt((np.abs(probe0) ** 2).sum())
    else:
        secondary = [(1.0 - mode0_frac) / (pmodes - 1)] * (pmodes - 1)
        stack = make_mixed_probe(probe0, pmodes, secondary)
    return torch.as_tensor(stack[:, None], dtype=torch.complex64, device=device)


class MixedProbeModesMixin:
    """The probe-mode accessors that every mixed-state probe module shares.

    A host :class:`torch.nn.Module` defines ``forward_single()`` (one coherent
    ``(MY, MX)`` mode), ``mixed_probe`` (bool), ``mixed_probe_modes`` (the
    ``(pmodes, MY, MX)`` stack, set when ``mixed_probe`` is True) and
    ``device``. Put the mixin first in the bases so that its ``forward``
    replaces :class:`~torch.nn.Module`'s.
    """

    mixed_probe: bool
    mixed_probe_modes: Tensor
    device: torch.device

    def forward_single(self) -> Tensor:
        """Return the single coherent probe mode; each host module defines it."""
        raise NotImplementedError

    def forward(self) -> Tensor:
        """Return the probe modes, mixed if enabled, else a single mode.

        Returns
        -------
        Tensor
            Complex probe modes shaped ``(pmodes, 1, MY, MX)`` if
            ``self.mixed_probe`` is True, otherwise ``(1, 1, MY, MX)``.
        """
        if self.mixed_probe:
            return self.mixed_probe_modes[:, None, :, :]
        return self.forward_single()[None, None, :, :]

    def get_probe_modes(self) -> Tensor:
        """Get the current probe modes (mixed or single).

        Returns
        -------
        Tensor
            Same as :meth:`forward`.
        """
        return self.forward()

    def get_mode_powers(self) -> Tensor:
        """Get the normalized power of each probe mode.

        Returns
        -------
        Tensor
            Power fractions of shape ``(pmodes,)``, summing to 1, or ``[1.0]``
            when ``self.mixed_probe`` is False.
        """
        if self.mixed_probe:
            powers = torch.sum(torch.abs(self.mixed_probe_modes) ** 2, dim=(-2, -1))
            return powers / torch.sum(powers)
        return torch.tensor([1.0], device=self.device)
