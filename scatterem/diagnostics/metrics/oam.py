"""Virtual orbital-angular-momentum (OAM) sorter for mixed-state wavefields.

The computational analogue of a hardware log-polar OAM sorter (Grillo et al.,
Nat. Commun. 8, 15536, 2017). Each mode ``psi_m`` of a mixed state is sampled
on a polar grid about a sorting axis ``(cy, cx)`` and decomposed into
azimuthal harmonics,

.. math::

    c_l^m(r) = \\frac{1}{2\\pi} \\int_0^{2\\pi} \\psi_m(r, \\theta)
               e^{-i l \\theta} d\\theta,

from which the OAM power spectrum :math:`P_l = \\sum_m \\sum_r r |c_l^m(r)|^2`
and the radially partial-traced OAM density matrix
:math:`\\rho_{ab} = \\sum_m \\sum_r r\\, c_a^m(r) \\overline{c_b^m(r)}` follow.
Unlike an intensity-based sorter, the virtual one sees coefficient phases, so
it separates an incoherent OAM mixture (``rho`` diagonal) from a coherent
superposition (``rho`` off-diagonal); see :func:`adjacent_coherence`.

Conventions shared by every function here:

* ``fields`` is a mode stack of shape ``(..., H, W)``; all leading dimensions
  are modes of **one** state and are summed over. A plain ``(H, W)`` array is
  a single coherent mode. Modes carry their populations in their norms (the
  usual ptychographic mixed-state convention), so no separate weights exist.
* Coordinates are array pixels, ``y`` = row index (increasing downwards),
  ``x`` = column index, and ``theta = atan2(y - cy, x - cx)``. With
  ``l_sign=+1`` a field ``exp(+i l theta)`` reports charge ``+l``; pass
  ``l_sign=-1`` to report in the opposite (e.g. a device) handedness.
* The sorter works in whatever plane it is given. For a probe the natural
  sorting plane is the aperture, :func:`to_aperture_plane`; the charge is the
  same in both planes (the Fourier transform keeps the azimuthal index).
* ``l`` is resolved to ``|l| < n_theta / 2``; spectra and density matrices are
  restricted to ``-l_max..l_max`` and renormalised over that window.

Inputs may be NumPy arrays or torch tensors (any device); outputs follow the
input. Nothing here is differentiable by design (the ring sampling is a
measurement, not a loss).
"""

import math

import torch

from scatterem.utils._array_compat import ArrayLike, as_complex, as_tensor, restore
from scatterem.utils.interpolation import sample_image

__all__ = [
    "adjacent_coherence",
    "mean_oam",
    "oam_density_matrix",
    "oam_spectrum",
    "to_aperture_plane",
]


def to_aperture_plane(fields: ArrayLike) -> ArrayLike:
    """Transform real-space-centred modes to the centred aperture plane.

    Computes ``fftshift(fft2(ifftshift(fields), norm="ortho"))`` over the last
    two axes, so the real-space centre ``(H // 2, W // 2)`` maps to zero
    frequency, zero frequency lands at ``(H // 2, W // 2)``, and the power of
    every mode is preserved.

    Parameters
    ----------
    fields : numpy.ndarray or torch.Tensor
        Complex mode stack, shape ``(..., H, W)``, real space, centred on
        ``(H // 2, W // 2)``.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        Aperture-plane modes, same shape and array family as ``fields``; one
        pixel is ``1 / (W * dx)`` in reciprocal space.
    """
    x, to_numpy = as_tensor(fields)
    dims = (-2, -1)
    out = torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(x, dim=dims), norm="ortho"), dim=dims
    )
    return restore(out, to_numpy)


def _intensity_centroid(x: torch.Tensor) -> tuple[float, float]:
    inten = (x.abs() ** 2).reshape(-1, *x.shape[-2:]).sum(0)
    h, w = inten.shape
    tot = inten.sum()
    ys = torch.arange(h, dtype=inten.dtype, device=inten.device)
    xs = torch.arange(w, dtype=inten.dtype, device=inten.device)
    cy = (ys[:, None] * inten).sum() / tot
    cx = (xs[None, :] * inten).sum() / tot
    return float(cy), float(cx)


def _harmonics(
    fields: ArrayLike,
    center: tuple[float, float] | None,
    l_max: int | None,
    r_min: float,
    r_max: float | None,
    n_r: int | None,
    n_theta: int,
    l_sign: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Return ``(ls, c, radii, to_numpy)`` with ``c`` of shape ``(M, n_r, L)``."""
    x, to_numpy = as_tensor(fields)
    x = as_complex(x)
    if to_numpy:
        x = x.to(torch.complex128)
    x = x.reshape(-1, *x.shape[-2:])
    l_max = n_theta // 2 - 1 if l_max is None else int(l_max)
    if not 0 <= l_max < n_theta / 2:
        raise ValueError(f"l_max={l_max} must satisfy 0 <= l_max < n_theta/2")
    if l_sign not in (1, -1):
        raise ValueError(f"l_sign must be +1 or -1, got {l_sign}")
    cy, cx = _intensity_centroid(x) if center is None else center
    h, w = x.shape[-2:]
    r_max = min(h, w) / 2 - 1 if r_max is None else float(r_max)
    n_r = max(2, math.ceil(r_max - r_min) + 1) if n_r is None else int(n_r)
    real = x.real.dtype
    radii = torch.linspace(r_min, r_max, n_r, dtype=real, device=x.device)
    thetas = torch.arange(n_theta, dtype=real, device=x.device) * (
        2 * math.pi / n_theta
    )
    ys = cy + radii[:, None] * torch.sin(thetas)[None, :]
    xs = cx + radii[:, None] * torch.cos(thetas)[None, :]
    rings = sample_image(x, torch.stack([ys, xs], dim=-1))  # (M, n_r, n_theta)
    c = torch.fft.fft(rings, dim=-1) / n_theta  # index = harmonic (mod n_theta)
    ls = torch.arange(-l_max, l_max + 1, device=x.device)
    c = c[..., (l_sign * ls) % n_theta]
    return ls, c, radii, to_numpy


def oam_spectrum(
    fields: ArrayLike,
    *,
    center: tuple[float, float] | None = None,
    l_max: int | None = None,
    r_min: float = 0.5,
    r_max: float | None = None,
    n_r: int | None = None,
    n_theta: int = 256,
    l_sign: int = 1,
) -> tuple[ArrayLike, ArrayLike]:
    """Compute the OAM power spectrum of a mixed state about a sorting axis.

    ``P_l = sum_m sum_r r |c_l^m(r)|^2`` (the ``r`` is the polar area element),
    normalised to ``sum_l P_l = 1`` over ``-l_max..l_max``. Bilinear ring
    sampling; samples outside the array count as zero.

    Parameters
    ----------
    fields : numpy.ndarray or torch.Tensor
        Complex mode stack, shape ``(..., H, W)``; leading dims are modes of
        one state, a ``(H, W)`` array is one mode. Real input is promoted.
    center : tuple of float, optional
        Sorting axis ``(cy, cx)`` in pixels. Default: the centroid of the
        total intensity ``sum_m |psi_m|^2``.
    l_max : int, optional
        Largest ``|l|`` reported; must be ``< n_theta / 2``. Default
        ``n_theta // 2 - 1`` (all resolvable charges).
    r_min, r_max : float, optional
        Radial sampling range in pixels. ``r_max`` defaults to
        ``min(H, W) / 2 - 1``.
    n_r : int, optional
        Number of radii, uniformly spaced. Default about one per pixel.
    n_theta : int, optional
        Azimuthal samples per ring. Default 256.
    l_sign : int, optional
        ``+1`` (default) reports ``exp(+i l theta)`` as ``+l``; ``-1`` flips
        the handedness.

    Returns
    -------
    ls : numpy.ndarray or torch.Tensor
        Integer charges ``-l_max..l_max``, shape ``(2 l_max + 1,)``.
    spectrum : numpy.ndarray or torch.Tensor
        Real populations, same shape, summing to 1.

    Raises
    ------
    ValueError
        If ``l_max`` is out of range or ``l_sign`` is not ``+-1``.
    """
    ls, c, radii, to_numpy = _harmonics(
        fields, center, l_max, r_min, r_max, n_r, n_theta, l_sign
    )
    spec = (radii[:, None] * c.abs() ** 2).sum((0, 1))
    return restore(ls, to_numpy), restore(spec / spec.sum(), to_numpy)


def oam_density_matrix(
    fields: ArrayLike,
    *,
    center: tuple[float, float] | None = None,
    l_max: int | None = None,
    r_min: float = 0.5,
    r_max: float | None = None,
    n_r: int | None = None,
    n_theta: int = 256,
    l_sign: int = 1,
) -> tuple[ArrayLike, ArrayLike]:
    """Project a mixed state onto the OAM basis, tracing out the radius.

    ``rho[a, b] = sum_m sum_r r c_a^m(r) conj(c_b^m(r))`` restricted to
    ``-l_max..l_max`` and normalised to ``Tr rho = 1``. It is Hermitian and
    positive semi-definite; its diagonal equals :func:`oam_spectrum` with the
    same arguments. Off-diagonal ``rho[a, a+1]`` is the coherence between
    adjacent OAM channels: zero for an incoherent mixture of OAM eigenstates.

    Parameters
    ----------
    fields : numpy.ndarray or torch.Tensor
        Complex mode stack, shape ``(..., H, W)``; see :func:`oam_spectrum`.
    center : tuple of float, optional
        Sorting axis ``(cy, cx)`` in pixels; default the intensity centroid.
    l_max : int, optional
        Largest ``|l|`` kept; default ``n_theta // 2 - 1``.
    r_min, r_max : float, optional
        Radial sampling range in pixels; ``r_max`` default ``min(H, W)/2 - 1``.
    n_r : int, optional
        Number of radii; default about one per pixel.
    n_theta : int, optional
        Azimuthal samples per ring. Default 256.
    l_sign : int, optional
        Handedness of the reported charge, ``+1`` (default) or ``-1``.

    Returns
    -------
    ls : numpy.ndarray or torch.Tensor
        Integer charges, shape ``(L,)`` with ``L = 2 l_max + 1``.
    rho : numpy.ndarray or torch.Tensor
        Complex density matrix, shape ``(L, L)``, rows/columns ordered as ``ls``.

    Raises
    ------
    ValueError
        If ``l_max`` is out of range or ``l_sign`` is not ``+-1``.
    """
    ls, c, radii, to_numpy = _harmonics(
        fields, center, l_max, r_min, r_max, n_r, n_theta, l_sign
    )
    weighted = c * radii[:, None].to(c.dtype)
    rho = torch.einsum("mra,mrb->ab", weighted, c.conj())
    rho = rho / torch.diagonal(rho).real.sum()
    return restore(ls, to_numpy), restore(rho, to_numpy)


def adjacent_coherence(rho: ArrayLike) -> tuple[float, int, int]:
    """Degree of coherence between the dominant OAM channel and a neighbour.

    ``g = |rho[i0, j]| / sqrt(rho[i0, i0] rho[j, j])`` where ``i0`` is the most
    populated channel and ``j`` the more populated of ``i0 - 1`` and
    ``i0 + 1``. ``g = 0`` for an incoherent OAM mixture, ``g = 1`` for a
    coherent superposition with a common radial profile.

    Parameters
    ----------
    rho : numpy.ndarray or torch.Tensor
        OAM density matrix, shape ``(L, L)``, e.g. from
        :func:`oam_density_matrix`.

    Returns
    -------
    g : float
        Degree of coherence in ``[0, 1]``.
    i0 : int
        Row index of the dominant channel.
    j : int
        Row index of the neighbour used.

    Raises
    ------
    ValueError
        If ``rho`` has fewer than two channels.
    """
    r, _ = as_tensor(rho)
    if r.shape[-1] < 2:
        raise ValueError("adjacent_coherence needs at least two OAM channels")
    pops = torch.diagonal(r).real.clamp_min(0)
    i0 = int(torch.argmax(pops))
    neighbours = [i for i in (i0 - 1, i0 + 1) if 0 <= i < pops.shape[0]]
    j = max(neighbours, key=lambda i: float(pops[i]))
    g = r[i0, j].abs() / torch.sqrt(pops[i0] * pops[j])
    return float(g), i0, j


def mean_oam(ls: ArrayLike, spectrum: ArrayLike) -> float:
    """Return the mean OAM ``sum_l l P_l / sum_l P_l`` in units of hbar.

    Parameters
    ----------
    ls : numpy.ndarray or torch.Tensor
        Integer charges, shape ``(L,)``.
    spectrum : numpy.ndarray or torch.Tensor
        Populations, shape ``(L,)``, e.g. from :func:`oam_spectrum`.

    Returns
    -------
    float
        Population-weighted mean charge.
    """
    lt, _ = as_tensor(ls)
    pt, _ = as_tensor(spectrum)
    pt = pt.to(torch.float64)
    return float((lt.to(pt) * pt).sum() / pt.sum())
