"""Reciprocal-space coordinate grids and Fourier-space band limiting.

Written from the published formulation of discrete Fourier sampling and
multislice anti-aliasing.  The two pieces of infrastructure it provides are

``q_space_array``
    the discrete reciprocal-space (spatial-frequency) grid of a periodic
    simulation cell, on which elastic scattering factors, probe-forming
    apertures, Fresnel propagators and detector masks are evaluated;

``bandwidth_limit_array_torch``
    the anti-aliasing band-width limit that keeps the multislice iteration
    stable (Kirkland's "2/3 rule").

Conventions used throughout (and relied on by every consumer in this package)
---------------------------------------------------------------------------
*Spatial frequency, not angular wavenumber.*  A plane wave is
``exp(2*pi*i*q.r)``, so ``q`` is in cycles per Angstrom and there is **no**
factor of ``2*pi`` anywhere.  Downstream formulae are written to match:
aperture cut at ``alpha/(1000*lambda)``, aberration ``chi = pi*lambda*df*q^2``,
propagator ``exp(-i*pi*lambda*dz*q^2)``, detector angle
``theta_mrad = |q|*lambda*1000``.

*Corner origin.*  Zero frequency sits at index ``[0, 0]``; the per-axis order is
the unshifted DFT order ``[0, 1, ..., ceil(M/2)-1, -floor(M/2), ..., -1]``.  No
``fftshift`` is applied here or expected downstream.

*Unnormalised forward transform.*  ``X[k] = sum_n x[n] exp(-2*pi*i*k*n/M)``,
with the whole ``1/M`` carried by the inverse (``norm='backward'`` in both
NumPy and PyTorch).  A two-dimensional inverse over the trailing axis pair
therefore carries ``1/(M_-2 * M_-1)``.

*Matrix ('ij') mesh indexing.*  Coordinate ``i`` varies along axis ``i``, so
``qy, qx = q_space_array(pixels, gridsize)`` has ``qy`` varying down rows.

References
----------
E. J. Kirkland, *Advanced Computing in Electron Microscopy*, Springer.
    Ch. 5-6 for multislice, Sec. 6.7 for the band-width limit and the 2/3 rule,
    App. C for scattering-factor parameterisations.
J. M. Cowley and A. F. Moodie, Acta Cryst. **10** (1957) 609.
    Original multislice formulation.
I. Lobato and D. Van Dyck, Acta Cryst. A **70** (2014) 636-649.
    Physically constrained parameterisation of the elastic scattering factors
    that are sampled on the grid built here.
S. A. Orszag, J. Atmos. Sci. **28** (1971) 1074.
    The 2/3 de-aliasing rule in its pseudo-spectral form.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from functools import reduce
from typing import Any, List, Optional, Sequence, Union

import numpy as np
import torch

__all__ = [
    "q_space_array",
    "q_space_squared",
    "bandwidth_limit_array_torch",
    "broadcast_from_unmeshed",
    "clear_caches",
]


# ---------------------------------------------------------------------------
# Small generic helpers
# ---------------------------------------------------------------------------


def _is_array_like(value: Any) -> bool:
    """True when ``value`` is a sequence rather than a scalar.

    Used only to decide whether the ``limit`` argument of
    :func:`bandwidth_limit_array_torch` carries one value per axis or a single
    value to be replicated across both axes.  Anything exposing ``ndim`` is
    judged by that (so a 0-d NumPy or torch scalar reads as a scalar, and a
    NumPy scalar type likewise); everything else is a sequence exactly when it
    has a length, so lists and tuples read as sequences and Python floats do
    not.
    """
    ndim = getattr(value, "ndim", None)
    if ndim is not None:
        return ndim > 0
    try:
        len(value)
    except TypeError:
        return False
    return True


def _broadcast_from_unmeshed(coords: Sequence[np.ndarray]) -> List[np.ndarray]:
    """'ij'-indexed mesh of ``N`` one-dimensional coordinate vectors, zero-copy.

    Equivalent in value to ``numpy.meshgrid(*coords, indexing='ij')``, but the
    returned arrays are read-only broadcast *views*: coordinate ``i`` is
    reshaped to extent ``N_i`` on axis ``i`` and extent 1 on every other axis,
    then broadcast to ``(N_0, ..., N_{N-1})``.  Each output therefore has stride
    zero on every axis but its own and costs no memory beyond the input
    vectors -- a three-dimensional mesh of ``(4, 3, 2)`` allocates 9 floats, not
    3 * 24.  Callers that need contiguous buffers (e.g. before
    ``torch.tensor``) call ``numpy.ascontiguousarray`` themselves.

    Parameters
    ----------
    coords : sequence of N one-dimensional ndarray
        Shapes ``(N_0,), ..., (N_{N-1},)``; dtype arbitrary and preserved.

    Returns
    -------
    list of N ndarray
        Each of shape ``(N_0, ..., N_{N-1})``, read-only.
    """
    ndim = len(coords)
    shape = tuple(int(c.shape[0]) for c in coords)
    out = []
    for i, c in enumerate(coords):
        # Reshape spec for axis i: 1 everywhere, N_i in slot i.  Built from
        # plain Python ints, so there is no integer-width question at all.
        spec = [1] * ndim
        spec[i] = shape[i]
        out.append(np.broadcast_to(c.reshape(spec), shape))
    return out


#: Public spelling of the mesh helper (the module-private name is used
#: internally; both refer to the same function).
broadcast_from_unmeshed = _broadcast_from_unmeshed


# ---------------------------------------------------------------------------
# A. Reciprocal-space coordinate grid
# ---------------------------------------------------------------------------

_AXIS_CACHE: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
_AXIS_CACHE_MAX = 64
_AXIS_LOCK = threading.Lock()


def _q_axis(n: int, length: float) -> np.ndarray:
    """One reciprocal-space axis: ``q[k] = m(k) / length``, float64, read-only.

    The DFT of an ``n``-point periodic cell of edge ``length`` represents
    exactly the reciprocal-lattice frequencies ``m/length`` with integer ``m``
    in unshifted order,

        m(k) = k              for k < ceil(n/2)
        m(k) = k - n          otherwise,

    bounded by the Nyquist frequency ``n/(2*length)``.  For even ``n`` that
    extreme value occurs once, at index ``n/2``, and is stored **negative**;
    for odd ``n`` the set is symmetric and no exact Nyquist sample exists.

    This is ``numpy.fft.fftfreq(n, d=length/n)`` -- the pixel count in the
    sampling interval ``d = length/n`` cancels against the ``n`` inside
    ``fftfreq`` -- and is computed that way so the values agree with the DFT
    frequency convention bit for bit.

    The result is memoised on ``(n, length)`` and marked read-only, because
    callers build the same grid once per probe position / detector / slice.
    """
    n = int(n)
    key: Optional[tuple]
    try:
        key = (n, float(length))
    except (TypeError, ValueError):
        key = None

    if key is not None:
        with _AXIS_LOCK:
            hit = _AXIS_CACHE.get(key)
            if hit is not None:
                _AXIS_CACHE.move_to_end(key)
                return hit

    axis = np.fft.fftfreq(n, d=length / n)
    axis.flags.writeable = False

    if key is not None:
        with _AXIS_LOCK:
            if len(_AXIS_CACHE) >= _AXIS_CACHE_MAX:
                _AXIS_CACHE.popitem(last=False)
            _AXIS_CACHE[key] = axis
    return axis


def q_space_array(
    pixels: Sequence[int],
    gridsize: Sequence[float],
    meshed: bool = True,
) -> List[np.ndarray]:
    """Reciprocal-space coordinate arrays of an N-dimensional real-space grid.

    A cell of edge lengths ``L_i = gridsize[i]`` sampled on ``N_i = pixels[i]``
    points has real-space sampling ``dr_i = L_i / N_i`` and represents the
    spatial frequencies

        q_i[k] = m(k) / L_i,    m(k) = k if k < ceil(N_i/2) else k - N_i,

    i.e. ``|q_i| <= q_Nyq,i = 1/(2*dr_i) = N_i/(2*L_i)``, in unshifted
    (corner-origin) DFT order.  Units are inverse Angstrom, *cycles* per
    Angstrom -- there is no factor of ``2*pi``.

    The quantity consumers actually form is ``gsq = q_0**2 + q_1**2``, the
    squared scattering-vector magnitude that parameterised elastic scattering
    factors ``f_e(|g|)`` (Kirkland App. C; Lobato and Van Dyck, Acta Cryst. A70
    (2014) 636), the aperture and defocus phase ``chi = pi*lambda*df*|q|^2``,
    and the Fresnel propagator ``exp(-i*pi*lambda*dz*|q|^2)`` are evaluated on.
    :func:`q_space_squared` computes it in one pass if that is all you need.

    Parameters
    ----------
    pixels : array_like of N ints
        Sample counts per axis.  ``N = len(pixels)`` sets the dimensionality;
        works for any ``N >= 1``.
    gridsize : array_like of N floats
        Real-space cell edge lengths in Angstrom.  Only the first ``N`` entries
        are consulted; extra entries are ignored.
    meshed : bool, optional
        If True (default) return N arrays broadcast to the full grid shape,
        'ij' (matrix) indexed: axis 0 varies with ``q_0``, axis 1 with ``q_1``.
        If False return the N unbroadcast one-dimensional axes.

    Returns
    -------
    list of N ndarray, dtype float64
        ``meshed=True``: each of shape ``tuple(pixels)``, returned as read-only
        zero-stride broadcast views (see :func:`broadcast_from_unmeshed`), so
        ``numpy.asarray(...)`` on the result has shape ``(N, *pixels)``.
        ``meshed=False``: writable 1-d arrays of shapes ``(N_0,), ...``.

    Notes
    -----
    Pure function, NumPy only, no validation: a zero cell edge yields
    ``inf``/``nan`` with a divide-by-zero warning and a length mismatch raises
    ``IndexError``.  The dtype is float64 unconditionally, whatever the dtype
    of ``gridsize``.
    """
    ndim = len(pixels)
    axes = [_q_axis(pixels[i], gridsize[i]) for i in range(ndim)]
    if meshed:
        return _broadcast_from_unmeshed(axes)
    # Hand out writable copies so a caller cannot poison the memoised axes;
    # these are O(N_i), not O(prod(pixels)).
    return [np.array(a) for a in axes]


def q_space_squared(
    pixels: Sequence[int],
    gridsize: Sequence[float],
) -> np.ndarray:
    """Squared scattering-vector magnitude ``|g|^2 = sum_i q_i^2`` on the grid.

    Identical in value to ``sum(q**2 for q in q_space_array(pixels, gridsize))``
    but formed as an outer sum of the N one-dimensional squared axes: ``O(sum
    N_i)`` arithmetic and a single full-size allocation, instead of squaring N
    broadcast views (which re-reads each value ``prod(pixels)/N_i`` times) and
    adding them.

    Returns
    -------
    ndarray of shape ``tuple(pixels)``, dtype float64, in 1/Angstrom^2.
    """
    ndim = len(pixels)
    sq = [np.square(_q_axis(pixels[i], gridsize[i])) for i in range(ndim)]
    # reduce: a of shape (..., ) gains a trailing axis, b broadcasts along it.
    return reduce(lambda a, b: a[..., None] + b, sq)


# ---------------------------------------------------------------------------
# B. Anti-aliasing band-width limit
# ---------------------------------------------------------------------------

_MASK_CACHE: "OrderedDict[tuple, torch.Tensor]" = OrderedDict()
_MASK_CACHE_MAX = 32
_MASK_LOCK = threading.Lock()

#: ``erf(x)`` differs from 1 by less than 2.2e-17 for ``x >= 6``, i.e. by less
#: than half a float64 ulp, so the soft profile's argument may be clipped there
#: without changing a single returned value.  Clipping keeps ``erf`` away from
#: its saturated tail and bounds the argument range.
_ERF_SATURATION = 6.0


def _device_key(device: Union[str, torch.device]) -> str:
    """Canonical string for a device, so 'cuda' and 'cuda:0' share one entry."""
    dev = torch.device(device)
    if dev.type == "cuda" and dev.index is None:
        return "cuda:%d" % torch.cuda.current_device()
    return str(dev)


def _resolve_rfft_length(n: int, rfft: Union[bool, int]) -> Optional[int]:
    """Full signal length behind a one-sided spectrum of ``n`` frequencies.

    ``rfft=False`` returns ``None`` (the axis is two-sided).  ``rfft=True``
    infers the length as ``2*(n - 1)``, which is exact for an even-length
    signal; a half-spectrum alone cannot resolve the parity, and for odd ``Nx``
    this treats the highest column as Nyquist, scaling that axis by
    ``Nx/(Nx-1)``.  Passing the true length as an integer (``rfft=Nx``, any
    ``int`` that is not a ``bool``) removes the ambiguity.
    """
    if not rfft:
        return None
    if isinstance(rfft, bool):
        return 2 * (int(n) - 1)
    return int(rfft)


def _nyquist_normalised_axis(
    n: int,
    limit: float,
    device: torch.device,
    n_full: Optional[int],
) -> torch.Tensor:
    """``(u/limit)**2`` along one axis, with ``u = q/q_Nyq`` in ``[-1, 1)``.

    Frequencies are taken with **unit sample spacing**, so ``fftfreq(n)`` runs
    over ``[-1/2, 1/2)`` cycles per pixel and ``u = 2*fftfreq(n)`` is the
    frequency as a fraction of Nyquist.  The band limit is therefore a fraction
    of Nyquist and is independent of the physical cell size -- that is
    deliberate: it is an aliasing constraint on the sampling grid, not a
    physical cutoff.

    When ``n_full`` is given the axis is a one-sided real-FFT spectrum of ``n``
    non-negative frequencies belonging to a real signal of ``n_full`` samples;
    its coordinates are ``rfftfreq(n_full)``, the non-negative half
    ``[0, ..., 1/2]`` of the corresponding two-sided axis.

    Computed in float64 so that samples lying exactly on the band-limit contour
    (which happens whenever ``n*limit/2`` is an integer -- e.g. ``n=12``,
    ``limit=2/3``) land on exactly 1.0 and are resolved by the strict
    inequality rather than by float32 rounding noise.
    """
    if n_full is None:
        u = torch.fft.fftfreq(int(n), dtype=torch.float64, device=device)
    else:
        u = torch.fft.rfftfreq(int(n_full), dtype=torch.float64, device=device)
        u = u[: int(n)]
    u = u * (2.0 / float(limit))  # u/limit in Nyquist-normalised units
    return u * u


def _build_bandwidth_mask(
    shp2: Sequence[int],
    lmt: Sequence[float],
    soft: Optional[float],
    rfft: bool,
    dtype: torch.dtype,
    device: Union[str, torch.device],
) -> torch.Tensor:
    """Construct the two-dimensional band-limit mask (uncached).

    With per-axis Nyquist-normalised coordinates ``u_j = q_j/q_Nyq,j`` and
    per-axis limits ``l_j``, the elliptic radius is

        rho^2 = (u_-2/l_-2)^2 + (u_-1/l_-1)^2,

    so ``rho = 1`` is the band-limit contour: a circle at ``limit * Nyquist``
    for a scalar limit, an axis-aligned ellipse for a two-vector (a circle in
    pixel-frequency space, hence an ellipse in physical 1/Angstrom on an
    anisotropically sampled grid).  The mask is

        hard (soft is None) :  W = 1 if rho^2 < 1 else 0        [strict <]
        soft (soft = s > 0) :  W = erf(max(1 - rho, 0) / s).

    The strict inequality means a frequency landing exactly on the contour is
    rejected: at ``limit=1`` the Nyquist row and column of an even grid are
    zeroed, and at ``limit=2/3`` with ``M`` divisible by 3 the two samples per
    axis at ``|q| = 2/3 q_Nyq`` are zeroed.

    The soft profile exists to suppress the Gibbs ringing that a sharp spectral
    cut imprints on a real-space projected potential (a hard cut convolves it
    with a jinc-like kernel whose sidelobes decay only as 1/r).  It reaches
    exactly zero for all ``rho >= 1`` and is *not* renormalised: its DC gain is
    ``erf(1/s)``, which is 1 to within 1e-9 at ``s = 0.05`` but only 0.9953 at
    ``s = 0.5``.

    Parameters
    ----------
    shp2 : 2-sequence of int
        ``(M_-2, M_-1)``, the trailing two dimensions of the spectrum.
    lmt : 2-sequence of float
        Per-axis limits as fractions of that axis' Nyquist frequency.
    soft : None or float
        ``None`` selects the hard mask; a positive float the erf taper of that
        characteristic width in normalised radius.
    rfft : bool or int
        The trailing axis is a one-sided real-FFT spectrum; an ``int`` gives
        the full signal length explicitly (see :func:`_resolve_rfft_length`).
    dtype : torch real dtype
    device : torch device or device string

    Returns
    -------
    torch.Tensor of shape ``tuple(shp2)``, given ``dtype`` and ``device``.

    Notes
    -----
    Built directly on the target device and in float64, then cast once: no host
    computation, no host-to-device copy of a full-size array, and no dependence
    on the global default dtype.  Boundary pixels are therefore resolved by
    exact-arithmetic-quality float64 comparison rather than at float32
    granularity (~1e-7 in normalised radius).
    """
    dev = torch.device(device)
    ay = _nyquist_normalised_axis(shp2[0], lmt[0], dev, None)
    ax = _nyquist_normalised_axis(
        shp2[-1], lmt[-1], dev, _resolve_rfft_length(shp2[-1], rfft)
    )

    # rho^2 as a rank-1 outer sum of the two squared axes: only M_-2 + M_-1
    # values are genuinely independent.
    rho_sq = ay.unsqueeze(-1) + ax

    if soft is None:
        # One fused compare-and-cast; no float mask intermediate is kept.
        return (rho_sq < 1.0).to(dtype=dtype)

    s = float(soft)
    if s <= 0.0:
        raise ValueError(
            "soft must be a positive taper width (or None for a hard limit); "
            "got %r" % (soft,)
        )
    if not dtype.is_floating_point:
        raise TypeError(
            "a soft band-limit taper cannot be represented in %r; the erf "
            "profile would truncate to 0/1" % (dtype,)
        )
    # erf(max(1 - rho, 0)/s).  In-place on the private rho_sq temporary.
    rho_sq.sqrt_().neg_().add_(1.0).clamp_(min=0.0, max=_ERF_SATURATION * s)
    rho_sq.div_(s)
    return torch.erf(rho_sq).to(dtype=dtype)


def _bandwidth_mask(
    shp2: Sequence[int],
    lmt: Sequence[float],
    soft: Optional[float],
    rfft: bool,
    dtype: torch.dtype,
    device: Union[str, torch.device],
) -> torch.Tensor:
    """Memoising wrapper over :func:`_build_bandwidth_mask`.

    The mask is a pure function of ``(shape, limit, soft, rfft, dtype,
    device)`` and never sees data values, so reuse is exact rather than
    tolerance-bounded.  The cache is a lock-protected LRU capped at 32 entries;
    the returned tensor is **shared and must be treated as immutable** (the
    only use here is an out-of-place multiply).  Call :func:`clear_caches` to
    release the entries, which for large grids pin device memory.

    A key that cannot be formed (a limit that will not cast to ``float``, say)
    falls back to an uncached build rather than raising.
    """
    try:
        key = (
            int(shp2[0]),
            int(shp2[-1]),
            float(lmt[0]),
            float(lmt[-1]),
            None if soft is None else float(soft),
            # Key on the *resolved* full signal length, not on ``rfft`` itself:
            # ``int(True)`` and the integer ``1`` are indistinguishable, so
            # ``int(rfft)`` would hand an ``rfft=True`` mask to an ``rfft=1``
            # call (and vice versa).  ``None`` (two-sided) never collides with
            # a length, and two spellings that resolve to the same length are
            # genuinely the same mask and *should* share an entry.
            _resolve_rfft_length(shp2[-1], rfft),
            dtype,
            _device_key(device),
        )
    except (TypeError, ValueError):
        return _build_bandwidth_mask(shp2, lmt, soft, rfft, dtype, device)

    with _MASK_LOCK:
        hit = _MASK_CACHE.get(key)
        if hit is not None:
            _MASK_CACHE.move_to_end(key)
            return hit

    mask = _build_bandwidth_mask(shp2, lmt, soft, rfft, dtype, device)

    with _MASK_LOCK:
        if key not in _MASK_CACHE and len(_MASK_CACHE) >= _MASK_CACHE_MAX:
            _MASK_CACHE.popitem(last=False)
        _MASK_CACHE[key] = mask
    return mask


def bandwidth_limit_array_torch(
    arrayin: torch.Tensor,
    limit: Union[float, Sequence[float], None] = 2 / 3,
    qspace_in: bool = True,
    qspace_out: bool = True,
    soft: Optional[float] = None,
    rfft: bool = False,
) -> torch.Tensor:
    """Band-width limit the trailing two dimensions of a tensor.

    Multislice (Cowley and Moodie, Acta Cryst. 10 (1957) 609; Kirkland,
    *Advanced Computing in Electron Microscopy*, Ch. 6, Sec. 6.7) alternates a
    real-space multiplication by the transmission function
    ``t(r) = exp(i*sigma*V(r))`` with a reciprocal-space multiplication by the
    Fresnel propagator.  Multiplication in real space is convolution in
    reciprocal space, and on a periodic DFT grid whatever a convolution pushes
    past Nyquist wraps around and reappears as spurious *low*-frequency signal.
    A product of two functions each band-limited to ``q_c`` has support out to
    ``2*q_c``, so cutting at

        q_c = (2/3) * q_Nyq   ==>   2*q_c = (4/3) * q_Nyq

    guarantees that everything which wraps lands in the band
    ``(2/3, 1] * q_Nyq`` that the next application of the mask discards.  Hence
    the default ``limit = 2/3`` (the same argument gives Orszag's 2/3 rule in
    pseudo-spectral fluid dynamics, J. Atmos. Sci. 28 (1971) 1074).

    The mask itself is described in :func:`_build_bandwidth_mask`: an indicator
    of ``rho < 1``, or an erf taper of width ``soft`` reaching exactly zero at
    ``rho = 1``, with ``rho`` the elliptic radius in per-axis Nyquist-normalised
    coordinates.  The limit is a fraction of Nyquist and never sees the
    physical cell size.

    Pipeline: optionally forward-FFT over ``(-2, -1)``, multiply by the mask
    broadcast over all leading batch dimensions, optionally inverse-FFT back.

    Parameters
    ----------
    arrayin : torch.Tensor of shape ``(..., M_-2, M_-1)``
        Any number of leading batch dimensions (including none), real or
        complex, CPU or CUDA.
    limit : float, sequence of float, or None, optional
        Cut as a fraction of Nyquist.  A scalar is replicated to both axes; a
        sequence contributes its **last two** entries, in axis order
        ``(-2, -1)``, allowing an elliptic cut.  ``None`` disables masking
        entirely -- and with ``qspace_in=qspace_out=True`` makes the call an
        identity that returns *the same object*, which callers must not assume
        they own.  Default 2/3.
    qspace_in : bool, optional
        The input is already a spectrum in corner-origin order (default True);
        if False it is forward-FFT'd first.
    qspace_out : bool, optional
        Return the spectrum (default True); if False, inverse-FFT back to real
        space.  The inverse is the complex ``ifft2``, so the result is complex
        even for a Hermitian-symmetric spectrum.
    soft : None or float, optional
        ``None`` (default) for the hard cut; a positive width for the erf
        taper.  See the DC-gain caveat in :func:`_build_bandwidth_mask`.
    rfft : bool or int, optional
        Declare that the last dimension holds one-sided real-FFT frequencies,
        i.e. ``M_-1 = Nx//2 + 1``.  Its coordinates are then rebuilt as
        ``rfftfreq(2*(M_-1 - 1))``, while the ``M_-2`` axis stays two-sided.
        The full length can only be inferred from a half-spectrum up to
        parity: that reconstruction is exact for even ``Nx``, and for odd
        ``Nx`` it treats the highest column as Nyquist, scaling that axis by
        ``Nx/(Nx-1)`` (an error of order one column, i.e. ``1/Nx``).  Pass the
        true length as an integer -- ``rfft=Nx`` -- to make the odd case exact;
        with that, band-limiting an ``rfft2`` and inverting with ``irfft2``
        agrees with the two-sided path to round-off for either parity.

    Returns
    -------
    torch.Tensor
        Same shape and device as the input.  Complex if the input is complex or
        if an FFT was taken; otherwise the input's real dtype is preserved (a
        real spectrum in, a real spectrum out -- nothing is forced complex).

    Notes
    -----
    Non-mutating and autograd-safe: the input is never written to and never
    copied, and the mask multiply is out-of-place, so gradients flow through to
    ``arrayin`` and anything upstream of it (per-atom weights reach the
    potential builder this way).  The FFTs use ``norm='backward'``: unnormalised
    forward, ``1/(M_-2 * M_-1)`` in the inverse.

    Examples
    --------
    Transmission functions band-limited at the 2/3 rule, spectrum in::

        T = bandwidth_limit_array_torch(fft2(exp(1j*sigma*V)), limit=2/3)

    Projected potentials, half-spectrum with a soft edge touching Nyquist::

        P = bandwidth_limit_array_torch(P, limit=1, soft=0.05, rfft=True)
    """
    # --- limit=None: skip the mask, but still honour the transform flags -----
    if limit is None:
        if qspace_in and qspace_out:
            return arrayin
        array = (
            arrayin if qspace_in else torch.fft.fft2(arrayin, dim=(-2, -1))
        )
        if qspace_out:
            return array
        return torch.fft.ifft2(array, dim=(-2, -1))

    # --- per-axis limits ----------------------------------------------------
    if _is_array_like(limit):
        lmt = (limit[-2], limit[-1])  # trailing pair, in axis order (-2, -1)
    else:
        lmt = (limit, limit)
    try:
        lmt = (float(lmt[0]), float(lmt[1]))
    except (TypeError, ValueError):
        pass  # left raw; the memo key will fail and the build will coerce

    # --- pipeline -----------------------------------------------------------
    array = arrayin if qspace_in else torch.fft.fft2(arrayin, dim=(-2, -1))

    # The mask is requested at the input's underlying *real* dtype, so
    # complex64 stays complex64 and complex128 stays complex128 through the
    # multiply, and on the input's own device so nothing crosses the bus.
    mask = _bandwidth_mask(
        array.shape[-2:], lmt, soft, rfft, array.real.dtype, array.device
    )

    array = array * mask  # out-of-place: never touches the caller's tensor

    if qspace_out:
        return array
    return torch.fft.ifft2(array, dim=(-2, -1))


def clear_caches() -> None:
    """Drop the memoised frequency axes and band-limit masks.

    Purely a memory-management convenience: every cached object is a pure
    function of its key, so clearing changes no numerical result.  Worth
    calling after a large simulation, since cached masks pin device memory for
    the lifetime of the process (4 MB per distinct 1024x1024 float32 key).
    """
    with _AXIS_LOCK:
        _AXIS_CACHE.clear()
    with _MASK_LOCK:
        _MASK_CACHE.clear()
