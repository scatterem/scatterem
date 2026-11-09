"""Fourier-space primitives: sub-pixel shifts, phase ramps, rfft layout, rescaling.

Every function acts on the **last** ``ndim`` axes of a tensor, in array order
(``(z, y, x)`` or ``(y, x)``), on a corner-origin (``fftfreq``-order) grid with
frequencies in cycles per pixel. A positive shift moves content to larger
indices, so an integer shift equals ``torch.roll`` by the same amount, up to
rounding.

Written for scatterem from the Fourier shift theorem and the DFT crop / zero-pad
resampling identity; it comes from no other implementation. :func:`fourier_shift`
moved here from ``scatterem.physics.propagation`` and became N-D. Its 2-D path
with a host shift is bit-identical to the function it replaced, which existing
bit-exactness gates rely on.

Some performance-critical copies keep their own phase arithmetic deliberately;
each names its contract at its definition.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor

__all__ = [
    "dft_center",
    "fftshift_last2",
    "fftshift_rfft",
    "fourier_rescale",
    "fourier_shift",
    "ifftshift_rfft",
    "phase_ramp",
    "rfft_shape",
]


def _route_grad_entries(shift: Sequence[float] | Tensor) -> Sequence[float] | Tensor:
    """Stack a host sequence into a tensor shift if any entry requires grad.

    ``float()`` on such an entry would silently drop its gradient. Any other
    shift is returned as is, so the host path stays bit-identical.
    """
    if isinstance(shift, Tensor):
        return shift
    grad = [v for v in shift if isinstance(v, Tensor) and v.requires_grad]
    if not grad:
        return shift
    ref = grad[0]
    return torch.stack(
        [
            torch.as_tensor(v, dtype=ref.dtype, device=ref.device).reshape(())
            for v in shift
        ]
    )


def _shift_ndim(shift: Sequence[float] | Tensor) -> int:
    """Number of spatial axes a shift addresses."""
    if isinstance(shift, Tensor):
        if shift.ndim == 0:
            raise ValueError("a tensor shift needs a trailing axis of length ndim")
        return int(shift.shape[-1])
    return len(shift)


def _check_shift(shift: Sequence[float] | Tensor, ndim: int) -> None:
    """Raise unless ``shift`` addresses exactly ``ndim >= 1`` axes."""
    got = _shift_ndim(shift)
    if ndim < 1 or got != ndim:
        raise ValueError(f"shift addresses {got} axes but ndim is {ndim}")


def _ramp_argument(
    shape: Sequence[int],
    shift: Sequence[float] | Tensor,
    rfft: bool,
    ramp_dtype: torch.dtype,
    device: str | torch.device,
) -> Tensor:
    """``sum_i k_i * s_i`` on the grid, accumulated in axis order."""
    ndim = len(shape)
    if isinstance(shift, Tensor):
        s = shift.to(device=device, dtype=ramp_dtype)
        lead = tuple(s.shape[:-1])
        parts: list = [s[..., i].reshape(*lead, *(1,) * ndim) for i in range(ndim)]
    else:
        parts = [float(v) for v in shift]
    terms = []
    for i, n in enumerate(shape):
        if rfft and i == ndim - 1:
            k = torch.fft.rfftfreq(int(n), device=device, dtype=ramp_dtype)
        else:
            k = torch.fft.fftfreq(int(n), device=device, dtype=ramp_dtype)
        view = [1] * ndim
        view[i] = int(k.shape[0])
        terms.append(k.view(view) * parts[i])
    arg = terms[0]
    for term in terms[1:]:
        arg = arg + term
    return arg


def phase_ramp(
    shape: Sequence[int],
    shift: Sequence[float] | Tensor,
    *,
    rfft: bool = False,
    dtype: torch.dtype = torch.complex64,
    ramp_dtype: torch.dtype = torch.float64,
    device: str | torch.device | None = None,
) -> Tensor:
    """Build the Fourier phase ramp that shifts a grid of ``shape`` by ``shift``.

    The ramp is ``exp(-2j * pi * (k_0 * s_0 + k_1 * s_1 + ...))``, summed in axis
    order, with ``k_i`` the ``fftfreq`` (cycles per pixel) of axis ``i``. Multiplying
    a spectrum by it and transforming back moves the content by ``+shift`` pixels.

    For real ``x``, ``irfftn(rfftn(x) * phase_ramp(x.shape, s, rfft=True), x.shape)``
    equals ``fourier_shift(x, s)`` (up to rounding) only when no axis but the last
    carries Nyquist content: exactly so on odd grids, on grids even only in the
    last axis, and for data with nothing in the Nyquist planes of the other axes.
    On white noise with an even non-last axis the two differ, because
    :func:`fourier_shift` keeps the real part of a full-spectrum shift there.

    Parameters
    ----------
    shape : sequence of int
        Real-space grid shape ``(n_0, ..., n_{D-1})``.
    shift : sequence of float or Tensor
        Shift in pixels: a host sequence of length ``D`` (each entry a float or a
        0-d tensor, read with ``float()``), or a tensor ``(..., D)`` whose leading
        axes become leading axes of the ramp. A sequence with an entry that
        requires grad is stacked and takes the tensor path, so gradients flow.
    rfft : bool, optional
        Build the half spectrum of ``torch.fft.rfftn``: the last axis has
        ``n_{D-1} // 2 + 1`` non-negative frequencies. Default False.
    dtype : torch.dtype, optional
        Complex dtype of the returned ramp. Default ``torch.complex64``.
    ramp_dtype : torch.dtype, optional
        Real dtype the phase is computed in before the cast. Default
        ``torch.float64``.
    device : str or torch.device, optional
        Device of the ramp. Default: the shift's device for a tensor shift (or a
        stacked sequence), otherwise the CPU.

    Returns
    -------
    Tensor
        ``(..., *shape)`` ramp (last axis ``n_{D-1} // 2 + 1`` when ``rfft``).

    Raises
    ------
    ValueError
        If ``shift`` does not address exactly ``len(shape)`` axes.
    """
    ndim = len(shape)
    shift = _route_grad_entries(shift)
    _check_shift(shift, ndim)
    if device is None:
        device = shift.device if isinstance(shift, Tensor) else "cpu"
    arg = _ramp_argument(shape, shift, rfft, ramp_dtype, device)
    return torch.exp(-2j * math.pi * arg).to(dtype)


def _check_broadcast(lead_x: Sequence[int], lead_s: Sequence[int]) -> None:
    """Raise a ValueError naming both shapes when they cannot broadcast."""
    try:
        torch.broadcast_shapes(tuple(lead_x), tuple(lead_s))
    except RuntimeError as err:
        raise ValueError(
            f"shift leading shape {tuple(lead_s)} does not broadcast against the "
            f"leading shape {tuple(lead_x)} of x"
        ) from err


def fourier_shift(
    x: Tensor,
    shift: Sequence[float] | Tensor,
    *,
    ndim: int | None = None,
    ramp_dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Shift ``x`` by ``shift`` pixels along its last axes (Fourier shift theorem).

    Periodic; equals ``torch.roll`` for whole-pixel shifts up to FFT rounding. For
    real input, a fractional shift on an even grid loses part of the Nyquist
    frequency component (the ramp is not Hermitian there, and the real part is
    returned), so it is not exactly power preserving. A host shift is read with
    ``float()`` (a device sync for 0-d CUDA tensors), unless an entry requires
    grad: then the entries are stacked and take the tensor path. A tensor shift
    stays on the device and carries gradients. With a batched tensor shift, CPU
    results equal a per-slice loop only up to floating-point rounding. The rfft
    route ``irfftn(rfftn(x) * phase_ramp(..., rfft=True))`` agrees with this
    function for real ``x`` only under the Nyquist condition :func:`phase_ramp`
    states.

    Parameters
    ----------
    x : Tensor
        Real or complex ``(..., n_0, ..., n_{D-1})``; the last ``D`` axes move.
    shift : sequence of float or Tensor
        Shift in pixels, array order: a host sequence of length ``D``, or a
        tensor ``(..., D)`` whose leading axes broadcast against the leading
        axes of ``x``.
    ndim : int, optional
        Number of shifted axes ``D``. Default: inferred from ``shift``; when
        given it must agree with it.
    ramp_dtype : torch.dtype, optional
        Real dtype of the phase ramp before it is cast to the spectrum's
        complex dtype. Default ``torch.float64``.

    Returns
    -------
    Tensor
        The shifted array. Complex input keeps its dtype; real floating input
        returns the real part in its dtype; integer input returns the real part
        in torch's default floating dtype.

    Raises
    ------
    ValueError
        If ``shift`` and ``ndim`` disagree, ``D`` exceeds ``x.ndim``, or the
        leading shapes of a tensor shift and ``x`` do not broadcast.
    """
    shift = _route_grad_entries(shift)
    d = _shift_ndim(shift) if ndim is None else int(ndim)
    _check_shift(shift, d)
    if d > x.ndim:
        raise ValueError(f"cannot shift {d} axes of a {x.ndim}-D tensor")
    if isinstance(shift, Tensor):
        _check_broadcast(x.shape[:-d], shift.shape[:-1])
    axes = tuple(range(-d, 0))
    spectrum = torch.fft.fftn(x, dim=axes)
    ramp = phase_ramp(
        x.shape[-d:],
        shift,
        dtype=spectrum.dtype,
        ramp_dtype=ramp_dtype,
        device=x.device,
    )
    out = torch.fft.ifftn(spectrum * ramp, dim=axes)
    if x.is_complex():
        return out
    return out.real.to(x.dtype) if x.is_floating_point() else out.real


def rfft_shape(shape: Sequence[int]) -> tuple[int, ...]:
    """Return the shape of ``torch.fft.rfftn`` over every axis of ``shape``.

    Parameters
    ----------
    shape : sequence of int
        Real-space shape.

    Returns
    -------
    tuple of int
        ``shape`` with its last entry replaced by ``shape[-1] // 2 + 1``.
    """
    dims = [int(n) for n in shape]
    return (*dims[:-1], dims[-1] // 2 + 1)


def dft_center(
    shape: Sequence[int], *, rfft: bool, fftshifted: bool
) -> tuple[int, ...]:
    """Return the index of the zero frequency in a spectrum of real-space ``shape``.

    Parameters
    ----------
    shape : sequence of int
        Real-space shape.
    rfft : bool
        Whether the spectrum is an rfft half spectrum; its last axis is never
        shifted, so the zero frequency stays at index 0 there.
    fftshifted : bool
        Whether the (other) axes were fftshifted, which puts the zero frequency
        at ``n // 2``.

    Returns
    -------
    tuple of int
        Index of the zero frequency, one entry per axis.
    """
    centre = [int(n) // 2 if fftshifted else 0 for n in shape]
    if rfft:
        centre[-1] = 0
    return tuple(centre)


def fftshift_last2(x: Tensor) -> Tensor:
    """Return ``torch.fft.fftshift(x, dim=(-2, -1))``, bit-identical, ONE buffer.

    Four quadrant copies instead of two rolls: half the memory.

    Parameters
    ----------
    x : Tensor
        Tensor ``(..., ny, nx)``, e.g. a 4D-STEM cube.

    Returns
    -------
    Tensor
        New tensor; element ``i`` of the last two dims is at ``(i + n//2) % n``.
    """
    ny, nx = x.shape[-2:]
    sy, sx = ny // 2, nx // 2
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    out[..., sy:, sx:] = x[..., : ny - sy, : nx - sx]
    out[..., sy:, :sx] = x[..., : ny - sy, nx - sx :]
    out[..., :sy, sx:] = x[..., ny - sy :, : nx - sx]
    out[..., :sy, :sx] = x[..., ny - sy :, nx - sx :]
    return out


def fftshift_rfft(x: Tensor, ndim: int) -> Tensor:
    """Apply ``fftshift`` to every spatial axis of an rfft spectrum but the last.

    Parameters
    ----------
    x : Tensor
        Half spectrum ``(..., n_0, ..., n_{D-1} // 2 + 1)``.
    ndim : int
        Number of spatial axes ``D``.

    Returns
    -------
    Tensor
        ``x`` with its zero frequency moved to ``n_i // 2`` on axes ``0 .. D-2``.
    """
    dims = tuple(range(-ndim, -1))
    return torch.fft.fftshift(x, dim=dims) if dims else x


def ifftshift_rfft(x: Tensor, ndim: int) -> Tensor:
    """Undo :func:`fftshift_rfft`.

    Parameters
    ----------
    x : Tensor
        Centred half spectrum ``(..., n_0, ..., n_{D-1} // 2 + 1)``.
    ndim : int
        Number of spatial axes ``D``.

    Returns
    -------
    Tensor
        ``x`` with its zero frequency back at index 0 on axes ``0 .. D-2``.
    """
    dims = tuple(range(-ndim, -1))
    return torch.fft.ifftshift(x, dim=dims) if dims else x


def _resize_centred_axis(spec: Tensor, axis: int, n_out: int) -> Tensor:
    """Crop or zero-pad one axis of a centred (fftshifted) spectrum to ``n_out``.

    The band both lengths represent is copied. Padding an even length splits its
    unpaired Nyquist bin evenly between ``-n/2`` and ``+n/2``; cropping to an even
    length folds ``+n_out/2`` onto ``-n_out/2``, the bin both alias to. So a pad
    followed by the matching crop returns the input exactly.
    """
    n_in = int(spec.shape[axis])
    if n_out == n_in:
        return spec
    lo = -min(n_in // 2, n_out // 2)
    width = min((n_in - 1) // 2, (n_out - 1) // 2) - lo + 1
    shape = list(spec.shape)
    shape[axis] = n_out
    out = spec.new_zeros(shape)
    out.narrow(axis, lo + n_out // 2, width).copy_(
        spec.narrow(axis, lo + n_in // 2, width)
    )
    if n_out > n_in and n_in % 2 == 0:
        half = out.narrow(axis, n_out // 2 - n_in // 2, 1) * 0.5
        out.narrow(axis, n_out // 2 - n_in // 2, 1).copy_(half)
        out.narrow(axis, n_out // 2 + n_in // 2, 1).copy_(half)
    elif n_out < n_in and n_out % 2 == 0:
        out.narrow(axis, 0, 1).add_(spec.narrow(axis, n_in // 2 + n_out // 2, 1))
    return out


def _centred_taper(
    n_in: Sequence[int], n_out: Sequence[int], taper: float, device
) -> Tensor:
    """Raised-cosine weight on the centred output spectrum, 1 at DC.

    The radius is measured per axis in units of the band edge both grids
    hold, ``min(n_in, n_out) / 2`` bins. The weight is 1 below ``1 - taper``
    and falls to 0 at 1.
    """
    r2 = torch.zeros((), dtype=torch.float64, device=device)
    for axis, (n, m) in enumerate(zip(n_in, n_out, strict=True)):
        j = torch.arange(m, dtype=torch.float64, device=device) - m // 2
        shape = [1] * len(n_out)
        shape[axis] = m
        r2 = r2 + ((j / (min(n, m) / 2.0)) ** 2).view(shape)
    ramp = ((1.0 - r2.sqrt()) / taper).clamp(0.0, 1.0)
    return 0.5 - 0.5 * torch.cos(math.pi * ramp)


def _check_samplings(name: str, values: Sequence[float]) -> None:
    """Raise ``ValueError`` naming ``name`` unless every entry is finite and > 0."""
    for i, v in enumerate(values):
        if not (math.isfinite(float(v)) and float(v) > 0.0):
            raise ValueError(f"{name}[{i}] must be finite and > 0; got {v!r}")


def fourier_rescale(
    x: Tensor,
    sampling_in: Sequence[float],
    sampling_out: Sequence[float],
    *,
    ndim: int | None = None,
    taper: float | None = None,
) -> tuple[Tensor, tuple[float, ...]]:
    """Resample ``x`` to a new pixel size by cropping or zero-padding its spectrum.

    This is the anti-aliased down-sampling: every frequency above the new
    Nyquist frequency is removed, not folded back. A box average
    (``F.interpolate(mode="area")``) folds a crystal lattice above the new
    Nyquist frequency onto the coarse grid. ``taper`` adds a raised-cosine
    roll-off below the band edge, which removes the ringing of a hard cut.

    Axis ``i`` gets ``round(n_i * sampling_in[i] / sampling_out[i])`` samples (half
    rounds up). The band-limited interpolant is preserved: up-sampling by an
    integer factor reproduces the input at the original sample positions, and the
    mean value is unchanged. Sample 0 stays at the origin (corner convention).

    Parameters
    ----------
    x : Tensor
        Real or complex ``(..., n_0, ..., n_{D-1})``; the last ``D`` axes are
        resampled.
    sampling_in : sequence of float
        Pixel size of ``x`` per resampled axis.
    sampling_out : sequence of float
        Requested pixel size per resampled axis.
    ndim : int, optional
        Number of resampled axes ``D``. Default ``len(sampling_in)``.
    taper : float or None, optional
        Width of a raised-cosine roll-off, as a fraction of the band edge (the
        lower of the two Nyquist frequencies, per axis, on an elliptical
        radius). The weight is 1 below ``1 - taper`` and 0 at the edge; the
        DC weight is 1, so the mean stays. ``None`` (the default) is a hard
        cut. The roll-off also applies when no axis changes size.

    Returns
    -------
    rescaled : Tensor
        The resampled array; real input gives real output in its dtype, and
        integer input a floating array (never truncated back to integers).
        When no axis changes size, ``x`` itself is returned, not a copy.
    sampling : tuple of float
        The pixel size actually achieved, ``n_i * sampling_in[i] / n_out_i``;
        exactly ``sampling_in[i]`` on an axis whose size did not change.

    Raises
    ------
    ValueError
        If the sampling lengths disagree with ``ndim``, ``ndim`` exceeds
        ``x.ndim``, a sampling is not finite and positive, an axis would get
        fewer than one sample, or ``taper`` is not in ``(0, 1]``.
    """
    d = len(sampling_in) if ndim is None else int(ndim)
    if not 1 <= d <= x.ndim or len(sampling_in) != d or len(sampling_out) != d:
        raise ValueError(
            f"need 1 <= ndim <= x.ndim and one sampling per axis; got ndim={d}, "
            f"x.ndim={x.ndim}, {len(sampling_in)} and {len(sampling_out)} samplings"
        )
    _check_samplings("sampling_in", sampling_in)
    _check_samplings("sampling_out", sampling_out)
    if taper is not None and not 0.0 < float(taper) <= 1.0:
        raise ValueError(f"taper must be None or in (0, 1]; got {taper!r}")
    n_in = [int(n) for n in x.shape[-d:]]
    ratio = [
        float(a) / float(b) for a, b in zip(sampling_in, sampling_out, strict=True)
    ]
    n_out = [math.floor(n * r + 0.5) for n, r in zip(n_in, ratio, strict=True)]
    if min(n_out) < 1:
        raise ValueError(f"resampling {tuple(n_in)} gives {tuple(n_out)} samples")
    achieved = tuple(
        float(s) if m == n else n * float(s) / m
        for n, s, m in zip(n_in, sampling_in, n_out, strict=True)
    )
    if n_out == n_in and taper is None:
        return x, achieved
    axes = tuple(range(-d, 0))
    spec = torch.fft.fftshift(torch.fft.fftn(x, dim=axes), dim=axes)
    for axis, m in zip(axes, n_out, strict=True):
        spec = _resize_centred_axis(spec, axis, m)
    if taper is not None:
        weight = _centred_taper(n_in, n_out, float(taper), spec.device)
        spec = spec * weight.to(spec.real.dtype)
    out = torch.fft.ifftn(torch.fft.ifftshift(spec, dim=axes), dim=axes)
    out = out * (math.prod(n_out) / math.prod(n_in))
    if x.is_complex():
        return out, achieved
    return (out.real.to(x.dtype) if x.is_floating_point() else out.real), achieved
