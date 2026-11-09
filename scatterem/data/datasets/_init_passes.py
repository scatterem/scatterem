"""Whole-cube passes of ``Dataset4DStem.__init__``.

``dataset_4dstem.py`` is frozen size debt, so these live here.
"""

from __future__ import annotations

import threading
import weakref
from collections.abc import Sequence

import torch
from torch import Tensor

from scatterem.physics.fourier import fftshift_last2

from . import _dose

__all__ = [
    "clamp_sqrt_total",
    "crop_array",
    "fourier_shift",
    "hand_over_row_sums",
    "normalization_const",
    "normalize_pending",
    "shift_and_clamp",
]

#: Per thread, the row sums that :func:`hand_over_row_sums` keeps for the buffer
#: that a fused producer (``tapering``, ``binning``, :func:`crop_array`) has just
#: written: ``(weak reference, buffer._version, sums, normalized)``, or ``None``.
_HANDED = threading.local()

#: Per thread, a weak reference to the cube that a fused pass has already divided
#: by the constructor's normalization constant, or ``None``.
#: :func:`normalize_pending` reads and clears it.
_NORMALIZED = threading.local()

#: The buffers that a producer handed over as divided, keyed by ``id`` and held
#: weakly. Unlike the per-thread slot of :data:`_HANDED`, a later hand-over does
#: not clear an entry: :func:`clamp_sqrt_total` refuses a divided buffer whose
#: sums were lost. Not a ``WeakSet``: its lookup compares tensors with ``==``.
_DIVIDED: dict[int, weakref.ref] = {}
_DIVIDED_LOCK = threading.Lock()


def _mark_divided(buffer: Tensor) -> None:
    """Record ``buffer`` as divided until its constructor takes it."""
    key = id(buffer)

    def forget(_ref: weakref.ref) -> None:
        with _DIVIDED_LOCK:
            if _DIVIDED.get(key) is _ref:
                del _DIVIDED[key]

    with _DIVIDED_LOCK:
        _DIVIDED[key] = weakref.ref(buffer, forget)


def _take_divided(array: Tensor) -> bool:
    """Whether ``array`` was handed over as divided; forget it."""
    with _DIVIDED_LOCK:
        ref = _DIVIDED.get(id(array))
        if ref is None or ref() is not array:
            return False
        del _DIVIDED[id(array)]
        return True


def fourier_shift(array: Tensor, dims: int | Sequence[int]) -> Tensor:
    """``torch.fft.fftshift(array, dim=dims)``, a new tensor.

    ``fftshift`` rolls one dim at a time: on two dims it holds two full-size
    buffers and makes two passes. When ``dims`` names exactly the last two dims
    (the detector of a 4D-STEM cube), :func:`~scatterem.physics.fourier.fftshift_last2` writes the same
    elements into ONE buffer (1.88x less own peak). Any other
    ``dims`` keeps torch's shift.

    Parameters
    ----------
    array : Tensor
        The cube.
    dims : int or sequence of int
        The dims to shift, as ``torch.fft.fftshift`` takes them.

    Returns
    -------
    Tensor
        The shifted cube.
    """
    if _is_last2(array, dims):
        return fftshift_last2(array)
    return torch.fft.fftshift(array, dim=dims)


def _is_last2(array: Tensor, dims: int | Sequence[int]) -> bool:
    """Whether ``dims`` names exactly the last two dims of ``array``."""
    nd = array.ndim
    tup = (dims,) if isinstance(dims, int) else tuple(dims)
    return (
        nd >= 2
        and all(-nd <= d < nd for d in tup)
        and sorted(d % nd for d in tup) == [nd - 2, nd - 1]
    )


def shift_and_clamp(
    array: Tensor,
    dims: int | Sequence[int] | None,
    clip_neg_values: bool,
    amplitudes: bool,
    normalize: bool = False,
) -> tuple[Tensor, Tensor | None]:
    """:func:`fourier_shift`, and on CUDA also the clamp, root and row sums.

    On CUDA float32, with ``dims`` on the last two dims and ``clip_neg_values``,
    one Warp pass of ``_shift_clamp_warp`` writes the shifted cube, clamps it,
    takes the root when ``amplitudes``, and adds the float64 row sums in
    torch's order. The bits are those of the shift followed
    by :func:`clamp_sqrt_total`. With ``normalize`` the pass of
    ``_shift_norm_warp`` also divides by the constructor's normalization
    constant, found first from ``array``; then
    :func:`normalize_pending` tells the constructor to skip its division.

    Parameters
    ----------
    array : Tensor
        The cube. It is only read.
    dims : int or sequence of int or None
        The dims to shift; ``None`` shifts nothing.
    clip_neg_values : bool
        Whether the constructor clamps negative values to zero.
    amplitudes : bool
        Whether the constructor takes the square root.
    normalize : bool, optional
        Whether the constructor normalizes.

    Returns
    -------
    tuple
        ``(cube, row_sums)``. ``cube`` is ``array`` itself when ``dims`` is
        ``None``, else a new tensor. ``row_sums`` is ``None``, or the float64
        row sums of a cube that is already clamped (and rooted when
        ``amplitudes``), before any division; pass it on to
        :func:`clamp_sqrt_total`.
    """
    if dims is None:
        return array, None
    if clip_neg_values and array.is_cuda and _is_last2(array, dims):
        try:
            from ._shift_clamp_warp import fftshift_last2_clamp_and_row_sums_f64
            from ._shift_norm_warp import (
                fftshift_last2_clamp_normalize_and_row_sums_f64 as shift_norm,
            )
        except ImportError:  # no fused Warp pass in this build
            pass
        else:
            fused = shift_norm(array, sqrt=amplitudes) if normalize else None
            if fused is not None:
                _NORMALIZED.ref = weakref.ref(fused[0])
                return fused
            fused = fftshift_last2_clamp_and_row_sums_f64(array, sqrt=amplitudes)
            if fused is not None:
                return fused
    return fourier_shift(array, dims), None


def _fused_crop(parent: Tensor, view: Tensor, normalize: bool) -> Tensor | None:
    """The crop pass of ``_crop_clamp_warp``, its sums handed over; or ``None``."""
    try:
        from ._crop_clamp_warp import (
            crop_clamp_and_row_sums_f64,
            crop_clamp_normalize_and_row_sums_f64,
        )
    except ImportError:  # no fused Warp pass in this build
        return None
    # Under inference mode the hand-over is refused, and a divided crop would
    # then be summed and divided again: only the undivided pass may run there.
    if normalize and not torch.is_inference_mode_enabled():
        fused = crop_clamp_normalize_and_row_sums_f64(parent, view)
        if fused is not None:
            hand_over_row_sums(*fused, normalized=True)
            return fused[0]
    fused = crop_clamp_and_row_sums_f64(parent, view)
    if fused is None:
        return None
    hand_over_row_sums(*fused)
    return fused[0]


def crop_array(
    parent: Tensor,
    index: tuple[slice, ...],
    clone: bool,
    normalize: bool,
    clip_neg_values: bool,
) -> Tensor:
    """``parent[index]``, cloned when ``clone``, for ``Dataset4DStem.crop``.

    On CUDA float32, with ``clone`` and ``clip_neg_values``, one Warp pass of
    ``_crop_clamp_warp`` reads the strided slice, clamps it, writes the new
    buffer and adds the float64 row sums in torch's order.
    With ``normalize`` the pass also divides by the constructor's normalization
    constant, found first from ``parent``.  The sums go to the
    constructor through :func:`hand_over_row_sums`.  The constructor's result
    has the same bits as clone, clamp, sum and divide.

    Parameters
    ----------
    parent : Tensor
        The dataset's cube.  It is only read.
    index : tuple of slice
        The crop.
    clone : bool
        Whether the crop gets its own buffer; else it is a view of ``parent``.
    normalize : bool
        Whether the constructor normalizes the crop.
    clip_neg_values : bool
        Whether the constructor clamps the crop.

    Returns
    -------
    Tensor
        The view, or a new buffer (already clamped, and divided, when a fused
        pass ran).
    """
    view = parent[index]
    if not clone:
        return view
    if clip_neg_values and view.is_cuda:
        fused = _fused_crop(parent, view, normalize)
        if fused is not None:
            return fused
    return view.clone()


def hand_over_row_sums(
    padded: Tensor, row_sums: Tensor, normalized: bool = False
) -> None:
    """Keep the clamp + dose sums of ``padded`` for :func:`clamp_sqrt_total`.

    The fused pad-and-taper pass calls this after it runs,
    and so do the crop and bin passes and the
    normalizing pad and crop passes. :func:`clamp_sqrt_total`
    takes the sums only for this exact buffer, and only if no in-place op
    changed it since.  An inference tensor has no version counter, so it hands
    over nothing: the constructor then clamps and sums itself, with the same
    bits.  ``normalized`` says that ``padded`` is already divided by the
    constructor's normalization constant; a normalizing producer must not run
    under inference mode.  A divided buffer is also recorded on its own, so a
    constructor that meets it after another hand-over cleared the slot raises.
    """
    if padded.is_inference():
        _HANDED.entry = None
        return
    if normalized:
        _mark_divided(padded)
    _HANDED.entry = (weakref.ref(padded), padded._version, row_sums, normalized)


def _handed_row_sums(array: Tensor) -> tuple[Tensor, bool] | None:
    """``(sums, normalized)`` that :func:`hand_over_row_sums` keeps for ``array``.

    ``None`` when there are none, or they belong to another buffer, or an
    in-place op changed ``array`` since.  A divided buffer that changed cannot
    give back its undivided values, so that case raises.
    """
    entry = getattr(_HANDED, "entry", None)
    _HANDED.entry = None
    if entry is None:
        return None
    ref, version, row_sums, normalized = entry
    if ref() is not array:
        return None
    if array._version != version:
        if normalized:
            raise RuntimeError("a normalized buffer changed before its constructor")
        return None
    return row_sums, normalized


def normalize_pending(array: Tensor, normalize: bool) -> bool:
    """Whether the constructor must still divide ``array`` by its constant.

    A fused pass may have divided already: the fftshift pass of
    :func:`shift_and_clamp`, or a crop or pad pass whose sums
    :func:`clamp_sqrt_total` took.

    Parameters
    ----------
    array : Tensor
        The constructor's cube.
    normalize : bool
        Whether the constructor normalizes.

    Returns
    -------
    bool
        ``normalize``, unless a fused pass has divided ``array`` already.

    Raises
    ------
    RuntimeError
        If ``array`` is already divided but ``normalize`` is False.
    """
    ref = getattr(_NORMALIZED, "ref", None)
    _NORMALIZED.ref = None
    done = ref is not None and ref() is array
    if done and not normalize:
        raise RuntimeError("a fused pass divided a cube that is not to be normalized")
    return normalize and not done


def normalization_const(array: Tensor, shape: Sequence[int]) -> Tensor:
    """The constructor's ``temp_3d.mean(0).max()``.

    Parameters
    ----------
    array : Tensor
        The clamped cube.
    shape : sequence of int
        Its 4-D shape ``(scan y, scan x, detector y, detector x)``.

    Returns
    -------
    Tensor
        The largest pixel of the mean diffraction pattern, a 0-d tensor.
    """
    # Calculate normalization before creating the view
    temp_3d = array.contiguous().view(shape[0] * shape[1], shape[2], shape[3])
    return temp_3d.mean(0).max()


def _fused_row_sums(array: Tensor, amplitudes: bool) -> Tensor | None:
    """The Warp pass's per-row sums, or ``None`` when the torch ops must run.

    :func:`._clamp_cast_warp.clamp_and_row_sums_f64` returns ``None`` where its
    Warp pass does not apply. The import is lazy, so Warp loads on first use.
    """
    from ._clamp_cast_warp import clamp_and_row_sums_f64

    return clamp_and_row_sums_f64(array, sqrt=amplitudes)


def clamp_sqrt_total(
    array: Tensor,
    clip_neg_values: bool,
    amplitudes: bool,
    row_sums: Tensor | None = None,
) -> float:
    """Clamp and square-root ``array`` IN PLACE; return its float64 total.

    The torch spelling is ``clamp_min_(0)`` when ``clip_neg_values``, then
    ``sqrt_()`` when ``amplitudes``, then :func:`._dose.float64_slice_total`.
    ``self._array[self._array < 0] = 0`` used to materialise a full-cube bool
    mask (3.06 GiB on the 512x512x112x112 figure-1 cube, at the peak) and make
    a second, masked pass; ``clamp_min_`` is one in-place pass (118.8 -> 50.0
    ms and -3136 MiB of peak). torch has no ``clamp`` for bool and no bool
    value is negative, so a bool cube skips the clamp.

    On CUDA float32 the Warp pass of ``_clamp_cast_warp`` clamps and writes the
    float64 copy of each row in one read, and takes the root
    in the same read when ``amplitudes``. For rows of 2048 elements or
    more it adds the sums itself, in the order of torch's own per-row
    ``.sum()``; else torch's ``.sum()`` makes them. Values and bits are
    the same. Without the clamp, torch's ``sqrt`` of a negative is NaN, so that
    case keeps the torch ops.

    The sums may also exist already: :func:`shift_and_clamp` returns them
    (``row_sums``), and :func:`hand_over_row_sums` keeps them for a padded,
    cropped or binned buffer. Then
    ``array`` is already clamped (and rooted), and only the host total runs.
    When the handed buffer is also divided, this function
    records it for :func:`normalize_pending`.  A divided buffer whose sums
    another hand-over replaced raises: its undivided values are gone.

    Parameters
    ----------
    array : Tensor
        The cube; its first axis is the outer slice.
    clip_neg_values : bool
        Clamp negative values to zero.
    amplitudes : bool
        Take the square root (``transform_to_amplitudes``).
    row_sums : Tensor or None, optional
        The float64 row sums of :func:`shift_and_clamp`: ``array`` is then
        already clamped, and rooted when ``amplitudes``.

    Returns
    -------
    float
        The float64 total of the result, ``0.0`` for an empty first axis.
    """
    divided = _take_divided(array)
    handed = _handed_row_sums(array)
    if divided and handed is None:
        raise RuntimeError("a normalized buffer lost its hand-over")
    if row_sums is not None:
        return _dose.host_total(row_sums)
    if handed is not None and clip_neg_values and not amplitudes:
        if handed[1]:
            _NORMALIZED.ref = weakref.ref(array)
        return _dose.host_total(handed[0])
    if handed is not None and handed[1]:
        raise RuntimeError("a normalized buffer reached a constructor that changes it")
    if clip_neg_values and array.is_cuda:
        row_sums = _fused_row_sums(array, amplitudes)
        if row_sums is not None:
            return _dose.host_total(row_sums)
    if clip_neg_values and array.dtype is not torch.bool:
        array.clamp_min_(0)
    if amplitudes:
        array.sqrt_()
    return _dose.float64_slice_total(array)
