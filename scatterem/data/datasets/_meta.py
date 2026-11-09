"""Keep a dataset's acquisition metadata in step with the dataset it describes.

``Dataset4DStem`` carries ``meta`` (an ``AcquisitionMetadata``) next to its own
``shape`` and ``sampling``. Two rules keep the
two from drifting apart:

* A dataset whose geometry differs from its source OWNS its metadata:
  ``copy``, ``crop``/``crop_``, ``pad_and_taper_*`` and ``bin_detector`` build
  theirs with :func:`copy_meta`, so binning, padding or moving a derived
  dataset can never reach back into its parent's calibration. Re-wraps over
  the very same array (the virtual bright field) share it BY
  REFERENCE on purpose: aberrations fitted on one are meant to show on the
  other (see ``_place_aberrations_on_device``).
* The ``array`` and ``sampling`` setters call :func:`sync_meta_shape` and
  :func:`sync_meta_sampling`, so a dataset's own metadata follows every
  geometry change it goes through (``bin``, ``pad``, a device move).

A metadata with ``parent_scan`` (a window of a parent raster) fails closed:
:func:`sync_meta_shape` refuses to change its scan dimensions unless the
caller moves the window in the same call (``parent_scan=``), because a stale
window would place every pattern at the wrong position without an error.
A ``detector_warp_centre`` (the disk centre in detector pixels) fails closed in
the same way on the detector dimensions: a crop (``index=``), ``bin_detector``,
``pad_detector`` and ``recentre_origin`` move it (:func:`moved_warp_centre`),
and any other change of the detector shape without ``warp_centre=`` raises.

The dataset's ``sampling`` is the source of truth and ``meta`` mirrors it.
:func:`meta_sampling` closes the loop at construction: a dataset built without
an explicit ``sampling`` starts from what its metadata describes, so the first
update does not push a ``1.0`` placeholder into a real calibration.
"""

import copy
import types
from collections.abc import Iterable, Sequence
from typing import Any, TypeVar

import numpy as np
from torch import Tensor

from scatterem.data._acquisition_fields import _parent_scan, _warp_centre
from scatterem.data.acquisition import AcquisitionMetadata

from ._parent_link import VBF_CACHE_ATTRS

#: ``parent_scan=`` default of :func:`copy_meta` / :func:`sync_meta_shape`.
KEEP = object()

#: Set to ``True`` on a :func:`with_meta` view whose metadata differs from its
#: source's; :class:`~scatterem.reconstruction.Provenance` records it.
META_OVERRIDE_ATTR = "_meta_override"


def copy_meta(
    meta: Any,
    shape: Sequence[int] | None = None,
    sampling: Any = None,
    parent_scan: Any = KEEP,
    index: Any = None,
    warp_centre: Any = KEEP,
    old_shape: Sequence[int] | None = None,
) -> Any:
    """Metadata for a dataset derived from the one ``meta`` describes.

    A deep copy, so the derived dataset's geometry updates stay on it, with
    ``shape`` and ``sampling`` already written in when given; ``parent_scan``
    (see :func:`sync_meta_shape`) moves the window with the shape. ``index``
    is the crop that made ``shape`` (``Dataset4DStem.crop``): it moves the
    window by the crop's start (:func:`cropped_parent_scan`), and the
    ``detector_warp_centre`` by the detector start (``warp_centre`` sets it
    instead; see :func:`sync_meta_shape`). ``old_shape`` is the shape of the
    dataset ``meta`` describes, for the detector check when ``meta.shape`` is
    unset (pass it whenever ``shape`` is given). Tensors still
    attached to an autograd graph (an ``aberrations.array`` a fit left
    non-leaf) are replaced by detached clones: ``copy.deepcopy`` refuses
    those, and the copy is a snapshot of the acquisition geometry, never part
    of a gradient path. ``None`` copies to ``None``; duck-typed metadata is
    deep-copied as is.
    """
    if meta is None:
        return None
    if index is not None and parent_scan is KEEP:
        parent_scan = cropped_parent_scan(meta, index, getattr(meta, "shape", None))
    if index is not None and warp_centre is KEEP:
        warp_centre = cropped_warp_centre(meta, index, getattr(meta, "shape", None))
    memo: dict[int, Any] = {}
    _detach_non_leaf_tensors(meta, memo, set())
    # A metadata snapshot: the measurement tensor is never inside ``meta``,
    # and the one autograd graph it can hold is detached above on purpose.
    new = copy.deepcopy(meta, memo)  # noqa: TID251
    if shape is not None:
        sync_meta_shape(
            new,
            shape,
            parent_scan=parent_scan,
            warp_centre=warp_centre,
            old_shape=old_shape,
        )
    elif parent_scan is not KEEP:
        raise ValueError("copy_meta: parent_scan= needs the new shape=")
    elif warp_centre is not KEEP and isinstance(new, AcquisitionMetadata):
        new.detector_warp_centre = _warp_centre(warp_centre)
    if sampling is not None:
        sync_meta_sampling(new, sampling)
    return new


def padded_meta(
    meta: Any,
    shape: Sequence[int],
    old_shape: Sequence[int],
    pad_ratio: float = 0.0,
) -> Any:
    """:func:`copy_meta` for a detector pad of ``int(n * pad_ratio)`` in front.

    ``old_shape`` is the shape before the pad (the detector check, also when
    ``meta.shape`` is unset). The leading margin of the pad
    (of the pad-and-taper pass) moves ``detector_warp_centre``;
    ``pad_ratio=0`` is a pad of the scan axes only.
    """
    lead = [int(int(n) * pad_ratio) for n in old_shape[-2:]]
    centre = moved_warp_centre(meta, lead)
    return copy_meta(meta, shape=shape, warp_centre=centre, old_shape=old_shape)


D = TypeVar("D")

def with_meta(dataset: D, meta: Any) -> D:
    """A cheap view of ``dataset`` that reads ``meta`` instead of its own.

    The view is a shallow copy: it shares the measurement array and every
    other attribute with ``dataset``, so it costs one attribute dictionary.
    Only ``meta`` differs, and the view starts with an empty virtual-bright-
    field cache, so a ``get_vbf`` on it builds a provider under ``meta``.
    ``dataset`` itself is not changed.

    Use it to reconstruct the same data with other metadata, for example two
    aberration fits::

        view = with_meta(dataset, fit.applied_to(dataset.meta))
        vbf = view.get_vbf(0.5)

    :func:`scatterem.reconstruction.run` takes ``meta=`` and does this for
    you.

    Parameters
    ----------
    dataset : object
        A dataset with a settable ``meta`` (a ``Dataset4DStem``, ...).
    meta : object
        The metadata the view reads. It is bound by reference, not copied, as
        an assignment ``dataset.meta = meta`` would bind it.

    Returns
    -------
    object
        The view, of the same type as ``dataset``.

    Raises
    ------
    TypeError
        If ``dataset`` has no ``meta`` attribute or it cannot be set.
    """
    if not hasattr(dataset, "meta"):
        raise TypeError(f"a {type(dataset).__name__} has no `meta` to override")
    view: Any = copy.copy(dataset)
    state = vars(view)
    for name in VBF_CACHE_ATTRS:
        state.pop(name, None)
    if meta is not view.meta:
        state[META_OVERRIDE_ATTR] = True
    try:
        view.meta = meta
    except AttributeError as error:
        raise TypeError(
            f"the `meta` of a {type(dataset).__name__} cannot be set"
        ) from error
    return view


def sync_meta_shape(
    meta: Any,
    shape: Sequence[int],
    parent_scan: Any = KEEP,
    warp_centre: Any = KEEP,
    old_shape: Sequence[int] | None = None,
) -> None:
    """Write the array shape ``meta`` describes.

    ``shape`` on an ``AcquisitionMetadata``. Anything else (``None``,
    duck-typed metadata) is left alone.

    Parameters
    ----------
    meta : Any
        The metadata to update.
    shape : sequence of int
        The new array shape.
    parent_scan : sequence of int or None, optional
        The window ``(y0, x0, NY, NX)`` that goes with ``shape`` (None: no
        window). Default: keep the current one, which is allowed only while
        the scan dimensions ``shape[:2]`` stay the same.
    warp_centre : (float, float) or None, optional
        The ``detector_warp_centre`` that goes with ``shape``. Default: keep
        the current one, which is allowed only while the detector dimensions
        ``shape[-2:]`` stay the same.
    old_shape : sequence of int, optional
        The shape before the change, for the detector check when
        ``meta.shape`` is unset (``from_array`` leaves it unset).

    Raises
    ------
    ValueError
        The scan dimensions change on a metadata with ``parent_scan`` and no
        new ``parent_scan`` is given, or the detector dimensions change on a
        metadata with ``detector_warp_centre`` and no new centre is given.
    """
    if not isinstance(meta, AcquisitionMetadata):
        return
    before = meta.shape if meta.shape is not None else old_shape
    if warp_centre is KEEP:
        centre = getattr(meta, "detector_warp_centre", None)
        if (
            centre is not None
            and before is not None
            and len(shape) >= 4
            and tuple(int(v) for v in before[-2:]) != tuple(int(v) for v in shape[-2:])
        ):
            raise ValueError(
                f"the detector changes from {tuple(int(v) for v in before[-2:])} "
                f"to {tuple(int(v) for v in shape[-2:])} but detector_warp_centre "
                f"{centre} is not updated: the warp would pivot about the wrong pixel"
            )
    new = tuple(int(v) for v in shape[:2])
    if parent_scan is KEEP:
        old = meta.shape
        if (
            meta.parent_scan is not None
            and old is not None
            and tuple(int(v) for v in old[:2]) != new
        ):
            raise ValueError(
                f"the scan changes from {tuple(int(v) for v in old[:2])} to "
                f"{new} but parent_scan {meta.parent_scan} is not updated: the "
                "patterns would sit at the wrong positions"
            )
    else:
        meta.parent_scan = _parent_scan(parent_scan)
    if warp_centre is not KEEP:
        meta.detector_warp_centre = _warp_centre(warp_centre)
    meta.shape = np.asarray(shape)


def sync_meta_sampling(meta: Any, sampling: Any) -> None:
    """Write the per-axis ``sampling`` ``meta`` describes.

    ``sampling`` on an ``AcquisitionMetadata`` (a numpy array is shared as is, the
    way the ``Dataset4DStem.sampling`` setter always has). Anything else is left
    alone.
    """
    if isinstance(meta, AcquisitionMetadata):
        meta.sampling = np.asarray(sampling)


def meta_sampling(meta: Any) -> Any:
    """The per-axis sampling ``meta`` describes, for a dataset built without one.

    ``None`` when ``meta`` describes no sampling at all.
    """
    return getattr(meta, "sampling", None)


def _detach_non_leaf_tensors(obj: Any, memo: dict[int, Any], seen: set[int]) -> None:
    """Seed a deepcopy ``memo`` so non-leaf grad tensors copy as detached clones.

    Walks containers and instance dictionaries; everything else keeps
    ``deepcopy``'s own behaviour (a leaf that requires grad still copies as
    one).
    """
    if id(obj) in seen:
        return
    seen.add(id(obj))
    if isinstance(obj, Tensor):
        if obj.requires_grad and not obj.is_leaf:
            memo[id(obj)] = obj.detach().clone()
        return
    children: Iterable[Any]
    if isinstance(obj, dict):
        children = obj.values()
    elif isinstance(obj, (list, tuple, set, frozenset)):
        children = obj
    elif isinstance(obj, (type, types.ModuleType, types.FunctionType)):
        return
    elif hasattr(obj, "__dict__"):
        children = vars(obj).values()
    else:
        return
    for child in children:
        _detach_non_leaf_tensors(child, memo, seen)


def cropped_parent_scan(meta: Any, index: Any, shape: Sequence[int] | None) -> Any:
    """The ``parent_scan`` of the crop ``index`` of a dataset of ``shape``.

    The window moves by the crop's start on each scan axis (``slice.indices``,
    so ``None`` and negative starts work). :data:`KEEP` when ``meta`` has no
    window.

    Raises
    ------
    ValueError
        A scan index that is not a slice, or a slice with a step other than 1:
        the crop would not be one contiguous block of the parent raster.
    """
    p = meta.parent_scan if isinstance(meta, AcquisitionMetadata) else None
    if p is None or shape is None:
        return KEEP
    idx = index if isinstance(index, tuple) else (index,)
    starts = []
    for ax in (0, 1):
        sl = idx[ax] if ax < len(idx) else slice(None)
        if not isinstance(sl, slice):
            raise ValueError(f"a crop of a parent_scan window needs slices, got {sl!r}")
        start, _, step = sl.indices(int(shape[ax]))
        if step != 1:
            raise ValueError(
                f"a strided scan crop (step {step}) of a parent_scan window is "
                "not one block of the parent raster"
            )
        starts.append(start)
    return (p[0] + starts[0], p[1] + starts[1], p[2], p[3])


def moved_warp_centre(
    meta: Any,
    offset: Sequence[float] = (0.0, 0.0),
    *,
    before: Sequence[float] = (0.0, 0.0),
    scale: float = 1.0,
    wrap: Sequence[int] | None = None,
) -> Any:
    """``meta.detector_warp_centre`` after a detector operation moved the pixels.

    Parameters
    ----------
    meta : Any
        The metadata before the operation.
    offset : (float, float), optional
        Pixels added to the centre after the binning (a pad adds its top and
        left margin, a crop subtracts its start).
    before : (float, float), optional
        Pixels added before the binning (the top and left zero padding of
        ``bin_detector``).
    scale : float, optional
        Binning factor: pixel ``p`` lands at ``(p + 0.5) / scale - 0.5``.
    wrap : (int, int), optional
        Frame size for a circular shift (``fftshift``): the result is taken
        modulo it.

    Returns
    -------
    tuple of float or object
        The moved centre, or :data:`KEEP` when ``meta`` has none.
    """
    centre = getattr(meta, "detector_warp_centre", None)
    if centre is None:
        return KEEP
    out = [
        (float(c) + float(b) + 0.5) / scale - 0.5 + float(o)
        for c, b, o in zip(centre, before, offset, strict=True)
    ]
    if wrap is not None:
        out = [v % int(n) for v, n in zip(out, wrap, strict=True)]
    return (out[0], out[1])


def cropped_warp_centre(meta: Any, index: Any, shape: Sequence[int] | None) -> Any:
    """The ``detector_warp_centre`` of the crop ``index`` of a dataset of ``shape``.

    :data:`KEEP` when ``meta`` has no centre. ``shape`` (None: unknown)
    resolves negative starts.

    Raises
    ------
    ValueError
        A detector index that is not a slice with step 1.
    """
    if getattr(meta, "detector_warp_centre", None) is None:
        return KEEP
    idx = index if isinstance(index, tuple) else (index,)
    starts = []
    for ax in (2, 3):
        sl = idx[ax] if ax < len(idx) else slice(None)
        if not isinstance(sl, slice):
            raise ValueError(
                f"a detector crop with a warp centre needs slices, got {sl!r}"
            )
        if shape is not None:
            start, _, step = sl.indices(int(shape[ax]))
        elif sl.start is not None and sl.start < 0:
            raise ValueError(
                "a detector crop from the end needs the detector shape in meta.shape"
            )
        else:
            start, step = sl.start or 0, sl.step or 1
        if step != 1:
            raise ValueError(
                f"a strided detector crop (step {step}) cannot carry "
                "detector_warp_centre"
            )
        starts.append(-start)
    return moved_warp_centre(meta, starts)


def binned_warp_centre(meta: Any, shape: Sequence[int], factor: int) -> Any:
    """The ``detector_warp_centre`` after ``bin_detector(factor)`` of ``shape``.

    ``bin_detector`` pads ``(factor - n % factor) % factor`` pixels per
    detector axis first, the smaller half on the top and left.
    """
    pad = [((factor - int(n) % factor) % factor) // 2 for n in shape[-2:]]
    return moved_warp_centre(meta, before=pad, scale=factor)
