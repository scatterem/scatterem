"""The link from a virtual bright field back to the dataset it was built from.

``Dataset4DStem.get_vbf`` caches every provider it builds on the dataset, in
the attributes ``VBF_CACHE_ATTRS``, and each provider records that dataset
as its ``parent_dataset``. Held strongly in both directions, the two formed a
reference cycle. Reference counting cannot free a cycle; only Python's cyclic
garbage collector can, and it is scheduled by object counts, not by the bytes a
tensor holds. ``del dataset`` therefore left the cube, the bright-field stack
and its cached FFT ``G`` allocated until a collection happened to run: 7.5 GiB
per EMPIAR-12236 scan in a batch loop, which ran a 96 GB GPU out of memory
after about 20 scans unless the caller knew to call ``gc.collect()``.

The dataset owns its cache, and the cache must stay strong: it is reused
across calls on a live dataset, and on full-size data its ``G`` alone is about
17 GiB. The cycle is broken on the provider's side instead. :class:`ParentLink`
holds the parent through a weak reference, backed by a strong reference to a
*stand-in*: a shallow copy of the parent, taken when the link is set, without
the vBF cache. The stand-in shares the parent's tensors and metadata, so it
costs one attribute dictionary, and it refers to no provider, so it closes no
cycle. ``parent_dataset`` then reads

* the parent itself while anything else keeps it alive, so
  ``dataset.get_vbf(t).parent_dataset is dataset`` holds as before;
* the stand-in once the parent is gone, so a provider taken from a temporary
  dataset (``Dataset4DStem.from_array(...).get_vbf(0.5)``) still reads the
  cube, sampling and metadata it came from -- the COM rotation prior and the
  out-of-core providers do.

The stand-in is the parent as it was when the link was set. An attribute
*reassigned* on the parent afterwards (a new ``array`` after an in-place
``bin``, ``pad`` or device move) does not reach it, and the stand-in keeps the
replaced value alive for as long as the provider lives; that value is the one
the provider's own bright-field stack was gathered from. Changes made in
place, to a shared tensor or to the shared metadata, reach both.
"""

from __future__ import annotations

import copy
import weakref
from typing import TYPE_CHECKING, overload

if TYPE_CHECKING:
    from typing import Self

    # Annotation only: dataset_4dstem imports virtual_bright_field, which
    # imports this module.
    from scatterem.data.acquisition import AcquisitionMetadata

    from .dataset_4dstem import Dataset4DStem

#: The attributes ``Dataset4DStem.get_vbf`` reads and writes: the current
#: provider, its mask threshold, and the demoted ``(threshold, provider)``.
VBF_CACHE_ATTRS = ("vBF", "_vbf_threshold", "_vbf_spare")


def without_vbf_cache(dataset: Dataset4DStem) -> Dataset4DStem:
    """A shallow copy of ``dataset`` that holds no virtual-bright-field cache.

    Parameters
    ----------
    dataset : Dataset4DStem
        The dataset to copy.

    Returns
    -------
    Dataset4DStem
        A new object sharing every attribute value of ``dataset`` (tensors and
        metadata by reference) except those in ``VBF_CACHE_ATTRS``.
    """
    standin = copy.copy(dataset)
    for name in VBF_CACHE_ATTRS:
        vars(standin).pop(name, None)
    return standin


class _Link:
    """A weak reference to a parent dataset, with a strong stand-in behind it."""

    __slots__ = ("_parent", "_standin")

    def __init__(self, parent: Dataset4DStem, standin: Dataset4DStem) -> None:
        self._parent = weakref.ref(parent)
        self._standin = standin

    def __call__(self) -> Dataset4DStem:
        parent = self._parent()
        return self._standin if parent is None else parent

    def __reduce__(self) -> tuple[type[_Link], tuple[Dataset4DStem, Dataset4DStem]]:
        # A weak reference does not pickle. The parent it resolves to does, and
        # pickle's memo keeps it the one object the rest of the graph refers to.
        return (_Link, (self(), self._standin))


class ParentLink:
    """Descriptor for ``parent_dataset``: the parent while it lives, then a stand-in.

    Assigning a dataset stores a :class:`_Link` to it in the instance
    dictionary; assigning ``None`` stores ``None``. Reading returns the live
    parent, the stand-in once the parent has been freed, or ``None``.
    """

    def __set_name__(self, owner: type, name: str) -> None:
        self._slot = f"_{name}_link"

    @overload
    def __get__(self, obj: None, objtype: type | None = None) -> Self: ...

    @overload
    def __get__(
        self, obj: object, objtype: type | None = None
    ) -> Dataset4DStem | None: ...

    def __get__(
        self, obj: object | None, objtype: type | None = None
    ) -> Self | Dataset4DStem | None:
        if obj is None:
            return self
        link: _Link | None = vars(obj).get(self._slot)
        return None if link is None else link()

    def __set__(self, obj: object, parent: Dataset4DStem | None) -> None:
        link = None if parent is None else _Link(parent, without_vbf_cache(parent))
        vars(obj)[self._slot] = link


class MetaDropsVbfCache:
    """Descriptor for ``Dataset4DStem.meta``: a new object drops the vBF cache.

    Each provider that :meth:`Dataset4DStem.get_vbf` caches reads the
    aberrations and the rotation from its own ``meta``. That ``meta`` is the
    metadata object that the dataset had when the provider was built.

    A caller can bind a different object to ``dataset.meta``, for example
    ``dataset.meta = fit.applied_to(dataset.meta)``. The cached providers then
    keep the old values, and a reconstruction at a cached threshold uses them
    without a warning. To prevent this, an assignment of a new object removes
    ``VBF_CACHE_ATTRS`` from the instance. An assignment of the object
    that is already bound keeps them. In-place changes to the metadata reach
    the providers, because the providers share the object.

    The value is in the instance dictionary, under the name of the attribute.
    """

    def __set_name__(self, owner: type, name: str) -> None:
        self._name = name

    @overload
    def __get__(self, obj: None, objtype: type | None = None) -> Self: ...

    @overload
    def __get__(
        self, obj: object, objtype: type | None = None
    ) -> AcquisitionMetadata | None: ...

    def __get__(
        self, obj: object | None, objtype: type | None = None
    ) -> Self | AcquisitionMetadata | None:
        if obj is None:
            return self
        return vars(obj).get(self._name)

    def __set__(self, obj: object, meta: AcquisitionMetadata | None) -> None:
        state = vars(obj)
        if state.get(self._name) is not meta:
            for name in VBF_CACHE_ATTRS:
                state.pop(name, None)
        state[self._name] = meta
