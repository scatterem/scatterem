"""Throwaway views that keep a reconstruction's caching off the caller's dataset.

The direct reconstructions build their virtual bright field through
``Dataset4DStem.get_vbf``, which caches the provider on the dataset (``vBF``,
``_vbf_threshold``, ``_vbf_spare``). A runner calls the very same code on a
shallow copy of the dataset instead: the copy shares the measurement array and
the metadata, has its own attribute dictionary, and is discarded afterwards,
so whatever the reconstruction caches lands on the copy.

Two ``reuse=`` keys seed the copy's cache:

* ``"vbf"`` -- one provider, built earlier at the config's threshold;
* ``"vbf_cache"`` -- a whole :class:`VbfCache` (what :func:`vbf_cache_of` read
  off a dataset). The result then reports the copy's final cache as
  ``diagnostics["vbf_cache"]``; the runner itself writes nothing onto the
  caller's dataset.
"""

import copy
from collections.abc import Mapping
from typing import Any, NamedTuple

from scatterem.data.datasets._meta import copy_meta
from scatterem.data.datasets._parent_link import VBF_CACHE_ATTRS

#: The attributes ``Dataset4DStem.get_vbf`` reads and writes.
VBF_ATTRS = VBF_CACHE_ATTRS

#: The ``reuse=`` keys that seed a view's vBF cache.
VBF_REUSE_KEYS = ("vbf", "vbf_cache")


class VbfCache(NamedTuple):
    """A dataset's virtual-bright-field cache, as ``get_vbf`` keeps it."""

    vbf: Any
    threshold: float | None
    spare: tuple[float, Any] | None


def vbf_cache_of(dataset: Any) -> VbfCache:
    """Read ``dataset``'s vBF cache (unset attributes read as ``None``)."""
    return VbfCache(*(getattr(dataset, name, None) for name in VBF_ATTRS))


def isolated_view(
    data: Any,
    reuse: Mapping[str, Any],
    threshold: float,
    *,
    own_meta: bool = False,
) -> Any:
    """A shallow copy of ``data`` whose vBF cache (and optionally meta) is its own.

    ``reuse["vbf"]`` or ``reuse["vbf_cache"]`` seed the cache; otherwise it
    starts empty. ``own_meta`` gives the view a detached copy of ``data.meta``,
    for reconstructions that sweep the aberrations in place (the tcDF depth
    section) -- such a view must not be handed a cache whose providers share
    the original metadata.
    """
    vbf, cache = reuse.get("vbf"), reuse.get("vbf_cache")
    if vbf is not None and cache is not None:
        raise ValueError(
            "pass reuse={'vbf': ...} or reuse={'vbf_cache': ...}, not both"
        )
    view = copy.copy(data)
    if own_meta:
        # Before seeding: binding new metadata drops the view's vBF cache.
        view.meta = copy_meta(data.meta)
    if cache is None:
        cache = VbfCache(vbf, threshold if vbf is not None else None, None)
    for name, value in zip(VBF_ATTRS, cache, strict=True):
        setattr(view, name, value)
    return view


def cache_diagnostics(view: Any, reuse: Mapping[str, Any]) -> dict[str, Any]:
    """``{"vbf_cache": ...}`` when the caller asked to carry the cache through.

    Only then: a result holding providers keeps their bright-field stack (and
    its ~17 GiB ``G`` on a full scan) alive for as long as the result lives.
    """
    if "vbf_cache" in reuse:
        return {"vbf_cache": vbf_cache_of(view)}
    return {}
