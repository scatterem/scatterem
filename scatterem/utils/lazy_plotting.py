"""Plotting modules that are imported on first use.

The direct-reconstruction modules draw diagnostic plots only when ``verbosity``
asks for them, but used to import :mod:`matplotlib.pyplot` and
:mod:`scatterem.vis` at module level -- so every reconstruction, including a
headless ``verbosity=0`` batch run, loaded a plotting stack it never used.
They now take ``plt`` and ``vis`` from here instead::

    from scatterem.utils.lazy_plotting import plt, vis

Every ``plt.<name>`` / ``vis.<name>`` call site stays exactly as written; the
real module is imported by the first such call, which only a plotting branch
makes, and the plots are the ones the real modules draw.

Unlike :class:`importlib.util.LazyLoader`, nothing is registered in
:data:`sys.modules` early: ``import matplotlib.pyplot`` elsewhere behaves
exactly as before.
"""

from __future__ import annotations

import importlib
import types
from typing import Any

__all__ = ["lazy_module", "plt", "vis"]


class _LazyModule(types.ModuleType):
    """A stand-in for a module that is imported on first attribute access."""

    def __getattr__(self, attr: str) -> Any:
        """Import the real module (a no-op after the first time), then read it.

        Parameters
        ----------
        attr : str
            Attribute requested on the stand-in.

        Returns
        -------
        Any
            The attribute of the real module.
        """
        return getattr(importlib.import_module(self.__name__), attr)


def lazy_module(name: str) -> types.ModuleType:
    """Return a reference to module ``name`` that imports it when first used.

    Parameters
    ----------
    name : str
        Absolute dotted module name, e.g. ``"matplotlib.pyplot"``.

    Returns
    -------
    types.ModuleType
        A stand-in whose attribute reads are served by the real module, which
        is imported on the first such read. The stand-in itself is not the
        module: compare the attributes it serves, not the object.

    Examples
    --------
    >>> json_ref = lazy_module("json")
    >>> json_ref.dumps([1, 2])
    '[1, 2]'
    """
    return _LazyModule(name)


plt = lazy_module("matplotlib.pyplot")
vis = lazy_module("scatterem.vis")
