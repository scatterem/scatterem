"""Find the view for an object: registered type first, then the protocol."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from scatterem.vis.show._context import View

_REGISTRY: dict[type | str, View] = {}


def register(
    target: type | str, *, name: str, frames: Callable | None = None
) -> Callable:
    """Register the panel function decorated by this as the view for ``target``.

    Parameters
    ----------
    target : type or str
        The type to show, or its qualified name ("package.module.Class") so the
        registering module need not import a heavy type. Subclasses use the view.
    name : str
        Short name used for file names (``"dataset4dstem"``).
    frames : callable, optional
        ``frames(obj, axis, ctx) -> FrameSource`` for ``show_animated``.

    Returns
    -------
    callable
        The decorator.
    """

    def decorator(fn: Callable) -> Callable:
        _REGISTRY[target] = View(name, fn, frames)
        return fn

    return decorator


def _protocol_view(obj: Any) -> View:
    name = getattr(obj, "__scatterem_name__", type(obj).__name__.lower())

    def panels(o, ctx):
        return o.__scatterem_figure__(ctx)

    frames: Callable[..., Any] | None = None
    if hasattr(obj, "__scatterem_frames__"):

        def _frames(o, axis, ctx):
            return o.__scatterem_frames__(axis, ctx)

        frames = _frames

    return View(name, panels, frames)


def resolve(obj: Any) -> View:
    """The view for ``obj``.

    Parameters
    ----------
    obj : Any
        The object to show.

    Returns
    -------
    View
        The registered view of its type (or a base type), else the protocol view.

    Raises
    ------
    TypeError
        If neither a registered type nor ``__scatterem_figure__`` fits.
    """
    for klass in type(obj).__mro__:
        if klass in _REGISTRY:
            return _REGISTRY[klass]
        qualified = f"{klass.__module__}.{klass.__qualname__}"
        if qualified in _REGISTRY:
            return _REGISTRY[qualified]
    if hasattr(obj, "__scatterem_figure__"):
        return _protocol_view(obj)
    raise TypeError(
        f"show() has no view for {type(obj).__name__}; registered views: "
        f"{', '.join(known_views())}; result classes provide __scatterem_figure__"
    )


def known_views() -> list[str]:
    """Names of the registered views, sorted."""
    return sorted(v.name for v in _REGISTRY.values())
