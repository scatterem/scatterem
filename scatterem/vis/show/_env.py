"""Where a figure goes: notebook, window, or a file."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import matplotlib

from scatterem.vis.show._figure import ShownFigure

ENVIRONMENTS = ("marimo", "jupyter", "gui", "headless", "none")


def _checked(name: str, source: str) -> str:
    if name not in ENVIRONMENTS:
        raise ValueError(f"{source} must be one of {ENVIRONMENTS}, not {name!r}")
    return name


def _in_marimo() -> bool:
    if "marimo" not in sys.modules:
        return False
    import marimo

    return bool(getattr(marimo, "running_in_notebook", lambda: False)())


def _in_jupyter() -> bool:
    ipython = sys.modules.get("IPython")
    if ipython is None:
        return False
    shell = ipython.get_ipython()
    return shell is not None and hasattr(shell, "kernel")


def _gui_backend() -> bool:
    from matplotlib.backends.registry import BackendFilter, backend_registry

    interactive = {
        b.lower() for b in backend_registry.list_builtin(BackendFilter.INTERACTIVE)
    }
    return matplotlib.get_backend().lower() in interactive


def detect() -> str:
    """The environment a figure goes to.

    Returns
    -------
    str
        ``SCATTEREM_SHOW`` if set, else "marimo", "jupyter", "gui" or "headless".

    Raises
    ------
    ValueError
        If ``SCATTEREM_SHOW`` names no known environment.
    """
    forced = os.environ.get("SCATTEREM_SHOW")
    if forced:
        return _checked(forced, "SCATTEREM_SHOW")
    if _in_marimo():
        return "marimo"
    if _in_jupyter():
        return "jupyter"
    if _gui_backend():
        return "gui"
    return "headless"


def default_path(name: str, suffix: str) -> Path:
    """A new file in ``SCATTEREM_FIGURES_DIR`` (default ./figures) for ``name``.

    Parameters
    ----------
    name : str
        The view name.
    suffix : str
        File suffix, for example ".png".

    Returns
    -------
    pathlib.Path
        ``<dir>/<name>_<YYYYmmdd-HHMMSS>[_<n>]<suffix>``, not yet existing.
    """
    folder = Path(os.environ.get("SCATTEREM_FIGURES_DIR", "figures"))
    stem = f"{name}_{time.strftime('%Y%m%d-%H%M%S')}"
    path = folder / f"{stem}{suffix}"
    n = 1
    while path.exists():
        path = folder / f"{stem}_{n}{suffix}"
        n += 1
    return path


def announce(path: Path) -> None:
    """Tell a headless user where the file went."""
    print(f"scatterem.vis.show: wrote {path}", file=sys.stderr)  # noqa: T201 -- the path is the output


def route(
    fig: ShownFigure, env: str, *, save, name: str, block: bool = True
) -> ShownFigure:
    """Save and display ``fig`` as ``env`` needs.

    Parameters
    ----------
    fig : ShownFigure
        The composed figure.
    env : str
        One of ``ENVIRONMENTS``.
    save : str or pathlib.Path or None
        Also write the figure here.
    name : str
        View name for the headless file.
    block : bool, default=True
        Passed to ``plt.show`` in a GUI.

    Returns
    -------
    ShownFigure
        ``fig``, with ``saved_path`` set when a file was written.
    """
    _checked(env, "env")
    if save is not None:
        fig.saved_path = fig.save(save)
    if env == "jupyter":
        fig.display_in_jupyter()
    elif env == "gui":
        import matplotlib.pyplot as plt

        plt.show(block=block)
    elif env == "headless" and save is None:
        fig.saved_path = fig.save(default_path(name, ".png"))
        announce(fig.saved_path)
    return fig
