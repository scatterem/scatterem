"""A matplotlib Figure that knows whether a notebook has shown it."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import matplotlib as mpl
from matplotlib.figure import Figure

FORMATS = (".png", ".pdf", ".svg")


class ShownFigure(Figure):
    """The figure ``show`` returns.

    In Jupyter, ``show`` displays it at once and ``_ipython_display_`` then does
    nothing, so it appears exactly once. In marimo it renders through
    ``_display_`` when it is a cell's last expression (marimo's convention).

    Parameters
    ----------
    *args, **kwargs
        Passed to :class:`matplotlib.figure.Figure`.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.shown = False
        self.saved_path: Path | None = None
        self.style_rc: dict[Any, Any] = {}

    def to_png(self, dpi: float | None = None) -> bytes:
        """The figure as PNG bytes, rendered with its style.

        Parameters
        ----------
        dpi : float, optional
            Resolution; the figure default when None.

        Returns
        -------
        bytes
            The PNG file content.
        """
        buffer = io.BytesIO()
        with mpl.rc_context(self.style_rc):
            self.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
        return buffer.getvalue()

    def savefig(self, *args: Any, **kwargs: Any) -> Any:
        """Save as :meth:`matplotlib.figure.Figure.savefig`, with the figure's style.

        The style's rc values (for example Type-42 PDF fonts in paper style)
        apply, so a plain ``fig.savefig("x.pdf")`` matches :meth:`save`.

        Parameters
        ----------
        *args, **kwargs
            Passed to :meth:`matplotlib.figure.Figure.savefig`.

        Returns
        -------
        Any
            What :meth:`matplotlib.figure.Figure.savefig` returns.
        """
        with mpl.rc_context(self.style_rc):
            return super().savefig(*args, **kwargs)

    def save(self, path: str | Path) -> Path:
        """Write the figure as PNG, PDF or SVG (chosen by the suffix).

        Parameters
        ----------
        path : str or pathlib.Path
            Destination. Parent folders are created.

        Returns
        -------
        pathlib.Path
            The written path.

        Raises
        ------
        ValueError
            If the suffix is not .png, .pdf or .svg.
        """
        target = Path(path)
        if target.suffix.lower() not in FORMATS:
            raise ValueError(f"save= must end in one of {FORMATS}, not {target.name!r}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with mpl.rc_context(self.style_rc):
            self.savefig(target, bbox_inches="tight")
        return target

    def display_in_jupyter(self) -> None:
        """Display the figure in the running Jupyter kernel and mark it shown."""
        from IPython.display import Image, display

        display(Image(data=self.to_png()))
        self.shown = True

    def _ipython_display_(self) -> None:
        if not self.shown:
            self.display_in_jupyter()

    def _display_(self) -> Any:
        import marimo

        return marimo.image(self.to_png())
