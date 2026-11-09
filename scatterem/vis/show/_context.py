"""What a view receives (FigureContext) and what it returns (Panel)."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from matplotlib.axes import Axes

STYLES = ("screen", "paper")
WIDTHS = (None, "single", "double")


@dataclass(frozen=True)
class FigureContext:
    """Options that every view and panel sees.

    Parameters
    ----------
    style : {"screen", "paper"}
        Screen keeps titles and larger text; paper is Nature style.
    width : {None, "single", "double"}
        Paper width: 89 mm (single, the default) or 183 mm (double).
    background : {"light", "dark"}
        Figure background, used by the colour policy.
    panels : tuple of str, optional
        Panel names to draw, in this order. None draws the default set.
    opts : mapping
        View-specific options (for example ``point_group``).
    """

    style: str = "screen"
    width: str | None = None
    background: str = "light"
    panels: tuple[str, ...] | None = None
    opts: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.style not in STYLES:
            raise ValueError(f"style must be one of {STYLES}, not {self.style!r}")
        if self.width not in WIDTHS:
            raise ValueError(f"width must be one of {WIDTHS}, not {self.width!r}")

    def wants(self, name: str) -> bool:
        """Whether the panel ``name`` is part of this figure.

        Parameters
        ----------
        name : str
            Panel name.

        Returns
        -------
        bool
            True when ``panels`` is None or lists ``name``.
        """
        return self.panels is None or name in self.panels


@dataclass(frozen=True)
class Panel:
    """One panel of a figure: a name, a drawing function and a screen title.

    Parameters
    ----------
    name : str
        Panel name, used by ``panels=``.
    draw : callable
        ``draw(ax, ctx)`` draws the panel into ``ax``.
    title : str, default=""
        Title shown in screen style.
    """

    name: str
    draw: Callable[[Axes, FigureContext], None]
    title: str = ""


@dataclass(frozen=True)
class View:
    """How to show one kind of object: its name, panels and optional frames."""

    name: str
    panels: Callable[[Any, FigureContext], Sequence[Panel]]
    frames: Callable[[Any, str | None, FigureContext], Any] | None = None
