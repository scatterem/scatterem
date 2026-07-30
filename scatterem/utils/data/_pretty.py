"""Shared pretty-printing / introspection for the dataset family.

Most dataset classes historically had no working ``__repr__`` (several raised
on ``repr()``) and none rendered in Jupyter. ``PrettyDatasetMixin`` centralises
all of that behind a single cheap hook, ``_summary_rows``: subclasses return an
ordered ``dict`` of the fields worth showing and inherit a type-labelled REPL
repr, a Jupyter ``_repr_html_`` table, ``.info()`` and ``.summary()`` for free.

Keep ``_summary_rows`` cheap — no FFTs, no ``.max()`` / device syncs, no
decompression — it runs every time an object is echoed in a notebook.
"""

from __future__ import annotations

from html import escape
from typing import Any


class PrettyDatasetMixin:
    """Mixin providing repr / html / info / summary from ``_summary_rows``."""

    def _summary_rows(self) -> dict[str, Any]:
        """Ordered {label: value} pairs shown by the repr/html/info helpers.

        The safe default reports name/shape/dtype/device when available;
        concrete datasets override this with a richer, domain-specific set.
        """
        rows: dict[str, Any] = {}
        name = getattr(self, "name", None)
        if name is not None:
            rows["name"] = name
        for attr in ("shape", "dtype", "device"):
            try:
                rows[attr] = getattr(self, attr)
            except Exception:
                pass
        return rows

    def __repr__(self) -> str:
        inner = ", ".join(f"{k}={v!r}" for k, v in self._summary_rows().items())
        return f"{type(self).__name__}({inner})"

    def __str__(self) -> str:
        body = "\n".join(f"  {k}: {v}" for k, v in self._summary_rows().items())
        return f"{type(self).__name__}\n{body}"

    def _repr_html_(self) -> str:
        cls = escape(type(self).__name__)
        rows = "".join(
            "<tr>"
            f"<th style='text-align:left;padding-right:1em'>{escape(str(k))}</th>"
            f"<td style='text-align:left'>{escape(str(v))}</td>"
            "</tr>"
            for k, v in self._summary_rows().items()
        )
        return f"<b>{cls}</b><table>{rows}</table>"

    def info(self) -> None:
        """Print the human-readable multi-line summary."""
        print(str(self))

    def summary(self) -> dict[str, Any]:
        """Return the summary fields as a plain ``dict``."""
        return dict(self._summary_rows())
