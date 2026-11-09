"""Warnings that point at the user's call, not at a frame inside scatterem.

``warnings.warn(..., skip_file_prefixes=...)`` does this, but the keyword exists
only from Python 3.12, and scatterem supports 3.10. :func:`warn_outside_package`
passes the keyword on 3.12 and later. On older versions it walks the stack and
computes the ``stacklevel`` of the first frame outside the package.
"""

from __future__ import annotations

import os
import sys
import warnings

__all__ = ["PACKAGE_DIR", "warn_outside_package"]

#: The directory of the ``scatterem`` package, with a trailing separator.
PACKAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "")

#: True when ``warnings.warn`` takes ``skip_file_prefixes`` (Python 3.12+).
_HAS_SKIP_FILE_PREFIXES = sys.version_info >= (3, 12)


def warn_outside_package(
    message: str, category: type[Warning], prefix: str = PACKAGE_DIR
) -> None:
    """Issue ``message`` at the first stack frame whose file is outside ``prefix``.

    Parameters
    ----------
    message
        The warning text.
    category
        The warning class.
    prefix
        Frames whose file name starts with this prefix are skipped. The default
        is the ``scatterem`` package directory.
    """
    if _HAS_SKIP_FILE_PREFIXES:
        warnings.warn(message, category, skip_file_prefixes=(prefix,))
        return
    # stacklevel=1 is this function. Count the frames inside the package from
    # here outward; the first frame outside it gets the warning.
    level = 1
    frame = sys._getframe(0)
    while frame is not None and frame.f_code.co_filename.startswith(prefix):
        frame = frame.f_back
        level += 1
    warnings.warn(message, category, stacklevel=level)
