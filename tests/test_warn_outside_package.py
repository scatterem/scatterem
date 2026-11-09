"""``warn_outside_package`` points at the caller on every supported Python.

``warnings.warn(skip_file_prefixes=...)`` exists only from Python 3.12. On 3.10
and 3.11 the helper walks the stack instead, and it must give the same warning
location. Both branches run here on whatever interpreter runs the test.
"""

from __future__ import annotations

import inspect
import os
import warnings

import pytest

from scatterem import _warn

BRANCHES = [
    pytest.param(True, id="skip_file_prefixes"),
    pytest.param(False, id="stack_walk"),
]


def _use_branch(monkeypatch, has_skip: bool) -> None:
    if has_skip and not _warn._HAS_SKIP_FILE_PREFIXES:
        pytest.skip("skip_file_prefixes needs Python 3.12")
    monkeypatch.setattr(_warn, "_HAS_SKIP_FILE_PREFIXES", has_skip)


def _inner(prefix: str) -> int:
    line = inspect.currentframe().f_lineno + 1
    _warn.warn_outside_package("planted", UserWarning, prefix)
    return line


def _outer(prefix: str) -> int:
    return _inner(prefix)


@pytest.mark.parametrize("has_skip", BRANCHES)
def test_a_package_warning_lands_on_the_callers_line(monkeypatch, has_skip):
    _use_branch(monkeypatch, has_skip)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        line = _outer(_warn.PACKAGE_DIR)
    assert len(caught) == 1
    assert caught[0].filename == os.path.abspath(__file__)
    assert caught[0].lineno == line


@pytest.mark.parametrize("has_skip", BRANCHES)
def test_every_frame_under_the_prefix_is_skipped(monkeypatch, has_skip):
    """A prefix that covers this file too moves the warning out of it."""
    _use_branch(monkeypatch, has_skip)
    here = os.path.abspath(__file__)
    prefix = os.path.join(os.path.commonpath([_warn.__file__, here]), "")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _outer(prefix)
    assert len(caught) == 1
    assert not caught[0].filename.startswith(prefix)
