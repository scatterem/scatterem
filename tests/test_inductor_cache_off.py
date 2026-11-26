"""The test session turns off the inductor FX graph and AOTAutograd caches.

``tests/conftest.py`` sets ``TORCHINDUCTOR_FX_GRAPH_CACHE=0`` and
``TORCHINDUCTOR_AUTOGRAD_CACHE=0``. The torch config modules read them only
once, at import. If a later import in the conftest loads a config module
before the ``setdefault``, the caches stay on and a stale graph from another
checkout can run against this checkout's custom ops. This test fails then.
"""

from __future__ import annotations

import os

import pytest


def test_inductor_caches_are_off_in_the_test_session() -> None:
    for name in ("TORCHINDUCTOR_FX_GRAPH_CACHE", "TORCHINDUCTOR_AUTOGRAD_CACHE"):
        if os.environ.get(name) != "0":
            pytest.skip(f"the caller set {name}={os.environ.get(name)!r}")
    import torch._functorch.config as functorch_config
    import torch._inductor.config as inductor_config

    assert inductor_config.fx_graph_cache is False
    assert functorch_config.enable_autograd_cache is False
