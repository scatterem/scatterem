"""NVIDIA Warp kernels.

This package re-exports nothing: each consumer imports the module it needs
directly.

``warp.init()`` is called once by ``scatterem/nn/__init__.py``, which runs
before any module of this package, so it is not repeated here.
"""
