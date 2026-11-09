"""Shared physics layer: primitives that nn/ and simulation/ both build on.

It may import only the standard library, numpy, torch, warp, itself, the
two foundation packages below it, ``scatterem.rotation`` and
``scatterem.crystal``, and the internal full-precision GEMM helper.
"""
