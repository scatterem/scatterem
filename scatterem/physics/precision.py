"""Precision configuration for PRISM scan-exit-wave computation.

Defines the three precision axes:
- ``build``   : dtype used for the S-matrix multislice build (fp32 or fp16).
- ``store``   : dtype used to store S_dt before projection (complex64 or complex32).
- ``project`` : dtype used inside the Warp projection kernel (fp32 or fp16).

Three named presets are provided:

``fp32``  (default)
    All axes in fp32/complex64 — bit-identical to the pre-Task-6 baseline.

``mixed``
    Build in fp32, store in complex32 (half the S-matrix memory), project in
    fp32 (upcast on the fly per chunk).  Expected <1% RMS vs fp32.

``fp16``
    Build in fp16, store in complex32, project in fp16 (true half-precision
    multiply in the Warp kernel; accumulate in fp32).  Expected <1% RMS.
"""

from __future__ import annotations

# Canonical preset table.  Each entry is {build, store, project}.
_PRESETS: dict[str, dict[str, str]] = {
    "fp32": {"build": "fp32", "store": "complex64", "project": "fp32"},
    "mixed": {"build": "fp32", "store": "complex32", "project": "fp32"},
    "fp16": {"build": "fp16", "store": "complex32", "project": "fp16"},
}


def resolve_precision(name: str | dict[str, str] | None) -> dict[str, str]:
    """Return a precision-config dict for the given preset name.

    Parameters
    ----------
    name
        One of ``"fp32"``, ``"mixed"``, ``"fp16"`` (case-sensitive), ``None``
        (equivalent to ``"fp32"``), or an explicit ``{build, store, project}``
        dict for per-axis override.

    Returns
    -------
    dict
        ``{"build": ..., "store": ..., "project": ...}`` — a plain dict copy
        so callers can safely mutate it.

    Raises
    ------
    ValueError
        If *name* is a string not in the preset table.
    """
    if name is None:
        return dict(_PRESETS["fp32"])

    if isinstance(name, dict):
        # Per-axis override: fill missing keys from fp32 baseline.
        base = dict(_PRESETS["fp32"])
        base.update(name)
        return base

    if name not in _PRESETS:
        raise ValueError(
            f"Unknown precision preset {name!r}. "
            f"Valid options: {list(_PRESETS)!r} or a dict with "
            "keys 'build', 'store', 'project'."
        )
    return dict(_PRESETS[name])
