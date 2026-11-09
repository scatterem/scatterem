"""Launch-geometry helpers for hand-tuned Warp kernels.

WHY THIS EXISTS
---------------
A ``block_dim`` larger than a kernel's register budget allows is a **silent**
failure in Warp.  The driver refuses the launch with ``CUDA error 701: too many
resources requested for launch``, Warp prints that to stderr and **does not
raise**, and the output buffers come back holding whatever they were allocated
with -- zeros, for a freshly allocated gradient.  In a sweep of one backward
kernel on an H200 (sm_90), every ``block_dim`` up to 896 gave the same
gradient and 897 gave zeros. ``cuFuncGetAttribute(MAX_THREADS_PER_BLOCK)``
reports **896** for that kernel -- i.e. the driver's own per-kernel ceiling is
*exactly* the boundary, so it is the right resource to key on rather than a
constant fitted on one GPU.

That ceiling is not a property of the device.  It varies per kernel, per
overload and per direction, so a tuned constant is right only for the cards it
was swept on; this module is the clamp that keeps it right anywhere else.

WHAT NOT TO USE IT ON
---------------------
A ``block_dim`` that is a **semantic** requirement rather than a tuning choice
must not be clamped.  ``nn/functional/ptychography/`` launches two kernels at
``block_dim=256`` because the cooperative ``wp.tile`` / ``wp.tile_atomic_add``
reductions inside them are written for that width; lowering it there would
quietly change the reduction instead of quietly skipping the launch, which is a
worse failure, not a better one.  Those two sites are deliberately left as plain
``block_dim=256``.

WHAT TO USE
-----------
:func:`ensure_warp_initialized` initialises the Warp runtime once. Nothing in
the library calls it at import: ``import scatterem`` and its subpackages leave
Warp unstarted, because starting Warp starts a thread, after which a process
cannot fork safely. Six Warp APIs do not start Warp themselves --
``wp.from_torch``, ``wp.device_from_torch``, ``wp.stream_from_torch``,
``wp.ScopedStream``, ``wp.Stream``, ``wp.synchronize_device`` -- so the
library calls them through ``scatterem.physics._warp_interop`` (imported as
``wi``), whose first call starts Warp. No module calls them on ``wp``
directly.

:func:`block_dim_kwargs` returns the ``block_dim=`` keyword for ``wp.launch``,
clamped to what the kernel can actually run, or an empty dict when the tuned
value cannot be honoured and Warp's default should stand.  It never raises and
never *raises* the requested value -- a tuned block size is only ever lowered.
"""

from __future__ import annotations

import ctypes
import warnings

#: ``CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK`` from ``cuda.h``.
_CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0

#: ``(kernel.adj, device alias, adjoint) -> ceiling or None``.  The query costs a
#: module load and a driver call; the launches that want it run in inner loops.
_CEILING: dict = {}

#: Warned-about (kernel, device, requested) triples, so a clamp is reported ONCE
#: rather than once per launch.
_WARNED: set = set()

#: Fully-resolved ``block_dim`` kwargs, keyed by the whole call.  Keyed on the
#: device's ALIAS and not on ``id(device)``: Warp does intern its ``Device``
#: objects (verified), so the id would be stable in practice -- but a recycled id
#: would hand back another card's ceiling silently, and silence is the exact
#: failure this module exists to remove.  The alias costs ~30 ns more.
_KWARGS: dict = {}

_libcuda = None
_libcuda_tried = False


def ensure_warp_initialized() -> None:
    """Initialise the Warp runtime if nothing in the process has yet.

    Idempotent and cheap after the first call (``wp.init`` returns at once).
    The first call costs ~50-60 ms, or ~0.4 s on a GPU host whose CUDA driver
    nothing has started yet. It leaves no CUDA context behind and, unlike a
    bare ``wp.init()``, creates none on the way: Warp's start-up banner, which
    would retain a context on every GPU of the node to read its memory size, is
    skipped (``scatterem.physics._warp_init``; checked on Warp 1.17). Warp's own ``wp.launch``,
    ``wp.array`` and ``wp.get_device`` initialise it on demand;
    ``wp.from_torch`` and the other torch-interop and stream calls listed in
    this module's docstring do not, and fail with ``AttributeError:
    'NoneType' object has no attribute ...`` without it.
    """
    from scatterem.physics._warp_init import (  # noqa: PLC0415 -- imports Warp
        ensure_warp_initialized as _init,
    )

    _init()


def _driver():
    """The CUDA driver library, or ``None`` if it cannot be loaded."""
    global _libcuda, _libcuda_tried
    if not _libcuda_tried:
        _libcuda_tried = True
        for name in ("libcuda.so.1", "libcuda.so", "nvcuda.dll"):
            try:
                _libcuda = ctypes.CDLL(name)
                break
            except OSError:
                continue
    return _libcuda


def max_block_dim(kernel, device, *, adjoint: bool = False) -> int | None:
    """Largest ``block_dim`` ``kernel`` can be launched at on ``device``.

    ``adjoint`` selects the backward function, which is what a
    ``wp.launch(..., adjoint=True)`` actually runs -- and which has a *different*
    (usually lower) ceiling than the forward, so asking about the wrong one is
    how a clamp would let the silent failure through anyway.

    Returns ``None`` when the ceiling cannot be determined -- a CPU device, a
    Warp whose private layout moved, a driver that will not answer.  Callers
    treat that as "do not pass block_dim" rather than as "any value is fine".
    """
    dev_alias = getattr(device, "alias", str(device))
    if not getattr(device, "is_cuda", False):
        return None
    key = (id(kernel.adj), dev_alias, bool(adjoint))
    if key in _CEILING:
        return _CEILING[key]
    ceiling = None
    lib = _driver()
    if lib is not None:
        try:
            hooks = kernel.module.load(device).get_kernel_hooks(kernel)
            handle = hooks.backward if adjoint else hooks.forward
            if handle:
                value = ctypes.c_int(0)
                rc = lib.cuFuncGetAttribute(
                    ctypes.byref(value),
                    ctypes.c_int(_CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK),
                    ctypes.c_void_p(handle),
                )
                if rc == 0 and value.value > 0:
                    ceiling = int(value.value)
        except Exception:  # pragma: no cover - any Warp/driver shape change
            ceiling = None
    _CEILING[key] = ceiling
    return ceiling


def block_dim_kwargs(kernel, device, requested, *, adjoint: bool = False) -> dict:
    """``{"block_dim": n}`` for ``wp.launch``, clamped to what will actually run.

    ``requested`` is the swept constant.  It is honoured when the kernel can take
    it, lowered to the kernel's ceiling when it cannot, and dropped entirely
    (leaving Warp's default) when ``requested`` is ``None`` or the ceiling is
    unknown.  A clamp warns once per (kernel, device, requested) so that a tuned
    value silently not applying is visible rather than merely safe.

    The answer is memoised on the whole call, not just on the ceiling: these
    launches sit on host-issue-bound rows where one ``wp.launch`` is ~5 us, so
    the steady-state cost here is one tuple build and one dict lookup (~70 ns)
    rather than a re-derivation.  The returned dict is shared, which is safe
    because every caller consumes it with ``**`` (a copy) and never mutates it.
    """
    key = (id(kernel.adj), getattr(device, "alias", None), requested, adjoint)
    hit = _KWARGS.get(key)
    if hit is not None:
        return hit
    return _resolve_kwargs(key, kernel, device, requested, adjoint)


def _resolve_kwargs(key, kernel, device, requested, adjoint) -> dict:
    """The cold path of :func:`block_dim_kwargs`; see it for the contract."""
    if requested is None:
        out: dict = {}
    else:
        requested = int(requested)
        ceiling = max_block_dim(kernel, device, adjoint=adjoint)
        if ceiling is None:
            out = {}
        elif requested <= ceiling:
            out = {"block_dim": requested}
        else:
            out = {"block_dim": ceiling}
            warn_key = (kernel.key, getattr(device, "alias", str(device)), requested)
            if warn_key not in _WARNED:
                _WARNED.add(warn_key)
                warnings.warn(
                    f"{kernel.key}: block_dim {requested} exceeds this kernel's "
                    f"{ceiling}-thread limit on {warn_key[1]}; launching at "
                    f"{ceiling}. The tuned value was swept on another device -- "
                    f"the result is correct, the speed may not be what that "
                    f"sweep recorded.",
                    RuntimeWarning,
                    stacklevel=3,
                )
    _KWARGS[key] = out
    return out
