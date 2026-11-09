"""Package-level pytest configuration shared by the maintained ``tests/`` suite
and the inline ``scatterem/**/tests/`` suites.

This file sits at the ``packages/scatterem`` root so that it applies to both
configured ``testpaths`` (``tests`` and ``scatterem``) regardless of whether
pytest is invoked from the repository root or from this package directory.

It owns what must hold for BOTH suites: hardware lane separation (below), the
guard against import-time ``SCATTEREM_*`` mutation, and the float32 matmul
precision -- the fixtures that pin it and the guard that every test leaves it
as it found it. The coverage policy runs two lanes:

* a CPU lane, which is the enforced gate and must be green with no CUDA device
  visible; and
* a GPU lane, which additionally exercises the CUDA/Warp kernels.

Before this hook existed the ``gpu`` marker declared in ``pyproject.toml`` was
never enforced, and tests that parametrize over ``device`` simply raised
``RuntimeError: No CUDA GPUs are available`` when CUDA was hidden. That turned a
hardware-availability condition into 301 hard failures and made the CPU lane
impossible to gate on. A test that cannot run without a GPU must *skip* on a
CPU-only host, not fail.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, MutableMapping
from typing import Any

import pytest
import torch


# Tests never open a window. An interactive pyvista, matplotlib or Qt window
# waits for a person to close it, so the test hangs (one pyvista window held a
# pooled run for 5 hours) and the window takes focus on the desktop. Force the
# headless backends for the whole session, also when a plugin has already
# imported a library.
def _force_headless() -> None:
    import sys

    os.environ["MPLBACKEND"] = "Agg"
    os.environ["PYVISTA_OFF_SCREEN"] = "true"
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    if "matplotlib" in sys.modules:
        sys.modules["matplotlib"].use("Agg", force=True)
    if "pyvista" in sys.modules:
        sys.modules["pyvista"].OFF_SCREEN = True


_force_headless()

# torch bug (torch >= 2.10 on Linux x86-64 MKL builds, measured on 2.10-2.12,
# on Intel and AMD CPUs): after any
# ``torch.set_num_threads(n)`` with n > 1, batched CPU LU (``linalg.lu_factor``,
# ``linalg.solve``, ``linalg.inv``; B > 1, N >= ~150) gives bad pivots, hangs, or
# a wrong result. A later call does not undo it. About 70 test fixtures and
# helpers set one thread and then restore the saved count; the restore call
# poisoned every later test in the process
# (a batched-solve test failed in suite order). Start
# the session at one thread, so each restore sets 1 again.
# When torch is fixed, this line can go.
torch.set_num_threads(1)

# The pin holds only while nothing in the process calls set_num_threads(n > 1):
# the call itself does the damage, even if the caller restores 1 afterwards.
# So refuse such a call for the whole session, before it reaches torch.
_torch_set_num_threads = torch.set_num_threads
_SET_NUM_THREADS_REFUSED = (
    "torch.set_num_threads({n}) is refused in the scatterem test session: after "
    "any call with n > 1, torch >= 2.10 on Linux x86-64 MKL builds (measured on "
    "2.10-2.12) corrupts batched "
    "CPU LU (linalg.solve / inv / lu_factor) for the rest of the process. See "
    "tests/conftest.py. Run the multi-threaded code in a subprocess instead."
)


def _set_num_threads_guarded(n: int) -> None:
    """``torch.set_num_threads`` that refuses ``n > 1`` (torch batched-LU bug)."""
    if int(n) > 1:
        raise RuntimeError(_SET_NUM_THREADS_REFUSED.format(n=n))
    _torch_set_num_threads(n)


torch.set_num_threads = _set_num_threads_guarded

_CUDA_SKIP_REASON = "requires a CUDA-capable GPU (CPU lane skips the GPU lane's tests)"


def _cuda_available() -> bool:
    """Whether a usable CUDA device is present.

    ``torch.cuda.is_available()`` can raise on hosts with a broken or partially
    initialised driver rather than returning ``False``; treat any failure as
    "no GPU" so the CPU lane degrades to skips instead of collection errors.
    """
    try:
        return torch.cuda.is_available()
    except Exception:  # pragma: no cover - driver-dependent, not reachable in CI
        return False


def _mentions_cuda(value: Any) -> bool:
    """Whether a single parametrized value names a CUDA device.

    Handles the three spellings used across this repository: the plain string
    ``"cuda"``/``"cuda:0"``, a ``torch.device``, and ``torch.dtype``-style
    objects are explicitly *not* matched. Substring matching is deliberately
    avoided so an unrelated string that merely contains "cuda" (a file path, a
    test id) does not silently skip a CPU-capable test.
    """
    if isinstance(value, torch.device):
        return value.type == "cuda"
    if isinstance(value, str):
        head = value.split(":", 1)[0]
        return head == "cuda"
    return False


def _requires_cuda(item: pytest.Item) -> bool:
    """Whether ``item`` can only run on a CUDA host.

    Two independent signals are honoured:

    1. an explicit ``@pytest.mark.gpu`` marker, and
    2. a parametrized argument whose value names a CUDA device -- the common
       ``@pytest.mark.parametrize("device", ["cuda", "cpu"])`` shape, where only
       the ``cuda`` half needs a GPU.

    Only the CUDA half of such a parametrization is skipped; the CPU half still
    runs and still contributes coverage on the CPU lane.
    """
    if item.get_closest_marker("gpu") is not None:
        return True
    callspec = getattr(item, "callspec", None)
    if callspec is None:
        return False
    return any(_mentions_cuda(value) for value in callspec.params.values())


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip GPU-only tests when no CUDA device is visible."""
    if _cuda_available():
        return
    skip_marker = pytest.mark.skip(reason=_CUDA_SKIP_REASON)
    for item in items:
        if _requires_cuda(item):
            item.add_marker(skip_marker)


# --------------------------------------------------------------------------
# Guard against import-time mutation of SCATTEREM_* environment variables.
# --------------------------------------------------------------------------
#
# pytest imports every test module during collection. A module that does
# `os.environ.setdefault("SCATTEREM_USE_WARP", "1")` at module scope therefore
# changes that variable for the WHOLE session, silently switching unrelated
# tests onto a different code path. That is not hypothetical: it happened here,
# and it broke six `_forward_core_real` vs `_forward_core` parity assertions
# that passed in isolation and failed only in a full-suite run. Separately,
# three other tests were found to pass *only because of* the same leak.
#
# The variables are read with `os.environ.get` at call time, so a test that
# needs one should set it with a scoped `monkeypatch.setenv` fixture. This hook
# makes the module-scope form fail loudly at collection instead of silently
# corrupting results hours later.

_ENV_PREFIX = "SCATTEREM_"
_env_at_startup: dict[str, str] = {}


def pytest_configure(config: pytest.Config) -> None:
    """Snapshot SCATTEREM_* before any test module is imported."""
    _env_at_startup.clear()
    _env_at_startup.update(
        {k: v for k, v in os.environ.items() if k.startswith(_ENV_PREFIX)}
    )
    _cap_worker_threads(config)


# --------------------------------------------------------------------------- #
# BLAS / OpenMP threads per xdist worker
# --------------------------------------------------------------------------- #
# torch, MKL and OpenBLAS each default to one thread per core, so `pytest -n N`
# runs N x cores threads. On a 20-core host `-n auto` meant 400 threads; one
# test (an ADF detector test) took > 56 min that way and 75 s with one
# thread per worker. CI sets these variables in every workflow, so this only
# fills in what a local run left unset (setdefault: an explicit value wins).
# It runs in the xdist CONTROLLER, before the workers are spawned, so each
# worker starts with the variables already in its environment: the libraries
# read them once at load, and plugins (pytest-randomly imports numpy) load
# before any conftest does in the worker. A serial run is left alone. torch
# itself runs at one thread in every run (serial or xdist): see the pin at the
# top of this file. The share here still applies to numpy / OpenBLAS.

_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def _usable_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # not on Linux
        return os.cpu_count() or 1


def _threads_per_worker(n_workers: int, n_cpus: int) -> int:
    return max(1, n_cpus // max(1, n_workers))


def _cap_worker_threads(
    config: pytest.Config, environ: MutableMapping[str, str] = os.environ
) -> int | None:
    """Give each xdist worker a share of the cores; return the share, or None."""
    if hasattr(config, "workerinput"):  # a worker: inherited from the controller
        return None
    workers = len(getattr(config.option, "tx", None) or [])
    if not workers or getattr(config.option, "dist", "no") == "no":
        return None
    # torch takes MKL_NUM_THREADS over OMP_NUM_THREADS, so a caller who set only
    # one must see it copied to the others, not overridden by our share.
    explicit = [environ[v] for v in _THREAD_VARS if environ.get(v, "").isdigit()]
    if explicit:
        threads = min(int(v) for v in explicit)
    else:
        threads = _threads_per_worker(workers, _usable_cpus())
    for var in _THREAD_VARS:
        environ.setdefault(var, str(threads))
    return threads


def pytest_collection_finish(session: pytest.Session) -> None:
    """Fail if importing the test modules changed any SCATTEREM_* variable."""
    after = {k: v for k, v in os.environ.items() if k.startswith(_ENV_PREFIX)}
    changed = {
        key: (_env_at_startup.get(key), after.get(key))
        for key in set(_env_at_startup) | set(after)
        if _env_at_startup.get(key) != after.get(key)
    }
    if changed:
        detail = ", ".join(
            f"{k}: {before!r} -> {now!r}"
            for k, (before, now) in sorted(changed.items())
        )
        raise pytest.UsageError(
            "Importing the test modules mutated the environment at module scope, "
            "which leaks into every other test in the session and silently changes "
            f"which code path they take: {detail}. Set the variable inside a "
            "scoped fixture (monkeypatch.setenv) instead of at module level."
        )


@pytest.fixture(scope="session")
def cuda_available() -> bool:
    """Session-scoped CUDA availability, for tests that branch on it directly."""
    return _cuda_available()


@pytest.fixture
def require_cuda() -> Iterator[torch.device]:
    """Yield a CUDA device, skipping the test when none is available.

    Preferred over an inline ``pytest.mark.skipif(not torch.cuda.is_available())``
    because availability is evaluated at call time rather than import time.
    """
    if not _cuda_available():
        pytest.skip(_CUDA_SKIP_REASON)
    yield torch.device("cuda")


# --------------------------------------------------------------------------
# The float32 matmul precision: pinned through ONE API, and left as found.
# --------------------------------------------------------------------------
#
# torch 2.11 keeps three records of it -- the legacy value, and the
# ``fp32_precision`` of ``backends.cuda.matmul`` and of ``backends.mkldnn.matmul``
# -- and its setters do not write the same subset.
# ``torch.set_float32_matmul_precision`` writes all three; scatterem calls it with
# ``"medium"`` (legacy MEDIUM, cuda tf32, mkldnn bf16) when
# ``scatterem.configure()`` is called. (Importing
# scatterem did it for the whole process until 2026-09; it sets nothing now, so a
# session starts at torch's default, ``"highest"`` with both records ``"none"``.)
# Writing the legacy FLAG, ``torch.backends.cuda.matmul.allow_tf32``, rewrites the
# first two and leaves mkldnn as it was, and torch accepts the result only
# where the records still agree. After ``"medium"`` (either value of the flag)
# or ``"high"`` (``False``), ``torch.get_float32_matmul_precision()`` RAISES, in
# that process, for good; at torch's default the flag happens to be harmless.
# Writing the flag back does not cure it (True means HIGH, not MEDIUM).
#
# Six inline test modules pinned true fp32 that way. Whatever ran after one of
# them in the same worker and read the precision failed with torch's "mix of
# the legacy and new APIs" error: the users of ``fp32_matmul_precision`` -- 17
# failures in one GPU run, none naming the module that caused them, all passing
# alone. Three fixtures had learned to restore a hardcoded ``"medium"`` instead
# of the value they found, because reading it back raised.
#
# So: the two fixtures below are the only way a test pins the precision (the
# flag is banned by Ruff, TID251), and the autouse guard checks the invariant
# on every test of both suites and fails the test that breaks it. Both put all
# three records back exactly, not merely the value the getter reports.


def _matmul_records() -> tuple[str | None, str | None]:
    """The two backend records of the float32 matmul precision.

    torch before 2.9 has neither record; each then reads as ``None``.
    """
    from scatterem._configure import _MKLDNN_MATMUL, _fp32_record

    return (
        _fp32_record(torch.backends.cuda.matmul),
        _fp32_record(_MKLDNN_MATMUL),
    )


def _restore_matmul(precision: str, records: tuple[str | None, str | None]) -> None:
    """Put the precision and both backend records back exactly."""
    from scatterem._configure import _restore_backend_records

    torch.set_float32_matmul_precision(precision)  # writes all three records
    _restore_backend_records(*records)


@pytest.fixture
def fp32_matmul_precision() -> Iterator[None]:
    """Pin true fp32 (TF32 off) matmul for kernel-vs-reference gradient checks.

    Under float32 matmul precision ``"medium"`` or ``"high"`` -- inside a
    process, after ``scatterem.configure()``, or wherever a
    caller set it -- ``torch.bmm`` (and its backward) run in TF32 on Ampere+
    GPUs. That injects ~1e-3 relative error into the *reference* gradients built
    from ``torch.bmm`` / ``grid_sample`` autograd, while the Warp kernels compute
    full-fp32 scalar FMA. A kernel-vs-reference equivalence check at a ~1e-5
    tolerance then fails on the reference's TF32 error, not on any kernel bug
    (the Warp gradient matches a float64 finite-difference reference to ~1e-6).
    Tests comparing a Warp kernel against a torch reference request this fixture
    so the reference is accurate whatever the session's precision is, and the
    tight tolerance stays meaningful.

    A ``unittest.TestCase`` takes it with
    ``@pytest.mark.usefixtures("fp32_matmul_precision")`` on the class.
    """
    prev = torch.get_float32_matmul_precision()
    records = _matmul_records()
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        _restore_matmul(prev, records)


@pytest.fixture
def ieee_float32(fp32_matmul_precision: None) -> Iterator[None]:
    """:func:`fp32_matmul_precision`, and no TF32 in cuDNN either.

    The cuDNN flag is self-consistent (it writes both of ITS records, and
    writing the saved value back is an exact round trip), so it is used as is.
    """
    prev = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32 = prev


def _matmul_precision() -> str | None:
    """The float32 matmul precision, or ``None`` where torch refuses to say."""
    try:
        return torch.get_float32_matmul_precision()
    except RuntimeError:
        return None


_UNREADABLE = (
    "torch.get_float32_matmul_precision() raises: torch's records of the float32 "
    "matmul precision disagree. The usual cause is a write to the legacy flag "
    "torch.backends.cuda.matmul.allow_tf32 while another precision was set "
    '(e.g. "medium" inside a scatterem fit or after scatterem.configure()). Pin '
    "true fp32 with the `fp32_matmul_precision` / `ieee_float32` fixtures (see "
    "tests/conftest.py) instead."
)


@pytest.fixture(autouse=True)
def matmul_precision_is_left_as_found() -> Iterator[None]:
    """Fail the test that leaves the float32 matmul precision unreadable or changed.

    The failure lands on the test that caused it, and torch's records are put
    back exactly afterwards, so one offender costs one test instead of every
    later reader in the worker. A change of the records alone that leaves the
    precision the same (``"none"`` -> ``"ieee"`` under ``"highest"``, which is
    what writing the legacy flag ``False`` at torch's default does) is repaired
    without a failure: it changes no numerics and no later reader.
    """
    before = _matmul_precision()
    if before is None:
        pytest.fail(f"before this test ran, {_UNREADABLE}", pytrace=False)
    records = _matmul_records()
    yield
    after = _matmul_precision()
    if after != before or _matmul_records() != records:
        _restore_matmul(before, records)
    if after != before:
        changed = f"this test left it at {after!r} (it was {before!r})"
        pytest.fail(_UNREADABLE if after is None else changed, pytrace=False)


