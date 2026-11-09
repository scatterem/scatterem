"""Process-wide settings scatterem uses, set only when a caller asks for them.

Importing scatterem changes no global state. Until 2026-09 it set
``torch.set_float32_matmul_precision("medium")`` at import, which reached every
matmul of the importing program, scatterem's or not. That setting is now made
here, explicitly, by the caller.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import torch

_PRECISIONS = ("highest", "high", "medium")

# torch attaches this at runtime; its stubs do not declare it. torch before 2.9
# has neither it nor ``backends.cuda.matmul.fp32_precision``: there the legacy
# precision is the only record, and the two backend records read as ``None``.
_MKLDNN_MATMUL = getattr(torch.backends.mkldnn, "matmul", None)


def _fp32_record(backend) -> str | None:
    """``backend.fp32_precision`` as a string, or ``None`` where torch lacks it."""
    if backend is None:
        return None
    try:
        return str(backend.fp32_precision)
    except AttributeError:
        return None


def _restore_backend_records(cuda: str | None, mkldnn: str | None) -> None:
    """Write back the two backend records that :func:`_fp32_record` read."""
    if cuda is not None:
        torch.backends.cuda.matmul.fp32_precision = cuda
    if mkldnn is not None and _MKLDNN_MATMUL is not None:
        _MKLDNN_MATMUL.fp32_precision = mkldnn


def _matmul_precision() -> str | None:
    """The float32 matmul precision, or ``None`` where torch cannot report it.

    torch raises when its records of the precision disagree, which is what
    writing the legacy flag ``torch.backends.cuda.matmul.allow_tf32`` after
    another precision was set leaves behind.
    """
    try:
        return torch.get_float32_matmul_precision()
    except RuntimeError:
        return None


def _matmul_state() -> tuple[str | None, str | None, str | None]:
    """Every record torch keeps of the float32 matmul precision, for an exact undo.

    ``torch.set_float32_matmul_precision`` writes the legacy value and the
    ``fp32_precision`` of ``backends.cuda.matmul`` and ``backends.mkldnn.matmul``.
    A fresh process holds ``"none"`` (inherit) in the last two, which setting
    ``"highest"`` back would turn into ``"ieee"``: the same numerics, other
    records.
    """
    return (
        _matmul_precision(),
        _fp32_record(torch.backends.cuda.matmul),
        _fp32_record(_MKLDNN_MATMUL),
    )


def _restore_matmul_state(state: tuple[str | None, str | None, str | None]) -> None:
    """Put back what :func:`_matmul_state` recorded.

    A state recorded while torch could not report the precision is not put
    back: that would recreate the inconsistency, not a setting.
    """
    precision, cuda, mkldnn = state
    if precision is None:
        return
    torch.set_float32_matmul_precision(precision)
    _restore_backend_records(cuda, mkldnn)


def _restore_records(state: tuple[str | None, str | None, str | None]) -> None:
    """Put back the records of a scoped switch's entry state, as far as torch can.

    A state torch could report is restored exactly
    (:func:`_restore_matmul_state`). One it could not (the legacy
    ``allow_tf32`` flag was written, or an enclosing :func:`cuda_matmul_tf32`
    holds the CUDA record) gets its two backend records back, which carry the
    numerics; writing the legacy record would end the enclosing state.
    """
    if state[0] is None:
        _restore_backend_records(state[1], state[2])
        return
    _restore_matmul_state(state)


@contextlib.contextmanager
def cuda_matmul_tf32(enabled: bool) -> Iterator[None]:
    """Allow or forbid TF32 in CUDA float32 matmuls for the block, then restore.

    Writes ``torch.backends.cuda.matmul.fp32_precision`` (``"tf32"`` or
    ``"ieee"``) and nothing else: the oneDNN CPU matmul record keeps its value,
    so every CPU result is unchanged. This is what the legacy flag
    ``torch.backends.cuda.matmul.allow_tf32`` changed for CUDA, without its
    other records (:func:`configure`, Notes). The records torch keeps
    (:func:`_matmul_state`) are put back on exit, also on an exception: all
    three, or the two backend records when the block was entered with records
    torch could not report (:func:`_restore_records`).

    Parameters
    ----------
    enabled : bool
        ``True`` lets CUDA float32 matmuls round their inputs to TF32;
        ``False`` keeps them at IEEE float32.

    Yields
    ------
    None

    Notes
    -----
    Process-wide, like the setting: do not use it from concurrent threads.
    Inside the block :func:`torch.get_float32_matmul_precision` may raise,
    because the CUDA record then disagrees with the legacy one; do not start a
    training fit inside it.
    """
    state = _matmul_state()
    torch.backends.cuda.matmul.fp32_precision = "tf32" if enabled else "ieee"
    try:
        yield
    finally:
        _restore_records(state)


def configure(*, matmul_precision: str | None = "medium") -> dict[str, str | None]:
    """Set the process-wide numeric settings scatterem is tuned for.

    Nothing is set at ``import scatterem``. Direct reconstructions,
    simulations and I/O run at whatever the process has -- torch's default
    ``"highest"`` unless the caller changes it. Call this to run them at TF32
    speed.

    Parameters
    ----------
    matmul_precision : {"highest", "high", "medium"} or None, default "medium"
        Passed to :func:`torch.set_float32_matmul_precision`. ``"medium"`` and
        ``"high"`` let float32 matmuls and convolutions run in TF32 on Ampere and
        later GPUs (and, for ``"medium"``, in bfloat16 through oneDNN on CPUs
        that support it): faster, with about three significant digits instead of
        seven. ``"highest"`` is true float32. ``None`` leaves the setting
        untouched.

    Returns
    -------
    dict of str to str or None
        The value every setting had before this call, keyed by parameter name,
        so ``configure(**previous)`` restores them. A value is ``None`` when
        torch could not report it (see Notes); passing that back sets nothing.

    Raises
    ------
    ValueError
        If ``matmul_precision`` is not one of the three torch knows (torch
        itself only warns, and sets nothing).

    Notes
    -----
    Change the precision with this function or
    :func:`torch.set_float32_matmul_precision`, never with the legacy flag
    ``torch.backends.cuda.matmul.allow_tf32``: torch (2.11) keeps three records
    of the precision, the flag rewrites two of them, and next to ``"medium"`` the
    third then makes ``torch.get_float32_matmul_precision()`` raise for the rest
    of the process -- which some training frameworks call on every CUDA fit. A call
    here with a precision rewrites all three and repairs that state.

    Examples
    --------
    >>> import torch
    >>> import scatterem
    >>> previous = scatterem.configure(matmul_precision="high")
    >>> torch.get_float32_matmul_precision()
    'high'
    >>> _ = scatterem.configure(**previous)
    """
    if matmul_precision is not None and matmul_precision not in _PRECISIONS:
        raise ValueError(
            f"matmul_precision must be one of {_PRECISIONS} or None, "
            f"not {matmul_precision!r}"
        )
    previous = {"matmul_precision": _matmul_precision()}
    if matmul_precision is not None:
        torch.set_float32_matmul_precision(matmul_precision)
    return previous
