"""Forward-mode autograd checks for kernels without a forward-mode derivative.

Fused kernels (Triton, Warp) and some torch ops (``grid_sampler_3d``) have no
forward-mode derivative. A fused kernel that is not a differentiable torch op
drops the tangent of a dual operand without an error; a torch op without a
formula raises. Callers check :func:`carries_tangent` and take a path that
keeps the tangent, or call :func:`refuse_forward_ad` to fail loudly.

The Warp custom ops are the second case. A ``torch.library.custom_op`` with
only ``register_autograd`` has no forward-mode derivative: ``torch.library``
has no forward-AD registration, and a forward mode needs an
``autograd.Function`` wrapper with ``jvp`` around the op. The generated
autograd kernel records the op only when an operand ``requires_grad``. A dual
operand (``torch.func.jvp``, or ``torch.autograd.forward_ad``) does not, so
the op runs below autograd and its output has no tangent: ``torch.func.jvp``
returns a ZERO tangent and no error. The public wrapper of each such op calls
:func:`refuse_forward_ad` first.

It lives in ``physics`` because that is the lowest layer, so every package may
import it.
"""

import torch.autograd.forward_ad as fwad
from torch import Tensor

__all__ = ["carries_tangent", "refuse_forward_ad"]


def _no_dual_level() -> bool:
    """Whether no forward-AD dual level is open (no tensor can carry a tangent).

    ``fwad.unpack_dual`` makes the same test on ``fwad._current_level`` before
    it does anything else. Reading it once here, instead of once per operand,
    makes a plain call of a guarded op cost one module-attribute read.
    """
    return fwad._current_level < 0  # torch's own test


#: The one error that ``fwad.unpack_dual`` raises for a ``vmap``-batched operand.
_VMAP_UNREADABLE = "Batching rule not implemented for aten::_unpack_dual"


def _read_tangents(tensors: tuple[Tensor, ...]) -> tuple[bool, RuntimeError | None]:
    """Whether an operand has a tangent, and the ``vmap`` error that hid one.

    Call only inside a dual level. Returns ``(True, err)`` when a
    ``vmap``-batched operand cannot be read (``err`` is torch's batching
    error). Any other error of ``fwad.unpack_dual`` propagates.
    """
    try:
        return any(fwad.unpack_dual(t).tangent is not None for t in tensors), None
    except RuntimeError as err:
        if _VMAP_UNREADABLE not in str(err):
            raise
        return True, err


def carries_tangent(*tensors: Tensor) -> bool:
    """Whether a tensor is a forward-mode autograd dual at the current level.

    Inside any open dual level (``torch.func.jvp``, or a plain
    ``torch.autograd.forward_ad.dual_level``), a tensor batched by ``vmap``
    cannot be read: torch has no batching rule for ``_unpack_dual``. The
    answer is then ``True``, also when no operand has a tangent. A caller then
    takes the path that keeps a tangent. That path is correct for a plain
    operand too, so a false ``True`` costs speed in forward mode only, not
    correctness.

    Parameters
    ----------
    *tensors : torch.Tensor
        The operands.

    Returns
    -------
    bool
        ``True`` if any operand has a tangent, or if a tangent cannot be
        read (a ``vmap``-batched operand in a dual level); always ``False``
        outside a ``torch.autograd.forward_ad.dual_level``.

    Raises
    ------
    RuntimeError
        Any error of ``fwad.unpack_dual`` other than the ``vmap`` batching
        error.
    """
    if _no_dual_level():
        return False
    return _read_tangents(tensors)[0]


def refuse_forward_ad(op: str, *tensors: Tensor | None) -> None:
    """Raise if an operand of a fused op carries a forward-mode tangent.

    Outside a dual level the check returns at once (one module-attribute read),
    with no device work.
    It sees the tangents of the current (innermost) level only: an operand
    that carries only an outer tangent inside a nested ``torch.func.jvp`` is
    not caught.

    Parameters
    ----------
    op : str
        Name of the op, for the error message.
    *tensors : torch.Tensor or None
        The operands (None: absent).

    Raises
    ------
    RuntimeError
        If an operand has a tangent at the current dual level, or if the check
        cannot read the tangent (a ``vmap``-batched operand inside a dual
        level). In the second case the message says so, and the batching
        error is the ``__cause__``.
    """
    if _no_dual_level():  # the common case: no forward mode, almost no cost
        return
    message = (
        f"forward-mode AD is not supported by {op}; use reverse mode, "
        "or finite differences"
    )
    dual, err = _read_tangents(tuple(t for t in tensors if t is not None))
    if err is not None:  # the answer comes from the vmap branch
        raise RuntimeError(f"{message} (a vmap-batched operand)") from err
    if dual:
        raise RuntimeError(message)
