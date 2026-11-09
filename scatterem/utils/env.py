"""Environment-variable switches and the backend registry.

Every boolean ``SCATTEREM_*`` switch is read through :func:`env_flag`, and
every choice between implementations of one kernel through
:func:`resolve_backend` (:func:`resolve_use_warp` is its Warp-or-torch
spelling).  They used to be spelled out at each call site, and one copy drifted
to ``bool(os.environ.get(...))``, which is True for ``"0"``:
``SCATTEREM_USE_WARP=0`` selected Warp for the PRISM projection and torch for
everything else.

A :class:`Backend` row describes one kernel: its implementations in preference
order, the environment variables that steer it, and the predicates that say
whether an implementation can run (``available``, cached: an import, a device,
a compiled extension) and whether it takes these operands (``accepts`` /
``declines``, per call: dtype, device, shape, grad, size).  Rows live with the
package that owns the kernels (``scatterem/nn/_backends.py`` and so on) and
name their predicates as ``"module:function"`` strings, resolved on first use,
so declaring a row imports nothing heavy and this module -- layer 1 -- never
imports the packages above it.
"""

from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

__all__ = [
    "FALSY",
    "TRUTHY",
    "Backend",
    "BackendUnavailable",
    "Switch",
    "backend_rows",
    "clear_backend_cache",
    "env_flag",
    "register_backend",
    "resolve_backend",
    "resolve_use_warp",
    "switched_on",
]

#: The spellings that switch a flag ON (compared case-insensitively).  Every
#: other value is OFF -- ``"0"``, ``"false"``, ``"no"``, ``""`` and also
#: ``"on"``: the set is closed, not a heuristic.
TRUTHY = ("1", "true", "yes")

#: The spellings that switch a DEFAULT-ON flag off (case-insensitive).
FALSY = ("0", "false", "no")

#: The variable every Warp-or-torch row with ``use_warp_env=True`` obeys.
USE_WARP = "SCATTEREM_USE_WARP"

#: The request that lets the registry choose.
AUTO = "auto"


def env_flag(name: str, default: bool = False) -> bool:
    """Whether the environment switch ``name`` is on.

    Parameters
    ----------
    name : str
        Environment variable name.
    default : bool, optional
        The state of the switch when the variable says nothing decisive.  An
        opt-in switch (``False``, the default) is on iff the value is one of
        :data:`TRUTHY`; a kill switch (``True``) is on unless the value is one
        of :data:`FALSY`.  Either way an unset, empty or unrecognised value
        leaves the switch at ``default``.

    Returns
    -------
    bool
        The state of the switch.
    """
    value = os.environ.get(name, "").lower()
    return value not in FALSY if default else value in TRUTHY


class BackendUnavailable(RuntimeError):
    """An explicitly requested implementation cannot run for these operands."""


@dataclass(frozen=True)
class Switch:
    """A boolean environment switch that gates one implementation.

    Parameters
    ----------
    env : str
        The variable, parsed by :func:`env_flag`.
    default : bool, optional
        ``True`` (the default) for a kill switch, on unless the variable is
        one of :data:`FALSY`; ``False`` for an opt-in, off unless it is one of
        :data:`TRUTHY`.
    """

    env: str
    default: bool = True

    def on(self) -> bool:
        """Whether the switch lets its implementation run (read per call).

        Returns
        -------
        bool
            :func:`env_flag` of the variable at this switch's default.
        """
        return env_flag(self.env, self.default)


@dataclass(frozen=True)
class Backend:
    """One kernel and the implementations :func:`resolve_backend` chooses from.

    Parameters
    ----------
    name : str
        The kernel's registry name, unique across the library.
    impls : tuple of str
        Implementation names in preference order.  The last one is the
        fallback and must carry no predicate, so ``"auto"`` always resolves.
    default : str, optional
        What the row does when neither the caller nor the environment asks
        for anything: ``"auto"`` (the default) walks ``impls``; an
        implementation name selects it without consulting its predicates.
    env : str or None, optional
        A string-valued variable naming an implementation (or ``"auto"``).
    use_warp_env : bool, optional
        Whether ``SCATTEREM_USE_WARP`` (set and non-empty) counts as a request:
        on (:func:`env_flag`) asks for ``"warp"``, anything else for
        ``"torch"``.
    switches : mapping of str to tuple of Switch, optional
        Per implementation, the boolean switches that must all be on for the
        ``"auto"`` walk to consider it.  An explicit request ignores them.
    available : mapping of str to tuple of str, optional
        Per implementation, ``"module:function"`` predicates taking no
        argument, cached until :func:`clear_backend_cache`.
    accepts, declines : mapping of str to tuple of str, optional
        Per implementation, ``"module:function"`` predicates evaluated on
        every call with the operands passed to :func:`resolve_backend`: every
        ``accepts`` must be true and every ``declines`` false.  A module that
        fails to import makes the implementation unavailable.
    """

    name: str
    impls: tuple[str, ...]
    default: str = AUTO
    env: str | None = None
    use_warp_env: bool = False
    switches: Mapping[str, tuple[Switch, ...]] = field(default_factory=dict)
    available: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    accepts: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    declines: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.impls or len(set(self.impls)) != len(self.impls):
            raise ValueError(f"{self.name}: impls must be distinct and non-empty")
        if self.default != AUTO and self.default not in self.impls:
            raise ValueError(f"{self.name}: default {self.default!r} not in impls")
        if self.use_warp_env and not {"warp", "torch"} <= set(self.impls):
            raise ValueError(f"{self.name}: use_warp_env needs warp and torch impls")
        keyed = (self.switches, self.available, self.accepts, self.declines)
        for mapping in keyed:
            unknown = set(mapping) - set(self.impls)
            if unknown:
                raise ValueError(f"{self.name}: unknown impls {sorted(unknown)}")
        if any(self.impls[-1] in mapping for mapping in keyed):
            raise ValueError(f"{self.name}: the fallback {self.impls[-1]!r} is gated")

    def predicates(self) -> tuple[str, ...]:
        """Every ``"module:function"`` this row names.

        Returns
        -------
        tuple of str
            The ``available``, ``accepts`` and ``declines`` references.
        """
        maps = (self.available, self.accepts, self.declines)
        return tuple(ref for m in maps for refs in m.values() for ref in refs)


_REGISTRY: dict[str, Backend] = {}
_AVAILABLE: dict[tuple[str, str], bool] = {}


def register_backend(row: Backend) -> Backend:
    """Add ``row`` to the registry.

    Registering an equal row again (a module reload) is a no-op; a different
    row under a registered name is refused, so two kernels cannot silently
    share one.

    Parameters
    ----------
    row : Backend
        The row.

    Returns
    -------
    Backend
        ``row``, so a module can write ``NAME = register_backend(Backend(...))``.

    Raises
    ------
    ValueError
        If another row is registered under ``row.name``.
    """
    existing = _REGISTRY.get(row.name)
    if existing is not None and existing != row:
        raise ValueError(f"a different backend row is registered as {row.name!r}")
    _REGISTRY[row.name] = row
    for impl in row.impls:
        _AVAILABLE.pop((row.name, impl), None)
    return row


def backend_rows() -> Mapping[str, Backend]:
    """The registered rows by name (a read-only view).

    Returns
    -------
    Mapping
        Name to :class:`Backend`.
    """
    return MappingProxyType(_REGISTRY)


def clear_backend_cache() -> None:
    """Forget every cached ``available`` verdict."""
    _AVAILABLE.clear()


def _resolve_ref(ref: str) -> Callable[..., bool] | None:
    """The function ``"module:function"`` names, or None if it cannot import.

    Looked up on every call (``sys.modules`` first), so a test that
    monkeypatches the function, or swaps the module in ``sys.modules``, is
    seen.
    """
    module_name, _, attr = ref.partition(":")
    module = sys.modules.get(module_name)
    if module is None:
        try:
            module = importlib.import_module(module_name)
        except ImportError:  # an optional dependency (Triton) is missing
            return None
    return getattr(module, attr)


def _available(row: Backend, impl: str) -> bool:
    """Every predicate of ``impl`` imports and its ``available`` checks pass."""
    key = (row.name, impl)
    if key not in _AVAILABLE:
        ok = True
        for ref in row.available.get(impl, ()):
            fn = _resolve_ref(ref)
            ok = ok and fn is not None and bool(fn())
        for ref in (*row.accepts.get(impl, ()), *row.declines.get(impl, ())):
            ok = ok and _resolve_ref(ref) is not None
        _AVAILABLE[key] = ok
    return _AVAILABLE[key]


def _takes(row: Backend, impl: str, args: tuple) -> bool:
    """Whether ``impl`` is available and its per-call predicates pass."""
    if not _available(row, impl):
        return False
    for ref in row.accepts.get(impl, ()):
        fn = _resolve_ref(ref)
        if fn is None or not fn(*args):
            return False
    for ref in row.declines.get(impl, ()):
        fn = _resolve_ref(ref)
        if fn is None or fn(*args):
            return False
    return True


#: A preference for Warp (``use_warp=True`` or ``SCATTEREM_USE_WARP`` on): Warp
#: where it takes the operands, else the ``"auto"`` walk.
_PREFER_WARP = "prefer-warp"


def _request(row: Backend, requested: str | None) -> tuple[str, str]:
    """``(request, where it came from)``; a request is an impl, auto or a preference."""
    if requested is not None:
        return requested, "argument"
    if row.env is not None and os.environ.get(row.env, ""):
        return os.environ[row.env].lower(), row.env
    if row.use_warp_env and os.environ.get(USE_WARP, ""):
        return (_PREFER_WARP if env_flag(USE_WARP) else "torch"), USE_WARP
    return row.default, "default"


def _walk(row: Backend, args: tuple, skip: str | None = None) -> str:
    """The ``"auto"`` choice: the first switched-on implementation that takes."""
    for impl in row.impls[:-1]:
        if impl == skip or not all(s.on() for s in row.switches.get(impl, ())):
            continue
        if _takes(row, impl, args):
            return impl
    return row.impls[-1]


def _row(kernel: str | Backend) -> Backend:
    if isinstance(kernel, Backend):
        return kernel
    try:
        return _REGISTRY[kernel]
    except KeyError:
        raise KeyError(
            f"no backend row {kernel!r}; registered: {sorted(_REGISTRY)}"
        ) from None


def resolve_backend(kernel: str | Backend, requested: str | None = None, *args) -> str:
    """The implementation of ``kernel`` to run on these operands.

    The request is the ``requested`` argument, else the row's ``env``
    variable, else ``SCATTEREM_USE_WARP`` (for a ``use_warp_env`` row), else
    the row's ``default``.  ``"auto"`` takes the first implementation whose
    switches are on, which is available, and whose predicates accept
    ``args``; the last implementation is the fallback.  An explicit request
    is honoured or refused, never silently replaced; ``SCATTEREM_USE_WARP``
    switched on is a preference for Warp (see :func:`resolve_use_warp`), and
    switched off a request for torch.

    Parameters
    ----------
    kernel : str or Backend
        A registered row's name, or the row itself.
    requested : str or None, optional
        ``None`` (not specified), ``"auto"`` or an implementation name.
    *args
        The operands, passed to the row's ``accepts`` / ``declines``.

    Returns
    -------
    str
        An implementation name from the row's ``impls``.

    Raises
    ------
    ValueError
        If the request names no implementation of this row.
    BackendUnavailable
        If an explicit request (argument or environment) cannot run on
        ``args``.
    """
    row = _row(kernel)
    choice, source = _request(row, requested)
    if choice == AUTO:
        return _walk(row, args)
    if choice == _PREFER_WARP and "warp" in row.impls:
        return "warp" if _takes(row, "warp", args) else _walk(row, args, skip="warp")
    if choice not in row.impls:
        raise ValueError(
            f"{row.name}: {source} asks for {choice!r}; choose one of "
            f"{(AUTO, *row.impls)}"
        )
    if source != "default" and not _takes(row, choice, args):
        raise BackendUnavailable(
            f"{row.name}: {source} asks for {choice!r}, which is unavailable or "
            f"does not accept these operands"
        )
    return choice


def switched_on(kernel: str | Backend, impl: str) -> bool:
    """Whether every switch of ``impl`` is on, read now.

    For a caller that arms an implementation ahead of time, outside the call
    that runs it (a ``torch.compile`` route resolves its kernel before tracing).

    Parameters
    ----------
    kernel : str or Backend
        A registered row's name, or the row itself.
    impl : str
        One of the row's implementations.

    Returns
    -------
    bool
        ``True`` if ``impl`` has no switch or all of them are on.
    """
    row = _row(kernel)
    if impl not in row.impls:
        raise ValueError(f"{row.name}: no implementation {impl!r}")
    return all(s.on() for s in row.switches.get(impl, ()))


def resolve_use_warp(
    kernel: str | Backend, use_warp: bool | None = None, *args
) -> bool:
    """Warp (True) or torch (False) for a Warp-or-torch row.

    :func:`resolve_backend` with ``use_warp`` spelled as a request: ``False``
    is ``"torch"``, ``None`` defers to the environment (``SCATTEREM_USE_WARP``
    for a ``use_warp_env`` row) and then to the row's default, and ``True`` --
    like ``SCATTEREM_USE_WARP`` switched on -- is a *preference*: Warp where
    the row's Warp predicates take the operands, else the ``"auto"`` choice.
    A row whose Warp implementation has no predicate therefore always gets
    Warp; to demand it, call ``resolve_backend(kernel, "warp", ...)``.  Only
    the default differs between operations.

    Parameters
    ----------
    kernel : str or Backend
        A registered row with ``"warp"`` and ``"torch"`` implementations.
    use_warp : bool or None, optional
        The caller's explicit choice; ``None`` means "not specified".
    *args
        The operands, passed to the row's predicates.

    Returns
    -------
    bool
        True for the Warp route.
    """
    if use_warp is None:
        requested = None
    else:
        requested = _PREFER_WARP if use_warp else "torch"
    return resolve_backend(kernel, requested, *args) == "warp"
