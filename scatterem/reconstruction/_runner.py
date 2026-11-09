"""The reconstruction convention: a frozen config in, a typed result out.

One call, :func:`run`, reaches every registered reconstruction family. The
family is chosen by the TYPE of the config, never by a string or a flag: each
family defines a frozen :class:`ReconstructionConfig` subclass whose fields and
defaults are exactly those of the function it wraps, and a :class:`Result`
subclass with named outputs. The implementing module registers the pair with
:func:`register_runner`, so importing a config type is what makes it runnable.

Three rules hold for every runner, and are what separates :func:`run` from the
older per-family entry points:

* **The device comes from the data.** Outputs are placed on ``data.device``.
  A runner that departs from this documents it and passes ``device=`` to
  ``Provenance.capture``, so ``provenance.device`` is where the outputs are.
* **No hidden writes.** A runner never assigns to ``data`` or ``data.meta``;
  intermediate products another call could reuse are passed in explicitly
  through ``reuse=`` instead of being cached on the input.
* **Every result says where it came from.** :class:`Provenance` records the
  method, the fully resolved config, the library version, the input's name, a
  snapshot copy of its metadata and the device. It carries no timestamp and no
  git SHA, so two identical runs compare equal field by field.

``run(data, config, meta=other)`` reconstructs the same data under other
metadata (for example a second aberration fit). The runner gets a cheap view of
``data`` (:func:`~scatterem.data.datasets.with_meta`) and ``data.meta`` stays
as it was; the provenance records the override.
"""

import dataclasses
import warnings
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar, Generic, TypeVar

import torch

__all__ = [
    "Provenance",
    "ReconstructionConfig",
    "Result",
    "register_runner",
    "run",
]


def _empty_mapping() -> Mapping[str, Any]:
    return MappingProxyType({})


def _scatterem_version() -> str:
    # Resolved at call time: the package ``__init__`` is still executing when
    # this module is first imported through ``scatterem.reconstruction``.
    import scatterem

    return str(scatterem.__version__)


def _data_device(data: Any) -> torch.device:
    device = getattr(data, "device", None)
    if device is None:
        raise TypeError(
            f"cannot tell which device a {type(data).__name__} lives on: it has "
            "no `device` attribute; pass `device=` to Provenance.capture"
        )
    return torch.device(device)


@dataclass(frozen=True, eq=False)
class Provenance:
    """Where a :class:`Result` came from.

    Deliberately free of a timestamp or git SHA: either would make two
    identical runs differ, and golden tests compare results field by field.

    Parameters
    ----------
    method : str
        The reconstruction family, the config type's ``method``.
    config : Mapping
        The fully resolved config, one entry per config field (read-only).
    scatterem_version : str
        ``scatterem.__version__`` at the time of the run.
    input_name : str or None
        The input's ``name``, when it has a string one.
    meta : object or None
        A detached snapshot copy of the input's ``meta``: later changes to the
        input's metadata do not reach it, and it holds no autograd graph.
    device : torch.device
        The device the input lived on, where the outputs are placed.
    meta_override : bool, optional
        ``True`` when the run read metadata other than the dataset's own
        (``run(..., meta=...)`` or a :func:`~scatterem.data.datasets.with_meta`
        view); ``meta`` is then that metadata. Default ``False``.
    """

    method: str
    config: Mapping[str, Any]
    scatterem_version: str
    input_name: str | None
    meta: Any | None
    device: torch.device
    meta_override: bool = False

    @classmethod
    def capture(
        cls,
        config: "ReconstructionConfig[Any]",
        data: Any,
        *,
        device: torch.device | None = None,
    ) -> "Provenance":
        """Record the provenance of running ``config`` on ``data``.

        Reads ``data`` and never writes to it: the metadata is copied with
        :func:`~scatterem.data.datasets._meta.copy_meta`.

        Parameters
        ----------
        config : ReconstructionConfig
            The config the run used.
        data : object
            The run's input. ``name``, ``meta`` and ``device`` are read when
            present.
        device : torch.device, optional
            Overrides ``data.device``, for inputs that do not carry one.

        Returns
        -------
        Provenance
            The record, with ``config`` resolved field by field.

        Raises
        ------
        TypeError
            If ``config`` is not a :class:`ReconstructionConfig`, or if
            ``device`` is omitted and ``data`` has no ``device``.
        """
        if not isinstance(config, ReconstructionConfig):
            raise TypeError(
                f"config must be a ReconstructionConfig, got {type(config).__name__}"
            )
        # Imported here: ``import scatterem.reconstruction`` must not load the
        # dataset classes.
        from scatterem.data.datasets._meta import META_OVERRIDE_ATTR, copy_meta

        name = getattr(data, "name", None)
        return cls(
            method=config.method,
            config=config.resolved(),
            scatterem_version=_scatterem_version(),
            input_name=name if isinstance(name, str) else None,
            meta=copy_meta(getattr(data, "meta", None)),
            device=_data_device(data) if device is None else torch.device(device),
            meta_override=getattr(data, META_OVERRIDE_ATTR, False) is True,
        )


@dataclass(frozen=True, eq=False, kw_only=True)
class Result:
    """Base of every reconstruction result.

    A family's result adds its named outputs as fields and implements
    :meth:`as_tuple`, which returns exactly what the family's older function
    returned, so a caller can switch entry point without changing a number.
    Results compare by identity: their fields are tensors.

    Parameters
    ----------
    provenance : Provenance
        Where the result came from.
    diagnostics : Mapping, optional
        Secondary outputs a family reports beside its named ones (read-only;
        empty by default).
    """

    provenance: Provenance
    diagnostics: Mapping[str, Any] = field(default_factory=_empty_mapping)

    def as_tuple(self) -> tuple[Any, ...]:
        """Return the outputs in the order the family's older function did.

        Returns
        -------
        tuple
            The older entry point's return value.

        Raises
        ------
        NotImplementedError
            On the base class; every family result overrides it.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not define as_tuple(); each family "
            "result must return its older entry point's tuple"
        )


R = TypeVar("R", bound=Result)

@dataclass(frozen=True)
class ReconstructionConfig(Generic[R]):
    """Base of every reconstruction config.

    A family subclasses it as a frozen dataclass whose fields and defaults are
    exactly the keyword arguments of the function it wraps, parameterised by
    its result type (``ReconstructionConfig[DirectPtychographyResult]``) so
    :func:`run` is typed per family, and sets the class attribute ``method``.
    """

    method: ClassVar[str]

    def resolved(self) -> Mapping[str, Any]:
        """Return every field of this config, by name.

        Returns
        -------
        Mapping
            A read-only ``{field name: value}`` view, in declaration order.
            Values are the config's own (configs hold scalars and strings).
        """
        return MappingProxyType(
            {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        )


Runner = Callable[[Any, Any, Mapping[str, Any]], Result]


@dataclass(frozen=True)
class _Registration:
    runner: Runner
    reuse_keys: frozenset[str]


_REGISTRY: dict[type, _Registration] = {}


def register_runner(
    config_type: type[ReconstructionConfig[Any]],
    *,
    reuse_keys: Iterable[str] = (),
) -> Callable[[Runner], Runner]:
    """Register the function :func:`run` calls for ``config_type``.

    Used as a decorator in the module that implements the family. The runner
    is called as ``runner(data, config, reuse)``, with ``reuse`` a read-only
    mapping holding only keys from ``reuse_keys``.

    Parameters
    ----------
    config_type : type
        A :class:`ReconstructionConfig` subclass that sets ``method``.
    reuse_keys : sequence of str, optional
        The ``reuse=`` keys this family accepts; :func:`run` rejects others.

    Returns
    -------
    callable
        A decorator that registers its argument and returns it unchanged.

    Raises
    ------
    TypeError
        If ``config_type`` is not a :class:`ReconstructionConfig` subclass
        with a string ``method``.
    ValueError
        If ``config_type`` already has a runner.
    """
    if not (
        isinstance(config_type, type)
        and issubclass(config_type, ReconstructionConfig)
        and isinstance(getattr(config_type, "method", None), str)
    ):
        raise TypeError(
            "config_type must be a ReconstructionConfig subclass that sets a "
            f"string `method`, got {config_type!r}"
        )
    if config_type in _REGISTRY:
        raise ValueError(f"{config_type.__name__} already has a runner")
    keys = frozenset(reuse_keys)

    def decorate(runner: Runner) -> Runner:
        _REGISTRY[config_type] = _Registration(runner, keys)
        return runner

    return decorate


def _lookup(config_type: type) -> _Registration:
    for klass in config_type.__mro__:
        registration = _REGISTRY.get(klass)
        if registration is not None:
            return registration
    known = ", ".join(sorted(k.__name__ for k in _REGISTRY)) or "none"
    raise TypeError(
        f"no reconstruction is registered for {config_type.__name__} "
        f"(registered: {known}); import the module that defines the config"
    )


def _with_meta_override(data: Any, meta: Any, reuse: Mapping[str, Any]) -> Any:
    """A view of ``data`` reading ``meta``; refuse vBFs built under other metadata."""
    # Imported here: ``import scatterem.reconstruction`` must not load the
    # dataset classes.
    from scatterem.data.datasets import with_meta

    providers = [reuse.get("vbf")]
    cache = reuse.get("vbf_cache")
    if cache is not None:
        providers += [cache.vbf, None if cache.spare is None else cache.spare[1]]
    if any(p is not None and getattr(p, "meta", None) is not meta for p in providers):
        raise ValueError(
            "a reused vBF was built under other metadata than meta=; build it "
            "on the view instead: with_meta(data, meta).get_vbf(threshold)"
        )
    return with_meta(data, meta)


def _warn_placeholder_reciprocal_sampling(data: Any) -> None:
    """Warn when a 4D-STEM input still has the ``dk = (1.0, 1.0)`` placeholder.

    ``Dataset4DStem.from_array(..., energy=...)`` without ``reciprocal_step``
    sets that placeholder. Every FF-STEM reconstruction reads ``dk``, so a run
    on the placeholder finishes but gives wrong aberrations and images.
    """
    sampling = getattr(data, "sampling", None)
    try:
        values = [float(v) for v in sampling]  # pyright: ignore[reportOptionalIterable]
    except (TypeError, ValueError):
        return
    if len(values) == 4 and values[2:] == [1.0, 1.0]:
        warnings.warn(
            "the reciprocal sampling dk is the uncalibrated (1.0, 1.0) placeholder; "
            "call dataset.calibrate_reciprocal_from_bright_field() or pass "
            "reciprocal_step= to Dataset4DStem.from_array before run()",
            UserWarning,
            stacklevel=3,
        )


def run(
    data: Any,
    config: ReconstructionConfig[R],
    *,
    reuse: Mapping[str, Any] | None = None,
    meta: Any = None,
) -> R:
    """Run the reconstruction ``config`` selects on ``data``.

    The family is chosen by ``type(config)`` (the nearest registered class in
    its MRO). ``data`` and ``data.meta`` are never written to; outputs are
    placed on ``data.device`` or, for a runner that departs from that rule
    (see the module docstring), where ``provenance.device`` says.

    To compare two aberration fits on one dataset, pass each fit's metadata::

        a = run(ds, config, meta=fit_a.applied_to(ds.meta))
        b = run(ds, config, meta=fit_b.applied_to(ds.meta))

    Parameters
    ----------
    data : object
        The input the family expects (a dataset, a series, ...).
    config : ReconstructionConfig
        The family's frozen config.
    reuse : Mapping, optional
        Intermediate products from earlier runs, by the key the family
        documents (for example a virtual bright field). Nothing is reused
        unless it is passed here.
    meta : object, optional
        Metadata to use instead of ``data.meta``. The runner gets a
        :func:`~scatterem.data.datasets.with_meta` view of ``data``: it shares
        the array, starts with an empty vBF cache and leaves ``data``
        unchanged. The result's ``provenance.meta_override`` is ``True``. A
        reused vBF (``"vbf"`` / ``"vbf_cache"``) must have been built under
        this ``meta``, for example ``with_meta(data, meta).get_vbf(t)``.

    Returns
    -------
    Result
        The family's result type, as ``config``'s type parameter declares.

    Raises
    ------
    TypeError
        If ``config`` is not a :class:`ReconstructionConfig`, if no runner is
        registered for its type, or if the runner does not return a
        :class:`Result`.
    ValueError
        If ``reuse`` holds a key the family does not accept, or, with
        ``meta``, a vBF provider built under other metadata.

    Warns
    -----
    UserWarning
        If ``data`` is 4D-STEM data whose reciprocal sampling ``dk`` is still
        the ``(1.0, 1.0)`` placeholder of ``Dataset4DStem.from_array``.
    """
    if not isinstance(config, ReconstructionConfig):
        raise TypeError(
            f"config must be a ReconstructionConfig, got {type(config).__name__}"
        )
    registration = _lookup(type(config))
    given = dict(reuse) if reuse is not None else {}
    unknown = sorted(set(given) - registration.reuse_keys)
    if unknown:
        accepted = sorted(registration.reuse_keys) or "none"
        raise ValueError(
            f"{type(config).__name__} does not accept reuse keys {unknown} "
            f"(accepted: {accepted})"
        )
    if meta is not None:
        data = _with_meta_override(data, meta, given)
    _warn_placeholder_reciprocal_sampling(data)
    result = registration.runner(data, config, MappingProxyType(given))
    if not isinstance(result, Result):
        raise TypeError(
            f"the runner for {type(config).__name__} returned "
            f"{type(result).__name__}, not a Result"
        )
    return result  # pyright: ignore[reportReturnType]
