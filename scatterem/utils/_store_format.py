"""Internals of the HDF5 format of :mod:`scatterem.utils.store`.

How one field value is laid out, and which class a stored group may become.

Scalars, strings and None become attributes of the object's group; arrays and
tensors become datasets. Objects, and lists of objects, are the caller's: they
become subgroups, written by :mod:`scatterem.utils.store` itself.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np
import torch

__all__ = [
    "FORMER_MODULE_PREFIXES",
    "RESERVED",
    "SerializationError",
    "compare_field_lists",
    "current_module",
    "declares_schema",
    "former_fields",
    "is_group",
    "nested_classes",
    "none_marker",
    "read_attribute",
    "read_dataset",
    "resolve_class",
    "stored_names",
    "subclass_tree",
    "write_leaf",
]

#: Arrays at least this large are compressed. Below it, gzip costs more time than
#: it saves space.
_COMPRESS_ABOVE_BYTES = 1 << 16

#: Attribute names the format itself uses on every object group.
RESERVED = frozenset({"class", "module", "mro", "fields", "sequence_of", "length"})


#: Module paths a file may record that have since moved, each prefix mapped to
#: the one that holds the same modules today. The package layout of 2026-09-28:
#: ``scatterem.utils.data`` became ``scatterem.data``, and the downloadable
#: datasets moved from ``scatterem.datasets`` into
#: ``scatterem.data.public``. The longest
#: matching prefix wins, so ``scatterem.datasets.public.x`` is not doubled.
FORMER_MODULE_PREFIXES: dict[str, str] = {
    "scatterem.utils.data": "scatterem.data",
    "scatterem.datasets.public": "scatterem.data.public",
    "scatterem.datasets": "scatterem.data.public",
}


def current_module(module: str) -> str:
    """Today's path of a module a file recorded; ``module`` itself if it never moved.

    Parameters
    ----------
    module : str
        A dotted module path as a file recorded it.

    Returns
    -------
    str
        The same path with a prefix of :data:`FORMER_MODULE_PREFIXES` replaced.
    """
    for old in sorted(FORMER_MODULE_PREFIXES, key=len, reverse=True):
        if module == old or module.startswith(old + "."):
            return FORMER_MODULE_PREFIXES[old] + module[len(old) :]
    return module


#: Marker attribute name for a field whose value is None. A separate attribute
#: rather than a sentinel string: any sentinel is a value some caller could
#: legitimately store, and HDF5 rejects the NUL byte that would make one safe.
def none_marker(name: str) -> str:
    return f"{name}__is_none"


class SerializationError(RuntimeError):
    """A file cannot be written, or cannot be loaded faithfully."""


def write_leaf(group, name: str, value: Any) -> None:
    """Write one non-object field value as an attribute or a dataset."""
    if value is None:
        group.attrs[none_marker(name)] = True
        return

    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.dtype == torch.bfloat16:  # numpy has no bfloat16: keep the bits
            tensor = tensor.view(torch.int16)
        _write_array(group, name, tensor.numpy(), torch_dtype=str(value.dtype))
        return

    if isinstance(value, np.ndarray):
        _write_array(group, name, value)
        return

    if type(value) in (bool, int, float):
        # h5py hands every attribute back as a numpy scalar; recording the Python
        # type lets the reader return what the writer was given.
        group.attrs[name] = value
        group.attrs[f"{name}__pytype"] = type(value).__name__
        return

    if isinstance(value, (str, bytes, np.bool_, np.integer, np.floating)):
        group.attrs[name] = value
        return

    if isinstance(value, (list, tuple)):
        # A homogeneous numeric sequence becomes an array; anything else is
        # written as JSON, which keeps str/None/mixed sequences exact rather than
        # coercing them into a numpy dtype.
        if value and all(
            isinstance(v, (int, float, np.integer, np.floating))
            and not isinstance(v, (bool, np.bool_))
            for v in value
        ):
            _write_array(group, name, np.asarray(value), kind=type(value).__name__)
        else:
            group.attrs[name] = json.dumps(list(value))
            group.attrs[f"{name}__json"] = type(value).__name__
        return

    raise SerializationError(
        f"cannot serialise field {name!r} of type {type(value).__name__}. Add a "
        f"case here rather than letting it be written as its repr."
    )


def _write_array(group, name: str, array: np.ndarray, *, torch_dtype=None, kind=None):
    if array.dtype == object:
        raise SerializationError(f"field {name!r} is an object-dtype array")
    compress = array.nbytes >= _COMPRESS_ABOVE_BYTES
    dset = group.create_dataset(
        name, data=array, compression="gzip" if compress else None
    )
    if torch_dtype is not None:
        dset.attrs["torch_dtype"] = torch_dtype
    if kind is not None:
        dset.attrs["sequence_kind"] = kind


def read_dataset(dset, device):
    array = dset[()]
    if "sequence_kind" in dset.attrs:
        values = np.asarray(array).tolist()
        return tuple(values) if dset.attrs["sequence_kind"] == "tuple" else values
    if "torch_dtype" in dset.attrs:
        dtype = getattr(torch, str(dset.attrs["torch_dtype"]).replace("torch.", ""))
        tensor = torch.as_tensor(np.asarray(array))
        if dtype == torch.bfloat16:
            tensor = tensor.view(torch.bfloat16)
        return tensor.to(dtype=dtype, device=device)
    return np.asarray(array)


def read_attribute(group, name: str):
    value = group.attrs[name]
    if isinstance(value, bytes):
        value = value.decode()
    kind = group.attrs.get(f"{name}__json")
    if kind is not None:
        decoded = json.loads(value)
        return tuple(decoded) if kind == "tuple" else decoded
    pytype = group.attrs.get(f"{name}__pytype")
    if pytype is not None:
        return {"bool": bool, "int": int, "float": float}[str(pytype)](value)
    return value


# -- which class a group becomes ----------------------------------------


def declares_schema(cls: type) -> bool:
    own = ("SERIAL_FIELDS", "SERIAL_ASSIGNED", "SERIAL_INIT_KWARGS", "_from_fields")
    return any(name in cls.__dict__ for name in own)


def subclass_tree(cls: type) -> Iterator[type]:
    yield cls
    for sub in cls.__subclasses__():
        yield from subclass_tree(sub)


def stored_names(cls: type) -> tuple[str, ...]:
    """The names a file may record for ``cls``: its own and its former ones."""
    own = vars(cls)
    return (
        cls.__name__,
        *own.get("SERIAL_FORMER_NAMES", ()),
        *own.get("SERIAL_FORMER_FIELDS", {}),
    )


def former_fields(cls: type, name: str) -> tuple[str, ...] | None:
    """The field list a file stored under the former name ``name`` records.

    ``None`` unless ``cls`` itself declares ``name`` in ``SERIAL_FORMER_FIELDS``:
    a former class whose schema differed, whose fields ``cls._from_former_fields``
    maps onto its own. A name in ``SERIAL_FORMER_NAMES`` shares the current schema.
    """
    declared = vars(cls).get("SERIAL_FORMER_FIELDS", {}).get(name)
    return None if declared is None else tuple(declared)


def resolve_class(group, candidates: tuple[type, ...]) -> tuple[type, str]:
    """The class to build for ``group``, and the stored name that selected it.

    The class is allowed, and as derived as the file says. The stored ancestry is
    walked from the most derived name. A name is honoured only when it names one
    of ``candidates`` or a subclass of one, and a subclass only when it declares
    its own schema (otherwise it holds nothing its ancestor does not, and the
    ancestor is built instead). A class also answers to the names in its own
    ``SERIAL_FORMER_NAMES`` and ``SERIAL_FORMER_FIELDS`` -- file-format
    migrations for a renamed or replaced class, deliberately not inherited by
    its subclasses. The recorded module, which breaks a tie between allowed
    classes of one name, is read through :func:`current_module`.
    """
    stored = group.attrs.get("class")
    module = group.attrs.get("module")
    if module is not None:
        module = current_module(module)
    ancestry = json.loads(group.attrs.get("mro", "[]")) or [stored]
    allowed = {c for base in candidates for c in subclass_tree(base)}
    for name in ancestry:
        matches = [
            c
            for c in allowed
            if name in stored_names(c) and (c in candidates or declares_schema(c))
        ]
        if len(matches) > 1 and name == stored and module is not None:
            matches = [c for c in matches if c.__module__ == module] or matches
        if len(matches) == 1:
            return matches[0], name
        if len(matches) > 1:
            raise SerializationError(
                f"file holds a {name!r}, which names {len(matches)} different "
                f"allowed classes: {sorted(c.__module__ for c in matches)}"
            )
    wanted = [c.__name__ for c in candidates]
    raise SerializationError(
        f"file holds a {stored!r} but {' or '.join(map(repr, wanted))} was "
        f"requested; the stored object's ancestry is {ancestry}"
    )


def compare_field_lists(
    class_name, stored_json, cls: type, former: Iterable[str] | None = None
) -> frozenset[str]:
    """Refuse a file whose field list differs from ``cls``'s, bar optional fields.

    Returns the fields of ``cls.SERIAL_OPTIONAL`` the file lacks, which the
    reader leaves to the constructor's default; any other difference raises.
    With ``former`` (a :func:`former_fields` list) the file must record exactly
    that list instead: a former schema has no optional fields.
    """
    expected: Iterable[str] = cls.SERIAL_FIELDS if former is None else former
    optional = getattr(cls, "SERIAL_OPTIONAL", ()) if former is None else ()
    if stored_json is None:
        raise SerializationError(f"{class_name}: file records no field list")
    stored = set(json.loads(stored_json))
    now = set(expected)
    absent = frozenset(now - stored)
    if stored <= now and absent <= set(optional):
        return absent
    added, gone = sorted(now - stored), sorted(stored - now)
    raise SerializationError(
        f"{class_name}: the file's fields do not match this version of the class."
        + (f" Missing from the file: {added}." if added else "")
        + (f" Present in the file but no longer declared: {gone}." if gone else "")
        + " Loading it would produce a silently incomplete object, so it is"
        " refused."
    )


def is_group(node) -> bool:
    import h5py

    return isinstance(node, h5py.Group)


def nested_classes(cls, name: str) -> tuple[type, ...]:
    """The Serializable subclasses a nested field may hold.

    Declared by the owner in ``SERIAL_NESTED``, because a group records only its
    class *name* and resolving that to a class by searching the interpreter would
    let a file choose what gets constructed.
    """
    nested = getattr(cls, "SERIAL_NESTED", {})
    if name not in nested:
        raise SerializationError(
            f"{cls.__name__}.SERIAL_NESTED does not say which class field "
            f"{name!r} holds, so it cannot be loaded. Declaring it keeps a file "
            f"from choosing which class to instantiate."
        )
    declared = nested[name]
    return tuple(declared) if isinstance(declared, tuple) else (declared,)
