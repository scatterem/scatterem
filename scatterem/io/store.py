"""Saving and loading containers, with a declared schema.

Replaces ``io/serialize.AutoSerialize``, which is derived from quantem (MIT) with
the copyright notice stripped. Written here from scratch, and deliberately not
the same design.

**What is different, and why.** ``AutoSerialize`` walks ``__dict__`` and writes
whatever it finds. That is convenient and it fails quietly:

* rename an attribute and the on-disk layout changes with no version bump, so old
  files load into new code as half-populated objects;
* add an attribute and old files silently load without it, leaving whatever the
  constructor defaulted to;
* the reader cannot tell a field that was *absent* from one that was *None*.

That is not hypothetical. ``Metadata4dstem.to_h5``/``from_h5`` -- hand-written
against the same reflective habit -- dropped ``sampling``, ``units`` and
``shape`` on the way out and returned a silently wrong object on the way back.

So here a class **declares** what it persists, in :attr:`Serializable.SERIAL_FIELDS`,
and the file records that list alongside a schema version. On load, the recorded
field list is compared with the class's current one and a mismatch **raises**,
naming exactly which fields were added and which went missing. A file that cannot
be loaded faithfully refuses to load at all.

**Format.** HDF5, via ``h5py``, which the package already requires and which the
electron-microscopy tools around it read. One group per object. Arrays become
datasets (gzip-compressed past a size threshold); scalars, strings and None
become attributes; nested :class:`Serializable` objects become subgroups.

**Devices are not persisted.** A tensor is written as plain data and comes back on
whatever device the caller asks for. A file that pins a GPU is a file that will
not open on someone else's machine.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

#: Bumped when the on-disk layout changes in a way older readers cannot handle.
SCHEMA_VERSION = 1

#: Arrays at least this large are compressed. Below it, gzip costs more time than
#: it saves space.
_COMPRESS_ABOVE_BYTES = 1 << 16

#: Marker attribute name for a field whose value is None. A separate attribute
#: rather than a sentinel string: any sentinel is a value some caller could
#: legitimately store, and HDF5 rejects the NUL byte that would make one safe.
def _none_marker(name: str) -> str:
    return f"{name}__is_none"


class SerializationError(RuntimeError):
    """A file cannot be written, or cannot be loaded faithfully."""


class Serializable:
    """Mixin giving :meth:`save` and :meth:`load` to a class with a declared schema.

    Subclasses set :attr:`SERIAL_FIELDS` to the attribute names to persist, and
    must be constructible from those names as keyword arguments -- or override
    :meth:`_from_fields` if construction needs more than that.
    """

    #: Attribute names this class persists. Order is irrelevant; membership is not.
    SERIAL_FIELDS: tuple[str, ...] = ()

    def save(self, path: str | Path, *, overwrite: bool = False) -> Path:
        """Write to ``path``. Returns the path written."""
        import h5py

        path = Path(path)
        if path.exists() and not overwrite:
            raise SerializationError(
                f"{path} exists; pass overwrite=True to replace it"
            )
        self._check_schema_declared()
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as fh:
            fh.attrs["scatterem_schema_version"] = SCHEMA_VERSION
            _write_object(fh, self)
        return path

    @classmethod
    def load(cls, path: str | Path, *, device: str | torch.device = "cpu"):
        """Read an instance of ``cls`` from ``path``.

        Raises:
            SerializationError: the file holds a different class, a schema version
                this code cannot read, or a field list that does not match
                ``cls.SERIAL_FIELDS``.
        """
        import h5py

        path = Path(path)
        if not path.is_file():
            raise SerializationError(f"no such file: {path}")
        with h5py.File(path, "r") as fh:
            version = int(fh.attrs.get("scatterem_schema_version", -1))
            if version != SCHEMA_VERSION:
                raise SerializationError(
                    f"{path} was written with schema version {version}; this code "
                    f"reads version {SCHEMA_VERSION}"
                )
            return _read_object(fh, cls, device)

    # -- hooks -------------------------------------------------------------

    @classmethod
    def _from_fields(cls, fields: dict[str, Any]):
        """Build an instance from loaded fields. Override if ``__init__`` differs."""
        return cls(**fields)

    def _check_schema_declared(self) -> None:
        if not self.SERIAL_FIELDS:
            raise SerializationError(
                f"{type(self).__name__} declares no SERIAL_FIELDS, so there is "
                f"nothing to save. Declaring the schema is deliberate -- writing "
                f"whatever happens to be in __dict__ is how a rename silently "
                f"changes a file format."
            )
        missing = [f for f in self.SERIAL_FIELDS if not hasattr(self, f)]
        if missing:
            raise SerializationError(
                f"{type(self).__name__} declares SERIAL_FIELDS it does not have: "
                f"{missing}"
            )


# -- writing ---------------------------------------------------------------


def _write_object(group, obj: Serializable) -> None:
    group.attrs["class"] = type(obj).__name__
    # The ancestry, so a subclass's file can be loaded as the base whose schema it
    # shares -- a You2026Carbon is a Dataset4dstem, and refusing that is unhelpful
    # when the field list already guarantees the load is faithful. Recorded as
    # names rather than resolved on read: a file must not choose what gets built.
    group.attrs["mro"] = json.dumps(
        [c.__name__ for c in type(obj).__mro__ if issubclass(c, Serializable)]
    )
    group.attrs["fields"] = json.dumps(sorted(obj.SERIAL_FIELDS))
    for name in obj.SERIAL_FIELDS:
        _write_value(group, name, getattr(obj, name))


def _write_value(group, name: str, value: Any) -> None:
    if isinstance(value, Serializable):
        sub = group.create_group(name)
        value._check_schema_declared()
        _write_object(sub, value)
        return

    if value is None:
        group.attrs[_none_marker(name)] = True
        return

    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
        _write_array(group, name, array, torch_dtype=str(value.dtype))
        return

    if isinstance(value, np.ndarray):
        _write_array(group, name, value)
        return

    if isinstance(value, (str, bytes, bool, int, float, np.integer, np.floating)):
        group.attrs[name] = value
        return

    if isinstance(value, (list, tuple)):
        # A homogeneous numeric sequence becomes an array; anything else is
        # written as JSON, which keeps str/None/mixed sequences exact rather than
        # coercing them into a numpy dtype.
        if value and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
            _write_array(group, name, np.asarray(value), kind=type(value).__name__)
        else:
            group.attrs[name] = json.dumps(list(value))
            group.attrs[f"{name}__json"] = True
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


# -- reading ---------------------------------------------------------------


def _read_object(group, cls, device):
    stored_class = group.attrs.get("class")
    ancestry = json.loads(group.attrs.get("mro", "[]")) or [stored_class]
    if cls.__name__ not in ancestry:
        raise SerializationError(
            f"file holds a {stored_class!r} but {cls.__name__!r} was requested; "
            f"the stored object's ancestry is {ancestry}"
        )
    _compare_field_lists(stored_class, group.attrs.get("fields"), cls.SERIAL_FIELDS)

    fields: dict[str, Any] = {}
    for name in cls.SERIAL_FIELDS:
        fields[name] = _read_value(group, name, cls, device)
    return cls._from_fields(fields)


def _compare_field_lists(class_name, stored_json, expected: Iterable[str]) -> None:
    if stored_json is None:
        raise SerializationError(f"{class_name}: file records no field list")
    stored = set(json.loads(stored_json))
    now = set(expected)
    if stored == now:
        return
    added, gone = sorted(now - stored), sorted(stored - now)
    raise SerializationError(
        f"{class_name}: the file's fields do not match this version of the class."
        + (f" Missing from the file: {added}." if added else "")
        + (f" Present in the file but no longer declared: {gone}." if gone else "")
        + " Loading it would produce a silently incomplete object, so it is"
        " refused."
    )


def _read_value(group, name: str, cls, device):
    if name in group and hasattr(group[name], "attrs") and _is_group(group[name]):
        nested = _nested_class(cls, name)
        return _read_object(group[name], nested, device)

    if name in group:  # a dataset
        dset = group[name]
        array = dset[()]
        if "sequence_kind" in dset.attrs:
            values = np.asarray(array).tolist()
            return tuple(values) if dset.attrs["sequence_kind"] == "tuple" else values
        if "torch_dtype" in dset.attrs:
            dtype = getattr(torch, str(dset.attrs["torch_dtype"]).replace("torch.", ""))
            return torch.as_tensor(np.asarray(array)).to(dtype=dtype, device=device)
        return np.asarray(array)

    if group.attrs.get(_none_marker(name)):
        return None

    if name in group.attrs:
        value = group.attrs[name]
        if isinstance(value, bytes):
            value = value.decode()
        if group.attrs.get(f"{name}__json"):
            return json.loads(value)
        return value

    raise SerializationError(
        f"{cls.__name__}: field {name!r} is declared but absent from the file"
    )


def _is_group(node) -> bool:
    import h5py

    return isinstance(node, h5py.Group)


def _nested_class(cls, name: str):
    """The Serializable subclass a nested field holds.

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
    return nested[name]


def read_schema(path: str | Path) -> dict[str, Any]:
    """The class, version and field list of a file, without constructing anything.

    Useful for telling why a load was refused.
    """
    import h5py

    with h5py.File(Path(path), "r") as fh:
        return {
            "schema_version": int(fh.attrs.get("scatterem_schema_version", -1)),
            "class": fh.attrs.get("class"),
            "fields": json.loads(fh.attrs.get("fields", "[]")),
        }
