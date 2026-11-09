"""Saving and loading containers, with a declared schema.

Replaces ``io/serialize.AutoSerialize`` (removed in 0.1.0), which was derived
from quantem (MIT) with the copyright notice stripped. Written here from scratch, and deliberately not
the same design.

**What is different, and why.** ``AutoSerialize`` walks ``__dict__`` and writes
whatever it finds. That is convenient and it fails quietly:

* rename an attribute and the on-disk layout changes with no version bump, so old
  files load into new code as half-populated objects;
* add an attribute and old files silently load without it, leaving whatever the
  constructor defaulted to;
* the reader cannot tell a field that was *absent* from one that was *None*;
* and the reader rebuilds the class from a pickled definition stored in the file,
  so opening a file runs code the file chose.

That is not hypothetical. ``AcquisitionMetadata.to_h5``/``from_h5`` -- hand-written
against the same reflective habit -- dropped ``sampling``, ``units`` and
``shape`` on the way out and returned a silently wrong object on the way back.

So here a class **declares** what it persists, in :attr:`Serializable.SERIAL_FIELDS`,
and the file records that list alongside a schema version. On load, the recorded
field list is compared with the class's current one and a mismatch **raises**,
naming exactly which fields were added and which went missing. A file that cannot
be loaded faithfully refuses to load at all. The one exception is declared: a
field in :attr:`Serializable.SERIAL_OPTIONAL` may be absent from a file written
before it existed, and then takes the constructor's default. Nothing in a file
is ever executed: the file records class *names*, and a name is only honoured when
it names a class the caller already allowed (the requested class, a subclass of
it, or a class the owner listed in ``SERIAL_NESTED``).

**Format.** HDF5, via ``h5py``, which the package already requires and which the
electron-microscopy tools around it read. One group per object. Arrays become
datasets (gzip-compressed past a size threshold); scalars, strings and None
become attributes; nested :class:`Serializable` objects become subgroups, and a
list of them a subgroup of numbered subgroups.

**Devices are not persisted.** A tensor is written as plain data and comes back on
whatever device the caller asks for. A file that pins a GPU is a file that will
not open on someone else's machine.

This module lives in ``scatterem.utils`` so the data containers in
``scatterem.data`` can derive from :class:`Serializable`.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, ClassVar

import torch

from scatterem.utils._store_format import (
    RESERVED,
    SerializationError,
    compare_field_lists,
    former_fields,
    is_group,
    nested_classes,
    none_marker,
    read_attribute,
    read_dataset,
    resolve_class,
    write_leaf,
)

__all__ = [
    "SCHEMA_VERSION",
    "Serializable",
    "SerializationError",
    "load_serializable",
    "read_schema",
]

#: Bumped when the on-disk layout changes in a way older readers cannot handle.
#: Additions an older reader simply ignores (the ``module`` and ``__pytype``
#: attributes, numbered subgroups for a list of objects) do not bump it.
SCHEMA_VERSION = 1


class Serializable:
    """Give a declared-schema class :meth:`~Serializable.save` and ``load``.

    Subclasses set :attr:`SERIAL_FIELDS` to the attribute names to persist, and
    must be constructible from those names as keyword arguments -- or override
    :meth:`_from_fields` if construction needs more than that. A field holding
    another :class:`Serializable` (or a list of them) names the class it holds
    in :attr:`SERIAL_NESTED`; a tuple there allows any one of several classes.

    A subclass that declares neither ``SERIAL_FIELDS`` nor ``_from_fields``
    itself adds behaviour but not state, and a file it wrote loads as its
    nearest ancestor that does -- a download wrapper whose constructor fetches
    data cannot be rebuilt from fields. A subclass that shares its ancestor's
    schema *and* is built the same way says so with a class keyword,
    ``class Sub(Base, load_as_self=True)``, and then loads as itself.
    """

    def __init_subclass__(cls, *, load_as_self: bool = False, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if load_as_self:
            # An entry in the class's own namespace is what marks a declared schema.
            cls.SERIAL_FIELDS = tuple(cls.SERIAL_FIELDS)

    #: Attribute names this class persists. Order is irrelevant; membership is not.
    SERIAL_FIELDS: tuple[str, ...] = ()
    #: For each nested field, the class (or tuple of classes) it may hold.
    SERIAL_NESTED: ClassVar[dict[str, type | tuple[type, ...]]] = {}
    #: Fields restored by assignment after construction rather than passed to
    #: the constructor -- state the constructor would otherwise recompute, or
    #: apply to data that already has it applied.
    SERIAL_ASSIGNED: tuple[str, ...] = ()
    #: Extra keyword arguments the default :meth:`_from_fields` passes to the
    #: constructor, e.g. a flag that keeps it from re-normalising stored data.
    SERIAL_INIT_KWARGS: ClassVar[dict[str, Any]] = {}
    #: Names this class was stored under before a rename. A file recording one
    #: of them loads as this class. Read from the class's own namespace only, so
    #: a subclass does not answer to its parent's former names.
    SERIAL_FORMER_NAMES: ClassVar[tuple[str, ...]] = ()
    #: Fields a file may lack. Files written before the field was declared load,
    #: and the constructor's default fills it (the field is not passed). Every
    #: other difference between the file's field list and the class's is refused.
    SERIAL_OPTIONAL: ClassVar[tuple[str, ...]] = ()
    #: Former class names whose files record another field list, each mapped to
    #: that list. Such a file is read field by field (a nested field resolves
    #: through this class's :attr:`SERIAL_NESTED`) and handed to
    #: :meth:`_from_former_fields`. Read from the class's own namespace only.
    SERIAL_FORMER_FIELDS: ClassVar[dict[str, tuple[str, ...]]] = {}

    def save(self, path: str | Path, *, overwrite: bool = False) -> Path:
        """Write every declared field to one HDF5 file.

        Parameters
        ----------
        path : str or pathlib.Path
            File to write. Missing parent directories are created.
        overwrite : bool, optional
            Replace an existing file. Default ``False``, which refuses.

        Returns
        -------
        pathlib.Path
            The path written.

        Raises
        ------
        SerializationError
            If the file exists and ``overwrite`` is false, or if a declared
            field cannot be written.
        """
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
        """Read an instance of ``cls`` (or of a subclass) from ``path``.

        Parameters
        ----------
        path : str or pathlib.Path
            A file written by :meth:`save`.
        device : str or torch.device, optional
            Device every tensor field is placed on. Default ``"cpu"``.

        Returns
        -------
        Serializable
            The object. When the file holds a subclass of ``cls`` that declares
            its own schema, an instance of that subclass.

        Raises
        ------
        SerializationError
            The file holds an unrelated class, a schema version this code cannot
            read, or a field list that does not match the class's.
        """
        return load_serializable(path, (cls,), device=device)

    def print_tree(self, name: str | None = None) -> None:
        """Print the declared fields, recursing into nested objects.

        Parameters
        ----------
        name : str, optional
            Label for the root. Defaults to the class name.
        """
        # The tree on stdout IS this method's output.
        print(name or type(self).__name__)  # noqa: T201
        for line in _tree_lines(self, ""):
            print(line)  # noqa: T201

    # -- hooks -------------------------------------------------------------

    @classmethod
    def _from_fields(cls, fields: dict[str, Any]):
        """Build an instance from loaded fields. Override if ``__init__`` differs.

        The default calls ``cls(**fields, **SERIAL_INIT_KWARGS)`` with the
        fields named in :attr:`SERIAL_ASSIGNED` held back, then assigns those
        on the new instance. An assigned field in :attr:`SERIAL_OPTIONAL` that
        the file lacks is not assigned, so the class default stays.
        """
        assigned = {
            name: fields.pop(name) for name in cls.SERIAL_ASSIGNED if name in fields
        }
        obj = cls(**fields, **cls.SERIAL_INIT_KWARGS)
        for name, value in assigned.items():
            setattr(obj, name, value)
        return obj

    @classmethod
    def _from_former_fields(cls, name: str, fields: dict[str, Any]):
        """Build an instance from the fields of a file stored as ``name``.

        Called for the names in :attr:`SERIAL_FORMER_FIELDS`; a class that
        declares one overrides this to map the former fields onto its own.
        """
        raise SerializationError(
            f"{cls.__name__} declares former fields for {name!r} but no "
            f"_from_former_fields to map them"
        )

    def _check_schema_declared(self) -> None:
        if not self.SERIAL_FIELDS:
            raise SerializationError(
                f"{type(self).__name__} declares no SERIAL_FIELDS, so there is "
                f"nothing to save. Declaring the schema is deliberate -- writing "
                f"whatever happens to be in __dict__ is how a rename silently "
                f"changes a file format."
            )
        clash = sorted(RESERVED.intersection(self.SERIAL_FIELDS))
        if clash:
            raise SerializationError(
                f"{type(self).__name__} declares fields the format reserves: {clash}"
            )
        missing = [f for f in self.SERIAL_FIELDS if not hasattr(self, f)]
        if missing:
            raise SerializationError(
                f"{type(self).__name__} declares SERIAL_FIELDS it does not have: "
                f"{missing}"
            )


def _tree_lines(obj: Serializable, prefix: str) -> Iterator[str]:
    names = sorted(obj.SERIAL_FIELDS)
    for i, key in enumerate(names):
        last = i == len(names) - 1
        branch, pad = ("└── ", "    ") if last else ("├── ", "│   ")
        value = getattr(obj, key)
        if isinstance(value, Serializable):
            yield prefix + branch + key
            yield from _tree_lines(value, prefix + pad)
        else:
            yield prefix + branch + f"{key}: {type(value).__name__}"


# -- writing ---------------------------------------------------------------


def _write_object(group, obj: Serializable) -> None:
    group.attrs["class"] = type(obj).__name__
    group.attrs["module"] = type(obj).__module__
    # The ancestry, so a subclass's file can be loaded as the base whose schema it
    # shares -- a You2026Carbon is a Dataset4DStem, and refusing that is unhelpful
    # when the field list already guarantees the load is faithful. Recorded as
    # names rather than resolved on read: a file must not choose what gets built.
    group.attrs["mro"] = json.dumps(
        [c.__name__ for c in type(obj).__mro__ if issubclass(c, Serializable)]
    )
    group.attrs["fields"] = json.dumps(sorted(obj.SERIAL_FIELDS))
    for name in obj.SERIAL_FIELDS:
        _write_value(group, name, getattr(obj, name))


def _is_object_sequence(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and bool(value)
        and all(isinstance(v, Serializable) for v in value)
    )


def _write_value(group, name: str, value: Any) -> None:
    if isinstance(value, Serializable):
        sub = group.create_group(name)
        value._check_schema_declared()
        _write_object(sub, value)
        return

    if _is_object_sequence(value):
        sub = group.create_group(name)
        sub.attrs["sequence_of"] = type(value).__name__
        sub.attrs["length"] = len(value)
        for i, item in enumerate(value):
            item._check_schema_declared()
            _write_object(sub.create_group(str(i)), item)
        return

    write_leaf(group, name, value)


# -- reading ---------------------------------------------------------------


def load_serializable(
    path: str | Path,
    candidates: Sequence[type],
    *,
    device: str | torch.device = "cpu",
):
    """Read a file written by :meth:`Serializable.save`, as one of ``candidates``.

    Parameters
    ----------
    path : str or pathlib.Path
        The HDF5 file.
    candidates : sequence of type
        The classes the caller allows, each a :class:`Serializable` subclass.
        Their subclasses are allowed too; nothing else is.
    device : str or torch.device, optional
        Device every tensor field is placed on. Default ``"cpu"``.

    Returns
    -------
    Serializable
        The loaded object.

    Raises
    ------
    SerializationError
        If the file is missing, has another schema version, holds no allowed
        class, or its field list disagrees with the class's.
    """
    import h5py

    path = Path(path)
    if not path.is_file():
        hint = (
            " (a directory is a legacy AutoSerialize store, which this version "
            "does not read)"
            if path.is_dir()
            else ""
        )
        raise SerializationError(f"no such file: {path}{hint}")
    if not h5py.is_hdf5(path):
        raise SerializationError(
            f"{path} is not an HDF5 file; this version does not read a legacy "
            f"AutoSerialize zip store"
        )
    with h5py.File(path, "r") as fh:
        version = int(fh.attrs.get("scatterem_schema_version", -1))
        if version != SCHEMA_VERSION:
            raise SerializationError(
                f"{path} was written with schema version {version}; this code "
                f"reads version {SCHEMA_VERSION}"
            )
        return _read_object(fh, tuple(candidates), device)


def _read_object(group, candidates: tuple[type, ...], device):
    cls, stored_as = resolve_class(group, candidates)
    former = former_fields(cls, stored_as)
    absent = compare_field_lists(
        group.attrs.get("class"), group.attrs.get("fields"), cls, former
    )
    fields: dict[str, Any] = {}
    for name in cls.SERIAL_FIELDS if former is None else former:
        if name not in absent:
            fields[name] = _read_value(group, name, cls, device)
    if former is not None:
        return cls._from_former_fields(stored_as, fields)
    return cls._from_fields(fields)


def _read_value(group, name: str, cls, device):
    if name in group and is_group(group[name]):
        sub = group[name]
        nested = nested_classes(cls, name)
        if "sequence_of" in sub.attrs:
            items = [
                _read_object(sub[str(i)], nested, device)
                for i in range(int(sub.attrs["length"]))
            ]
            return tuple(items) if sub.attrs["sequence_of"] == "tuple" else items
        return _read_object(sub, nested, device)

    if name in group:  # a dataset
        return read_dataset(group[name], device)

    if group.attrs.get(none_marker(name)):
        return None

    if name in group.attrs:
        return read_attribute(group, name)

    raise SerializationError(
        f"{cls.__name__}: field {name!r} is declared but absent from the file"
    )


def read_schema(path: str | Path) -> dict[str, Any]:
    """The class, version and field list of a file, without constructing anything.

    Useful for telling why a load was refused.

    Parameters
    ----------
    path : str or pathlib.Path
        A file written by :meth:`Serializable.save`.

    Returns
    -------
    dict
        ``schema_version`` (int, ``-1`` if absent), ``class`` (the stored class
        name), ``module`` (its module, ``None`` for files that predate it) and
        ``fields`` (the recorded field list).
    """
    import h5py

    with h5py.File(Path(path), "r") as fh:
        return {
            "schema_version": int(fh.attrs.get("scatterem_schema_version", -1)),
            "class": fh.attrs.get("class"),
            "module": fh.attrs.get("module"),
            "fields": json.loads(fh.attrs.get("fields", "[]")),
        }
