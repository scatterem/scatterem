"""The declared-schema serializer that replaces the quantem-derived AutoSerialize.

The point of the redesign is that it refuses rather than half-loads, so most of
these tests are about what it declines to do.
"""

import numpy as np
import pytest
import torch

from scatterem.io.store import (
    SCHEMA_VERSION,
    Serializable,
    SerializationError,
    read_schema,
)


class Inner(Serializable):
    SERIAL_FIELDS = ("value",)

    def __init__(self, value):
        self.value = value


class Thing(Serializable):
    SERIAL_FIELDS = ("array", "tensor", "name", "count", "ratio", "units", "nothing", "inner")
    SERIAL_NESTED = {"inner": Inner}

    def __init__(self, array, tensor, name, count, ratio, units, nothing, inner):
        self.array = array
        self.tensor = tensor
        self.name = name
        self.count = count
        self.ratio = ratio
        self.units = units
        self.nothing = nothing
        self.inner = inner


def _thing():
    return Thing(
        array=np.arange(12, dtype=np.float64).reshape(3, 4),
        tensor=torch.arange(6, dtype=torch.float32).reshape(2, 3),
        name="carbon",
        count=7,
        ratio=0.25,
        units=["A", "A", "1/A", "1/A"],
        nothing=None,
        inner=Inner(value=np.ones(3)),
    )


class TestRoundTrip:
    def test_every_declared_field_survives(self, tmp_path):
        original, path = _thing(), tmp_path / "t.h5"
        original.save(path)
        back = Thing.load(path)

        np.testing.assert_array_equal(back.array, original.array)
        assert torch.equal(back.tensor, original.tensor)
        assert back.tensor.dtype == original.tensor.dtype
        assert back.name == "carbon"
        assert back.count == 7
        assert back.ratio == 0.25
        assert back.units == ["A", "A", "1/A", "1/A"]
        assert back.nothing is None
        np.testing.assert_array_equal(back.inner.value, original.inner.value)

    def test_none_is_distinguishable_from_absent(self, tmp_path):
        """AutoSerialize's reflective write could not tell these apart."""
        path = tmp_path / "t.h5"
        _thing().save(path)
        assert Thing.load(path).nothing is None

    def test_device_is_not_persisted(self, tmp_path):
        """A file that pins a GPU will not open on someone else's machine."""
        path = tmp_path / "t.h5"
        _thing().save(path)
        assert Thing.load(path, device="cpu").tensor.device.type == "cpu"

    def test_tuple_stays_a_tuple(self, tmp_path):
        class WithTuple(Serializable):
            SERIAL_FIELDS = ("pair",)

            def __init__(self, pair):
                self.pair = pair

        path = tmp_path / "t.h5"
        WithTuple(pair=(1.5, 2.5)).save(path)
        assert WithTuple.load(path).pair == (1.5, 2.5)

    def test_large_array_is_compressed(self, tmp_path):
        import h5py

        class Big(Serializable):
            SERIAL_FIELDS = ("data",)

            def __init__(self, data):
                self.data = data

        path = tmp_path / "big.h5"
        Big(data=np.zeros((256, 256))).save(path)
        with h5py.File(path) as fh:
            assert fh["data"].compression == "gzip"


class TestRefusals:
    def test_added_field_refuses_with_names(self, tmp_path):
        """The failure mode the redesign exists to prevent."""
        path = tmp_path / "t.h5"
        _thing().save(path)
        try:
            Thing.SERIAL_FIELDS = Thing.SERIAL_FIELDS + ("added_later",)
            with pytest.raises(SerializationError, match=r"Missing from the file.*added_later"):
                Thing.load(path)
        finally:
            Thing.SERIAL_FIELDS = tuple(f for f in Thing.SERIAL_FIELDS if f != "added_later")

    def test_removed_field_refuses_with_names(self, tmp_path):
        path = tmp_path / "t.h5"
        _thing().save(path)
        original = Thing.SERIAL_FIELDS
        try:
            Thing.SERIAL_FIELDS = tuple(f for f in original if f != "ratio")
            with pytest.raises(SerializationError, match="no longer declared.*ratio"):
                Thing.load(path)
        finally:
            Thing.SERIAL_FIELDS = original

    def test_wrong_class_refuses(self, tmp_path):
        path = tmp_path / "t.h5"
        _thing().save(path)
        with pytest.raises(SerializationError, match="holds a 'Thing'"):
            Inner.load(path)

    def test_schema_version_mismatch_refuses(self, tmp_path):
        import h5py

        path = tmp_path / "t.h5"
        _thing().save(path)
        with h5py.File(path, "a") as fh:
            fh.attrs["scatterem_schema_version"] = SCHEMA_VERSION + 1
        with pytest.raises(SerializationError, match="schema version"):
            Thing.load(path)

    def test_no_declared_schema_refuses(self, tmp_path):
        class Undeclared(Serializable):
            pass

        with pytest.raises(SerializationError, match="declares no SERIAL_FIELDS"):
            Undeclared().save(tmp_path / "x.h5")

    def test_declared_but_absent_attribute_refuses(self, tmp_path):
        class Wrong(Serializable):
            SERIAL_FIELDS = ("nope",)

        with pytest.raises(SerializationError, match="does not have"):
            Wrong().save(tmp_path / "x.h5")

    def test_unserialisable_type_refuses_rather_than_writing_a_repr(self, tmp_path):
        class Odd(Serializable):
            SERIAL_FIELDS = ("thing",)

            def __init__(self):
                self.thing = object()

        with pytest.raises(SerializationError, match="cannot serialise"):
            Odd().save(tmp_path / "x.h5")

    def test_nested_class_must_be_declared(self, tmp_path):
        """A file must not get to choose which class is constructed."""
        class Outer(Serializable):
            SERIAL_FIELDS = ("inner",)

            def __init__(self, inner):
                self.inner = inner

        path = tmp_path / "x.h5"
        Outer(inner=Inner(value=np.ones(2))).save(path)
        with pytest.raises(SerializationError, match="SERIAL_NESTED"):
            Outer.load(path)

    def test_existing_file_is_not_clobbered(self, tmp_path):
        path = tmp_path / "t.h5"
        _thing().save(path)
        with pytest.raises(SerializationError, match="pass overwrite=True"):
            _thing().save(path)
        _thing().save(path, overwrite=True)

    def test_missing_file_says_so(self, tmp_path):
        with pytest.raises(SerializationError, match="no such file"):
            Thing.load(tmp_path / "absent.h5")


def test_read_schema_explains_a_refusal(tmp_path):
    path = tmp_path / "t.h5"
    _thing().save(path)
    info = read_schema(path)
    assert info["class"] == "Thing"
    assert info["schema_version"] == SCHEMA_VERSION
    assert "ratio" in info["fields"]


class Derived(Thing):
    """A subclass that adds behaviour but not schema."""


def test_subclass_file_loads_as_its_base(tmp_path):
    """A You2026Carbon is a Dataset4dstem; refusing that load is unhelpful when
    the field list already guarantees faithfulness."""
    t = _thing()
    obj = Derived(**{f: getattr(t, f) for f in Thing.SERIAL_FIELDS})
    path = tmp_path / "d.h5"
    obj.save(path)
    back = Thing.load(path)
    assert back.name == "carbon"
    assert isinstance(Derived.load(path), Derived)


def test_unrelated_class_still_refuses(tmp_path):
    path = tmp_path / "d.h5"
    _thing().save(path)
    with pytest.raises(SerializationError, match="ancestry"):
        Inner.load(path)
