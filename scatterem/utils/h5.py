"""Narrowing accessors for HDF5 nodes.

``h5py.Group.__getitem__`` is typed as the union of everything a node can be --
``Group | Dataset | Datatype`` plus the three link classes -- because HDF5 gives
no static guarantee about what a name holds. A reader that knows its file's
layout therefore pays one diagnostic per union member at every lookup, which
says nothing the layout has not already settled.

These say which member the layout puts at a path, once, at the read. They are
narrowing only: neither performs a check, so a file that does not match its
reader's layout still fails exactly where and how it did before (an
``AttributeError`` or ``TypeError`` on the node that came back).
"""

from __future__ import annotations

from typing import cast

import h5py


def h5_group(parent: h5py.File | h5py.Group, name: str) -> h5py.Group:
    """``parent[name]``, read as the subgroup the caller's layout puts there."""
    return cast(h5py.Group, parent[name])


def h5_dataset(parent: h5py.File | h5py.Group, name: str) -> h5py.Dataset:
    """``parent[name]``, read as the dataset the caller's layout puts there."""
    return cast(h5py.Dataset, parent[name])
