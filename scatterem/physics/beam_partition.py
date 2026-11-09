"""Slicing of a periodic cell along the beam, with exact fractional edges.

Literature: the multislice method cuts the cell into thin slices along the
beam and projects the atoms of each slice (Kirkland 2020, *Advanced Computing
in Electron Microscopy*, 3rd ed., ch. 6). Written for scatterem from this
literature.

Units: slice edges and beam coordinates are fractional coordinates of the
box [dimensionless]. Slice lengths are in [Angstrom].

A slicing is a :class:`BeamPartition`: ``S + 1`` strictly increasing edges
``e_0 < ... < e_S`` in ``[0, 1]``, stored as exact ``fractions.Fraction``.
Slice ``k`` is the half-open interval ``[e_k, e_{k+1})``. Slice 0 is the
slice that the beam reaches first.
"""

from __future__ import annotations

import logging
import math
import numbers
from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property

import numpy as np
import torch

__all__ = [
    "EDGE_SNAP_ULPS",
    "BeamPartition",
    "edge_snap_margin",
    "species_order",
]

_log = logging.getLogger(__name__)

#: Snap margin of float64 coordinates [fractional]: 1e-9 of the cell depth
#: (the v0.3 value). Reason: coordinates from files carry fewer decimals
#: than float64. A coordinate written
#: with 12 decimals, 0.333333333333, is 3.3e-13 from 1/3. In length the
#: margin is 1e-9 * depth, far below any distance between atomic layers.
EDGE_SNAP_RELATIVE = 1e-9

#: Lower bound of the snap margin in units of the machine epsilon of the
#: coordinate dtype. Reason: a computed float32 coordinate such as
#: ``float32(0.7)`` is 1.2e-8 below the edge 7/10, more than
#: ``EDGE_SNAP_RELATIVE``. 512 ``eps`` covers the few ``eps`` of such an
#: error with a factor of more than 10. It sets the float32 margin and the
#: end tolerance of :meth:`BeamPartition.from_thicknesses`.
EDGE_SNAP_ULPS = 512


def edge_snap_margin(dtype: torch.dtype = torch.float64) -> float:
    """Return the snap margin ``mu`` for coordinates of one dtype.

    Parameters
    ----------
    dtype : torch.dtype, optional
        Real floating dtype of the coordinate source. Default float64.

    Returns
    -------
    float
        ``max(EDGE_SNAP_RELATIVE, EDGE_SNAP_ULPS * torch.finfo(dtype).eps)``
        [dimensionless]: ``1e-9`` for float64 and ``6.10e-5`` for float32.
        For cells deeper than about 100 Angstrom, use float64 coordinates:
        the float32 margin is ``6.1e-5 * depth``.

    Raises
    ------
    TypeError
        If ``dtype`` is not a real floating torch dtype.
    """
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise TypeError(f"dtype: must be a real floating torch dtype; got {dtype!r}")
    return max(EDGE_SNAP_RELATIVE, EDGE_SNAP_ULPS * torch.finfo(dtype).eps)


def _exact(value, name: str) -> Fraction:
    """Convert one real number exactly to a Fraction."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name}: must be a real number; got {value!r}")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, numbers.Integral):
        return Fraction(int(value))
    as_float = float(value)
    if not math.isfinite(as_float):
        raise ValueError(f"{name}: must be finite; got {value!r}")
    if isinstance(value, (float, np.floating)):
        return Fraction(as_float)
    return Fraction(value)  # pyright: ignore[reportCallIssue, reportArgumentType]


def _snap_dtype_of(values) -> torch.dtype:
    """Return the dtype that sets the snap margin of a coordinate source."""
    dt = values.dtype
    if isinstance(dt, torch.dtype):
        return torch.float32 if dt in (torch.float32, torch.float16) else torch.float64
    return torch.float32 if dt in (np.float32, np.float16) else torch.float64


@dataclass(frozen=True)
class BeamPartition:
    """Slicing of the cell along the beam, with exact fractional edges.

    Parameters
    ----------
    edges : sequence of numbers.Real
        ``S + 1`` strictly increasing edges in ``[0, 1]`` [fractional]. Each
        value is converted exactly with ``Fraction``; a float converts
        exactly. Python and numpy reals and ``Fraction`` are accepted.

    Raises
    ------
    TypeError
        If an edge is not a real number (also ``bool``, ``str``, complex).
    ValueError
        If there are fewer than 2 edges, an edge is not finite, the edges
        are not strictly increasing, ``e_0 < 0`` or ``e_S > 1``.

    Notes
    -----
    Slice ``k`` is ``[e_k, e_{k+1})``. Exact edges give the rule
    "slice ``k`` is ``[k/n, (k+1)/n)``" one exact meaning, and the slicing
    does not depend on the box depth. A slice of zero thickness is refused:
    it has no propagation length and holds no atom under the half-open rule.
    The object is hashable; partitions with equal edges are equal.
    """

    edges: tuple[Fraction, ...]

    def __post_init__(self) -> None:
        if isinstance(self.edges, (str, bytes)):
            raise TypeError(f"edges: must be a sequence of reals; got {self.edges!r}")
        try:
            raw = list(self.edges)
        except TypeError:
            raise TypeError(
                f"edges: must be a sequence of reals; got {self.edges!r}"
            ) from None
        if len(raw) < 2:
            raise ValueError(f"edges: need at least 2 values; got {len(raw)}")
        exact = tuple(_exact(v, f"edges[{i}]") for i, v in enumerate(raw))
        for k in range(len(exact) - 1):
            if exact[k + 1] <= exact[k]:
                raise ValueError(
                    f"edges: must be strictly increasing; edges[{k + 1}] = "
                    f"{exact[k + 1]} <= edges[{k}] = {exact[k]}"
                )
        if exact[0] < 0 or exact[-1] > 1:
            raise ValueError(
                f"edges: must lie in [0, 1]; got first {exact[0]}, last {exact[-1]}"
            )
        object.__setattr__(self, "edges", exact)

    # ------------------------------------------------------------ makers
    @classmethod
    def uniform(cls, count) -> BeamPartition:
        """Return the slicing of ``count`` equal slices of the cell.

        Parameters
        ----------
        count : int
            Number of slices, at least 1. Python or numpy integer, not bool.

        Returns
        -------
        BeamPartition
            Edges ``Fraction(k, count)`` for ``k = 0 .. count``.

        Raises
        ------
        TypeError
            If ``count`` is not an integer.
        ValueError
            If ``count < 1``.
        """
        if isinstance(count, (bool, np.bool_)) or not isinstance(
            count, numbers.Integral
        ):
            raise TypeError(f"count: must be an integer; got {count!r}")
        n = int(count)
        if n < 1:
            raise ValueError(f"count: must be at least 1; got {n}")
        return cls(tuple(Fraction(k, n) for k in range(n + 1)))

    @classmethod
    def from_thicknesses(cls, thicknesses, depth) -> BeamPartition:
        """Return the slicing with given slice lengths along the beam.

        Parameters
        ----------
        thicknesses : sequence of float
            Slice lengths [Angstrom], each finite and > 0.
        depth : float
            Box length along the beam ``sides[2]`` [Angstrom], finite, > 0.

        Returns
        -------
        BeamPartition
            Edges are the exact partial sums of ``Fraction(t_k)`` divided by
            ``Fraction(depth)``, starting at 0. A last edge within
            ``EDGE_SNAP_ULPS * 2**-53`` of 1 becomes exactly 1, because a sum
            of float lengths carries rounding.

        Raises
        ------
        TypeError
            If a value is not a real number.
        ValueError
            If a value is not finite or not > 0, or if the edges leave
            ``[0, 1]``.
        """
        d = _exact(depth, "depth")
        if d <= 0:
            raise ValueError(f"depth: must be > 0; got {depth!r}")
        ts = [_exact(t, f"thicknesses[{i}]") for i, t in enumerate(thicknesses)]
        if not ts:
            raise ValueError("thicknesses: need at least one slice length")
        for i, t in enumerate(ts):
            if t <= 0:
                raise ValueError(f"thicknesses[{i}]: must be > 0; got {float(t)!r}")
        edges = [Fraction(0)]
        for t in ts:
            edges.append(edges[-1] + t / d)
        if abs(edges[-1] - 1) <= Fraction(EDGE_SNAP_ULPS, 2**53):
            edges[-1] = Fraction(1)
        return cls(tuple(edges))

    # ------------------------------------------------------------ properties
    @property
    def count(self) -> int:
        """Number of slices ``S``."""
        return len(self.edges) - 1

    @property
    def covers_cell(self) -> bool:
        """True if ``e_0 == 0`` and ``e_S == 1``."""
        return self.edges[0] == 0 and self.edges[-1] == 1

    @cached_property
    def _float_edges(self) -> np.ndarray:
        return np.array([float(e) for e in self.edges], dtype=np.float64)

    def edge_values(self, dtype=torch.float64, device=None) -> torch.Tensor:
        """Return the edges rounded once to ``dtype``.

        Parameters
        ----------
        dtype : torch.dtype, optional
            Real floating dtype. Default float64.
        device : torch.device or str, optional
            Device. Default the CPU.

        Returns
        -------
        torch.Tensor
            Shape ``(S + 1,)`` [fractional]. The float64 values are the
            correctly rounded edges; other dtypes round these once more.
        """
        return torch.tensor(self._float_edges, dtype=dtype, device=device)

    def thicknesses(self, depth) -> tuple[float, ...]:
        """Return the slice lengths for a box of length ``depth``.

        Parameters
        ----------
        depth : float
            Box length along the beam [Angstrom], finite and > 0.

        Returns
        -------
        tuple of float
            ``float((e_{k+1} - e_k) * Fraction(depth))`` [Angstrom], each
            rounded once.
        """
        d = _exact(depth, "depth")
        if d <= 0:
            raise ValueError(f"depth: must be > 0; got {depth!r}")
        e = self.edges
        return tuple(float((e[k + 1] - e[k]) * d) for k in range(self.count))

    def subrange(self, first: int, stop: int) -> BeamPartition:
        """Return the partition of slices ``first .. stop - 1``.

        Parameters
        ----------
        first, stop : int
            Slice range, ``0 <= first < stop <= S``.

        Returns
        -------
        BeamPartition
            Edges ``e_first .. e_stop``.

        Raises
        ------
        ValueError
            If the range is empty or outside ``0 .. S``.
        """
        for name, v in (("first", first), ("stop", stop)):
            if isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Integral):
                raise TypeError(f"{name}: must be an integer; got {v!r}")
        if not 0 <= int(first) < int(stop) <= self.count:
            raise ValueError(
                f"first, stop: need 0 <= first < stop <= {self.count}; "
                f"got {first}, {stop}"
            )
        return BeamPartition(self.edges[int(first) : int(stop) + 1])

    # ------------------------------------------------------------ slice lookup
    def _locate(self, z: torch.Tensor, margin: float, snap: bool) -> torch.Tensor:
        """Return ``k`` in ``-1 .. S``: -1 below ``e_0``, ``S`` at/after ``e_S``.

        The lookup uses the float64 edges. A coordinate equal to a float edge
        goes to the slice after it, also when the float edge is below the
        exact edge: a computed layer ``k / n`` then lands in slice ``k``.
        """
        e = torch.as_tensor(self._float_edges, device=z.device)
        s = self.count
        k = torch.searchsorted(e, z.contiguous(), right=True) - 1
        if snap:
            inner = (k >= 0) & (k < s - 1)
            nxt = e[(k + 1).clamp(0, s)]
            k = k + (inner & (nxt - z <= margin)).to(k.dtype)
        return k

    def _prepare(self, beam_fraction, margin_dtype):
        """Return a detached float64 tensor, a numpy flag and the snap dtype."""
        if margin_dtype is not None:
            edge_snap_margin(margin_dtype)
        if isinstance(beam_fraction, torch.Tensor):
            if beam_fraction.is_complex() or beam_fraction.dtype == torch.bool:
                raise TypeError(
                    f"beam_fraction: must be real; got dtype {beam_fraction.dtype}"
                )
            snap_dtype = margin_dtype or _snap_dtype_of(beam_fraction)
            return beam_fraction.detach().to(torch.float64), False, snap_dtype
        arr = np.asarray(beam_fraction)
        if arr.dtype.kind not in "iuf":
            raise TypeError(f"beam_fraction: must be real; got dtype {arr.dtype}")
        snap_dtype = margin_dtype or _snap_dtype_of(arr)
        return torch.from_numpy(arr.astype(np.float64)), True, snap_dtype

    def bin_index(self, beam_fraction, *, edge_snap: bool = True, margin_dtype=None):
        """Return the slice of each atom in the periodic cell.

        Rules, in this order:

        1. Reduce into ``[0, 1)``: ``r = z - floor(z)``; a rounded ``r == 1``
           becomes 0 (the cell is periodic along the beam).
        2. Interval test on the float64 edges ``f_k = float(e_k)``: slice
           ``k`` with ``f_k <= r < f_{k+1}``. A coordinate equal to an edge
           goes to the slice after the edge (the upstream end is closed).
           This also holds when ``f_k`` is below ``e_k``, so the computed
           layer ``k / n`` is in slice ``k`` of ``uniform(n)``.
        3. With ``edge_snap``, a coordinate with ``e_{k+1} - mu <= r`` for an
           inner edge (``k + 1 < S``) goes to slice ``k + 1``, with
           ``mu = edge_snap_margin(dtype)`` of the input dtype. The last
           edge never moves.
        4. A reduced coordinate below ``e_0`` or at/after ``e_S`` (only for a
           partition that does not cover the cell) gives ``-1``.

        Parameters
        ----------
        beam_fraction : numpy.ndarray or torch.Tensor
            Fractional beam coordinates, any shape, real.
        edge_snap : bool, optional
            Snap atomic layers that lie less than ``mu`` before an inner edge
            into the slice after it. Default True: a layer that a file or a
            computation puts a little before its edge is then in the slice
            of exact arithmetic.
        margin_dtype : torch.dtype, optional
            The dtype of the coordinate source that sets the snap margin, when
            the values come converted (for example float32 data in a float64
            tensor). Default: the dtype of ``beam_fraction``.

        Returns
        -------
        numpy.ndarray or torch.Tensor
            The int64 slice indices of the input shape: numpy for numpy input, a
            tensor on the input device for tensor input. No autograd.
        """
        z, as_numpy, snap_dtype = self._prepare(beam_fraction, margin_dtype)
        _check_switch(edge_snap)
        r = z - torch.floor(z)
        r = torch.where(r >= 1.0, torch.zeros_like(r), r)
        k = self._locate(r.reshape(-1), edge_snap_margin(snap_dtype), edge_snap)
        k = torch.where((k < 0) | (k >= self.count), torch.full_like(k, -1), k)
        k = k.reshape(z.shape)
        return k.numpy() if as_numpy else k

    def bin_index_clamped(
        self, beam_fraction, *, edge_snap: bool = True, margin_dtype=None
    ):
        """Return the slice of each source term, without periodic reduction.

        A coordinate with ``z < e_0`` gives slice 0; a coordinate with
        ``z >= e_S`` gives slice ``S - 1``. Other coordinates follow the
        interval test and the edge snap of :meth:`bin_index`.

        Parameters
        ----------
        beam_fraction : numpy.ndarray or torch.Tensor
            Fractional beam coordinates, any shape, real.
        edge_snap : bool, optional
            As in :meth:`bin_index`. Default True.
        margin_dtype : torch.dtype, optional
            As in :meth:`bin_index`.

        Returns
        -------
        numpy.ndarray or torch.Tensor
            The int64 slice indices in ``0 .. S - 1``, as in :meth:`bin_index`.
        """
        z, as_numpy, snap_dtype = self._prepare(beam_fraction, margin_dtype)
        _check_switch(edge_snap)
        k = self._locate(z.reshape(-1), edge_snap_margin(snap_dtype), edge_snap)
        k = k.clamp(0, self.count - 1).reshape(z.shape)
        return k.numpy() if as_numpy else k


def _check_switch(value) -> None:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"edge_snap: must be a bool; got {value!r}")


def species_order(atom_set) -> tuple[int, ...]:
    """Return the distinct atomic numbers of an atom set in ascending order.

    Parameters
    ----------
    atom_set : AtomSet
        The structure.

    Returns
    -------
    tuple of int
        Ascending distinct ``Z``; ``()`` for an empty set. The order depends
        only on the content, so a permuted atom set gives the same sequence.
        The rows of per-element form spectra follow this sequence.
    """
    z = np.asarray(atom_set.element_z)
    return tuple(int(v) for v in np.unique(z))
