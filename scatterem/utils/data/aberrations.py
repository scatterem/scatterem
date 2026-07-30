"""Container for the lens aberration coefficients of a STEM probe.

The canonical storage is a flat, 12-element **Cartesian** Zernike coefficient
vector (``Aberrations.array``) in the layout consumed by
:func:`scatterem.utils.aberration_basis.cartesian_chi` and its NVIDIA Warp
counterpart :func:`scatterem.utils.warp.aberrations.aberration_function_cartesian`::

    index  name   description
    -----  -----  ------------------------------------------------
      0    C10    defocus (C1); ``defocus == -C10``
      1    C12a   two-fold astigmatism
      2    C12b
      3    C21a   axial coma
      4    C21b
      5    C23a   three-fold astigmatism (trefoil)
      6    C23b
      7    C30    spherical aberration (C3, Cs)
      8    C32a   axial star aberration
      9    C32b
     10    C34a   four-fold astigmatism (quadrafoil)
     11    C34b

On top of that vector this class adds named key/attribute access, polar
``(magnitude, angle)`` conversions, device/dtype helpers and human-readable
printing, while remaining a thin wrapper so existing ``aberrations.array``
access keeps working unchanged.
"""

from __future__ import annotations

import math
from typing import Iterator, Mapping

import torch
from torch import Tensor

from scatterem.io.store import Serializable



#: Names of the 12 Cartesian coefficients, in storage order.
CARTESIAN_NAMES: tuple[str, ...] = (
    "C10",
    "C12a",
    "C12b",
    "C21a",
    "C21b",
    "C23a",
    "C23b",
    "C30",
    "C32a",
    "C32b",
    "C34a",
    "C34b",
)

#: Names of the 12 polar coefficients, paired with :data:`CARTESIAN_NAMES` by
#: :func:`scatterem.utils.aberration_basis.cartesian_to_polar`.
POLAR_NAMES: tuple[str, ...] = (
    "C10",
    "C12",
    "C21",
    "C23",
    "C30",
    "C32",
    "C34",
    "phi12",
    "phi21",
    "phi23",
    "phi32",
    "phi34",
)

_NAME_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(CARTESIAN_NAMES)}

# Convenient human aliases that resolve to a single Cartesian slot.
_ALIASES: dict[str, str] = {
    "C1": "C10",
    "C3": "C30",
    "Cs": "C30",
}

# Polar magnitude/angle groups: (magnitude, angle, fold m, (a_index, b_index)).
# The Cartesian pair relates to the polar form by
#   a = C * cos(m * phi),  b = C * sin(m * phi)
# so   C = hypot(a, b),  phi = atan2(b, a) / m.
_POLAR_GROUPS: tuple[tuple[str, str, int, tuple[int, int]], ...] = (
    ("C12", "phi12", 2, (1, 2)),
    ("C21", "phi21", 1, (3, 4)),
    ("C23", "phi23", 3, (5, 6)),
    ("C32", "phi32", 2, (8, 9)),
    ("C34", "phi34", 4, (10, 11)),
)


# Set of every string key understood by attribute access.
_ATTR_KEYS: frozenset[str] = frozenset(
    set(CARTESIAN_NAMES) | set(_ALIASES) | {"defocus"}
)


class Aberrations(Serializable):
    SERIAL_FIELDS = ("array",)

    """Lens aberration coefficients with named access and convention helpers.

    Parameters
    ----------
    array:
        Optional 12-element 1-D tensor (or anything :func:`torch.as_tensor`
        accepts) in Cartesian layout. Defaults to a fresh zero vector.
    **named:
        Coefficients to set by name after construction, e.g.
        ``Aberrations(C10=-150.0, C30=1.2e7)`` or ``Aberrations(defocus=150.0)``.

    Examples
    --------
    >>> ab = Aberrations(defocus=150.0, C30=1e7)
    >>> ab["C10"], ab.defocus
    (tensor(-150.), tensor(150.))
    >>> ab.to_polar()["C30"]
    10000000.0
    """

    #: Number of Cartesian coefficients.
    N: int = 12

    # Equal-valued instances may differ in identity; treat as unhashable since
    # the underlying tensor is mutable.
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, array: Tensor | None = None, **named: float) -> None:
        if array is None:
            array = torch.zeros(self.N)
        elif not isinstance(array, Tensor):
            array = torch.as_tensor(array, dtype=torch.float32)

        if array.ndim != 1 or array.shape[0] != self.N:
            raise ValueError(
                f"Aberrations expects a 1-D tensor of length {self.N}, "
                f"got shape {tuple(array.shape)}."
            )

        # Bypass our own __setattr__ for the backing store.
        object.__setattr__(self, "array", array)

        for key, value in named.items():
            self[key] = value

    # -- factories ---------------------------------------------------------

    @classmethod
    def zeros(
        cls,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "Aberrations":
        """Return an all-zero :class:`Aberrations` on the requested device."""
        return cls(torch.zeros(cls.N, device=device, dtype=dtype))

    @classmethod
    def from_dict(
        cls,
        coefficients: Mapping[str, float],
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "Aberrations":
        """Build from a mapping of Cartesian names (or aliases) to values."""
        ab = cls.zeros(device=device, dtype=dtype)
        for key, value in coefficients.items():
            ab[key] = value
        return ab

    @classmethod
    def from_polar(
        cls,
        polar: Mapping[str, float],
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "Aberrations":
        """Build from polar ``(magnitude, angle[rad])`` coefficients.

        Recognised keys are the magnitudes/angles in :data:`POLAR_NAMES`
        (``C10``, ``C12``, ``phi12``, ...) plus the ``defocus``/``Cs`` aliases.
        Missing keys default to zero.
        """
        get = lambda k: float(polar.get(k, 0.0))  # noqa: E731
        arr = torch.zeros(cls.N, dtype=dtype)

        arr[0] = get("C10")
        if "defocus" in polar:
            arr[0] = -float(polar["defocus"])
        arr[7] = get("C30")
        if "Cs" in polar:
            arr[7] = float(polar["Cs"])

        for mag_name, ang_name, fold, (ai, bi) in _POLAR_GROUPS:
            magnitude = get(mag_name)
            angle = get(ang_name)
            arr[ai] = magnitude * math.cos(fold * angle)
            arr[bi] = magnitude * math.sin(fold * angle)

        if device is not None:
            arr = arr.to(device)
        return cls(arr)

    @classmethod
    def from_polar_array(
        cls, polar_array: Tensor, **kwargs
    ) -> "Aberrations":
        """Build from the 12-element polar vector ordered as :data:`POLAR_NAMES`."""
        polar_array = torch.as_tensor(polar_array)
        if polar_array.shape[0] != cls.N:
            raise ValueError(
                f"polar_array must have length {cls.N}, got {polar_array.shape[0]}."
            )
        mapping = {
            name: float(polar_array[i]) for i, name in enumerate(POLAR_NAMES)
        }
        return cls.from_polar(mapping, **kwargs)

    # -- key / attribute access -------------------------------------------

    @classmethod
    def _index(cls, name: str) -> int:
        canonical = _ALIASES.get(name, name)
        try:
            return _NAME_TO_INDEX[canonical]
        except KeyError:
            valid = ", ".join(CARTESIAN_NAMES)
            aliases = ", ".join((*_ALIASES, "defocus"))
            raise KeyError(
                f"Unknown aberration coefficient {name!r}. "
                f"Valid names: {valid} (aliases: {aliases})."
            ) from None

    def __getitem__(self, key: str | int | slice | Tensor):
        if isinstance(key, str):
            if key == "defocus":
                return -self.array[0]
            return self.array[self._index(key)]
        return self.array[key]

    def __setitem__(self, key: str | int | slice | Tensor, value) -> None:
        if isinstance(key, str):
            if key == "defocus":
                self.array[0] = -_as_scalar(value)
                return
            self.array[self._index(key)] = _as_scalar(value)
            return
        self.array[key] = value

    def __getattr__(self, name: str):
        # Only reached when normal attribute lookup fails (e.g. ``array`` is
        # always found first). Guard against dunder probing by copy/pickle.
        if name in _ATTR_KEYS:
            return self[name]
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )

    def __setattr__(self, name: str, value) -> None:
        if name in _ATTR_KEYS:
            self[name] = value
        else:
            object.__setattr__(self, name, value)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and (
            name in _ATTR_KEYS or _ALIASES.get(name, name) in _NAME_TO_INDEX
        )

    def keys(self) -> tuple[str, ...]:
        """Cartesian coefficient names, in storage order."""
        return CARTESIAN_NAMES

    def values(self) -> Iterator[Tensor]:
        return iter(self.array)

    def items(self) -> Iterator[tuple[str, Tensor]]:
        return zip(CARTESIAN_NAMES, self.array)

    def nonzero_names(self) -> tuple[str, ...]:
        """Names of the Cartesian coefficients that are currently non-zero."""
        return tuple(
            name for name, value in self.items() if float(value) != 0.0
        )

    # -- convention conversions -------------------------------------------

    def to_cartesian_array(self) -> Tensor:
        """Return the underlying 12-element Cartesian vector (a view)."""
        return self.array

    def to_polar_array(self) -> Tensor:
        """Return the 12-element polar vector ordered as :data:`POLAR_NAMES`.

        Magnitudes are non-negative; angles are in radians, following
        :func:`scatterem.utils.aberration_basis.cartesian_to_polar`.
        """
        a = self.array
        out = torch.zeros_like(a)
        # Radial terms map straight across.
        out[0] = a[0]  # C10
        out[4] = a[7]  # C30
        polar_index = {name: i for i, name in enumerate(POLAR_NAMES)}
        for mag_name, ang_name, fold, (ai, bi) in _POLAR_GROUPS:
            magnitude = torch.hypot(a[ai], a[bi])
            angle = torch.atan2(a[bi], a[ai]) / fold
            out[polar_index[mag_name]] = magnitude
            out[polar_index[ang_name]] = angle
        return out

    def to_polar(self) -> dict[str, float]:
        """Return polar ``(magnitude, angle[rad])`` coefficients as a dict."""
        polar = self.to_polar_array()
        return {name: float(polar[i]) for i, name in enumerate(POLAR_NAMES)}

    def to_dict(self, nonzero_only: bool = False) -> dict[str, float]:
        """Return the Cartesian coefficients as a ``{name: value}`` dict."""
        out = {name: float(value) for name, value in self.items()}
        if nonzero_only:
            out = {k: v for k, v in out.items() if v != 0.0}
        return out

    # -- evaluation --------------------------------------------------------

    def evaluate(self, qy: Tensor, qx: Tensor, wavelength: float) -> Tensor:
        """Evaluate the aberration phase ``chi`` on a reciprocal-space grid.

        Thin wrapper around
        :func:`scatterem.utils.aberration_basis.cartesian_chi` (imported lazily
        to avoid an import cycle).
        """
        from scatterem.utils.aberration_basis import cartesian_chi

        return cartesian_chi(qy, qx, wavelength, self.array)

    # -- tensor-like helpers ----------------------------------------------

    @property
    def device(self) -> torch.device:
        return self.array.device

    @property
    def dtype(self) -> torch.dtype:
        return self.array.dtype

    @property
    def is_zero(self) -> bool:
        """``True`` when every coefficient is exactly zero."""
        return bool(torch.all(self.array == 0).item())

    def to(self, *args, **kwargs) -> "Aberrations":
        """Return a copy with ``array`` moved/cast via :meth:`torch.Tensor.to`."""
        return Aberrations(self.array.to(*args, **kwargs))

    def cpu(self) -> "Aberrations":
        return Aberrations(self.array.cpu())

    def cuda(self) -> "Aberrations":
        return Aberrations(self.array.cuda())

    def clone(self) -> "Aberrations":
        return Aberrations(self.array.clone())

    def detach(self) -> "Aberrations":
        return Aberrations(self.array.detach())

    def numpy(self):
        return self.array.detach().cpu().numpy()

    def __array__(self, dtype=None):
        arr = self.numpy()
        return arr.astype(dtype) if dtype is not None else arr

    def __len__(self) -> int:
        return self.N

    def __iter__(self) -> Iterator[Tensor]:
        return iter(self.array)

    def allclose(self, other: "Aberrations | Tensor", **kwargs) -> bool:
        other_array = other.array if isinstance(other, Aberrations) else other
        return bool(torch.allclose(self.array, torch.as_tensor(other_array), **kwargs))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Aberrations):
            return bool(torch.equal(self.array, other.array))
        if isinstance(other, Tensor):
            return bool(torch.equal(self.array, other))
        return NotImplemented

    # -- arithmetic --------------------------------------------------------

    def __add__(self, other: "Aberrations") -> "Aberrations":
        return Aberrations(self.array + _other_array(other))

    def __sub__(self, other: "Aberrations") -> "Aberrations":
        return Aberrations(self.array - _other_array(other))

    def __mul__(self, scalar: float) -> "Aberrations":
        return Aberrations(self.array * scalar)

    __rmul__ = __mul__

    def __copy__(self) -> "Aberrations":
        return self.clone()

    # -- printing ----------------------------------------------------------

    def __repr__(self) -> str:
        values = ", ".join(f"{float(v):.4g}" for v in self.array)
        return f"{type(self).__name__}([{values}])"

    def __str__(self) -> str:
        lines = [
            f"{type(self).__name__} (Cartesian, {self.N} coefficients, "
            f"device={self.device}, dtype={self.dtype})"
        ]
        if self.is_zero:
            lines.append("  (all coefficients zero)")
            return "\n".join(lines)

        polar = self.to_polar()
        # Defocus first, since it is the most commonly inspected quantity.
        lines.append(f"  defocus           : {float(self['defocus']):+.4g}")
        radial = {"C10": "defocus C1", "C30": "spherical Cs"}
        for name, label in radial.items():
            value = float(self[name])
            if value != 0.0:
                lines.append(f"  {name:<5} ({label:<11}): {value:+.6g}")
        descriptions = {
            "C12": "astigmatism",
            "C21": "coma",
            "C23": "trefoil",
            "C32": "star",
            "C34": "quadrafoil",
        }
        for mag_name, ang_name, _, _ in _POLAR_GROUPS:
            magnitude = polar[mag_name]
            if magnitude != 0.0:
                angle_deg = math.degrees(polar[ang_name])
                label = descriptions[mag_name]
                lines.append(
                    f"  {mag_name:<5} ({label:<11}): "
                    f"{magnitude:.6g} @ {angle_deg:+.3f} deg"
                )
        return "\n".join(lines)


def _as_scalar(value):
    """Coerce a Python/tensor scalar to a form assignable into ``array[idx]``."""
    if isinstance(value, Tensor):
        return value
    return float(value)


def _other_array(other: "Aberrations | Tensor") -> Tensor:
    if isinstance(other, Aberrations):
        return other.array
    return torch.as_tensor(other)
