"""Container for the lens aberration coefficients of a STEM probe.

The canonical storage is a flat, 12-element **Cartesian** Zernike coefficient
vector (``Aberrations.array``): a lens vector of :func:`scatterem.physics.optics.lens_wavefront`.
The names are the Krivanek labels of
:data:`scatterem.physics.optics.LENS_LAYOUT`, in the same order::

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
from collections.abc import Iterator, Mapping

import torch
from torch import Tensor

from scatterem.physics.optics import (
    AMPLITUDE_LAYOUT,
    LENS_LAYOUT,
    lens_from_amplitude,
    lens_to_amplitude,
)
from scatterem.utils.store import Serializable



#: Names of the 12 Cartesian coefficients, in storage order. This is the
#: tuple :data:`scatterem.physics.optics.LENS_LAYOUT` itself, so the package
#: has one set of aberration labels.
CARTESIAN_NAMES: tuple[str, ...] = LENS_LAYOUT

#: Names of the 12 polar coefficients: the tuple
#: :data:`scatterem.physics.optics.AMPLITUDE_LAYOUT` itself (seven magnitudes,
#: then five angles). :meth:`Aberrations.to_polar_array` uses this order.
POLAR_NAMES: tuple[str, ...] = AMPLITUDE_LAYOUT

_NAME_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(CARTESIAN_NAMES)}

# Convenient human aliases that resolve to a single Cartesian slot.
_ALIASES: dict[str, str] = {
    "C1": "C10",
    "C3": "C30",
    "Cs": "C30",
}

# Polar magnitude/angle pairs with m > 0, for printing:
# (magnitude label, angle label).
_POLAR_PAIRS: tuple[tuple[str, str], ...] = tuple(
    (f"C{label[3:]}", label) for label in AMPLITUDE_LAYOUT if label.startswith("phi")
)

# Set of every string key understood by attribute access.
_ATTR_KEYS: frozenset[str] = frozenset(
    set(CARTESIAN_NAMES) | set(_ALIASES) | {"defocus"}
)


class Aberrations(Serializable):
    """Lens aberration coefficients with named access and convention helpers.

    Parameters
    ----------
    array : Tensor, optional
        12-element 1-D tensor (or anything :func:`torch.as_tensor` accepts) in
        Cartesian layout. Defaults to a fresh zero vector.
    **named : float
        Coefficients to set by name after construction, e.g.
        ``Aberrations(C10=-150.0, C30=1.2e7)`` or ``Aberrations(defocus=150.0)``.

    Examples
    --------
    >>> from scatterem.data.aberrations import Aberrations
    >>> ab = Aberrations(defocus=150.0, C30=1e7)
    >>> ab["C10"], ab.defocus
    (tensor(-150.), tensor(150.))
    >>> ab.to_polar()["C30"]
    10000000.0
    """

    #: Number of Cartesian coefficients.
    N: int = 12
    SERIAL_FIELDS = ("array",)

    # Equal-valued instances may differ in identity; treat as unhashable since
    # the underlying tensor is mutable.
    __hash__ = None

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
    ) -> Aberrations:
        """Return an all-zero :class:`Aberrations` on the requested device.

        Parameters
        ----------
        device : torch.device or str, optional
            Device to allocate the backing tensor on. Defaults to the current
            device.
        dtype : torch.dtype, optional
            Floating-point dtype of the backing tensor. Defaults to
            ``torch.float32``.

        Returns
        -------
        Aberrations
            All-zero aberration coefficients.
        """
        return cls(torch.zeros(cls.N, device=device, dtype=dtype))

    @classmethod
    def from_dict(
        cls,
        coefficients: Mapping[str, float],
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> Aberrations:
        """Build from a mapping of Cartesian names (or aliases) to values.

        Parameters
        ----------
        coefficients : Mapping[str, float]
            Cartesian coefficient names (or aliases such as ``C1``, ``Cs``) to
            values. Missing coefficients default to zero.
        device : torch.device or str, optional
            Device to allocate the backing tensor on. Defaults to the current
            device.
        dtype : torch.dtype, optional
            Floating-point dtype of the backing tensor. Defaults to
            ``torch.float32``.

        Returns
        -------
        Aberrations
            Instance with the given coefficients set.
        """
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
    ) -> Aberrations:
        """Build from polar ``(magnitude, angle[rad])`` coefficients.

        Recognised keys are the magnitudes/angles in ``POLAR_NAMES``
        (``C10``, ``C12``, ..., ``phi12``, ...) plus the ``defocus``/``Cs``
        aliases. Missing keys default to zero. The conversion is
        :func:`scatterem.physics.optics.lens_from_amplitude`, so
        ``a = C cos(m phi)`` and ``b = C sin(m phi)`` for any real angle.

        Parameters
        ----------
        polar : Mapping[str, float]
            Polar coefficient names to values (magnitudes and angles in
            radians), plus the optional ``defocus``/``Cs`` aliases.
        device : torch.device or str, optional
            Device to allocate the backing tensor on. Defaults to the current
            device.
        dtype : torch.dtype, optional
            Floating-point dtype of the backing tensor. Defaults to
            ``torch.float32``.

        Returns
        -------
        Aberrations
            Instance built from the polar coefficients.
        """
        amp = [float(polar.get(name, 0.0)) for name in AMPLITUDE_LAYOUT]
        if "defocus" in polar:
            amp[AMPLITUDE_LAYOUT.index("C10")] = -float(polar["defocus"])
        if "Cs" in polar:
            amp[AMPLITUDE_LAYOUT.index("C30")] = float(polar["Cs"])
        lens = lens_from_amplitude(torch.tensor(amp, dtype=torch.float64))
        return cls(lens.to(device=device, dtype=dtype))

    @classmethod
    def from_polar_array(cls, polar_array: Tensor, **kwargs) -> Aberrations:
        """Build from the 12-element polar vector ordered as ``POLAR_NAMES``.

        Parameters
        ----------
        polar_array : Tensor
            12-element 1-D tensor of polar coefficients, ordered as
            ``POLAR_NAMES``. The conversion is
            :func:`scatterem.physics.optics.lens_from_amplitude`.
        **kwargs
            ``device`` and ``dtype`` of the result, as in :meth:`from_polar`.

        Returns
        -------
        Aberrations
            Instance built from the polar coefficients.
        """
        polar_array = torch.as_tensor(polar_array)
        if polar_array.ndim != 1 or polar_array.shape[0] != cls.N:
            raise ValueError(
                f"polar_array must have length {cls.N}, got shape "
                f"{tuple(polar_array.shape)}."
            )
        device = kwargs.pop("device", None)
        dtype = kwargs.pop("dtype", torch.float32)
        if kwargs:
            raise TypeError(f"unexpected keyword arguments: {sorted(kwargs)}")
        if not polar_array.is_floating_point():
            polar_array = polar_array.to(torch.float64)
        lens = lens_from_amplitude(polar_array)
        if device is None:
            device = polar_array.device
        return cls(lens.to(device=device, dtype=dtype))

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
        """Return the Cartesian coefficient names, in storage order.

        Returns
        -------
        tuple of str
            The 12 Cartesian coefficient names, in storage order.
        """
        return CARTESIAN_NAMES

    def values(self) -> Iterator[Tensor]:
        """Iterate over the Cartesian coefficient values, in storage order.

        Returns
        -------
        Iterator[Tensor]
            Iterator over the 12 scalar coefficient tensors.
        """
        return iter(self.array)

    def items(self) -> Iterator[tuple[str, Tensor]]:
        """Iterate over ``(name, value)`` pairs, in storage order.

        Returns
        -------
        Iterator[tuple[str, Tensor]]
            Iterator over ``(name, value)`` pairs.
        """
        return zip(CARTESIAN_NAMES, self.array, strict=True)

    def nonzero_names(self) -> tuple[str, ...]:
        """Return names of the Cartesian coefficients that are non-zero.

        Returns
        -------
        tuple of str
            Names of the Cartesian coefficients currently non-zero.
        """
        return tuple(name for name, value in self.items() if float(value) != 0.0)

    # -- convention conversions -------------------------------------------

    def to_cartesian_array(self) -> Tensor:
        """Return the underlying 12-element Cartesian vector (a view).

        Returns
        -------
        Tensor
            12-element 1-D Cartesian coefficient vector.
        """
        return self.array

    def to_polar_array(self) -> Tensor:
        """Return the 12-element polar vector ordered as ``POLAR_NAMES``.

        This is :func:`scatterem.physics.optics.lens_to_amplitude` of
        ``array``. Magnitudes are non-negative; an angle of fold ``m`` is
        in radians, in ``[0, 2 pi / m)``.

        Returns
        -------
        Tensor
            12-element 1-D polar coefficient vector, ordered as
            ``POLAR_NAMES``.
        """
        return lens_to_amplitude(self.array)

    def to_polar(self) -> dict[str, float]:
        """Return polar ``(magnitude, angle[rad])`` coefficients as a dict.

        Returns
        -------
        dict of str to float
            Polar coefficient names (``POLAR_NAMES``) mapped to their values.
        """
        polar = self.to_polar_array()
        return {name: float(polar[i]) for i, name in enumerate(POLAR_NAMES)}

    def to_dict(self, nonzero_only: bool = False) -> dict[str, float]:
        """Return the Cartesian coefficients as a ``{name: value}`` dict.

        Parameters
        ----------
        nonzero_only : bool, optional
            If ``True``, omit coefficients that are exactly zero. Default is
            ``False``.

        Returns
        -------
        dict of str to float
            Cartesian coefficient names mapped to their values.
        """
        out = {name: float(value) for name, value in self.items()}
        if nonzero_only:
            out = {k: v for k, v in out.items() if v != 0.0}
        return out

    # -- evaluation --------------------------------------------------------

    def evaluate(self, qy: Tensor, qx: Tensor, wavelength: float) -> Tensor:
        """Evaluate the aberration phase ``chi`` on a reciprocal-space grid.

        Thin wrapper around
        :func:`scatterem.physics.optics.lens_wavefront` (imported lazily
        to avoid an import cycle).

        Parameters
        ----------
        qy : Tensor
            Reciprocal-space y-coordinates.
        qx : Tensor
            Reciprocal-space x-coordinates, broadcastable with `qy`.
        wavelength : float
            Electron wavelength, in the same length unit as ``qy``/``qx``
            are reciprocal to.

        Returns
        -------
        Tensor
            Aberration phase ``chi`` evaluated on the ``qy``/``qx`` grid.
        """
        from scatterem.physics.optics import lens_wavefront

        return lens_wavefront(self.array, qy, qx, lambda_a=wavelength)

    # -- tensor-like helpers ----------------------------------------------

    @property
    def device(self) -> torch.device:
        """Device of the underlying coefficient tensor."""
        return self.array.device

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the underlying coefficient tensor."""
        return self.array.dtype

    @property
    def is_zero(self) -> bool:
        """``True`` when every coefficient is exactly zero."""
        return bool(torch.all(self.array == 0).item())

    def to(self, *args, **kwargs) -> Aberrations:
        """Return a copy with ``array`` moved/cast via :meth:`torch.Tensor.to`.

        Parameters
        ----------
        *args
            Forwarded to :meth:`torch.Tensor.to`.
        **kwargs
            Forwarded to :meth:`torch.Tensor.to`.

        Returns
        -------
        Aberrations
            Copy with the coefficient tensor moved/cast.
        """
        return Aberrations(self.array.to(*args, **kwargs))

    def cpu(self) -> Aberrations:
        """Return a copy with ``array`` moved to CPU.

        Returns
        -------
        Aberrations
            Copy with the coefficient tensor on CPU.
        """
        return Aberrations(self.array.cpu())

    def cuda(self) -> Aberrations:
        """Return a copy with ``array`` moved to the current CUDA device.

        Returns
        -------
        Aberrations
            Copy with the coefficient tensor on the current CUDA device.
        """
        return Aberrations(self.array.cuda())

    def clone(self) -> Aberrations:
        """Return a copy with a cloned ``array`` tensor.

        Returns
        -------
        Aberrations
            Copy with an independent coefficient tensor.
        """
        return Aberrations(self.array.clone())

    def detach(self) -> Aberrations:
        """Return a copy with ``array`` detached from the autograd graph.

        Returns
        -------
        Aberrations
            Copy with the coefficient tensor detached from autograd.
        """
        return Aberrations(self.array.detach())

    def numpy(self):
        """Return the coefficients as a detached CPU :class:`numpy.ndarray`.

        Returns
        -------
        numpy.ndarray
            12-element array of the Cartesian coefficients.
        """
        return self.array.detach().cpu().numpy()

    def __array__(self, dtype=None):
        arr = self.numpy()
        return arr.astype(dtype) if dtype is not None else arr

    def __len__(self) -> int:
        return self.N

    def __iter__(self) -> Iterator[Tensor]:
        return iter(self.array)

    def allclose(self, other: Aberrations | Tensor, **kwargs) -> bool:
        """Check elementwise closeness of the Cartesian coefficient arrays.

        Parameters
        ----------
        other : Aberrations or Tensor
            Other coefficients to compare against.
        **kwargs
            Forwarded to :func:`torch.allclose` (e.g. ``rtol``, ``atol``).

        Returns
        -------
        bool
            ``True`` if all coefficients are close within tolerance.
        """
        other_array = other.array if isinstance(other, Aberrations) else other
        return bool(torch.allclose(self.array, torch.as_tensor(other_array), **kwargs))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Aberrations):
            return bool(torch.equal(self.array, other.array))
        if isinstance(other, Tensor):
            return bool(torch.equal(self.array, other))
        return NotImplemented

    # -- arithmetic --------------------------------------------------------

    def __add__(self, other: Aberrations) -> Aberrations:
        return Aberrations(self.array + _other_array(other))

    def __sub__(self, other: Aberrations) -> Aberrations:
        return Aberrations(self.array - _other_array(other))

    def __mul__(self, scalar: float) -> Aberrations:
        return Aberrations(self.array * scalar)

    __rmul__ = __mul__

    def __copy__(self) -> Aberrations:
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
        for mag_name, ang_name in _POLAR_PAIRS:
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


def _other_array(other: Aberrations | Tensor) -> Tensor:
    if isinstance(other, Aberrations):
        return other.array
    return torch.as_tensor(other)
