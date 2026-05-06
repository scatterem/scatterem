import re
from typing import Any, List, Optional, Sequence, Tuple, Union

import torch


class UnitTensor(torch.Tensor):
    """
    A tensor class that supports physical units for each dimension.

    This class extends PyTorch's Tensor to include unit information and
    sampling metadata as properties. Units can be specified per dimension.
    """

    # ----------------------------------------------
    # construction
    # ----------------------------------------------
    def __new__(
        cls,
        data: Any,
        units: Optional[
            Union[str, Sequence[Union[str, Tuple[str, ...], None]], None]
        ] = None,
        sampling: Any = None,
        *,
        requires_grad: Optional[bool] = None,
    ) -> "UnitTensor":
        # 1.  Turn data into a regular Tensor first
        dt = torch.as_tensor(data)
        # 2.  Make the subclass instance that shares storage
        obj = torch.Tensor._make_subclass(cls, dt)  # real subclass
        # 3.  Set requires_grad if specified
        if requires_grad is not None:
            obj.requires_grad = requires_grad
        # 4.  Attach metadata
        obj.units = cls._validate_units(units, dt.shape)
        obj.sampling = sampling
        return obj

    @staticmethod
    def _validate_units(
        units: Optional[Union[str, Sequence[Union[str, Tuple[str, ...], None]], None]],
        shape: torch.Size,
    ) -> Optional[List[Optional[Tuple[str, ...]]]]:
        """Validate and normalize units for each dimension."""
        if units is None:
            return None

        if isinstance(units, str):
            # Single unit string - apply to all dimensions
            unit_tuple = UnitTensor._parse_unit_string(units)
            return [unit_tuple] * len(shape)
        elif isinstance(units, (list, tuple)):
            # List/tuple of units - one per dimension
            if len(units) != len(shape):
                raise ValueError(
                    f"Number of units ({len(units)}) must match tensor dimensions ({len(shape)})"
                )
            return [
                UnitTensor._parse_unit_string(unit) if isinstance(unit, str) else unit
                for unit in units
            ]
        else:
            raise ValueError(f"Invalid units format: {units}")

    @staticmethod
    def _parse_unit_string(unit_str: str) -> Optional[Tuple[str, ...]]:
        """Parse a single unit string into tuple format."""
        if not unit_str or unit_str.strip() == "":
            return None

        # Simple parser for basic units like "kg*m/s^2"
        parts = re.split(r"([*/])", unit_str)
        units: List[str] = []

        for part in parts:
            part = part.strip()
            if part in ["*", "/"]:
                units.append(part)
            elif part:
                # Handle exponents like "m^2"
                if "^" in part:
                    base, exp = part.split("^")
                    exp_val = int(exp)
                    if exp_val > 0:
                        units.extend([base] * exp_val)
                    else:
                        # Negative exponents become division
                        units.append("/")
                        units.extend([base] * abs(exp_val))
                else:
                    units.append(part)

        return tuple(units)

    # ----------------------------------------------
    # pretty-printing
    # ----------------------------------------------
    def __repr__(self) -> str:
        core = super().__repr__()
        if hasattr(self, "units"):
            unit_str = (
                self._units_to_string(self.units) if self.units else "dimensionless"
            )
            sampling_str = (
                f", sampling={self.sampling}"
                if hasattr(self, "sampling") and self.sampling is not None
                else ""
            )
            return f"{core}  # units={unit_str}{sampling_str}"
        else:
            return core

    def _units_to_string(self, units: Optional[List[Optional[Tuple[str, ...]]]]) -> str:
        """Convert units list back to string representation."""
        if not units:
            return "dimensionless"

        # Convert each dimension's units to string
        dim_strings = []
        for i, dim_units in enumerate(units):
            if dim_units is None:
                dim_strings.append("dimensionless")
            else:
                dim_strings.append(self._dim_units_to_string(dim_units))

        return f"({', '.join(dim_strings)})"

    def _dim_units_to_string(self, units: Tuple[str, ...]) -> str:
        """Convert a single dimension's units tuple to string representation."""
        if not units:
            return "dimensionless"

        # Group units by type and count, handling division
        numerator_units: dict[str, int] = {}
        denominator_units: dict[str, int] = {}
        current_group = numerator_units

        for unit in units:
            if unit == "/":
                current_group = denominator_units
            elif unit == "*":
                continue
            else:
                current_group[unit] = current_group.get(unit, 0) + 1

        # Build string representation
        parts = []

        # Numerator
        if numerator_units:
            num_parts = []
            for unit, count in numerator_units.items():
                if count == 1:
                    num_parts.append(unit)
                else:
                    num_parts.append(f"{unit}^{count}")
            parts.append("*".join(num_parts))

        # Denominator
        if denominator_units:
            if parts:  # If we have a numerator, add division
                parts.append("/")
            denom_parts = []
            for unit, count in denominator_units.items():
                if count == 1:
                    denom_parts.append(unit)
                else:
                    denom_parts.append(f"{unit}^{count}")
            parts.append("*".join(denom_parts))

        return "".join(parts) if parts else "dimensionless"

    # ----------------------------------------------
    # unit operations
    # ----------------------------------------------
    def get_dimension_units(self, dim: int) -> Optional[Tuple[str, ...]]:
        """Get units for a specific dimension."""
        if self.units is None or dim >= len(self.units):
            return None
        return self.units[dim]

    def has_same_units(self, other: Any) -> bool:
        """Check if this tensor has the same units as another."""
        if isinstance(other, UnitTensor):
            return self.units == other.units
        return self.units is None

    def unit_multiply(
        self, other_units: Optional[List[Optional[Tuple[str, ...]]]]
    ) -> Optional[List[Optional[Tuple[str, ...]]]]:
        """Multiply units (for tensor multiplication)."""
        if self.units is None and other_units is None:
            return None
        if self.units is None:
            return other_units
        if other_units is None:
            return self.units

        # For per-dimension units, we need to handle this differently
        # This is a simplified version - could be enhanced
        return self.units + other_units

    def unit_divide(
        self, other_units: Optional[List[Optional[Tuple[str, ...]]]]
    ) -> Optional[List[Optional[Tuple[str, ...]]]]:
        """Divide units (for tensor division)."""
        if self.units is None and other_units is None:
            return None
        if self.units is None:
            return [("/",)] + other_units
        if other_units is None:
            return self.units

        return self.units + [("/",)] + other_units

    # ----------------------------------------------
    # convenience constructors
    # ----------------------------------------------
    @classmethod
    def zeros(
        cls,
        *size: int,
        units: Optional[
            Union[str, Sequence[Union[str, Tuple[str, ...], None]], None]
        ] = None,
        sampling: Any = None,
        **kwargs: Any,
    ) -> "UnitTensor":
        """Create a UnitTensor filled with zeros."""
        return cls(torch.zeros(*size, **kwargs), units, sampling)

    @classmethod
    def ones(
        cls,
        *size: int,
        units: Optional[
            Union[str, Sequence[Union[str, Tuple[str, ...], None]], None]
        ] = None,
        sampling: Any = None,
        **kwargs: Any,
    ) -> "UnitTensor":
        """Create a UnitTensor filled with ones."""
        return cls(torch.ones(*size, **kwargs), units, sampling)

    @classmethod
    def randn(
        cls,
        *size: int,
        units: Optional[
            Union[str, Sequence[Union[str, Tuple[str, ...], None]], None]
        ] = None,
        sampling: Any = None,
        **kwargs: Any,
    ) -> "UnitTensor":
        """Create a UnitTensor with random normal values."""
        return cls(torch.randn(*size, **kwargs), units, sampling)

    @classmethod
    def arange(
        cls,
        *args: Any,
        units: Optional[
            Union[str, Sequence[Union[str, Tuple[str, ...], None]], None]
        ] = None,
        sampling: Any = None,
        **kwargs: Any,
    ) -> "UnitTensor":
        """Create a UnitTensor with arange values."""
        return cls(torch.arange(*args, **kwargs), units, sampling)

    @classmethod
    def linspace(
        cls,
        start: float,
        end: float,
        steps: int,
        units: Optional[
            Union[str, Sequence[Union[str, Tuple[str, ...], None]], None]
        ] = None,
        sampling: Any = None,
        **kwargs: Any,
    ) -> "UnitTensor":
        """Create a UnitTensor with linspace values."""
        return cls(torch.linspace(start, end, steps, **kwargs), units, sampling)
