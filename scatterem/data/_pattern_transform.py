"""The streamed pattern conversion (pattern streaming T1).

A streamed series item keeps its patterns in the source encoding
(uint8 or uint16 counts, or float16 noise-free intensities with a per-tilt
power-of-two scale) and converts one batch at a time with
:class:`RowTransform`, which replays the resident constructor's operations
(``Dataset4DStem.__init__``) in the same order:

1. cast to float32 (exact for every encoding; a float32 input is copied);
1a. float16 only: ``x * 2**-e`` (exact), then optionally keyed Poisson counts
    at ``n_e * x`` (:func:`~scatterem.utils.warp._keyed_poisson.keyed_poisson`);
2. an optional pre-divisor (a source scale);
3. ``fftshift`` over ``fourier_shift_dim``;
4. ``clamp_min_(0)`` when ``clip_neg_values`` (float and signed integer
   encodings: a no-op on unsigned counts and bool, so it is skipped there);
5. ``sqrt_()`` when ``transform_to_amplitudes``: CPU and CUDA ``sqrt`` differ
   in the last bit, so steps 1-5 run on the item's ``transform_device`` (unset:
   the dataset's device in ``Dataset4DStem.from_source``, else the CPU);
6. ``x / c`` with ``c`` a 0-dim float32 tensor **on the batch device**: CUDA
   divides by a CPU tensor or a Python float through its reciprocal, which
   changes the bits (§5.3).

Given the same ``c``, the result is bitwise the resident ``_array3d[rows]``.

Pass 0's statistics are in :mod:`scatterem.data._pass0`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import Tensor

__all__ = [
    "ENCODINGS",
    "PoissonSpec",
    "STREAM_ENCODINGS",
    "RowTransform",
    "cast_only",
    "check_stream_encoding",
    "poisson_key",
]

#: The host encodings of a streamed item (bfloat16 is refused).
STREAM_ENCODINGS = (torch.uint8, torch.uint16, torch.float16)
#: What a RowTransform reads: every real dtype ``Dataset4DStem.__init__`` takes. Only
#: the streamed tier restricts dtypes (``check_stream_encoding``).
ENCODINGS = (
    torch.bool, torch.uint8, torch.int8, torch.uint16, torch.int16, torch.uint32,
    torch.int32, torch.uint64, torch.int64, torch.float16, torch.bfloat16,
    torch.float32, torch.float64,
)  # fmt: skip


def _check_encoding(dtype: torch.dtype) -> None:
    if dtype not in ENCODINGS:
        raise ValueError(
            f"a RowTransform reads a real dtype (bool, an integer or a float), got {dtype}"
        )


def check_stream_encoding(dtype: torch.dtype) -> None:
    """Refuse a dtype that is not a host encoding of a streamed item.

    Parameters
    ----------
    dtype : torch.dtype
        The patterns' dtype.

    Raises
    ------
    ValueError
        Anything but uint8, uint16 or float16 (float32 is ingested to float16).
    """
    if dtype == torch.bfloat16:
        raise ValueError(
            "bfloat16 patterns are refused: its 8-bit significand puts the "
            "noise-free error floor at ~2e-3; use float16"
        )
    if dtype not in STREAM_ENCODINGS:
        raise ValueError(
            f"a streamed item holds uint8, uint16 or float16 patterns, got {dtype}; "
            "ingest float data to float16 (encode_float16)"
        )


# ---- keyed Poisson ----------------------------------------------------------


@dataclass(frozen=True)
class PoissonSpec:
    """On-the-fly Poisson counts at a dose.

    Attributes
    ----------
    electrons_per_probe : float
        ``N_e = dose * scan_step_y * scan_step_x``; the mean is ``N_e * I``.
    key : int
        The 63-bit key, :func:`poisson_key` of ``(noise_seed, tilt)``.
    """

    electrons_per_probe: float
    key: int


def poisson_key(noise_seed: int, tilt: int) -> int:
    """``keyed_seed(noise_seed, tilt)``, the key of one tilt's draws."""
    from scatterem.distributed._seeding import keyed_seed  # noqa: PLC0415

    return keyed_seed(noise_seed, tilt)


# ---- the conversion ------------------------------------------------------


def _scalar(value: Tensor | None, name: str) -> None:
    if value is None:
        return
    if not isinstance(value, Tensor) or value.ndim != 0 or value.dtype != torch.float32:
        raise TypeError(
            f"{name} must be a 0-dim float32 tensor (a Python float divides with "
            "different bits on CUDA)"
        )


def _same_device(value: Tensor | None, x: Tensor, name: str) -> None:
    if value is not None and value.device != x.device:
        raise ValueError(
            f"{name} is on {value.device} but the batch is on {x.device}; place "
            "the transform first (RowTransform.placed)"
        )


def _detector_dims(dims: int | tuple[int, ...]) -> tuple[int, ...]:
    """The resident ``fourier_shift_dim`` (4-D numbering) as batch dims (-2, -1).

    A batch is ``(n, ky, kx)``: a scan axis cannot be replayed per row.
    """
    seq = (dims,) if isinstance(dims, int) else tuple(dims)
    out = tuple(int(d) - 4 if int(d) >= 0 else int(d) for d in seq)
    if not out or any(d not in (-2, -1) for d in out):
        raise ValueError(
            f"fourier_shift_dim {dims!r} must name detector axes only (2, 3 or "
            "-2, -1): a streamed batch cannot shift a scan axis"
        )
    return out


@dataclass(frozen=True)
class RowTransform:
    """The resident operation chain for one tilt (module docstring).

    Every divisor is a 0-dim float32 tensor on the device of its step (U2): a
    Python float divides by its reciprocal on CUDA and gives other bits.

    Parameters
    ----------
    encoding : torch.dtype
        The stored dtype: any real dtype (``ENCODINGS``). A streamed item takes
        only uint8, uint16 and float16 (``check_stream_encoding``).
    fourier_shift_dim : tuple of int or None
        As for ``Dataset4DStem``.
    clip_neg_values : bool
        As for ``Dataset4DStem``; skipped for unsigned encodings and bool.
    transform_to_amplitudes : bool
        As for ``Dataset4DStem``.
    c : Tensor or None
        0-dim float32 normalisation constant; None when not normalising.
    inv_scale : Tensor or None
        For float16 only: the 0-dim float32 ``2**-e``.
    pre : Tensor or None
        0-dim float32 pre-divisor (a source scale), or None.
    poisson : PoissonSpec or None
        For float16 only: draw counts at a dose.
    transform_device : torch.device or None
        Where steps 1-5 run (the device type ``sqrt`` must run on). None (not
        set): ``Dataset4DStem.from_source`` uses the dataset's device and moves
        it with the dataset; a transform used on its own runs on the CPU.
    """

    encoding: torch.dtype
    fourier_shift_dim: tuple[int, ...] | None = None
    clip_neg_values: bool = True
    transform_to_amplitudes: bool = False
    c: Tensor | None = None
    inv_scale: Tensor | None = None
    pre: Tensor | None = None
    poisson: PoissonSpec | None = None
    transform_device: torch.device | None = None

    def __post_init__(self) -> None:
        _check_encoding(self.encoding)
        for name in ("c", "inv_scale", "pre"):
            _scalar(getattr(self, name), name)
        is_f16 = self.encoding == torch.float16
        if is_f16 != (self.inv_scale is not None):
            raise ValueError("inv_scale is required for float16 and only for it")
        if self.poisson is not None and not is_f16:
            raise ValueError("Poisson sampling applies to float16 noise-free data")
        if self.fourier_shift_dim is not None:
            object.__setattr__(
                self, "fourier_shift_dim", _detector_dims(self.fourier_shift_dim)
            )
        if self.pre is not None and not float(self.pre) > 0:
            raise ValueError(f"the pre-divisor must be positive, got {float(self.pre)}")
        if self.transform_device is not None:
            object.__setattr__(
                self, "transform_device", torch.device(self.transform_device)
            )

    @property
    def steps_device(self) -> torch.device:
        """Where steps 1-5 run: ``transform_device``, or the CPU when it is not set."""
        return self.transform_device or torch.device("cpu")

    @property
    def is_identity(self) -> bool:
        """Whether ``stage1``/``stage2`` return the input unchanged (float32, no step)."""
        return (
            self.encoding == torch.float32
            and self.fourier_shift_dim is None
            and not self.clip_neg_values
            and not self.transform_to_amplitudes
            and self.c is None
            and self.pre is None
        )

    @property
    def integer_values(self) -> bool:
        """Whether the step-5 values are integers (pass 0 sums exactly)."""
        floats = self.encoding.is_floating_point
        counts = not floats or self.poisson is not None
        return counts and self.pre is None and not self.transform_to_amplitudes

    @property
    def device_exact(self) -> bool:
        """Whether steps 1-5 give the same bits on every device.

        Only ``sqrt`` (step 5) and the Poisson draws (step 1a) differ between the CPU
        and CUDA; the cast, the power-of-two scale, the IEEE ``pre`` divide, fftshift
        and the clamp do not.
        """
        return not self.transform_to_amplitudes and self.poisson is None

    def placed(self, batch_device: torch.device | str) -> RowTransform:
        """Return this transform placed for a batch device (once per ring).

        Parameters
        ----------
        batch_device : torch.device or str
            Where step 6 runs; ``c`` moves there.

        Returns
        -------
        RowTransform
            A copy with ``inv_scale`` and ``pre`` on ``transform_device`` (the
            CPU when it is not set) and ``c`` on ``batch_device``.
        """
        tdev = self.steps_device
        return replace(
            self,
            c=None if self.c is None else self.c.to(batch_device),
            inv_scale=None if self.inv_scale is None else self.inv_scale.to(tdev),
            pre=None if self.pre is None else self.pre.to(tdev),
        )

    def with_c(self, c: Tensor | None) -> RowTransform:
        """Return this transform with another normalisation constant.

        Parameters
        ----------
        c : Tensor or None
            The 0-dim float32 constant, or None for no division.

        Returns
        -------
        RowTransform
            A copy with ``c`` replaced.
        """
        return replace(self, c=c)

    def stage1(self, raw: Tensor, rows: Tensor | None = None) -> Tensor:
        """Run steps 1-5 on ``raw``'s device, which must be ``transform_device``'s type.

        Parameters
        ----------
        raw : Tensor
            ``(n, ky, kx)`` patterns in the encoding.
        rows : Tensor or None
            ``(n,)`` int64 global rows; required with Poisson sampling.

        Returns
        -------
        Tensor
            The float32 ``(n, ky, kx)`` values before the division by ``c``; ``raw``
            itself when :attr:`is_identity`, otherwise a new tensor.

        Raises
        ------
        ValueError
            ``raw`` is not in the encoding or not on ``transform_device``'s
            type, a scalar sits on another device, or Poisson sampling lacks
            ``rows``.
        """
        if raw.dtype != self.encoding:
            raise ValueError(f"raw patterns are {raw.dtype}, encoding {self.encoding}")
        if raw.device.type != self.steps_device.type:
            raise ValueError(
                f"steps 1-5 run on {self.steps_device.type} (transform_device), "
                f"the patterns are on {raw.device}"
            )
        if self.is_identity:
            return raw
        # A float32 input is copied, so the in-place steps below never write into
        # the caller's tensor (a TensorSource read aliases the stored cube).
        x = raw.to(torch.float32, copy=raw.dtype == torch.float32)
        if self.inv_scale is not None:
            _same_device(self.inv_scale, x, "inv_scale")
            x = x * self.inv_scale
            if self.poisson is not None:
                if rows is None:
                    raise ValueError("Poisson sampling needs the global rows")
                x = self._sample(x, rows)
        if self.pre is not None:
            _same_device(self.pre, x, "pre")
            x = x / self.pre
        if self.fourier_shift_dim is not None:
            x = torch.fft.fftshift(x, dim=self.fourier_shift_dim)
        if self.clip_neg_values and self.encoding.is_signed:  # floats, signed ints
            x.clamp_min_(0)
        if self.transform_to_amplitudes:
            x.sqrt_()
        return x

    def _sample(self, x: Tensor, rows: Tensor) -> Tensor:
        from scatterem.utils.warp._keyed_poisson import keyed_poisson  # noqa: PLC0415

        assert self.poisson is not None
        lam = x * self.poisson.electrons_per_probe
        return keyed_poisson(lam, rows, key=self.poisson.key)

    def stage2(self, x: Tensor) -> Tensor:
        """Step 6: ``x / c`` with ``c`` on ``x``'s device.

        Parameters
        ----------
        x : Tensor
            The :meth:`stage1` values.

        Returns
        -------
        Tensor
            ``x / c``, or ``x`` itself without ``c``.

        Raises
        ------
        ValueError
            ``c`` is on another device than ``x``.
        """
        if self.c is None:
            return x
        _same_device(self.c, x, "c")
        return x / self.c

    def apply(self, raw: Tensor, rows: Tensor | None = None) -> Tensor:
        """Run steps 1-6 on ``raw``'s device (:meth:`stage1`, then :meth:`stage2`).

        Parameters
        ----------
        raw : Tensor
            ``(n, ky, kx)`` patterns in the encoding.
        rows : Tensor or None
            ``(n,)`` int64 global rows; required with Poisson sampling.

        Returns
        -------
        Tensor
            The float32 ``(n, ky, kx)`` batch; ``raw`` itself when :attr:`is_identity`.

        Raises
        ------
        ValueError
            As :meth:`stage1` and :meth:`stage2`.
        """
        return self.stage2(self.stage1(raw, rows))


def cast_only(encoding: torch.dtype) -> RowTransform:
    """The transform that only casts ``encoding`` to float32 (no clip, no ``c``).

    A float16 encoding gets ``inv_scale = 1``, so its values stay as stored.

    Parameters
    ----------
    encoding : torch.dtype
        The stored dtype.

    Returns
    -------
    RowTransform
        Reads give the stored values as float32.
    """
    one = torch.tensor(1.0) if encoding == torch.float16 else None
    return RowTransform(encoding, clip_neg_values=False, inv_scale=one)
