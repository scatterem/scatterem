"""Base class for downloadable single-tilt 4D-STEM datasets.

Unlike :class:`~scatterem.datasets.scanning_diffraction.PublicScanningDiffractionDataset`
(which extends ``Dataset4dstemTomo`` and attaches a ``Metadata4D`` for the
iterative tomography models), this base produces a plain
:class:`~scatterem.utils.data.datasets.Dataset4dstem` carrying a
``Metadata4dstem`` -- what the direct-ptychography / tilt-corrected-dark-field /
fused-full-field reconstruction methods expect.
"""

import json
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, ClassVar, Optional, Sequence, Union

import numpy as np
import torch
from numpy.typing import DTypeLike

from scatterem.datasets.utils import check_md5, download_url, zenodo_file_url
from scatterem.utils.data.datasets import (
    Dataset4dstem,
    _metadata4dstem_from_physics,
)
import os

#: Below this, a plain ``.to(device)`` already saturates the link and the
#: pinned staging buffers are not worth their allocation.
_H2D_STAGE_MIN_BYTES = 256 << 20

#: Chunk size of the staged copy. Two buffers of this size are in flight, so
#: the pinned footprint is twice it; 16/32/64 MiB measure within 1.1% of each
#: other, so this is not a sharp knob.
_H2D_STAGE_CHUNK_BYTES = 16 << 20


def _h2d_staging_enabled() -> bool:
    """``SCATTEREM_H2D_STAGING=0`` restores the plain ``.to(device)``.

    An escape hatch for a host that cannot spare the page-locked memory.
    """
    return os.environ.get("SCATTEREM_H2D_STAGING", "1") != "0"

#: Class attributes every concrete subclass must set to a non-None value
#: before it can be instantiated (checked in __init__; see
#: ``_check_subclass_contract``).
_REQUIRED_CONSTANTS = (
    "zenodo_record_id",
    "energy",
    "semiconvergence_angle",
    "scan_step",
)

#: Sidecar written beside a resource once its declared md5 has been verified, so
#: the next construction can skip the hash. md5 runs at ~0.8 GiB/s single-
#: threaded, so re-verifying a multi-GiB cube costs seconds of pure re-hashing on
#: every ``__init__`` (9.0 s for the 6.1 GiB Fig. 1 cube, 62 % of its
#: construction) and produces no byte anything downstream can see.
_MD5_STAMP_SUFFIX = ".md5-stamp.json"

#: Staging chunk used by :func:`load_npy_to_device` when it converts dtype on
#: arrival -- larger than the plain ``_H2D_STAGE_CHUNK_BYTES`` because the extra
#: per-chunk ``copy_`` makes the granularity matter (see the comment there).
_NPY_CAST_CHUNK_BYTES = 128 << 20


def _md5_stamp_path(fpath: Path) -> Path:
    """Hidden sidecar beside ``fpath``, named so it can never collide with a
    declared resource (those are the published file names)."""
    return fpath.with_name(f".{fpath.name}{_MD5_STAMP_SUFFIX}")


def _md5_stamps_enabled() -> bool:
    """Read at call time, not import time, so a test/caller can force the full
    hash back on for one process."""
    return os.environ.get("SCATTEREM_MD5_STAMP", "1") not in ("0", "false", "False")


def _md5_stamp_is_current(fpath: Path, md5: str, st: os.stat_result) -> bool:
    """True if ``fpath`` was verified against exactly ``md5`` and has not been
    touched since -- same size AND same ``mtime_ns``.

    Deliberately conservative: any unreadable, malformed, absent or disagreeing
    stamp returns False and the caller falls back to hashing, so the stamp can
    only ever save work, never manufacture a pass.
    """
    try:
        with open(_md5_stamp_path(fpath), "rb") as fh:
            stamp = json.load(fh)
    except (OSError, ValueError):
        return False
    if not isinstance(stamp, dict):
        return False
    return (
        stamp.get("md5") == md5
        and stamp.get("size") == st.st_size
        and stamp.get("mtime_ns") == st.st_mtime_ns
    )


def _write_md5_stamp(fpath: Path, md5: str, st: os.stat_result) -> None:
    """Best effort -- a read-only or full cache directory must not turn a
    successful verification into a construction failure.

    Written to a temporary and ``os.replace``d so an interrupted write leaves
    either the old stamp or none, never a half-written one that would be
    silently believed.
    """
    dest = _md5_stamp_path(fpath)
    tmp = dest.with_name(dest.name + ".tmp")
    try:
        with open(tmp, "w") as fh:
            json.dump({"md5": md5, "size": st.st_size, "mtime_ns": st.st_mtime_ns}, fh)
        os.replace(tmp, dest)
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _unlink_md5_stamp(fpath: Path) -> None:
    """Drop a stamp that no longer describes the file (also best effort)."""
    try:
        os.unlink(_md5_stamp_path(fpath))
    except OSError:
        pass


#: ``.npy`` reads at or above this size get the parallel destination prefault in
#: :func:`load_npy`. This is NOT a tuning knob: 32 MiB is glibc's
#: ``DEFAULT_MMAP_THRESHOLD_MAX``, so a malloc request at or above it is always a
#: fresh anonymous mmap whose pages are cold, while below it the recycled heap
#: hands back already-faulted ones. Measured on a 6.1 GiB cube: the crossover is
#: exactly here -- ``np.load`` holds 3.2-6.7 GiB/s below 32 MiB and drops to
#: 1.67-1.78 GiB/s at and above it, and the prefault LOSES (0.15x at 4 MiB,
#: 0.65x at 16, 0.76x at 24) before winning (1.06x at 32, 1.33x at 48, 1.39x at
#: 128, 2.1x at 6.1 GiB).
_NPY_PREFAULT_MIN_BYTES = 32 << 20

#: Upper bound on prefault threads. The speedup plateaus from ~12 up (8 threads
#: 1.99x, 12 2.12x, 16 2.08x, 20 2.11x on the 6.1 GiB cube), so the exact value
#: is not load-bearing; it is bounded only so a many-core node does not spawn one
#: thread per core to run a memset.
_NPY_PREFAULT_MAX_THREADS = 16

#: Chunk for the sequential read. One 64 MiB ``readinto`` per call keeps a single
#: sequential I/O stream, which is the whole point (see :func:`load_npy`).
_NPY_READ_CHUNK = 64 << 20


def _npy_prefault_enabled() -> bool:
    """Read at call time, not import time, so a caller can force plain
    ``np.load`` back on for one process."""
    return os.environ.get("SCATTEREM_NPY_PREFAULT", "1") not in ("0", "false", "False")


def _prefault(flat: np.ndarray, nthreads: int) -> None:
    """Take ``flat``'s page faults on ``nthreads`` threads instead of one.

    Writing zeros looks like wasted bandwidth and is not: the stores ARE the
    mechanism, because each one faults in a page of the fresh anonymous mapping.
    Touching one byte per page instead is *slower* -- too few elements to clear
    numpy's threading grain size, so it runs single-threaded. ``np.zeros`` does
    not work either: calloc hands back lazily-zero-filled pages, so the faults
    are simply deferred to whoever reads them first.

    numpy releases the GIL for the assignment, so this scales (1 thread 3.05 s,
    2 2.27, 4 1.82, 8 1.61, 12 1.51 for read+prefault of 6.1 GiB).
    """
    n = flat.size
    chunk = -(-n // nthreads)

    def zero(i: int) -> None:
        start = i * chunk
        end = min(n, start + chunk)
        if start < end:
            flat[start:end] = 0

    with ThreadPoolExecutor(nthreads) as pool:
        list(pool.map(zero, range(nthreads)))


def load_npy(path: Union[str, Path]) -> np.ndarray:
    """Read a ``.npy`` file into memory, faster than ``np.load`` for big cubes.

    Returns exactly what ``np.load(path)`` returns -- same dtype (byte order
    included), shape, memory order and bytes.

    ``np.load`` on a multi-GiB cube is not disk-bound and not read-bound: it is
    bound by the page faults on its own fresh destination buffer. On the 6.1 GiB
    Fig. 1 cube, warm page cache, it runs at 1.9 GiB/s, while the identical
    sequential read into an already-faulted buffer runs at 4.8 GiB/s -- i.e. 1.6
    million minor faults, taken one at a time by the single thread that is also
    doing the copy, are ~2/3 of the stage. So take the faults first, in parallel,
    and then do the read: 3.20 s -> 1.51 s.

    The read itself stays a SINGLE sequential stream, deliberately. Reading the
    file with N threads is much faster still when the page cache holds it
    (0.50 s, 6.4x) but it is *slower* when it does not, because these cubes live
    on spinning disks and N concurrent streams seek-thrash: measured cold, with
    ``posix_fadvise(POSIX_FADV_DONTNEED)`` to evict, 24.5 s sequential vs 29.5 s
    with 16 threads. The prefault costs 0.3 s of RAM-side work before any I/O is
    issued, so it cannot change that trade-off in either direction.

    Falls back to ``np.load`` for anything it cannot handle byte-for-byte:
    pickled object arrays, ``.npz`` archives, dtypes without a plain buffer
    protocol, files below ``_NPY_PREFAULT_MIN_BYTES``, and
    ``SCATTEREM_NPY_PREFAULT=0``.
    """
    if not _npy_prefault_enabled():
        return np.load(path)

    # np.load(mmap_mode="r") makes numpy's own header parser do the work -- every
    # format version, and it raises for exactly the cases that must fall back
    # (pickled object dtypes, zip archives). The mapping is dropped immediately;
    # only the metadata is kept.
    try:
        probe = np.load(path, mmap_mode="r")
        shape = probe.shape
        dtype = probe.dtype
        offset = probe.offset
        # numpy does not surface the header's `fortran_order` field on the
        # memmap, so infer it from the layout it built. The one case the
        # inference cannot see is a shape that is BOTH C- and F-contiguous (1-D,
        # or any shape containing a 1) whose header nonetheless says
        # fortran_order=True -- there the two reshapes differ only in the stride
        # of a length-1 axis, which no index can reach. `np.save` never writes
        # that combination.
        fortran_order = probe.flags.f_contiguous and not probe.flags.c_contiguous
        del probe
    except Exception:
        return np.load(path)

    count = int(np.prod(shape, dtype=np.int64))
    nbytes = count * dtype.itemsize
    if nbytes < _NPY_PREFAULT_MIN_BYTES:
        return np.load(path)

    flat = np.empty(count, dtype=dtype)
    try:
        buf = memoryview(flat).cast("B")
    except (TypeError, ValueError):
        # A dtype numpy cannot expose as a plain byte buffer (some structured
        # dtypes). Nothing here is worth a special case.
        del flat
        return np.load(path)

    threads = os.cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):
        threads = len(os.sched_getaffinity(0)) or threads
    _prefault(flat, min(threads, _NPY_PREFAULT_MAX_THREADS))

    with open(path, "rb") as fh:
        fh.seek(offset)
        got = 0
        while got < nbytes:
            read = fh.readinto(buf[got : got + _NPY_READ_CHUNK])
            if not read:
                break
            got += read
    if got != nbytes:
        raise OSError(
            f"{path}: .npy payload is short -- expected {nbytes} bytes of data "
            f"after the {offset}-byte header, read {got}. The file is truncated."
        )

    # Exactly numpy's own reshape in `np.lib.format.read_array`, so the returned
    # array's shape, strides and flags are identical to np.load's by
    # construction rather than by argument.
    if fortran_order:
        flat.shape = shape[::-1]
        return flat.transpose()
    flat.shape = shape
    return flat


def _npy_direct_enabled() -> bool:
    """``SCATTEREM_NPY_DIRECT=0`` restores the two-step host read + staged copy.

    Read at call time, not import time. This is the in-process control arm the A/B
    measurement is quoted against, and an escape hatch for a caller that wants the
    host array materialised the way it always was.
    """
    return os.environ.get("SCATTEREM_NPY_DIRECT", "1") not in ("0", "false", "False")


def load_npy_to_device(
    path: Union[str, Path],
    device: Union[str, torch.device],
    dtype: Optional[torch.dtype] = None,
) -> Optional[torch.Tensor]:
    """Read a ``.npy`` file straight onto a CUDA device, or return ``None``.

    The result is byte-for-byte
    ``torch.from_numpy(load_npy(path)).to(device).to(dtype)`` --
    pure data movement plus, at most, the one elementwise cast the caller was
    going to make anyway, so it is bit-identical by construction rather than by
    tolerance. ``None`` means "not handled here": the caller must then do exactly
    what it did before, so every fallback is neutral by inspection.

    ``dtype`` (default: the file's own) is applied *on arrival*, per staged chunk,
    by the same ``Tensor.copy_`` that ``.to(dtype)`` dispatches to. The point is
    again what does not happen: reading the 6.125 GiB uint16 figure-1 cube with
    ``dtype=torch.float32`` means the uint16 *device* cube never exists, so the
    construction's peak drops by exactly its 6.125 GiB (20498 -> 14226 MiB). The
    time is close to a wash and the peak is the point: folding the cast in costs
    ~7 ms inside the staging loop and saves the 31 ms bulk convert (1143.5 vs
    1167.3 ms on that cube). The 6.125 GiB of first-touch ``cudaMalloc`` it also
    deletes is worth ~275 ms only on a COLD allocator -- a caller that builds the
    dataset repeatedly in one process has already had those blocks cached back.
    A converting pinned-host -> device ``copy_``
    returns to the CPU in 25 us for a 64 MiB source against 5.8 ms of device time
    (``_probe_r94_async.py``), i.e. it is as asynchronous as the same-dtype one, so
    the read/DMA pipeline below is unchanged.

    The point is what does NOT happen. The two-step form reads the file into a fresh
    6.1 GiB anonymous host buffer (whose 1.6 M page faults are ~2/3 of the read --
    see :func:`load_npy`) and then copies that buffer to the device in pinned chunks
    (the staged host-to-device copy described above). Reading each chunk
    *directly into the pinned slot* deletes the host cube, its faults and its
    prefault, and overlaps the read of chunk ``i+1`` with the DMA of chunk ``i``.
    Measured on the 6.125 GiB Fig. 1 cube, warm page cache, A6000
    (``benchmarks/lab/_probe_r92_roof.py``): read 1434 ms + staged H2D 539 ms =
    1973 ms, against **1121 ms fused** = 1.76x, and 1121 ms is **91.6 % of the
    read-only roof** (1027 ms for the same sequential read into a reused 16 MiB
    buffer with no device in the picture), i.e. the 539 ms DMA is ~91 % hidden.
    Chunk size is not a sharp knob here either -- 4/8/16/32/64/128 MiB span
    5.28-5.55 GiB/s -- so this reuses ``_H2D_STAGE_CHUNK_BYTES`` rather than
    introducing a second one, and the same ``_H2D_STAGE_MIN_BYTES`` gate for the
    same reason (below it the ~10 ms first pin is not repaid).

    Returns ``None`` for: a non-CUDA destination, no CUDA at all, either escape
    hatch set, anything ``np.load``'s own header parser refuses (pickled object
    dtypes, ``.npz``), a non-native byte order or a dtype torch cannot hold, a
    Fortran-order file, a payload below the gate, and a host that refuses to
    page-lock.
    """
    device = torch.device(device)
    if (
        device.type != "cuda"
        or not torch.cuda.is_available()
        or not _npy_direct_enabled()
        or not _h2d_staging_enabled()
    ):
        return None

    # Same probe as load_npy: numpy's own header parser, mapping dropped at once.
    try:
        probe = np.load(path, mmap_mode="r")
        shape = probe.shape
        file_dtype = probe.dtype
        offset = probe.offset
        fortran_order = probe.flags.f_contiguous and not probe.flags.c_contiguous
        del probe
    except Exception:
        return None

    # A big-endian file must keep raising from torch.from_numpy exactly as it does
    # today, and a Fortran-order one would need the reversed-stride reconstruction
    # this path deliberately does not do. Both are handed back to the old route.
    if fortran_order or not file_dtype.isnative:
        return None
    try:
        torch_dtype = torch.from_numpy(np.empty(0, dtype=file_dtype)).dtype
    except (TypeError, ValueError):
        return None

    count = int(np.prod(shape, dtype=np.int64))
    nbytes = count * file_dtype.itemsize
    if nbytes < _H2D_STAGE_MIN_BYTES:
        return None

    # Converting on arrival adds one `copy_` per chunk, and THAT is what makes the
    # staging granularity matter here: at the plain 16 MiB chunk the fused loop pays
    # +112 ms over the same-dtype one on the 6.125 GiB cube (391 chunks), at 32 MiB
    # +8, at 128 MiB +6, at 256 MiB -5 (``_probe_r94_chunk.py``) -- i.e. it is
    # per-chunk dispatch, not bandwidth. So the casting path stages in larger pieces.
    # It costs 2x the chunk pinned and 2x on the device, both freed on return, and
    # neither is anywhere near the construction's peak.
    casting = dtype is not None and dtype != torch_dtype
    chunk = max(
        1,
        (_NPY_CAST_CHUNK_BYTES if casting else _H2D_STAGE_CHUNK_BYTES)
        // file_dtype.itemsize,
    )
    try:
        bufs = [
            torch.empty(chunk, dtype=torch_dtype, pin_memory=True) for _ in range(2)
        ]
    except RuntimeError:
        # Page-locking can fail outright (cgroup limit, exhausted host RAM).
        return None
    # memoryview over the pinned tensor's own storage: `readinto` writes the file
    # bytes into the very buffer the DMA reads from, which is the whole point.
    views = [memoryview(buf.numpy()).cast("B") for buf in bufs]

    # The ONE cube this function allocates. With `dtype` given it is the caller's
    # final dtype, so the file's own dtype never occupies a full cube anywhere.
    out = torch.empty(shape, dtype=dtype or torch_dtype, device=device)
    flat = out.reshape(-1)
    # Converting on arrival costs one device-side staging chunk per slot. Doing it
    # explicitly, in buffers that live for the whole loop, keeps the transfer itself
    # on `copy_`'s same-dtype path and spares the caching allocator an allocate/free
    # per chunk.
    stages = (
        [torch.empty(chunk, dtype=torch_dtype, device=device) for _ in range(2)]
        if casting
        else None
    )
    events = [torch.cuda.Event() for _ in range(2)]
    recorded = [False, False]
    slot = 0
    with open(path, "rb") as fh:
        fh.seek(offset)
        lo = 0
        while lo < count:
            hi = min(lo + chunk, count)
            want = (hi - lo) * file_dtype.itemsize
            # A slot may not be refilled until its DMA has drained.
            if recorded[slot]:
                events[slot].synchronize()
            got = 0
            while got < want:
                read = fh.readinto(views[slot][got:want])
                if not read:
                    raise OSError(
                        f"{path}: .npy payload is short -- expected {nbytes} bytes "
                        f"of data after the {offset}-byte header, read "
                        f"{lo * file_dtype.itemsize + got}. The file is truncated."
                    )
                got += read
            if casting:
                # Both on the current stream, so the cast cannot start before its
                # own DMA has landed, and the event below covers both.
                stages[slot][: hi - lo].copy_(bufs[slot][: hi - lo], non_blocking=True)
                flat[lo:hi].copy_(stages[slot][: hi - lo])
            else:
                flat[lo:hi].copy_(bufs[slot][: hi - lo], non_blocking=True)
            events[slot].record()
            recorded[slot] = True
            slot ^= 1
            lo = hi
    # The staging buffers are dropped on return, so the last DMA out of each must
    # have landed first. Every DMA was issued on the CURRENT stream, so the returned
    # tensor needs no cross-stream bookkeeping.
    for evt, was in zip(events, recorded):
        if was:
            evt.synchronize()
    return out


def _loader_owns(array: Any, device: Union[str, torch.device]) -> bool:
    """True when ``_load_array`` built ``array`` on the destination device itself.

    Only :func:`load_npy_to_device` and its kind return a tensor already sitting on
    the device the dataset was asked for; a host loader returns a numpy array. So
    this is exactly "the array is a freshly-allocated buffer nobody outside this
    constructor holds a reference to", which is what ``Dataset4dstem``'s ``copy``
    guard exists to detect (it protects a *caller's* buffer from the in-place clip /
    sqrt / normalize that follow). Comparing device *type* rather than the full
    device is deliberate: a different index means ``Dataset4dstem`` moves the tensor
    and thereby makes its own private copy anyway, so the answer stays safe.
    """
    return (
        isinstance(array, torch.Tensor)
        and array.device.type == torch.device(device).type
    )


class PublicDataset4dstem(Dataset4dstem):
    """A published 4D-STEM dataset that downloads itself and IS a ``Dataset4dstem``.

    Subclasses declare the Zenodo record, the files (with md5 checksums) and the
    acquisition constants, and implement :meth:`_load_array`.

    The cache layout is FLAT: files live directly in ``root``, not in
    ``root/<ClassName>/raw``. Datasets published in a single record have unique
    file names, and a flat layout means an existing local copy of the data is
    reused as-is rather than re-downloaded.

    Args:
        root: Directory holding (or to receive) the raw files.
        download: Fetch any missing/corrupt file from Zenodo.
        device: Torch device for the data array.
        calibrate: Run ``calibrate_reciprocal_from_bright_field()`` after
            construction, replacing the placeholder ``dk``. All the paper's
            figure scripts do this, so it defaults to True. If the bright-field
            disk can't be measured (e.g. it sits near/at the detector edge),
            the measurement yields a non-finite/non-positive ``dk``, or the
            measured radius is implausibly large for a real bright-field disk
            (see ``_MAX_PLAUSIBLE_RADIUS_FRACTION``), the dataset is still
            constructed with the placeholder ``dk`` and a warning is raised
            rather than propagating the error -- pass ``calibrate=False`` to
            skip the attempt outright.
        **dataset_kwargs: Forwarded to ``Dataset4dstem.__init__`` (e.g.
            ``normalize``, ``clip_neg_values``, ``transform_to_amplitudes``).
    """

    zenodo_record_id: ClassVar[Optional[str]] = None
    #: ``[(filename, md5), ...]`` -- every file needed to load this dataset.
    resources: ClassVar[list[tuple[str, str]]] = []
    energy: ClassVar[Optional[float]] = None  # eV
    semiconvergence_angle: ClassVar[Optional[float]] = None  # rad
    scan_step: ClassVar[Optional[float]] = None  # Angstrom
    rotation: ClassVar[float] = 0.0  # deg
    #: Host-side dtype cast applied to the loaded array (None = leave as read).
    host_dtype: ClassVar[DTypeLike] = None
    reference: ClassVar[str] = ""

    def __init__(
        self,
        root: Union[str, Path],
        download: bool = False,
        device: Union[str, torch.device] = "cpu",
        calibrate: bool = True,
        **dataset_kwargs: Any,
    ) -> None:
        self._check_subclass_contract()
        self.root = Path(os.path.expanduser(str(root)))

        # Single md5 pass over the resources declared missing/corrupt; if a
        # download is requested, fetch exactly that subset (download_url
        # verifies each fetched file itself and raises on mismatch, so no
        # second verification pass is needed) and trust the result rather than
        # re-hashing everything again.
        missing = self._missing_resources()
        if missing and download:
            self._download_resources([(fn, md5) for fn, md5, _reason in missing])
            missing = []

        if missing:
            detail = ", ".join(f"{fn} ({reason})" for fn, _md5, reason in missing)
            raise RuntimeError(
                f"{type(self).__name__}: missing or corrupt data file(s) in "
                f"{self.raw_folder}: {detail}. Pass download=True to fetch them "
                f"from https://zenodo.org/records/{self.zenodo_record_id}"
            )

        # Published where _load_array can see it so a loader may put the array
        # straight on the destination device instead of building a host copy for
        # Dataset4dstem.__init__ to move (see load_npy_to_device).
        self._load_device = torch.device(device)
        self._load_dtype = self._load_target_dtype(dataset_kwargs)
        array = self._load_array()
        if _loader_owns(array, self._load_device):
            # `_load_array` allocated this tensor on the destination device, so the
            # defensive clone Dataset4dstem takes for a caller-supplied buffer is
            # pure cost -- 12.25 GiB and ~550 ms of first-touch cudaMalloc on the
            # figure-1 cube, which is the entire prize of loading it as float32 in
            # the first place. An explicit `copy=` from the caller still wins.
            dataset_kwargs.setdefault("copy", False)
        if self.host_dtype is not None:
            if isinstance(array, torch.Tensor):
                # Same cast, one device up. np.dtype -> torch.dtype via numpy itself
                # rather than a hand-written table.
                array = array.to(
                    dtype=torch.from_numpy(np.empty(0, dtype=self.host_dtype)).dtype
                )
            else:
                array = array.astype(self.host_dtype)

        physics = _metadata4dstem_from_physics(
            tuple(array.shape),
            energy=self.energy,
            semiconvergence_angle=self.semiconvergence_angle,
            scan_step=self.scan_step,
            rotation=self.rotation,
        )

        Dataset4dstem.__init__(
            self,
            array=array,
            name=type(self).__name__,
            origin=physics.origin,
            sampling=physics.sampling,
            units=physics.meta.units,
            meta=physics.meta,
            device=device,
            _token=type(self)._token,
            **dataset_kwargs,
        )

        if calibrate:
            self._calibrate_or_warn(placeholder_sampling=physics.sampling)

    # --- data access hook ---------------------------------------------------
    #: Device the constructed dataset is headed for, set by ``__init__`` before it
    #: calls :meth:`_load_array`. Class-level default so a subclass that calls the
    #: hook on its own still sees something sane.
    _load_device: torch.device = torch.device("cpu")
    #: dtype the constructed array will end up in, when a loader can produce it
    #: directly without changing a single bit; ``None`` = "read the file's own".
    #: Set by ``__init__`` before it calls :meth:`_load_array`, like
    #: ``_load_device``.
    _load_dtype: Optional[torch.dtype] = None

    def _load_target_dtype(self, dataset_kwargs: dict) -> Optional[torch.dtype]:
        """The dtype :meth:`_load_array` should aim for, or ``None`` for the file's.

        ``Dataset4dstem.__init__`` casts the array to float32 unless
        ``astype_float32=False``, so on the default path a loader that reads
        straight into float32 saves a whole extra cube -- the file-dtype one --
        rather than allocating it, filling it and immediately converting it away.

        ``None`` whenever the composition would not be the *identical single cast*:
        no float32 conversion requested, or a ``host_dtype`` that is not float32
        (that would round through a third dtype first, which can lose bits, e.g.
        float16).
        """
        if not dataset_kwargs.get("astype_float32", True):
            return None
        if self.host_dtype is not None and np.dtype(self.host_dtype) != np.dtype(
            np.float32
        ):
            return None
        return torch.float32

    def _load_array(self) -> Union[np.ndarray, torch.Tensor]:
        """Read (and repair) the raw files into a ``(ny, nx, M, M)`` array.

        May return either a numpy array on the host or a torch tensor already on
        ``self._load_device`` -- ``Dataset4dstem.__init__`` accepts both, and a
        loader that can write straight to the device saves the host round trip.
        """
        raise NotImplementedError

    # --- construction guards -------------------------------------------------
    def _check_subclass_contract(self) -> None:
        """Fail fast with a clear message if a subclass forgot a constant.

        Without this, a subclass that forgets e.g. ``energy`` constructs fine
        (every field defaults to ``None``/``[]``) and only breaks later, deep
        inside ``calibrate_reciprocal_from_bright_field`` or ``_load_array``,
        with a confusing ``TypeError``. An empty ``resources`` is equally
        silent: ``all(...)`` over zero items is ``True``, so
        ``_missing_resources`` would report nothing missing and every
        integrity check would be skipped.
        """
        missing_attrs = [
            name for name in _REQUIRED_CONSTANTS if getattr(self, name, None) is None
        ]
        if not self.resources:
            missing_attrs.append("resources")
        if missing_attrs:
            raise TypeError(
                f"{type(self).__name__} is missing required class attribute(s): "
                f"{', '.join(missing_attrs)}. Subclasses of PublicDataset4dstem "
                "must set these before they can be instantiated."
            )

    #: A real bright-field disk can't have a radius larger than this fraction
    #: of the detector's shorter side -- validated against the four datasets
    #: this base ships for: Gd2O3 (112 px, ratio 0.17), carbon (96 px, 0.25),
    #: Co3O4 (64 px, 0.22) and Au low-dose (96 px after crop, 0.17), all
    #: comfortably below; a uniform/featureless detector (no real disk edge)
    #: measures ~0.59.
    _MAX_PLAUSIBLE_RADIUS_FRACTION = 0.45

    def _calibrate_or_warn(self, placeholder_sampling: Sequence[float]) -> None:
        """Best-effort ``calibrate_reciprocal_from_bright_field()``.

        The underlying bright-field crop/measurement isn't robust to a disk
        that sits near/at the detector edge (an off-centre crop box can clip
        to an empty slice, raising ``RuntimeError``/``ValueError`` deep inside
        torch) or to genuinely featureless data: that case doesn't crash, but
        every other failure mode here warns loudly while this one would
        silently produce a plausible-looking, physically wrong ``dk`` that
        propagates uncaught into direct ptychography/tcDF/FFF. Caught via the
        one unambiguous tell available: a real disk's radius can't exceed the
        detector half-width, and in practice sits well under it (see
        ``_MAX_PLAUSIBLE_RADIUS_FRACTION``). Since all of this runs inside
        ``__init__`` after a potentially multi-GB load, a raised exception
        here would make an otherwise-valid dataset unconstructible -- warn and
        keep the placeholder ``dk`` instead.
        """
        try:
            dk = self.calibrate_reciprocal_from_bright_field()
            if not np.isfinite(dk) or dk <= 0:
                raise ValueError(f"non-finite or non-positive dk ({dk!r})")
            rBF = self.radius_bright_field
            detector_shape = tuple(int(s) for s in self.detector_shape)
            if rBF is not None and rBF > self._MAX_PLAUSIBLE_RADIUS_FRACTION * min(
                detector_shape
            ):
                raise ValueError(
                    f"implausible bright-field radius (rBF={rBF!r} px on a "
                    f"{detector_shape} detector) -- a real disk can't occupy "
                    "this much of the detector; this is either a broken "
                    "acquisition/threshold, or the data is already cropped "
                    "to its bright-field disk"
                )
        except torch.cuda.OutOfMemoryError:
            # torch.cuda.OutOfMemoryError subclasses RuntimeError, so it would
            # otherwise be swallowed by the broad clause below and the
            # placeholder dk kept -- clause order (this must come first) is
            # what makes it win. A GPU resource failure is not a calibration
            # failure: it must crash loudly rather than silently degrade dk
            # to a physically wrong (1.0, 1.0), which would then run
            # undetected through aberration fitting / SSB / tcDF / FFF.
            raise
        except (RuntimeError, ValueError, ZeroDivisionError) as exc:
            self.sampling = placeholder_sampling
            warnings.warn(
                f"{type(self).__name__}: automatic bright-field calibration "
                f"failed ({exc!r}). dk is left at its (1.0, 1.0) placeholder "
                "-- inspect the data and call "
                "calibrate_reciprocal_from_bright_field() manually.",
                # stacklevel=3: __init__ (frame 2) calls this method (frame 1,
                # the warn() call itself); the common case has no subclass
                # __init__ override, so frame 3 is the user's construction
                # line. Approximate if a subclass does add its own __init__.
                stacklevel=3,
            )

    # --- cache management ---------------------------------------------------
    @property
    def raw_folder(self) -> Path:
        """Flat cache directory -- the files live directly in ``root``."""
        return self.root

    def _missing_resources(self) -> list[tuple[str, str, str]]:
        """``[(filename, md5, reason)]`` for resources not present with the
        declared md5 -- at most one md5 pass per file. ``reason`` distinguishes
        "missing" from "corrupt" using ``Path.is_file()`` (a stat, not a
        hash), so it's free information, not an extra pass.
        """
        missing = []
        for filename, md5 in self.resources:
            fpath = self.raw_folder / filename
            if not self._verify_resource(fpath, md5):
                reason = "missing" if not fpath.is_file() else "corrupt (md5 mismatch)"
                missing.append((filename, md5, reason))
        return missing

    def _verify_resource(self, fpath: Path, md5: str) -> bool:
        """``check_integrity(fpath, md5)`` with a stat-based fast path.

        A resource already verified against this exact md5, and unchanged in
        size and ``mtime_ns`` since, is accepted on the strength of its stamp
        instead of being re-hashed; otherwise it is hashed exactly as before and
        a stamp is recorded on success. **Fails closed** -- a missing, stale,
        malformed or disagreeing stamp means the full hash runs, so a corrupt
        cache is still detected (``tests/test_public_you2026.py::
        test_base_rejects_corrupt_cache``).

        What this trades away, stated plainly: corruption that changes neither
        the size nor the mtime of an already-verified file -- silent bit rot on
        a filesystem without its own checksums -- is no longer caught on every
        construction. Truncated/partial/re-downloaded/edited files all move one
        of the two, and ``SCATTEREM_MD5_STAMP=0`` forces the unconditional hash
        back for a process that wants the stronger check.
        """
        if not fpath.is_file():
            return False
        stamps = _md5_stamps_enabled()
        # Stat BEFORE hashing: if the file changes while we read it the stamp
        # records the pre-change stat, so the next call disagrees and re-hashes.
        # The other order would stamp a mismatched digest as current.
        st = fpath.stat()
        if stamps and _md5_stamp_is_current(fpath, md5, st):
            return True
        if not check_md5(fpath, md5):
            _unlink_md5_stamp(fpath)
            return False
        if stamps:
            _write_md5_stamp(fpath, md5, st)
        return True

    def download(self) -> None:
        """Fetch every declared resource from Zenodo.

        Argument-free by design: the three sibling public-dataset bases
        (``chen2021.py``, ``sha2022.py``, ``you2024.py``) all declare
        ``def download(self) -> None``, and subclasses following that
        established pattern would ``TypeError`` on construction if this
        signature required an argument. ``__init__`` uses the private
        ``_download_resources`` instead so it can fetch only the subset it
        found missing/corrupt.
        """
        self._download_resources(self.resources)

    def _download_resources(self, resources: list[tuple[str, str]]) -> None:
        """Fetch exactly ``resources`` from Zenodo.

        ``download_url`` already no-ops for a file whose md5 matches -- but
        that no-op still costs a full md5 pass over the file, so ``__init__``
        passes a narrower list (just what it found missing) to skip
        re-hashing files already known to be valid.
        """
        os.makedirs(self.raw_folder, exist_ok=True)
        for filename, md5 in resources:
            download_url(
                zenodo_file_url(self.zenodo_record_id, filename),
                root=self.raw_folder,
                filename=filename,
                md5=md5,
            )

    # --- pretty-printing -----------------------------------------------------
    def _summary_rows(self) -> dict[str, Any]:
        rows = super()._summary_rows()
        rows["root"] = self.root
        if self.reference:
            rows["reference"] = self.reference
        if self.zenodo_record_id:
            rows["data"] = (
                f"https://zenodo.org/records/{self.zenodo_record_id} (CC-BY-4.0)"
            )
        return rows
