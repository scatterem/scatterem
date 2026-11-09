"""Verification stamps: a sidecar that lets a cache skip re-hashing a file.

A stamp is written beside a resource once its declared digest has been
verified. It records the digest and the file's size and ``mtime_ns``; a later
check believes it only if all three still agree. The Zenodo (md5) and Hugging
Face (sha256) caches both use these helpers, each with its own sidecar suffix
and digest key.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

__all__ = ["stamp_is_current", "stamp_path", "unlink_stamp", "write_stamp"]


def stamp_path(fpath: Path, suffix: str) -> Path:
    """Return the hidden sidecar path beside ``fpath``.

    The leading dot keeps it from colliding with a declared resource (those
    are the published file names).

    Parameters
    ----------
    fpath : pathlib.Path
        The verified resource.
    suffix : str
        Sidecar suffix, e.g. ``".md5-stamp.json"``.

    Returns
    -------
    pathlib.Path
        ``fpath.parent / f".{fpath.name}{suffix}"``.
    """
    return fpath.with_name(f".{fpath.name}{suffix}")


def stamp_is_current(
    fpath: Path, suffix: str, key: str, digest: str, st: os.stat_result
) -> bool:
    """Return whether ``fpath`` was verified against ``digest`` and is untouched.

    Deliberately conservative: any unreadable, malformed, absent or disagreeing
    stamp returns False and the caller falls back to hashing, so the stamp can
    only ever save work, never manufacture a pass.

    Parameters
    ----------
    fpath : pathlib.Path
        The resource.
    suffix : str
        Sidecar suffix.
    key : str
        Digest key in the stamp, e.g. ``"md5"`` or ``"sha256"``.
    digest : str
        Expected digest.
    st : os.stat_result
        ``fpath.stat()``, taken before any hashing.

    Returns
    -------
    bool
        True if the stamp records ``digest`` and the same size and
        ``mtime_ns``.
    """
    try:
        with open(stamp_path(fpath, suffix), "rb") as fh:
            stamp = json.load(fh)
    except (OSError, ValueError):
        return False
    if not isinstance(stamp, dict):
        return False
    return (
        stamp.get(key) == digest
        and stamp.get("size") == st.st_size
        and stamp.get("mtime_ns") == st.st_mtime_ns
    )


def write_stamp(
    fpath: Path, suffix: str, key: str, digest: str, st: os.stat_result
) -> None:
    """Record that ``fpath`` (as described by ``st``) has digest ``digest``.

    Best effort: a read-only or full cache directory must not turn a
    successful verification into a failure. Written to a temporary and
    ``os.replace``d, so an interrupted write leaves either the old stamp or
    none, never a half-written one that would be silently believed.

    Parameters
    ----------
    fpath : pathlib.Path
        The verified resource.
    suffix : str
        Sidecar suffix.
    key : str
        Digest key in the stamp.
    digest : str
        The verified digest.
    st : os.stat_result
        ``fpath.stat()``, taken before hashing.
    """
    dest = stamp_path(fpath, suffix)
    tmp = dest.with_name(dest.name + ".tmp")
    try:
        with open(tmp, "w") as fh:
            json.dump({key: digest, "size": st.st_size, "mtime_ns": st.st_mtime_ns}, fh)
        os.replace(tmp, dest)
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def unlink_stamp(fpath: Path, suffix: str) -> None:
    """Drop a stamp that no longer describes the file (also best effort).

    Parameters
    ----------
    fpath : pathlib.Path
        The resource.
    suffix : str
        Sidecar suffix.
    """
    try:
        os.unlink(stamp_path(fpath, suffix))
    except OSError:
        pass
