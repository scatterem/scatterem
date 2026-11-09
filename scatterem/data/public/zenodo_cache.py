"""Zenodo download / md5-verification machinery for the public datasets.

:class:`ZenodoResourceCache` is the flat on-disk cache behind
:class:`~scatterem.data.public.base.PublicDataset4DStem`, extracted so a
dataset class on any base can reuse it by composition instead of inheritance. The md5 stamps
that make re-verification cheap use :mod:`scatterem.data.public._stamp`.
"""

import os
from pathlib import Path

from scatterem.data.public._stamp import (
    stamp_is_current,
    stamp_path,
    unlink_stamp,
    write_stamp,
)
from scatterem.data.public.utils import check_md5, download_url, zenodo_file_url
from scatterem.utils.env import env_flag

#: Sidecar written beside a resource once its declared md5 has been verified, so
#: the next construction can skip the hash. md5 runs at ~0.8 GiB/s single-
#: threaded, so re-verifying a multi-GiB cube costs seconds of pure re-hashing on
#: every ``__init__`` (9.0 s for the 6.1 GiB Fig. 1 cube, 62 % of its
#: construction) and produces no byte anything downstream can see.
_MD5_STAMP_SUFFIX = ".md5-stamp.json"


def _md5_stamp_path(fpath: Path) -> Path:
    """Hidden md5 sidecar beside ``fpath`` (:func:`._stamp.stamp_path`)."""
    return stamp_path(fpath, _MD5_STAMP_SUFFIX)


def _md5_stamps_enabled() -> bool:
    """Read at call time, not import time, so a test/caller can force the full
    hash back on for one process."""
    return env_flag("SCATTEREM_MD5_STAMP", default=True)


class ZenodoResourceCache:
    """Flat on-disk cache of a Zenodo record's files, verified by md5.

    The download/verify machinery of :class:`PublicDataset4DStem`, extracted
    so a dataset class on any base can reuse it by
    composition instead of inheritance.

    Parameters
    ----------
    root : pathlib.Path or str
        Flat cache directory; files live directly in it.
    record_id : str or None
        Zenodo record the files are published under.
    resources : list of tuple of (str, str)
        ``(filename, md5)`` for every file the dataset needs.
    """

    def __init__(
        self,
        root: Path | str,
        record_id: str | None,
        resources: list[tuple[str, str]],
    ) -> None:
        self.root = Path(root)
        self.record_id = record_id
        self.resources = list(resources)

    def missing_resources(self) -> list[tuple[str, str, str]]:
        """``[(filename, md5, reason)]`` for resources not present with the
        declared md5 -- at most one md5 pass per file. ``reason`` distinguishes
        "missing" from "corrupt" using ``Path.is_file()`` (a stat, not a
        hash), so it's free information, not an extra pass.
        """
        missing = []
        for filename, md5 in self.resources:
            fpath = self.root / filename
            if not self.verify_resource(fpath, md5):
                reason = "missing" if not fpath.is_file() else "corrupt (md5 mismatch)"
                missing.append((filename, md5, reason))
        return missing

    def verify_resource(self, fpath: Path, md5: str) -> bool:
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
        construction. Truncated, partial and re-downloaded files move one of the
        two. An in-place edit of the same length usually moves the mtime, but
        not always: if it lands inside one filesystem timestamp tick of the
        verification, ``st_mtime_ns`` is byte-identical and the edit is invisible
        to the stamp (pinned by ``tests/test_public_you2026.py::
        test_md5_stamp_believes_an_edit_inside_one_mtime_tick``).
        ``SCATTEREM_MD5_STAMP=0`` forces the unconditional hash
        back for a process that wants the stronger check.
        """
        if not fpath.is_file():
            return False
        stamps = _md5_stamps_enabled()
        # Stat BEFORE hashing: if the file changes while we read it the stamp
        # records the pre-change stat, so the next call disagrees and re-hashes.
        # The other order would stamp a mismatched digest as current.
        st = fpath.stat()
        if stamps and stamp_is_current(fpath, _MD5_STAMP_SUFFIX, "md5", md5, st):
            return True
        if not check_md5(fpath, md5):
            unlink_stamp(fpath, _MD5_STAMP_SUFFIX)
            return False
        if stamps:
            write_stamp(fpath, _MD5_STAMP_SUFFIX, "md5", md5, st)
        return True

    def download(self, resources: list[tuple[str, str]] | None = None) -> None:
        """Fetch ``resources`` (default: all declared) from Zenodo.

        ``download_url`` already no-ops for a file whose md5 matches -- but
        that no-op still costs a full md5 pass over the file, so callers that
        know which subset is missing pass exactly that subset.
        """
        if resources is None:
            resources = self.resources
        os.makedirs(self.root, exist_ok=True)
        for filename, md5 in resources:
            download_url(
                zenodo_file_url(self.record_id or "", filename),
                root=self.root,
                filename=filename,
                md5=md5,
            )
