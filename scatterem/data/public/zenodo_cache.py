"""Zenodo download / md5-verification machinery for the public datasets.

:class:`ZenodoResourceCache` is the flat on-disk cache behind
:class:`~scatterem.data.public.base.PublicDataset4DStem`, extracted so a
dataset class on any base can reuse it by composition instead of inheritance. The md5 stamps
that make re-verification cheap use :mod:`scatterem.data.public._stamp`.

Download sources. The Zenodo record is the canonical source and the one to
cite. For the records in :data:`ZENODO_MIRRORS`, the cache first downloads
each file from a byte-identical mirror on the Hugging Face Hub, which is
faster. It uses plain HTTPS (no ``huggingface_hub``) and the same md5 check.
If the mirror fails (HTTP error, timeout or md5 mismatch), the cache deletes
the bad file, warns once, and downloads the file from Zenodo.
``SCATTEREM_DATA_MIRROR=0`` switches the mirror off.
"""

import http.client
import os
import warnings
from pathlib import Path

from scatterem.data.public._stamp import (
    stamp_is_current,
    stamp_path,
    unlink_stamp,
    write_stamp,
)
from scatterem.data.public.utils import check_md5, download_url, zenodo_file_url
from scatterem.utils.env import env_flag

#: Commit of the Hugging Face dataset repository ``philipp-pelz/public-4dstem``
#: that the mirror URLs resolve against. A fixed commit, not ``main``, so that
#: later edits of the repository cannot change the files that scatterem fetches.
_HF_MIRROR_REVISION = "67b94bb6e79ddfda2f8be1d09479cb06a987d30d"

#: Zenodo record id -> base URL of a byte-identical mirror of its files.
#: ``<base>/<filename>`` is the direct URL of one file.
ZENODO_MIRRORS: dict[str, str] = {
    "18008901": (
        "https://huggingface.co/datasets/philipp-pelz/public-4dstem/resolve/"
        f"{_HF_MIRROR_REVISION}/mirrors/zenodo-18008901"
    ),
}

#: Socket timeout (s) of each blocking network operation of a mirror download.
#: A stalled mirror then fails over to Zenodo instead of hanging.
MIRROR_TIMEOUT_S = 60.0

#: The errors after which a mirror download falls back to Zenodo: HTTP errors
#: and timeouts (``urllib.error.URLError`` and ``TimeoutError`` are
#: ``OSError``), a connection that drops in the body (``HTTPException``) and
#: the md5 mismatch of ``download_url`` (``RuntimeError``).
_MIRROR_ERRORS = (OSError, http.client.HTTPException, RuntimeError)


def mirror_file_url(record_id: str | int | None, filename: str) -> str | None:
    """Mirror URL of one file of a Zenodo record, or None if it has no mirror."""
    base = ZENODO_MIRRORS.get(str(record_id)) if record_id is not None else None
    return None if base is None else f"{base}/{filename}"


def download_sources(record_id: str | int | None) -> str:
    """Where a download of ``record_id`` fetches its files from, for messages."""
    zenodo = f"https://zenodo.org/records/{record_id}"
    mirror = ZENODO_MIRRORS.get(str(record_id))
    if mirror is None:
        return zenodo
    return f"the Hugging Face mirror {mirror} or, if that fails, {zenodo}"


def _mirror_enabled() -> bool:
    """``SCATTEREM_DATA_MIRROR`` (default on), read at call time."""
    return env_flag("SCATTEREM_DATA_MIRROR", default=True)


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

    Files of a record in :data:`ZENODO_MIRRORS` download from the mirror
    first and from Zenodo if the mirror fails (see :meth:`download`).

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
        """Fetch ``resources`` (default: all declared), mirror first, then Zenodo.

        For a record in :data:`ZENODO_MIRRORS`, each file comes from the
        mirror. If that fails (HTTP error, timeout or md5 mismatch), the bad
        or partial file is deleted, one warning names the error, and the file
        comes from Zenodo. ``SCATTEREM_DATA_MIRROR=0`` skips the mirror. Both
        sources go through ``download_url`` with the declared md5.

        ``download_url`` already no-ops for a file whose md5 matches -- but
        that no-op still costs a full md5 pass over the file, so callers that
        know which subset is missing pass exactly that subset.
        """
        if resources is None:
            resources = self.resources
        os.makedirs(self.root, exist_ok=True)
        for filename, md5 in resources:
            if self._download_from_mirror(filename, md5):
                continue
            download_url(
                zenodo_file_url(self.record_id or "", filename),
                root=self.root,
                filename=filename,
                md5=md5,
            )

    def _download_from_mirror(self, filename: str, md5: str) -> bool:
        """True if ``filename`` came from the mirror with the right md5.

        False if the record has no mirror, the mirror is switched off, or the
        mirror download failed; a failed download leaves no file behind.
        """
        url = mirror_file_url(self.record_id, filename)
        if url is None or not _mirror_enabled():
            return False
        try:
            download_url(
                url,
                root=self.root,
                filename=filename,
                md5=md5,
                timeout=MIRROR_TIMEOUT_S,
            )
        except _MIRROR_ERRORS as exc:
            fpath = self.root / filename
            fpath.unlink(missing_ok=True)
            unlink_stamp(fpath, _MD5_STAMP_SUFFIX)
            warnings.warn(
                f"mirror download of {filename} failed ({type(exc).__name__}: "
                f"{exc}); downloading it from Zenodo record {self.record_id} "
                "instead",
                stacklevel=3,
            )
            return False
        return True
