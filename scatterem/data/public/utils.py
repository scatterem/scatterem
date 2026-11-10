# Download, checksum and archive helpers for the public datasets.
#
# Parts of this file derive from torchvision/datasets/utils.py
# (https://github.com/pytorch/vision), which is licensed under the
# BSD 3-Clause License:
#
#   Copyright (c) Soumith Chintala 2016,
#   All rights reserved.
#
# THIRD-PARTY-NOTICES.md (section "torchvision") holds the full licence text.
# The derived parts are the URL retrieval, the MD5 checks, the Google Drive
# download, and the archive extraction and decompression helpers.

import bz2
import gzip
import hashlib
import lzma
import os
import os.path
import pathlib
import re
import tarfile
import urllib
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from typing import (
    IO,
    Any,
    Literal,
)
from urllib.parse import urlparse

from tqdm import tqdm

USER_AGENT = "scatterem"


def _timeout_kwargs(timeout: float | None) -> dict[str, float]:
    """``urlopen`` keyword for a socket timeout; none means the global default."""
    return {} if timeout is None else {"timeout": timeout}


def _urlretrieve(
    url: str,
    filename: str | pathlib.Path,
    chunk_size: int = 1024 * 32,
    timeout: float | None = None,
    desc: str | None = None,
) -> None:
    with urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": USER_AGENT}),
        **_timeout_kwargs(timeout),
    ) as response:
        with (
            open(filename, "wb") as fh,
            tqdm(total=response.length, unit="B", unit_scale=True, desc=desc) as pbar,
        ):
            while chunk := response.read(chunk_size):
                fh.write(chunk)
                pbar.update(len(chunk))


def calculate_md5(fpath: str | pathlib.Path, chunk_size: int = 1024 * 1024) -> str:
    md5 = hashlib.md5(usedforsecurity=False)
    with open(fpath, "rb") as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()


def check_md5(fpath: str | pathlib.Path, md5: str, **kwargs: Any) -> bool:
    return md5 == calculate_md5(fpath, **kwargs)


def check_integrity(fpath: str | pathlib.Path, md5: str | None = None) -> bool:
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True
    return check_md5(fpath, md5)


def download_file_from_google_drive(
    file_id: str,
    root: str | pathlib.Path,
    filename: str | pathlib.Path | None = None,
    md5: str | None = None,
) -> None:
    """Download a Google Drive file from  and place it in root.

    Args:
        file_id (str): id of file to be downloaded
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the id of the file.
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """
    try:
        import gdown
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "To download files from GDrive, 'gdown' is required. You can install it with 'pip install gdown'."
        ) from exc

    root = os.path.expanduser(root)
    if not filename:
        filename = file_id
    fpath = os.fspath(os.path.join(root, filename))

    os.makedirs(root, exist_ok=True)

    if check_integrity(fpath, md5):
        return

    gdown.download(id=file_id, output=fpath, quiet=False, user_agent=USER_AGENT)

    if not check_integrity(fpath, md5):
        raise RuntimeError("File not found or corrupted.")


def _get_google_drive_file_id(url: str) -> str | None:
    parts = urlparse(url)

    if re.match(r"(drive|docs)[.]google[.]com", parts.netloc) is None:
        return None

    match = re.match(r"/file/d/(?P<id>[^/]*)", parts.path)
    if match is None:
        return None

    return match.group("id")


def _get_redirect_url(url: str, max_hops: int = 3, timeout: float | None = None) -> str:
    initial_url = url
    headers = {"Method": "HEAD", "User-Agent": USER_AGENT}

    for _ in range(max_hops + 1):
        with urllib.request.urlopen(
            urllib.request.Request(url, headers=headers), **_timeout_kwargs(timeout)
        ) as response:
            if response.url == url or response.url is None:
                return url

            url = response.url
    else:
        raise RecursionError(
            f"Request to {initial_url} exceeded {max_hops} redirects. The last redirect points to {url}."
        )


def download_url(
    url: str,
    root: str | pathlib.Path,
    filename: str | pathlib.Path | None = None,
    md5: str | None = None,
    max_redirect_hops: int = 3,
    timeout: float | None = None,
) -> None:
    """Download a file from a url and place it in root.

    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the basename of the URL
        md5 (str, optional): MD5 checksum of the download. If None, do not check
        max_redirect_hops (int, optional): Maximum number of redirect hops allowed
        timeout (float, optional): Socket timeout in seconds for each blocking
            network operation. If None, use the global default (no timeout).

    The progress bar names the file and the host of ``url``, so a log shows
    where each file came from.
    """
    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.fspath(os.path.join(root, filename))

    os.makedirs(root, exist_ok=True)

    # check if file is already present locally
    if check_integrity(fpath, md5):
        return

    desc = f"{os.path.basename(fpath)} from {urlparse(url).netloc}"

    # expand redirect chain if needed
    url = _get_redirect_url(url, max_hops=max_redirect_hops, timeout=timeout)

    # check if file is located on Google Drive
    file_id = _get_google_drive_file_id(url)
    if file_id is not None:
        return download_file_from_google_drive(file_id, root, filename, md5)

    # download the file
    try:
        _urlretrieve(url, fpath, timeout=timeout, desc=desc)
    except (urllib.error.URLError, OSError) as e:
        if url[:5] == "https":
            url = url.replace("https:", "http:")
            _urlretrieve(url, fpath, timeout=timeout, desc=desc)
        else:
            raise e

    # check integrity of downloaded file
    if not check_integrity(fpath, md5):
        raise RuntimeError("File not found or corrupted.")


def _extract_tar(
    from_path: str | pathlib.Path,
    to_path: str | pathlib.Path,
    compression: str | None,
) -> None:
    with tarfile.open(
        from_path, "r" if compression is None else _TAR_READ_MODES[compression]
    ) as tar:
        tar.extractall(to_path)


_ZIP_COMPRESSION_MAP: dict[str, int] = {
    ".bz2": zipfile.ZIP_BZIP2,
    ".xz": zipfile.ZIP_LZMA,
}
_TAR_READ_MODES: dict[str, Literal["r:gz", "r:bz2", "r:xz"]] = {
    ".gz": "r:gz",
    ".bz2": "r:bz2",
    ".xz": "r:xz",
}


def _extract_zip(
    from_path: str | pathlib.Path,
    to_path: str | pathlib.Path,
    compression: str | None,
) -> None:
    with zipfile.ZipFile(
        from_path,
        "r",
        compression=(
            _ZIP_COMPRESSION_MAP[compression] if compression else zipfile.ZIP_STORED
        ),
    ) as zip:
        zip.extractall(to_path)


def _extract_mat(
    from_path: str | pathlib.Path,
    to_path: str | pathlib.Path,
    compression: str | None,
) -> None:
    pass


_ARCHIVE_EXTRACTORS: dict[
    str,
    Callable[[str | pathlib.Path, str | pathlib.Path, str | None], None],
] = {
    ".tar": _extract_tar,
    ".zip": _extract_zip,
    ".mat": _extract_mat,
}
_COMPRESSED_FILE_OPENERS: dict[str, Callable[..., IO]] = {
    ".bz2": bz2.open,
    ".gz": gzip.open,
    ".xz": lzma.open,
}
_FILE_TYPE_ALIASES: dict[str, tuple[str | None, str | None | Literal[False]]] = {
    ".tbz": (".tar", ".bz2"),
    ".tbz2": (".tar", ".bz2"),
    ".tgz": (".tar", ".gz"),
    ".mat": (".mat", False),
}


def _detect_file_type(
    file: str | pathlib.Path,
) -> tuple[str, str | None, str | None | Literal[False]]:
    """Detect the archive type and/or compression of a file.

    Args:
        file (str): the filename

    Returns:
        (tuple): tuple of suffix, archive type, and compression

    Raises:
        RuntimeError: if file has no suffix or suffix is not supported
    """
    suffixes = pathlib.Path(file).suffixes
    if not suffixes:
        raise RuntimeError(
            f"File '{file}' has no suffixes that could be used to detect the archive type and compression."
        )
    suffix = suffixes[-1]

    # check if the suffix is a known alias
    if suffix in _FILE_TYPE_ALIASES:
        return suffix, *_FILE_TYPE_ALIASES[suffix]

    # check if the suffix is an archive type
    if suffix in _ARCHIVE_EXTRACTORS:
        return suffix, suffix, None

    # check if the suffix is a compression
    if suffix in _COMPRESSED_FILE_OPENERS:
        # check for suffix hierarchy
        if len(suffixes) > 1:
            suffix2 = suffixes[-2]

            # check if the suffix2 is an archive type
            if suffix2 in _ARCHIVE_EXTRACTORS:
                return suffix2 + suffix, suffix2, suffix

        return suffix, None, suffix

    valid_suffixes = sorted(
        set(_FILE_TYPE_ALIASES)
        | set(_ARCHIVE_EXTRACTORS)
        | set(_COMPRESSED_FILE_OPENERS)
    )
    raise RuntimeError(
        f"Unknown compression or archive type: '{suffix}'.\nKnown suffixes are: '{valid_suffixes}'."
    )


def _decompress(
    from_path: str | pathlib.Path,
    to_path: str | pathlib.Path | None = None,
    remove_finished: bool = False,
) -> pathlib.Path:
    r"""Decompress a file.

    The compression is automatically detected from the file name.

    Args:
        from_path (str): Path to the file to be decompressed.
        to_path (str): Path to the decompressed file. If omitted, ``from_path`` without compression extension is used.
        remove_finished (bool): If ``True``, remove the file after the extraction.

    Returns:
        (str): Path to the decompressed file.
    """
    suffix, archive_type, compression = _detect_file_type(from_path)
    if not compression:
        raise RuntimeError(f"Couldn't detect a compression from suffix {suffix}.")

    if to_path is None:
        to_path = pathlib.Path(
            os.fspath(from_path).replace(
                suffix, archive_type if archive_type is not None else ""
            )
        )

    # We don't need to check for a missing key here, since this was already done in _detect_file_type()
    compressed_file_opener = _COMPRESSED_FILE_OPENERS[compression]

    with compressed_file_opener(from_path, "rb") as rfh, open(to_path, "wb") as wfh:
        wfh.write(rfh.read())

    if remove_finished:
        os.remove(from_path)

    return pathlib.Path(to_path)


def extract_archive(
    from_path: str | pathlib.Path,
    to_path: str | pathlib.Path | None = None,
    remove_finished: bool = False,
) -> str | pathlib.Path:
    """Extract an archive.

    The archive type and a possible compression is automatically detected from the file name. If the file is compressed
    but not an archive, the file is decompressed.

    Args:
        from_path (str): Path to the file to be extracted.
        to_path (str): Path to the directory the file will be extracted to. If omitted, the directory of the file is
            used.
        remove_finished (bool): If ``True``, remove the file after the extraction.

    Returns:
        (str): Path to the directory the file was extracted to.
    """

    def path_or_str(ret_path: pathlib.Path) -> str | pathlib.Path:
        if isinstance(from_path, str):
            return os.fspath(ret_path)
        else:
            return ret_path

    if to_path is None:
        to_path = os.path.dirname(from_path)

    suffix, archive_type, compression = _detect_file_type(from_path)
    if not archive_type:
        ret_path = _decompress(
            from_path,
            os.path.join(to_path, os.path.basename(from_path).replace(suffix, "")),
            remove_finished=remove_finished,
        )
        return path_or_str(ret_path)

    # We don't need to check for a missing key here, since this was already done in _detect_file_type()
    extractor = _ARCHIVE_EXTRACTORS[archive_type]

    extractor(from_path, to_path, None if compression is False else compression)
    if remove_finished:
        os.remove(from_path)

    return path_or_str(pathlib.Path(to_path))


def download_and_extract_archive(
    url: str,
    download_root: str | pathlib.Path,
    extract_root: str | pathlib.Path | None = None,
    filename: str | pathlib.Path | None = None,
    md5: str | None = None,
    remove_finished: bool = False,
) -> None:
    download_root = os.path.expanduser(download_root)
    if extract_root is None:
        extract_root = download_root
    if not filename:
        filename = os.path.basename(url)

    download_url(url, download_root, filename, md5)

    archive = os.path.join(download_root, filename)
    extract_archive(archive, extract_root, remove_finished)


def zenodo_file_url(record_id: str | int, filename: str) -> str:
    """Direct-content download URL for one file in a Zenodo record.

    Args:
        record_id: Numeric Zenodo record id, e.g. ``"18008901"``.
        filename: File name exactly as listed in the record.

    Returns:
        A URL suitable for :func:`download_url`.
    """
    return f"https://zenodo.org/api/records/{record_id}/files/{filename}/content"
