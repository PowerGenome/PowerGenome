"""Download data records from Zenodo.

Zenodo records often contain hundreds of files, some of them multi-GB, that
have to be fetched one at a time. This module provides a small programmatic
API for that job and the ``download_zenodo_data`` command line interface.

Identifiers
-----------
Every helper accepts any of the forms a user is likely to have copied from a
paper, a browser, or the Zenodo API:

* a bare record ID: ``22233228``
* a DOI: ``10.5281/zenodo.4426096``, ``doi:10.5281/zenodo.4426096``, or
  ``https://doi.org/10.5281/zenodo.4426096``
* a web URL: ``https://zenodo.org/records/22233228`` (or the legacy
  ``/record/22233228`` form)
* an API URL: ``https://zenodo.org/api/records/22233228``
* a single-file URL: ``https://zenodo.org/records/22233228/files/cpi_data.csv``
* any of the above on ``sandbox.zenodo.org``

Concept DOIs (the "always latest version" DOI) are resolved by following the
redirect from ``https://doi.org`` to the concrete record.

Examples
--------
::

    from pathlib import Path

    from powergenome.zenodo import download_record_files

    summary = download_record_files(
        "10.5281/zenodo.22233228",
        Path("data/zenodo"),
        files=["*.csv"],
    )
    print(summary.downloaded, summary.failed, summary.total_bytes)

Command line::

    download_zenodo_data 10.5281/zenodo.22233228 -sf settings
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import hmac
import logging
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union
from urllib.parse import quote, unquote, urlsplit

import requests

logger = logging.getLogger(__name__)

ZENODO_WEB_BASE = "https://zenodo.org"
ZENODO_SANDBOX_BASE = "https://sandbox.zenodo.org"
DOI_BASE = "https://doi.org"

#: Status codes that are worth retrying. Everything else in the 4xx range is
#: treated as permanent (bad record, bad key, auth problem, ...).
_RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

#: Base delay (seconds) for exponential backoff. Defined at module level so
#: tests can shrink it.
_RETRY_BASE_DELAY = 1.0
_RETRY_MAX_DELAY = 30.0

#: Maximum ``Retry-After`` value we are willing to wait for.
_MAX_RETRY_AFTER = 120.0

_DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")
_RECORD_PATH_RE = re.compile(r"^/(?:api/)?records?/(\d+)(?:/|$)")
_FILE_PATH_RE = re.compile(r"^files/(.+)$")


# ──────────────────────────────────────────────────────────────────────────────
# Errors
# ──────────────────────────────────────────────────────────────────────────────


class ZenodoError(RuntimeError):
    """Base error for anything that goes wrong talking to Zenodo."""


class ZenodoNotFoundError(ZenodoError):
    """A record, DOI, or file does not exist on Zenodo."""


class ChecksumError(ZenodoError):
    """A downloaded file does not match the checksum published by Zenodo."""


# ──────────────────────────────────────────────────────────────────────────────
# Identifier parsing
# ──────────────────────────────────────────────────────────────────────────────


class ZenodoIdentifier(tuple):
    """Parsed Zenodo identifier.

    Behaves exactly like the documented four-tuple
    ``(kind, record_id, doi, sandbox)`` so it can be unpacked, compared, and
    indexed as such. A single-file URL additionally exposes ``file_key`` (the
    fifth piece of information, kept off the tuple so unpacking stays stable).
    """

    def __new__(
        cls,
        kind: str,
        record_id: Optional[int],
        doi: Optional[str],
        sandbox: bool,
        file_key: Optional[str] = None,
    ) -> "ZenodoIdentifier":
        self = super().__new__(cls, (kind, record_id, doi, sandbox))
        self.file_key = file_key
        return self

    @property
    def kind(self) -> str:
        return self[0]

    @property
    def record_id(self) -> Optional[int]:
        return self[1]

    @property
    def doi(self) -> Optional[str]:
        return self[2]

    @property
    def sandbox(self) -> bool:
        return self[3]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"ZenodoIdentifier(kind={self.kind!r}, record_id={self.record_id!r}, "
            f"doi={self.doi!r}, sandbox={self.sandbox!r}, "
            f"file_key={self.file_key!r})"
        )


def _strip_doi_prefix(value: str) -> str:
    value = value.strip()
    if value.lower().startswith("doi:"):
        value = value[4:].strip()
    return value


def _split_file_key(
    path: str, identifier: Optional[str] = None
) -> Tuple[Optional[int], Optional[str]]:
    """Return ``(record_id, file_key)`` from a Zenodo URL path.

    A path that names the ``/files`` collection without a file key raises
    ``ValueError``: silently treating it as the whole record would download far
    more data than the user asked for.
    """
    match = _RECORD_PATH_RE.match(path)
    if not match:
        return None, None
    record_id = int(match.group(1))
    rest = path[match.end() :]
    if rest.rstrip("/") == "files":
        raise ValueError(
            "Zenodo file URL is missing its file key: "
            f"{(identifier or path)!r}. Use the record URL without '/files' to "
            "download the whole record, or put the file key after '/files/'."
        )
    file_match = _FILE_PATH_RE.match(rest)
    if not file_match:
        return record_id, None
    file_key = file_match.group(1)
    # API download links end in ``/content``; drop it.
    if file_key.endswith("/content"):
        file_key = file_key[: -len("/content")]
    file_key = file_key.rstrip("/")
    return record_id, file_key or None


def parse_zenodo_identifier(identifier: Union[str, int]) -> ZenodoIdentifier:
    """Parse any supported Zenodo identifier form.

    Parameters
    ----------
    identifier
        Record ID, DOI, or Zenodo URL. See the module docstring for the
        supported forms.

    Returns
    -------
    ZenodoIdentifier
        A four-tuple ``(kind, record_id, doi, sandbox)`` where ``kind`` is one
        of ``"record"``, ``"doi"``, or ``"url"``. Single-file URLs also expose
        ``file_key``.

    Raises
    ------
    ValueError
        If the identifier is not a recognised Zenodo form.
    """
    if isinstance(identifier, bool):
        raise ValueError(f"Invalid Zenodo identifier: {identifier!r}")
    if isinstance(identifier, int):
        if identifier <= 0:
            raise ValueError(f"Invalid Zenodo record ID: {identifier!r}")
        return ZenodoIdentifier("record", identifier, None, False)

    raw = str(identifier).strip()
    if not raw:
        raise ValueError("Invalid Zenodo identifier: empty string")

    # Bare record ID.
    if raw.isdigit():
        record_id = int(raw)
        if record_id <= 0:
            raise ValueError(f"Invalid Zenodo record ID: {raw!r}")
        return ZenodoIdentifier("record", record_id, None, False)

    # DOI with an optional ``doi:`` prefix.
    doi_candidate = _strip_doi_prefix(raw)
    if _DOI_RE.match(doi_candidate):
        return ZenodoIdentifier("doi", None, doi_candidate, False)

    parts = urlsplit(raw)
    if not parts.scheme or not parts.netloc:
        raise ValueError(
            f"Unrecognised Zenodo identifier: {identifier!r}. Expected a record "
            "ID (e.g. 22233228), a DOI (e.g. 10.5281/zenodo.22233228), or a "
            "Zenodo URL (e.g. https://zenodo.org/records/22233228)."
        )

    host = (parts.hostname or "").lower()
    sandbox = host == "sandbox.zenodo.org"
    if host in {"doi.org", "www.doi.org"}:
        doi = unquote(parts.path).lstrip("/")
        if not _DOI_RE.match(doi):
            raise ValueError(
                f"Unrecognised DOI in URL: {identifier!r}. Expected something "
                "like https://doi.org/10.5281/zenodo.22233228."
            )
        return ZenodoIdentifier("doi", None, doi, sandbox)

    if host not in {
        "zenodo.org",
        "www.zenodo.org",
        "sandbox.zenodo.org",
    }:
        raise ValueError(
            f"Unrecognised Zenodo host {host!r} in {identifier!r}. Expected "
            "zenodo.org, www.zenodo.org, sandbox.zenodo.org, or doi.org."
        )

    path = unquote(parts.path) or "/"

    # Zenodo's own DOI redirect path: https://zenodo.org/doi/10.5281/zenodo.123
    if path.startswith("/doi/"):
        doi = path[len("/doi/") :].strip("/")
        if _DOI_RE.match(doi):
            return ZenodoIdentifier("doi", None, doi, sandbox)
        raise ValueError(f"Unrecognised DOI path in {identifier!r}")

    record_id, file_key = _split_file_key(path, identifier=raw)
    if record_id is not None:
        return ZenodoIdentifier("url", record_id, None, sandbox, file_key)

    raise ValueError(
        f"Unrecognised Zenodo URL: {identifier!r}. Expected a record URL such "
        "as https://zenodo.org/records/22233228, an API URL such as "
        "https://zenodo.org/api/records/22233228, or a file URL such as "
        "https://zenodo.org/records/22233228/files/cpi_data.csv."
    )


def _web_base_for(sandbox: bool, web_base: str) -> str:
    if sandbox and web_base.rstrip("/") == ZENODO_WEB_BASE:
        return ZENODO_SANDBOX_BASE
    return web_base.rstrip("/")


def _record_id_from_url(url: str) -> Optional[int]:
    """Extract a record ID from a Zenodo URL or URL path."""
    if not url:
        return None
    path = urlsplit(url).path if "://" in url else url
    try:
        record_id, _ = _split_file_key(path)
    except ValueError:
        record_id = None
    if record_id is not None:
        return record_id
    # Tolerate ``/records/<id>/versions/latest`` style paths as well.
    match = re.search(r"/records?/(\d+)", path)
    return int(match.group(1)) if match else None


def _doi_suffix(doi: str) -> Optional[int]:
    suffix = doi.rsplit(".", 1)[-1]
    return int(suffix) if suffix.isdigit() else None


def _session_get(session: Any, url: str, **kwargs: Any) -> Any:
    if session is not None:
        return session.get(url, **kwargs)
    return requests.get(url, **kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ZenodoFile:
    key: str
    size: int
    checksum: Optional[str]
    download_url: str


@dataclass(frozen=True)
class ZenodoRecord:
    record_id: int
    title: str
    doi: Optional[str]
    concept_doi: Optional[str]
    version: Optional[str]
    publication_date: Optional[str]
    files: Tuple[ZenodoFile, ...]
    html_url: str
    api_url: str


@dataclass(frozen=True)
class FileResult:
    key: str
    path: Path
    status: str  # "downloaded" | "skipped" | "failed"
    bytes_written: int
    error: Optional[str] = None


@dataclass(frozen=True)
class DownloadSummary:
    record: ZenodoRecord
    destination: Path
    results: Tuple[FileResult, ...]

    @property
    def downloaded(self) -> Tuple[FileResult, ...]:
        return tuple(r for r in self.results if r.status == "downloaded")

    @property
    def skipped(self) -> Tuple[FileResult, ...]:
        return tuple(r for r in self.results if r.status == "skipped")

    @property
    def failed(self) -> Tuple[FileResult, ...]:
        return tuple(r for r in self.results if r.status == "failed")

    @property
    def ok(self) -> bool:
        return not self.failed

    @property
    def total_bytes(self) -> int:
        return sum(r.bytes_written for r in self.results)


# ──────────────────────────────────────────────────────────────────────────────
# Record lookup
# ──────────────────────────────────────────────────────────────────────────────


def resolve_record_id(
    identifier: Union[str, int],
    *,
    timeout: float = 30,
    session: Any = None,
    web_base: str = ZENODO_WEB_BASE,
) -> int:
    """Resolve any identifier to a concrete Zenodo record ID.

    Concept DOIs are resolved by following the ``https://doi.org`` redirect to
    the latest version, which is the only reliable way to resolve them:
    Zenodo's search API does not index concept DOIs. The API lookup
    ``/api/records/<suffix>`` is used as a fallback.

    Raises
    ------
    ZenodoNotFoundError
        If the identifier cannot be resolved to a record.
    """
    parsed = parse_zenodo_identifier(identifier)
    if parsed.record_id is not None:
        return parsed.record_id

    doi = parsed.doi
    assert doi is not None  # parse() guarantees doi or record_id
    base = _web_base_for(parsed.sandbox, web_base)

    # Primary path: follow the DOI redirect chain to the concrete record URL.
    response = None
    try:
        response = _session_get(
            session,
            f"{DOI_BASE}/{quote(doi, safe='/')}",
            allow_redirects=True,
            timeout=timeout,
        )
        candidates = [h.headers.get("Location") for h in response.history or []]
        candidates.append(response.url)
        for url in candidates:
            if not url:
                continue
            record_id = _record_id_from_url(url)
            if record_id is not None:
                logger.debug("Resolved DOI %s to record %s", doi, record_id)
                return record_id
    except requests.RequestException as exc:  # pragma: no cover - network
        logger.debug("DOI redirect resolution failed for %s: %s", doi, exc)
    finally:
        _close_quietly(response)

    # Fallback: assume the DOI suffix is the record ID.
    suffix = _doi_suffix(doi)
    if suffix is not None:
        response = None
        try:
            response = _session_get(
                session,
                f"{base}/api/records/{suffix}",
                timeout=timeout,
                allow_redirects=True,
            )
            if getattr(response, "ok", False):
                data = response.json()
                record_id = data.get("id")
                if record_id is not None:
                    logger.debug(
                        "Resolved DOI %s via API fallback to record %s",
                        doi,
                        record_id,
                    )
                    return int(record_id)
        except (requests.RequestException, ValueError) as exc:  # pragma: no cover
            logger.debug("DOI API fallback failed for %s: %s", doi, exc)
        finally:
            _close_quietly(response)

    raise ZenodoNotFoundError(
        f"Could not resolve Zenodo DOI {doi!r} to a record. Check the DOI, or "
        "pass the numeric record ID instead."
    )


def _close_quietly(response: Any) -> None:
    if response is None:
        return
    close = getattr(response, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # pragma: no cover - best effort
            pass


def _http_error(response: Any, url: str) -> ZenodoError:
    status = getattr(response, "status_code", None)
    body = ""
    try:
        body = (response.text or "").strip().replace("\n", " ")[:300]
    except Exception:  # pragma: no cover - best effort
        body = ""
    if status == 404:
        return ZenodoNotFoundError(f"Zenodo resource not found (404): {url}")
    return ZenodoError(f"Zenodo request failed ({status}) for {url}: {body}")


def _parse_record_files(
    data: dict,
    record_id: int,
    web_base: str,
    timeout: float,
    session: Any,
) -> list[ZenodoFile]:
    raw_files = data.get("files") or []
    entries: list[dict] = []

    if isinstance(raw_files, dict):
        entries.extend(raw_files.get("entries") or [])
        next_url = (raw_files.get("links") or {}).get("next")
        seen_pages = set()
        while next_url and next_url not in seen_pages:
            seen_pages.add(next_url)
            page = _session_get(
                session, next_url, timeout=timeout, allow_redirects=True
            )
            try:
                if not getattr(page, "ok", False):
                    raise _http_error(page, next_url)
                page_data = page.json()
            finally:
                _close_quietly(page)
            entries.extend(page_data.get("entries") or [])
            next_url = (page_data.get("links") or {}).get("next")
    elif isinstance(raw_files, (list, tuple)):
        entries.extend(raw_files)
    else:  # pragma: no cover - defensive
        raise ZenodoError(
            f"Unexpected 'files' value on record {record_id}: {type(raw_files)!r}"
        )

    files: list[ZenodoFile] = []
    for entry in entries:
        key = entry.get("key")
        if not key:
            logger.warning("Skipping file without a key on record %s", record_id)
            continue
        links = entry.get("links") or {}
        download_url = links.get("self") or (
            f"{web_base}/api/records/{record_id}/files/{quote(key)}/"
        )
        files.append(
            ZenodoFile(
                key=str(key),
                size=int(entry.get("size") or 0),
                checksum=entry.get("checksum"),
                download_url=download_url,
            )
        )
    return files


def get_record(
    identifier: Union[str, int],
    *,
    timeout: float = 30,
    session: Any = None,
    web_base: str = ZENODO_WEB_BASE,
) -> ZenodoRecord:
    """Fetch a record's metadata and file list from the Zenodo API."""
    parsed = parse_zenodo_identifier(identifier)
    base = _web_base_for(parsed.sandbox, web_base)
    record_id = parsed.record_id
    if record_id is None:
        record_id = resolve_record_id(
            identifier, timeout=timeout, session=session, web_base=base
        )

    api_url = f"{base}/api/records/{record_id}"
    response = _session_get(session, api_url, timeout=timeout, allow_redirects=True)
    try:
        if not getattr(response, "ok", False):
            raise _http_error(response, api_url)
        try:
            data = response.json()
        except ValueError as exc:
            raise ZenodoError(
                f"Zenodo returned invalid JSON for {api_url}: {exc}"
            ) from exc
    finally:
        _close_quietly(response)

    metadata = data.get("metadata") or {}
    links = data.get("links") or {}
    html_url = links.get("self_html") or f"{base}/records/{record_id}"
    canonical_api_url = links.get("self") or api_url

    return ZenodoRecord(
        record_id=int(data.get("id") or record_id),
        title=str(metadata.get("title") or data.get("title") or ""),
        doi=data.get("doi"),
        concept_doi=data.get("conceptdoi"),
        version=metadata.get("version"),
        publication_date=metadata.get("publication_date"),
        files=tuple(_parse_record_files(data, record_id, base, timeout, session)),
        html_url=html_url,
        api_url=canonical_api_url,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Download helpers
# ──────────────────────────────────────────────────────────────────────────────


def _safe_relative_path(key: str) -> Path:
    """Return *key* as a destination-relative path, or raise ``ValueError``.

    Zenodo file keys may contain ``/``; they must never escape the destination
    directory.
    """
    if not key or not key.strip():
        raise ValueError("empty file key")
    normalized = key.replace("\\", "/")
    pure = PurePosixPath(normalized)
    if pure.is_absolute():
        raise ValueError(f"absolute path is not allowed: {key!r}")
    parts = [part for part in pure.parts if part != "."]
    if not parts:
        raise ValueError(f"empty file key: {key!r}")
    if any(part == ".." for part in parts):
        raise ValueError(f"path traversal is not allowed: {key!r}")
    if ":" in parts[0]:
        raise ValueError(f"absolute path is not allowed: {key!r}")
    if "\x00" in normalized:
        raise ValueError(f"invalid character in file key: {key!r}")
    return Path(*parts)


def _expected_md5(checksum: Optional[str]) -> Optional[str]:
    if not checksum:
        return None
    algorithm, _, value = checksum.partition(":")
    if algorithm.lower() != "md5" or not value:
        return None
    return value.strip().lower()


def _md5_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    hasher = hashlib.md5()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def format_bytes(num_bytes: int) -> str:
    """Human readable byte count, e.g. ``1.2 MB``."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"  # pragma: no cover - unreachable


def _retry_delay(
    attempt: int, response: Any = None, *, base: Optional[float] = None
) -> float:
    """Exponential backoff with jitter; honours a sane ``Retry-After`` header."""
    if base is None:
        base = _RETRY_BASE_DELAY
    retry_after = None
    if response is not None:
        header = (getattr(response, "headers", None) or {}).get("Retry-After")
        try:
            retry_after = float(header) if header is not None else None
        except (TypeError, ValueError):
            retry_after = None
    if retry_after is not None and 0 <= retry_after <= _MAX_RETRY_AFTER:
        return retry_after
    delay = min(base * (2 ** max(attempt - 1, 0)), _RETRY_MAX_DELAY)
    return delay + random.uniform(0, delay * 0.25)


class _ProgressReporter:
    """Rate-limit progress logging so small files do not flood the log."""

    def __init__(self, key: str, total: int) -> None:
        self.key = key
        self.total = total
        self._started = time.monotonic()
        self._last_log = self._started
        self._last_fraction = 0.0
        self._logged = False

    def maybe_log(self, downloaded: int) -> None:
        now = time.monotonic()
        fraction = downloaded / self.total if self.total else 0.0
        elapsed = now - self._last_log
        if elapsed >= 5 or (self.total and fraction - self._last_fraction >= 0.05):
            logger.info(
                "downloading %s: %s / %s (%.0f%%)",
                self.key,
                format_bytes(downloaded),
                format_bytes(self.total),
                fraction * 100,
            )
            self._last_log = now
            self._last_fraction = fraction
            self._logged = True


def download_file(
    file: ZenodoFile,
    destination: Path,
    *,
    overwrite: bool = False,
    verify_checksum: bool = True,
    resume: bool = True,
    timeout: Union[float, Tuple[float, float]] = 60,
    retries: int = 3,
    session: Any = None,
    chunk_size: int = 1 << 20,
    progress: Optional[Callable[[str, int, int], None]] = None,
    verify_existing: bool = False,
) -> FileResult:
    """Download one Zenodo file to *destination*.

    The write is atomic: bytes go to ``<final>.part`` and the part file is
    renamed only after the size and (when available) the MD5 checksum match.
    Interrupted downloads can be resumed from the existing part file, and
    concurrent callers are safe as long as they do not share one
    ``requests.Session``.

    Returns a :class:`FileResult`; per-file failures never raise, so a caller
    can continue with the rest of a record.
    """
    destination = Path(destination)
    try:
        relative = _safe_relative_path(file.key)
    except ValueError as exc:
        logger.warning("refusing to write %r: %s", file.key, exc)
        return FileResult(
            key=file.key,
            path=destination,
            status="failed",
            bytes_written=0,
            error=f"unsafe file key: {exc}",
        )

    final_path = destination / relative
    part_path = final_path.with_name(final_path.name + ".part")
    expected_md5 = _expected_md5(file.checksum)
    if verify_checksum and file.checksum and expected_md5 is None:
        logger.debug(
            "No md5 available for %s (checksum=%r); skipping verification",
            file.key,
            file.checksum,
        )
        verify_checksum = False

    try:
        final_path.parent.mkdir(parents=True, exist_ok=True)

        # ── Skip existing complete files ──────────────────────────────────────
        if not overwrite and final_path.exists():
            existing_size = final_path.stat().st_size
            size_matches = not file.size or existing_size == file.size
            if not size_matches:
                logger.warning(
                    "existing %s has size %d, expected %d; downloading again",
                    file.key,
                    existing_size,
                    file.size,
                )
            else:
                if not verify_existing:
                    logger.info(
                        "skipped %s (already present, %s)",
                        file.key,
                        format_bytes(existing_size),
                    )
                    return FileResult(
                        key=file.key,
                        path=final_path,
                        status="skipped",
                        bytes_written=0,
                    )
                if expected_md5 is None:
                    logger.info(
                        "skipped %s (already present; no checksum to verify)",
                        file.key,
                    )
                    return FileResult(
                        key=file.key,
                        path=final_path,
                        status="skipped",
                        bytes_written=0,
                    )
                existing_md5 = _md5_of_file(final_path, chunk_size)
                if hmac.compare_digest(existing_md5, expected_md5):
                    logger.info(
                        "skipped %s (already present, checksum verified)", file.key
                    )
                    return FileResult(
                        key=file.key,
                        path=final_path,
                        status="skipped",
                        bytes_written=0,
                    )
                logger.warning(
                    "existing %s failed checksum verification; downloading again",
                    file.key,
                )

        # ── Resume bookkeeping ────────────────────────────────────────────────
        if overwrite or not resume:
            if part_path.exists():
                part_path.unlink()
        resume_offset = part_path.stat().st_size if part_path.exists() else 0
        if file.size and resume_offset > file.size:
            logger.warning(
                "partial file for %s is larger than expected; starting over",
                file.key,
            )
            part_path.unlink()
            resume_offset = 0
        if file.size and resume_offset == file.size and resume_offset > 0:
            logger.info("partial file for %s is complete; verifying", file.key)

        hasher = hashlib.md5()
        if resume_offset:
            # Hash the bytes already on disk once, then append.
            with open(part_path, "rb") as handle:
                while True:
                    chunk = handle.read(chunk_size)
                    if not chunk:
                        break
                    hasher.update(chunk)
            logger.info("resuming %s at %s", file.key, format_bytes(resume_offset))

        reporter = _ProgressReporter(file.key, file.size)
        bytes_written = 0
        attempt = 0
        max_attempts = max(1, int(retries) + 1)

        while True:
            if resume_offset and file.size and resume_offset == file.size:
                break  # nothing left to fetch

            attempt += 1
            headers = {}
            if resume_offset:
                headers["Range"] = f"bytes={resume_offset}-"

            response = None
            try:
                response = _session_get(
                    session,
                    file.download_url,
                    stream=True,
                    headers=headers,
                    timeout=timeout,
                    allow_redirects=True,
                )
                status = getattr(response, "status_code", 200)

                if status == 416 and "Range" in headers:
                    logger.info(
                        "server rejected resume range for %s; restarting",
                        file.key,
                    )
                    _close_quietly(response)
                    response = None
                    if part_path.exists():
                        part_path.unlink()
                    resume_offset = 0
                    hasher = hashlib.md5()
                    continue

                if not (200 <= status < 300):
                    error = _http_error(response, file.download_url)
                    if status in _RETRYABLE_STATUS and attempt < max_attempts:
                        delay = _retry_delay(attempt, response)
                        logger.warning(
                            "retrying %s after HTTP %s in %.1fs (attempt %d/%d)",
                            file.key,
                            status,
                            delay,
                            attempt,
                            max_attempts,
                        )
                        _close_quietly(response)
                        response = None
                        time.sleep(delay)
                        continue
                    if status not in _RETRYABLE_STATUS and part_path.exists():
                        # Unrecoverable (404, 403, ...): a stale partial file
                        # would make a rerun resume from the wrong bytes.
                        part_path.unlink(missing_ok=True)
                    raise error

                append = status == 206 and resume_offset > 0
                if not append and resume_offset:
                    # Server ignored the range; restart from zero.
                    logger.info(
                        "server returned HTTP %s for range request; restarting %s",
                        status,
                        file.key,
                    )
                    resume_offset = 0
                    hasher = hashlib.md5()
                    if part_path.exists():
                        part_path.unlink()

                mode = "ab" if append else "wb"
                with open(part_path, mode) as handle:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        hasher.update(chunk)
                        resume_offset += len(chunk)
                        bytes_written += len(chunk)
                        reporter.maybe_log(resume_offset)
                        if progress is not None:
                            try:
                                progress(file.key, resume_offset, file.size)
                            except Exception:  # pragma: no cover - user callback
                                logger.debug(
                                    "progress callback raised for %s",
                                    file.key,
                                    exc_info=True,
                                )
                break
            except (
                requests.ConnectionError,
                requests.Timeout,
                requests.exceptions.ChunkedEncodingError,
            ) as exc:
                if attempt < max_attempts:
                    delay = _retry_delay(attempt)
                    logger.warning(
                        "retrying %s after %s in %.1fs (attempt %d/%d)",
                        file.key,
                        type(exc).__name__,
                        delay,
                        attempt,
                        max_attempts,
                    )
                    time.sleep(delay)
                    continue
                raise
            finally:
                _close_quietly(response)

        # ── Verify and publish ────────────────────────────────────────────────
        actual_size = part_path.stat().st_size if part_path.exists() else 0
        if file.size and actual_size != file.size:
            message = f"incomplete download: got {actual_size} of {file.size} bytes"
            logger.warning("%s: %s (partial file kept for resume)", file.key, message)
            return FileResult(
                key=file.key,
                path=final_path,
                status="failed",
                bytes_written=bytes_written,
                error=message,
            )

        if verify_checksum and expected_md5 is not None:
            actual_md5 = (
                hasher.hexdigest()
                if actual_size
                else _md5_of_file(part_path, chunk_size)
            )
            if not hmac.compare_digest(actual_md5, expected_md5):
                part_path.unlink(missing_ok=True)
                error = ChecksumError(
                    f"checksum mismatch: expected md5:{expected_md5}, "
                    f"got md5:{actual_md5}"
                )
                logger.error("%s: %s (partial file removed)", file.key, error)
                return FileResult(
                    key=file.key,
                    path=final_path,
                    status="failed",
                    bytes_written=bytes_written,
                    error=str(error),
                )

        os.replace(part_path, final_path)
        logger.info(
            "downloaded %s (%s)", file.key, format_bytes(final_path.stat().st_size)
        )
        return FileResult(
            key=file.key,
            path=final_path,
            status="downloaded",
            bytes_written=bytes_written,
        )

    except KeyboardInterrupt:
        # Leave .part in place for a later resume.
        logger.warning("interrupted while downloading %s", file.key)
        raise
    except ZenodoError as exc:
        logger.warning("failed to download %s: %s", file.key, exc)
        return FileResult(
            key=file.key,
            path=final_path,
            status="failed",
            bytes_written=0,
            error=str(exc),
        )
    except OSError as exc:
        logger.warning("failed to write %s: %s", file.key, exc)
        return FileResult(
            key=file.key,
            path=final_path,
            status="failed",
            bytes_written=0,
            error=str(exc),
        )


def select_record_files(
    record: ZenodoRecord,
    files: Optional[Iterable[str]] = None,
    *,
    default_key: Optional[str] = None,
) -> list[ZenodoFile]:
    """Filter *record*'s files by exact key or glob pattern.

    Patterns are matched against the full key and against the basename, so
    ``--file '*.csv'`` and ``--file cpi_data.csv`` both work for nested keys.
    Every pattern must match at least one file.
    """
    if files is None:
        if default_key is None:
            return list(record.files)
        selectors = [default_key]
    elif isinstance(files, str):
        selectors = [files]
    else:
        selectors = [str(pattern) for pattern in files]

    unmatched: list[str] = []
    matched_keys: set[str] = set()
    for selector in selectors:
        matches = [
            f
            for f in record.files
            if fnmatch.fnmatchcase(f.key, selector)
            or fnmatch.fnmatchcase(PurePosixPath(f.key).name, selector)
        ]
        if not matches:
            unmatched.append(selector)
            continue
        matched_keys.update(f.key for f in matches)
    if unmatched:
        raise ZenodoError(
            "No files in Zenodo record "
            f"{record.record_id} match: {', '.join(repr(p) for p in unmatched)}. "
            "Use --list-files to see the available keys."
        )

    # Preserve the record's order and drop duplicates.
    return [f for f in record.files if f.key in matched_keys]


def download_record_files(
    identifier: Union[str, int],
    destination: Union[str, Path],
    *,
    files: Optional[Iterable[str]] = None,
    overwrite: bool = False,
    verify_checksums: bool = True,
    verify_existing: bool = False,
    max_workers: int = 4,
    timeout: Union[float, Tuple[float, float]] = 60,
    retries: int = 3,
    resume: bool = True,
    sandbox: bool = False,
    session: Any = None,
    record: Optional[ZenodoRecord] = None,
    progress: Optional[Callable[[str, int, int], None]] = None,
) -> DownloadSummary:
    """Download all (or selected) files from a Zenodo record.

    Parameters
    ----------
    identifier
        Any identifier accepted by :func:`parse_zenodo_identifier`.
    destination
        Directory the record's files are written into. Relative subdirectories
        in file keys are preserved.
    files
        Exact keys or glob patterns. When omitted, a single-file URL selects
        that one file and every other identifier selects the whole record.
    overwrite
        Always download, even when a complete file is already present.
    verify_checksums
        Verify each downloaded file against Zenodo's MD5 checksum.
    verify_existing
        Also verify already-present files before skipping them.
    max_workers
        Number of concurrent downloads.
    timeout, retries, resume
        Passed through to :func:`download_file`.
    sandbox
        Use ``https://sandbox.zenodo.org``. Sandbox URLs also select it.
    session
        Optional injected ``requests.Session``. A single session is used for
        all downloads, so an injected session must be thread-safe; without one
        each worker thread creates its own session.
    record
        Pre-fetched record metadata, to avoid a second API call.
    progress
        Optional ``progress(key, downloaded, total)`` callback.

    Returns
    -------
    DownloadSummary
        Results in the record's file order, regardless of completion order.
    """
    parsed = parse_zenodo_identifier(identifier)
    web_base = _web_base_for(sandbox or parsed.sandbox, ZENODO_WEB_BASE)

    if record is None:
        record = get_record(
            identifier, timeout=timeout, session=session, web_base=web_base
        )

    selected = select_record_files(record, files, default_key=parsed.file_key)

    destination = Path(destination)
    try:
        destination.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ZenodoError(
            f"Could not create destination directory {destination}: {exc}"
        ) from exc

    if not selected:
        logger.info("No files selected from record %s", record.record_id)
        return DownloadSummary(record=record, destination=destination, results=tuple())

    logger.info(
        "Downloading %d file(s) from %s to %s",
        len(selected),
        record.title or f"record {record.record_id}",
        destination,
    )

    def _download(file: ZenodoFile, thread_session: Any) -> FileResult:
        try:
            return download_file(
                file,
                destination,
                overwrite=overwrite,
                verify_checksum=verify_checksums,
                verify_existing=verify_existing,
                resume=resume,
                timeout=timeout,
                retries=retries,
                session=thread_session or session,
                progress=progress,
            )
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # one bad file must not stop the record
            logger.warning("failed to download %s: %s", file.key, exc)
            return FileResult(
                key=file.key,
                path=destination,
                status="failed",
                bytes_written=0,
                error=str(exc),
            )

    workers = max(1, int(max_workers))
    results: list[FileResult] = []

    if workers == 1 or len(selected) == 1:
        for file in selected:
            results.append(_download(file, session))
    else:
        thread_local = threading.local()

        def _worker(file: ZenodoFile) -> FileResult:
            thread_session = session
            if thread_session is None:
                thread_session = getattr(thread_local, "session", None)
                if thread_session is None:
                    thread_session = requests.Session()
                    thread_local.session = thread_session
            return _download(file, thread_session)

        executor = ThreadPoolExecutor(max_workers=workers)
        futures = [executor.submit(_worker, file) for file in selected]
        try:
            results = [future.result() for future in futures]
        except KeyboardInterrupt:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            logger.warning(
                "Interrupted; partial downloads were kept as .part files for resume"
            )
            raise
        else:
            executor.shutdown(wait=True)

    return DownloadSummary(
        record=record, destination=destination, results=tuple(results)
    )


# ──────────────────────────────────────────────────────────────────────────────
# Command line interface
# ──────────────────────────────────────────────────────────────────────────────


def _configure_logging(level: str) -> None:
    """Configure the ``powergenome`` logger once, even across repeated calls."""
    package_logger = logging.getLogger("powergenome")
    package_logger.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    for handler in package_logger.handlers:
        if getattr(handler, "_powergenome_cli_handler", False):
            return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)-8s %(name)s: %(message)s"))
    handler._powergenome_cli_handler = True  # type: ignore[attr-defined]
    package_logger.addHandler(handler)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="download_zenodo_data",
        description=(
            "Download data files from Zenodo records.\n\n"
            "IDENTIFIER may be a record ID, a DOI, or a Zenodo URL. Files can be "
            "filtered with --file and are written into the destination "
            "directory, preserving any relative subdirectories."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "identifiers",
        metavar="IDENTIFIER",
        nargs="+",
        help=(
            "One or more Zenodo record IDs (e.g. 22233228), DOIs "
            "(e.g. 10.5281/zenodo.22233228), or Zenodo URLs."
        ),
    )
    parser.add_argument(
        "-d",
        "--destination",
        dest="destination",
        type=str,
        default=None,
        help=(
            "Directory to write files into. Defaults to 'data_location' from the "
            "settings file passed with -sf/--settings-file."
        ),
    )
    parser.add_argument(
        "-sf",
        "--settings-file",
        dest="settings_file",
        type=str,
        default=None,
        help=(
            "Settings YAML file, or a directory of YAML files, used to read "
            "'data_location' when -d/--destination is not given."
        ),
    )
    parser.add_argument(
        "--file",
        dest="file_patterns",
        action="append",
        default=None,
        metavar="PATTERN",
        help=(
            "Only download files whose key or basename matches this glob "
            "pattern. Repeat the flag for several patterns."
        ),
    )
    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        default=False,
        help="Download files again even when a complete copy already exists.",
    )
    parser.add_argument(
        "--no-verify-checksums",
        dest="verify_checksums",
        action="store_false",
        default=True,
        help="Skip MD5 checksum verification of downloaded files.",
    )
    parser.add_argument(
        "--verify-existing",
        dest="verify_existing",
        action="store_true",
        default=False,
        help=(
            "Verify the checksum of files that are already present before "
            "skipping them (slower, but detects corrupted copies)."
        ),
    )
    parser.add_argument(
        "--jobs",
        dest="jobs",
        type=int,
        default=4,
        metavar="N",
        help="Number of concurrent downloads (default: 4).",
    )
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=False,
        help=(
            "Resolve records and log the files that would be downloaded without "
            "downloading anything."
        ),
    )
    parser.add_argument(
        "--list-files",
        dest="list_files",
        action="store_true",
        default=False,
        help=(
            "Print one line per file (key, size in bytes, checksum) and exit "
            "without creating directories or downloading."
        ),
    )
    parser.add_argument(
        "--sandbox",
        dest="sandbox",
        action="store_true",
        default=False,
        help="Use the Zenodo sandbox (sandbox.zenodo.org) instead of zenodo.org.",
    )
    parser.add_argument(
        "--timeout",
        dest="timeout",
        type=float,
        default=60.0,
        metavar="SECONDS",
        help="Network timeout in seconds (default: 60).",
    )
    parser.add_argument(
        "--retries",
        dest="retries",
        type=int,
        default=3,
        metavar="N",
        help="Retry attempts for transient failures (default: 3).",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        default=True,
        help="Do not resume partially downloaded .part files; start over.",
    )
    parser.add_argument(
        "--log-level",
        dest="log_level",
        type=str,
        default="INFO",
        metavar="LEVEL",
        help="Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO).",
    )
    return parser


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    return _build_parser().parse_args(argv)


def _absolute_data_path(value: Any, relative_base: Optional[Path]) -> Path:
    """Return *value* as an absolute path.

    ``relative_base`` is the directory a raw YAML value is relative to. When it
    is ``None`` the value has already been anchored by ``load_settings`` (which
    keeps relative input paths relative to the current directory), so resolving
    against the working directory is correct.
    """
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    if relative_base is None:
        return path.resolve()
    return (relative_base / path).resolve()


def _absolutise_data_location(settings: dict, relative_base: Optional[Path]) -> dict:
    """Normalise ``data_location`` to absolute paths, in place."""
    value = settings.get("data_location")
    if value is None:
        return settings
    if isinstance(value, (list, tuple)):
        settings["data_location"] = [
            _absolute_data_path(item, relative_base) for item in value
        ]
    else:
        settings["data_location"] = _absolute_data_path(value, relative_base)
    return settings


def _load_settings_dict(path: Path) -> dict:
    """Load settings from a directory (via ``load_settings``) or a YAML file.

    ``data_location`` is always returned as an absolute path. A directory load
    goes through :func:`powergenome.settings.load_settings`, which anchors
    relative values at the settings folder; a plain YAML file is resolved
    against its own parent. This keeps the CLI destination identical to the
    location the pipeline reads, without resolving the path twice.
    """
    if path.is_dir():
        from powergenome.settings import load_settings

        try:
            settings = dict(load_settings(path))
        except Exception as exc:  # missing required keys, etc.
            logger.debug(
                "load_settings(%s) failed (%s); merging YAML files directly",
                path,
                exc,
            )
            from ruamel.yaml import YAML

            merged: dict = {}
            for settings_file in sorted(path.glob("*.yml")):
                yaml = YAML(typ="safe")
                with open(settings_file, "r") as handle:
                    data = yaml.load(handle)
                if data:
                    merged.update(data)
            # Raw values are relative to the settings folder, matching
            # load_settings' own semantics.
            return _absolutise_data_location(merged, path.parent)
        # load_settings has already anchored relative values (possibly as a
        # path relative to the working directory).
        return _absolutise_data_location(settings, None)

    from ruamel.yaml import YAML

    yaml = YAML(typ="safe")
    with open(path, "r") as handle:
        settings = dict(yaml.load(handle) or {})
    return _absolutise_data_location(settings, path.parent)


def _resolve_destination(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> Path:
    if args.destination:
        return Path(args.destination).expanduser()

    if args.settings_file:
        settings_path = Path(args.settings_file).expanduser()
        if not settings_path.exists():
            parser.error(f"Settings path does not exist: {args.settings_file}")
        settings = _load_settings_dict(settings_path)
        data_location = settings.get("data_location")
        if isinstance(data_location, (list, tuple)):
            if not data_location:
                data_location = None
            else:
                logger.warning(
                    "Settings define %d data locations; using the first (%s). "
                    "Pass -d/--destination to choose explicitly.",
                    len(data_location),
                    data_location[0],
                )
                data_location = data_location[0]
        if not data_location:
            parser.error(
                f"No 'data_location' found in {args.settings_file}. Pass "
                "-d/--destination to choose a download directory."
            )
        destination = Path(data_location).expanduser()
        if not destination.is_absolute():
            # _load_settings_dict normally makes this absolute; fall back to
            # the working directory if a custom settings object did not.
            destination = destination.resolve()
        return destination

    parser.error(
        "A destination is required: pass -d/--destination PATH, or "
        "-sf/--settings-file to use 'data_location' from your settings."
    )
    raise AssertionError("unreachable")  # pragma: no cover


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for ``download_zenodo_data``.

    Returns the process exit code (0 success, 1 failure, 130 interrupted).
    Argument errors raise ``SystemExit`` with code 2 through argparse.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.log_level)

    destination: Optional[Path] = None
    if not args.list_files:
        destination = _resolve_destination(args, parser)
        if destination.exists() and not destination.is_dir():
            logger.error("Destination exists and is not a directory: %s", destination)
            return 1

    exit_code = 0
    totals = {"downloaded": 0, "skipped": 0, "failed": 0, "bytes": 0}
    records_seen = 0

    try:
        for identifier in args.identifiers:
            try:
                parsed = parse_zenodo_identifier(identifier)
            except ValueError as exc:
                logger.error("%s", exc)
                exit_code = 1
                totals["failed"] += 1
                continue

            sandbox = bool(args.sandbox or parsed.sandbox)
            web_base = _web_base_for(sandbox, ZENODO_WEB_BASE)
            selectors = args.file_patterns
            if selectors is None and parsed.file_key:
                selectors = [parsed.file_key]

            try:
                record = get_record(identifier, timeout=args.timeout, web_base=web_base)
            except (ZenodoError, requests.RequestException) as exc:
                logger.error("Could not fetch %s: %s", identifier, exc)
                exit_code = 1
                totals["failed"] += 1
                continue

            records_seen += 1

            try:
                selected = select_record_files(
                    record, selectors, default_key=parsed.file_key
                )
            except ZenodoError as exc:
                logger.error("%s", exc)
                exit_code = 1
                totals["failed"] += 1
                continue

            if args.list_files:
                for file in selected:
                    print(f"{file.key}\t{file.size}\t{file.checksum or ''}")
                continue

            assert destination is not None
            if args.dry_run:
                for file in selected:
                    try:
                        target = destination / _safe_relative_path(file.key)
                    except ValueError as exc:
                        logger.warning("would skip unsafe key %r: %s", file.key, exc)
                        continue
                    state = "exists" if target.exists() else "missing"
                    logger.info(
                        "would download %s (%s, %s)%s",
                        file.key,
                        format_bytes(file.size),
                        state,
                        "" if target.exists() else f" -> {target}",
                    )
                continue

            try:
                summary = download_record_files(
                    identifier,
                    destination,
                    files=selectors,
                    overwrite=args.overwrite,
                    verify_checksums=args.verify_checksums,
                    verify_existing=args.verify_existing,
                    max_workers=args.jobs,
                    timeout=args.timeout,
                    retries=args.retries,
                    resume=args.resume,
                    sandbox=sandbox,
                    record=record,
                )
            except (ZenodoError, requests.RequestException) as exc:
                logger.error("Download failed for %s: %s", identifier, exc)
                exit_code = 1
                totals["failed"] += 1
                continue

            totals["downloaded"] += len(summary.downloaded)
            totals["skipped"] += len(summary.skipped)
            totals["failed"] += len(summary.failed)
            totals["bytes"] += summary.total_bytes
            if summary.failed:
                exit_code = 1
                for result in summary.failed:
                    logger.error("failed: %s: %s", result.key, result.error)

    except KeyboardInterrupt:
        logger.warning(
            "Interrupted. Partial downloads remain as .part files and will "
            "resume on the next run."
        )
        return 130

    if args.list_files:
        return exit_code

    assert destination is not None
    if totals["downloaded"] or totals["skipped"]:
        logger.info(
            "Summary: %d downloaded, %d skipped, %d failed, %s written to %s",
            totals["downloaded"],
            totals["skipped"],
            totals["failed"],
            format_bytes(totals["bytes"]),
            destination,
        )
    else:
        # Nothing landed in the destination (bad identifiers, every selected
        # file failed, or a dry run), so do not claim anything was written.
        logger.info(
            "Summary: %d downloaded, %d skipped, %d failed",
            totals["downloaded"],
            totals["skipped"],
            totals["failed"],
        )
    if records_seen == 0 and exit_code == 0:  # pragma: no cover - defensive
        exit_code = 1
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
