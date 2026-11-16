"""Offline tests for :mod:`powergenome.zenodo`.

Every network interaction is replaced with a small fake session so the suite
runs without touching Zenodo.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from pathlib import Path

import pytest
import requests

import powergenome.zenodo as zenodo
from powergenome.zenodo import (
    ChecksumError,
    DownloadSummary,
    FileResult,
    ZenodoError,
    ZenodoFile,
    ZenodoNotFoundError,
    ZenodoRecord,
    download_file,
    download_record_files,
    get_record,
    parse_zenodo_identifier,
    resolve_record_id,
    select_record_files,
)

# ──────────────────────────────────────────────────────────────────────────────
# Fakes
# ──────────────────────────────────────────────────────────────────────────────


def md5_checksum(data: bytes) -> str:
    return "md5:" + hashlib.md5(data).hexdigest()


class FakeResponse:
    def __init__(
        self,
        status_code: int = 200,
        *,
        json_data=None,
        content: bytes = b"",
        headers=None,
        url: str = "",
        history=None,
    ) -> None:
        self.status_code = status_code
        self._json = json_data
        self.content = content
        self.headers = headers or {}
        self.url = url
        self.history = history or []
        self.closed = False

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", "replace")

    def json(self):
        if self._json is None:
            raise ValueError("no JSON body")
        return self._json

    def iter_content(self, chunk_size: int = 1):
        for index in range(0, len(self.content), chunk_size):
            yield self.content[index : index + chunk_size]

    def close(self) -> None:
        self.closed = True


class FakeSession:
    """Thread-safe stand-in for ``requests.Session``."""

    def __init__(self, routes=None, default=None) -> None:
        self.routes = dict(routes or {})
        self.default = default
        self.calls: list[tuple[str, dict]] = []
        self._lock = threading.RLock()

    def get(self, url: str, **kwargs):
        with self._lock:
            self.calls.append((url, kwargs))
            handler = self._lookup(url)
        if callable(handler):
            return handler(url, kwargs)
        return handler

    def _lookup(self, url: str):
        with self._lock:
            for key in (url, url.rstrip("/")):
                if key in self.routes:
                    value = self.routes[key]
                    if isinstance(value, list):
                        return value.pop(0) if len(value) > 1 else value[0]
                    return value
        if self.default is not None:
            return self.default
        return FakeResponse(status_code=404, content=b"not found", url=url)

    def urls(self) -> list[str]:
        return [url for url, _ in self.calls]


def make_file(key: str, data: bytes = b"", *, checksum=None, size=None) -> ZenodoFile:
    return ZenodoFile(
        key=key,
        size=len(data) if size is None else size,
        checksum=md5_checksum(data) if checksum is None else checksum,
        download_url=f"https://zenodo.org/api/records/1/files/{key}/content",
    )


def make_record(files, record_id: int = 22233228) -> ZenodoRecord:
    return ZenodoRecord(
        record_id=record_id,
        title="Test record",
        doi="10.5281/zenodo.22233228",
        concept_doi="10.5281/zenodo.22233227",
        version="1.0",
        publication_date="2026-01-01",
        files=tuple(files),
        html_url="https://zenodo.org/records/22233228",
        api_url="https://zenodo.org/api/records/22233228",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Identifier parsing
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "identifier,record_id",
    [
        ("22233228", 22233228),
        (22233228, 22233228),
        ("https://zenodo.org/records/22233228", 22233228),
        ("https://zenodo.org/records/22233228/", 22233228),
        ("https://zenodo.org/record/22233228", 22233228),
        ("https://zenodo.org/api/records/22233228", 22233228),
        (
            "https://zenodo.org/records/22233228/files/cpi_data.csv",
            22233228,
        ),
    ],
)
def test_parse_record_forms(identifier, record_id):
    parsed = parse_zenodo_identifier(identifier)
    kind, parsed_id, doi, sandbox = parsed
    assert parsed.kind == kind
    assert parsed_id == record_id
    assert doi is None
    assert sandbox is False


@pytest.mark.parametrize(
    "identifier",
    [
        "10.5281/zenodo.4426096",
        "doi:10.5281/zenodo.4426096",
        "https://doi.org/10.5281/zenodo.4426096",
        "https://zenodo.org/doi/10.5281/zenodo.4426096",
    ],
)
def test_parse_doi_forms(identifier):
    parsed = parse_zenodo_identifier(identifier)
    assert parsed.kind == "doi"
    assert parsed.record_id is None
    assert parsed.doi == "10.5281/zenodo.4426096"
    assert parsed.sandbox is False


def test_parse_single_file_url_returns_file_key():
    parsed = parse_zenodo_identifier(
        "https://zenodo.org/records/22233228/files/cpi_data.csv"
    )
    assert parsed.record_id == 22233228
    assert parsed.file_key == "cpi_data.csv"


def test_parse_single_file_url_nested_key():
    parsed = parse_zenodo_identifier(
        "https://zenodo.org/records/22233228/files/data/profiles.parquet"
    )
    assert parsed.file_key == "data/profiles.parquet"


def test_parse_api_content_url_strips_content_suffix():
    parsed = parse_zenodo_identifier(
        "https://zenodo.org/api/records/22233228/files/cpi_data.csv/content"
    )
    assert parsed.record_id == 22233228
    assert parsed.file_key == "cpi_data.csv"


def test_parse_sandbox_url_sets_sandbox():
    parsed = parse_zenodo_identifier(
        "https://sandbox.zenodo.org/records/1234/files/a.csv"
    )
    assert parsed.sandbox is True
    assert parsed.record_id == 1234
    assert parsed.file_key == "a.csv"


@pytest.mark.parametrize(
    "identifier",
    [
        "https://zenodo.org/records/22233228/files",
        "https://zenodo.org/records/22233228/files/",
        "https://zenodo.org/api/records/22233228/files/",
    ],
)
def test_parse_truncated_file_url_raises_value_error(identifier):
    with pytest.raises(ValueError, match="missing its file key"):
        parse_zenodo_identifier(identifier)


def test_parse_record_url_still_selects_whole_record():
    parsed = parse_zenodo_identifier("https://zenodo.org/records/22233228")
    assert parsed.record_id == 22233228
    assert parsed.file_key is None


@pytest.mark.parametrize(
    "identifier",
    [
        "",
        "not a record",
        "https://example.com/records/1",
        "https://zenodo.org/communities/powergenome",
        "doi:nope",
        True,
        -5,
    ],
)
def test_parse_invalid_identifiers_raise_value_error(identifier):
    with pytest.raises(ValueError):
        parse_zenodo_identifier(identifier)


def test_parsed_identifier_unpacks_as_four_tuple():
    kind, record_id, doi, sandbox = parse_zenodo_identifier(
        "https://zenodo.org/records/5/files/a.csv"
    )
    assert (kind, record_id, doi, sandbox) == ("url", 5, None, False)


# ──────────────────────────────────────────────────────────────────────────────
# DOI resolution
# ──────────────────────────────────────────────────────────────────────────────


def test_resolve_concept_doi_follows_redirect():
    redirect = FakeResponse(
        status_code=302,
        headers={"Location": "https://zenodo.org/doi/10.5281/zenodo.4426096"},
    )
    final = FakeResponse(
        status_code=200,
        url="https://zenodo.org/records/24174963",
        history=[redirect],
        json_data={"id": 24174963},
    )
    session = FakeSession(routes={"https://doi.org/10.5281/zenodo.4426096": final})
    assert resolve_record_id("10.5281/zenodo.4426096", session=session) == 24174963


def test_resolve_doi_uses_api_fallback_when_redirect_has_no_record():
    session = FakeSession(
        routes={
            "https://doi.org/10.5281/zenodo.4426096": FakeResponse(
                status_code=200, url="https://zenodo.org/"
            ),
            "https://zenodo.org/api/records/4426096": FakeResponse(
                json_data={"id": 4426096}
            ),
        }
    )
    assert resolve_record_id("10.5281/zenodo.4426096", session=session) == 4426096


def test_resolve_doi_raises_with_doi_in_message():
    session = FakeSession(
        routes={
            "https://doi.org/10.5281/zenodo.9999999": FakeResponse(
                status_code=200, url="https://zenodo.org/"
            ),
            "https://zenodo.org/api/records/9999999": FakeResponse(
                status_code=404, content=b"{}"
            ),
        }
    )
    with pytest.raises(ZenodoNotFoundError) as excinfo:
        resolve_record_id("10.5281/zenodo.9999999", session=session)
    assert "10.5281/zenodo.9999999" in str(excinfo.value)


def test_resolve_bare_record_id_does_not_use_network():
    session = FakeSession()
    assert resolve_record_id(22233228, session=session) == 22233228
    assert session.calls == []


# ──────────────────────────────────────────────────────────────────────────────
# Record parsing
# ──────────────────────────────────────────────────────────────────────────────


def _api_payload(files, **overrides):
    payload = {
        "id": 22233228,
        "doi": "10.5281/zenodo.22233228",
        "conceptdoi": "10.5281/zenodo.22233227",
        "metadata": {
            "title": "PowerGenome Input Data",
            "version": "2026.08.14.v1",
            "publication_date": "2026-09-02",
        },
        "files": files,
        "links": {
            "self": "https://zenodo.org/api/records/22233228",
            "self_html": "https://zenodo.org/records/22233228",
        },
    }
    payload.update(overrides)
    return payload


def test_get_record_list_shaped_files():
    payload = _api_payload(
        [
            {
                "key": "cpi_data.csv",
                "size": 1234,
                "checksum": "md5:abc",
                "links": {"self": "https://zenodo.org/api/records/1/files/x/content"},
            }
        ]
    )
    session = FakeSession(
        routes={
            "https://zenodo.org/api/records/22233228": FakeResponse(json_data=payload)
        }
    )
    record = get_record("22233228", session=session)
    assert record.record_id == 22233228
    assert record.title == "PowerGenome Input Data"
    assert record.doi == "10.5281/zenodo.22233228"
    assert record.concept_doi == "10.5281/zenodo.22233227"
    assert record.version == "2026.08.14.v1"
    assert record.publication_date == "2026-09-02"
    assert record.html_url == "https://zenodo.org/records/22233228"
    assert record.api_url == "https://zenodo.org/api/records/22233228"
    assert len(record.files) == 1
    assert record.files[0].key == "cpi_data.csv"
    assert record.files[0].size == 1234
    assert record.files[0].checksum == "md5:abc"
    assert record.files[0].download_url.endswith("/x/content")


def test_get_record_dict_shaped_files_and_fallback_download_url():
    payload = _api_payload(
        {
            "entries": [
                {"key": "a.csv", "size": 10, "checksum": "md5:aaa"},
                {"key": "nested/b.csv", "size": 20, "checksum": "md5:bbb"},
            ],
            "links": {},
        }
    )
    session = FakeSession(
        routes={
            "https://zenodo.org/api/records/22233228": FakeResponse(json_data=payload)
        }
    )
    record = get_record("22233228", session=session)
    assert [f.key for f in record.files] == ["a.csv", "nested/b.csv"]
    assert (
        record.files[0].download_url
        == "https://zenodo.org/api/records/22233228/files/a.csv/content"
    )
    assert (
        record.files[1].download_url
        == "https://zenodo.org/api/records/22233228/files/nested/b.csv/content"
    )


def test_get_record_uses_content_link_when_self_is_absent():
    payload = _api_payload(
        [
            {
                "key": "a.csv",
                "size": 10,
                "checksum": "md5:aaa",
                "links": {
                    "content": "https://zenodo.org/api/records/22233228/files/abc/content"
                },
            }
        ]
    )
    session = FakeSession(
        routes={
            "https://zenodo.org/api/records/22233228": FakeResponse(json_data=payload)
        }
    )
    record = get_record("22233228", session=session)
    assert (
        record.files[0].download_url
        == "https://zenodo.org/api/records/22233228/files/abc/content"
    )


def test_get_record_prefers_content_link_over_self():
    payload = _api_payload(
        [
            {
                "key": "a.csv",
                "size": 10,
                "checksum": "md5:aaa",
                "links": {
                    "self": "https://zenodo.org/api/records/22233228/files/self/content",
                    "content": "https://zenodo.org/api/records/22233228/files/c/content",
                },
            }
        ]
    )
    session = FakeSession(
        routes={
            "https://zenodo.org/api/records/22233228": FakeResponse(json_data=payload)
        }
    )
    record = get_record("22233228", session=session)
    assert record.files[0].download_url.endswith("/files/c/content")


def test_get_record_follows_file_pagination():
    first_page = {
        "entries": [{"key": "a.csv", "size": 1, "checksum": "md5:a"}],
        "links": {"next": "https://zenodo.org/api/records/1/files?page=2"},
    }
    second_page = {
        "entries": [{"key": "b.csv", "size": 2, "checksum": "md5:b"}],
        "links": {},
    }
    session = FakeSession(
        routes={
            "https://zenodo.org/api/records/1": FakeResponse(
                json_data=_api_payload(first_page, id=1)
            ),
            "https://zenodo.org/api/records/1/files?page=2": FakeResponse(
                json_data=second_page
            ),
        }
    )
    record = get_record("1", session=session)
    assert [f.key for f in record.files] == ["a.csv", "b.csv"]


def test_get_record_missing_optional_metadata_and_links():
    payload = {"id": 7, "files": [], "metadata": {}}
    session = FakeSession(
        routes={"https://zenodo.org/api/records/7": FakeResponse(json_data=payload)}
    )
    record = get_record("7", session=session)
    assert record.record_id == 7
    assert record.title == ""
    assert record.doi is None
    assert record.concept_doi is None
    assert record.version is None
    assert record.publication_date is None
    assert record.files == ()
    assert record.html_url == "https://zenodo.org/records/7"
    assert record.api_url == "https://zenodo.org/api/records/7"


def test_get_record_404_raises_not_found():
    session = FakeSession(
        routes={
            "https://zenodo.org/api/records/5": FakeResponse(
                status_code=404, content=b"no such record"
            )
        }
    )
    with pytest.raises(ZenodoNotFoundError):
        get_record("5", session=session)


def test_get_record_other_error_includes_status_and_body():
    session = FakeSession(
        routes={
            "https://zenodo.org/api/records/5": FakeResponse(
                status_code=500, content=b"boom"
            )
        }
    )
    with pytest.raises(ZenodoError) as excinfo:
        get_record("5", session=session)
    assert "500" in str(excinfo.value)
    assert "boom" in str(excinfo.value)


# ──────────────────────────────────────────────────────────────────────────────
# download_file
# ──────────────────────────────────────────────────────────────────────────────


def test_download_file_writes_content_and_renames_part(tmp_path):
    data = b"hello zenodo"
    file = make_file("cpi_data.csv", data)
    session = FakeSession(
        routes={file.download_url: FakeResponse(content=data, url=file.download_url)}
    )
    result = download_file(file, tmp_path, session=session)

    assert result.status == "downloaded"
    assert result.bytes_written == len(data)
    assert result.path == tmp_path / "cpi_data.csv"
    assert result.path.read_bytes() == data
    assert not (tmp_path / "cpi_data.csv.part").exists()


def test_download_file_preserves_nested_keys(tmp_path):
    data = b"nested"
    file = make_file("profiles/wind.csv", data)
    session = FakeSession(
        routes={file.download_url: FakeResponse(content=data, url=file.download_url)}
    )
    result = download_file(file, tmp_path, session=session)
    assert result.status == "downloaded"
    assert result.path == tmp_path / "profiles" / "wind.csv"
    assert result.path.read_bytes() == data


def test_download_file_skips_existing_same_size(tmp_path):
    data = b"already here"
    file = make_file("a.csv", data)
    (tmp_path / "a.csv").write_bytes(data)
    session = FakeSession()
    result = download_file(file, tmp_path, session=session)

    assert result.status == "skipped"
    assert result.bytes_written == 0
    assert session.calls == []


def test_download_file_overwrite_forces_rewrite(tmp_path):
    file = make_file("a.csv", b"new contents")
    (tmp_path / "a.csv").write_bytes(b"old")
    session = FakeSession(
        routes={file.download_url: FakeResponse(content=b"new contents")}
    )
    result = download_file(file, tmp_path, overwrite=True, session=session)
    assert result.status == "downloaded"
    assert (tmp_path / "a.csv").read_bytes() == b"new contents"


def test_download_file_verify_existing_redownloads_corrupt_copy(tmp_path):
    file = make_file("a.csv", b"good data")
    (tmp_path / "a.csv").write_bytes(b"bad data!")
    session = FakeSession(
        routes={file.download_url: FakeResponse(content=b"good data")}
    )
    result = download_file(file, tmp_path, session=session, verify_existing=True)
    assert result.status == "downloaded"
    assert (tmp_path / "a.csv").read_bytes() == b"good data"


def test_download_file_checksum_mismatch_fails_and_removes_output(tmp_path):
    file = make_file("a.csv", b"payload", checksum="md5:" + "0" * 32)
    session = FakeSession(routes={file.download_url: FakeResponse(content=b"payload")})
    result = download_file(file, tmp_path, session=session)

    assert result.status == "failed"
    assert "checksum" in (result.error or "")
    assert not (tmp_path / "a.csv").exists()
    assert not (tmp_path / "a.csv.part").exists()


def test_download_file_resume_sends_range_and_appends(tmp_path):
    partial = b"abc"
    tail = b"def"
    file = make_file("a.csv", partial + tail)
    (tmp_path / "a.csv.part").write_bytes(partial)
    session = FakeSession(
        routes={
            file.download_url: FakeResponse(
                status_code=206, content=tail, url=file.download_url
            )
        }
    )
    result = download_file(file, tmp_path, session=session)

    assert result.status == "downloaded"
    assert result.bytes_written == len(tail)
    assert (tmp_path / "a.csv").read_bytes() == partial + tail
    sent_headers = session.calls[0][1]["headers"]
    assert sent_headers["Range"] == f"bytes={len(partial)}-"


def test_download_file_restarts_when_server_ignores_range(tmp_path):
    partial = b"old-partial"
    full = b"full content"
    file = make_file("a.csv", full)
    (tmp_path / "a.csv.part").write_bytes(partial)
    session = FakeSession(routes={file.download_url: FakeResponse(content=full)})
    result = download_file(file, tmp_path, session=session)

    assert result.status == "downloaded"
    assert (tmp_path / "a.csv").read_bytes() == full


def test_download_file_no_resume_starts_over(tmp_path):
    full = b"complete"
    file = make_file("a.csv", full)
    (tmp_path / "a.csv.part").write_bytes(b"stale")
    session = FakeSession(routes={file.download_url: FakeResponse(content=full)})
    result = download_file(file, tmp_path, resume=False, session=session)

    assert result.status == "downloaded"
    assert (tmp_path / "a.csv").read_bytes() == full
    assert session.calls[0][1]["headers"] == {}


@pytest.mark.parametrize("key", ["../escape.csv", "/abs/path.csv", "", "a/../../b.csv"])
def test_download_file_rejects_unsafe_keys(tmp_path, key):
    file = make_file(key, b"x")
    result = download_file(file, tmp_path, session=FakeSession())
    assert result.status == "failed"
    assert not (tmp_path.parent / "escape.csv").exists()


def test_download_file_rejects_symlinked_subdirectory(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    destination = tmp_path / "data"
    destination.mkdir()
    (destination / "profiles").symlink_to(outside, target_is_directory=True)

    file = make_file("profiles/wind.csv", b"wind")
    session = FakeSession(routes={file.download_url: FakeResponse(content=b"wind")})
    result = download_file(file, destination, session=session)

    assert result.status == "failed"
    assert "outside the destination" in (result.error or "")
    assert list(outside.iterdir()) == []
    assert session.calls == []


def test_download_file_rejects_nested_write_through_symlinked_subdirectory(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    destination = tmp_path / "data"
    destination.mkdir()
    (destination / "profiles").symlink_to(outside, target_is_directory=True)

    file = make_file("profiles/sub/wind.csv", b"wind")
    session = FakeSession(routes={file.download_url: FakeResponse(content=b"wind")})
    result = download_file(file, destination, session=session)

    assert result.status == "failed"
    assert not (outside / "sub").exists()


def test_download_file_rejects_symlinked_final_file(tmp_path):
    outside = tmp_path / "outside.csv"
    outside.write_bytes(b"original")
    destination = tmp_path / "data"
    destination.mkdir()
    (destination / "a.csv").symlink_to(outside)

    file = make_file("a.csv", b"new data that is longer")
    session = FakeSession(
        routes={file.download_url: FakeResponse(content=b"new data that is longer")}
    )
    result = download_file(file, destination, session=session)

    assert result.status == "failed"
    assert "symlink" in (result.error or "")
    assert outside.read_bytes() == b"original"
    assert session.calls == []


def test_download_file_rejects_symlinked_part_file(tmp_path):
    outside = tmp_path / "outside.part"
    outside.write_bytes(b"")
    destination = tmp_path / "data"
    destination.mkdir()
    (destination / "a.csv.part").symlink_to(outside)

    file = make_file("a.csv", b"payload")
    session = FakeSession(routes={file.download_url: FakeResponse(content=b"payload")})
    result = download_file(file, destination, session=session)

    assert result.status == "failed"
    assert "symlink" in (result.error or "")
    assert outside.read_bytes() == b""


def test_download_file_allows_symlinked_destination_root(tmp_path):
    """A symlinked destination is not an escape; files land in the real root."""
    real_dest = tmp_path / "real_dest"
    real_dest.mkdir()
    destination = tmp_path / "dest"
    destination.symlink_to(real_dest, target_is_directory=True)

    flat = make_file("cpi_data.csv", b"flat")
    nested = make_file("profiles/wind.csv", b"nested")
    session = FakeSession(
        routes={
            flat.download_url: FakeResponse(content=b"flat", url=flat.download_url),
            nested.download_url: FakeResponse(
                content=b"nested", url=nested.download_url
            ),
        }
    )

    flat_result = download_file(flat, destination, session=session)
    nested_result = download_file(nested, destination, session=session)

    assert flat_result.status == "downloaded"
    assert nested_result.status == "downloaded"
    assert (real_dest / "cpi_data.csv").read_bytes() == b"flat"
    assert (real_dest / "profiles" / "wind.csv").read_bytes() == b"nested"
    assert not (real_dest / "cpi_data.csv.part").exists()
    assert not (real_dest / "profiles" / "wind.csv.part").exists()


def test_download_file_rejects_escape_inside_symlinked_destination_root(tmp_path):
    """Resolving the root must not excuse an escaping symlink beneath it."""
    real_dest = tmp_path / "real_dest"
    real_dest.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (real_dest / "profiles").symlink_to(outside, target_is_directory=True)
    destination = tmp_path / "dest"
    destination.symlink_to(real_dest, target_is_directory=True)

    file = make_file("profiles/wind.csv", b"wind")
    session = FakeSession(routes={file.download_url: FakeResponse(content=b"wind")})
    result = download_file(file, destination, session=session)

    assert result.status == "failed"
    assert "outside the destination" in (result.error or "")
    assert list(outside.iterdir()) == []
    assert session.calls == []


def test_download_file_cancel_event_mid_stream_keeps_part(tmp_path):
    event = threading.Event()

    class CancellingResponse(FakeResponse):
        def iter_content(self, chunk_size=1):
            yield b"first"
            event.set()

    data = b"firstsecond"
    file = make_file("a.csv", data)
    session = FakeSession(
        routes={
            file.download_url: CancellingResponse(content=data, url=file.download_url)
        }
    )
    result = download_file(file, tmp_path, session=session, cancel_event=event)

    assert result.status == "failed"
    assert result.error == "cancelled"
    assert result.bytes_written == len(b"first")
    assert (tmp_path / "a.csv.part").read_bytes() == b"first"
    assert not (tmp_path / "a.csv").exists()


def test_download_file_cancel_event_before_call_skips_stream(tmp_path):
    event = threading.Event()
    event.set()

    file = make_file("a.csv", b"payload")
    session = FakeSession(routes={file.download_url: FakeResponse(content=b"payload")})
    result = download_file(file, tmp_path, session=session, cancel_event=event)

    assert result.status == "failed"
    assert result.error == "cancelled"
    assert session.calls == []
    assert not (tmp_path / "a.csv.part").exists()
    assert not (tmp_path / "a.csv").exists()


def test_download_file_cancel_event_interrupts_retry_backoff(tmp_path):
    event = threading.Event()

    def busy(url, kwargs):
        event.set()
        return FakeResponse(
            status_code=503, headers={"Retry-After": "30"}, content=b"busy", url=url
        )

    file = make_file("a.csv", b"never arrives")
    session = FakeSession(routes={file.download_url: busy})

    started = time.monotonic()
    result = download_file(file, tmp_path, session=session, cancel_event=event)
    elapsed = time.monotonic() - started

    assert result.status == "failed"
    assert result.error == "cancelled"
    assert elapsed < 1  # the 30 s Retry-After wait was interrupted
    assert len(session.calls) == 1


def test_download_file_retries_transient_503(monkeypatch, tmp_path):
    monkeypatch.setattr(zenodo, "_RETRY_BASE_DELAY", 0)
    data = b"recovered after retry"
    file = make_file("a.csv", data)
    session = FakeSession(
        routes={
            file.download_url: [
                FakeResponse(status_code=503, content=b"busy"),  # no Retry-After
                FakeResponse(status_code=200, content=data),
            ]
        }
    )
    result = download_file(file, tmp_path, session=session, retries=2)

    assert result.status == "downloaded"
    assert result.bytes_written == len(data)
    assert len(session.calls) == 2
    assert (tmp_path / "a.csv").read_bytes() == data
    assert not (tmp_path / "a.csv.part").exists()


def test_download_file_retry_exhaustion_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(zenodo, "_RETRY_BASE_DELAY", 0)
    file = make_file("a.csv", b"never arrives")
    session = FakeSession(
        routes={file.download_url: FakeResponse(status_code=503, content=b"overloaded")}
    )
    result = download_file(file, tmp_path, session=session, retries=1)

    assert result.status == "failed"
    assert "503" in (result.error or "")
    assert len(session.calls) == 2
    assert not (tmp_path / "a.csv").exists()
    assert not (tmp_path / "a.csv.part").exists()


def test_download_file_connection_error_exhaustion_returns_failed(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(zenodo, "_RETRY_BASE_DELAY", 0)

    def boom(url, kwargs):
        raise requests.ConnectionError("connection reset")

    file = make_file("a.csv", b"never arrives")
    session = FakeSession(routes={file.download_url: boom})
    result = download_file(file, tmp_path, session=session, retries=1)

    assert result.status == "failed"
    assert "connection reset" in (result.error or "")
    assert len(session.calls) == 2


@pytest.mark.parametrize(
    "exc",
    [
        requests.TooManyRedirects("exceeded 30 redirects"),
        requests.exceptions.InvalidURL("no host supplied"),
    ],
)
def test_download_file_transport_errors_return_failed(tmp_path, exc):
    def boom(url, kwargs):
        raise exc

    file = make_file("a.csv", b"payload")
    session = FakeSession(routes={file.download_url: boom})
    result = download_file(file, tmp_path, session=session)

    assert result.status == "failed"
    assert str(exc) in (result.error or "")
    assert len(session.calls) == 1
    assert not (tmp_path / "a.csv").exists()


def test_download_file_short_read_fails_and_keeps_part(tmp_path):
    file = make_file("a.csv", b"abc", size=10)
    session = FakeSession(routes={file.download_url: FakeResponse(content=b"abc")})
    result = download_file(file, tmp_path, session=session)

    assert result.status == "failed"
    assert "incomplete" in (result.error or "")
    assert not (tmp_path / "a.csv").exists()
    assert (tmp_path / "a.csv.part").read_bytes() == b"abc"


def test_download_file_keyboard_interrupt_keeps_part(tmp_path):
    class InterruptingResponse(FakeResponse):
        def iter_content(self, chunk_size=1):
            yield b"partial"
            raise KeyboardInterrupt

    file = make_file("a.csv", b"partial rest")
    session = FakeSession(routes={file.download_url: InterruptingResponse(content=b"")})
    with pytest.raises(KeyboardInterrupt):
        download_file(file, tmp_path, session=session)

    assert (tmp_path / "a.csv.part").read_bytes() == b"partial"
    assert not (tmp_path / "a.csv").exists()


def test_download_file_verify_existing_skips_identical(tmp_path):
    data = b"already correct"
    file = make_file("a.csv", data)
    (tmp_path / "a.csv").write_bytes(data)
    session = FakeSession()
    result = download_file(file, tmp_path, session=session, verify_existing=True)

    assert result.status == "skipped"
    assert session.calls == []


def test_download_file_reports_404_without_retry(tmp_path):
    file = make_file("missing.csv", b"x")
    session = FakeSession(
        routes={file.download_url: FakeResponse(status_code=404, content=b"gone")}
    )
    result = download_file(file, tmp_path, session=session)
    assert result.status == "failed"
    assert "404" in (result.error or "")
    assert len(session.calls) == 1


# ──────────────────────────────────────────────────────────────────────────────
# download_record_files
# ──────────────────────────────────────────────────────────────────────────────


def _record_with_content():
    contents = {
        "a.csv": b"alpha",
        "b.csv": b"bravo",
        "data/cpi_data.csv": b"cpi",
    }
    files = [make_file(key, data) for key, data in contents.items()]
    routes = {f.download_url: FakeResponse(content=contents[f.key]) for f in files}
    return make_record(files), FakeSession(routes=routes)


def test_download_record_files_downloads_all_in_order(tmp_path):
    record, session = _record_with_content()
    summary = download_record_files(
        "22233228", tmp_path, record=record, session=session, max_workers=3
    )

    assert isinstance(summary, DownloadSummary)
    assert [r.key for r in summary.results] == ["a.csv", "b.csv", "data/cpi_data.csv"]
    assert [r.status for r in summary.results] == ["downloaded"] * 3
    assert summary.ok
    assert summary.total_bytes == 5 + 5 + 3
    assert (tmp_path / "data" / "cpi_data.csv").read_bytes() == b"cpi"


def test_download_record_files_applies_glob_filters(tmp_path):
    record, session = _record_with_content()
    summary = download_record_files(
        "22233228", tmp_path, files=["*.csv"], record=record, session=session
    )
    assert [r.key for r in summary.results] == ["a.csv", "b.csv", "data/cpi_data.csv"]

    summary = download_record_files(
        "22233228", tmp_path, files=["cpi_data.csv"], record=record, session=session
    )
    assert [r.key for r in summary.results] == ["data/cpi_data.csv"]

    summary = download_record_files(
        "22233228", tmp_path, files=["a.*"], record=record, session=session
    )
    assert [r.key for r in summary.results] == ["a.csv"]


def test_download_record_files_one_failure_does_not_abort_others(tmp_path):
    good_one = make_file("a.csv", b"alpha")
    bad = make_file("b.csv", b"bravo")
    good_two = make_file("c.csv", b"charlie")
    session = FakeSession(
        routes={
            good_one.download_url: FakeResponse(content=b"alpha"),
            bad.download_url: FakeResponse(status_code=404, content=b"missing"),
            good_two.download_url: FakeResponse(content=b"charlie"),
        }
    )
    record = make_record([good_one, bad, good_two])
    summary = download_record_files(
        "22233228", tmp_path, record=record, session=session, max_workers=2
    )

    assert [r.status for r in summary.results] == ["downloaded", "failed", "downloaded"]
    assert not summary.ok
    assert len(summary.downloaded) == 2
    assert len(summary.failed) == 1
    assert (tmp_path / "a.csv").exists()
    assert (tmp_path / "c.csv").exists()


def test_download_record_files_unmatched_pattern_raises(tmp_path):
    record, session = _record_with_content()
    with pytest.raises(ZenodoError) as excinfo:
        download_record_files(
            "22233228", tmp_path, files=["*.parquet"], record=record, session=session
        )
    assert "*.parquet" in str(excinfo.value)
    assert not (tmp_path / "a.csv").exists()


def test_download_record_files_defaults_to_single_file_url(tmp_path):
    record, session = _record_with_content()
    summary = download_record_files(
        "https://zenodo.org/records/22233228/files/a.csv",
        tmp_path,
        record=record,
        session=session,
    )
    assert [r.key for r in summary.results] == ["a.csv"]


def test_download_record_files_skips_existing(tmp_path):
    record, session = _record_with_content()
    (tmp_path / "a.csv").write_bytes(b"alpha")
    summary = download_record_files(
        "22233228", tmp_path, record=record, session=session
    )
    statuses = {r.key: r.status for r in summary.results}
    assert statuses["a.csv"] == "skipped"
    assert statuses["b.csv"] == "downloaded"


def test_download_record_files_cancel_event_stops_streaming(tmp_path, monkeypatch):
    """Concurrent streams stop at the next chunk once the event is set.

    The shared event is set from a deterministic point instead of by whichever
    worker happens to reach its first chunk first. ``b.csv`` reports that its
    ``.part`` file is open and then holds its first chunk until released, so
    the cancel always lands while it waits inside the chunk loop; ``a.csv``
    sets the event only after seeing that signal. Both cancellation checks are
    therefore exercised on every run: ``a.csv`` after writing its first chunk
    and ``b.csv`` in the chunk loop, before it writes anything.
    """
    events = {}
    b_streaming = threading.Event()
    b_release = threading.Event()
    real_download = zenodo.download_file

    def capture(file, destination, **kwargs):
        events[file.key] = kwargs.get("cancel_event")
        return real_download(file, destination, **kwargs)

    monkeypatch.setattr(zenodo, "download_file", capture)

    class CancellingResponse(FakeResponse):
        """Write one chunk, then cancel once the other worker is streaming."""

        def iter_content(self, chunk_size=1):
            yield b"first"
            assert b_streaming.wait(5)
            events["a.csv"].set()
            b_release.set()

    class HoldingResponse(FakeResponse):
        """The .part file is open; hold the first chunk until the cancel."""

        def iter_content(self, chunk_size=1):
            b_streaming.set()
            assert b_release.wait(5)
            yield b"first"

    data = b"firstsecond"
    files = [make_file(key, data) for key in ("a.csv", "b.csv")]
    responses = {
        "a.csv": CancellingResponse(content=data),
        "b.csv": HoldingResponse(content=data),
    }
    session = FakeSession(
        routes={file.download_url: responses[file.key] for file in files}
    )
    record = make_record(files)

    started = time.monotonic()
    summary = download_record_files(
        "22233228", tmp_path, record=record, session=session, max_workers=2
    )
    elapsed = time.monotonic() - started

    assert [r.status for r in summary.results] == ["failed", "failed"]
    assert [r.error for r in summary.results] == ["cancelled", "cancelled"]
    # Every worker gets the same event, and both keep a .part file so a later
    # run can resume: a.csv holds its first chunk, b.csv was stopped before it
    # wrote one.
    assert events["a.csv"] is not None
    assert events["a.csv"] is events["b.csv"]
    parts = {key: (tmp_path / f"{key}.part").read_bytes() for key in ("a.csv", "b.csv")}
    assert parts == {"a.csv": b"first", "b.csv": b""}
    assert sum(len(part) for part in parts.values()) == sum(
        r.bytes_written for r in summary.results
    )
    for key in ("a.csv", "b.csv"):
        assert not (tmp_path / key).exists()
    assert elapsed < 2


def test_download_record_files_signals_workers_on_interrupt(monkeypatch, tmp_path):
    """A Ctrl-C must set the cancel event and wait for running workers."""
    second_started = threading.Event()
    finished = []
    events = []

    def fake_download(file, destination, **kwargs):
        event = kwargs.get("cancel_event")
        events.append(event)
        if file.key == "a.csv":
            assert second_started.wait(5)
            raise KeyboardInterrupt
        second_started.set()
        assert event is not None
        assert event.wait(5)
        finished.append(file.key)
        return FileResult(
            key=file.key,
            path=Path(destination) / file.key,
            status="failed",
            bytes_written=0,
            error="cancelled",
        )

    monkeypatch.setattr(zenodo, "download_file", fake_download)
    files = [make_file("a.csv", b"a"), make_file("b.csv", b"b")]
    record = make_record(files)

    started = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        download_record_files(
            "22233228", tmp_path, record=record, session=FakeSession(), max_workers=2
        )
    elapsed = time.monotonic() - started

    assert finished == ["b.csv"]
    assert all(event is not None and event.is_set() for event in events)
    assert elapsed < 2


def test_select_record_files_deduplicates_preserving_order():
    record = make_record(
        [make_file("a.csv", b"a"), make_file("b.csv", b"b"), make_file("c.csv", b"c")]
    )
    selected = select_record_files(record, ["*.csv", "a.csv"])
    assert [f.key for f in selected] == ["a.csv", "b.csv", "c.csv"]


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def _stub_record():
    return make_record([make_file("cpi_data.csv", b"cpi", checksum="md5:abc")])


def test_parse_args_flags():
    args = zenodo._parse_args(
        [
            "22233228",
            "-d",
            "out",
            "--file",
            "*.csv",
            "--file",
            "*.parquet",
            "--overwrite",
            "--no-verify-checksums",
            "--verify-existing",
            "--jobs",
            "8",
            "--no-resume",
            "--timeout",
            "5",
            "--retries",
            "1",
            "--log-level",
            "DEBUG",
        ]
    )
    assert args.identifiers == ["22233228"]
    assert args.destination == "out"
    assert args.file_patterns == ["*.csv", "*.parquet"]
    assert args.overwrite is True
    assert args.verify_checksums is False
    assert args.verify_existing is True
    assert args.jobs == 8
    assert args.resume is False
    assert args.timeout == 5
    assert args.retries == 1
    assert args.log_level == "DEBUG"


@pytest.mark.parametrize(
    "option",
    [
        ["--timeout", "0"],
        ["--timeout", "-1"],
        ["--timeout", "nan"],
        ["--timeout", "inf"],
        ["--jobs", "0"],
        ["--jobs", "-2"],
        ["--retries", "-1"],
    ],
)
def test_cli_rejects_out_of_range_numbers(option, capsys):
    with pytest.raises(SystemExit) as excinfo:
        zenodo.main(["22233228", "-d", "out", *option])
    assert excinfo.value.code == 2
    assert option[0] in capsys.readouterr().err


def test_cli_accepts_zero_retries():
    args = zenodo._parse_args(["22233228", "-d", "out", "--retries", "0"])
    assert args.retries == 0


def test_cli_list_files_prints_and_writes_nothing(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(zenodo, "get_record", lambda *a, **k: _stub_record())
    rc = zenodo.main(["--list-files", "22233228", "-d", str(tmp_path / "out")])

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == "cpi_data.csv\t3\tmd5:abc"
    assert not (tmp_path / "out").exists()


def test_cli_list_files_does_not_require_destination(monkeypatch, capsys):
    monkeypatch.setattr(zenodo, "get_record", lambda *a, **k: _stub_record())
    assert zenodo.main(["--list-files", "22233228"]) == 0
    assert capsys.readouterr().out.startswith("cpi_data.csv")


def test_cli_dry_run_writes_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(zenodo, "get_record", lambda *a, **k: _stub_record())
    destination = tmp_path / "out"
    rc = zenodo.main(["--dry-run", "22233228", "-d", str(destination)])

    assert rc == 0
    assert not destination.exists()


def test_cli_destination_from_settings_file(monkeypatch, tmp_path):
    destination = tmp_path / "data"
    settings_file = tmp_path / "settings.yml"
    settings_file.write_text(f"data_location: {destination}\n")

    seen = {}

    def fake_download(identifier, dest, **kwargs):
        seen["destination"] = Path(dest)
        return DownloadSummary(
            record=_stub_record(),
            destination=Path(dest),
            results=(
                FileResult(
                    key="cpi_data.csv",
                    path=Path(dest) / "cpi_data.csv",
                    status="downloaded",
                    bytes_written=3,
                ),
            ),
        )

    monkeypatch.setattr(zenodo, "get_record", lambda *a, **k: _stub_record())
    monkeypatch.setattr(zenodo, "download_record_files", fake_download)

    rc = zenodo.main(["22233228", "-sf", str(settings_file)])
    assert rc == 0
    assert seen["destination"] == destination


def test_cli_relative_data_location_resolves_from_settings_folder(
    monkeypatch, tmp_path
):
    settings_file = tmp_path / "data.yml"
    settings_file.write_text("data_location: my_data\n")
    seen = {}

    def fake_download(identifier, dest, **kwargs):
        seen["destination"] = Path(dest)
        return DownloadSummary(
            record=_stub_record(), destination=Path(dest), results=tuple()
        )

    monkeypatch.setattr(zenodo, "get_record", lambda *a, **k: _stub_record())
    monkeypatch.setattr(zenodo, "download_record_files", fake_download)

    assert zenodo.main(["22233228", "-sf", str(settings_file)]) == 0
    assert seen["destination"] == tmp_path / "my_data"


def test_cli_relative_settings_dir_resolves_destination_once(monkeypatch, tmp_path):
    """A relative settings folder must not be prefixed with its parent twice."""
    monkeypatch.chdir(tmp_path)
    settings_dir = tmp_path / "config" / "settings"
    settings_dir.mkdir(parents=True)
    (settings_dir / "data.yml").write_text(
        "model_regions:\n  - A\ndata_location: test_data\n"
    )
    seen = {}

    def fake_download(identifier, dest, **kwargs):
        seen["destination"] = Path(dest)
        return DownloadSummary(
            record=_stub_record(),
            destination=Path(dest),
            results=(
                FileResult(
                    key="cpi_data.csv",
                    path=Path(dest) / "cpi_data.csv",
                    status="downloaded",
                    bytes_written=3,
                ),
            ),
        )

    monkeypatch.setattr(zenodo, "get_record", lambda *a, **k: _stub_record())
    monkeypatch.setattr(zenodo, "download_record_files", fake_download)

    assert zenodo.main(["22233228", "-sf", "config/settings"]) == 0

    from powergenome.settings import load_settings

    expected = Path(load_settings("config/settings")["data_location"]).resolve()
    assert seen["destination"] == expected
    assert seen["destination"] == tmp_path / "config" / "test_data"
    assert "config/config" not in str(seen["destination"])


def test_cli_destination_is_file_exits_1_before_fetch(monkeypatch, tmp_path, caplog):
    destination = tmp_path / "not_a_directory"
    destination.write_text("regular file")
    calls = []

    def fake_get_record(*args, **kwargs):
        calls.append((args, kwargs))
        return _stub_record()

    monkeypatch.setattr(zenodo, "get_record", fake_get_record)
    with caplog.at_level(logging.INFO, logger="powergenome.zenodo"):
        rc = zenodo.main(["22233228", "-d", str(destination)])

    assert rc == 1
    assert calls == []
    assert "not a directory" in caplog.text


def test_download_record_files_mkdir_failure_raises_zenodo_error(monkeypatch, tmp_path):
    record, session = _record_with_content()

    def deny(self, parents=False, exist_ok=False):
        raise PermissionError("permission denied")

    monkeypatch.setattr(Path, "mkdir", deny)
    with pytest.raises(ZenodoError, match="Could not create destination"):
        download_record_files(
            "22233228", tmp_path / "data", record=record, session=session
        )


def test_cli_mkdir_failure_exits_1_without_traceback(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(zenodo, "get_record", lambda *a, **k: _stub_record())

    def deny(self, parents=False, exist_ok=False):
        raise PermissionError("permission denied")

    monkeypatch.setattr(Path, "mkdir", deny)
    with caplog.at_level(logging.INFO, logger="powergenome.zenodo"):
        rc = zenodo.main(["22233228", "-d", str(tmp_path / "data")])

    assert rc == 1
    assert "Could not create destination" in caplog.text
    assert "Traceback" not in caplog.text


def test_cli_invalid_identifier_summary_omits_destination(
    monkeypatch, tmp_path, caplog
):
    destination = tmp_path / "never_created"
    with caplog.at_level(logging.INFO, logger="powergenome.zenodo"):
        rc = zenodo.main(["not-a-record", "-d", str(destination)])

    assert rc == 1
    assert "written to" not in caplog.text
    assert str(destination) not in caplog.text
    assert not destination.exists()


def test_cli_missing_destination_exits_with_argparse_error(capsys):
    with pytest.raises(SystemExit) as excinfo:
        zenodo.main(["22233228", "--dry-run"])
    assert excinfo.value.code == 2
    assert "destination" in capsys.readouterr().err.lower()


def test_cli_invalid_identifier_exits_1(capsys):
    rc = zenodo.main(["--dry-run", "not-a-record", "-d", "out"])
    assert rc == 1


def test_cli_failed_file_exits_1(monkeypatch, tmp_path):
    def fake_download(identifier, dest, **kwargs):
        return DownloadSummary(
            record=_stub_record(),
            destination=Path(dest),
            results=(
                FileResult(
                    key="cpi_data.csv",
                    path=Path(dest) / "cpi_data.csv",
                    status="failed",
                    bytes_written=0,
                    error="checksum mismatch",
                ),
            ),
        )

    monkeypatch.setattr(zenodo, "get_record", lambda *a, **k: _stub_record())
    monkeypatch.setattr(zenodo, "download_record_files", fake_download)
    assert zenodo.main(["22233228", "-d", str(tmp_path)]) == 1


def test_cli_keyboard_interrupt_exits_130(monkeypatch, tmp_path):
    def interrupt(identifier, dest, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(zenodo, "get_record", lambda *a, **k: _stub_record())
    monkeypatch.setattr(zenodo, "download_record_files", interrupt)
    assert zenodo.main(["22233228", "-d", str(tmp_path)]) == 130


def test_cli_success_exits_0(monkeypatch, tmp_path, capsys):
    def fake_download(identifier, dest, **kwargs):
        return DownloadSummary(
            record=_stub_record(),
            destination=Path(dest),
            results=(
                FileResult(
                    key="cpi_data.csv",
                    path=Path(dest) / "cpi_data.csv",
                    status="downloaded",
                    bytes_written=3,
                ),
                FileResult(
                    key="other.csv",
                    path=Path(dest) / "other.csv",
                    status="skipped",
                    bytes_written=0,
                ),
            ),
        )

    monkeypatch.setattr(zenodo, "get_record", lambda *a, **k: _stub_record())
    monkeypatch.setattr(zenodo, "download_record_files", fake_download)
    assert zenodo.main(["22233228", "-d", str(tmp_path)]) == 0


def test_cli_sandbox_url_selects_sandbox_base(monkeypatch, tmp_path):
    seen = {}

    def fake_get_record(identifier, **kwargs):
        seen.update(kwargs)
        return _stub_record()

    monkeypatch.setattr(zenodo, "get_record", fake_get_record)

    def fake_download(identifier, dest, **kwargs):
        return DownloadSummary(
            record=_stub_record(), destination=Path(dest), results=tuple()
        )

    monkeypatch.setattr(zenodo, "download_record_files", fake_download)
    rc = zenodo.main(
        ["https://sandbox.zenodo.org/records/22233228", "-d", str(tmp_path)]
    )
    assert rc == 0
    assert seen["web_base"] == zenodo.ZENODO_SANDBOX_BASE


def test_configure_logging_is_idempotent():
    package_logger = __import__("logging").getLogger("powergenome")
    before = [
        handler
        for handler in package_logger.handlers
        if getattr(handler, "_powergenome_cli_handler", False)
    ]
    zenodo._configure_logging("INFO")
    zenodo._configure_logging("DEBUG")
    tagged = [
        handler
        for handler in package_logger.handlers
        if getattr(handler, "_powergenome_cli_handler", False)
    ]
    assert len(tagged) == 1
    for handler in tagged:
        if handler not in before:
            package_logger.removeHandler(handler)


def test_file_result_and_summary_conveniences():
    record = _stub_record()
    results = (
        FileResult(key="a", path=Path("a"), status="downloaded", bytes_written=10),
        FileResult(key="b", path=Path("b"), status="skipped", bytes_written=0),
        FileResult(
            key="c", path=Path("c"), status="failed", bytes_written=0, error="boom"
        ),
    )
    summary = DownloadSummary(record=record, destination=Path("."), results=results)
    assert summary.ok is False
    assert summary.total_bytes == 10
    assert [r.key for r in summary.downloaded] == ["a"]
    assert [r.key for r in summary.skipped] == ["b"]
    assert [r.key for r in summary.failed] == ["c"]


def test_checksum_error_is_a_zenodo_error():
    assert issubclass(ChecksumError, ZenodoError)
    assert issubclass(ZenodoNotFoundError, ZenodoError)
