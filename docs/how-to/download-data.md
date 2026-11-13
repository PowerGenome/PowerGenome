# Download Data from Zenodo

PowerGenome's input datasets — generator costs, demand profiles, renewable
resource profiles, and similar source data — are published as Zenodo records.
Each record is a versioned snapshot with a DOI, and its data is split into
many individual files (some multi-GB). The `download_zenodo_data` command
fetches those files for you: it downloads them concurrently, resumes
interrupted transfers, and verifies the checksums Zenodo publishes.

This guide shows how to find the right record, download it into your
`data_location`, fetch only a subset, and use the same functionality from
Python.

!!! note "Why Zenodo?"
    Zenodo gives every dataset a citable DOI and keeps old versions online.
    That makes a PowerGenome study reproducible: a settings file can point at
    a specific record version instead of at a moving web page.

## 1. Find the DOI or record ID

Zenodo links contain the record ID, and the DOI points at either a specific
version or the concept (all versions). Either works:

| Where you copied it from | Example |
|--------------------------|---------|
| Zenodo project page | `https://zenodo.org/records/22233228` |
| DOI from a paper or README | `10.5281/zenodo.22233228` |
| Zenodo API | `https://zenodo.org/api/records/22233228` |
| Record ID only | `22233228` |

The two PowerGenome data records are:

| Dataset | Record ID | Concept DOI |
|---------|-----------|-------------|
| PowerGenome Input Data | `22233228` | `10.5281/zenodo.22233228` |
| Renewable Resource Profiles | `22235855` | `10.5281/zenodo.22235855` |

A **version DOI** (for example `10.5281/zenodo.22233228`) always resolves to
that exact snapshot. A **concept DOI** resolves to the latest version, which is
usually what you want while following these docs.

## 2. List the files first

Before downloading hundreds of files, inspect the record:

```bash
download_zenodo_data 22233228 --list-files
```

Each line is `key`, tab, `size in bytes`, tab, `checksum`:

```text
cpi_data.csv	4123456	md5:2a6c49b1cd1fc175b776ce853f06485e
reeds_load_transformed.parquet	853684596	md5:...
...
```

`--list-files` does not create a directory or download anything, so it is safe
to run anywhere. It is also the quickest way to see the exact key you want to
pass to `--file`.

## 3. Download into `data_location`

The most common workflow is to reuse the same settings you pass to
`run_powergenome`. Point `-sf` at your settings file or settings directory:

```bash
download_zenodo_data 10.5281/zenodo.22233228 -sf settings
```

The destination comes from `data_location` in those settings, so the files land
where PowerGenome expects them. If you would rather choose a directory
explicitly, pass `-d` instead (it wins when both are given):

```bash
download_zenodo_data 22233228 -d data/zenodo
```

Relative subdirectories inside a record are preserved. A key such as
`profiles/wind.csv` is written to `data/zenodo/profiles/wind.csv`.

## 4. Download only part of a record

Use `--file` with one or more glob patterns. Patterns are matched against the
full key and against the basename, so `--file cpi_data.csv` finds a nested
`data/cpi_data.csv`:

```bash
# Only CSV and Parquet files
download_zenodo_data 22233228 -d data --file '*.csv' --file '*.parquet'

# One file by exact name
download_zenodo_data 22233228 -d data --file cpi_data.csv
```

A pattern that matches nothing is an error rather than a silent no-op, so a
typo cannot leave you with an empty data folder:

```text
ERROR    powergenome.zenodo: No files in Zenodo record 22233228 match: '*.cap'.
         Use --list-files to see the available keys.
```

If the identifier itself is a single-file URL, only that file is downloaded:

```bash
download_zenodo_data \
  https://zenodo.org/records/22233228/files/cpi_data.csv \
  -d data
```

## 5. Resume interrupted multi-GB downloads

Downloads stream into `<file>.part` and are renamed to their final name only
when complete. If your connection drops, re-run the same command: it sends a
range request and continues from the existing `.part` file instead of starting
over.

Passing `--no-resume` makes the command delete partial files and start fresh,
which is rarely needed.

## 6. Verify checksums

Zenodo publishes an MD5 checksum for every file. `download_zenodo_data` checks
each file while it streams and fails only that file on a mismatch, deleting the
partial output so the next run downloads it cleanly. Verification is on by
default; `--no-verify-checksums` skips it.

Existing files whose size matches are skipped without re-reading them. To also
verify an existing copy's checksum — useful after copying a data directory
between machines — add `--verify-existing`:

```bash
download_zenodo_data 22233228 -d data --verify-existing
```

Use `--overwrite` to re-download everything regardless of what is present.

## 7. Preview without downloading

`--dry-run` resolves every record and logs each file it would download, its
size, and whether a copy already exists. It does not create directories or
write files:

```bash
download_zenodo_data 22233228 -d data --dry-run
```

This is a good way to confirm the destination and selection before starting a
multi-GB transfer.

## 8. Use the helper from Python

The same capabilities are available as a public API when you are scripting a
study:

```python
from pathlib import Path

from powergenome.zenodo import download_record_files, get_record

# Inspect a record without downloading
record = get_record("10.5281/zenodo.22233228")
print(record.title, record.version)
for file in record.files:
    print(file.key, file.size, file.checksum)

# Download a subset into data_location
summary = download_record_files(
    "22233228",
    Path("data/zenodo"),
    files=["*.csv"],
    max_workers=8,
)
print(f"{len(summary.downloaded)} downloaded, {len(summary.failed)} failed")
print(f"{summary.total_bytes} bytes written")
```

`download_record_files` returns a `DownloadSummary` with per-file `results` in
the record's original order. Each `FileResult` reports `status`
(`downloaded`, `skipped`, or `failed`), `bytes_written`, and an `error` message
when something went wrong. A single failed file never raises and never stops
the rest of the run; `summary.ok` is `False` when anything failed.

To limit concurrency, pass `max_workers=1`. To report progress in a notebook or
GUI, pass a callback:

```python
def show(key, downloaded, total):
    print(f"{key}: {downloaded}/{total}")

download_record_files("22233228", Path("data"), files=["*.csv"], progress=show)
```

## Troubleshooting

### `A destination is required`

Pass `-d PATH`, or `-sf` with settings that define `data_location`:

```bash
download_zenodo_data 22233228 -sf settings/data.yml
```

### `Could not resolve Zenodo DOI`

Check the DOI, or pass the numeric record ID (visible in the Zenodo URL)
instead. Concept DOIs resolve through a redirect to `doi.org`, so a network
that blocks that host will fail even though the record exists; the API
fallback needs a suffix that matches the record ID.

### `zenodo.org` is unreachable

Check your network or proxy configuration. If you are testing an upload, use
`--sandbox` (or a `sandbox.zenodo.org` URL) to work against Zenodo's test
instance; sandbox records have their own record IDs, which differ from the
production IDs above.

### A download keeps failing

Run with `--log-level DEBUG` and retry. Partial output is kept as `.part`
files, so a later run resumes rather than restarts. HTTP 4xx errors are not
retried — they usually mean the record or file key does not exist, which
`--list-files` will confirm.
