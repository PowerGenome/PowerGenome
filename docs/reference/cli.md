# Command Line Interface

PowerGenome provides three commands:

| Command | Purpose |
|---------|---------|
| [`run_powergenome`](#run_powergenome) | Execute the full data pipeline and write model input files (GenX and/or Macro) |
| [`validate_powergenome`](#validate_powergenome) | Check settings and data for configuration mistakes without running the pipeline |
| [`download_zenodo_data`](#download_zenodo_data) | Download PowerGenome data files from Zenodo records |

---

## `run_powergenome`

### Basic Usage

```bash
run_powergenome --settings_file SETTINGS_PATH --results_folder OUTPUT_PATH
```

### Flags

#### `--settings_file` / `-sf`

Path to a settings file or a directory containing settings files.

**Type**: String (path)
**Default**: `example_settings.yml`
**Example**: `--settings_file settings`

!!! note
    Directory mode loads immediate `*.yml` files only, not subdirectories or
    `.yaml` files. Avoid duplicate top-level keys across files: later files
    replace whole values, and file order is not a configuration interface.

#### `--results_folder` / `-rf`

Directory where output files will be saved. Created if it does not exist;
its parent directory must already exist.

**Type**: String (path)
**Default**: A timestamp, for example `2026-01-15 14.30.00`
**Example**: `--results_folder my_study_outputs`

#### `--case-id` / `-c`

Run only the specified case IDs from `scenario_definitions_fn`. Without a
scenario definitions file, this selection is ignored with a warning.

**Type**: One or more case IDs, separated by spaces
**Default**: Run all cases
**Example**: `--case-id baseline high_demand`

```bash
run_powergenome -sf settings -rf output --case-id baseline high_demand
```

#### `--sort-gens`

Sort generator resources alphabetically within each region. Existing and
new-build resources remain in separate groups.

**Type**: Boolean flag
**Default**: False
**Example**: `--sort-gens`

Useful for:

- Comparing outputs across runs
- Version control (reduces diff noise)
- Easier manual inspection

#### `--no-current-gens`

Skip existing generator clustering. Only new-build resources will be included.

**Type**: Boolean flag
**Default**: False (existing generators are clustered)
**Example**: `--no-current-gens`

Use when:

- Building greenfield scenarios
- Testing new-build resource configurations
- Debugging new resource issues

#### `--no-load`

Skip demand profile generation. Load files must be provided manually.

**Type**: Boolean flag
**Default**: False (load profiles are generated)
**Example**: `--no-load`

Use when:

- Testing generator configurations
- Load profiles are pre-generated
- Debugging non-demand issues

#### `--macro` {#macro}

Write a MacroEnergy.jl (`Macro`) case in the `simpleCSVinputs` format, in addition to the default GenX output —
a single run writes both formats. Can also be enabled with the `macro_output: true` setting. Set
`genx_output: false` (or pass [`--no-genx`](#no-genx)) to write Macro inputs only.

**Type**: Boolean flag
**Default**: False
**Example**: `--macro`

!!! note
    Writing both formats in one run reuses all the intermediate data processing and is faster than running
    PowerGenome once per model.

#### `--genx` {#genx}

Write GenX `Inputs` files (the default output when no output flag or output setting is set). Can be combined with
[`--macro`](#macro) to explicitly request both formats in a single run. Can also be controlled with the
`genx_output: true` setting.

**Type**: Boolean flag
**Default**: False (GenX is written by default regardless; this flag is explicit)
**Example**: `--genx`

#### `--no-genx` {#no-genx}

Do not write GenX `Inputs` files. A hard override that wins over both `--genx` and an explicit
`genx_output: true` setting. Combine with `--macro` (or `macro_output: true`) to write Macro inputs only from the
command line, without editing a settings file.

**Type**: Boolean flag
**Default**: False
**Example**: `--no-genx`

```bash
# Write Macro inputs only
run_powergenome -sf settings -rf output --macro --no-genx
```

#### `--no-validation-cache` / `--force-validation` {#no-validation-cache}

Always run the Phase 1/2 validation checks, ignoring any cached results. By default,
validation results are replayed from a cache keyed on the settings and a fingerprint
(size, modification time, and deterministic content samples) of the data files backing the configured tables, so a re-run
with unchanged inputs skips the check computation. Equivalent to
`use_validation_cache: false` in settings.

**Type**: Boolean flag
**Default**: Not set (follows `use_validation_cache`; caching is enabled by default)
**Example**: `--force-validation`

```bash
run_powergenome -sf settings --no-validation-cache
```

#### `--no-gens`

Skip generator processing entirely, including existing and new-build resources.
This does not disable demand or transmission processing.

#### `--no-transmission`

Skip transmission processing and network output.

#### `--multi-period` / `-mp` and `--no-multi-period`

Multi-period output is enabled by default. These two flags are mutually
exclusive and only make the choice explicit:

- `--multi-period` / `-mp` enables it.
- `--no-multi-period` disables it.

Disabling omits the `Min_Retired_*` retirement requirements and the
multi-stage transmission fields. It does not change which planning years are
processed; use `model_year` and `model_first_planning_year` for that.

#### `--help` / `-h`

Display help message with all available flags.

```bash
run_powergenome --help
```

### Examples

#### Basic run

```bash
run_powergenome \
    --settings_file my_study/settings \
    --results_folder my_study/results
```

#### Run specific scenarios

```bash
run_powergenome \
    -sf settings \
    -rf output \
    --case-id baseline high_renewable low_cost
```

#### Greenfield with sorted output

```bash
run_powergenome \
    -sf settings \
    -rf output \
    --no-current-gens \
    --sort-gens
```

#### Debug new resources only

```bash
run_powergenome \
    -sf settings \
    -rf output \
    --no-current-gens \
    --no-load
```

### Output organization

With a scenario definitions file, results are organized by case and planning period:

```text
results_folder/
├── log.txt
├── case_id_1/
│   └── Inputs/
│       ├── Inputs_p1/
│       │   ├── resources/
│       │   ├── system/
│       │   └── policies/
│       └── Inputs_p2/
└── case_id_2/
    └── Inputs/
        └── Inputs_p1/
```

Without `scenario_definitions_fn`, period folders are directly under
`results_folder/Inputs/`, for example `results_folder/Inputs/Inputs_p1/`.

- **Inputs/Inputs_p#/**: GenX input folders `resources/`, `system/`, and
  `policies/` by default; `old_genx_format: true` selects legacy flat CSVs
  (written unless `--no-genx`)
- **Macro case**: written when `--macro`/`macro_output: true` is enabled;
  see [Output File Format](../explanation/output-format.md) for its layout
- **log.txt**: One detailed execution log at the results root, not per case

### Exit codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error (check log for details) |
| 2 | Invalid arguments |

---

## `validate_powergenome`

Validates your settings for common configuration mistakes without running the full pipeline.
See [Validate Settings Before Running](../how-to/validate-settings.md) for a full guide.

### Basic Usage

```bash
validate_powergenome --settings_file SETTINGS_PATH
```

### Flags

#### `--settings_file` / `-sf`

**Required.** Path to the folder containing your settings YAML files (same value you pass to `run_powergenome`).

#### `--skip-data-checks`

Only run Phase 1 (settings-only checks). DataManager is not initialized, so no data files need to be present.

**Default**: Both phases run.

```bash
validate_powergenome -sf settings --skip-data-checks
```

#### `--no-fail`

Exit with code `0` even when errors are found. Errors are still logged at `ERROR` level. Useful in scripts or CI pipelines where you want to review the output without triggering a failure.

**Default**: Exit with code `1` when any `ERROR`-level issue is found.

```bash
validate_powergenome -sf settings --no-fail
```

#### `--no-validation-cache` / `--force-validation`

Always run the checks, ignoring any cached validation results. By default, results are
replayed from a cache keyed on the settings and a fingerprint (size, modification time, and deterministic content samples) of
the data files, so unchanged inputs skip the checks — and Phase 2 skips loading the
DataManager entirely. See
[Skipping validation when nothing changed](../how-to/validate-settings.md#skipping-validation-when-nothing-changed).

**Default**: Cache enabled (unless `use_validation_cache: false` is set in settings).

```bash
validate_powergenome -sf settings --no-validation-cache
```

### Exit codes

| Code | Meaning |
|------|---------|
| 0 | No errors found (warnings may still be present) |
| 1 | One or more `ERROR`-level issues found |

---

## `download_zenodo_data`

Downloads the individual files that make up a Zenodo record. PowerGenome data
records contain many files — some multi-GB — so the command downloads them
concurrently, resumes interrupted transfers, and verifies the MD5 checksum
Zenodo publishes for every file. See
[Download Data from Zenodo](../how-to/download-data.md) for a task-oriented guide.

### Basic Usage

```bash
download_zenodo_data IDENTIFIER [IDENTIFIER ...] --destination PATH
```

`IDENTIFIER` accepts any of these forms:

| Form | Example |
|------|---------|
| Record ID | `22233228` |
| DOI | `10.5281/zenodo.4426096` |
| DOI with prefix | `doi:10.5281/zenodo.4426096` |
| DOI URL | `https://doi.org/10.5281/zenodo.4426096` |
| Web URL | `https://zenodo.org/records/22233228` |
| Legacy web URL | `https://zenodo.org/record/22233228` |
| API URL | `https://zenodo.org/api/records/22233228` |
| Single-file URL | `https://zenodo.org/records/22233228/files/cpi_data.csv` |
| Sandbox URL | `https://sandbox.zenodo.org/records/1234` |

Concept DOIs resolve to the latest version of a record by following the
`doi.org` redirect. A single-file URL downloads only that file unless
`--file` patterns are also given.

### Flags

#### `IDENTIFIER` (positional)

One or more records. Multiple records are processed in the order given; a
failure on one record does not stop the others.

#### `--destination` / `-d`

Directory to write files into. Relative subdirectories in file keys are
preserved (a key `profiles/wind.csv` is written to
`DESTINATION/profiles/wind.csv`). The directory is created if needed.

**Type**: String (path)
**Default**: `data_location` from `--settings-file`

#### `--settings-file` / `-sf`

Settings YAML file, or a directory of YAML files, used to read `data_location`
when `--destination` is not given. This is the same value passed to
`run_powergenome`.

**Type**: String (path)
**Default**: None

```bash
download_zenodo_data 22233228 -sf settings
```

`-d` wins when both options are provided. If neither option supplies a
destination, the command exits with code 2 and explains both options.

#### `--file`

Only download files whose full key or basename matches a glob pattern
(case-sensitive). Repeat the flag for several patterns. A pattern that matches
nothing is an error, so a typo cannot silently download nothing. When several
`IDENTIFIER`s are given, every pattern must match at least one file **in every
record**, so heterogeneous records need separate invocations.

**Type**: Glob pattern (repeatable)
**Default**: All files, or the one file named in a single-file URL

```bash
download_zenodo_data 22233228 -d data --file '*.csv' --file '*.parquet'
```

#### `--overwrite`

Download files again even when a complete copy already exists. Without this
flag a file whose size matches is skipped.

**Type**: Boolean flag
**Default**: False

#### `--no-verify-checksums`

Skip MD5 verification of downloaded files. Verification is on by default;
each file is hashed while it streams and a mismatch fails that file and
removes the partial output.

**Type**: Boolean flag
**Default**: Verify checksums

#### `--verify-existing`

Also verify the checksum of files that are already present before skipping
them. Slower on large existing datasets, but detects corrupted copies.

**Type**: Boolean flag
**Default**: False (size comparison only)

#### `--jobs`

Number of concurrent downloads.

**Type**: Integer
**Default**: 4

```bash
download_zenodo_data 22233228 -d data --jobs 8
```

#### `--dry-run`

Resolve the records and log each file that would be downloaded, including its
size and whether a copy already exists. Nothing is written and no directories
are created.

**Type**: Boolean flag
**Default**: False

#### `--list-files`

Print one line per selected file and exit without creating directories or
downloading. Fields are tab-separated: `key`, `size in bytes`, `checksum`.

**Type**: Boolean flag
**Default**: False

```bash
download_zenodo_data 22233228 --list-files
```

#### `--sandbox`

Use `https://sandbox.zenodo.org` instead of `https://zenodo.org`. A sandbox
URL in `IDENTIFIER` also selects the sandbox.

**Type**: Boolean flag
**Default**: False

#### `--timeout`

Network timeout in seconds, applied to connecting and reading.

**Type**: Float
**Default**: 60

#### `--retries`

Number of retry attempts for transient failures (connection errors, timeouts,
HTTP 429, and HTTP 5xx). Retries use exponential backoff with jitter and honour
a reasonable `Retry-After` header. HTTP 4xx responses are not retried.

**Type**: Integer
**Default**: 3

#### `--no-resume`

Start partial downloads over instead of resuming from the existing `.part`
file. Resuming is on by default and sends an HTTP `Range` request.

**Type**: Boolean flag
**Default**: Resume

#### `--log-level`

Logging verbosity.

**Type**: String (`DEBUG`, `INFO`, `WARNING`, `ERROR`)
**Default**: `INFO`

### Examples

```bash
# Download the PowerGenome input data record into data_location from settings
download_zenodo_data 10.5281/zenodo.22233228 -sf settings

# Download only CSV and Parquet files from the renewable profiles record
download_zenodo_data 22235855 -d data/zenodo --file '*.csv' --file '*.parquet'

# Inspect a record without downloading anything
download_zenodo_data 22233228 --list-files
download_zenodo_data 22233228 -d data --dry-run
```

### Download behavior

- **Atomic writes.** Bytes stream into `<file>.part`; the part file is renamed
  to the final name only after the size and checksum match.
- **Resume.** An interrupted download leaves its `.part` file in place. The
  next run sends `Range: bytes=<size>-` and appends on HTTP 206, or starts over
  if the server returns the whole file.
- **Checksums.** Zenodo publishes `md5:<hex>` for each file. A mismatch fails
  only that file and removes its partial output so a rerun starts clean.
- **Skip existing.** A file whose size matches is skipped; use
  `--verify-existing` to require a checksum match as well.
- **Safe keys.** File keys may contain `/`; relative subdirectories are
  preserved. Absolute paths and `..` components are rejected per file.
- **Partial failure.** One failed file is reported and recorded, and the rest
  of the record still downloads.
- **Ctrl-C.** Stops scheduling, keeps `.part` files for resume, and exits 130.

### Exit codes

| Code | Meaning |
|------|---------|
| 0 | All selected files downloaded or skipped |
| 1 | A record could not be resolved, or at least one file failed |
| 2 | Invalid arguments (including a missing destination) |
| 130 | Interrupted with Ctrl-C |

The run ends with a summary line such as:

```text
INFO     powergenome.zenodo: Summary: 15 downloaded, 0 skipped, 0 failed, 853.4 MB written to /data/zenodo
```

---

## Environment variables

PowerGenome respects these environment variables (legacy, prefer settings YAML):

- `RESOURCE_GROUPS`: Path to renewable resource group definitions
- `RESOURCE_GROUP_PROFILES`: Path to generation profiles
- `EFS_DATA`: Path to NREL EFS electrification data
- `DISTRIBUTED_GEN_DATA`: Legacy distributed-generation data path

`DATA_LOCATION` is not a supported environment-variable substitute for the
`data_location` YAML setting.

Environment variables are always single paths. When using settings YAML,
`RESOURCE_GROUPS` and `RESOURCE_GROUP_PROFILES` may instead be a list of paths
(split across multiple folders).

!!! warning "Deprecated"
    Using environment variables for paths is deprecated. Use settings YAML parameters instead:

    ```yaml
    data_location: /path/to/data
    RESOURCE_GROUPS: /path/to/groups
    RESOURCE_GROUP_PROFILES: /path/to/profiles
    ```

## Logging

PowerGenome uses Python's logging module. Logs show:

- Data loading progress
- Clustering operations
- Missing data warnings
- Regional mismatches
- Cost calculations

Example log output:

```
INFO:powergenome.run_powergenome:Loading settings from settings
INFO:powergenome.database:Initializing DataManager
INFO:powergenome.generators:Clustering 145 generators into 12 clusters
WARNING:powergenome.generators:Region p5 not found in fuel price data
INFO:powergenome.GenX:Writing output files to results/case1/Inputs/Inputs_p1/
```

## Programmatic usage

`run_powergenome` can also be called from Python:

```python
from powergenome.run_powergenome import main

main(
    settings_file="path/to/settings",
    results_folder="path/to/results",
    case_id=["case1", "case2"],
)
```

`main()` still parses the process command line before applying these keyword
arguments. In notebooks or applications with their own arguments, prefer calling
the CLI in a subprocess with an explicit argument list. This is an internal entry
point, not a separate stable Python API.

## Troubleshooting

### Settings folder not found

```
Error: Settings path 'settings' does not exist
```

**Solution**: Provide absolute path or path relative to current directory

### No scenario definitions file

A scenario CSV is optional. Without one, PowerGenome runs the configured planning
years as a single case. If you intended to use cases, set
`scenario_definitions_fn` to a CSV under `input_folder`, with `case_id` and `year`
columns.

### Permission denied writing output

```
PermissionError: [Errno 13] Permission denied: 'results/case1'
```

**Solution**: Ensure write permissions on output directory or choose different location

See [Debugging](../how-to/debugging.md) for more troubleshooting tips.
