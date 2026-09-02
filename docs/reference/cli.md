# Command Line Interface

PowerGenome provides two commands:

| Command | Purpose |
|---------|---------|
| [`run_powergenome`](#run_powergenome) | Execute the full data pipeline and write model input files (GenX and/or Macro) |
| [`validate_powergenome`](#validate_powergenome) | Check settings and data for configuration mistakes without running the pipeline |

---

## `run_powergenome`

### Basic Usage

```bash
run_powergenome --settings_file SETTINGS_PATH --results_folder OUTPUT_PATH
```

### Flags

#### `--settings_file` / `-sf`

Path to the folder containing settings YAML files.

**Type**: String (path)
**Example**: `--settings_file tests/test_system/settings`

!!! note
    PowerGenome loads **all** YAML files in the folder and merges them. Files loaded later override earlier values for duplicate keys.

#### `--results_folder` / `-rf`

Directory where output files will be saved. Created if it doesn't exist.

**Type**: String (path)
**Example**: `--results_folder my_study_outputs`

#### `--case_id`

Run only specific scenarios from the scenario definitions file.

**Type**: String or list of strings
**Default**: Run all scenarios
**Example**: `--case_id p1 p2` or `--case_id "high_demand"`

```bash
# Run only scenarios p1 and p3
run_powergenome -sf settings -rf output --case_id p1 p3
```

#### `--sort-gens`

Sort generator resources alphabetically in output files.

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
(size + first/last 1 MB) of the data files backing the configured tables, so a re-run
with unchanged inputs skips the check computation. Equivalent to
`use_validation_cache: false` in settings.

**Type**: Boolean flag
**Default**: False (cache enabled)
**Example**: `--force-validation`

```bash
run_powergenome -sf settings --no-validation-cache
```

#### `--verbose` / `-v`

Increase logging verbosity.

**Type**: Boolean flag
**Default**: INFO level logging
**Example**: `--verbose`

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
    --case_id baseline high_renewable low_cost
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
    --no-load \
    --verbose
```

### Output organization

Results are organized by scenario and planning period:

```
results_folder/
├── case_id_1/
│   ├── Inputs/
│   │   └── Inputs_p1/
│   │       ├── Generators_data.csv
│   │       ├── Load_data.csv
│   │       ├── Fuels_data.csv
│   │       └── ...
│   └── log.txt
└── case_id_2/
    ├── Inputs/
    │   └── Inputs_p1/
    └── log.txt
```

Each scenario folder contains:

- **Inputs/Inputs_p#/**: GenX input CSV files (written unless `--no-genx`)
- **Macro `simpleCSVinputs` case**: written alongside GenX when `--macro`/`macro_output: true` is enabled
  (see [Output File Format](../explanation/output-format.md) for the layout)
- **log.txt**: Execution log with warnings and info messages

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
replayed from a cache keyed on the settings and a fingerprint (size + first/last 1 MB) of
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

## Environment variables

PowerGenome respects these environment variables (legacy, prefer settings YAML):

- `DATA_LOCATION`: Path to data folder
- `RESOURCE_GROUPS`: Path to renewable resource group definitions
- `RESOURCE_GROUP_PROFILES`: Path to generation profiles
- `EFS_DATA`: Path to NREL EFS electrification data

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

See the source code in `powergenome/run_powergenome.py` for details.

## Troubleshooting

### Settings folder not found

```
Error: Settings path 'settings' does not exist
```

**Solution**: Provide absolute path or path relative to current directory

### No scenarios found

```
Warning: No scenario definitions found
```

**Solution**: Check `scenario_definitions_fn` points to valid CSV file in `extra_inputs/`

### Permission denied writing output

```
PermissionError: [Errno 13] Permission denied: 'results/case1'
```

**Solution**: Ensure write permissions on output directory or choose different location

See [Debugging](../how-to/debugging.md) for more troubleshooting tips.
