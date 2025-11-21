# Command-Line Interface Reference

PowerGenome provides the `run_powergenome_multiple` command for executing the data pipeline.

## Basic Usage

```bash
run_powergenome_multiple --settings_file SETTINGS_PATH --results_folder OUTPUT_PATH
```

## Command-Line Flags

### Required Flags

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

### Optional Flags

#### `--case_id`

Run only specific scenarios from the scenario definitions file.

**Type**: String or list of strings
**Default**: Run all scenarios
**Example**: `--case_id p1 p2` or `--case_id "high_demand"`

```bash
# Run only scenarios p1 and p3
run_powergenome_multiple -sf settings -rf output --case_id p1 p3
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

#### `--verbose` / `-v`

Increase logging verbosity.

**Type**: Boolean flag
**Default**: INFO level logging
**Example**: `--verbose`

#### `--help` / `-h`

Display help message with all available flags.

```bash
run_powergenome_multiple --help
```

## Complete Examples

### Basic Run

```bash
run_powergenome_multiple \
    --settings_file my_study/settings \
    --results_folder my_study/results
```

### Run Specific Scenarios

```bash
run_powergenome_multiple \
    -sf settings \
    -rf output \
    --case_id baseline high_renewable low_cost
```

### Greenfield with Sorted Output

```bash
run_powergenome_multiple \
    -sf settings \
    -rf output \
    --no-current-gens \
    --sort-gens
```

### Debug New Resources Only

```bash
run_powergenome_multiple \
    -sf settings \
    -rf output \
    --no-current-gens \
    --no-load \
    --verbose
```

## Output Organization

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

- **Inputs/Inputs_p#/**: GenX input CSV files
- **log.txt**: Execution log with warnings and info messages

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error (check log for details) |
| 2 | Invalid arguments |

## Environment Variables

PowerGenome respects these environment variables (legacy, prefer settings YAML):

- `DATA_LOCATION`: Path to data folder
- `RESOURCE_GROUPS`: Path to renewable resource group definitions
- `RESOURCE_GROUP_PROFILES`: Path to generation profiles
- `EFS_DATA`: Path to NREL EFS electrification data

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

## Programmatic Usage

PowerGenome can also be used programmatically:

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
