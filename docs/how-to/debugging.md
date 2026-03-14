# Debugging

This guide covers the most common errors and problems you may encounter when running PowerGenome and how to fix them.

## Where to look first

PowerGenome writes a detailed log file to your results folder:

```
results/
└── log.txt   ← check here first
```

The log captures `DEBUG`-level messages, including which tables were loaded, what clusters were built, and any warnings about missing data. The terminal only shows `INFO` and above.

---

## Common errors

### `KeyError: 'model_year'` (or other required parameter)

**Cause**: A required settings parameter is missing from your YAML files.

**Fix**: Check that the parameter is present in one of your settings files:

```bash
grep -r "model_year" settings/
```

Common required parameters:

```yaml
model_periods:
  - [2020, 2030]
target_usd_year: 2022
model_regions: [CA_N, CA_S]
model_tag_names: [THERM, VRE, MUST_RUN, STOR, FLEX, HYDRO, LDS, ELECTROLYZER]
```

See [Model Definition](../reference/settings/model-definition.md) for the full list.

---

### `RuntimeError: Failed to create table <name>`

**Cause**: DataManager could not load a table specified in settings.

**Fix**: Check three things:

1. **File exists** — confirm the file is in `data_location`:

    ```bash
    ls /path/to/data_location/
    ```

2. **Correct key** — verify the setting name matches the expected key (e.g., `generation_table`, not `generators`):

    ```yaml
    # Correct
    generation_table: generators.parquet

    # Wrong key — will be silently ignored
    generators: generators.parquet
    ```

3. **File extension** — always include `.csv` or `.parquet`:

    ```yaml
    # Correct
    demand_table: demand.parquet

    # May fail if auto-detection doesn't find it
    demand_table: demand
    ```

---

### `FileNotFoundError: No such file or directory`

**Cause**: A path in settings doesn't exist. Most commonly `data_location`, `input_folder`, or a file pointed to by a `_fn` parameter.

**Fix**: Use an absolute path to rule out working-directory issues:

```yaml
# Use absolute path
data_location: /Users/me/powergenome_data
input_folder: extra_inputs   # Relative paths are resolved from the settings folder
```

Run PowerGenome from the same directory that contains your settings folder.

---

### `AssertionError: number of years in model_year must equal model_first_planning_year`

**Cause**: Using the legacy `model_year` + `model_first_planning_year` format with lists of different lengths.

**Fix**: Switch to `model_periods` (preferred), or ensure both lists are the same length — one entry per planning period:

```yaml
# Preferred
model_periods:
  - [2020, 2030]
  - [2031, 2040]
  - [2041, 2050]

# Legacy equivalent (lists must be the same length)
model_year: [2030, 2040, 2050]
model_first_planning_year: [2020, 2031, 2041]  # same length
```

---

### `ValueError: The requested case IDs … are not in your scenario inputs file`

**Cause**: You passed `--case-id` values to the CLI that don't exist in `scenario_definitions_fn`.

**Fix**: Check the `case_id` column in your scenario definitions CSV:

```bash
head -1 extra_inputs/scenario_definitions.csv
```

Or omit `--case-id` to run all cases.

---

### `Warning: One or more model regions is not valid`

**Cause**: A region name in `model_regions` doesn't match any IPM region or key in `region_aggregations`.

**Fix**: Check spelling carefully — region names are case-sensitive:

```yaml
model_regions:
  - CA_N    # must match keys in region_aggregations
  - CA_S

region_aggregations:
  CA_N: [WEC_CALN]
  CA_S: [WEC_LADW, WEC_SCE, WEC_SDGE, WECC_TID]
```

---

### `Warning: The years in scenario_definitions_fn do not match model_year`

**Cause**: The `year` column of your scenario definitions CSV doesn't match the planning period end years.

**Fix**: Make sure the `year` values in the CSV match the last year of each period in `model_periods`:

```yaml
# settings
model_periods:
  - [2020, 2030]
  - [2031, 2040]
```

```csv
# scenario_definitions.csv
case_id,year,...
baseline,2030,...
baseline,2040,...
```

---

### Generator clustering produces unexpected groups

**Cause**: Multiple possible causes — technology name mismatches, incorrect cluster counts, or plants being filtered out.

**Diagnostics**:

1. Check `log.txt` for messages about the number of plants being clustered per region/technology.
2. Verify that technology names in your data match what's expected in `tech_fuel_map` and `model_tag_values`. Technology name matching is case-insensitive substring matching.
3. Check `num_clusters` and `alt_num_clusters`:

    ```yaml
    num_clusters: 1  # Default clusters per region/tech
    alt_num_clusters:
      CA_N:
        NaturalGas_CCAvgCF_Moderate: 3  # Override for a specific region/tech
    ```

---

### Load profiles are all zeros or missing regions

**Cause**: `demand_table` configuration doesn't include all model regions, or the region column name differs from what PowerGenome expects.

**Fix**:

1. Check the column names in your demand file — PowerGenome auto-detects the region column but expects values to match `model_regions`.
2. If using a filtered demand table, ensure the filter includes all required regions:

    ```yaml
    demand_table:
      table_name: demand.parquet
      filters:
        - - [region, 'in', ['CA_N', 'CA_S']]
          - [year, '=', 2030]
    ```

---

### Time domain reduction produces errors

**Cause**: Usually a mismatch between `time_domain_periods` and the number of hours in your data, or missing required parameters.

**Fix**: Ensure all four required parameters are present when `reduce_time_domain: true`:

```yaml
reduce_time_domain: true
time_domain_periods: 12          # number of representative periods
time_domain_days_per_period: 7   # days per period (7 = weekly)
include_peak_day: true
demand_weight_factor: 5
```

Time domain reduction only works on data with ≤ 8760 hourly rows. Check `log.txt` for the warning message if it's being skipped.

---

### Output files are empty or very small

**Cause**: Missing resource tag assignments, or all generators were filtered out before output.

**Fix**:

1. Check that `model_tag_names` is fully defined and all resources have tag values assigned.
2. Verify `model_tag_values` covers your resource types (use substring matching — `NaturalGas` will match `NaturalGas_CCAvgCF_Moderate`):

    ```yaml
    model_tag_names: [THERM, VRE, MUST_RUN, STOR]
    model_tag_values:
      NaturalGas: {THERM: 1, VRE: 0}
      UtilityPV:  {THERM: 0, VRE: 1}
    ```

---

## Checking settings are correct

Validate settings load without running the full pipeline:

```python
from powergenome.settings import load_settings
from pathlib import Path

settings = load_settings(Path("settings"))
print(settings.get("model_year"))
print(settings.get("model_regions"))
```

Confirm DataManager can see your tables:

```python
from powergenome.database import initialize_data_manager, get_data_manager

initialize_data_manager(settings, settings["data_location"])
dm = get_data_manager()
print(dm.list_tables())
```

---

## Getting help

- Review the [Settings Reference](../reference/settings/index.md) to check parameter names and types
- Use the [System Design tool](https://gschivley.github.io/PowerGenome-tools/web/) to generate a known-good settings baseline to compare against
- Open an issue on [GitHub](https://github.com/PowerGenome/PowerGenome) with your log file attached
