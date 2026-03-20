# Validate Settings Before Running

PowerGenome includes a built-in validation system that checks your settings and data for common configuration mistakes before they can cause errors or produce silently incorrect results. This guide explains what the checks cover, how to run them, and how to interpret the output.

!!! warning "Validation is not exhaustive"
    The validation checks catch many common mistakes, but they do not cover every possible
    configuration error. Always review your output files and logs, especially on the first run
    with a new configuration.

---

## How validation works

Validation runs automatically at the start of every `run_powergenome` call, in two phases:

**Phase 1 — settings-only checks**
: Runs before any data files are read. Checks internal consistency of the settings
  dictionary: required keys are present, paths exist, planning year lists have consistent
  lengths, region names are consistent across settings parameters, fuel definitions are
  complete, and new-build resources have the model tags they need.

**Phase 2 — settings vs. data checks**
: Runs after DataManager has loaded your data tables. Checks that the region names in
  `region_aggregations` and `model_regions` actually appear in your data, that the
  transmission cost table doesn't reference unknown regions, that fuel prices exist for
  every fuel/region/year combination you need, and that new-resource cost tables cover
  your planning periods.

If Phase 1 finds **errors**, the pipeline stops before loading any data. Phase 2 can
produce **warnings** or **errors**: warnings allow the pipeline to continue (though
results may be wrong), while errors will stop execution when validation is run as part
of `run_powergenome`.

---

## Running validation on its own

To validate your settings without running the full pipeline, use the `validate_powergenome`
command:

```bash
validate_powergenome --settings_file path/to/settings
```

This is useful when:

- You're setting up a new study and want to catch mistakes early
- You've modified settings and want a quick sanity check before a long run
- You're debugging unexpected results

### Flags

| Flag | Description |
|------|-------------|
| `--settings_file SETTINGS_PATH` | Path to your settings folder (required) |
| `--skip-data-checks` | Only run Phase 1 (no DataManager needed) |
| `--no-fail` | Log errors but exit with code 0 (useful in CI pipelines) |

```bash
# Check settings structure only, without loading data files
validate_powergenome --settings_file settings --skip-data-checks

# Run full validation, don't exit with an error code
validate_powergenome --settings_file settings --no-fail
```

---

## Understanding the output

Each issue is reported with a severity level and a category:

```
ERROR    powergenome.validate: [ERROR] required_keys: Required settings key 'model_regions' is missing or empty
WARNING  powergenome.validate: [WARNING] fuel_price_coverage: 2 fuel/scenario/region/year combination(s) are missing from the fuel_price table — missing prices will become $0/MMBtu via fillna(0)
    Detail: fuel=naturalgas, scenario=reference, region=p5, year=2030
    fuel=naturalgas, scenario=reference, region=p5, year=2040
```

### Severity levels

**ERROR**
: A definite problem that will cause an exception or clearly wrong results. The pipeline
  stops (or reports a fatal error). Fix errors before proceeding.

**WARNING**
: A likely mistake that produces silently incorrect results — for example, a missing fuel
  price that defaults to $0/MMBtu, or a transmission line that gets dropped without any
  logged message. Warnings do not stop the pipeline, but should be investigated.

### Categories

| Category | Phase | What it checks |
|----------|-------|----------------|
| `required_keys` | 1 | Required settings parameters (`model_regions`, `target_usd_year`, etc.) |
| `planning_years` | 1 | `model_year` and `model_first_planning_year` have matching lengths; `model_periods` entries are valid 2-element lists |
| `paths` | 1 | `data_location`, `RESOURCE_GROUPS`, `RESOURCE_GROUP_PROFILES`, `input_folder` paths exist |
| `region_consistency` | 1 | Region-keyed settings (`regional_tag_values`, `alt_num_clusters`, etc.) only reference known `model_regions` |
| `model_tag_coverage` | 1 | Each entry in `new_resources` matches at least one tag in `model_tag_values` |
| `fuel_consistency` | 1 | Fuels in `tech_fuel_map` are defined in `fuel_scenarios`; CCS fuels have capture rates and emission factors |
| `data_tables` | 2 | Every settings-configured table is present in DataManager |
| `aggregation_base_regions` | 2 | Every base region in `region_aggregations` and every pass-through in `model_regions` appears in at least one data table |
| `transmission_regions` | 2 | `transmission_cost_table` only references known model or base regions |
| `fuel_price_coverage` | 2 | Fuel price table has rows for every fuel/scenario/region/year combination needed |
| `new_resource_cost_years` | 2 | `resource_cost` table has rows whose `basis_year` overlaps each planning period |

---

## Common issues and fixes

### Region name typo in aggregation

```
WARNING  powergenome.validate: [WARNING] aggregation_base_regions: 1 base region(s) referenced in 'model_regions' or 'region_aggregations' do not appear in any data table (plant_region, demand, fuel_price, transmission_cost) — check for typos
    Detail: AZ (aggregation): ['q2']
```

**Cause**: `region_aggregations` maps `AZ` to `[p1, q2]`, but `q2` doesn't exist in any data table (it was probably meant to be `p2`).

**Fix**: Correct the typo in `region_aggregations`:

```yaml
region_aggregations:
  AZ:
    - p1
    - p2  # was q2
```

### Missing fuel prices

```
WARNING  powergenome.validate: [WARNING] fuel_price_coverage: 3 fuel/scenario/region/year combination(s) are missing from the fuel_price table — missing prices will become $0/MMBtu via fillna(0)
    Detail: fuel=coal, scenario=reference, region=p5, year=2030
    fuel=naturalgas, scenario=reference, region=p5, year=2030
    fuel=naturalgas, scenario=reference, region=p5, year=2040
```

**Cause**: The fuel price table doesn't have rows for some fuel/region/year combinations needed by the model. The `fuels.fuel_cost_table()` function silently fills missing values with `$0`.

**Fix**: Check the `Detail` output to see which combinations are missing, then either:

- Add the missing rows to your fuel price table, or
- Check that `fuel_scenarios` and `region_aggregations` match the data in your table

### New resource with no planning-period cost data

```
WARNING  powergenome.validate: [WARNING] new_resource_cost_years: 1 new resource/period combination(s) have no matching basis_year in the resource_cost table — costs will be $0 via fillna(0)
    Detail: OffshoreWind/Class1/Moderate: no basis_year in 2035–2040 (available: [2020, 2025, 2030]…)
```

**Cause**: The `resource_cost` table has ATB cost data for some years, but none fall within this planning period's range. `new_build.single_generator_row()` averages over the period window; with no matching rows, the average is `NaN`, which becomes `$0`.

**Fix**: Extend your cost table to include data for the relevant years, or adjust the `model_periods` planning windows so they overlap with available ATB data years.

### Required key missing

```
ERROR    powergenome.validate: [ERROR] required_keys: Required settings key 'model_regions' is missing or empty
```

**Fix**: Add the missing key to one of your settings YAML files.
See [Model Definition](../reference/settings/model-definition.md) for a list of required parameters.
