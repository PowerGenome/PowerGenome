# Debugging

## Where to look first

1. Run `validate_powergenome --settings_file settings` from your study directory.
2. Read the first error and its preceding warnings in `<results_folder>/log.txt`.
3. Check the resolved `powergenome_case_settings.yml` in the affected period's
   output folder, not just the original YAML. Scenario swaps may have changed it.

The log contains `DEBUG` messages; the terminal shows `INFO` and above. A run
that completes with warnings may still have unsuitable inputs. See
[Validate Settings](validate-settings.md) for coverage and limitations.

## Common errors

| Symptom | Likely cause and next check |
| --- | --- |
| Missing settings key, such as `model_year` | Check the exact parameter name, type, and planning-period configuration in [Model Definition](../reference/settings/model-definition.md). |
| Table creation fails or table is missing | Check the source exists, the setting key is correct, and the table resolves in exactly one configured location. For files, include `.csv` or `.parquet`; for databases, use a table name. |
| `FileNotFoundError` | Check relative-path rules below and whether the command was run from the intended study directory. |
| Planning-year list lengths differ | Use valid `model_periods`, or one matching entry in each legacy year list per period. |
| Requested case IDs are not in scenario inputs | Compare `--case-id` values with the CSV's `case_id` column. Omit the flag to run all cases. |
| Scenario years do not match settings | CSV `year` values must match the planning-period end years. |
| Unknown or missing region | Compare case-sensitive names in tables, `model_regions`, and `region_aggregations`. |
| Missing fuel prices or zero fuel costs | Check fuel/scenario/base-region/year coverage and the selected data vintage. Do not assume a zero price is intentional. |
| Zero new-build costs | Check resource identifiers and whether cost-table `basis_year` values cover each planning window. |

Use the exact keys in [Data Tables Settings](../reference/settings/data-tables.md).
For example, `generation_table` is recognized; a similar-looking key such as
`generators` does not configure that table.

### Check paths and settings loading

- A settings **directory** loads its immediate `*.yml` files, not nested folders
  or `*.yaml` files. Avoid duplicate top-level keys across files: loading performs
  shallow replacement, not a recursive merge.
- Relative `data_location` and `input_folder` paths resolve from the settings
  directory's parent, or from a single settings file's containing directory.
- Relative `RESOURCE_GROUPS` and `RESOURCE_GROUP_PROFILES` paths resolve from the
  process working directory instead.
- Configured tables are searched in both `data_location` and `input_folder`;
  duplicate matches are an error.

The published examples split the two rules on purpose. Their Texas settings use
`data_location: data`, which resolves to `example_systems/TX-3-zone/data`, but
`RESOURCE_GROUPS: example_systems/TX-3-zone/resource_groups`, which only resolves
when you run from the examples repository root. Running the same command from
inside `example_systems/TX-3-zone` silently looks for resource groups that are not
there.

See [Configure Settings](configure-settings.md) for organization and
[Run Multi-Scenario Studies](run-scenarios.md) for override behavior.

## Unexpected outputs

### Generator clustering produces unexpected groups

Check the logged number of units per region and technology, then inspect:

- `num_clusters` and `alt_num_clusters`: existing technologies omitted from both
  are not included merely because they appear in the generation table.
- `tech_groups`, `tech_fuel_map`, and model tags: matching rules differ between
  settings. Do not assume every technology field uses the same substring rules.
- Operating/retirement years and source filters. Blank `retirement_year` means
  no planned retirement; filtering it with `> 2030` drops blank values.

For myopic multi-period runs, use the same generation source in every period so
cluster membership stays consistent. See [Existing Generators](../reference/settings/existing-generators.md).

### Demand or generation profiles are missing

Check that source rows cover every required base region/site, planning year,
weather year, and time step. Confirm filters have not removed needed rows and
that profile identifiers match resource metadata. For aggregated demand regions,
supply the constituent base regions, not just an aggregate label.

Review [Demand Settings](../reference/settings/demand.md) and
[Configure Renewable Clusters](configure-renewable-clusters.md).

### Resource files are empty or absent

Default GenX outputs are split into `resources/Thermal.csv`, `Vre.csv`, and other
resource types. Files for empty types are omitted. Inspect the
[output layout](../reference/schemas/output-files.md) before assuming a missing
legacy `Generators_data.csv` indicates failure.

If expected resources are missing, check source coverage, filters, existing-unit
cluster settings, model tags, and `new_gen_not_available` — its exclusions are
applied per model region while new-build resources are built, so an over-broad
pattern (or an `all`/`all_regions` entry) can remove a resource from the output.
A setting passing validation does not prove it affects output: the demand-growth
settings `growth_scenario`, `load_eia_aeo_year`, `historical_load_region_map`,
`future_load_region_map`, and `alt_growth_rate` are still accepted but no longer
change demand (see
[Multi-Scenario Management](../reference/settings/scenario-management.md#fuel-prices-and-demand)).

## Inspect settings and tables in Python

For deeper diagnostics, load the settings from the study directory:

```python
from pathlib import Path

from powergenome.settings import load_settings
from powergenome.database import initialize_data_manager, list_tables

settings = load_settings(Path("settings"))
print(settings.get("model_year"))
print(settings.get("model_regions"))
initialize_data_manager(settings, settings["data_location"])
print(list_tables())
```

This inspects base settings. Compare them with the saved case/period settings
when investigating scenario-specific behavior.

## Getting help

Share the command, PowerGenome version, first relevant error, and a minimal
reproducible configuration on [GitHub](https://github.com/PowerGenome/PowerGenome/issues)
or [groups.io](https://groups.io/g/powergenome). Redact credentials, private paths,
and sensitive data from settings and logs before sharing.

The [System Design tool](https://gschivley.github.io/PowerGenome-tools/web/)
provides starting settings, not a guarantee that your local data covers them.
