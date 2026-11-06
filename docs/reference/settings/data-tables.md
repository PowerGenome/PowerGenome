# Data Tables Settings

DataManager exposes configured CSV, Parquet, SQLite, and DuckDB sources under
standard internal names. For a setup walkthrough, see
[Configure Data Tables](../../how-to/configure-data-tables.md). For column types,
units, and coverage requirements, use [Input Table Schemas](../schemas/input-tables.md).

## Data Location

### `data_location`

**Type**: path or list of paths. **Required** for the data pipeline.

A location can be a directory of CSV/Parquet files or a SQLite/DuckDB database:

```yaml
data_location:
  - data
  - /shared/powergenome_data.duckdb
input_folder: extra_inputs
```

Relative `data_location` and `input_folder` paths resolve from the settings
folder's parent directory, or a single settings file's directory. Configured
tables are searched across these locations and `input_folder`. Every table must
resolve in exactly one location; duplicate matches are rejected.

For flat files, include `.csv` or `.parquet` in the table setting. Extension
auto-detection exists, but explicit filenames are clearer. For databases, name
the table without a file extension. SQLite (`.db`, `.sqlite`) and DuckDB
(`.duckdb`) sources are supported.

## Table Configuration Patterns

### Simple Configuration

```yaml
generation_table: generators.parquet
demand_table: hourly_demand.csv
```

### Advanced Configuration

```yaml
demand_table:
  table_name: hourly_demand.parquet
  scenario: high_ev
  filters:
    - - [year, 'in', [2030, 2040]]
      - [region, 'in', [p1, p2, p3, p4]]
```

| Field | Type | Meaning |
| --- | --- | --- |
| `table_name` | string; required | Source filename or database table name. |
| `scenario` | scalar | Convenience equality filter on the source `scenario` column. |
| `filters` | nested lists | Row predicates; see below. |
| `columns` | list of strings | Source columns to expose. Omit to retain all columns. |

These examples configure table access, not a complete runnable study. Source
columns must match downstream schemas; DataManager does not rename them to
repair an incompatible dataset.

## Filter Syntax

Filters are an **OR of AND groups** (disjunctive normal form):

```yaml
filters:
  - - [year, '=', 2030]
    - [region, '=', p1]  # Both conditions must match.
  - - [year, '=', 2040]
    - [region, '=', p2]  # Or both conditions in this group.
```

Each condition is `[column, operator, value]`. Supported operators:
`=`, `==`, `!=`, `>`, `>=`, `<`, `<=`, `in`, and `not in`. Membership operators
require a list value. Match value types to source types; quote operators.

### Scenario Convenience Filter

`scenario: high_ev` adds `scenario = high_ev` to **each** AND group that has no
existing condition on `scenario`. A group already containing a scenario
condition keeps that condition rather than receiving another one. With no
explicit filters, the convenience setting creates the scenario filter itself.
The source must have a `scenario` column.

### Column Selection

`columns` limits the exposed source columns. Retain all fields required by
processing, custom modifiers, and output generation—not just fields displayed
in a short schema example. An incomplete projection can load successfully and
fail later. Omit `columns` unless the workflow's requirements are known.

### Multi-period and scenario use

Use [year-keyed values](year-keyed-values.md) or
[scenario swaps](scenario-management.md) when a source configuration genuinely
must differ by period or case. Retain coverage for every requested planning year
when sharing a multi-year demand or cost source.

For myopic multi-period models, use the same generation source across periods
to preserve cluster membership. Do not filter units out by retirement year:
blank `retirement_year` means no planned retirement, but ordinary comparison
filters exclude blank values.

## Standard Table Mappings

| Setting | Internal name | Contents |
| --- | --- | --- |
| `generation_table` | `generation` | Existing generator units. |
| `plant_region_table` | `plant_region` | Plant-to-base-region assignments. |
| `resource_heat_rate_table` | `resource_heat_rate` | New-resource heat rates. |
| `resource_cost_table` | `resource_cost` | New-resource capital and operating costs. |
| `operational_constraints_table` | `operational_constraints` | Technology operating parameters. |
| `transmission_constraints_table` | `transmission_constraints` | Base-region transmission capacities. |
| `transmission_cost_table` | `transmission_cost` | Inter-regional costs and losses. |
| `fuel_price_table` | `fuel_price` | Fuel prices by scenario, region, year, and data vintage. |
| `dollar_year_table` | `dollar_year` | CPI values for dollar-year conversion. |
| `regional_cost_factor_table` | `regional_cost_factor` | Regional construction multipliers. |
| `demand_table` | `demand` | Hourly demand. |
| `supplemental_demand_table` | `supplemental_demand` | Additional demand profiles. |
| `emission_policies_table` | `emission_policies` | Regional policy requirements. |
| `demand_segments_table` | `demand_segments` | Demand segmentation and unserved-energy costs. |
| `distributed_capacity_table` | `distributed_capacity` | Distributed capacity by model region and year. |
| `distributed_profiles_table` | `distributed_profiles` | Distributed generation profiles. |

`distributed_profile_table` is also accepted for `distributed_profiles`.
Legacy `emission_policies_fn` and `demand_segments_fn` alias their corresponding
`*_table` keys; the modern key takes precedence when both are supplied.

See [Input Table Schemas](../schemas/input-tables.md) for all table fields.
In particular, fuel prices require `data_year` even for a single vintage;
regional coverage and cost-year coverage matter beyond merely loading a table.

## Renewable Resource Tables

Renewable resource groups use JSON definitions and referenced metadata/profile
files rather than additional standard table settings. See
[Configure Renewable Clusters](../../how-to/configure-renewable-clusters.md)
for their schemas and configuration.

### `RESOURCE_GROUPS`

**Type**: directory path or list of directory paths.

All `.json` group definitions are discovered recursively under these folders.
Unlike `data_location`, relative paths resolve from the process working directory.

```yaml
RESOURCE_GROUPS: test_data/resource_groups
```

This is the bundled study's path; use your own group directory for another study.

### `RESOURCE_GROUP_PROFILES`

**Type**: directory path or list of directory paths; optional override.

Profile filenames come from the group definitions; they are not required to
follow a particular naming pattern. When directories are listed, profile lookup
searches them in order, then the environment-variable location if set, before
falling back to the group JSON's folder. Relative paths resolve from the working
directory. Tidy profiles contain `site_id`, `time_index`, `value`, and optionally
`weather_year`; weather-year selection requires tidy profiles.

## Other input files

Not every file setting is a DataManager table. For example,
`capacity_limit_spur_fn` names a legacy capacity-limit/spur CSV under
`input_folder`. See [New-build settings](new-build.md#interconnection-costs)
for that workflow and the preferred direct `interconnect_capex_mw` setting.

## Troubleshooting

- **Missing table**: verify the setting key, filename/database table, and locations.
- **Missing column**: check the canonical schema and remove incomplete projections.
- **Unexpected rows**: check AND/OR nesting, scenario conditions, and value types.
- **Incomplete period coverage**: inspect the resolved case settings and source
  years; a successful load does not prove every period has data.

Run [validation](../../how-to/validate-settings.md) after changing sources, then
inspect the generated outputs and warnings.
