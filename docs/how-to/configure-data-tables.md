# Configure Data Tables

Use this guide to point an existing study at source files and select rows from a
multi-scenario dataset. It does not create or download the data. See
[Input Table Schemas](../reference/schemas/input-tables.md) for required columns
and [Data Tables Settings](../reference/settings/data-tables.md) for complete syntax.

## 1. Set the search locations

Put paths in your settings, for example `settings/data.yml`:

```yaml
data_location: data
input_folder: extra_inputs
```

With a settings directory, these paths resolve from its parent directory; with
a single settings file, they resolve from that file's directory. This example
expects `data/`, `extra_inputs/`, and `settings/` alongside one another.

PowerGenome searches both locations. Each configured table must resolve in
exactly one place; do not keep duplicate copies with the same table name in both.
`data_location` can also be a list of locations or a SQLite/DuckDB file.

`RESOURCE_GROUPS` and `RESOURCE_GROUP_PROFILES` are the exception: they resolve
from the process working directory, not from `data_location` or the settings
folder. The published examples write them relative to the examples root
(`example_systems/TX-3-zone/resource_groups`), which is why their instructions
run every command from that root.

## 2. Name your source tables

Use exact settings keys and source filenames:

```yaml
generation_table: generators.parquet
plant_region_table: plant_region.csv
demand_table: hourly_demand.parquet
fuel_price_table: fuel_prices.csv
transmission_constraints_table: transmission_capacity.csv
transmission_cost_table: network_costs.csv
```

These are illustrative filenames, not a complete study. Keep the other table
settings and processing parameters your workflow needs. For a database backend,
use database table names without file extensions.

### Published example

The [Texas `settings/data.yml`](https://github.com/PowerGenome/PowerGenome-examples/blob/7e34f714469e68fca173cdb849f74c47be779420/example_systems/TX-3-zone/settings/data.yml)
maps the same keys to real files:

```yaml
data_location: data
RESOURCE_GROUPS: example_systems/TX-3-zone/resource_groups
RESOURCE_GROUP_PROFILES: example_systems/TX-3-zone/resource_profiles

generation_table: reeds_generators_tx_3_zone.csv
plant_region_table: plant_region_map_tx_3_zone.csv
demand_table: reeds_load_tx_3_zone.csv
fuel_price_table: fuel_prices_tx_3_zone.csv
transmission_constraints_table: transmission_capacity_tx_3_zone.csv
transmission_cost_table: network_costs_tx_3_zone.csv
regional_cost_factor_table: regional_cost_multipliers_tx_3_zone.csv
```

It also configures heat-rate, cost, operational-constraint, dollar-year, and
distributed-generation tables. The CA/AZ example uses the same keys with plain
names (`generators.csv`, `demand_load.csv`, `fuel_prices.csv`) because its data
folder holds a single study. The filenames are yours to choose; the setting keys
are not.

## 3. Select a demand scenario

Suppose `hourly_demand.parquet` contains multiple scenarios and planning years.
Replace its string setting with a dictionary:

```yaml
demand_table:
  table_name: hourly_demand.parquet
  scenario: high_ev
  filters:
    - - [region, 'in', [p1, p2, p3, p4]]
      - [year, 'in', [2030, 2040]]
```

This selects `high_ev` rows for the four base regions and both planning years.
Use names and years present in your own data. Demand processing selects the
requested planning year; do not accidentally filter out later periods.

The two conditions in one group are **ANDed**. Separate groups are **ORed**;
see the [filter reference](../reference/settings/data-tables.md#filter-syntax)
for that syntax and the interaction with `scenario`.

!!! warning "Keep generation membership stable"
    For myopic multi-period models, point every planning period at the same
    generation data. Per-period filters can change cluster membership. Also,
    `retirement_year > 2030` removes blank retirement years, although blank means
    no planned retirement. Let the generator workflow label operating capacity.

## 4. Validate and inspect

Run from your study directory:

```bash
validate_powergenome --settings_file settings
```

Fix missing paths, unknown regions, missing columns, and coverage warnings before
using the inputs for analysis. Then run your study and inspect the saved
`powergenome_case_settings.yml` to confirm the intended table configuration was
used in each period.

| Problem | Check |
| --- | --- |
| Table not found | Exact filename/table name, recognized setting key, and configured locations. |
| Duplicate table match | Remove the duplicate source or give sources distinct names. |
| Required column missing | Source schema and any `columns` projection. Omit `columns` unless you know all downstream requirements. |
| Wrong rows or empty result | Filter nesting, data types, scenario spelling, and required region/year coverage. |

## Next steps

- [Data Tables Settings](../reference/settings/data-tables.md): mappings,
  database sources, filters, and renewable path settings.
- [Run Multi-Scenario Studies](run-scenarios.md): swap sources between cases.
- [Use Year-Keyed Settings](use-year-keyed-settings.md): change a configuration
  by planning year when the source requires it.
