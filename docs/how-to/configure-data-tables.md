# Configure Data Tables

PowerGenome reads all input data through a central **DataManager** that loads tables from a folder of (CSV or Parquet) files or a (DuckDB or Sqlite) database. This guide walks through how to point PowerGenome at your data files and how to filter or select the data you need.

For the full parameter reference, see [Data Tables Settings](../reference/settings/data-tables.md).

## Prerequisites

- Completed the [Getting Started](../tutorials/getting-started.md) tutorial
- A folder containing your source data files (CSV or Parquet)

---

## Step 1: Set the data location

Tell PowerGenome where your data files live with `data_location`:

```yaml
# settings/data.yml
data_location: /path/to/your/data_folder # Can also be /path/to/database.db
```

Paths may be relative to the settings file (or settings directory), and multiple
locations may be provided:

```yaml
data_location:
  - data
  - /shared/powergenome_data.db
input_folder: extra_inputs
```

PowerGenome searches `input_folder` as well as `data_location` for configured
tables. Each table must be found in exactly one location; duplicate table names
are rejected.

---

## Step 2: Configure each table

### Simple configuration (filename only)

Point each setting to a file by name:

```yaml
generation_table: generators.parquet
demand_table: hourly_demand.parquet
fuel_price_table: fuel_prices.csv
transmission_constraints_table: transmission_constraints.csv
```

PowerGenome resolves each filename inside the configured data locations.

### Advanced configuration (with filters)

Use a dictionary when you need to filter rows, select columns, or pick a
specific scenario from a multi-scenario file:

```yaml
generation_table:
  table_name: generators.parquet
  scenario: baseline
  filters:
    - - [capacity_mw, '>', 10]
```

**Keys:**

| Key | Required | Description |
|-----|----------|-------------|
| `table_name` | Yes | Filename in `data_location` |
| `scenario`   | No  | Filter rows where `scenario` column equals this value |
| `filters`    | No  | DNF filter logic (see below) |
| `columns`    | No  | Load only these columns (useful for large files) |

---

## Step 3: Write filter expressions

### DNF filter syntax

Filters use **Disjunctive Normal Form**: a list of AND-groups that are OR'd together.

```
(A AND B) OR (C AND D)
```

```yaml
filters:
  - - [column, operator, value]   # Group 1, condition A
    - [column, operator, value]   # Group 1, condition B  (AND)
  - - [column, operator, value]   # Group 2, condition C  (OR)
```

**Supported operators**: `=`, `==`, `!=`, `>`, `>=`, `<`, `<=`, `in`, `not in`

### Common filter patterns

**Single condition:**

```yaml
demand_table:
  table_name: demand.parquet
  filters:
    - - [year, '=', 2030]
```

**AND (two conditions must both be true):**

```yaml
generation_table:
  table_name: generators.parquet
  filters:
    - - [operating_year, '<=', 2030]
      - [retirement_year, '>', 2030]
```

**OR (either region is included):**

```yaml
demand_table:
  table_name: demand.parquet
  filters:
    - - [region, '=', 'CA_N']
    - - [region, '=', 'CA_S']
```

**List membership (`in`):**

```yaml
generation_table:
  table_name: generators.parquet
  filters:
    - - [region, 'in', ['CA_N', 'CA_S', 'AZ']]
```

---

## Step 4: Handle multi-year data

If your data file spans multiple planning years, use a `year` filter so
each run only sees the rows for that period:

```yaml
demand_table:
  table_name: demand_all_years.parquet
  filters:
    - - [year, '=', 2030]
```

For settings that change across planning years, use a [year-keyed dictionary](use-year-keyed-settings.md):

```yaml
demand_table:
  2030:
    table_name: demand_all_years.parquet
    filters:
      - - [year, '=', 2030]
  2040:
    table_name: demand_all_years.parquet
    filters:
      - - [year, '=', 2040]
```

---

## Step 5: Combine `scenario` with filters

`scenario` is a shorthand that adds a `scenario = <value>` filter. You can
combine it with additional filters — PowerGenome ANDs the scenario condition
into every OR-group:

```yaml
generation_table:
  table_name: generators.parquet
  scenario: high_retirements
  filters:
    - - [capacity_mw, '>=', 1]
```

This is equivalent to:

```yaml
generation_table:
  table_name: generators.parquet
  filters:
    - - [scenario, '=', 'high_retirements']
      - [capacity_mw, '>=', 1]
```

---

## Step 6: Reduce memory use with column selection

For large Parquet files, load only the columns your model needs:

```yaml
generation_table:
  table_name: generators.parquet
  columns:
    - plant_id
    - technology
    - capacity_mw
    - heat_rate_mmbtu_mwh
    - region
    - operating_year
```

---

## Step 7: Use a DuckDB database (optional)

If your data is stored in a database (`.db` or `.duckdb`) file rather than a folder of
flat files, point `data_location` at the database file:

```yaml
data_location: /path/to/powergenome_data.db
```

Table names then refer to tables inside the database:

```yaml
generation_table: generators
demand_table: hourly_demand
```

Do **not** include a file extension when using a database backend.

---

## Table reference

The following settings parameters map to tables in DataManager:

| Setting parameter | Standard table name | Typical use |
|---|---|---|
| `generation_table` | `generation` | Existing power plants |
| `demand_table` | `demand` | Hourly load time series |
| `fuel_price_table` | `fuel_price` | Fuel cost time series |
| `transmission_constraints_table` | `transmission_constraints` | Network topology |
| `plant_region_table` | `plant_region` | Plant-to-region mapping |
| `distributed_capacity_table` | `distributed_capacity` | Rooftop solar capacity |
| `distributed_profiles_table` | `distributed_profiles` | Rooftop solar generation |

---

## Troubleshooting

**"Table not found" error**

Check that the filename is spelled correctly and is present in `data_location`:

```bash
ls /path/to/your/data_folder
```

Verify the setting key is exact (e.g., `generation_table`, not `generators_table`).

**Wrong data being loaded**

Use `list_tables()` to confirm what DataManager sees:

```python
from powergenome.database import get_data_manager
dm = get_data_manager()
print(dm.list_tables())
```

**File has no extension**

PowerGenome tries to auto-detect `.csv` or `.parquet` files by appending extensions, but it's best practice to always include the extension in your setting.

---

## Related documentation

- [Data Tables Settings Reference](../reference/settings/data-tables.md): Full parameter documentation and filter syntax
- [Configure Settings](configure-settings.md): Settings file organization
- [Run Multi-Scenario Studies](run-scenarios.md): Varying data sources across scenarios
