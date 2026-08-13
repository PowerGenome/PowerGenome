# Add Supplemental Hourly Demand

This guide explains how to inject additional hourly demand (e.g. data-center load forecasts, new industrial loads) into an existing PowerGenome scenario on top of the baseline demand profiles.

For the full parameter reference see [Demand Settings](../reference/settings/demand.md#supplemental_demand_table).

---

## Prerequisites

- Completed the [Getting Started](../tutorials/getting-started.md) tutorial
- Baseline demand already configured via [`demand_table`](../reference/settings/demand.md#demand_table)

---

## Step 1: Create the supplemental demand file

Create a CSV (or Parquet) file with the additional load you want to add. At minimum you need three columns:

| Column | Description |
| -------- | ------------- |
| `region` | Model region name (must match your `model_regions`) |
| `time_index` | Integer hour index (1-based) **or** the string `all` / `all_hours` |
| `load_mw` | MW of demand to add |

Optional columns that PowerGenome will automatically use if present:

| Column | Description |
| -------- | ------------- |
| `year` | Planning year; when present, rows are filtered to the current `model_year` |
| `scenario` | Scenario identifier; when present **exactly one** scenario must remain after loading (see [Step 3](#step-3-select-a-scenario-optional)) |
| `weather_year` | Weather year; use `all` to apply to every weather year, a specific year (e.g. `2012`) to apply only to that block, or a blank value to **skip** the row (see [Step 4](#step-4-handle-multiple-weather-years)) |

### Using `all` / `all_hours`

Setting `time_index` to `all` (or the equivalent `all_hours`) adds a flat increment to **every** hour in the load curves. This is the most common way to add a new constant load:

```csv
region,time_index,load_mw,year
WEST,all_hours,500,2035
EAST,all,200,2035
```

### Using specific hour indices

Supply an integer `time_index` when the extra load should only appear at certain hours:

```csv
region,time_index,load_mw,year
WEST,1,300,2035
WEST,2,310,2035
WEST,3,295,2035
```

### Mixing `all_hours` and specific hours

You can include both in the same file:

```csv
region,time_index,load_mw,year,scenario
WEST,all_hours,500,2035,high_data_center
EAST,1,200,2035,high_data_center
```

---

## Step 2: Point settings at the file

### Data location is a folder

If `data_location` is a folder, place the file in that folder and reference it by name:

```yaml
# settings/data.yml
data_location: /path/to/data_folder

supplemental_demand_table: supplemental_demand.csv
```

### Data location is a database file

You can place the CSV or Parquet file **next to the database** and reference it by filename. PowerGenome resolves it relative to the database's parent directory:

```yaml
data_location: /data/pg_data.db

# resolved as /data/supplemental_demand.csv
supplemental_demand_table: supplemental_demand.csv
```

Alternatively, if the supplemental demand is stored as a table **inside** the database, reference it without an extension:

```yaml
data_location: /data/pg_data.db

supplemental_demand_table: supplemental_demand  # table inside pg_data.db
```

---

## Step 3: Select a scenario (optional)

If your supplemental demand file contains multiple scenarios (identified by a `scenario` column), you **must** specify which one to use. Without a selection, PowerGenome raises an error listing the available options.

Use the dictionary format:

```yaml
supplemental_demand_table:
  table_name: supplemental_demand.csv
  scenario: high_data_center
```

PowerGenome will filter the table to rows where `scenario = "high_data_center"` before applying the demand additions.

!!! tip "Multi-scenario studies"
    You can vary the `supplemental_demand_table` scenario across runs using the [scenario management](run-scenarios.md) system, just like any other settings value.

---

## Step 4: Handle multiple weather years

If your load curves span several weather years (e.g. three years of 8 760 hours = 26 280 hours total), the `weather_year` column controls which weather-year block a row is applied to:

**Option A — apply to every weather year (most common)**

Leave out the `weather_year` column entirely, and use `all_hours` or a specific `time_index`. PowerGenome automatically adds the supplemental demand across every weather-year block.

```csv
region,time_index,load_mw,year
WEST,all_hours,500,2035
```

```csv
region,time_index,load_mw,year,weather_year
WEST,all_hours,500,2035,all
```

**Option B — apply only to a specific weather year**

Give the row a specific `weather_year` value. An `all` / `all_hours` row then applies to every hour of only that year's block; a specific integer `time_index` applies to just that hour within the block.

```csv
region,time_index,load_mw,year,weather_year
WEST,all_hours,500,2035,2012      # every hour of the 2012 block
WEST,all_hours,300,2035,2013      # every hour of the 2013 block
WEST,3,400,2035,2013              # hour 3 of the 2013 block only
```

!!! warning "Blank `weather_year` rows are skipped"
    A row with a blank/empty `weather_year` is **not** applied. Use `weather_year: all` when you want a row to apply across every weather year.

!!! warning "Coverage check"
    When your load data uses multiple weather years, every weather year must be covered by the supplemental demand table — either by a row for that specific year or by a row with `weather_year: all`. A missing weather year raises an error naming the years that aren't covered. No coverage check is performed when the supplemental demand table has no `weather_year` column.

The block size is controlled by `hours_per_year` (default 8 760).

---

## Complete example

**`supplemental_demand.csv`**:

```csv
region,time_index,load_mw,year,scenario
WEST,all_hours,500,2030,base_data_center
WEST,all_hours,800,2035,base_data_center
EAST,all_hours,200,2030,base_data_center
EAST,all_hours,300,2035,base_data_center
WEST,all_hours,1000,2030,high_data_center
WEST,all_hours,1500,2035,high_data_center
EAST,all_hours,400,2030,high_data_center
EAST,all_hours,600,2035,high_data_center
```

**`settings/data.yml`**:

```yaml
data_location: /data/pg_data.db

supplemental_demand_table:
  table_name: supplemental_demand.csv
  scenario: high_data_center
```

PowerGenome will:

1. Load rows where `scenario = "high_data_center"`
2. Filter to the current `model_year` (because the `year` column is present)
3. Add the `load_mw` values to every hour in every matching region

---

## Related documentation

- [Demand Settings Reference](../reference/settings/demand.md): Full `supplemental_demand_table` parameter reference
- [Configure Data Tables](configure-data-tables.md): General data table configuration guide
- [Run Multi-Scenario Studies](run-scenarios.md): Vary supplemental demand across scenarios
