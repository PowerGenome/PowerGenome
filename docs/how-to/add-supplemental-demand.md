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
|--------|-------------|
| `region` | Base region **or** model region name (see [Step 5: Choose base or model regions](#step-5-choose-base-or-model-regions)) |
| `time_index` | Integer hour index (1-based) **or** the string `all` / `all_hours` |
| `load_mw` | MW of demand to add |

Optional columns that PowerGenome will automatically use if present:

| Column | Description |
|--------|-------------|
| `year` | Planning year; when present, rows are filtered to the current `model_year` |
| `scenario` | Scenario identifier; when present **exactly one** scenario must remain after loading (see [Step 3](#step-3-select-a-scenario-optional)) |
| `weather_year` | Weather year; use `all` to apply to every weather year, a specific year (e.g. `2012`) to apply only to that weather year, or a blank value to **skip** the row (see [Step 4](#step-4-handle-multiple-weather-years)) |

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

If your load curves span several weather years (e.g. three years of 8 760 hours = 26 280 hours total), the `weather_year` column controls which weather year a row is applied to:

**Option A — apply to every weather year (most common)**

Leave out the `weather_year` column entirely and use `all` / `all_hours`, or set `weather_year` to `all`. PowerGenome applies the row to every weather year present in the load data.

```csv
region,time_index,load_mw,year
WEST,all_hours,500,2035
```

```csv
region,time_index,load_mw,year,weather_year
WEST,all_hours,500,2035,all
```

**Option B — apply only to a specific weather year**

Give the row a specific `weather_year` value. An `all` / `all_hours` row then applies to every hour of only that weather year; a specific integer `time_index` applies to just that hour within the weather year.

```csv
region,time_index,load_mw,year,weather_year
WEST,all_hours,500,2035,2012      # every hour of 2012
WEST,all_hours,300,2035,2013      # every hour of 2013
WEST,3,400,2035,2013              # hour 3 of 2013 only
```

!!! warning "Blank `weather_year` rows are skipped"
    A row with a blank/empty `weather_year` is **not** applied. Use `weather_year: all` when you want a row to apply across every weather year.

!!! warning "Coverage check"
    When the supplemental demand table includes a `weather_year` column **and** the load data's weather years are known (via the `weather_year` setting), every weather year present in the load data must be covered by the supplemental demand table — either by a row for that specific year or by a row with `weather_year: all`. A missing weather year raises an error naming the years that aren't covered. No coverage check is performed when the supplemental demand table has no `weather_year` column.

!!! tip "How supplemental demand is added"
    Supplemental demand is joined into the **base load data stage**, before the per-weather-year hours are renumbered 1..N and before base regions are aggregated into model regions. Rows that share the same `(region, weather_year, time_index)` are summed together, so there is **no tiling** and **no fixed weather-year length assumption**: weather years of different lengths (e.g. a 8 784-hour leap year next to a normal 8 760-hour year) are handled correctly. This is unlike the older implementation, which block-tiled supplemental load using a fixed `hours_per_year`.

---

## Step 5: Choose base or model regions

The `region` column accepts **either** a base region name or a model region name:

- **Base region name** — the row is mapped to the model region that contains that base region (as configured by `region_aggregations`), and its demand is added to the aggregate's load. If a base region is part of an aggregation, the added demand contributes to the aggregated model region.
- **Model region name** — the row is applied as-is to the model region.

When the name is a base region that is part of a model-region aggregation, the supplemental demand is added to that one base region, and the aggregation then includes it in the model region. This is equivalent to adding it directly to the model region. Names that match neither a known base region nor a known model region are logged as a warning and ignored.

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
