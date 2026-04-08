# Demand Settings

These parameters control electricity demand data sources for the model.

## Demand Data

### `demand_table`

**Type**: String or dictionary
**Required**: Yes
**Example**: See below

Table containing hourly electricity demand projections for all model regions and planning periods.

**Simple**:

```yaml
demand_table: load_curves.csv
```

**Advanced** (with scenario selection):

```yaml
demand_table:
  table_name: demand_timeseries.parquet
  scenario: high_electrification
```

**Required columns**:

- `region`: Model region name
- `time_index`: Hour index
- `load_mw`: Demand in MW
- `year`: Planning year

**Optional columns**:

- `scenario`: Demand scenario identifier (reference, high_ev, high_electrification, etc.)
- `weather_year`: Weather data vintage year

**Format**: Tidy/long format with one row per region-time-year observation.

**Example demand CSV**:

```csv
time_index,weather_year,region,load_mw,year,scenario
1,2012,CA_N,15234.5,2030,reference
2,2012,CA_N,14123.2,2030,reference
3,2012,CA_N,13890.4,2030,reference
1,2012,CA_S,12450.8,2030,reference
...
1,2012,CA_N,16890.2,2040,reference
2,2012,CA_N,15678.9,2040,reference
...
```

!!! important "Multi-Period Coverage"
    The `demand_table` should contain hourly demand projections for **all future modeling periods**. For example, if your model runs from 2030 to 2050 in 5-year increments, the table should include demand data for 2030, 2035, 2040, 2045, and 2050.

### Demand Scenarios

Use the `scenario` column to manage different demand futures:

```yaml
demand_table:
  table_name: demand_projections.parquet
  scenario: high_electrification  # Options: reference, high_ev, high_electrification
```

Common scenarios:

- **reference**: Base case load growth
- **high_ev**: Increased electric vehicle adoption
- **high_electrification**: Widespread building/industrial electrification
- **low_growth**: Slower demand growth with efficiency improvements

## Supplemental Demand

### `supplemental_demand_table`

**Type**: String or dictionary
**Required**: No
**Example**: See below

Optional table of additional hourly demand (e.g. data-center forecasts, new industrial loads) to add on top of the baseline demand profiles. Added after distributed-generation subtraction and before integer conversion.

**Simple** (file in `data_location` folder, or co-located with a database):

```yaml
supplemental_demand_table: supplemental_demand.csv
```

**Advanced** (with scenario selection):

```yaml
supplemental_demand_table:
  table_name: supplemental_demand.csv
  scenario: high_data_center
```

**Required columns**:

- `region`: Model region name (must match `model_regions`)
- `time_index`: Integer hour index (1-based) **or** the string `all_hours`
- `load_mw`: MW of demand to add

**Optional columns**:

- `year`: When present, rows are filtered to the current `model_year`
- `scenario`: Scenario identifier; when present, exactly one scenario must remain after
  loading. If multiple scenarios are found and none has been selected, PowerGenome raises
  a descriptive error listing the available scenario names and showing how to select one
  (see below).
- `weather_year`: When present, rows with a blank `weather_year` are tiled across all
  weather-year blocks (block size controlled by `hours_per_year`, default 8760); rows
  with a specific `weather_year` are applied only to that block.

**`time_index` values**:

- **`all_hours`** – adds the demand increment to every hour in the load curves (flat load addition).
- **Integer** – adds the demand increment only to the hour with that index.

**Scenario handling**:

If the supplemental demand file contains multiple scenarios and no scenario is specified, PowerGenome raises:

```
ValueError: The supplemental_demand table contains multiple scenarios
('baseline', 'high_data_center') but no scenario has been selected.
Specify which scenario to use in your settings file, for example:

    supplemental_demand_table:
      table_name: supplemental_demand.csv
      scenario: baseline
```

**File co-located with a database**:

When `data_location` points to a database file (`.db`, `.duckdb`), a filename with a
`.csv` or `.parquet` extension is resolved relative to the database's parent directory:

```yaml
data_location: /data/pg_data.db
supplemental_demand_table: supplemental_demand.csv  # → /data/supplemental_demand.csv
```

**Example supplemental demand CSV**:

```csv
region,time_index,load_mw,year,scenario
WEST,all_hours,500,2030,base_data_center
WEST,all_hours,800,2035,base_data_center
EAST,all_hours,200,2030,base_data_center
EAST,1,300,2030,base_data_center
WEST,all_hours,1000,2030,high_data_center
WEST,all_hours,1500,2035,high_data_center
EAST,all_hours,400,2030,high_data_center
```

For a step-by-step guide see [Add Supplemental Hourly Demand](../../how-to/add-supplemental-demand.md).

## Related Settings

- [Data Tables](data-tables.md): Format specification for all data tables
- [Regions](regions.md): Model region definitions
- [Time Reduction](time-reduction.md): Representative period selection
- [Model Definition](model-definition.md): Planning years configuration
