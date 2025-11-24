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

## Related Settings

- [Data Tables](data-tables.md): Format specification for `demand_table`
- [Regions](regions.md): Model region definitions
- [Time Reduction](time-reduction.md): Representative period selection
- [Model Definition](model-definition.md): Planning years configuration
