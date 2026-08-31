# Data Tables Settings

PowerGenome loads data from tables (CSV, Parquet, DuckDB) configured in settings. Table configurations support filtering, column selection, and scenario-based data loading through the DataManager.

## Table Configuration Patterns

### Simple Configuration

Point to a filename in `data_location`:

```yaml
generation_table: generators.csv
fuel_prices_table: fuel_prices.parquet
demand_table: hourly_demand.csv
```

Files are loaded from the folder specified in `data_location`.

### Advanced Configuration

Use dictionary format for filtering and column selection:

```yaml
generation_table:
  table_name: generators.parquet
  scenario: high_retirements
  filters:
    - - [operating_year, '<=', 2030]
      - [retirement_year, '>', 2030]
  columns: [plant_id, technology, capacity_mw, heat_rate, region]
```

**Parameters**:

- `table_name`: Filename or database table
- `scenario`: Quick filter for scenario column
- `filters`: DNF (Disjunctive Normal Form) filter logic
- `columns`: Limit columns loaded (for large files)

## Filter Syntax

### DNF (Disjunctive Normal Form)

Filters use nested lists representing `(A AND B) OR (C AND D)` logic:

```yaml
filters:
  - - [column1, operator, value1]
    - [column2, operator, value2]  # AND with above
  - - [column3, operator, value3]  # OR with first group
```

**Operators**:

- `=` or `==`: Equality
- `!=`: Not equal
- `>`: Greater than
- `>=`: Greater than or equal
- `<`: Less than
- `<=`: Less than or equal
- `in`: Value in list
- `not in`: Value not in list

### Filter Examples

**Simple filter** (single condition):

```yaml
filters:
  - - [year, '=', 2030]
```

**AND filter** (both conditions must be true):

```yaml
filters:
  - - [year, '>=', 2030]
    - [region, '=', 'CA_N']
```

**OR filter** (either condition can be true):

```yaml
filters:
  - - [region, '=', 'CA_N']
  - - [region, '=', 'CA_S']
```

**Complex filter** ((A AND B) OR C):

```yaml
filters:
  - - [year, '=', 2030]
    - [scenario, '=', 'high']
  - - [year, '=', 2040]
```

**List membership**:

```yaml
filters:
  - - [region, 'in', ['CA_N', 'CA_S', 'AZ']]
    - [technology, 'not in', ['Coal', 'Oil']]
```

### Scenario Convenience Filter

The `scenario` parameter is shorthand for a scenario column filter:

```yaml
# These are equivalent:
generation_table:
  table_name: generators.parquet
  scenario: baseline

generation_table:
  table_name: generators.parquet
  filters:
    - - [scenario, '=', 'baseline']
```

## Standard Table Mappings

PowerGenome expects certain table names for core data:

| Setting Parameter | Standard Name | Purpose |
|-------------------|---------------|---------|
| `generation_table` | `generation` | Existing generators |
| `demand_table` | `demand` | Hourly demand profiles |
| `fuel_prices_table` | `fuel_prices` | Fuel cost time series |
| `transmission_table` | `transmission` | Network constraints |
| `plant_region_table` | `plant_region` | Plant location mapping |
| `capacity_limit_spur_table` | `capacity_limit_spur` | New-build capacity limits |
| `dg_capacity_table` | `dg_capacity` | Distributed gen capacity |
| `dg_profiles_table` | `dg_profiles` | Distributed gen profiles |
| `supplemental_demand_table` | `supplemental_demand` | Extra hourly demand (data centers, etc.) |

## Core Data Tables

### Generation Table

**Setting**: `generation_table`
**Purpose**: Existing power plant data

**Required columns**:

- `plant_id` or `plant_id_eia`: Unique plant identifier
- `technology`: Technology type
- `capacity_mw`: Nameplate capacity
- `region`: Geographic region

**Optional columns**:

- `heat_rate_mmbtu_mwh`: Thermal efficiency
- `operating_year`: Year plant started
- `retirement_year`: Planned retirement
- `fixed_o_m_mw`: Fixed O&M costs
- `variable_o_m_mwh`: Variable O&M costs
- `minimum_load_mw`: Minimum stable operation
- `fuel`: Fuel type

**Example**:

```yaml
generation_table:
  table_name: generators.parquet
  filters:
    - - [operating_year, '<=', 2030]
      - [capacity_mw, '>', 10]  # Exclude tiny plants
```

### Demand Table

**Setting**: `demand_table`
**Purpose**: Hourly electricity demand in tidy format

**Required columns**:

- `region`: Model region name
- `time_index`: Hour index (1 to number of hours)
- `load_mw`: Demand in MW

**Optional columns**:

- `year`: Model year (for multi-year data)
- `scenario`: Scenario identifier
- `weather_year`: Weather data year

**Format**: Tidy/long format with one row per region-time observation.

**Example**:

```yaml
demand_table:
  table_name: demand_timeseries.parquet
  scenario: high_ev
  filters:
    - - [year, '=', 2030]
```

**Example demand CSV (tidy format)**:

```csv
time_index,weather_year,region,load_mw,year
1,2012,CA_N,15234.5,2030
2,2012,CA_N,14123.2,2030
3,2012,CA_N,13890.4,2030
1,2012,CA_S,12450.8,2030
2,2012,CA_S,11234.5,2030
...
```

### Fuel Prices Table

**Setting**: `fuel_prices_table`
**Purpose**: Fuel cost projections

**Required columns**:

- `fuel`: Fuel name (coal, naturalgas, distillate, uranium, etc.)
- `region`: Region name
- `year`: Calendar year
- `price`: Price ($/MMBtu)

**Optional columns**:

- `scenario`: Price scenario
- `data_year`: Data vintage year
- `dollar_year`: USD year (for inflation adjustment)
- `month`: Monthly prices (if seasonal)

!!! important "Regional Coverage"
    The fuel price table must include prices for **all base regions** in your model. For example, if you have an aggregated region `CA` composed of base regions `CA_N` and `CA_S`, your fuel price table must contain separate rows for `CA_N` and `CA_S`. PowerGenome will automatically calculate the average price for `CA` from its constituent base regions.

**Example**:

```yaml
fuel_prices_table:
  table_name: fuel_prices.parquet
  scenario: high_gas
```

**Example fuel_prices.csv**:

```csv
fuel,region,year,price,scenario,dollar_year
coal,CA_N,2030,2.5,reference,2024
coal,CA_S,2030,2.6,reference,2024
naturalgas,CA_N,2030,4.2,reference,2024
naturalgas,CA_S,2030,4.3,reference,2024
uranium,CA_N,2030,0.8,reference,2024
uranium,CA_S,2030,0.8,reference,2024
```

### Transmission Table

**Setting**: `transmission_table`
**Purpose**: Inter-regional transmission constraints

**Required columns**:

- `transmission_path_name`: Line identifier
- `start_region`: Origin region
- `dest_region`: Destination region
- `transmission_line_mw`: Existing capacity

**Optional columns**:

- `max_transmission_mw`: Maximum capacity
- `distance_miles`: Line length
- `line_loss_pct`: Transmission losses

**Example**:

```yaml
transmission_table: transmission_network.csv
```

## Distributed Generation Tables

### DG Capacity Table

**Setting**: `dg_capacity_table`
**Purpose**: Distributed generation capacity by region/year

**Required columns**:

- `region`: Model region
- `year`: Model year
- `capacity_mw`: DG capacity

**Optional columns**:

- `scenario`: DG adoption scenario
- `technology`: DG technology type

**Example**:

```yaml
dg_capacity_table:
  table_name: dg_capacity.parquet
  scenario: high_solar
```

### DG Profiles Table

**Setting**: `dg_profiles_table`
**Purpose**: Hourly generation profiles for DG

**Required columns**:

- `region`: Model region
- `hour`: Hour of year
- `cf`: Capacity factor (0-1)

**Optional columns**:

- `weather_year`: Weather data vintage
- `technology`: DG type

**Example**:

```yaml
dg_profiles_table: dg_generation_profiles.parquet
```

## Renewable Resource Tables

### Resource Groups

**Setting**: `RESOURCE_GROUPS` (path, or list of paths)
**Purpose**: Folder(s) containing renewable resource group definition JSON files
(one JSON per resource group). All `.json` files are discovered recursively in
each configured folder, so you can point at a single folder or split the JSON files
across several folders.

**Example**:

```yaml
RESOURCE_GROUPS: /data/resource_groups
```

Or from multiple folders:

```yaml
RESOURCE_GROUPS:
  - /data/resource_groups/onshore
  - /data/resource_groups/offshore
```

### Resource Group Profiles

**Setting**: `RESOURCE_GROUP_PROFILES` (path, or list of paths)
**Purpose**: Hourly generation profiles for renewable clusters

`RESOURCE_GROUP_PROFILES` accepts either a single folder or a list of folders.
When a list is provided, each profile file named in a resource group JSON is
searched for in the listed folders (in order, plus the `RESOURCE_GROUP_PROFILES`
environment variable if set) before falling back to the resource group JSON's own
folder. This lets you, for example, keep wind profiles in one folder and solar
profiles in another.

**File naming**: `{technology}_profiles_{suffix}.csv` or `.parquet`

**Format**: Tidy format with columns:

- `weather_year`: Weather data year
- `time_index`: Hour index (1 to number of hours)
- `site_id`: Resource cluster/site identifier
- `value`: Capacity factor (0.0 to 1.0)

**Example**:

```yaml
RESOURCE_GROUP_PROFILES: /data/nrel_profiles/
```

Or from multiple folders:

```yaml
RESOURCE_GROUP_PROFILES:
  - /data/nrel_profiles/wind
  - /data/nrel_profiles/solar
```

Files in folder:

```
landbasedwind_profiles_class3.parquet
utilitypv_profiles_class1.parquet
offshorewind_profiles_class1.parquet
```

**Example profile CSV (tidy format)**:

```csv
weather_year,time_index,site_id,value
2012,1,3244,0.45
2012,2,3244,0.42
2012,3,3244,0.48
2012,1,3818,0.52
2012,2,3818,0.48
...
```

## Policy and Cost Tables

### Emission Policies

**Setting**: `emission_policies_fn`
**Purpose**: RPS, CES, carbon constraints by case/region/year

This setting names a DataManager table source. It can use the same file or
database table configuration supported by other DataManager inputs; it is not
limited to a CSV in `input_folder`.

**Required columns**:

- `case_id`: Case identifier
- `year`: Model year
- `region`: Model region (or "all")

**Policy columns** (example):

- `RPS`: Renewable portfolio standard (fraction)
- `CES`: Clean energy standard (fraction)
- `CO2_cap`: CO2 emissions limit (tonnes)

**Example**:

```yaml
emission_policies_fn: emission_policies.csv
```

### Demand Segments

**Setting**: `demand_segments_fn`
**Purpose**: Value of lost load and demand segmentation parameters

This setting names a DataManager table source and supports the same file and
database table configurations as other DataManager inputs.

### Cost Multipliers

**Setting**: `cost_multiplier_fn`
**Purpose**: Regional construction cost adjustments

**Required columns**:

- `region`: Region name
- `technology`: Technology name
- `multiplier`: Cost multiplier (1.2 = 20% increase)

**Example**:

```yaml
cost_multiplier_fn: regional_cost_multipliers.csv
```

### Capacity Limits and Spur Lines

**Setting**: `capacity_limit_spur_fn`
**Purpose**: New-build capacity limits and connection costs

**Required columns**:

- `region`: Model region
- `technology`: Technology name
- `max_capacity`: Maximum capacity (MW)
- `spur_miles`: Spur line distance

**Optional columns**:

- `cluster`: Resource cluster ID
- `spur_capex_mw_mile`: Spur line cost

**Example**:

```yaml
capacity_limit_spur_fn: capacity_limits.csv
```

## Data File Formats

### CSV Files

Standard CSV format with header row:

```csv
plant_id,technology,capacity_mw,region
1,Coal,500,CA_N
2,NaturalGas_CC,300,CA_S
```

### Parquet Files

Binary columnar format (more efficient for large datasets):

```python
import pandas as pd
df = pd.read_parquet("generators.parquet")
```

PowerGenome automatically detects and loads Parquet files.

### Database Tables

Database tables (SQLite or DuckDB format):

```yaml
data_location: /path/to/database.duckdb

generation_table:
  table_name: existing_generators  # Table name in database
```

Both SQLite (`.db`, `.sqlite`) and DuckDB (`.duckdb`) databases are supported:

```yaml
# SQLite database
data_location: /path/to/powergenome_data.db

# DuckDB database
data_location: /path/to/powergenome_data.duckdb
```

## Data Location

### `data_location`

**Type**: String (path)
**Required**: Yes
**Example**: `"/path/to/data"`

Root directory or database file for all data tables.

**Folder structure**:

```yaml
data_location: /Users/me/powergenome_data
```

Files loaded from:

```
/Users/me/powergenome_data/generators.csv
/Users/me/powergenome_data/demand.parquet
/Users/me/powergenome_data/fuel_prices.csv
```

**Database file**:

```yaml
data_location: /Users/me/powergenome_data/model_data.duckdb
```

Tables loaded from database.

## Column Selection

Limit columns loaded to reduce memory usage:

```yaml
generation_table:
  table_name: generators_full.parquet
  columns:
    - plant_id
    - technology
    - capacity_mw
    - heat_rate_mmbtu_mwh
    - region
  filters:
    - - [region, 'in', ['CA_N', 'CA_S']]
```

Only specified columns are loaded. Useful for large datasets with many unused columns.

## Example Configurations

### Minimal Configuration

```yaml
data_location: /data/powergenome

generation_table: generators.csv
demand_table: demand.parquet
fuel_prices_table: fuel_prices.csv
```

### Advanced Multi-Scenario

```yaml
data_location: /data/powergenome_v2

generation_table:
  table_name: generators_all_scenarios.parquet
  scenario: baseline
  filters:
    - - [operating_year, '<=', 2030]
  columns:
    - plant_id
    - technology
    - capacity_mw
    - heat_rate_mmbtu_mwh
    - region
    - operating_year

demand_table:
  table_name: demand_timeseries.parquet
  scenario: high_ev
  filters:
    - - [year, '=', 2030]
    - - [weather_year, 'in', [2012, 2013, 2014]]

fuel_prices_table:
  table_name: fuel_costs.parquet
  scenario: reference
  filters:
    - - [year, '>=', 2025]
    - - [year, '<=', 2050]

dg_capacity_table:
  table_name: distributed_generation.parquet
  scenario: high_solar
  filters:
    - - [technology, '=', 'rooftop_pv']

transmission_table: transmission_network.csv
```

## Troubleshooting

### Table Not Found

**Error**: `Table 'generation' not found in data_location`

**Solutions**:

- Check `data_location` path is correct
- Verify filename matches `table_name`
- Ensure file has correct extension (.csv, .parquet)
- For databases, verify table exists with `list_tables()`

### Column Missing

**Error**: `Column 'capacity_mw' not found in table`

**Solutions**:

- Check column names in source data (case-sensitive)
- Verify data schema matches expectations
- PowerGenome converts column names to snake_case automatically

### Filter Errors

**Error**: `Invalid filter syntax`

**Solutions**:

- Check filter uses proper DNF nested list format
- Verify operators are quoted: `'='` not `=`
- Ensure values match column data types (string vs. numeric)

## Related Settings

- [Model Definition](model-definition.md): `data_location` path
- [Existing Generators](existing-generators.md): `generation_table` schema
- [Demand](demand.md): `demand_table`, `dg_capacity_table`, `dg_profiles_table`
- [Fuels](fuels.md): `fuel_prices_table`
- [Transmission](transmission.md): `transmission_table`
