# Demand Settings

These parameters control load profile construction, growth projections, distributed generation subtraction, and demand response configuration.

## Load Profile Sources

### `load_zones`

**Type**: List of strings or dictionary
**Required**: Yes
**Example**: See below

Defines demand zones in the model. Can be simple list or nested structure.

**Simple** (one zone per region):

```yaml
load_zones: [CA_N, CA_S, AZ, NM]
```

**Complex** (multiple zones per region):

```yaml
load_zones:
  CA_N: [CA_N_URBAN, CA_N_RURAL]
  CA_S: [CA_S_URBAN, CA_S_RURAL]
  AZ: [AZ]
```

Load zones may or may not align with `model_regions` depending on model granularity.

<!-- regional_load_fn parameter removed. Use demand_table in data table configuration instead. -->

<!-- regional_load_includes_demand_response also removed with regional_load_fn. -->

<!-- Load growth parameters (growth_scenario, default_growth_rate, alt_growth_rate, load_growth_table, base_load_year, load_multiplier) removed. Load growth should be applied externally to input data tables before loading into PowerGenome. -->

<!-- Load scaling parameters (base_load_year, load_multiplier) removed. Scale load externally before loading. -->

## Distributed Generation

<!-- distributed_gen_method and distributed_gen_values removed. Use dg_capacity_table and dg_profiles_table instead for all DG configuration. -->

### `dg_capacity_table`

**Type**: String or dictionary
**Required**: No (modern alternative to `distributed_gen_values`)
**Example**: See below

Table with DG capacity by region/year.

```yaml
dg_capacity_table:
  table_name: dg_capacity.parquet
  scenario: high_adoption
```

**Expected columns**:

- `region`: Load zone
- `year`: Model year
- `capacity_mw`: DG capacity

### `dg_profiles_table`

**Type**: String or dictionary
**Required**: For DG subtraction
**Example**: `"dg_profiles.parquet"`

Hourly generation profiles for distributed generation.

```yaml
dg_profiles_table: dg_generation_profiles.parquet
```

**Expected format**:

- `region`: Load zone
- `hour`: Hour of year (1-8760)
- `cf`: Capacity factor (0-1)

### `avg_distribution_loss`

**Type**: Float (0-1)
**Required**: For DG subtraction
**Default**: `0.0`
**Example**: `0.06`

Average distribution system losses. Used when subtracting DG from load.

```yaml
avg_distribution_loss: 0.06  # 6% distribution loss
```

DG reduces net load at transmission level:

```text
net_load = gross_load - (dg_generation / (1 - dist_loss))
```

<!-- Demand response parameters (flexible_demand_resources, demand_response_fn, demand_response_scenario) removed. Demand response modeling is not currently supported. -->

## Electrification Scenarios

### `ev_load_profile_fn`

**Type**: String or dictionary
**Required**: For EV load addition
**Example**: See below

Electric vehicle charging load profiles.

**Simple**:

```yaml
ev_load_profile_fn: ev_charging_profiles.csv
```

**Year-specific**:

```yaml
ev_load_profile_fn:
  2030: ev_load_2030.csv
  2040: ev_load_2040.csv
```

Profiles are added to base load from demand_table.

### `electrification_stock_fn`

**Type**: String
**Required**: No
**Example**: `"electrification_stocks.csv"`

Electrification technology adoption (heat pumps, EVs, etc.).

```yaml
electrification_stock_fn: electrification/adoption_projections.csv
```

Used to calculate additional load from building/transport electrification.

## Load Modifications

### `reduce_time_domain`

**Type**: Boolean
**Required**: No
**Default**: `false`
**Example**: `true`

Whether to reduce 8760 hours to representative periods (see [Time Reduction](time-reduction.md)).

```yaml
reduce_time_domain: true
```

If `true`, demand profiles are clustered with generation profiles to select representative hours.

### `demand_weight_factor`

**Type**: Float
**Required**: No (used if `reduce_time_domain: true`)
**Default**: `1.0`
**Example**: `5.0`

Weight applied to demand when clustering hours. Higher values ensure peak demand is captured.

```yaml
reduce_time_domain: true
demand_weight_factor: 5  # Weight demand 5× more than generation
```

## Example Configuration

Complete demand configuration:

```yaml
# Load zones
load_zones: [CA_N, CA_S, AZ, NM]

# Distributed generation (modern approach via tables)
dg_capacity_table: dg_capacity.parquet
dg_profiles_table: dg_generation_profiles.parquet
avg_distribution_loss: 0.06

# Time reduction
reduce_time_domain: true
demand_weight_factor: 5
```

## Demand in GenX Outputs

PowerGenome generates **Load_data.csv** with:

- Columns for each time period
- Rows for each load zone
- Optional demand response resources (if configured)

The time dimension depends on `reduce_time_domain`:

- `false`: 8760 columns (hourly)
- `true`: Reduced to representative periods (e.g., 168 hours for 7 days)

## Related Settings

- [Time Reduction](time-reduction.md): Representative period selection
- [Regions](regions.md): Load zones and regional mappings
- [Model Definition](model-definition.md): Planning years for growth projections
- [Distributed Generation](../explanation/distributed-generation.md): DG methodology details
