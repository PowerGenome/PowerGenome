# Transmission Settings

These parameters define inter-regional transmission network expansion costs, line losses, and reference data tables containing network topology.

!!! important "Network Topology from Data Tables"
    Unlike generators and demand, transmission network structure (which regions connect, capacities, distances) is **not** defined in settings YAML. Instead, PowerGenome loads this from data tables configured via `transmission_constraints_table` and `transmission_cost_table`.

## Data Table References

### `transmission_constraints_table`

**Type**: String or dictionary
**Required**: For database-based transmission data
**Example**: `"trancap_init_energy_wecc_pg.csv"`

Table containing transmission capacity between base regions.

**Simple**:

```yaml
transmission_constraints_table: transmission_capacity.csv
```

**Advanced**:

```yaml
transmission_constraints_table:
  table_name: transmission.parquet
  scenario: high_capacity
  filters:
    - - [year, '=', 2030]
```

**Expected columns**:

- `region_from`: Origin base region
- `region_to`: Destination base region
- `firm_ttc_mw`: Firm transmission capacity (MW)
- `nonfirm_ttc_mw`: Non-firm capacity (optional)

PowerGenome aggregates these constraints across base regions according to `region_aggregations`.

### `transmission_cost_table`

**Type**: String or dictionary
**Required**: For pre-calculated transmission costs
**Example**: `"transmission_distance_cost_500kVac_annuity.csv"`

Table with transmission line costs and losses calculated from least-cost-path analysis.

```yaml
transmission_cost_table: network_costs.csv
```

**Expected columns**:

- `start_region`: Origin model region
- `dest_region`: Destination model region
- `total_interconnect_annuity_mw`: Annualized cost ($/MW-yr)
- `total_interconnect_cost_mw`: Total capital cost ($/MW)
- `total_line_loss_frac`: Line loss as fraction (0.05 = 5%)
- `dollar_year`: Cost basis year

### `user_transmission_costs`

**Type**: String (filename in `input_folder`)
**Required**: For user-specified transmission costs
**Example**: `"network_costs_wecc_6_zone.csv"`

CSV file with transmission costs between model regions. Overrides database values.

```yaml
user_transmission_costs: network_costs.csv
```

**File format** (`extra_inputs/network_costs.csv`):

```csv
start_region,dest_region,total_interconnect_annuity_mw,total_interconnect_cost_mw,total_line_loss_frac,dollar_year
northeast,midwest,15000,350000,0.08,2018
midwest,south,12000,280000,0.05,2018
```

!!! tip "Use Pre-Calculated Costs"
    PowerGenome can calculate costs from region centroids, but **pre-calculated costs from least-cost-path analysis are strongly recommended**. See [Gagnon et al. (2023)](https://www.sciencedirect.com/science/article/abs/pii/S2666278723000144) for methodology.

## Capacity Selection

### `tx_value_col`

**Type**: String
**Required**: No
**Default**: `"firm_ttc_mw"`
**Options**: `"firm_ttc_mw"` or `"nonfirm_ttc_mw"`

Which capacity column to use from transmission data tables.

```yaml
tx_value_col: firm_ttc_mw
```

**Firm vs non-firm**:

- **Firm**: Guaranteed capacity (lower value, more conservative)
- **Non-firm**: Opportunistic capacity (higher value, less reliable)

PowerGenome v0.8+ defaults to `firm_ttc_mw`. Previous versions used `nonfirm_ttc_mw`.

### `distribution_loss_factor`

**Type**: Float
**Required**: For distributed generation
**Default**: `0.0`
**Example**: `0.06`

Distribution-level line loss (transmission substation to customer).

```yaml
distribution_loss_factor: 0.06  # 6% distribution loss
```

Used when subtracting distributed generation from total demand. Total demand includes distribution losses, so DG must be adjusted.

## Expansion Controls

### `tx_expansion_per_period`

**Type**: Float
**Required**: For transmission expansion
**Example**: `1.0`

Maximum fractional expansion of existing transmission per planning period.

```yaml
tx_expansion_per_period: 1.0  # Can double existing capacity
```

**Interpretation**:

- `1.0`: Can add 100% of existing capacity (500 MW → 1000 MW max)
- `0.5`: Can add 50% of existing (500 MW → 750 MW max)
- `2.0`: Can triple existing capacity

Combined with `tx_expansion_mw_per_period` — **larger value governs** each line.

### `tx_expansion_mw_per_period`

**Type**: Integer
**Required**: No
**Example**: `500`

Absolute expansion limit per line per period (MW).

```yaml
tx_expansion_mw_per_period: 500  # Can add up to 500 MW
```

Useful for setting minimum buildable increments (e.g., one 230kV line ≈ 200-400 MW).

**Expansion calculation per line**:

```python
max_expansion = max(
    existing_capacity * tx_expansion_per_period,
    tx_expansion_mw_per_period
)
```

## GenX Output Files

PowerGenome generates these GenX transmission files:

**`Network.csv`**:

- `Network_Lines`: Line ID number
- `z1`, `z2`, ...: Network topology matrix
- `Line_Max_Flow_MW`: Existing capacity
- `Line_Min_Flow_MW`: Reverse flow limit (typically negative of max)
- `transmission_path_name`: Human-readable name
- `Line_Loss_Percentage`: Loss fraction
- `Line_Reinforcement_Cost_per_MWyr`: Annualized expansion cost

GenX uses these files to optimize transmission expansion and inter-regional power flows.

## Related Settings

- [Regions](regions.md): Defines `model_regions` used in transmission
- [Model Definition](model-definition.md): `target_usd_year` for cost inflation adjustment
- [Scenario Management](scenario-management.md): Multi-period transmission expansion
