# Transmission Settings

These parameters define inter-regional transmission network topology, capacity limits, expansion costs, and line losses.

## Network Topology

### `transmission_constraints`

**Type**: Dictionary (line → parameters)
**Required**: For inter-regional transmission
**Example**: See below

Defines transmission lines between regions with capacity limits and characteristics.

```yaml
transmission_constraints:
  CA_N_to_CA_S:
    transmission_path_name: CA_N_to_CA_S
    start_region: CA_N
    dest_region: CA_S
    transmission_line_mw: 5000  # Existing capacity
    max_transmission_mw: 8000   # Maximum with expansion
  CA_S_to_AZ:
    transmission_path_name: CA_S_to_AZ
    start_region: CA_S
    dest_region: AZ
    transmission_line_mw: 2000
    max_transmission_mw: 3500
```

**Required fields**:

- `transmission_path_name`: Unique identifier
- `start_region`: Origin region
- `dest_region`: Destination region
- `transmission_line_mw`: Existing transfer capacity (MW)

**Optional fields**:

- `max_transmission_mw`: Maximum capacity with expansion
- `distance_miles`: Line length (for cost calculations)

### `model_regions_gdf`

**Type**: String (path to GeoJSON)
**Required**: For geographic calculations
**Example**: `"data/ipm_regions_simple.geojson"`

GeoDataFrame with region geometries for distance calculations.

```yaml
model_regions_gdf: data/model_regions.geojson
```

Used to automatically calculate transmission line distances if not specified in `transmission_constraints`.

## Line Losses

### `tx_line_loss_pct`

**Type**: Float or dictionary
**Required**: No
**Default**: 0.0
**Example**: See below

Transmission line loss as percentage of transmitted energy.

**Global**:

```yaml
tx_line_loss_pct: 0.02  # 2% loss on all lines
```

**Line-specific**:

```yaml
tx_line_loss_pct:
  CA_N_to_CA_S: 0.015  # 1.5% loss
  CA_S_to_AZ: 0.025    # 2.5% loss (longer line)
```

Losses reduce delivered power: `delivered_mw = sent_mw × (1 - loss_pct)`.

### `distribution_loss_pct`

**Type**: Float
**Required**: No
**Default**: 0.0
**Example**: `0.05`

Distribution-level losses (load bus to end user).

```yaml
distribution_loss_pct: 0.05  # 5% distribution loss
```

Applied to demand values to account for losses between transmission and final consumption.

## Expansion Costs

### `tx_expansion_per_mw`

**Type**: Float or dictionary
**Required**: For transmission expansion
**Example**: See below

Cost to expand transmission capacity ($/MW of new capacity).

**Global**:

```yaml
tx_expansion_per_mw: 1000  # $/MW
```

**Line-specific**:

```yaml
tx_expansion_per_mw:
  CA_N_to_CA_S: 800   # Shorter line
  CA_S_to_AZ: 1500    # Longer, more difficult terrain
```

Total expansion cost = `(new_capacity - existing_capacity) × cost_per_mw`.

### `tx_line_capex_mw_mile`

**Type**: Float
**Required**: No (alternative to `tx_expansion_per_mw`)
**Example**: `2000`

Transmission capex per MW-mile (accounts for line length).

```yaml
tx_line_capex_mw_mile: 2000  # $/MW-mile
```

Expansion cost = `new_capacity × distance_miles × capex_mw_mile`.

More accurate than flat `tx_expansion_per_mw` if line lengths vary significantly.

### `tx_fixed_om_mw_mile`

**Type**: Float
**Required**: No
**Example**: `25`

Annual fixed O&M cost for transmission ($/MW-mile-year).

```yaml
tx_fixed_om_mw_mile: 25  # $/MW-mile-yr
```

Applied to existing and new transmission capacity.

## Capacity Limits

### `transmission_line_mw`

**Type**: Specified in `transmission_constraints`
**Required**: Yes (per line)
**Example**: See `transmission_constraints` above

Existing transfer capacity for each line (MW). This is the baseline before any expansion.

### `max_transmission_mw`

**Type**: Specified in `transmission_constraints`
**Required**: No (per line)
**Example**: See `transmission_constraints` above

Maximum allowable transmission capacity including expansion (MW).

If omitted, unlimited expansion is allowed (subject to costs).

### `enforce_constraints`

**Type**: Boolean
**Required**: No
**Default**: `true`
**Example**: `false`

Whether to enforce transmission capacity constraints in the model.

```yaml
enforce_constraints: false  # Copper plate (no transmission limits)
```

Setting to `false` creates a "copper plate" model where regions can trade unlimited power.

## Network Data Tables

### `transmission_table`

**Type**: String or dictionary
**Required**: Alternative to `transmission_constraints`
**Example**: See below

Load transmission network from data table instead of settings file.

**Simple**:

```yaml
transmission_table: transmission_network.csv
```

**Advanced**:

```yaml
transmission_table:
  table_name: transmission.parquet
  scenario: high_expansion
  filters:
    - - [year, '=', 2030]
```

**Expected columns**:

- `transmission_path_name`: Line ID
- `start_region`: Origin
- `dest_region`: Destination
- `transmission_line_mw`: Existing capacity
- `max_transmission_mw`: Maximum capacity (optional)
- `distance_miles`: Line length (optional)

This approach is cleaner for large networks (dozens of lines).

## Bidirectional Flow

### `allow_reversed_flow`

**Type**: Boolean
**Required**: No
**Default**: `true`
**Example**: `false`

Whether transmission lines allow bidirectional power flow.

```yaml
allow_reversed_flow: true
```

If `true`, power can flow either direction on each line (most realistic). If `false`, flow is unidirectional (start_region → dest_region only).

### `create_reversed_lines`

**Type**: Boolean
**Required**: No
**Default**: `false`
**Example**: `true`

Automatically create reversed transmission lines (dest → start) for each defined line.

```yaml
create_reversed_lines: true
```

Alternative to `allow_reversed_flow` that creates explicit bidirectional lines. Useful if costs/losses differ by direction.

## HVDC Lines

### `hvdc_transmission_lines`

**Type**: List of line names
**Required**: No
**Example**: `["CA_N_to_DESERT_HVDC"]`

Transmission lines that are HVDC (high voltage direct current) rather than AC.

```yaml
hvdc_transmission_lines:
  - CA_N_to_DESERT_HVDC
  - OFFSHORE_to_CA_S_HVDC
```

HVDC lines may have:

- Different loss characteristics
- Higher capex per MW-mile
- Converter station costs

### `hvdc_capex_mw_mile`

**Type**: Float
**Required**: For HVDC lines
**Example**: `3000`

HVDC line capex ($/MW-mile), typically higher than AC.

```yaml
hvdc_capex_mw_mile: 3000  # $/MW-mile
```

### `hvdc_converter_capex_mw`

**Type**: Float
**Required**: For HVDC lines
**Example**: `200000`

HVDC converter station cost ($/MW).

```yaml
hvdc_converter_capex_mw: 200000  # $/MW
```

Each HVDC line requires two converter stations (AC→DC and DC→AC).

## Regional Interconnection

### `region_interconnections`

**Type**: Dictionary (region → interconnection)
**Required**: No
**Example**: See below

Assigns regions to electricity interconnections (for synchronous grid modeling).

```yaml
region_interconnections:
  CA_N: Western
  CA_S: Western
  AZ: Western
  TX_N: ERCOT
  TX_S: ERCOT
  FL: Eastern
```

Used to:

- Limit transmission between asynchronous grids
- Apply interconnection-specific constraints
- Model seams between grid operators

### `interconnection_transmission_limit`

**Type**: Float
**Required**: No
**Example**: `500`

Maximum power transfer between interconnections (MW).

```yaml
interconnection_transmission_limit: 500
```

Models limited DC ties between asynchronous grids (e.g., ERCOT ↔ Western).

## Hurdle Rates

### `hurdle_rate_per_mwh`

**Type**: Float or dictionary
**Required**: No
**Example**: See below

Transaction cost for power transfers ($/MWh), representing wheeling charges or market friction.

**Global**:

```yaml
hurdle_rate_per_mwh: 5  # $5/MWh transaction cost
```

**Line-specific**:

```yaml
hurdle_rate_per_mwh:
  CA_N_to_CA_S: 2
  CA_S_to_AZ: 8  # Higher wheeling charge
```

Discourages unrealistic arbitrage in models with simplified market representation.

## Network Reinforcement

### `max_network_reinforcement_mw`

**Type**: Float or dictionary
**Required**: No
**Example**: See below

Annual limit on network expansion (MW/year).

**Global**:

```yaml
max_network_reinforcement_mw: 1000  # Max 1 GW expansion per year
```

**Line-specific**:

```yaml
max_network_reinforcement_mw:
  CA_N_to_CA_S: 500
  CA_S_to_AZ: 200
```

Models construction bottlenecks and permitting constraints.

## Example Configuration

Complete transmission configuration:

```yaml
# Network topology
transmission_constraints:
  CA_N_to_CA_S:
    transmission_path_name: CA_N_to_CA_S
    start_region: CA_N
    dest_region: CA_S
    transmission_line_mw: 5000
    max_transmission_mw: 8000
    distance_miles: 350

  CA_S_to_AZ:
    transmission_path_name: CA_S_to_AZ
    start_region: CA_S
    dest_region: AZ
    transmission_line_mw: 2000
    max_transmission_mw: 3500
    distance_miles: 280

# Costs and losses
tx_line_loss_pct: 0.02
tx_line_capex_mw_mile: 2000
tx_fixed_om_mw_mile: 25

# Flow characteristics
allow_reversed_flow: true
enforce_constraints: true

# Interconnections
region_interconnections:
  CA_N: Western
  CA_S: Western
  AZ: Western

# Expansion limits
max_network_reinforcement_mw:
  CA_N_to_CA_S: 500
  CA_S_to_AZ: 200
```

## Transmission in GenX Outputs

PowerGenome generates these GenX transmission files:

**Network.csv**:

- Line topology (start → end regions)
- Existing capacity (`Transmission_Line_MW`)
- Maximum capacity (`Max_Transmission_MW`)
- Loss percentage
- Distance (miles)

**Network_expansion.csv**:

- Expansion costs (`Inv_Cost_per_MWyr`)
- O&M costs (`Fixed_OM_per_MWyr`)
- Maximum reinforcement limits

See [GenX documentation](https://genxproject.github.io/GenX/) for how these files are used in optimization.

## Related Settings

- [Regions](regions.md): Defines `model_regions` used in network
- [Model Definition](model-definition.md): `target_usd_year` for cost conversions
- [Demand](demand.md): Load zones may differ from transmission zones
