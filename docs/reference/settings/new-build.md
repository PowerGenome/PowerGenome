# New-Build Resources Settings

These parameters control technology costs, new resource definitions, regional availability, and capacity constraints for candidate generators.

## Technology Cost Sources

### `new_resources`

**Type**: List of lists
**Required**: For ATB-based technologies
**Example**: See below

List of technologies from NREL Annual Technology Baseline (ATB) to include as new-build candidates. Each item is a list: `[technology, tech_detail, cost_case, size_mw]`.

```yaml
new_resources:
  - [NaturalGas, CCCCSAvgCF, Moderate, 500]
  - [NaturalGas, CTAvgCF, Moderate, 100]
  - [Battery, "*", Moderate, 100]
  - [UtilityPV, Class1, Moderate, 100]
  - [LandbasedWind, Class3, Moderate, 100]
  - [OffshoreWind, Class3, Moderate, 100]
  - [Nuclear, Nuclear - Large, Moderate, 1000]
```

**Format**: `[technology, tech_detail, cost_case, size_mw]`

- `technology`: Technology type (e.g., `NaturalGas`, `Battery`, `UtilityPV`)
- `tech_detail`: Specific variant (e.g., `CCAvgCF`, `Class1`, `*` for wildcard)
- `cost_case`: Cost trajectory (`Conservative`, `Moderate`, `Advanced`)
- `size_mw`: Unit size in MW

### `resource_data_year`year`

**Type**: Integer
**Required**: No
**Default**: Latest available
**Example**: `2023`

Which year's ATB data to use for technology costs.

```yaml
resource_data_year: 2023
```

ATB is updated annually with revised cost projections. Specify the data vintage for reproducibility.

### `resource_financial_case`case`

**Type**: String
**Required**: No
**Default**: `"Market"`
**Example**: `"Market"`

ATB financial assumptions case. Options:

- `Market`: Market-based financing
- `R&D`: Optimistic R&D case with policy support

```yaml
resource_financial_case: Market
```

### `resource_cap_recovery_years`

**Type**: Integer
**Required**: No
**Default**: `20`
**Example**: `20`

Default capital recovery period (economic lifetime) for new generators, used to calculate annualized capital costs.

```yaml
resource_cap_recovery_years: 20
```

### `alt_resource_cap_recovery_years`

**Type**: Dictionary (technology → years)
**Required**: No
**Example**: See below

Override capital recovery period for specific technologies (matched using string contains).

```yaml
alt_resource_cap_recovery_years:
  NaturalGas_CCAvg: 15
  NaturalGas_CTAvg: 15
  Battery: 15
  Nuclear: 40
```

## Cost and Performance Adjustments

### `resource_modifiers`

**Type**: Dictionary (tech_name → modifications)
**Required**: No
**Example**: See below

Modify ATB technology parameters. Each entry specifies technology/tech_detail matching and parameter changes.

```yaml
resource_modifiers:
  batteries:
    technology: Battery
    tech_detail: "*"
    Var_OM_Cost_per_MWh: [add, 0.15]
    Var_OM_Cost_per_MWh_In: 0.15
    size_mw: 100
```

Each entry creates a candidate resource with the specified unit size.

### `modified_new_resources`

**Type**: Dictionary (new_name → base + modifications)
**Required**: No
**Example**: See below

Create new technology variants by copying and modifying existing ATB technologies.

```yaml
modified_new_resources:
  NGCCS100:
    technology: NaturalGas
    tech_detail: CCCCSAvgCF
    cost_case: Conservative
    size_mw: 500
    new_technology: NaturalGas
    new_tech_detail: CCS100
    new_cost_case: Moderate
    capex_mw: [add, 116000]
    heat_rate: [add, 0.365]
    fixed_o_m_mw: [add, 9670]
    variable_o_m_mwh: [mul, 1.076]

  hydrogen_turbine:
    technology: NaturalGas
    tech_detail: 1-on-1 Combined Cycle (H-Frame)
    cost_case: Moderate
    size_mw: 100
    new_technology: hydrogen
    new_tech_detail: 1-on-1 Combined Cycle (H-Frame)
    new_cost_case: Moderate
```

**Base parameters** (copy from existing ATB technology):

- `technology`: ATB technology to copy from
- `tech_detail`: ATB tech detail to copy from
- `cost_case`: ATB cost case to copy from
- `size_mw`: Unit size

**New parameters** (create new technology with this name):

- `new_technology`: New technology name
- `new_tech_detail`: New tech detail name
- `new_cost_case`: New cost case name

**Modification operators**:

- `[add, value]`: Add to base value
- `[mul, value]`: Multiply base value
- `[sub, value]`: Subtract from base value
- `[truediv, value]`: Divide base value
- Direct value: Override base value

### `resource_modifiers`

**Type**: Dictionary (technology → parameter → values)
**Required**: No
**Example**: See below

Modify technology costs in-place (applies to all instances of that technology).

```yaml
resource_modifiers:
  UtilityPV_Class1_Moderate:
    capex_mw:
      2030: 0.9  # 10% capex reduction in 2030
      2040: 0.8  # 20% reduction in 2040
    fixed_o_m_mw:
      2030: 0.95
      2040: 0.90
```

Multipliers are applied by model year. Use this for across-the-board adjustments.

## Regional Availability

### `new_gen_not_available`

**Type**: Dictionary (region → list of technologies)
**Required**: No
**Example**: See below

Exclude specific new-build technologies from certain regions.

```yaml
new_gen_not_available:
  AZ:
    - OffshoreWind_Class1_Moderate_-1
    - Geothermal_HydroFlash_Moderate
  CA_N:
    - Coal_new_Moderate
```

Use cases:

- Physical constraints (no offshore wind in landlocked states)
- Policy constraints (coal banned)
- Resource availability (no geothermal potential)

## Capacity Limits

### `new_build_max_capacity`

**Type**: Dictionary (region → technology → MW)
**Required**: No
**Example**: See below

Maximum new capacity allowed for each technology in each region.

```yaml
new_build_max_capacity:
  CA_N:
    Nuclear_Nuclear_Moderate: 2000  # Max 2 GW new nuclear
    OffshoreWind_Class1_Moderate: 5000
  AZ:
    UtilityPV_Class1_Moderate: 10000
```

Enforces siting constraints, policy limits, or resource potential.

### `new_build_min_capacity`

**Type**: Dictionary (region → technology → MW)
**Required**: No
**Example**: See below

Minimum required new capacity (forced build).

```yaml
new_build_min_capacity:
  CA_S:
    Battery_4Hr_Moderate: 500  # Must build at least 500 MW storage
```

Implements policy mandates or planned projects.

## Technology Parameters

### `new_gen_unit_size`

**Type**: Dictionary (technology → MW)
**Required**: No
**Example**: See below

Override default unit sizes from ATB.

```yaml
new_gen_unit_size:
  NaturalGas_CCAvgCF_Moderate: 500
  Nuclear_Nuclear_Moderate: 1000
  Battery_4Hr_Moderate: 100
```

Unit size affects:

- Integer constraints (if enabled)
- Investment granularity
- Minimum build increments

### `min_power`

**Type**: Dictionary (technology → fraction)
**Required**: No
**Default**: From source data
**Example**: See below

Minimum stable operating level (fraction of capacity).

```yaml
min_power:
  NaturalGas_CCAvgCF_Moderate: 0.4  # 40% minimum load
  Coal_new_Moderate: 0.5
```

Lower values = more operational flexibility.

## Storage-Specific Settings

### `battery_energy_to_power`

**Type**: Dictionary (technology → hours)
**Required**: For batteries without explicit duration
**Example**: See below

Energy-to-power ratio for battery storage (hours of discharge).

```yaml
battery_energy_to_power:
  Battery_2Hr_Moderate: 2
  Battery_4Hr_Moderate: 4
  Battery_8Hr_Moderate: 8
```

Modern ATB includes this in technology name, but this parameter overrides it.

### `storage_efficiency`

**Type**: Dictionary (technology → efficiency)
**Required**: No
**Default**: From ATB
**Example**: See below

Round-trip efficiency for storage technologies.

```yaml
storage_efficiency:
  Battery_4Hr_Moderate: 0.85  # 85% round-trip
  Pumped_Hydro: 0.80
```

## Interconnection Costs

### `interconnect_capex_mw`

**Type**: Number or dictionary
**Required**: No
**Default**: None (falls back to legacy `transmission_investment_cost` if provided)
**Example**: See patterns below

Flexible specification of per-MW interconnection capital cost (USD/MW) applied to new and existing resources. Replaces the deprecated spur line mileage system (`transmission_investment_cost` with `capacity_limit_spur_fn` file).

**Important**: All cost values must be in the same dollar year as `target_usd_year` (the target dollar year for cost normalization). PowerGenome does not automatically adjust interconnection costs for inflation.

**Supported Patterns** (mutually exclusive; `default` key may accompany any pattern):

#### 1. Scalar

Apply uniform cost to all resources:

```yaml
interconnect_capex_mw: 150000  # $150k/MW for all resources
```

#### 2. Region-Only

Different costs by region (exact region name matching):

```yaml
interconnect_capex_mw:
  default: 120000
  CA_N: 140000
  CA_S: 130000
  AZ: 125000
```

#### 3. Technology-Only

Different costs by technology using case-insensitive substring matching with shortest-first precedence:

```yaml
interconnect_capex_mw:
  default: 100000
  wind: 120000          # matches 'LandbasedWind_Class3_Moderate', 'OffshoreWind_Class1_Moderate'
  offshore_wind: 200000 # longer substring overwrites previous 'wind' assignment
  battery: 50000        # matches 'Battery_4Hr_Moderate', 'Battery_*_Moderate'
  solar: 110000         # matches 'UtilityPV_Class1_Moderate'
```

**Precedence rule**: Shorter substrings applied first, then longer (more specific) substrings override previous assignments. This allows general categories with specific exceptions.

#### 4. Region → Technology Nested

Region-specific technology costs (region keys at top level):

```yaml
interconnect_capex_mw:
  default: 110000
  CA_N:
    battery: 60000
    offshore_wind: 210000
    solar: 105000
  CA_S:
    solar: 95000
    wind: 115000
  AZ: 125000  # Can mix dict and numeric values for different regions
```

Within each region, technology substrings follow shortest-first precedence.

#### 5. Technology → Region Nested

Technology-specific regional costs (technology keys at top level):

```yaml
interconnect_capex_mw:
  default: 110000
  wind:
    CA_N: 125000
    CA_S: 115000
    AZ: 120000
  battery:
    CA_N: 55000
    CA_S: 52000
  offshore_wind:
    CA_N: 205000  # More specific than 'wind', overrides for offshore
```

Technology substrings applied shortest-first; within each technology, regions matched exactly.

**Matching Rules**:

- **Technology matching**: Case-insensitive substring search against the `technology` column
- **Substring precedence**: Shortest substrings processed first; longer substrings override
- **Region matching**: Exact match against model region names (case-sensitive)
- **Invalid mixing**: Cannot mix region and technology keys at the same top level (excluding `default`)

**Examples of Invalid Configuration**:

```yaml
# ERROR: Cannot mix region and technology keys
interconnect_capex_mw:
  CA_N: 140000      # Region key
  wind: 120000      # Technology key - mixing not allowed!
```

**Behavior**:

- **Non-destructive**: Existing non-zero `interconnect_capex_mw` values in the resource dataframe are never overwritten
- **Automatic annuity**: `interconnect_annuity` is computed automatically for:
  - Newly assigned capex rows
  - Pre-existing capex rows with zero/blank annuity
- **Financial parameters**: Annuity calculation uses plant-specific `wacc_real` and `cap_recovery_years` from the resource dataframe
- **Legacy bypass**: When `interconnect_capex_mw` is provided, legacy spur mileage columns (`spur_miles`, `offshore_spur_miles`, `tx_miles`) are ignored (warning logged if present)

**Migration from Legacy System**:

Old approach (deprecated):

```yaml
# Settings
transmission_investment_cost:
  spur:
    capex_mw_mile:
      CA_N: 3000
      CA_S: 3200
    wacc: 0.069
    investment_years: 60

# Required extra_inputs file: capacity_limit_spur_fn
# File: resource_capacity_spur.csv
# resource,region,spur_miles
# offshore_wind_fixed,CA_N,45
# solar_pv,CA_N,12
```

New approach (recommended):

```yaml
interconnect_capex_mw:
  default: 120000
  offshore_wind: 200000  # Direct cost specification
  solar: 105000
```

**Benefits of new system**:

- No external file dependency
- More intuitive (direct $/MW vs. mileage calculation)
- Flexible pattern matching (region, technology, or nested)
- Automatic annuity handling
- Better error messages

### `transmission_investment_cost`

**Type**: Dictionary
**Required**: No
**Deprecated**: Use `interconnect_capex_mw` instead
**Example**: See legacy documentation

Legacy spur line mileage-based interconnection cost system. Contains nested dictionaries for `spur`, `offshore_spur`, and `tx` with keys:

- `capex_mw_mile`: Cost per MW-mile (by region)
- `wacc`: Weighted average cost of capital
- `investment_years`: Capital recovery period

**Deprecation notice**: This parameter is deprecated and will be removed in a future release. It is only used when `interconnect_capex_mw` is not provided, and emits warnings during execution. Migrate to `interconnect_capex_mw` for simplified, more flexible cost specification.

## Renewable Resource Groups

### `renewable_clusters`

**Type**: List of dictionaries
**Required**: For renewable resources with location-specific profiles
**Example**: See below

Define renewable resource sites with generation profiles and capacity limits.

```yaml
renewable_clusters:
  - region: CA_N
    technology: LandbasedWind
    cluster: 1
    max_capacity: 2000
    profile_id: CA_N_wind_class3_profile
  - region: CA_N
    technology: UtilityPV
    cluster: 1
    max_capacity: 5000
    profile_id: CA_N_solar_class1_profile
```

Each entry represents a specific site with:

- Geographic location (`region`)
- Resource type (`technology`)
- Unique identifier (`cluster`)
- Capacity potential (`max_capacity`)
- Hourly generation profile (`profile_id`)

See [Renewable Resources Tutorial](../../tutorials/renewable-resources.md) for detailed workflow.

### `RESOURCE_GROUP_PROFILES`

**Type**: String (path)
**Required**: For renewable clusters
**Example**: `"/data/generation_profiles"`

Directory containing hourly generation profiles for renewable resource groups. This is typically set as an environment variable or path constant.

```yaml
RESOURCE_GROUP_PROFILES: /data/nrel_profiles
```

Files should be named: `{technology}_{tech_detail}_{cluster_id}.csv` with hourly capacity factors.

## Additional Technologies

### `additional_technologies_fn`

**Type**: String or dictionary (year → filename)
**Required**: No
**Example**: See below

CSV file with user-defined technologies not in ATB.

**Single file**:

```yaml
additional_technologies_fn: custom_tech.csv
```

**Year-specific**:

```yaml
additional_technologies_fn:
  2030: additional_tech_2030.csv
  2040: additional_tech_2040.csv
```

File columns:

- `technology`: Technology name
- `tech_detail`: Technology detail
- `cost_case`: Cost case
- `planning_year`: Model year
- `capex_mw`: Capital cost ($/MW)
- `fixed_o_m_mw`: Fixed O&M ($/MW-yr)
- `variable_o_m_mwh`: Variable O&M ($/MWh)
- `heat_rate_mmbtu_mwh`: Heat rate (if thermal)
- Plus other technology parameters

See `data/additional_technologies/` for examples.

## Cost Multipliers

Cost multipliers adjust technology capital costs by region (e.g., labor costs, permitting difficulty). PowerGenome now uses a **table-driven approach** via the `regional_cost_factor` data table.

### `cost_multiplier_fn`

**Type**: String (path) or table configuration
**Required**: No (can use `regional_cost_factor` table instead)
**Example**: `"regional_cost_multipliers.csv"`

Legacy parameter for regional capital cost multipliers. **Prefer using `regional_cost_factor` in data table configuration.**

```yaml
cost_multiplier_fn: cost_multipliers/regional_multipliers.csv
```

**File format**:

- `region`: Region name
- `technology`: Technology name
- `multiplier`: Cost multiplier (1.2 = 20% increase)

**Modern approach** (recommended):

```yaml
# In data table configuration
regional_cost_factor:
  table_name: regional_cost_factors.parquet
  scenario: baseline
```

### `cost_multiplier_region_map`

**Type**: Dictionary (region → cost region)
**Required**: No (automatically generated if omitted)
**Example**: See below

Map model regions to cost multiplier region names. **Optional**: If not provided, PowerGenome automatically creates a mapping using exact region name matches.

```yaml
cost_multiplier_region_map:
  CA_N: California_North
  CA_S: California_South
```

**Auto-generation**: The `auto_create_region_map()` function in `new_build.py` automatically generates this mapping by matching model region names against available cost factor regions.

### `cost_multiplier_technology_map`

**Type**: Dictionary (technology → cost tech name)
**Required**: No (automatically generated if omitted)
**Example**: See below

Map model technologies to cost multiplier technology names. **Optional**: If not provided, PowerGenome automatically creates a mapping using substring matching.

```yaml
cost_multiplier_technology_map:
  NaturalGas_CCAvgCF_Moderate: NaturalGas_CC
  UtilityPV_Class1_Moderate: Solar_PV
```

**Auto-generation**: The `auto_create_technology_map()` function automatically maps technologies by finding the best substring match between model technology names and cost factor technology names.

## Example Configuration

Complete new-build configuration:

```yaml
# New-build technologies
resource_data_year: 2023
resource_financial_case: Market
resource_cap_recovery_years: 20

new_resources:
  - [NaturalGas, CCCCSAvgCF, Conservative, 500]
  - [NaturalGas, CTAvgCF, Moderate, 100]
  - [Battery, "*", Moderate, 100]
  - [UtilityPV, Class1, Moderate, 100]
  - [LandbasedWind, Class3, Moderate, 100]
  - [Nuclear, Nuclear - Large, Moderate, 1000]

# Technology modifications
resource_modifiers:
  batteries:
    technology: Battery
    tech_detail: "*"
    Var_OM_Cost_per_MWh: [add, 0.15]

# Regional restrictions
new_gen_not_available:
  AZ:
    - OffshoreWind_Class1_Moderate
  CA_N:
    - Coal_newAvgCF_Moderate

# Capacity limits
new_build_max_capacity:
  CA_N:
    Nuclear_Nuclear_Moderate: 2000
    UtilityPV_Class1_Moderate: 10000

# Renewable sites
renewable_clusters:
  - region: CA_N
    technology: LandbasedWind
    cluster: 1
    max_capacity: 2000
    profile_id: CA_N_wind_class3

RESOURCE_GROUP_PROFILES: /data/nrel_profiles
```

## Related Settings

- [Model Definition](model-definition.md): `model_year` affects cost projections
- [Regions](regions.md): Regional availability and cost multipliers
- [Fuels](fuels.md): Fuel costs for thermal technologies
- [Resource Tags](resource-tags.md): Tag assignment for new resources
