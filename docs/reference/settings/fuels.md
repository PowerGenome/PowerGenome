# Fuel Settings

These parameters control fuel price sources, emission factors, and regional fuel price mappings.

## Overview

PowerGenome supports two fuel pricing workflows:

1. **Simplified (Recommended)**: Provide a fuel price table with prices for all model regions and fuels. PowerGenome automatically constructs fuel names and calculates aggregated region prices.

2. **Legacy (Larger fuel regions)**: Use raw AEO data with mapping parameters to create regional fuel prices.

**Most users should use the simplified workflow** with a complete fuel price table containing all base regions.

## Simplified Workflow (Recommended)

### Required Settings

These settings select a supplied fuel table and map fuels to generators:

```yaml
# Assign scenario names to fuels
fuel_scenarios:
  coal: reference
  naturalgas: reference
  uranium: reference

# Map technologies to fuels
tech_fuel_map:
  Conventional Steam Coal: coal
  Natural Gas Fired Combined Cycle: naturalgas
  Nuclear: uranium

# Point to your fuel price table
fuel_price_table: fuel_prices.csv
fuel_data_year: 2025
```

You can select different scenarios for different fuels if those scenarios are
present in the table.

### Fuel Price Table Format

Your fuel price table must include prices for **every base region** in your model:

**Required columns**:

- `fuel`: Fuel name (coal, naturalgas, distillate, uranium, etc.)
- `region`: Base region name (must match regions in your model)
- `year`: Calendar year
- `price`: Price ($/MMBtu)
- `scenario`: Scenario identifier matching `fuel_scenarios`
- `data_year`: Data vintage year (include even for one vintage)

**Optional columns**:

- `dollar_year`: Price year (for inflation adjustment)

**Example fuel_prices.csv**:

```csv
fuel,region,year,price,scenario,data_year,dollar_year
coal,CA_N,2030,2.5,reference,2025,2024
coal,CA_S,2030,2.6,reference,2025,2024
naturalgas,CA_N,2030,4.2,reference,2025,2024
naturalgas,CA_S,2030,4.3,reference,2025,2024
uranium,CA_N,2030,0.8,reference,2025,2024
uranium,CA_S,2030,0.8,reference,2025,2024
```

**Important**: The table must contain prices for all base regions (e.g., CA_N, CA_S). PowerGenome will automatically average base region prices to create prices for aggregated regions defined in `region_aggregations`.

### How It Works

PowerGenome automatically:

1. **Constructs full fuel names** as `{region}_{scenario}_{fuel}` (e.g., `CA_N_reference_coal`)
2. **Averages prices for aggregated regions** (e.g., if CA = [CA_N, CA_S], then CA prices = average of CA_N and CA_S)
3. **Assigns fuels to generators** using `tech_fuel_map`

No mapping parameters (`fuel_series_*` or `fuel_region_map`) are needed.

## Legacy Workflow (AEO Direct)

If you're working directly with EIA AEO data, you'll need additional mapping parameters.

### `fuel_scenarios`

**Type**: Dictionary (fuel → scenario name)
**Required**: Yes
**Example**: See below

Maps fuels to price scenario names.

```yaml
fuel_scenarios:
  coal: reference
  naturalgas: reference
  uranium: reference
  distillate: high_resource
```

**Scenario names** must match keys in `fuel_series_scenario_names` (legacy workflow) or the `scenario` column in your fuel price table (simplified workflow).

### `fuel_data_year`

**Type**: Integer
**Required**: Only if your fuel price table has multiple data years
**Example**: `2025`

Specifies which data year to use if your fuel price table contains multiple vintages.

```yaml
fuel_data_year: 2025
```

Selects the vintage when the table contains multiple `data_year` values. Include
the column even for a single vintage; without `fuel_data_year`, the loader checks
the number of unique values in that column.

## Legacy Mapping Parameters (Optional)

!!! warning "Legacy Parameters"
    The parameters below are **only needed for the legacy AEO direct workflow**. If your fuel price table already contains prices for all base regions with the fuel/scenario names you want, **skip this entire section**.

These parameters map region, fuel, and scenario codes in a supplied fuel price table.

### `fuel_series_scenario_names`

**Type**: Dictionary (user scenario → AEO scenario code)
**Required**: No (legacy workflow only)
**Example**: See below

Maps user-friendly scenario names to official EIA AEO scenario codes.

```yaml
fuel_series_scenario_names:
  reference: REF2025
  no_111d: NOCAA111
  low_price: LOWPRICE
  high_price: HIGHPRICE
  high_resource: HIGHOGS
  low_resource: LOWOGS
```

**AEO scenario codes** change with each AEO release. Check [EIA OpenData](https://www.eia.gov/opendata/) for current codes.

### `fuel_series_names`

**Type**: Dictionary (fuel name → AEO fuel code)
**Required**: No (legacy workflow only)
**Example**: See below

Maps fuel names to EIA AEO fuel codes.

```yaml
fuel_series_names:
  coal: STC  # Steam coal
  naturalgas: NG
  distillate: DFO
  uranium: U
```

### `fuel_series_region_names`

**Type**: Dictionary (region name → AEO region code)
**Required**: No (legacy workflow only)
**Example**: See below

Maps region names to EIA AEO region codes.

```yaml
fuel_series_region_names:
  mountain: MTN
  pacific: PCF
  west_south_central: WSC
```

### `fuel_region_map`

**Type**: Dictionary (AEO region → list of model regions)
**Required**: No (legacy workflow only)
**Example**: See below

Maps EIA AEO fuel price regions to model regions. **Only used in legacy workflow** to create modified copies of base region prices.

```yaml
fuel_region_map:
  pacific: [CA_N, CA_S]
  mountain: [AZ, NM]
  west_south_central: [TX_N]
```

!!! note "Simplified Workflow Alternative"
    In the simplified workflow, you don't need `fuel_region_map`. Instead, provide prices for all base regions directly in your fuel price table. PowerGenome will automatically average base region prices for aggregated regions.

**AEO fuel regions** (common names):

- `pacific`: Pacific
- `mountain`: Mountain
- `west_south_central`: West South Central
- `east_north_central`: East North Central
- `west_north_central`: West North Central
- `new_england`: New England
- `middle_atlantic`: Middle Atlantic
- `south_atlantic`: South Atlantic
- `east_south_central`: East South Central

See EIA documentation for complete region definitions.

## Fuel Emission Factors

### `fuel_emission_factors`

**Type**: Dictionary (fuel → number)
**Required**: For emissions modeling
**Example**: See below

CO₂ emission factors by fuel type (tonnes CO₂ per MMBtu).

```yaml
fuel_emission_factors:
  coal: 0.09552
  naturalgas: 0.05306
  distillate: 0.07315
  uranium: 0.0
  biomass: 0.0
```

Values are based on fuel carbon content and combustion chemistry.

**Common factors** (tonnes CO₂/MMBtu):

- Bituminous coal: 0.0955
- Sub-bituminous coal: 0.0972
- Natural gas: 0.0531
- Diesel: 0.0732
- Biomass: 0.0 (policy-dependent)

### `fuel_price_table`

**Type**: String or dictionary
**Required**: For custom fuel prices
**Example**: See below

Custom fuel price time series (alternative to AEO).

**Simple**:

```yaml
fuel_price_table: fuel_prices.csv
fuel_data_year: 2025
```

**Advanced**:

```yaml
fuel_price_table:
  table_name: fuel_prices.parquet
  scenario: high_gas
  filters:
    - - [year, '>=', 2025]
```

**Expected columns**:

- `fuel`: Fuel name (coal, naturalgas, etc.)
- `region`: Region name
- `year`: Calendar year
- `price`: Price ($/MMBtu)

## User-Defined Fuel Prices

### `user_fuel_price`

**Type**: Dictionary (fuel → price or region → price)
**Required**: No
**Example**: See below

Supply prices for user-defined fuels. Values can be scalar across regions or
specified by model region; use fuels not already assigned in `fuel_scenarios`.

```yaml
user_fuel_price:
  hydrogen: 16.0  # $/MMBtu in every region
  biomass:
    CA_N: 5.0
    CA_S: 6.0
```

Use [year-keyed values](year-keyed-values.md) or scenario management for prices
that change by planning period. `user_fuel_usd_year` optionally maps fuel names
to the dollar year of each user-supplied price.

## Fuel Technology Mappings

### `tech_fuel_map`

**Type**: Dictionary (technology → fuel)
**Required**: For assigning fuels to generators
**Example**: See below

Maps technology-name substrings to fuels in `fuel_scenarios` or
`user_fuel_price`. Matching is case-insensitive.

```yaml
tech_fuel_map:
  NaturalGas: naturalgas
  Conventional Steam Coal: coal
  Nuclear: uranium
  Biomass: biomass
```

## Regional Fuel Prices

### `regional_fuel_adjustments`

**Type**: Dictionary (region → fuel → value or modifier)
**Required**: No
**Example**: See below

Regional price adjustments, either relative to base prices or as a relacement value.

```yaml
regional_fuel_adjustments:
  CA_N:
    naturalgas: [add, 1.5]  # Add $1.5/mmbtu to gas prices in CA_N
  CA_S: [mul, 0.9] # Multiply all fuel costs in CA_S by 0.9
  AZ:
    coal: 2.0  # $2/mmbtu fuel cost in AZ (dollar year must match target_usd_year)
```

Accounts for regional market conditions, transportation costs, or local policies.

## Carbon Pricing

### `carbon_tax`

**Type**: Float or dictionary (year → $/tonne)
**Required**: No
**Example**: See below

Carbon price ($/tonne CO₂) added to fuel costs.

**Single value**:

```yaml
carbon_tax: 50  # $/tonne CO₂
```

Carbon costs are calculated as:

```text
carbon_cost_per_mmbtu = carbon_tax * fuel_emission_factors[fuel]
```

## Example Configurations

### Simplified Workflow Example

Complete fuel pricing with custom fuel price table:

```yaml
# Fuel price table location
fuel_price_table: fuel_prices.csv

# Map scenario names to fuels
fuel_scenarios:
  coal: reference
  naturalgas: reference
  uranium: reference

# Map technologies to fuels
tech_fuel_map:
  Conventional Steam Coal: coal
  Natural Gas Fired Combined Cycle: naturalgas
  Natural Gas Fired Combustion Turbine: naturalgas
  Nuclear: uranium

# Emission factors (tonnes CO2 per MMBtu)
fuel_emission_factors:
  coal: 0.09552
  naturalgas: 0.05306
  distillate: 0.07315
  uranium: 0.0

# CCS fuel mapping
ccs_fuel_map:
  NaturalGas_CCS90: naturalgas_ccs90
  Coal_CCS90: coal_ccs90

ccs_capture_rate:
  naturalgas_ccs90: 0.9
  coal_ccs90: 0.9

ccs_disposal_cost: 10  # $/tonne CO2
```

Your `fuel_prices.csv` must contain all base regions:

```csv
fuel,region,year,price,scenario,data_year,dollar_year
coal,CA_N,2030,2.5,reference,2025,2024
coal,CA_S,2030,2.6,reference,2025,2024
coal,AZ,2030,2.4,reference,2025,2024
naturalgas,CA_N,2030,4.2,reference,2025,2024
naturalgas,CA_S,2030,4.3,reference,2025,2024
naturalgas,AZ,2030,3.8,reference,2025,2024
uranium,CA_N,2030,0.8,reference,2025,2024
uranium,CA_S,2030,0.8,reference,2025,2024
uranium,AZ,2030,0.8,reference,2025,2024
```

### Legacy Workflow Example (AEO Direct)

If using EIA AEO data directly:

```yaml
# AEO fuel prices
fuel_data_year: 2025

fuel_series_scenario_names:
  reference: REF2025
  high_resource: HIGHOGS
  low_price: LOWPRICE

fuel_series_names:
  coal: STC
  naturalgas: NG
  distillate: DFO
  uranium: U

fuel_series_region_names:
  pacific: PCF
  mountain: MTN

fuel_scenarios:
  coal: reference
  naturalgas: reference
  uranium: reference
  distillate: high_resource

fuel_region_map:
  pacific: [CA_N, CA_S]
  mountain: [AZ, NM]

# Emission factors
fuel_emission_factors:
  coal: 0.09552
  naturalgas: 0.05306
  distillate: 0.07315
  uranium: 0.0
  biomass: 0.0

# Regional adjustments
regional_fuel_adjustments:
  CA_N:
    naturalgas: [mul, 1.2]
  CA_S:
    naturalgas: [mul, 1.2]

# Carbon pricing
carbon_tax:
  2030: 50
  2040: 75
  2050: 100

```

## Data Sources

Prepare an AEO-derived table separately, then configure it with `fuel_price_table`.

### Custom Data Tables

Alternative to AEO, provide complete fuel price time series:

**fuel_prices.csv**:

```csv
fuel,region,year,price,scenario,data_year,dollar_year
naturalgas,CA_N,2030,4.5,reference,2025,2024
naturalgas,CA_S,2030,4.7,reference,2025,2024
naturalgas,CA_N,2040,5.2,reference,2025,2024
naturalgas,CA_S,2040,5.4,reference,2025,2024
```

Configure:

```yaml
fuel_price_table: fuel_prices.csv
fuel_data_year: 2025
```

## Fuel Cost Calculation

Final fuel cost per MMBtu:

```text
total_fuel_cost = base_price
                  × regional_adjustment
                  + (carbon_tax × emission_factor)
```

Example (natural gas in CA_N, 2030):

```text
base_price = 4.00 $/MMBtu (from AEO)
regional_adjustment = 1.2
carbon_tax = 100 $/tonne
emission_factor = 0.05306 tonnes/MMBtu

total_cost = 4.00 × 1.2 + (100 × 0.05306)
          = 4.80 + 5.31
          = 10.11 $/MMBtu
```

This cost is combined with heat rates to calculate fuel costs per MWh for each generator.

## Related Settings

- [Model Definition](model-definition.md): `target_usd_year` for price conversions
- [Regions](regions.md): `fuel_region_map` maps AEO regions to model regions
- [Existing Generators](existing-generators.md): Heat rates convert fuel → electricity costs
- [New-Build Resources](new-build.md): Heat rates for new thermal plants
