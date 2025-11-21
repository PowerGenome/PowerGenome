# Fuel Settings

These parameters control fuel price sources, emission factors, and regional fuel price mappings.

## Overview

PowerGenome supports two fuel pricing workflows:

1. **Simplified (Recommended)**: Provide a fuel price table with prices for all model regions and fuels. PowerGenome automatically constructs fuel names and calculates aggregated region prices.

2. **Legacy (Larger fuel regions)**: Use raw AEO data with mapping parameters to create regional fuel prices.

**Most users should use the simplified workflow** with a complete fuel price table containing all base regions.

## Simplified Workflow (Recommended)

### Required Settings

Only three parameters are needed:

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
```

### Fuel Price Table Format

Your fuel price table must include prices for **every base region** in your model:

**Required columns**:

- `fuel`: Fuel name (coal, naturalgas, distillate, uranium, etc.)
- `region`: Base region name (must match regions in your model)
- `year`: Calendar year
- `price`: Price ($/MMBtu)

**Optional columns**:

- `scenario`: Scenario identifier (if using multiple price scenarios)
- `data_year`: Data vintage year
- `dollar_year`: Price year (for inflation adjustment)

**Example fuel_prices.csv**:

```csv
fuel,region,year,price,scenario,dollar_year
coal,CA_N,2030,2.5,reference,2024
coal,CA_S,2030,2.6,reference,2024
naturalgas,CA_N,2030,4.2,reference,2024
naturalgas,CA_S,2030,4.3,reference,2024
uranium,CA_N,2030,0.8,reference,2024
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

Only needed if your `fuel_price_table` has a `data_year` column with multiple years.

## Legacy Mapping Parameters (Optional)

!!! warning "Legacy Parameters"
    The parameters below are **only needed for the legacy AEO direct workflow**. If your fuel price table already contains prices for all base regions with the fuel/scenario names you want, **skip this entire section**.

These parameters were originally designed for working with EIA AEO data directly via API or raw AEO files with region/fuel codes that need mapping.

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

**Type**: Dictionary (fuel → emissions dict)
**Required**: For emissions modeling
**Example**: See below

CO₂ emission factors by fuel type (tonnes CO₂ per MMBtu).

```yaml
fuel_emission_factors:
  coal:
    co2_tons_per_mmbtu: 0.09552  # Bituminous coal
  naturalgas:
    co2_tons_per_mmbtu: 0.05306  # Natural gas
  distillate:
    co2_tons_per_mmbtu: 0.07315  # Diesel/fuel oil
  uranium:
    co2_tons_per_mmbtu: 0.0  # Nuclear (no combustion)
  biomass:
    co2_tons_per_mmbtu: 0.0  # Often counted as carbon neutral
```

Values are based on fuel carbon content and combustion chemistry.

**Common factors** (tonnes CO₂/MMBtu):

- Bituminous coal: 0.0955
- Sub-bituminous coal: 0.0972
- Natural gas: 0.0531
- Diesel: 0.0732
- Biomass: 0.0 (policy-dependent)

### `fuel_prices_table`

**Type**: String or dictionary
**Required**: For custom fuel prices
**Example**: See below

Custom fuel price time series (alternative to AEO).

**Simple**:

```yaml
fuel_prices_table: fuel_prices.csv
```

**Advanced**:

```yaml
fuel_prices_table:
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

**Type**: Dictionary (fuel → year → price)
**Required**: No
**Example**: See below

Directly specify fuel prices by year, overriding AEO or table data.

```yaml
user_fuel_price:
  naturalgas:
    2030: 4.5  # $/MMBtu
    2040: 5.2
    2050: 5.8
  coal:
    2030: 2.1
    2040: 2.3
    2050: 2.5
```

Useful for sensitivity analyses or policy scenarios (e.g., carbon pricing embedded in fuel costs).

### `user_fuel_price_regions`

**Type**: List of regions
**Required**: If using `user_fuel_price` with regional variation
**Example**: `["CA_N", "CA_S", "AZ"]`

Regions where `user_fuel_price` should apply. If omitted, applies to all regions.

```yaml
user_fuel_price_regions: [CA_N, CA_S]
```

## Fuel Technology Mappings

### `tech_fuel_map`

**Type**: Dictionary (technology → fuel)
**Required**: No (has defaults)
**Example**: See below

Maps technologies to fuel types. PowerGenome has defaults, but you can override.

```yaml
tech_fuel_map:
  NaturalGas_CCAvgCF_Moderate: naturalgas
  Coal_new_Moderate: coal
  Nuclear_Nuclear_Moderate: uranium
  Biomass_Dedicated_Moderate: biomass
```

**Default mappings**:

- Natural gas technologies → `naturalgas`
- Coal technologies → `coal`
- Nuclear → `uranium`
- Biomass/MSW → `biomass`
- Oil/diesel → `distillate`

Custom mappings are only needed for non-standard technology names.

### `eia_atb_tech_map`

**Type**: Dictionary (EIA tech → ATB tech)
**Required**: No
**Example**: See below

Maps EIA technology names to ATB names for startup cost calculations.

```yaml
eia_atb_tech_map:
  Natural Gas Fired Combined Cycle: NaturalGas_CCAvgCF
  Natural Gas Fired Combustion Turbine: NaturalGas_CTAvgCF
  Conventional Steam Coal: Coal_newAvgCF
```

Used to assign ATB startup costs to existing generators based on their EIA technology classification.

## Regional Fuel Prices

### `regional_fuel_adjustments`

**Type**: Dictionary (region → fuel → multiplier)
**Required**: No
**Example**: See below

Regional price adjustments relative to AEO base prices.

```yaml
regional_fuel_adjustments:
  CA_N:
    naturalgas: 1.2  # 20% higher gas prices in CA
  AZ:
    coal: 0.9  # 10% lower coal prices
```

Accounts for regional market conditions, transportation costs, or local policies.

## Fuel Price Escalation

### `fuel_price_escalation`

**Type**: Dictionary (fuel → annual rate)
**Required**: No
**Example**: See below

Annual escalation rate for fuel prices beyond projection period.

```yaml
fuel_price_escalation:
  naturalgas: 0.02  # 2% annual escalation
  coal: 0.01        # 1% annual escalation
  uranium: 0.015
```

Applied to extrapolate prices beyond AEO projection horizon (typically 2050).

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

**Escalating**:

```yaml
carbon_tax:
  2030: 50
  2040: 75
  2050: 100
```

Carbon costs are calculated as:

```
carbon_cost_per_mmbtu = carbon_tax * fuel_emission_factors[fuel]['co2_tons_per_mmbtu']
```

### `regional_carbon_tax`

**Type**: Dictionary (region → carbon tax)
**Required**: No
**Example**: See below

Region-specific carbon pricing (overrides `carbon_tax`).

```yaml
regional_carbon_tax:
  CA_N: 100  # California carbon price
  CA_S: 100
  AZ: 25     # Lower carbon price in AZ
```

Implements regional climate policies (e.g., RGGI, California cap-and-trade).

## Biomass Constraints

### `biomass_supply_curve`

**Type**: Dictionary or table reference
**Required**: No
**Example**: See below

Biomass fuel supply curve (price vs. quantity available).

```yaml
biomass_supply_curve:
  CA_N:
    - quantity: 1000  # Tonnes/yr
      price: 2.5      # $/MMBtu
    - quantity: 2000
      price: 3.0
    - quantity: 3000
      price: 4.0
```

Models limited biomass resource availability with price escalation at high utilization.

## Example Configurations

### Simplified Workflow (Recommended)

Complete fuel pricing with custom fuel price table:

```yaml
# Fuel price table location
fuel_price_table: fuel_prices.csv

# Map scenario names to fuels
fuel_scenarios:
  coal: reference
  naturalgas: reference
  uranium: reference
  distillate: reference

# Map technologies to fuels
tech_fuel_map:
  Conventional Steam Coal: coal
  Natural Gas Fired Combined Cycle: naturalgas
  Natural Gas Fired Combustion Turbine: naturalgas
  Nuclear: uranium
  Petroleum Liquids: distillate

# Emission factors (tonnes CO2 per MMBtu)
fuel_emission_factors:
  coal: 0.09552
  naturalgas: 0.05306
  distillate: 0.07315
  uranium: 0.0

# Carbon pricing
carbon_tax:
  2030: 50
  2040: 75
  2050: 100

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
fuel,region,year,price,scenario,dollar_year
coal,CA_N,2030,2.5,reference,2024
coal,CA_S,2030,2.6,reference,2024
coal,AZ,2030,2.4,reference,2024
naturalgas,CA_N,2030,4.2,reference,2024
naturalgas,CA_S,2030,4.3,reference,2024
naturalgas,AZ,2030,3.8,reference,2024
uranium,CA_N,2030,0.8,reference,2024
uranium,CA_S,2030,0.8,reference,2024
uranium,AZ,2030,0.8,reference,2024
```

### Legacy Workflow (AEO Direct)

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
  biomass:
    co2_tons_per_mmbtu: 0.0

# Regional adjustments
regional_fuel_adjustments:
  CA_N:
    naturalgas: 1.2
  CA_S:
    naturalgas: 1.2

# Carbon pricing
carbon_tax:
  2030: 50
  2040: 75
  2050: 100

regional_carbon_tax:
  CA_N: 100  # Higher CA carbon price
  CA_S: 100
```

## Data Sources

### AEO API

PowerGenome can fetch fuel prices directly from EIA's AEO API:

```python
# Requires EIA API key in settings or environment
eia_api_key: "YOUR_API_KEY_HERE"
```

Set `EIA_API_KEY` environment variable or include in settings.

### Custom Data Tables

Alternative to AEO, provide complete fuel price time series:

**fuel_prices.csv**:

```csv
fuel,region,year,price
naturalgas,CA_N,2030,4.5
naturalgas,CA_N,2040,5.2
naturalgas,CA_S,2030,4.7
coal,AZ,2030,2.1
```

Configure:

```yaml
fuel_prices_table: fuel_prices.csv
```

## Fuel Cost Calculation

Final fuel cost per MMBtu:

```
total_fuel_cost = base_price
                  × regional_adjustment
                  + (carbon_tax × emission_factor)
```

Example (natural gas in CA_N, 2030):

```
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
