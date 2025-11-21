# Regional Configuration

Regional settings define the geographic structure of your model—how base regions (e.g., from IPM or ReEDS) aggregate into model regions, and region-specific customizations.

## Core Regional Parameters

### `model_regions`

**Type**: List of strings
**Required**: Yes
**Example**: `["CA_N", "CA_S", "AZ", "NM"]`

Names of the regions that will appear in output files. These can be:

- Single base regions (no aggregation)
- Aggregated region names (defined in `region_aggregations`)

!!! tip "Base Regions"
    "Base regions" refer to the geographic units in your source data. While PowerGenome examples use US IPM or ReEDS regions, base regions can represent any geography: European countries, Chinese provinces, market zones, utility territories, etc. Define regions that match your input data tables.

```yaml
model_regions: [CA_N, CA_S, AZ]
```

All regional parameters use these names as keys.

### `region_aggregations`

**Type**: Dictionary (model region → list of base regions)
**Required**: Only if using aggregations
**Example**: See below

Maps model region names to the base regions they contain. Base regions come from your data source (IPM, ReEDS, etc.).

```yaml
region_aggregations:
  CA_N:
    - CA_IID
    - CA_LADWP
    - CA_BANC
  CA_S:
    - CA_SCE
    - CA_SDGE
  MOUNTAIN:
    - AZ
    - NM
    - CO_S
```

**Important**:

- Model region names in `region_aggregations` must appear in `model_regions`
- Base region names must match those in your generation/demand data tables
- If `model_regions` contains a name NOT in `region_aggregations`, it's treated as a pass-through (1:1 mapping to a base region)

### `alt_region_names`

**Type**: Dictionary (old name → new name)
**Required**: No
**Example**: `{"CA_N": "Northern California"}`

Renames regions in output files. Keys are model region names, values are display names.

```yaml
alt_region_names:
  CA_N: Northern California
  CA_S: Southern California
```

## Regional Capacity Reserves

### `regional_capacity_reserves`

**Type**: Nested dictionary
**Required**: No
**Example**: See below

Defines capacity reserve requirements by zone. Top-level keys are `CapRes_<num>` tags, second level keys are model regions.

```yaml
regional_capacity_reserves:
  CapRes_1:
    CA_N: 0.15  # 15% reserve margin
    CA_S: 0.15
    AZ: 0.12
  CapRes_2:
    CA_N: 0.20  # Additional winter reserve
    CA_S: 0.18
```

Each `CapRes_X` tag creates a corresponding column in generators output, indicating which reserve zone each resource can contribute to.

### `cap_res_network_derate_default`

**Type**: Float (0-1)
**Required**: No
**Default**: 1.0
**Example**: `0.95`

Derating factor for transmission imports used to meet capacity reserves. A value of 0.95 means transmission can only contribute 95% of its capacity toward reserve requirements.

```yaml
cap_res_network_derate_default: 0.95
```

## Technology-Specific Regional Settings

### `cogen_tech`

**Type**: Dictionary (region → list of technologies)
**Required**: No
**Example**: See below

Identifies cogeneration (combined heat and power) technologies by region. These are often must-run resources.

```yaml
cogen_tech:
  CA_N: [Natural Gas Steam Turbine]
  CA_S: [Biomass, Natural Gas Steam Turbine]
```

Cogen plants are typically:

- Tagged as `MUST_RUN: 1`
- Excluded from clustering (each plant is its own resource)
- Have special dispatch constraints

### `new_gen_not_available`

**Type**: Dictionary (region → list of new-build technologies)
**Required**: No
**Example**: See below

Specifies new-build resources that should NOT be available in certain regions. Useful for policy/physical constraints.

```yaml
new_gen_not_available:
  AZ:
    - OffshoreWind_Class1_Moderate_-1  # No offshore wind in AZ
    - Geothermal_HydroFlash_Moderate  # No geothermal potential
  CA_N:
    - Coal_new_Moderate  # Coal banned in CA
```

Technology names must match those in `new_resources`.

### `regional_no_grouping`

**Type**: Dictionary (region → list of technologies)
**Required**: No
**Example**: See below

Technologies that should NOT be clustered in specific regions. Each existing plant becomes its own resource.

```yaml
regional_no_grouping:
  CA_N:
    - Nuclear  # Each reactor is unique
    - Hydroelectric Pumped Storage
  AZ:
    - Coal  # Large, distinct coal plants
```

This overrides `num_clusters` and `tech_groups` for specified technologies.

## Load Zones

### `load_zones`

**Type**: Dictionary (region → list of sub-zones) OR list of regions
**Required**: No
**Example**: See below

Defines demand zones within model regions. If omitted, one zone per region is assumed.

**Simple** (one zone per region):

```yaml
load_zones: [CA_N, CA_S, AZ]
```

**Complex** (multiple zones per region):

```yaml
load_zones:
  CA_N:
    - CA_N_URBAN
    - CA_N_RURAL
  CA_S:
    - CA_S_URBAN
    - CA_S_RURAL
  AZ: [AZ]  # Single zone
```

### `load_region_map`

**Type**: Dictionary (region → demand file region name)
**Required**: No
**Example**: See below

Maps model regions to region names in demand data files. Needed when region names differ between settings and data sources.

```yaml
load_region_map:
  CA_N: CAMX_N  # Model region → demand file region
  CA_S: CAMX_S
  AZ: AZNM
```

### `future_load_region_map`

**Type**: Dictionary (region → future demand region name)
**Required**: No
**Example**: Similar to `load_region_map`

Separate mapping for future demand projections if they use different region names.

## Cost Multipliers

### `cost_multiplier_region_map`

**Type**: Dictionary (region → cost region name)
**Required**: No
**Example**: See below

Maps model regions to cost multiplier region names. Used with `cost_multiplier_fn` to apply regional construction cost adjustments.

```yaml
cost_multiplier_region_map:
  CA_N: CA_N_SPUR
  CA_S: CA_S_SPUR
  AZ: Southwest
```

This allows using generic cost multiplier tables (e.g., from NREL) that use different region names.

### `cost_multiplier_technology_map`

**Type**: Dictionary (PowerGenome tech → cost tech name)
**Required**: No
**Example**: See below

Maps technology names to those in cost multiplier files.

```yaml
cost_multiplier_technology_map:
  NaturalGas_CCAvgCF_Moderate: NaturalGas_CC
  UtilityPV_Class1_Moderate: LandbasedWind  # Use wind costs for PV
```

## Fuel Regions

### `fuel_region_map`

**Type**: Dictionary (AEO region → list of model regions)
**Required**: For fuel costs from EIA AEO
**Example**: See below

Maps EIA Annual Energy Outlook (AEO) fuel price regions to model regions.

```yaml
fuel_region_map:
  pacific: [CA_N, CA_S]
  mountain: [AZ, NM]
  west_south_central: [TX]
```

AEO region names:

- `pacific`: Pacific
- `mountain`: Mountain
- `west_south_central`: West South Central
- `east_north_central`: East North Central
- `new_england`: New England

## Renewable Resource Bins

### `new_wind_solar_regional_bins`

**Type**: Dictionary (region → technology → number of bins)
**Required**: No (deprecated)
**Example**: See below

**Deprecated**: Use `renewable_clusters` instead.

Previously controlled how many resource quality bins for wind/solar in each region.

```yaml
# Old approach (deprecated)
new_wind_solar_regional_bins:
  CA_N:
    UtilityPV: 3
    LandbasedWind: 5
```

Modern approach uses `renewable_clusters` to explicitly define resource sites.

## UTC Offset

### `utc_offset`

**Type**: Integer
**Required**: No
**Default**: 0
**Example**: `-8`

Hour offset from UTC for the model timezone. All time-series data is stored in UTC and converted using this offset.

```yaml
utc_offset: -8  # Pacific Time (UTC-8)
```

US time zones:

- Pacific: `-8`
- Mountain: `-7`
- Central: `-6`
- Eastern: `-5`

## Example Configuration

Complete regional settings for a California + Southwest model:

```yaml
# Base model regions
model_regions: [CA_N, CA_S, AZ, NM]

# Aggregate IPM regions
region_aggregations:
  CA_N: [CA_IID, CA_LADWP, CA_BANC]
  CA_S: [CA_SCE, CA_SDGE]

# Regional capacity reserves
regional_capacity_reserves:
  CapRes_1:
    CA_N: 0.15
    CA_S: 0.15
    AZ: 0.12
    NM: 0.12

cap_res_network_derate_default: 0.95

# Technology restrictions
cogen_tech:
  CA_N: [Natural Gas Steam Turbine]
  CA_S: [Natural Gas Steam Turbine]

new_gen_not_available:
  CA_N: [Coal_new, OffshoreWind]
  CA_S: [Coal_new]
  AZ: [OffshoreWind, Geothermal]
  NM: [OffshoreWind, Geothermal]

# Load zones
load_zones: [CA_N, CA_S, AZ, NM]

load_region_map:
  CA_N: CAMX_N
  CA_S: CAMX_S
  AZ: AZNM
  NM: AZNM

# Regional mappings
fuel_region_map:
  pacific: [CA_N, CA_S]
  mountain: [AZ, NM]

cost_multiplier_region_map:
  CA_N: CA_N_spur
  CA_S: CA_S_spur
  AZ: Southwest
  NM: Southwest

# Timezone
utc_offset: -8
```

## Related Settings

- [Model Definition](model-definition.md): Planning years and model scope
- [Existing Generators](existing-generators.md): Region-specific clustering
- [New-Build Resources](new-build.md): Regional availability constraints
- [Transmission](transmission.md): Inter-regional network topology
