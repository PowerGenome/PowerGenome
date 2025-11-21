# Model Definition Settings

These core parameters define the overall model scope, planning periods, and currency adjustments.

## Planning Years

### `model_year`

**Type**: List of integers
**Required**: Yes
**Example**: `[2030, 2040, 2050]`

The primary planning year(s) for the model. All costs are calculated for this year, and it determines which technology cost projections to use.

```yaml
model_year: [2030, 2040, 2050]
```

### `model_first_planning_year`

**Type**: List of integers
**Required**: Yes
**Example**: `[2025, 2031, 2041]`

The first year in a multi-period planning horizon. Used to calculate period lengths and determine which existing generators are retired.

```yaml
model_year: [2030, 2040, 2050]
model_first_planning_year: [2025, 2031, 2041]
```

This creates planning periods:

- 2025-2030 (5 years)
- 2031-2040 (10 years)
- 2041-2050 (10 years)

## Currency Conversion

### `target_usd_year`

**Type**: Integer
**Required**: Yes
**Example**: `2023`

All monetary values are converted to this base year USD using CPI data.

```yaml
target_usd_year: 2023
```

PowerGenome loads CPI data from `data/cpi_data/` to perform conversions. Technology costs from ATB (in their native years) and EIA data (various years) are all normalized to this target year.

### `cpi_data_file`

**Type**: String (path)
**Required**: No
**Default**: Built-in CPI data
**Example**: `"data/cpi_data/CPI_2024.csv"`

Path to custom CPI data file for currency conversion.

```yaml
cpi_data_file: custom_data/inflation_factors.csv
```

The file should have columns:

- `year`: Calendar year
- `cpi`: Consumer Price Index value

## Model Tags

### `model_tag_names`

**Type**: List of strings
**Required**: For GenX outputs
**Example**: `[THERM, VRE, STOR, FLEX, HYDRO, MUST_RUN, LDS]`

Defines which resource tags are used in the model. These control GenX dispatch behavior:

- **THERM**: Thermal generators (fossil/nuclear)
- **VRE**: Variable renewable energy (wind/solar)
- **STOR**: Storage resources (batteries)
- **FLEX**: Flexible load/demand response
- **HYDRO**: Hydroelectric resources
- **MUST_RUN**: Must-run generators (cogen, etc.)
- **LDS**: Long-duration storage

```yaml
model_tag_names: [THERM, VRE, STOR, FLEX, HYDRO, MUST_RUN, LDS]
```

Each resource must be assigned a value for each tag (typically 0 or 1).

!!! note "Custom Tags"
    You can define custom tags for specialized model variants. All tags must have values assigned in `model_tag_values` or `regional_tag_values`.

### `default_model_tag`

**Type**: Integer
**Required**: No
**Default**: `0`
**Example**: `0`

Default value for all tag assignments. This allows you to specify only non-default values in `model_tag_values`, reducing configuration verbosity.

```yaml
default_model_tag: 0

model_tag_names: [THERM, VRE, STOR]
model_tag_values:
  THERM:
    NaturalGas_CCAvgCF_Moderate: 1  # Only specify non-zero values
  VRE:
    UtilityPV_Class1_Moderate: 1
  STOR:
    Battery_*_Moderate: 1
```

With `default_model_tag: 0`, any technology not explicitly listed under a tag automatically gets value `0`. Without this parameter, you would need to explicitly set all zero values.

### `model_tag_values`

**Type**: Dictionary (tag → technology → value)
**Required**: Yes
**Example**: See below

Assigns tag values to each technology. Structure has tags as top-level keys, with technologies underneath. These are technology-level defaults that can be overridden regionally.

**Matching behavior**: Technology names are **substring matched** against actual resource names. Names are sorted by length (shortest first) before matching, ensuring more specific names are applied after generic ones.

```yaml
model_tag_values:
  THERM:
    NaturalGas_CCCCSAvgCF_Moderate: 1
  VRE:
    UtilityPV_Class1_Moderate: 1
  STOR:
    Battery_*_Moderate: 1
  New_Build:
    Nuclear_Nuclear: 1      # Matched second (longer string)
    Nuclear: 0              # Matched first (shorter string)
```

**Example matching**: For a resource named `Nuclear_Nuclear - Large_Moderate`:

1. First tries `Nuclear_Nuclear` (longest match) → finds match, applies value `1`
2. Shorter strings like `Nuclear` are not checked once a match is found

This allows you to set broad defaults (e.g., `Battery`) and specific overrides (e.g., `Battery_8Hr_Conservative`).

### `regional_tag_values`

**Type**: Dictionary (region → tag → technology → value)
**Required**: No
**Example**: See below

Overrides `model_tag_values` for specific regions. Useful when technology behavior differs by location.

```yaml
regional_tag_values:
  CA_N:
    THERM:
      NaturalGas_CCCCSAvgCF_Moderate: 2  # Different THERM zone in CA
  AZ:
    THERM:
      NaturalGas_CCCCSAvgCF_Moderate: 1
```

Regional values take precedence over model-level defaults.

## Execution Control

### `capacity_col`

**Type**: String
**Required**: No
**Default**: `"capacity_mw"`
**Example**: `"summer_capacity_mw"`

Column name in generation data containing capacity values.

```yaml
capacity_col: summer_capacity_mw
```

This allows using different capacity measures (e.g., nameplate vs. summer vs. winter).

## Data Paths

### `data_location`

**Type**: String (path)
**Required**: Yes (replaces legacy env vars)
**Example**: `"/path/to/data_folder"`

Root directory for data files. DataManager loads tables from this location.

```yaml
data_location: /Users/me/powergenome_data
```

This directory should contain:

- Generation data (CSV/Parquet)
- Demand profiles
- Fuel prices
- Transmission constraints

### `input_folder`

**Type**: String (path)
**Required**: No
**Default**: Current directory
**Example**: `"extra_inputs"`

Folder containing supplementary input files:

- `emission_policies.csv`
- `misc_gen_inputs.csv`
- `fuel_emissions.csv`

```yaml
input_folder: extra_inputs
```

Paths in this folder can be relative to settings location.

### `RESOURCE_GROUP_PROFILES`

**Type**: String (path)
**Required**: For renewable resources
**Example**: `"/path/to/generation_profiles"`

Directory containing hourly generation profiles for renewable resource groups.

```yaml
RESOURCE_GROUP_PROFILES: /data/nrel_profiles
```

Files should be named like `{technology}_{tech_detail}_{cluster_id}.csv` with hourly capacity factors.

### `DISTRIBUTED_GEN_DATA`

**Type**: String (path)
**Required**: No (deprecated)
**Example**: `"/path/to/dg_data"`

Legacy path for distributed generation data. Modern approach uses table configurations:

```yaml
# Old approach
DISTRIBUTED_GEN_DATA: /path/to/dg_data

# New approach (preferred)
dg_capacity_table: distributed_gen_capacity.parquet
dg_profiles_table: distributed_gen_profiles.parquet
```

## Example Configuration

Complete model definition for a multi-period study:

```yaml
# Planning horizon
model_year: [2030, 2040, 2050]
model_first_planning_year: 2025

# Currency
target_usd_year: 2023

# Tags
default_model_tag: 0
model_tag_names: [THERM, VRE, STOR, FLEX, HYDRO, MUST_RUN, LDS]
model_tag_values:
  THERM:
    NaturalGas_CCAvgCF_Moderate: 1
  VRE:
    UtilityPV_Class1_Moderate: 1
  STOR:
    Battery_*_Moderate: 1
  # Other tags default to 0 for all technologies

# Data paths
data_location: /Users/me/pg_data
input_folder: extra_inputs
RESOURCE_GROUP_PROFILES: /Users/me/nrel_profiles

# Execution
sort_gens: true
capacity_col: capacity_mw
```

## Related Settings

- [Regions](regions.md): Define model regions and aggregations
- [Scenario Management](scenario-management.md): Multi-scenario parameter swapping
- [Resource Tags](resource-tags.md): Detailed tag configuration
- [Data Tables](data-tables.md): Input data source configuration
