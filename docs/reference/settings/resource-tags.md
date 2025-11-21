# Resource Tags Settings

Resource tags control how generators are classified in GenX and other capacity expansion models. Tags determine dispatch behavior, policy eligibility, and operational constraints.

## Tag System Overview

GenX uses binary or integer tags to categorize resources. Common tags include:

- **THERM**: Thermal zone for reserves/capacity requirements
- **VRE**: Variable renewable energy (wind/solar)
- **STOR**: Storage resources (batteries, pumped hydro)
- **FLEX**: Flexible demand/demand response
- **HYDRO**: Hydroelectric resources
- **MUST_RUN**: Must-run generators (cogen, nuclear)
- **LDS**: Long-duration storage (>12 hours)

Each resource must have a value for every tag defined in the model.

## Defining Tags

### `model_tag_names`

**Type**: List of strings
**Required**: Yes
**Example**: See below

List of all resource tags used in the model.

```yaml
model_tag_names:
  - THERM
  - VRE
  - STOR
  - FLEX
  - HYDRO
  - MUST_RUN
  - LDS
```

All resources must be assigned a value for each tag in this list.

!!! note "Custom Tags"
    You can define custom tags for specialized models. For example, `OFFSHORE` for offshore wind, `BIOMASS` for biomass resources, or `NEW_BUILD` to distinguish new vs. existing capacity.

## Technology-Level Tags

### `model_tag_values`

**Type**: Dictionary (technology → tag → value)
**Required**: Yes
**Example**: See below

Default tag values for each technology. These apply across all regions unless overridden.

```yaml
model_tag_values:
  NaturalGas_CCCCSAvgCF_Moderate:
    THERM: 1
    VRE: 0
    STOR: 0
    FLEX: 0
    HYDRO: 0
    MUST_RUN: 0
    LDS: 0

  UtilityPV_Class1_Moderate:
    THERM: 0
    VRE: 1
    STOR: 0
    FLEX: 0
    HYDRO: 0
    MUST_RUN: 0
    LDS: 0

  Battery_4Hr_Moderate:
    THERM: 0
    VRE: 0
    STOR: 1
    FLEX: 0
    HYDRO: 0
    MUST_RUN: 0
    LDS: 0

  Hydroelectric_Pumped_Storage:
    THERM: 0
    VRE: 0
    STOR: 1
    FLEX: 0
    HYDRO: 1
    MUST_RUN: 0
    LDS: 1
```

**Tag values**:

- Binary (0/1): Most tags are binary flags
- Integer (1, 2, 3...): THERM tag often uses integers for thermal zones
- Wildcard matching: Use `*` in technology names to match patterns

### Wildcard Patterns

Use wildcards to assign tags to multiple technologies at once:

```yaml
model_tag_values:
  # Match all battery durations
  Battery_*_Moderate:
    THERM: 0
    VRE: 0
    STOR: 1
    FLEX: 0

  # Match all natural gas technologies
  NaturalGas_*:
    THERM: 1
    VRE: 0
    STOR: 0

  # Match all wind classes
  LandbasedWind_Class*_Moderate:
    THERM: 0
    VRE: 1
    STOR: 0
```

Wildcards make configuration more maintainable when you have many similar technologies.

## Regional Tag Overrides

### `regional_tag_values`

**Type**: Dictionary (region → technology → tag → value)
**Required**: No
**Example**: See below

Override tag values for specific technologies in specific regions. Regional values take precedence over `model_tag_values`.

```yaml
regional_tag_values:
  CA_N:
    NaturalGas_CCCCSAvgCF_Moderate:
      THERM: 2  # Different thermal zone in CA
    Conventional_Hydroelectric:
      MUST_RUN: 1  # California hydro is must-run

  AZ:
    NaturalGas_CCCCSAvgCF_Moderate:
      THERM: 1  # Arizona thermal zone
    Conventional_Hydroelectric:
      MUST_RUN: 0  # Arizona hydro is flexible
```

**Use cases**:

- Different thermal reserve zones by region
- Region-specific must-run requirements (e.g., cogeneration contracts)
- Policy differences (e.g., renewable portfolio standard eligibility)

## Common Tag Configurations

### Thermal Generators

```yaml
model_tag_values:
  NaturalGas_CCAvgCF_Moderate:
    THERM: 1      # Thermal generator
    VRE: 0        # Not variable renewable
    STOR: 0       # Not storage
    MUST_RUN: 0   # Can be dispatched
    HYDRO: 0
    FLEX: 0
    LDS: 0

  Coal_NewAvgCF_Moderate:
    THERM: 1
    VRE: 0
    STOR: 0
    MUST_RUN: 0
    HYDRO: 0
    FLEX: 0
    LDS: 0

  Nuclear_Nuclear_Moderate:
    THERM: 1
    VRE: 0
    STOR: 0
    MUST_RUN: 1   # Nuclear often must-run
    HYDRO: 0
    FLEX: 0
    LDS: 0
```

### Variable Renewables

```yaml
model_tag_values:
  UtilityPV_Class1_Moderate:
    THERM: 0
    VRE: 1        # Variable renewable
    STOR: 0
    MUST_RUN: 0
    HYDRO: 0
    FLEX: 0
    LDS: 0

  LandbasedWind_Class3_Moderate:
    THERM: 0
    VRE: 1
    STOR: 0
    MUST_RUN: 0
    HYDRO: 0
    FLEX: 0
    LDS: 0

  OffshoreWind_Class1_Moderate:
    THERM: 0
    VRE: 1
    STOR: 0
    MUST_RUN: 0
    HYDRO: 0
    FLEX: 0
    LDS: 0
```

### Storage Resources

```yaml
model_tag_values:
  Battery_2Hr_Moderate:
    THERM: 0
    VRE: 0
    STOR: 1       # Storage resource
    MUST_RUN: 0
    HYDRO: 0
    FLEX: 0
    LDS: 0        # Not long-duration (<12hr)

  Battery_8Hr_Moderate:
    THERM: 0
    VRE: 0
    STOR: 1
    MUST_RUN: 0
    HYDRO: 0
    FLEX: 0
    LDS: 0        # 8hr not typically "long-duration"

  Hydroelectric_Pumped_Storage:
    THERM: 0
    VRE: 0
    STOR: 1
    MUST_RUN: 0
    HYDRO: 1      # Also tagged as hydro
    FLEX: 0
    LDS: 1        # Long-duration storage
```

### Hydroelectric

```yaml
model_tag_values:
  Conventional_Hydroelectric:
    THERM: 0
    VRE: 0
    STOR: 0
    MUST_RUN: 0   # Flexible (unless regional override)
    HYDRO: 1      # Hydroelectric resource
    FLEX: 0
    LDS: 0

  Hydroelectric_Pumped_Storage:
    THERM: 0
    VRE: 0
    STOR: 1       # Pumped storage is storage + hydro
    MUST_RUN: 0
    HYDRO: 1
    FLEX: 0
    LDS: 1
```

### Demand Response

```yaml
model_tag_values:
  EV_Charging:
    THERM: 0
    VRE: 0
    STOR: 0
    MUST_RUN: 0
    HYDRO: 0
    FLEX: 1       # Flexible demand
    LDS: 0

  Water_Heating:
    THERM: 0
    VRE: 0
    STOR: 0
    MUST_RUN: 0
    HYDRO: 0
    FLEX: 1
    LDS: 0
```

## Policy-Related Tags

### Energy Share Requirements (ESR)

Tags can define renewable/clean energy eligibility:

```yaml
model_tag_names:
  - THERM
  - VRE
  - STOR
  - ESR_1    # Renewable Portfolio Standard
  - ESR_2    # Clean Energy Standard

model_tag_values:
  UtilityPV_Class1_Moderate:
    ESR_1: 1  # Eligible for RPS
    ESR_2: 1  # Eligible for CES

  NaturalGas_CCCCSAvgCF_Moderate:
    ESR_1: 0  # Not eligible for RPS
    ESR_2: 1  # May be eligible for CES (with CCS)

  Nuclear_Nuclear_Moderate:
    ESR_1: 0  # Not eligible for RPS
    ESR_2: 1  # Eligible for CES (carbon-free)
```

See [Energy Share Requirements How-To](../../how-to/energy-share-requirements.md) for policy configuration.

### Minimum/Maximum Capacity Tags

Tags can enforce capacity constraints:

```yaml
model_tag_names:
  - THERM
  - VRE
  - MinCapTag_1  # Minimum solar requirement
  - MaxCapTag_1  # Maximum gas limit

model_tag_values:
  UtilityPV_Class1_Moderate:
    MinCapTag_1: 1  # Counts toward solar minimum
    MaxCapTag_1: 0

  NaturalGas_CCAvgCF_Moderate:
    MinCapTag_1: 0
    MaxCapTag_1: 1  # Counts toward gas maximum
```

Configure requirements in `emission_policies_fn` or via GenX settings.

## Capacity Reserve Tags

### `CapRes_<num>` Tags

Define which reserve zones each resource contributes to:

```yaml
model_tag_names:
  - CapRes_1  # Primary reserve zone
  - CapRes_2  # Secondary reserve zone

model_tag_values:
  NaturalGas_CCAvgCF_Moderate:
    CapRes_1: 1  # Contributes to zone 1
    CapRes_2: 0

  Battery_4Hr_Moderate:
    CapRes_1: 1  # Storage can provide reserves
    CapRes_2: 0

regional_tag_values:
  CA_N:
    NaturalGas_CCAvgCF_Moderate:
      CapRes_1: 1
      CapRes_2: 1  # CA resources contribute to both zones
```

Capacity reserve requirements are set in `regional_capacity_reserves`.

## Tag Validation

PowerGenome validates tag assignments:

**Missing tags**: Error if a resource lacks a required tag

```
ResourceTagError: Technology 'UtilityPV' missing tag 'THERM'
```

**Solution**: Add missing tag to `model_tag_values` or `regional_tag_values`.

**Too many tags**: Warning if resource has tags not in `model_tag_names`

```
Warning: Resource has tag 'OFFSHORE' not in model_tag_names
```

**Solution**: Add tag to `model_tag_names` or remove from resource definitions.

## Generator Columns Output

Tags appear as columns in `Generators_data.csv`:

```csv
Resource,technology,THERM,VRE,STOR,FLEX,HYDRO,MUST_RUN,LDS
CA_N_NaturalGas_CC_1,NaturalGas_CC,1,0,0,0,0,0,0
CA_N_Solar_PV_1,UtilityPV,0,1,0,0,0,0,0
CA_N_Battery_1,Battery_4Hr,0,0,1,0,0,0,0
```

GenX reads these columns to determine resource behavior in optimization.

## Example Configuration

Complete tag configuration for a multi-region model:

```yaml
# Define all tags
model_tag_names:
  - THERM
  - VRE
  - STOR
  - FLEX
  - HYDRO
  - MUST_RUN
  - LDS
  - ESR_1
  - CapRes_1

# Technology defaults
model_tag_values:
  # Natural gas
  NaturalGas_*:
    THERM: 1
    VRE: 0
    STOR: 0
    FLEX: 0
    HYDRO: 0
    MUST_RUN: 0
    LDS: 0
    ESR_1: 0
    CapRes_1: 1

  # Solar
  UtilityPV_*:
    THERM: 0
    VRE: 1
    STOR: 0
    FLEX: 0
    HYDRO: 0
    MUST_RUN: 0
    LDS: 0
    ESR_1: 1  # RPS-eligible
    CapRes_1: 1

  # Batteries
  Battery_*:
    THERM: 0
    VRE: 0
    STOR: 1
    FLEX: 0
    HYDRO: 0
    MUST_RUN: 0
    LDS: 0
    ESR_1: 0
    CapRes_1: 1

# Regional overrides
regional_tag_values:
  CA_N:
    NaturalGas_CCAvgCF_Moderate:
      THERM: 2  # Different thermal zone
  CA_S:
    NaturalGas_CCAvgCF_Moderate:
      THERM: 2
```

## Related Settings

- [Model Definition](model-definition.md): `model_tag_names` defined here
- [Existing Generators](existing-generators.md): Tags assigned to clustered generators
- [New-Build Resources](new-build.md): Tags for candidate technologies
- [Regions](regions.md): Regional tag overrides by model region
