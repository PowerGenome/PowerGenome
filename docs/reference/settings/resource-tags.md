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

**Type**: Dictionary (tag → technology → value)
**Required**: Yes
**Example**: See below

Default tag values for each technology. These apply across all regions unless overridden.

!!! info "Technology Matching"
    Technology names are matched using **case-insensitive substring matching**. Shortest (most general) patterns are applied first, then longer (more specific) patterns override previous assignments. This allows you to set broad defaults and create specific exceptions.

```yaml
model_tag_values:
  THERM:
    NaturalGas: 1
    UtilityPV: 0
    Battery: 0
    Hydroelectric: 0

  VRE:
    NaturalGas: 0
    UtilityPV: 1
    Battery: 0
    Hydroelectric: 0

  STOR:
    NaturalGas: 0
    UtilityPV: 0
    Battery: 1
    Hydroelectric_Pumped: 1  # More specific, overrides 'Hydroelectric'

  FLEX:
    NaturalGas: 0
    UtilityPV: 0
    Battery: 0
    Hydroelectric: 0

  HYDRO:
    NaturalGas: 0
    UtilityPV: 0
    Battery: 0
    Hydroelectric: 1

  MUST_RUN:
    NaturalGas: 0
    UtilityPV: 0
    Battery: 0
    Hydroelectric: 0

  LDS:
    NaturalGas: 0
    UtilityPV: 0
    Battery: 0
    Hydroelectric_Pumped: 1
```

**Tag values**:

- Binary (0/1): Most tags are binary flags
- Integer (1, 2, 3...): THERM tag often uses integers for thermal zones
- Wildcard matching: Use `*` in technology names to match patterns

### Wildcard Patterns

Use wildcards to assign tags to multiple technologies at once. The `*` matches any characters.

```yaml
model_tag_values:
  THERM:
    Battery: 0  # Matches all battery technologies
    NaturalGas: 1  # Matches all natural gas technologies
    LandbasedWind: 0  # Matches all wind technologies
    OffShoreWind: 0  # More specific, can override 'Wind' if both present

  VRE:
    Battery: 0
    NaturalGas: 0
    LandbasedWind: 1
    OffShoreWind: 1

  STOR:
    Battery: 1
    NaturalGas: 0
    LandbasedWind: 0

  FLEX:
    Battery: 0
    NaturalGas: 0
    LandbasedWind: 0
```

**Substring precedence**: Shorter substrings are processed first, longer substrings override:

```yaml
model_tag_values:
  THERM:
    Wind: 0          # Applied first - matches 'LandbasedWind_Class3', 'OffShoreWind_Class12'
    OffShoreWind: 1  # Applied second - overwrites only offshore wind technologies
```

This makes configuration maintainable when you have many similar technologies.

## Regional Tag Overrides

### `regional_tag_values`

**Type**: Dictionary (region → tag → technology → value)
**Required**: No
**Example**: See below

Override tag values for specific technologies in specific regions. Regional values take precedence over `model_tag_values`.

```yaml
regional_tag_values:
  CA_N:
    THERM:
      NaturalGas: 2  # Different thermal zone in CA
    MUST_RUN:
      Hydroelectric: 1  # California hydro is must-run

  AZ:
    THERM:
      NaturalGas: 1  # Arizona thermal zone
    MUST_RUN:
      Hydroelectric: 0  # Arizona hydro is flexible
```

**Use cases**:

- Different thermal reserve zones by region
- Region-specific must-run requirements (e.g., cogeneration contracts)
- Policy differences (e.g., renewable portfolio standard eligibility)

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
  ESR_1:
    UtilityPV: 1  # Eligible for RPS
    NaturalGas_CCS: 0  # Not eligible for RPS
    Nuclear: 0  # Not eligible for RPS

  ESR_2:
    UtilityPV: 1  # Eligible for CES
    NaturalGas_CCS: 1  # May be eligible for CES (with CCS)
    Nuclear: 1  # Eligible for CES (carbon-free)
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
  MinCapTag_1:
    UtilityPV: 1  # Counts toward solar minimum
    NaturalGas: 0

  MaxCapTag_1:
    UtilityPV: 0
    NaturalGas: 1  # Counts toward gas maximum
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
  # Here, all matched resources contribute to the two different reserve requirements
  CapRes_1:
    NaturalGas: 0.9
    Battery: 0.95

  CapRes_2:
    NaturalGas: 0.9
    Battery: 0.9

############################
# *OR*
# Resources in specific regions contribute to different capacity reserve requirements
regional_tag_values:
  CA_N:
    CapRes_1:
      NaturalGas: 0.9
      Battery: 0.95
  AZ:
    CapRes_2:
      NaturalGas: 0.8
      Battery: 0.9
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
  THERM:
    NaturalGas: 1
    UtilityPV: 0
    Battery: 0

  VRE:
    NaturalGas: 0
    UtilityPV: 1
    Battery: 0

  STOR:
    NaturalGas: 0
    UtilityPV: 0
    Battery: 1

  FLEX:
    NaturalGas: 0
    UtilityPV: 0
    Battery: 0

  HYDRO:
    NaturalGas: 0
    UtilityPV: 0
    Battery: 0

  MUST_RUN:
    NaturalGas: 0
    UtilityPV: 0
    Battery: 0

  LDS:
    NaturalGas: 0
    UtilityPV: 0
    Battery: 0

  ESR_1:
    NaturalGas: 0
    UtilityPV: 1  # RPS-eligible
    Battery: 0

  CapRes_1:
    NaturalGas: 1
    UtilityPV: 1
    Battery: 1

# Regional overrides
regional_tag_values:
  CA_N:
    THERM:
      NaturalGas: 2  # Different thermal zone
  CA_S:
    THERM:
      NaturalGas: 2
```

## Related Settings

- [Model Definition](model-definition.md): `model_tag_names` defined here
- [Existing Generators](existing-generators.md): Tags assigned to clustered generators
- [New-Build Resources](new-build.md): Tags for candidate technologies
- [Regions](regions.md): Regional tag overrides by model region
