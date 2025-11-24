# Configure Settings Files

This guide shows how to organize and structure settings files for PowerGenome projects.

## Settings Folder Structure

PowerGenome loads settings from a folder containing multiple YAML files:

```
my_project/
├── settings/
│   ├── model_definition.yml
│   ├── regions.yml
│   ├── generators.yml
│   ├── new_build.yml
│   ├── fuels.yml
│   ├── transmission.yml
│   ├── demand.yml
│   ├── time_reduction.yml
│   ├── scenario_management.yml
│   └── data.yml
└── extra_inputs/
    └── emission_policies.csv
```

**Loading behavior**:

- Files load in alphabetical order
- Later files override earlier ones
- Split settings logically across files for maintainability

## Starting from Examples

### Copy Example System

Start with a working example and modify it:

```bash
# Navigate to PowerGenome repository
cd PowerGenome/example_systems

# Copy example that matches your region
cp -r CA_AZ ~/my_project/settings

# Or for multi-zone systems
cp -r WECC-6-zone ~/my_project/settings
```

Available examples:

- **CA_AZ**: 2-region California + Arizona
- **CONUS-3-zone**: 3-region Continental US
- **WECC-6-zone**: 6-region Western Interconnection
- **ISONE**: ISO New England

### Minimal Settings

Create minimal settings from scratch:

**settings/model_definition.yml**:

```yaml
# Planning horizon
model_year: [2030]
model_first_planning_year: 2030
model_periods: [1]

# Currency
target_usd_year: 2023

# Data location
data_location: /path/to/data

# Model tags
model_tag_names:
  - THERM
  - VRE
  - STOR

model_tag_values:
  NaturalGas_CCAvgCF_Moderate:
    THERM: 1
    VRE: 0
    STOR: 0
  UtilityPV_Class1_Moderate:
    THERM: 0
    VRE: 1
    STOR: 0
```

**settings/regions.yml**:

```yaml
# Model regions
model_regions:
  - CA_N
  - CA_S

# Region aggregations (if using base regions)
region_aggregations:
  CA_N: [CA_N]
  CA_S: [CA_S]
```

## File Organization Strategies

### By Category (Recommended)

Split settings by functional area:

**model_definition.yml**: Planning years, tags, execution control
**regions.yml**: Regional configuration, aggregations, reserves
**generators.yml**: Existing generator clustering, retirement
**new_build.yml**: New-build technologies, costs, renewable clusters
**fuels.yml**: Fuel prices, emissions, carbon pricing
**transmission.yml**: Network constraints, expansion costs
**demand.yml**: Load profiles, growth, distributed generation
**time_reduction.yml**: Representative period configuration
**scenario_management.yml**: Multi-scenario definitions

### By Model Year

For multi-period models with year-specific parameters, use `settings_management`:

```text
settings/
├── base.yml                    # Core configuration
├── regions.yml                 # Regional definition
├── scenario_management.yml     # Year-specific overrides
└── scenario_definitions.csv    # Case definitions
```

**base.yml**:

```yaml
# Planning periods
model_year: [2030, 2040, 2050]
model_first_planning_year: [2020, 2031, 2041]
target_usd_year: 2023

model_regions:
  - CA_N
  - CA_S
```

**scenario_management.yml**:

```yaml
settings_management:
  all_years:
    policy:  # Column name from scenario_definitions.csv
      high:
        minimum_cap_req_fn: high_res_policy.csv
      low:
        minimum_cap_req_fn: low_res_policy.csv
  2030:
    all_cases:  # Applied to ALL scenarios in 2030
      carbon_tax: 50
      default_load_multiplier: 1.0
  2040:
    all_cases:  # Applied to ALL scenarios in 2040
      carbon_tax: 75
      default_load_multiplier: 1.15
  2050:
    all_cases:  # Applied to ALL scenarios in 2050
      carbon_tax: 100
      default_load_multiplier: 1.30
```

**scenario_definitions.csv**:

```csv
case_id,year,policy
high_policy,2030,high
high_policy,2040,high
high_policy,2050,high
low_policy,2030,low
low_policy,2040,low
low_policy,2050,low
```

In this example:

- `all_cases` sets carbon tax and load growth for each year
- `policy` column varies RES policy files across scenarios
- Both are applied together for each case/year combination

### By Scenario

For scenario studies, use `settings_management`:

```texttext
settings/
├── base.yml                    # Base configuration
├── scenario_definitions.csv    # Scenario matrix
└── settings_management.yml     # Parameter swaps
```

See [Run Multiple Scenarios](run-scenarios.md) for details.

## Environment-Specific Settings

### Local Paths

Create `data.yml` for machine-specific paths (add to `.gitignore`):

**settings/data.yml**:

```yaml
# Data paths (local machine)
data_location: /Users/me/powergenome_data
RESOURCE_GROUP_PROFILES: /Users/me/nrel_profiles
input_folder: extra_inputs

# Optional legacy paths
PUDL_DB: /Users/me/databases/pudl.db
EIA_AEO_YEAR: 2023
```

**settings/.gitignore**:

```
data.yml
```

Team members create their own `data.yml` without committing it.

### Shared Paths

For shared data locations (HPC cluster, cloud storage), commit paths:

**settings/data_paths.yml**:

```yaml
# Shared data paths (committed to repo)
data_location: /shared/powergenome/v2_data
RESOURCE_GROUP_PROFILES: /shared/nrel/generation_profiles
```

## Override Patterns

### Parameter Override

Later files override earlier values:

**settings/01_base.yml**:

```yaml
atb_cost_case: Moderate
default_growth_rate: 0.01
```

**settings/02_overrides.yml**:

```yaml
atb_cost_case: Advanced  # Overrides Moderate
# default_growth_rate: 0.01 remains unchanged
```

### List Merging

Lists are **replaced**, not merged:

**settings/technologies.yml**:

```yaml
new_resources:
  - [NaturalGas, CCAvgCF, Moderate, 500]
  - [UtilityPV, Class1, Moderate, 100]
  - [LandbasedWind, Class3, Moderate, 100]
```

**settings/more_technologies.yml**:

```yaml
new_resources:
  - [Battery, "*", Moderate, 100]  # Replaces entire list above!
```

**Result**: Only `Battery` is in `new_resources`.

**Solution**: Keep lists in one file or use complete lists in later files:

```yaml
new_resources:
  - [NaturalGas, CCAvgCF, Moderate, 500]
  - [UtilityPV, Class1, Moderate, 100]
  - [LandbasedWind, Class3, Moderate, 100]
  - [Battery, "*", Moderate, 100]  # Complete list
```

### Dictionary Merging

Nested dictionaries merge recursively:

**settings/tags.yml**:

```yaml
model_tag_values:
  NaturalGas_CCAvgCF_Moderate:
    THERM: 1
    VRE: 0
```

**settings/more_tags.yml**:

```yaml
model_tag_values:
  UtilityPV_Class1_Moderate:  # Adds to model_tag_values
    THERM: 0
    VRE: 1
```

**Result**: Both technologies are in `model_tag_values`.

## Validation

### Check Settings Load

Verify settings load correctly:

```python
from powergenome.settings import load_settings
from pathlib import Path

settings = load_settings(Path("settings"))
print(settings.get("model_year"))
print(settings.get("model_regions"))
```

### Validate Required Parameters

Check for missing required parameters:

```python
from powergenome.settings import load_settings

settings = load_settings(Path("settings"))

required = [
    "model_year",
    "model_first_planning_year",
    "target_usd_year",
    "model_regions",
    "model_tag_names",
]

missing = [p for p in required if p not in settings]
if missing:
    print(f"Missing required parameters: {missing}")
```

### Test Data Loading

Verify data tables are accessible:

```python
from powergenome.settings import load_settings
from powergenome.database import initialize_data_manager

settings = load_settings(Path("settings"))
data_location = settings.get("data_location")

initialize_data_manager(settings, data_location)

# Test table loading
from powergenome.database import get_data_manager
dm = get_data_manager()
print(dm.list_tables())
```

## Common Patterns

### Regional Cost Multipliers

Apply regional construction cost differences:

**settings/regional_costs.yml**:

```yaml
cost_multiplier_fn: regional_multipliers.csv
cost_multiplier_region_map:
  CA_N: PCA_WECC
  CA_S: PCA_WECC
  AZ: PCA_WECC
```

**extra_inputs/regional_multipliers.csv**:

```csv
region,technology,multiplier
PCA_WECC,UtilityPV,1.2
PCA_WECC,LandbasedWind,1.15
PCA_WECC,NaturalGas_CC,1.1
```

### Technology Restrictions

Prohibit technologies in specific regions:

**settings/tech_availability.yml**:

```yaml
new_gen_not_available:
  AZ:
    - OffshoreWind_*  # No offshore wind in Arizona
  CA_N:
    - Coal_*  # No new coal in California
```

### Custom Technology Costs

Modify costs for specific technologies:

**settings/custom_costs.yml**:

```yaml
resource_modifiers:
  UtilityPV_Class1_Moderate:
    capex_mw:
      2030: 1.1  # 10% higher than ATB
    fixed_o_m_mw:
      2030: 15000  # Override ATB value

modified_new_resources:
  UtilityPV_Class1_High:  # Create new variant
    base_resource: UtilityPV_Class1_Moderate
    capex_mw:
      2030: 1.3  # 30% higher
```

### Emission Policies

Configure renewable portfolio standards:

**settings/policies.yml**:

```yaml
emission_policies_fn: emission_policies.csv
energy_share_req_fn: energy_share_requirements.csv
```

**extra_inputs/emission_policies.csv**:

```csv
case_id,year,region,RPS,CES,CO2_cap
baseline,2030,all,0.5,0.8,
baseline,2040,all,0.7,0.9,
baseline,2050,all,0.9,1.0,
```

## Troubleshooting

### Settings Not Loading

**Problem**: Settings folder not recognized

**Solution**: Pass folder path, not individual file:

```bash
# Correct
run_powergenome --settings_file settings --results_folder results

# Incorrect
run_powergenome --settings_file settings/model.yml --results_folder results
```

### Parameter Not Found

**Problem**: `KeyError: 'model_year'`

**Solution**: Check parameter spelling and ensure it's in one of the YAML files:

```bash
# Search for parameter in settings
grep -r "model_year" settings/
```

### Override Not Working

**Problem**: Parameter value not updating from later file

**Solution**: Check file alphabetical order:

```bash
# Files load alphabetically
ls settings/
# 01_base.yml loads before 02_overrides.yml
```

Rename files to control load order if needed.

### Path Not Found

**Problem**: `FileNotFoundError: [Errno 2] No such file or directory: '/path/to/data'`

**Solution**: Use absolute paths in `data_location`:

```yaml
# Use absolute path
data_location: /Users/me/powergenome_data

# Or expand home directory
data_location: ~/powergenome_data
```

## Best Practices

1. **Split logically**: Organize by functional area (generators, transmission, etc.)
2. **Use data.yml**: Keep machine-specific paths out of version control
3. **Comment generously**: Explain non-obvious parameter choices
4. **Version control**: Commit settings folder (except data.yml) to Git
5. **Test incrementally**: Validate after major changes
6. **Document assumptions**: Note data sources, policy assumptions, cost adjustments
7. **Use examples**: Start from working example_systems configurations

## Next Steps

- [Add Custom Technologies](add-technologies.md): Define new resources
- [Configure Data Tables](configure-data-tables.md): Set up table loading
- [Run Multiple Scenarios](run-scenarios.md): Multi-scenario workflows
- [Settings Reference](../reference/settings/index.md): Complete parameter documentation
