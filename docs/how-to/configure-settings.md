# Configure Settings Files

This guide shows how to organize and structure settings files for PowerGenome projects.

## Settings Folder Structure

PowerGenome loads settings from a folder containing multiple YAML files:

```text
my_project/
├── settings/
│   ├── model_definition.yml
│   ├── regions.yml
│   ├── generators.yml
│   ├── new_build.yml
│   ├── fuels.yml
│   ├── transmission.yml
│   ├── demand.yml
│   ├── scenario_management.yml
│   └── data.yml
└── extra_inputs/
    └── emission_policies.csv
```

**Loading behavior**:

- A directory loads its `.yml` files; a single YAML file can also be loaded.
- Duplicate top-level keys replace the entire earlier value. File order is not
  guaranteed, so define each top-level setting in only one file.
- Split settings logically across files for maintainability.

## Starting from Examples

### Published Example Systems

The [PowerGenome-examples](https://github.com/PowerGenome/PowerGenome-examples)
repository contains complete studies whose settings you can copy and adapt:

- [TX-3-zone](https://github.com/PowerGenome/PowerGenome-examples/tree/7e34f714469e68fca173cdb849f74c47be779420/example_systems/TX-3-zone):
  three ERCOT model zones (`TX1`, `TX2`, `TX3`) aggregated from seven ReEDS
  balancing areas.
- [CA_AZ](https://github.com/PowerGenome/PowerGenome-examples/tree/7e34f714469e68fca173cdb849f74c47be779420/example_systems/CA_AZ):
  California (`CA`) and Arizona (`AZ`) aggregated from eight ReEDS balancing
  areas.

The Texas settings folder shows how the files in this guide fit together:

```text
example_systems/TX-3-zone/settings/
├── model_definition.yml   # regions, years, tags, output columns
├── resources.yml          # existing fleet, new build, renewable clusters
├── resource_tags.yml      # model_tag_names and model_tag_values
├── fuels.yml              # fuel prices, assignments, carbon tax
├── demand.yml
├── distributed_gen.yml
├── transmission.yml
├── startup_costs.yml
├── time_clustering.yml
├── data.yml               # source tables and data paths
├── extra_inputs.yml       # policy and scenario files
├── scenario_management.yml
└── env.yml
```

The CA/AZ study uses the same layout with different values. Both are small
demonstration datasets rather than calibrated studies, but each runs end to end
and is a working reference for every file described here. The
[Getting Started](../tutorials/getting-started.md#1-download-the-texas-example)
tutorial downloads a pinned snapshot and runs the Texas study.

### Copy Example System

Use the [PowerGenome System Design tool](https://gschivley.github.io/PowerGenome-tools/web/) to generate a starting settings folder for your study area. The tool walks you through selecting regions, technologies, and policies, then produces a settings folder you can download and modify.

See the [System Design tool documentation](https://gschivley.github.io/PowerGenome-tools/) for a walkthrough of each step.

### Minimal Settings

These fragments illustrate the structure; they are not a complete runnable
configuration. Start with a generated configuration or a published example for
the required tables and other settings.

**settings/model_definition.yml**:

```yaml
# Planning horizon
model_periods: [[2026, 2030]]

# Currency
target_usd_year: 2023

# Alternatively, define the planning horizon using model_year and model_first_planning_year:
# model_year: [2030]
# model_first_planning_year: 2030

# Data location
data_location: /path/to/data

# Model tags
model_tag_names:
  - THERM
  - VRE
  - STOR

model_tag_values:
  THERM:
    NaturalGas: 1
  VRE:
    UtilityPV: 1
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

**model_definition.yml**: Planning years, model regions and aggregations, tags, execution control
**regions.yml**: Regional configuration, aggregations, reserves
**generators.yml**: Existing generator clustering, retirement
**new_build.yml**: New-build technologies, costs, renewable clusters
**fuels.yml**: Fuel prices, emissions, carbon pricing
**transmission.yml**: Network constraints, expansion costs
**demand.yml**: Load profiles, growth, distributed generation
**scenario_management.yml**: Multi-scenario definitions

How you group these is a project choice. Both published examples keep existing
generators, new-build resources, and renewable clusters in `resources.yml`, and
model tags in `resource_tags.yml`.

### By Model Year

For multi-period models with year-specific parameters, you now have two options:

- **Year-keyed values**: Best for by-year changes without full scenario matrices.
- **`settings_management`**: Best for multi-case studies with scenario dimensions.

### Advanced: Year-Keyed Values Across Any Settings Section

Year-keyed values are an advanced pattern that can be used for any
settings parameter value (including nested values):

```yaml
carbon_tax:
  2030: 25
  2040: 50
```

Resolution behavior:

1. Exact year match
2. `default` fallback

If a year-keyed dictionary does not cover all planning years and has no
fallback, PowerGenome raises a validation error.

See [Use Year-Keyed Settings Values](use-year-keyed-settings.md) for
step-by-step usage and
[Year-Keyed Values (Reference)](../reference/settings/year-keyed-values.md)
for full rules.

### Scenario Management For Multi-Case Workflows

Use `settings_management` when you need parameter swaps across
multiple scenario dimensions:

```text
settings/
├── base.yml                    # Core configuration
├── regions.yml                 # Regional definition
└── scenario_management.yml     # Year-specific overrides
extra_inputs/
└── scenario_definitions.csv    # Case definitions
```

**base.yml**:

```yaml
# Planning periods
model_periods:
  - [2020, 2030]
  - [2031, 2040]
  - [2041, 2050]
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
        carbon_tax: 100
      low:
        carbon_tax: 25
  2030:
    all_cases:  # Applied to ALL scenarios in 2030
      weather_year: 2012
  2040:
    all_cases:  # Applied to ALL scenarios in 2040
      weather_year: 2012
  2050:
    all_cases:  # Applied to ALL scenarios in 2050
      weather_year: 2012

scenario_definitions_fn: scenario_definitions.csv
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

- `all_cases` selects the weather year for each planning year
- `policy` varies carbon tax across scenarios
- Both are applied together for each case/year combination

### By Scenario

For scenario studies, use `settings_management`:

```text
settings/
├── base.yml                    # Base configuration
└── settings_management.yml     # Parameter swaps
extra_inputs/
└── scenario_definitions.csv    # Scenario matrix
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

```

Relative `data_location` and scalar `input_folder` paths resolve from the
parent of the settings directory (or the directory containing a settings file).
Resource group paths are relative to the process working directory.

`RESOURCE_GROUPS` and `RESOURCE_GROUP_PROFILES` accept **either a single path or a
list of paths**, so data can be split across multiple folders:

```yaml
RESOURCE_GROUPS:
  - /Users/me/resource_groups/onshore
  - /Users/me/resource_groups/offshore
RESOURCE_GROUP_PROFILES:
  - /Users/me/nrel_profiles/wind
  - /Users/me/nrel_profiles/solar
```

**settings/.gitignore**:

```text
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

Avoid defining the same top-level parameter in multiple files. For example,
these two files conflict:

**settings/01_base.yml**:

```yaml
resource_modifiers:
  utilitypv:
    technology: UtilityPV
    tech_detail: Class1
    capex_mw: [mul, 1.0]
carbon_tax: 25
```

**settings/02_overrides.yml**:

```yaml
resource_modifiers:
  utilitypv:
    capex_mw: [mul, 0.9]
```

The whole `resource_modifiers` dictionary is replaced, not recursively merged;
the surviving entry lacks the required `technology` and `tech_detail` keys.
Since directory iteration order is not guaranteed, do not rely on either file
winning. Keep all entries for a top-level key together.

### List Merging

Lists are **replaced**, not merged:

**settings/technologies.yml**:

```yaml
new_resources:
  - [NaturalGas, 2-on-1 Combined Cycle (F-Frame), Moderate, 500]
  - [UtilityPV, Class1, Moderate, 100]
  - [LandbasedWind, Class3, Moderate, 100]
```

**settings/more_technologies.yml**:

```yaml
new_resources:
  - [Utility-Scale Battery Storage, Lithium Ion, Moderate, 100]
```

**Result**: One entire list replaces the other; which one wins is not guaranteed.

**Solution**: Keep lists in one file or use complete lists in later files:

```yaml
new_resources:
  - [NaturalGas, 2-on-1 Combined Cycle (F-Frame), Moderate, 500]
  - [UtilityPV, Class1, Moderate, 100]
  - [LandbasedWind, Class3, Moderate, 100]
  - [Utility-Scale Battery Storage, Lithium Ion, Moderate, 100]
```

### Dictionary Merging

Nested dictionaries are not merged across files:

**settings/tags.yml**:

```yaml
model_tag_values:
  THERM:
    NaturalGas: 1
```

**settings/more_tags.yml**:

```yaml
model_tag_values:
  VRE:
    UtilityPV: 1
```

**Result**: One entire `model_tag_values` dictionary replaces the other. Combine
the tag entries in a single file instead. The Python `Settings.update()` and
`Settings.load_settings()` methods also update top-level keys rather than
deep-merging them.

## Validation

### Check Settings Load

Verify settings load correctly:

```python
from powergenome.settings import load_settings
from pathlib import Path

settings = load_settings(Path("settings"))
print(settings.get("model_periods"))
print(settings.get("model_regions"))
```

### Validate Required Parameters

Check for settings used in the example above:

```python
from powergenome.settings import load_settings
from pathlib import Path

settings = load_settings(Path("settings"))

required = [
    "model_periods",
    "target_usd_year",
    "model_regions",
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
from pathlib import Path

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
regional_cost_factor_table: regional_multipliers.csv
cost_multiplier_region_map:
  PCA_WECC: [CA_N, CA_S, AZ]
```

**extra_inputs/regional_multipliers.csv**:

```csv
technology,region,value
UtilityPV_Class1,PCA_WECC,1.2
LandbasedWind_Class3,PCA_WECC,1.15
NaturalGas_2-on-1 Combined Cycle (F-Frame),PCA_WECC,1.1
```

### Technology Restrictions

Prohibit technologies in specific regions:

!!! warning "Legacy setting — not applied"
    `new_gen_not_available` is only validated for region-name consistency; it is **not**
    applied to exclude technologies. Non-renewable `new_resources` are built in
    every model region; renewable resource clusters are scoped per region through
    `renewables_clusters`.

**settings/tech_availability.yml**:

```yaml
# Legacy — not applied in this version
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
  utility_pv:
    technology: UtilityPV
    tech_detail: Class1
    capex_mw:
      2030: [mul, 1.1]  # 10% higher than source data
    Fixed_OM_Cost_per_MWyr:
      2030: 15000  # Override source value

modified_new_resources:
  utility_pv_high:
    technology: UtilityPV
    tech_detail: Class1
    cost_case: Moderate
    size_mw: 100
    new_technology: UtilityPV
    new_tech_detail: Class1_High
    new_cost_case: High
    capex_mw:
      2030: [mul, 1.3]
```

Add entries for every planning year or a `default` to the year-keyed values.

### Emission Policies

Configure renewable portfolio standards:

**settings/policies.yml**:

```yaml
emission_policies_table: emission_policies.csv
```

See [Energy Share Requirements](energy-share-requirements.md) for the policy
table format.

## Troubleshooting

### Settings Not Loading

**Problem**: Settings folder not recognized

**Solution**: Pass an existing settings directory or YAML file:

```bash
# Correct
run_powergenome --settings_file settings --results_folder results

# Also supported: a single YAML file
run_powergenome --settings_file settings/model.yml --results_folder results
```

### Parameter Not Found

**Problem**: `KeyError: 'model_periods'`

**Solution**: Check parameter spelling and ensure it's in one of the YAML files:

```bash
# Search for parameter in settings
grep -r "model_periods" settings/
```

### Override Not Working

**Problem**: Parameter value not updating from later file

**Solution**: Check for duplicate top-level keys and combine their values in one
file. Renaming files does not establish a guaranteed override order.

### Path Not Found

**Problem**: `FileNotFoundError: [Errno 2] No such file or directory: '/path/to/data'`

**Solution**: Use absolute paths in `data_location`:

```yaml
# Use absolute path
data_location: /Users/me/powergenome_data

```

The loader does not expand `~`; use the full home-directory path.

## Best Practices

1. **Split logically**: Organize by functional area (generators, transmission, etc.)
2. **Use data.yml**: Keep machine-specific paths out of version control
3. **Comment generously**: Explain non-obvious parameter choices
4. **Version control**: Commit settings folder (except data.yml) to Git
5. **Test incrementally**: Validate after major changes
6. **Document assumptions**: Note data sources, policy assumptions, cost adjustments
7. **Start from a generated configuration**: Use the [System Design tool](https://gschivley.github.io/PowerGenome-tools/web/) to produce a working settings folder for your region

## Next Steps

- [Add Custom Technologies](add-technologies.md): Define new resources
- [Configure Data Tables](configure-data-tables.md): Set up table loading
- [Run Multiple Scenarios](run-scenarios.md): Multi-scenario workflows
- [Settings Reference](../reference/settings/index.md): Complete parameter documentation
