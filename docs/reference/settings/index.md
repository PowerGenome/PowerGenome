# Settings Reference

This section documents all configuration parameters that control PowerGenome's behavior.

Settings are defined in YAML files and loaded from a directory. PowerGenome recursively reads all `.yml` and `.yaml` files, with later files overriding earlier ones.

## Organization

Settings are grouped by functional area:

<div class="grid cards" markdown>

- :material-cube-outline: **[Model Definition](model-definition.md)**

    ---

    Core parameters defining planning years, currency conversion, and run scope.

    Parameters: `model_year`, `model_first_planning_year`, `model_tag_names`, etc.

  - :material-calendar-range: **[Year-Keyed Values](year-keyed-values.md)**

    ---

    Advanced pattern for resolving parameter values automatically by planning year.

    Works across settings sections, including nested values and cost modifiers.

- :material-map-marker-radius: **[Regional Configuration](regions.md)**

    ---

    Region definitions, aggregations, and base-to-model region mapping.

    Parameters: `model_regions`, `region_aggregations`, `cogen_tech`, etc.

- :material-factory: **[Existing Generators](existing-generators.md)**

    ---

    Clustering parameters, technology groupings, cost overrides.

    Parameters: `num_clusters`, `tech_groups`, `tech_fuel_map`, etc.

- :material-wind-turbine: **[New-Build Resources](new-build.md)**

    ---

    Technology costs, new resource definitions, regional availability, capacity limits.

    Parameters: `new_resources`, `resource_modifiers`, `modified_new_resources`, etc.

- :material-gas-station: **[Fuels](fuels.md)**

    ---

    Fuel price sources, emission factors, regional price adjustments.

    Parameters: `fuel_scenarios`, `fuel_price_table`, `fuel_emission_factors`, `tech_fuel_map`, etc.

    !!! tip "Simplified Workflow"
        PowerGenome now uses a simplified fuel workflow by default. Provide a complete fuel price table with all base regions instead of using legacy AEO mapping parameters.

- :material-transmission-tower: **[Transmission](transmission.md)**

    ---

    Network topology, line limits, expansion costs, loss percentages.

    Parameters: `transmission_constraints`, `tx_line_loss_pct`, `tx_expansion_per_mw`, etc.

- :material-power-plug-outline: **[Demand](demand.md)**

    ---

    Load profile sources, distributed generation, time weighting.

    Parameters: `load_zones`, `dg_capacity_table`, `dg_profiles_table`, etc.

- :material-clock-fast: **[Time Reduction](time-reduction.md)**

    ---

    Representative period selection, clustering methodology, time domain control.

    Parameters: `reduce_time_domain`, `time_domain_periods`, `include_peak_day`, etc.

- :material-tag-multiple: **[Resource Tags](resource-tags.md)**

    ---

    GenX-specific tags controlling dispatch, storage, flexibility, must-run status.

    Parameters: `regional_tag_values`, `model_tag_values`, `THERM`, `VRE`, `STOR`, etc.

- :material-file-tree: **[Data Tables](data-tables.md)**

    ---

    Input data source configuration with filtering and scenario selection.

    Parameters: `generation_table`, `demand_table`, `fuel_prices_table`, etc.

- :material-vector-combine: **[Scenario Management](scenario-management.md)**

    ---

    Multi-scenario execution, parameter swapping, case definitions.

    Parameters: `scenario_definitions_fn`, `settings_management`, `input_folder`, etc.

</div>

## Common Patterns

### Basic Value Types

```yaml
# Scalars
model_year: 2030
target_usd_year: 2023
reduce_time_domain: true

# Lists
model_regions: [CA_N, CA_S, AZ]
new_resources:
  - [NaturalGas, CCAvgCF, Moderate, 500]
  - [UtilityPV, Class1, Moderate, 100]

# Dictionaries
region_aggregations:
  CA_N: [CA_IID, CA_LADWP, CA_BANC]
  CA_S: [CA_SCE, CA_SDGE]
```

### Table Configurations

**Simple** (just filename):

```yaml
generation_table: "generators.csv"
```

**Advanced** (with filters):

```yaml
demand_table:
  table_name: demand_timeseries.parquet
  scenario: HighEV
  filters:
    - - [region, '=', 'CA']
      - [year, '>=', 2030]
  columns: [time_index, weather_year, region, load_mw, year]
```

Filter logic uses **DNF (Disjunctive Normal Form)**:

- Inner lists are AND conditions
- Outer list elements are OR'd together

Example: `filters: [[[A, =, 1], [B, =, 2]], [[C, =, 3]]]`
→ `(A=1 AND B=2) OR (C=3)`

### Multi-Level Dictionaries

Many settings use technology → parameter structure:

```yaml
tech_groups:
  Combined Cycle:
    - Natural Gas Fired Combined Cycle
    - Natural Gas Steam Turbine
  Coal:
    - Conventional Steam Coal
    - Coal Integrated Gasification Combined Cycle

num_clusters:
  CA_N:
    Conventional Steam Coal: 5
    Natural Gas Fired Combined Cycle: 10
```

Access pattern in code:

```python
combined_cycle_techs = settings["tech_groups"]["Combined Cycle"]
clusters = settings["num_clusters"]["CA_N"]["Conventional Steam Coal"]
```

### Regional Parameters

Regional settings can use model region names as keys:

```yaml
regional_tag_values:
  CA_N:
    THERM: 1
    VRE: 0
  AZ:
    THERM: 2
    VRE: 1

new_gen_not_available:
  CA_N: [Coal_new]
  AZ: [OffshoreWind]
```

### Parameter Modifiers

Cost and technology modifiers use nested dictionaries:

```yaml
resource_modifiers:
  solar_adjusted:
    technology: UtilityPV
    tech_detail: Class1
    capex_mw: [mul, 0.9]  # 10% cost reduction
    fixed_o_m_mw: [mul, 0.95]  # 5% O&M reduction
```

These apply changes to base technology costs using operators like `mul`, `add`, `sub`.

### Technology Matching

Technology names are matched **case-insensitively**:

```yaml
# These all match "naturalgas_ccavgcf_moderate" resource
tech_groups:
  Combined Cycle: [NaturalGas_CCAvgCF, naturalgas_ccavgcf]
```

Region names are **case-sensitive** and must match exactly.

## Loading Mechanism

PowerGenome loads settings from a **folder**, not individual files:

```python
from powergenome.settings import load_settings
from pathlib import Path

# Correct: point to folder
settings = load_settings(Path("my_study/settings"))

# Incorrect: don't point to specific file
# settings = load_settings(Path("my_study/settings/generators.yml"))
```

**Load Order**:

1. All `.yml` and `.yaml` files found recursively
2. Files sorted alphabetically
3. Later files override earlier ones
4. Nested dictionaries are merged (not replaced)

**Example structure**:

```
settings/
├── 01_model_definition.yml   # Loaded first
├── 02_regions.yml
├── 03_generators.yml
└── 04_overrides.yml           # Loaded last, takes precedence
```

## Validation

PowerGenome validates settings at runtime:

- **Missing required parameters**: Raises `KeyError`
- **Type mismatches**: May raise errors during execution
- **Invalid table names**: Logged as warnings
- **Missing data files**: Errors during DataManager initialization

Use `.get(key, default)` for optional parameters:

```python
# Required (error if missing)
year = settings["model_year"]

# Optional (use default)
reduce_time = settings.get("reduce_time_domain", False)
```

## Environment Variables (Deprecated)

Older versions used `.env` files for data paths:

```bash
# Legacy approach (still works but discouraged)
PUDL_DB=/path/to/pudl.sqlite
PG_DB=/path/to/powergenome.sqlite
```

**Modern approach** uses settings YAML:

```yaml
data_location: /path/to/data_folder
RESOURCE_GROUP_PROFILES: /path/to/generation_profiles
```

## Override Priority

When the same parameter appears in multiple places:

1. **Command-line arguments** (highest priority)
2. **Scenario-specific settings_management** swaps
3. **YAML files** (last file wins)
4. **Default values** in code (lowest priority)

## Next Steps

- [Model Definition Parameters](model-definition.md): Start here for core settings
- [Settings How-To Guide](../how-to/configure-settings.md): Practical examples
- [Scenario Management](scenario-management.md): Multi-scenario workflows
- [Data Tables](data-tables.md): Advanced table configuration
