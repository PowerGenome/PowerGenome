# Run Multiple Scenarios

This guide shows how to configure and execute multiple scenarios in a single PowerGenome run, enabling sensitivity analyses, policy comparisons, and uncertainty exploration.

## Scenario Workflow Overview

1. **Define scenarios** in CSV file (parameter combinations)
2. **Configure parameter swaps** in settings (what changes per scenario)
3. **Run batch execution** with command-line tool
4. **Outputs** saved to separate folders per case

## Basic Setup

### 1. Create Scenario Definitions

Create a CSV file listing all scenario variations:

**scenario_definitions.csv**:

```csv
case_id,year,cost_scenario,carbon_policy
baseline,2030,mid,none
low_cost,2030,low,none
high_carbon,2030,mid,aggressive
low_cost_carbon,2030,low,aggressive
baseline,2040,mid,none
low_cost,2040,low,none
```

**Required columns**:

- `case_id`: Unique scenario identifier
- `year`: Model year (must match `model_year` in settings)

**User-defined columns**: Any parameter dimensions you want to vary

### 2. Configure Settings Management

Define what changes for each parameter value:

**settings/scenario_management.yml**:

```yaml
scenario_definitions_fn: scenario_definitions.csv

settings_management:
  2030:
    cost_scenario:
      low:
        new_resources:
          - [NaturalGas, CCAvgCF, Advanced, 500]
          - [UtilityPV, Class1, Advanced, 100]
          - [LandbasedWind, Class3, Advanced, 100]
      mid:
        new_resources:
          - [NaturalGas, CCAvgCF, Moderate, 500]
          - [UtilityPV, Class1, Moderate, 100]
          - [LandbasedWind, Class3, Moderate, 100]
      high:
        new_resources:
          - [NaturalGas, CCAvgCF, Conservative, 500]
          - [UtilityPV, Class1, Conservative, 100]
          - [LandbasedWind, Class3, Conservative, 100]

    carbon_policy:
      none:
        carbon_tax: 0
      moderate:
        carbon_tax: 50
      aggressive:
        carbon_tax: 100

  2040:
    cost_scenario:
      low:
        new_resources:
          - [NaturalGas, CCAvgCF, Advanced, 500]
          - [UtilityPV, Class1, Advanced, 100]
          - [LandbasedWind, Class3, Advanced, 100]
      mid:
        new_resources:
          - [NaturalGas, CCAvgCF, Moderate, 500]
          - [UtilityPV, Class1, Moderate, 100]
          - [LandbasedWind, Class3, Moderate, 100]
      high:
        new_resources:
          - [NaturalGas, CCAvgCF, Conservative, 500]
          - [UtilityPV, Class1, Conservative, 100]
          - [LandbasedWind, Class3, Conservative, 100]

    carbon_policy:
      none:
        carbon_tax: 0
      moderate:
        carbon_tax: 75
      aggressive:
        carbon_tax: 150
```

!!! note "Selecting the ATB cost case"
    The ATB cost trajectory (`Advanced`, `Moderate`, `Conservative`) is not a
    standalone setting. It is embedded in the `cost_case` element of each
    `new_resources` entry (`[technology, tech_detail, cost_case, size_mw]`), so a
    `settings_management` swap must provide the full `new_resources` list for the
    desired cost case. For percentage-based cost changes, use `resource_modifiers`
    with `[mul, value]` instead (see [Technology Costs](#technology-costs)).

### 3. Run Multi-Scenario Execution

```bash
run_powergenome \
  --settings_file settings \
  --results_folder results
```

**Output structure**:

```
results/
├── baseline_2030/
│   ├── Generators_data.csv
│   ├── Load_data.csv
│   └── ...
├── low_cost_2030/
├── high_carbon_2030/
├── low_cost_carbon_2030/
├── baseline_2040/
└── low_cost_2040/
```

To also write a MacroEnergy.jl `simpleCSVinputs` case for each scenario, add `--macro` (or set
`macro_output: true` in settings). GenX inputs are still written by default, so one run produces both formats —
reusing the intermediate data processing is faster than running PowerGenome once per model. Use `--no-genx` for
Macro-only output:

```bash
run_powergenome \
  --settings_file settings \
  --results_folder results \
  --macro --no-genx
```

See [Macro Output Settings](../reference/settings/macro-output.md) for the full set of Macro output options.

## Parameter Swap Examples

### Technology Costs

Vary technology costs across scenarios:

```yaml
settings_management:
  2030:
    solar_cost:
      low:
        resource_modifiers:
          solar:
            technology: UtilityPV
            tech_detail: Class1
            capex_mw: [mul, 0.8]
      mid: {}  # No change from baseline
      high:
        resource_modifiers:
          solar:
            technology: UtilityPV
            tech_detail: Class1
            capex_mw: [mul, 1.3]

    battery_cost:
      low:
        resource_modifiers:
          batteries:
            technology: Battery
            tech_detail: "*"
            capex_mw: [mul, 0.7]
            capex_mwh: [mul, 0.7]
      mid: {}
      high:
        resource_modifiers:
          batteries:
            technology: Battery
            tech_detail: "*"
            capex_mw: [mul, 1.4]
            capex_mwh: [mul, 1.4]
```

!!! info "`resource_modifiers` format"
    Each top-level key is a short name you choose; the nested dict **must** contain
    `technology` and `tech_detail` keys that match a resource in `new_resources`.
    Parameter values use `[operator, value]` (operators: `add`, `mul`, `sub`,
    `truediv`) or a plain number to set an absolute value. Values can be
    year-keyed (e.g. `capex_mw: {2030: [mul, 0.8], 2040: [mul, 0.9]}`) to vary
    across planning years.

### Fuel Prices

Vary fuel price scenarios:

```yaml
settings_management:
  2030:
    gas_price:
      low:
        fuel_scenarios:
          naturalgas: low_price
      reference:
        fuel_scenarios:
          naturalgas: reference
      high:
        fuel_scenarios:
          naturalgas: high_price
```

### Demand Growth

Vary electrification assumptions:

```yaml
settings_management:
  2030:
    demand:
      low:
        growth_scenario: REF2020
        alt_growth_rate:
          CA_N: 0.005
          CA_S: 0.005
      mid:
        growth_scenario: REF2020
        alt_growth_rate:
          CA_N: 0.015
          CA_S: 0.015
      high:
        growth_scenario: REF2020
        alt_growth_rate:
          CA_N: 0.03
          CA_S: 0.03
```

!!! note "Demand growth and distributed generation"
    Load growth is set with `growth_scenario` (an EIA AEO scenario code, e.g. `REF2020`)
    plus the optional `alt_growth_rate` override (a per-region rate or a per-region/sector
    dict). There is no `default_growth_rate` setting. Distributed generation capacity is
    no longer set through a settings value — it comes from the DataManager
    (`distributed_capacity_table` / `distributed_profiles_table`). See
    [Distributed Generation](../explanation/distributed-generation.md).

### Technology Availability

Enable/disable technologies:

```yaml
settings_management:
  2030:
    nuclear:
      allowed:
        new_resources:
          - [NaturalGas, CCAvgCF, Moderate, 500]
          - [UtilityPV, Class1, Moderate, 100]
          - [LandbasedWind, Class3, Moderate, 100]
          - [Battery, "*", Moderate, 100]
          - [Nuclear, Nuclear - Large, Moderate, 1000]
      prohibited:
        # Omit Nuclear from the resource list so it is not available as a candidate
        new_resources:
          - [NaturalGas, CCAvgCF, Moderate, 500]
          - [UtilityPV, Class1, Moderate, 100]
          - [LandbasedWind, Class3, Moderate, 100]
          - [Battery, "*", Moderate, 100]

    ccs:
      available:
        new_resources:
          - [NaturalGas, CCAvgCF, Moderate, 500]
          - [NaturalGas, CCCCSAvgCF, Conservative, 500]
          - [Coal, CCS90AvgCF, Moderate, 500]
      unavailable:
        new_resources:
          - [NaturalGas, CCAvgCF, Moderate, 500]
          - [Coal, CCS90AvgCF, Moderate, 500]
```

!!! note "How resource availability is controlled"
    A new-build resource is only included if it appears in `new_resources` (or is
    produced by `renewables_clusters`). To disable a technology for a scenario, leave
    it out of `new_resources` — there is no separate "exclude" list. Note that
    `new_resources` is not region-scoped: omitting a technology removes it from **every**
    region. The legacy `new_gen_not_available` key is not applied to exclude resources
    (it is only checked for region-name consistency during validation), and `ALL_REGIONS`
    is not a recognized region key.

### Retirement Assumptions

Existing-generator retirements are driven by the `retirement_year` column in the
generation input data (see
[Existing Generators](../reference/settings/existing-generators.md)) — there is no
`retirement_ages` setting, that code path is no longer used. To vary retirements
across scenarios, point each scenario at its own generation data (per-scenario input
tables or filters) rather than swapping a settings value, e.g.:

```yaml
settings_management:
  2030:
    coal_retirement:
      early:
        generation_table:
          table_name: generation_early_retirement.parquet
```

### Transmission Expansion

Vary transmission constraints:

```yaml
settings_management:
  2030:
    transmission:
      limited:
        tx_expansion_per_period: 0.0
        tx_expansion_mw_per_period: 500
      baseline:
        tx_expansion_per_period: 1.0
        tx_expansion_mw_per_period: 1000
      unlimited:
        tx_expansion_per_period: 10.0
        tx_expansion_mw_per_period: 10000
```

!!! note "Transmission expansion settings"
    Intertie expansion is set with `tx_expansion_per_period` (fraction of existing
    capacity that may be added, e.g. `1.0` doubles it) and `tx_expansion_mw_per_period`
    (fixed MW cap); the larger of the two governs each line. There is no
    `max_network_reinforcement_mw`, `tx_expansion_per_mw`, or `enforce_constraints`
    transmission setting. See [Transmission Settings](../reference/settings/transmission.md).

## Multi-Dimensional Scenarios

### Scenario Matrix

Create scenarios varying multiple dimensions:

**scenario_definitions.csv**:

```csv
case_id,year,tech_cost,fuel_price,carbon,demand
base_ref_none_mid,2030,mid,reference,none,mid
lowtech_ref_none_mid,2030,low,reference,none,mid
base_high_none_mid,2030,mid,high,none,mid
base_ref_50_mid,2030,mid,reference,50,mid
base_ref_none_high,2030,mid,reference,none,high
lowtech_low_100_high,2030,low,low,100,high
```

This creates a partial factorial design exploring combinations of interest.

### Full Factorial Design

Generate all combinations programmatically:

```python
import pandas as pd
import itertools

# Define dimensions
years = [2030, 2040]
tech_costs = ['low', 'mid', 'high']
fuel_prices = ['low', 'reference', 'high']
carbon_policies = ['none', '50', '100']

# Generate all combinations
combinations = itertools.product(years, tech_costs, fuel_prices, carbon_policies)

# Create dataframe
scenarios = []
for year, tech, fuel, carbon in combinations:
    case_id = f"tech{tech}_fuel{fuel}_carbon{carbon}_{year}"
    scenarios.append({
        'case_id': case_id,
        'year': year,
        'tech_cost': tech,
        'fuel_price': fuel,
        'carbon': carbon
    })

df = pd.DataFrame(scenarios)
df.to_csv('scenario_definitions.csv', index=False)
```

This generates 54 scenarios (2 years × 3 tech costs × 3 fuel prices × 3 carbon policies).

## Running Scenarios Sequentially

PowerGenome runs the cases in `scenario_definitions.csv` in the order they appear,
one after another (there is no `--num_workers`/parallel option — renewable
*clustering* has an internal `clustering_n_jobs` setting, but scenario
execution is sequential). To run only a subset, use `--case-id`:

```bash
run_powergenome \
  --settings_file settings \
  --results_folder results \
  --case-id baseline low_cost
```

**Performance**:

- Each case is run in the order listed in the scenario file
- To parallelize, launch separate PowerGenome processes (one per group of cases) or
  use `--case-id` to distribute cases across machines
- Monitor memory/disk usage; large models take significant disk per case

### Check Progress

Monitor which scenarios are running:

```bash
# In separate terminal
tail -f results/*/powergenome.log
```

Or use process monitoring:

```bash
# Count running PowerGenome processes
ps aux | grep powergenome | wc -l
```

## Multi-Period Scenarios

### Configure Multi-Period Model

For myopic or perfect foresight multi-period models:

**scenario_definitions.csv**:

```csv
case_id,year,tech_cost,carbon
baseline,2030,mid,50
baseline,2040,mid,75
baseline,2050,mid,100
high_tech,2030,low,50
high_tech,2040,low,75
high_tech,2050,low,100
```

**Requirements**:

- Each `case_id` must have entry for every model year
- Number of rows = (# unique case_ids) × (# model years)

**Settings**:

```yaml
model_periods: [[2026, 2030], [2031, 2040], [2041, 2050]]

settings_management:
  2030:
    tech_cost:
      low:
        resource_modifiers:
          solar:
            technology: UtilityPV
            tech_detail: Class1
            capex_mw: [mul, 0.9]
      mid:
        resource_modifiers:
          solar:
            technology: UtilityPV
            tech_detail: Class1
            capex_mw: [mul, 1.0]
    carbon:
      50:
        carbon_tax: 50

  2040:
    tech_cost:
      low:
        resource_modifiers:
          solar:
            technology: UtilityPV
            tech_detail: Class1
            capex_mw: [mul, 0.9]
      mid:
        resource_modifiers:
          solar:
            technology: UtilityPV
            tech_detail: Class1
            capex_mw: [mul, 1.0]
    carbon:
      75:
        carbon_tax: 75

  2050:
    tech_cost:
      low:
        resource_modifiers:
          solar:
            technology: UtilityPV
            tech_detail: Class1
            capex_mw: [mul, 0.9]
      mid:
        resource_modifiers:
          solar:
            technology: UtilityPV
            tech_detail: Class1
            capex_mw: [mul, 1.0]
    carbon:
      100:
        carbon_tax: 100
```

## Advanced Techniques

### Nested Parameter Changes

Modify multiple nested settings:

```yaml
settings_management:
  2030:
    renewable_scenario:
      high:
        # Multiple changes for high renewable scenario
        resource_modifiers:
          solar:
            technology: UtilityPV
            tech_detail: Class1
            capex_mw: [mul, 0.8]
          wind:
            technology: LandbasedWind
            tech_detail: Class3
            capex_mw: [mul, 0.85]
        renewables_clusters:
          - region: CA_N
            technology: utilitypv
            filter:
              - feature: lcoe
                max: 50
            cluster:
              - feature: [longitude, latitude]
                n_clusters: 6
                method: kmeans
        alt_growth_rate:
          CA_N: 0.02
          CA_S: 0.02
```

### Conditional Parameter Swaps

Different settings for different regions:

```yaml
settings_management:
  2030:
    policy_region:
      california:
        carbon_tax: 100
        regional_capacity_reserves:
          CapRes_1:
            CA_N: 1.15
            CA_S: 1.15

      arizona:
        carbon_tax: 0
        regional_capacity_reserves:
          CapRes_1:
            AZ: 1.10
```

!!! note "Region-keyed settings"
    `regional_capacity_reserves` is nested as
    `constraint → region → value`, where each `CapRes_<num>` creates a reserve
    zone. A flat region→value mapping is not valid. Note that `new_resources` is **not**
    region-scoped — it builds each listed technology in every model region — so per-region
    new-build availability is not currently supported. Renewable resource clusters are
    scoped per region through `renewables_clusters`.

### Copy Case Policies

Reuse emission policies across scenarios:

**emission_policies.csv**:

```csv
case_id,year,region,copy_case_id,RPS,CES,CO2_cap
baseline,2030,all,,0.50,0.80,
high_re,2030,all,baseline,,
low_carbon,2030,all,,0.50,0.90,
baseline,2040,all,,0.70,0.90,
```

`high_re` copies policies from `baseline` instead of redefining.

## Validation and Debugging

### Validate Scenario Definitions

Check scenario file before running:

```python
import pandas as pd

df = pd.read_csv('scenario_definitions.csv')

# Check required columns
required_cols = ['case_id', 'year']
missing = [c for c in required_cols if c not in df.columns]
if missing:
    print(f"Missing columns: {missing}")

# Check for duplicates
duplicates = df[df.duplicated(['case_id', 'year'], keep=False)]
if not duplicates.empty:
    print("Duplicate case_id × year combinations:")
    print(duplicates)

# Check years match model_year
model_years = [2030, 2040, 2050]
invalid_years = df[~df['year'].isin(model_years)]
if not invalid_years.empty:
    print("Invalid years (not in model_year):")
    print(invalid_years)

# Count scenarios
n_cases = df['case_id'].nunique()
n_years = df['year'].nunique()
print(f"\nTotal scenarios: {n_cases} cases × {n_years} years = {len(df)} runs")
```

### Test Single Scenario

Run one scenario before batch execution:

```bash
# Create test scenario file
echo "case_id,year,tech_cost" > test_scenario.csv
echo "test,2030,mid" >> test_scenario.csv

# Run single scenario
run_powergenome \
  --settings_file settings \
  --results_folder test_results
```

Check outputs before running full scenario set.

### Debug Parameter Swaps

Verify parameter changes are applied:

```python
from powergenome.settings import load_settings
from pathlib import Path

settings = load_settings(Path("settings"))

# Check base settings
print("Base new_resources:", settings.get('new_resources'))
print("Base carbon_tax:", settings.get('carbon_tax'))

# Manually apply parameter swap (for testing)
settings_mgmt = settings.get('settings_management', {})
year_settings = settings_mgmt.get(2030, {})
cost_settings = year_settings.get('tech_cost', {}).get('low', {})

print("\nLow tech_cost changes:")
print(cost_settings)
```

## Command-Line Options

### Key Flags

```bash
run_powergenome \
  --settings_file settings \           # Settings folder
  --results_folder results \           # Output folder
  --no-current-gens \                  # Skip existing generator clustering
  --no-load \                          # Skip load profile generation
  --sort-gens \                        # Sort output by resource name
  --case-id case1 case2                # Run specific cases only
```

### Selective Execution

Run specific scenarios only:

```bash
# Option 1: Edit scenario_definitions.csv to include only the desired cases

# Option 2: Select cases with --case-id
run_powergenome \
  --settings_file settings \
  --results_folder results \
  --case-id baseline low_cost
```

### Resume Failed Runs

If some scenarios fail, rerun only failed cases:

```bash
# Check which scenarios completed
ls results/*/Generators_data.csv

# Remove failed scenario folders
rm -rf results/failed_case_2030

# Rerun (PowerGenome skips existing outputs)
run_powergenome \
  --settings_file settings \
  --results_folder results
```

## Best Practices

1. **Start small**: Test with 2-3 scenarios before full matrix
2. **Incremental complexity**: Add one dimension at a time
3. **Meaningful case_ids**: Use descriptive names (`high_re_carbon_2030` not `case_1`)
4. **Document scenarios**: Create `case_descriptions.csv` with explanations
5. **Version control**: Commit scenario definitions and settings to Git
6. **Monitor resources**: Watch memory/CPU usage with large parallel runs
7. **Validate early**: Check one scenario output before running all
8. **Archive results**: Compress/archive results after analysis

## Troubleshooting

### Scenario Not Running

**Problem**: Some scenarios skipped

**Check**:

- Scenario exists in `scenario_definitions.csv`?
- `case_id` × `year` combination unique?
- Parameter values exist in `settings_management`?

### Parameter Not Changing

**Problem**: Settings don't vary across scenarios

**Check**:

- Parameter spelled correctly in `settings_management`?
- Year matches between scenario file and `settings_management`?
- Parameter value matches exactly (case-sensitive)?

## Next Steps

- [Configure Settings](configure-settings.md): Settings file organization
- [Settings Reference - Scenario Management](../reference/settings/scenario-management.md): Complete parameter documentation
- [Explanation - Multi-Scenario Architecture](../explanation/scenarios.md): How scenario management works internally
