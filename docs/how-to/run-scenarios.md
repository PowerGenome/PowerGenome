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
        atb_cost_case: Advanced
      mid:
        atb_cost_case: Moderate
      high:
        atb_cost_case: Conservative

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
        atb_cost_case: Advanced
      mid:
        atb_cost_case: Moderate
      high:
        atb_cost_case: Conservative

    carbon_policy:
      none:
        carbon_tax: 0
      moderate:
        carbon_tax: 75
      aggressive:
        carbon_tax: 150
```

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

## Parameter Swap Examples

### Technology Costs

Vary technology costs across scenarios:

```yaml
settings_management:
  2030:
    solar_cost:
      low:
        resource_modifiers:
          UtilityPV_*:
            capex_mw:
              2030: 0.8
      mid: {}  # No change from baseline
      high:
        resource_modifiers:
          UtilityPV_*:
            capex_mw:
              2030: 1.3

    battery_cost:
      low:
        resource_modifiers:
          Battery_*:
            capex_mw:
              2030: 0.7
            capex_mwh:
              2030: 0.7
      mid: {}
      high:
        resource_modifiers:
          Battery_*:
            capex_mw:
              2030: 1.4
            capex_mwh:
              2030: 1.4
```

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
        growth_scenario: reference
        default_growth_rate: 0.005
      mid:
        growth_scenario: moderate
        default_growth_rate: 0.015
      high:
        growth_scenario: high_electrification
        default_growth_rate: 0.03
        distributed_gen_values:
          2030:
            CA_N: 2500
            CA_S: 3000
```

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
        new_resources:
          - [NaturalGas, CCAvgCF, Moderate, 500]
          - [UtilityPV, Class1, Moderate, 100]
          - [LandbasedWind, Class3, Moderate, 100]
          - [Battery, "*", Moderate, 100]
        new_gen_not_available:
          ALL_REGIONS:
            - Nuclear_Nuclear - Large_Moderate

    ccs:
      allowed:
        new_resources:
          - [NaturalGas, CCCCSAvgCF, Conservative, 500]
          - [Coal, CCS90AvgCF, Moderate, 500]
      not_allowed:
        new_gen_not_available:
          ALL_REGIONS:
            - NaturalGas_CCCCSAvgCF_Conservative
            - Coal_CCS90AvgCF_Moderate
```

### Retirement Assumptions

Vary existing generator retirements:

```yaml
settings_management:
  2030:
    coal_retirement:
      early:
        retirement_ages:
          Conventional Steam Coal: 45
      baseline:
        retirement_ages:
          Conventional Steam Coal: 60
      extended:
        retirement_ages:
          Conventional Steam Coal: 75

    gas_retirement:
      early:
        retirement_ages:
          Natural Gas Fired Combined Cycle: 35
      baseline:
        retirement_ages:
          Natural Gas Fired Combined Cycle: 50
```

### Transmission Expansion

Vary transmission constraints:

```yaml
settings_management:
  2030:
    transmission:
      limited:
        max_network_reinforcement_mw: 500
        tx_expansion_per_mw: 2000
      baseline:
        max_network_reinforcement_mw: 1500
        tx_expansion_per_mw: 1200
      unlimited:
        enforce_constraints: false
```

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

## Parallel Execution

### Run Scenarios in Parallel

Use multiple workers to speed up execution:

```bash
run_powergenome \
  --settings_file settings \
  --results_folder results \
  --num_workers 8
```

**Performance**:

- Each worker runs one scenario at a time
- Optimal `num_workers` ≈ number of CPU cores
- Memory usage scales with workers (monitor for large models)

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
model_periods:
  - [2020, 2030]
  - [2031, 2040]
  - [2041, 2050]

settings_management:
  2030:
    tech_cost:
      low:
        atb_cost_case: Advanced
      mid:
        atb_cost_case: Moderate
    carbon:
      50:
        carbon_tax: 50

  2040:
    tech_cost:
      low:
        atb_cost_case: Advanced
      mid:
        atb_cost_case: Moderate
    carbon:
      75:
        carbon_tax: 75

  2050:
    tech_cost:
      low:
        atb_cost_case: Advanced
      mid:
        atb_cost_case: Moderate
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
        atb_cost_case: Advanced
        resource_modifiers:
          UtilityPV_*:
            capex_mw:
              2030: 0.8
          LandbasedWind_*:
            capex_mw:
              2030: 0.85
        renewable_clusters:
          UtilityPV_Class1_Moderate:
            - region: CA_N
              cluster: 1
              capacity_mw: 5000  # Higher capacity
          LandbasedWind_Class3_Moderate:
            - region: CA_N
              cluster: 1
              capacity_mw: 3000
        default_growth_rate: 0.02
```

### Conditional Parameter Swaps

Different settings for different regions:

```yaml
settings_management:
  2030:
    policy_region:
      california:
        carbon_tax: 100
        new_gen_not_available:
          CA_N:
            - Coal_*
            - NaturalGas_CT*
          CA_S:
            - Coal_*
            - NaturalGas_CT*
        regional_capacity_reserves:
          CA_N: 1.15
          CA_S: 1.15

      arizona:
        carbon_tax: 0
        new_gen_not_available:
          AZ:
            - OffshoreWind_*
        regional_capacity_reserves:
          AZ: 1.10
```

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
print("Base atb_cost_case:", settings.get('atb_cost_case'))
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
# Option 1: Edit scenario_definitions.csv to include only desired cases

# Option 2: Use --filter flag (if available in your version)
run_powergenome \
  --settings_file settings \
  --results_folder results \
  --filter "case_id.startswith('baseline')"
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
