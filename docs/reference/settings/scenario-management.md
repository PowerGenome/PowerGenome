# Scenario Management Settings

PowerGenome supports running multiple scenarios in a single execution, systematically varying parameters across cases. This enables sensitivity analyses, policy comparisons, and uncertainty exploration.

## Scenario Workflow

1. **Define scenarios** in a CSV file with parameter variations
2. **Configure parameter swaps** in `settings_management`
3. **Run batch execution** with `run_powergenome_multiple`
4. **Output** to separate folders per case

## Scenario Definition File

### `scenario_definitions_fn`

**Type**: String (path)
**Required**: For multi-scenario runs
**Example**: `"scenario_definitions.csv"`

CSV file defining all scenario variations.

```yaml
scenario_definitions_fn: scenario_definitions.csv
```

**File structure**:

**Mandatory columns**:

- `case_id`: Unique identifier for each case
- `year`: Model year (must match `model_year` values)

**User-defined columns**: Parameter dimensions to vary across scenarios

**Example file**:

```csv
case_id,year,solar_cost,wind_cost,gas_price,carbon_policy
baseline_2030,2030,mid,mid,reference,none
high_re_2030,2030,low,low,reference,none
low_re_2030,2030,high,high,reference,none
carbon_2030,2030,mid,mid,reference,aggressive
baseline_2040,2040,mid,mid,reference,none
high_re_2040,2040,low,low,reference,none
```

**Requirements**:

- Every `case_id` × `year` combination must be unique
- Number of rows = (# unique case_ids) × (# model years)
- User-defined columns reference keys in `settings_management`

## Settings Management

### `settings_management`

**Type**: Nested dictionary (year → parameter → value → settings)
**Required**: For multi-scenario runs
**Example**: See below

Defines how settings parameters change based on scenario dimensions.

**Structure**:

```yaml
settings_management:
  <year>:
    <parameter_name>:
      <parameter_value>:
        <settings_to_modify>
```

**Detailed example**:

```yaml
settings_management:
  2030:
    solar_cost:
      low:
        atb_cost_case: Advanced
        resource_modifiers:
          UtilityPV_Class1_Moderate:
            capex_mw:
              2030: 0.8
      mid:
        atb_cost_case: Moderate
      high:
        atb_cost_case: Conservative
        resource_modifiers:
          UtilityPV_Class1_Moderate:
            capex_mw:
              2030: 1.2

    wind_cost:
      low:
        resource_modifiers:
          LandbasedWind_Class3_Moderate:
            capex_mw:
              2030: 0.85
      mid: {}  # No changes from baseline
      high:
        resource_modifiers:
          LandbasedWind_Class3_Moderate:
            capex_mw:
              2030: 1.15

    gas_price:
      reference:
        fuel_scenarios:
          naturalgas: reference
      high:
        fuel_scenarios:
          naturalgas: high_price
      low:
        fuel_scenarios:
          naturalgas: low_price

    carbon_policy:
      none:
        carbon_tax: 0
      moderate:
        carbon_tax:
          2030: 50
          2040: 75
      aggressive:
        carbon_tax:
          2030: 100
          2040: 150
```

**How it works**:

1. For each row in `scenario_definitions.csv`, PowerGenome reads parameter values
2. Looks up corresponding settings in `settings_management`
3. Applies all settings modifications for that scenario
4. Runs model with modified settings
5. Writes outputs to `results/{case_id}/`

## Parameter Swap Examples

### Technology Costs

```yaml
settings_management:
  2030:
    battery_cost:
      low:
        resource_modifiers:
          batteries:
            technology: Battery
            tech_detail: "*"
            capex_mw: [mul, 0.7]
            capex_mwh: [mul, 0.7]
      high:
        resource_modifiers:
          batteries:
            technology: Battery
            tech_detail: "*"
            capex_mw: [mul, 1.3]
            capex_mwh: [mul, 1.3]
```

### Technology Availability

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
```

<!-- Retirement age and demand growth scenario swapping removed. These concepts are now handled directly by input data and not via settings_management parameters. -->

### Transmission Expansion

```yaml
settings_management:
  2030:
    tx_expansion:
      limited:
        max_network_reinforcement_mw: 500
        tx_expansion_per_mw: 1500
      baseline:
        max_network_reinforcement_mw: 1000
        tx_expansion_per_mw: 1000
      unlimited:
        enforce_constraints: false  # Copper plate
```

<!-- case_id_description_fn removed. If descriptive labels are desired, manage externally; not a supported setting. -->

## Multi-Period Scenarios

For multi-period models, each period can have different parameter values:

**scenario_definitions.csv** (multi-period, technology cost & carbon only):

```csv
case_id,year,tech_cost,carbon
baseline,2030,moderate,none
baseline,2040,moderate,none
baseline,2050,moderate,none
high_tech,2030,advanced,aggressive
high_tech,2040,advanced,aggressive
high_tech,2050,advanced,aggressive
```

**settings_management.yml**:

```yaml
settings_management:
  2030:
    tech_cost:
      moderate:
        atb_cost_case: Moderate
      advanced:
        atb_cost_case: Advanced
    carbon:
      none:
        carbon_tax: 0
      aggressive:
        carbon_tax: 100
  2040:
    tech_cost:
      moderate:
        atb_cost_case: Moderate
      advanced:
        atb_cost_case: Advanced
    carbon:
      none:
        carbon_tax: 0
      aggressive:
        carbon_tax: 150
  2050:
    tech_cost:
      moderate:
        atb_cost_case: Moderate
      advanced:
        atb_cost_case: Advanced
    carbon:
      none:
        carbon_tax: 0
      aggressive:
        carbon_tax: 200
```

## Input/Output Folders

### `input_folder`

**Type**: String (path)
**Required**: No
**Default**: Current directory
**Example**: `"extra_inputs"`

Folder with supplementary input files:

- `emission_policies.csv`
- `misc_gen_inputs.csv`
- `demand_response_profiles.csv`

```yaml
input_folder: extra_inputs
```

Referenced files are relative to this folder.

### Results Folder

**Not a setting** - specified on command line:

```bash
run_powergenome_multiple \
  --settings_file settings \
  --results_folder results
```

Creates structure:

```text
results/
├── baseline_2030/
│   ├── Generators_data.csv
│   ├── Load_data.csv
│   └── ...
├── high_re_2030/
│   ├── Generators_data.csv
│   └── ...
└── carbon_2030/
    └── ...
```

## Scenario Copying

### `copy_case_id`

Use in `emission_policies.csv` to copy policies from another case:

```csv
case_id,year,region,copy_case_id,RPS,CES
baseline,2030,all,,0.5,0.8
high_re,2030,all,baseline,,
carbon,2030,all,,0.5,0.9
```

`high_re` copies policies from `baseline` instead of specifying them explicitly.

## Validation

PowerGenome validates scenario definitions:

**Missing case_id × year combinations**:

```text
Error: case_id 'baseline' missing year 2040
```

**Invalid parameter values**:

```text
Error: Parameter value 'ultra_low' not found in settings_management['solar_cost']
```

**Mismatched years**:

```text
Error: scenario_definitions contains year 2035, not in model_year [2030, 2040, 2050]
```

## Example Complete Configuration

**scenario_definitions.csv** (single-period example):

```csv
case_id,year,cost_case,carbon
base,2030,mid,none
low_cost,2030,low,none
carbon_50,2030,mid,50
carbon_100,2030,mid,100
```

**settings/scenario_management.yml**:

```yaml
scenario_definitions_fn: scenario_definitions.csv

settings_management:
  2030:
    cost_case:
      low:
        resource_financial_case: R&D
        new_resources:
          - [NaturalGas, CCAvgCF, Advanced, 500]
          - [UtilityPV, Class1, Advanced, 100]
      mid:
        resource_financial_case: Market
        new_resources:
          - [NaturalGas, CCAvgCF, Moderate, 500]
          - [UtilityPV, Class1, Moderate, 100]
      high:
        resource_financial_case: Market
        new_resources:
          - [NaturalGas, CCAvgCF, Conservative, 500]
          - [UtilityPV, Class1, Conservative, 100]
    carbon:
      none:
        carbon_tax: 0
      50:
        carbon_tax: 50
      100:
        carbon_tax: 100
```

**Run command**:

```bash
run_powergenome_multiple \
  --settings_file settings \
  --results_folder results \
  --num_workers 4
```

This generates 6 cases (6 rows in scenario_definitions.csv), running up to 4 in parallel.

## Parallel Execution

Use `--num_workers` to run scenarios in parallel:

```bash
run_powergenome_multiple \
  --settings_file settings \
  --results_folder results \
  --num_workers 8
```

Significantly speeds up large scenario ensembles. Each worker runs one case at a time.

## Related Settings

- [Model Definition](model-definition.md): `model_year` must match scenario years
- [New-Build Resources](new-build.md): Technology cost modifications
- [Fuels](fuels.md): Fuel price scenario swapping
- [Transmission](transmission.md): Network reinforcement parameters
