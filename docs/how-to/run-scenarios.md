# Run Multiple Scenarios

Compare cases by adding one scenario CSV and one settings swap to a working
configuration. This guide uses the bundled three-region test system to compare
`baseline` with a `carbon_50` case in 2030 and 2040.

Use this page for the practical workflow. For supported shapes and combinations,
see [Scenario Management Settings](../reference/settings/scenario-management.md).
For precedence and execution order, see
[Multi-Scenario Architecture](../explanation/scenarios.md).

## Basic Setup

### 1. Copy a working system

The three-region test system lives in `tests/test_system/` of the PowerGenome
source repository. It is **not** part of the installed package: the distribution
ships only the `powergenome` package and its bundled `data/` folder, so
`pip install`/`uv` alone leaves you without the fixture. Pull just that folder
with a sparse, blobless clone rather than the whole repository:

```bash
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/PowerGenome/PowerGenome.git
cd PowerGenome
git sparse-checkout set tests/test_system
```

That downloads the folder and nothing else (~20 MB, a couple of seconds). If you
installed a release and want the fixture to match it, add `--branch <tag>` to the
clone.

The fixture is small demonstration data, not a production study. If you only need
values to change over time within one case, use
[year-keyed settings](use-year-keyed-settings.md) instead.

From the repository root, copy the fixture outside the checkout:

```bash
cp -R tests/test_system ../scenario-study
cd ../scenario-study
```

Choose a new destination if `../scenario-study` already exists. Run the remaining
commands from the copied `scenario-study/` directory. Its settings already define:

```yaml
model_regions: [p1_2, p3, p4]
region_aggregations:
  p1_2: [p1, p2]
model_year: [2030, 2040]
model_first_planning_year: [2025, 2031]
```

These are existing values in `settings/model_definition.yml`, not settings to add
again. The first planning period is 2025–2030 and the second is 2031–2040.
Keep the two planning-year lists in corresponding chronological order; they are
paired by position, not sorted automatically.

### 2. Create Scenario Definitions

Replace the contents of the copied `extra_inputs/scenario_inputs.csv` with:

```csv
case_id,year,carbon_policy
baseline,2030,unchanged
carbon_50,2030,tax_50
baseline,2040,unchanged
carbon_50,2040,tax_50
```

`case_id` and `year` are mandatory. Keep the same case ID across periods: each
row is one case/year, not a new independent case. `carbon_policy` is a dimension
name you choose; its values select the swaps in the next step.

Include every planning year for each complete case, chronologically within that
case. Duplicate case/year pairs fail, but incomplete cases are not automatically
rejected. Period numbers follow row order within each case, so a missing or
out-of-order row can put a year in the wrong `Inputs_pN` directory.

The published example systems ship a minimal version of this file with a single
`base` case covering both planning years, paired with `settings_management: {}` —
see the [Texas three-zone `scenario_inputs.csv`](https://github.com/PowerGenome/PowerGenome-examples/blob/7e34f714469e68fca173cdb849f74c47be779420/example_systems/TX-3-zone/extra_inputs/scenario_inputs.csv)
and its [`scenario_management.yml`](https://github.com/PowerGenome/PowerGenome-examples/blob/7e34f714469e68fca173cdb849f74c47be779420/example_systems/TX-3-zone/settings/scenario_management.yml).
That is a valid single-case setup, not a template for variation: copy its shape,
then add the case and policy rows you need.

Two things break if you add cases to an example system without checking the rest
of its inputs. Its emission policy table is keyed by `case_id` (the Texas rows use
`base`), so a new case with no matching rows stops the run with a `KeyError` such
as `('baseline', 2030)`; duplicate the `base` rows for the new case ID or set
[`copy_case_id`](../reference/settings/scenario-management.md#copy_case_id) to
reuse them. The `carbon_tax` swap used below also needs a fuel price table with a
`CO2_content_tons_per_MMBtu` column, which the Texas example does not include.

### 3. Configure Settings Management

Replace the contents of the copied `settings/scenario_management.yml` with:

```yaml
scenario_definitions_fn: scenario_inputs.csv

settings_management:
  all_years:
    carbon_policy:
      unchanged: {}  # Keep the base settings
      tax_50:
        carbon_tax: 50
```

This is one swap: `tax_50` sets `carbon_tax` to 50 USD per metric tonne of CO₂
for both years. The tax is added to fuel costs using fuel CO₂ content; it is not
an emissions-cap setting.
The bundled emission policies remain in place for both cases.

The copied `settings/extra_inputs.yml` already sets
`input_folder: extra_inputs`. `scenario_definitions_fn` is relative to that
folder, so it resolves to `scenario-study/extra_inputs/scenario_inputs.csv`.
The commented-out filename in `extra_inputs.yml` can stay commented out.

!!! warning "Keep each top-level setting in one file"
    A settings directory loads only its immediate `*.yml` files, not subfolders
    or `.yaml` files. These files are combined with shallow updates and no
    reliable file order. Do not add a second `settings_management` block in a
    different file: it could replace the whole mapping. Edit the existing block.

Configured relative `data_location` and `input_folder` paths are rooted at the
parent of the settings directory. Relative `RESOURCE_GROUPS` and
`RESOURCE_GROUP_PROFILES` paths instead use the working directory. Running from
the copied fixture keeps its `test_data/resource_groups` path valid.

## Validation and Debugging

### Validate Scenario Definitions

First validate the settings and source data:

```bash
validate_powergenome --settings_file settings
```

This is not a substitute for checking the complete scenario matrix. To check the
CSV and inspect the actual resolved swaps before generating files, run:

```python
from pathlib import Path

import pandas as pd

from powergenome.settings import build_scenario_settings, load_settings

settings = load_settings(Path("settings"))
scenarios = pd.read_csv(
    Path(settings["input_folder"]) / settings["scenario_definitions_fn"]
)
years = settings["model_year"]
assert {"case_id", "year"}.issubset(scenarios.columns)
assert not scenarios.duplicated(["case_id", "year"]).any()
for case_id, rows in scenarios.groupby("case_id", sort=False):
    assert rows["year"].tolist() == years, case_id

resolved = build_scenario_settings(settings, scenarios)
for year in years:
    for case_id, case_settings in resolved[year].items():
        print(year, case_id, case_settings.get("carbon_tax", 0))
```

Expected values are 0 for `baseline` and 50 for `carbon_50` in both years.
Unknown dimension/level selections warn and leave settings unchanged; an explicit
`{}` means intentionally keep the base value without a missing-selection warning.
The fixture can also emit warnings about incomplete fuel-price coverage and
resource tags; review these before adapting it for a real study.

### Test Single Scenario

Generate both periods of the baseline first:

```bash
run_powergenome \
  --settings_file settings \
  --results_folder baseline_check \
  --case-id baseline
```

Inspect `baseline_check/log.txt` and the two period folders before running both
cases. PowerGenome generates model inputs; it does not solve the GenX model.

## Run the scenario set

```bash
run_powergenome \
  --settings_file settings \
  --results_folder results
```

With default GenX output, the directory layout is:

```text
results/
├── log.txt
├── baseline/
│   └── Inputs/
│       ├── Inputs_p1/
│       │   ├── resources/
│       │   ├── system/
│       │   ├── policies/
│       │   └── powergenome_case_settings.yml
│       └── Inputs_p2/
└── carbon_50/
    └── Inputs/
        ├── Inputs_p1/
        └── Inputs_p2/
```

Each period has the same subdirectory layout. Confirm the resolved `carbon_tax`,
`model_year`, and `model_first_planning_year` in
`powergenome_case_settings.yml`. Compare `system/Fuels_data.csv` between cases to
check the tax effect on emitting fuels.

## Running Scenarios Sequentially

PowerGenome runs sequentially, grouped by year in the order years first appear
in the CSV, then by case order within each year. This example runs both 2030
cases before both 2040 cases. There is no scenario `--num_workers` option;
`clustering_n_jobs` controls renewable clustering, not case execution.

Monitor a run from another terminal in `scenario-study/`:

```bash
tail -f results/log.txt
```

## Command-Line Options

### Selective Execution

Select one or more existing case IDs; each selection includes all its CSV years:

```bash
run_powergenome \
  --settings_file settings \
  --results_folder selected_results \
  --case-id baseline carbon_50
```

Use `--case-id` with a hyphen. See the [CLI Reference](../reference/cli.md) for
flags such as `--no-current-gens`, `--no-load`, and `--sort-gens`.
For MacroEnergy output alongside GenX, add `--macro`; for Macro-only output, add
`--macro --no-genx`. See [Macro Output Settings](../reference/settings/macro-output.md).

### Resume Failed Runs

There is no automatic resume or skip-completed behavior. Check the log, then
rerun only the affected case IDs into a new results folder, using the same
`--case-id` pattern. Do not delete successful outputs to retry a failed case.

## Extend the study

Keep the first addition small: one dimension and one expected output change.
Use the reference rather than duplicating a large matrix of combinations:

- [Technology Costs](../reference/settings/scenario-management.md#technology-costs):
  ATB cost cases and numeric resource modifiers.
- [Technology Availability](../reference/settings/scenario-management.md#technology-availability):
  complete replacement lists and region limitations.
- [Fuel prices and demand](../reference/settings/scenario-management.md#fuel-prices-and-demand):
  matching source-data scenarios rather than unsupported growth settings.
- [Retirement assumptions](../reference/settings/scenario-management.md#retirement-assumptions):
  change source data between cases, not periods within a myopic case.
- [Transmission and regional reserves](../reference/settings/scenario-management.md#transmission-and-regional-reserves):
  reinforcement limits and nested reserve-margin shapes.
- [Multi-period combinations](../reference/settings/scenario-management.md#multi-period-scenarios)
  and [merge precedence](../explanation/scenarios.md#how-settings-are-constructed-for-each-case):
  combine year-keyed values, `all_years`, and year-specific swaps deliberately.
- [Copy Case Policies](../reference/settings/scenario-management.md#copy_case_id):
  reuse policy rows without copying an entire scenario.

## Troubleshooting

### Scenario Not Running

Check that the filename resolves under `input_folder`, required CSV columns are
present, the selected case ID exists, and each case has all its planning years.
A scenario CSV must be enabled with `scenario_definitions_fn`; without it,
`settings_management` swaps are not applied.

### Parameter Not Changing

Check dimension and level spelling, YAML value types, and the applicable year
block. Inspect the resolved case settings, not just the base YAML. Scenario
swaps deep-merge nested dictionaries but replace lists entirely. Two selected
dimensions cannot modify the same leaf setting within one year block; see
[conflict rules](../explanation/scenarios.md#deep-merge-behavior).
