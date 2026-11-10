# Run Multiple Scenarios

Compare cases by adding one scenario CSV and one settings swap to a working
configuration. This guide uses the published Texas three-zone example to compare
`base` with a `carbon_50` case in the 2030 and 2045 planning periods.

Use this page for the practical workflow. For supported shapes and combinations,
see [Scenario Management Settings](../reference/settings/scenario-management.md).
For precedence and execution order, see
[Multi-Scenario Architecture](../explanation/scenarios.md).

## Basic Setup

### 1. Download a working system

The published examples live in a separate
[PowerGenome-examples](https://github.com/PowerGenome/PowerGenome-examples)
repository, so no PowerGenome source checkout is involved. Download the same
pinned Texas snapshot that
[Getting Started](../tutorials/getting-started.md#1-download-the-texas-example)
uses:

```bash
mkdir scenario-study
cd scenario-study
curl --fail --location \
  https://codeload.github.com/PowerGenome/PowerGenome-examples/tar.gz/7e34f714469e68fca173cdb849f74c47be779420 \
  --output powergenome-examples.tar.gz
tar -xzf powergenome-examples.tar.gz --strip-components=1
rm powergenome-examples.tar.gz
```

That archive is under 1 MB. Choose another folder name if `scenario-study`
already exists, and run every command below from the new folder.

Stay at the top of `scenario-study/` and address the study as
`example_systems/TX-3-zone/`; do not change into that folder. Its `data_location`
and `input_folder` resolve from the settings folder's parent, but its published
`RESOURCE_GROUPS` and `RESOURCE_GROUP_PROFILES` paths resolve from your working
directory.

The example is demonstration data (240 synthetic profile time steps), not a
production study. If you only need values to change over time within one case,
use [year-keyed settings](use-year-keyed-settings.md) instead.

Its `example_systems/TX-3-zone/settings/model_definition.yml` already defines:

```yaml
model_regions: [TX1, TX2, TX3]
region_aggregations:
  TX1: [p60, p61, p62]
  TX2: [p63, p64]
  TX3: [p65, p67]
model_year: [2030, 2045]
model_first_planning_year: [2025, 2031]
```

These are existing values, not settings to add again. The first planning period
is 2025–2030 and the second is 2031–2045. Keep the two planning-year lists in
corresponding chronological order; they are paired by position, not sorted
automatically.

### 2. Create Scenario Definitions

Replace the contents of
`example_systems/TX-3-zone/extra_inputs/scenario_inputs.csv` with:

```csv
case_id,year,carbon_policy
base,2030,unchanged
carbon_50,2030,tax_50
base,2045,unchanged
carbon_50,2045,tax_50
```

The published file has a single `base` case and no dimension column; this version
keeps `base` and adds the `carbon_50` case. `case_id` and `year` are mandatory.
Keep the same case ID across periods: each row is one case/year, not a new
independent case. `carbon_policy` is a dimension name you choose; its values
select the swaps in the next step.

Include every planning year for each complete case, chronologically within that
case. Duplicate case/year pairs fail, but incomplete cases are not automatically
rejected. Period numbers follow row order within each case, so a missing or
out-of-order row can put a year in the wrong `Inputs_pN` directory.

#### Add policy rows for the new case

The study's emission policy table is keyed by `case_id` (the published rows all
use `base`), so a case with no matching rows stops the run with a `KeyError` such
as `('carbon_50', 2030)`. Replace the contents of
`example_systems/TX-3-zone/extra_inputs/emission_policies.csv` with:

```csv
case_id,year,region,ESR_1,ESR_2,ESR_3,copy_case_id
base,2030,TX1,0,0,0,
base,2030,TX2,0,0,0,
base,2030,TX3,0,0,0,
base,2045,TX1,0,0,0,
base,2045,TX2,0,0,0,
base,2045,TX3,0,0,0,
carbon_50,2030,TX1,,,,base
carbon_50,2030,TX2,,,,base
carbon_50,2030,TX3,,,,base
carbon_50,2045,TX1,,,,base
carbon_50,2045,TX2,,,,base
carbon_50,2045,TX3,,,,base
```

The [`copy_case_id`](../reference/settings/scenario-management.md#copy_case_id)
entries reuse the `base` values for `carbon_50`, so the cases carry identical
policies and only one set of values needs maintenance. Duplicating the six `base`
rows under `case_id,carbon_50` works too. Every `ESR_*` value here is zero, so
this step only satisfies the per-case lookup; real policies would need the values
you intend for each case.

!!! note "The carbon tax does not need extra fuel data"
    `carbon_tax` is applied using CO₂ content built from the
    [`fuel_emission_factors` settings parameter](../reference/settings/fuels.md#fuel_emission_factors),
    not from a column in the fuel price table. The Texas `fuels.yml` already
    defines factors for natural gas, coal, distillate, and hydrogen, so the swap
    below works as written.

### 3. Configure Settings Management

Replace the contents of
`example_systems/TX-3-zone/settings/scenario_management.yml` with:

```yaml
scenario_definitions_fn: scenario_inputs.csv

settings_management:
  all_years:
    carbon_policy:
      unchanged: {}  # Keep the base settings
      tax_50:
        carbon_tax: 50
```

The published file is `settings_management: {}`, so replace it rather than adding
a second mapping.

This is one swap: `tax_50` sets `carbon_tax` to 50 USD per metric tonne of CO₂
for both years. The tax is added to fuel costs using fuel CO₂ content; it is not
an emissions-cap setting.
The emission policies from the previous step remain in place for both cases.

The published `settings/extra_inputs.yml` already sets
`input_folder: extra_inputs` and
`scenario_definitions_fn: scenario_inputs.csv`, and it also names this study's
emission policy table (`emission_policies_fn: emission_policies.csv`) from the
previous step. `scenario_definitions_fn` is relative to `input_folder`, so it
resolves to `example_systems/TX-3-zone/extra_inputs/scenario_inputs.csv`. The
`scenario_definitions_fn` line in the block above therefore only repeats a value
that is already set.

!!! warning "Keep each top-level setting in one file"
    A settings directory loads only its immediate `*.yml` files, not subfolders
    or `.yaml` files. These files are combined with shallow updates and no
    reliable file order. Do not add a second `settings_management` block in a
    different file: it could replace the whole mapping. Edit the existing block.

Configured relative `data_location` and `input_folder` paths are rooted at the
parent of the settings directory (`example_systems/TX-3-zone/`). Relative
`RESOURCE_GROUPS` and `RESOURCE_GROUP_PROFILES` paths instead use the working
directory, which is why the commands stay at the top of `scenario-study/`.

## Validation and Debugging

### Validate Scenario Definitions

First validate the settings and source data:

```bash
validate_powergenome --settings_file example_systems/TX-3-zone/settings
```

This is not a substitute for checking the complete scenario matrix. To check the
CSV and inspect the actual resolved swaps before generating files, run:

```python
from pathlib import Path

import pandas as pd

from powergenome.settings import build_scenario_settings, load_settings

settings = load_settings(Path("example_systems/TX-3-zone/settings"))
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

Expected values are 0 for `base` and 50 for `carbon_50` in 2030 and 2045.
Unknown dimension/level selections warn and leave settings unchanged; an explicit
`{}` means intentionally keep the base value without a missing-selection warning.
The example also emits warnings about unassigned model resource tags, resources
missing from the operational-constraints table, missing transmission investment
costs, and a hydro resource without a variable profile. Review these before
adapting it for a real study.

### Test Single Scenario

Generate both periods of the base case first:

```bash
run_powergenome \
  --settings_file example_systems/TX-3-zone/settings \
  --results_folder base_check \
  --case-id base
```

Inspect `base_check/log.txt` and the two period folders before running both
cases. PowerGenome generates model inputs; it does not solve the GenX model.

## Run the scenario set

```bash
run_powergenome \
  --settings_file example_systems/TX-3-zone/settings \
  --results_folder results
```

With default GenX output, the directory layout is:

```text
results/
├── log.txt
├── base/
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
`powergenome_case_settings.yml`. Paying 50 USD per tonne of CO₂ raises the
delivered price of emitting fuels, which you can check in
`system/Fuels_data.csv`: in this example the first period goes from 3.16 to 5.82
USD/MMBtu for natural gas and from 2.07 to 6.84 for coal, and the second period
from 4.86 to 7.51 and from 2.13 to 6.91. Read the row labeled with `Time_Index`
`1`; row `0` holds the CO₂ content per MMBtu.

## Running Scenarios Sequentially

PowerGenome runs sequentially, grouped by year in the order years first appear
in the CSV, then by case order within each year. This example runs both 2030
cases before both 2045 cases. There is no scenario `--num_workers` option;
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
  --settings_file example_systems/TX-3-zone/settings \
  --results_folder selected_results \
  --case-id base carbon_50
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
