# Getting Started with PowerGenome

Install PowerGenome, run a small bundled study, and inspect the generated GenX
inputs. PowerGenome prepares inputs; it does not solve the capacity expansion
model.

**Prerequisites**: Basic command-line knowledge, Git, and
[uv](https://docs.astral.sh/uv/getting-started/installation/). The commands below
use a macOS/Linux shell and Python 3.11.

## Installation

Use a source checkout for this tutorial so the installed code and bundled data
match. From a directory where you keep projects:

```bash
git clone https://github.com/PowerGenome/PowerGenome.git
cd PowerGenome
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead of
`source .venv/bin/activate`, and use the equivalent PowerShell copy commands
below. Keep the environment active when changing directories.

??? note "Installing a published release instead"
    For an existing study, you can install a release with `uv pip install powergenome`
    in an activated uv environment. Use documentation and example
    data matching that release. The bundled `tests/test_system/` data is not
    installed by the package; the source workflow above avoids that mismatch.

Verify installation:

```bash
run_powergenome --help
```

## Project Structure

PowerGenome projects follow this structure:

```
my_study/
├── settings/              # Configuration files (YAML)
│   ├── model_definition.yml
│   ├── resources.yml
│   ├── fuels.yml
│   ├── demand.yml
│   ├── transmission.yml
│   └── ...
├── extra_inputs/          # User-provided data files
│   ├── scenario_inputs.csv
│   ├── emission_policies.csv
│   └── ...
├── data/                  # Source data (optional, can be elsewhere)
│   ├── generators.csv
│   ├── demand_profiles.parquet
│   └── ...
└── results/               # Output folder (created by PowerGenome)
    ├── log.txt
    └── Inputs/            # Under a case ID if using a scenario CSV
        └── Inputs_p1/
            ├── resources/ # Thermal.csv, Vre.csv, Storage.csv, ...
            ├── system/    # Demand_data.csv, Fuels_data.csv, Network.csv, ...
            └── policies/
```

!!! tip "Keep Projects Separate"
    Create project folders **outside** the PowerGenome repository to avoid git conflicts and make updates easier.

## Running Your First Model

### 1. Use the Test System

From the `PowerGenome` checkout root, copy the bundled study outside the
repository so later tutorial edits do not change the test fixtures. Choose an
unused destination; the commands below use `powergenome-study`:

```bash
cp -R tests/test_system ../powergenome-study
cd ../powergenome-study
validate_powergenome --settings_file settings
run_powergenome --settings_file settings --results_folder test_output
```

Run from this study directory unless a later tutorial explicitly creates a
separate copy. Its relative resource-group path depends on that working directory.

This will:

1. Load the `.yml` files directly inside `settings/`
2. Read source tables from `test_data/` and policy inputs from `extra_inputs/`
3. Write two planning periods to `test_output/Inputs/Inputs_p1/` and
   `test_output/Inputs/Inputs_p2/`

`data_location` and `input_folder` resolve relative to the directory containing
`settings/`. In contrast, relative `RESOURCE_GROUPS` and
`RESOURCE_GROUP_PROFILES` paths resolve from your current working directory.

For a study-oriented example, see the
[CA/AZ and Texas configurations in the examples update](https://github.com/PowerGenome/PowerGenome-examples/pull/1).
These use short, synthetic profiles for demonstration, not production studies.
The [System Design tool](https://gschivley.github.io/PowerGenome-tools/web/)
is another way to generate a starting settings folder; obtain its required data
files before running it.

### 2. Check completion

Both commands should exit successfully. Confirm that
`test_output/Inputs/Inputs_p1/system/Demand_data.csv` and the corresponding
`Inputs_p2/` directory exist. If the run fails, read the first error in the
terminal and [check the log](../how-to/debugging.md).

The [CLI Reference](../reference/cli.md) covers optional flags. Do not skip
pipeline stages for this first run; the analysis below needs their outputs.

### 3. Monitor Progress

PowerGenome logs progress and warnings to the terminal and writes a more detailed
`test_output/log.txt`. The bundled test run completes with warnings, including
missing uranium and distillate fuel prices; it is a software demonstration, not
a calibrated study. Resolve missing prices, unmatched technologies, and incomplete
regional data before using your own results.

## Examining Outputs

PowerGenome generates multiple CSV files formatted for GenX:

### Key Output Files

**resources/Thermal.csv, Vre.csv, Storage.csv, Hydro.csv, Must_run.csv**
: Resource characteristics grouped by resource type; files are written only
  when that type is present

**system/Demand_data.csv**
: Demand by zone and time step, plus representative-period metadata

**system/Fuels_data.csv**
: Fuel prices by region and time step

**system/Network.csv**
: Transmission topology and constraints

**system/Generators_variability.csv**
: Generation profiles for variable resources

These are the default GenX outputs. Set `old_genx_format: true` only when your
GenX workflow requires the legacy flat `Generators_data.csv` and `Load_data.csv`
layout. See [Output Format](../explanation/output-format.md) for both layouts.

### Exploring Resource Data

```python
from pathlib import Path

import pandas as pd

period = Path("test_output/Inputs/Inputs_p1")
resource_files = [
    period / "resources" / f"{name}.csv"
    for name in ["Thermal", "Vre", "Storage", "Hydro", "Must_run", "Flex_demand"]
]
gens = pd.concat(
    [pd.read_csv(path) for path in resource_files if path.exists()],
    ignore_index=True,
)

# View summary
print(f"Total resources: {len(gens)}")
print(f"Regions: {gens['region'].unique()}")
print(f"Technologies: {gens['technology'].nunique()}")

# Check existing vs new capacity
print(f"\nExisting capacity: {gens['Existing_Cap_MW'].sum():.0f} MW")
print(f"New-build available: {gens[gens['New_Build']==1]['technology'].unique()}")

# View by technology
gens.groupby('technology')['Existing_Cap_MW'].sum()
```

### Understanding Resource Columns

Common columns in the resource files (some are resource-type specific):

- **Resource**: Unique resource name
- **region**: Model region
- **technology**: Technology type
- **Existing_Cap_MW**: Existing capacity
- **New_Build**: Can new capacity be built? (0/1)
- **Inv_Cost_per_MWyr**: Investment cost ($/MW-yr)
- **Fixed_OM_Cost_per_MWyr**: Fixed O&M ($/MW-yr)
- **Var_OM_Cost_per_MWh**: Variable O&M ($/MWh)
- **Heat_Rate_MMBTU_per_MWh**: Heat rate
- **Fuel**: Fuel type

Resource tags (GenX behavior):

- **THERM**: Thermal generator (has heat rate, startup costs)
- **VRE**: Variable renewable energy (wind/solar)
- **STOR**: Storage resource
- **HYDRO**: Hydroelectric resource

See [Output Format](../explanation/output-format.md) for complete documentation.

## Next Steps

Continue with the bundled study:

1. [Building a Multi-Region Model](multi-region-model.md): use the same copy to
   inspect its regions and change transmission expansion limits.
2. [Working with Renewable Resources](renewable-resources.md): follow its
   separate-copy setup to change renewable resource resolution.
3. For your own study, generate starting settings with the
   [System Design tool](https://gschivley.github.io/PowerGenome-tools/web/) and
   [supply the referenced data tables](../how-to/configure-data-tables.md).

Do not change regions or planning years without checking data coverage. The
bundled dataset is deliberately small.

## Troubleshooting

### Common Issues

**"Table not found" error**
: Check `data_location` in settings points to correct folder

**"Region not found" error**
: Verify region names in `model_regions` match data tables

**Import errors**
: Activate the environment where PowerGenome is installed. For an editable source
  install, run `uv pip install -e .` from the PowerGenome directory.

See [Debugging](../how-to/debugging.md) for more solutions.

## Learning Resources

- **Settings reference**: [Model Definition](../reference/settings/model-definition.md)
- **How settings work**: [Configure Settings](../how-to/configure-settings.md)
- **Understanding the pipeline**: [Data Pipeline Flow](../explanation/pipeline.md)
- **System Design tool**: [Generate settings files for your study area](https://gschivley.github.io/PowerGenome-tools/web/)

## Get Help

- Questions: [groups.io/g/powergenome](https://groups.io/g/powergenome)
- Issues: [GitHub Issues](https://github.com/PowerGenome/PowerGenome/issues)
