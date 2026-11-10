# Getting Started with PowerGenome

Install a published PowerGenome release, run the Texas three-zone example from
[PowerGenome-examples](https://github.com/PowerGenome/PowerGenome-examples), and
inspect the generated GenX inputs. PowerGenome prepares inputs; it does not
solve the capacity expansion model.

**Prerequisites**: Basic command-line knowledge,
[uv](https://docs.astral.sh/uv/getting-started/installation/), `curl`, and `tar`.
The commands below use a macOS/Linux shell and Python 3.11. Git and a PowerGenome
source checkout are not required.

## Installation

From a directory where you keep projects, create a new tutorial folder and
install the published release used by this exercise:

```bash
mkdir powergenome-tutorial
cd powergenome-tutorial
uv venv --python 3.11
source .venv/bin/activate
uv pip install powergenome==0.8.0
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead of
`source .venv/bin/activate`. Use `curl.exe` for the download below, or download
and extract the linked archive manually. Keep the environment active when
changing directories.

The package version and example snapshot below are pinned together for a
repeatable first run. For another release, check example compatibility before
upgrading.

Verify installation:

```bash
run_powergenome --help
```

## Project Structure

PowerGenome projects follow this structure:

```text
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

### 1. Download the Texas example

From `powergenome-tutorial/`, download and extract the pinned
[examples snapshot](https://github.com/PowerGenome/PowerGenome-examples/tree/7e34f714469e68fca173cdb849f74c47be779420):

```bash
curl --fail --location \
  https://codeload.github.com/PowerGenome/PowerGenome-examples/tar.gz/7e34f714469e68fca173cdb849f74c47be779420 \
  --output powergenome-examples.tar.gz
tar -xzf powergenome-examples.tar.gz --strip-components=1
rm powergenome-examples.tar.gz
```

This downloads the examples, not PowerGenome's source code. The Texas study is
under `example_systems/TX-3-zone/`, with `settings/`, `data/`, `extra_inputs/`,
`resource_groups/`, and `resource_profiles/` already populated. The archive also
includes a CA/AZ study; browse the
[examples repository](https://github.com/PowerGenome/PowerGenome-examples) for
its instructions and other available cases.

Stay in `powergenome-tutorial/` for all commands below; **do not change into
`example_systems/TX-3-zone/`**. The published resource-group paths require this
working directory.

```bash
validate_powergenome --settings_file example_systems/TX-3-zone/settings
run_powergenome \
  --settings_file example_systems/TX-3-zone/settings \
  --results_folder tx_output
```

This will:

1. Load the `.yml` files directly inside `example_systems/TX-3-zone/settings/`
2. Read the study's source tables from `data/` and policy inputs from `extra_inputs/`
3. Aggregate seven base regions into three model zones: `TX1`, `TX2`, and `TX3`
4. Write the `base` case's two planning periods (2030 and 2045) to
   `tx_output/base/Inputs/Inputs_p1/` and `Inputs_p2/`

`data_location` and `input_folder` resolve relative to the parent of the settings
directory (`example_systems/TX-3-zone/`). In contrast, relative `RESOURCE_GROUPS`
and `RESOURCE_GROUP_PROFILES` paths resolve from your current working directory
(`powergenome-tutorial/`).

!!! note "Demonstration data, not a calibrated study"
    The Texas example contains 240 synthetic profile time steps, not a complete
    year. It is intended to demonstrate input generation, not support economic
    conclusions. Do not change regions, years, or technologies without checking
    that every source table and profile covers the new configuration.

### 2. Check completion

Both commands should exit successfully. Confirm that
`tx_output/base/Inputs/Inputs_p1/system/Demand_data.csv` and the corresponding
`Inputs_p2/` directory exist. If the run fails, read the first error in the
terminal and [check the log](../how-to/debugging.md).

The [CLI Reference](../reference/cli.md) covers optional flags. Do not skip
pipeline stages for this first run; the analysis below needs their outputs.

### 3. Monitor Progress

PowerGenome logs progress and warnings to the terminal and writes a more detailed
`tx_output/log.txt`. The example can warn about unassigned resource tags,
missing output columns, and missing transmission investment costs for some
resources. Review warnings rather than treating a completed run as proof of
study quality. Resolve missing costs and prices, unmatched technologies, and
incomplete regional data before using your own results.

## Examining Outputs

PowerGenome generates multiple CSV files formatted for GenX:

### Key Output Files

**resources/Thermal.csv, Vre.csv, Storage.csv, Hydro.csv, Must_run.csv**
: Resource characteristics grouped by resource type; files are written only
  when that type is present

**system/Demand_data.csv**
: Demand by zone and time step

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

period = Path("tx_output/base/Inputs/Inputs_p1")
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

Continue with the downloaded Texas study:

1. [Building a Multi-Region Model](multi-region-model.md): use the same study to
   inspect its regions and change transmission expansion limits.
2. [Working with Renewable Resources](renewable-resources.md): follow its
   separate-settings setup to change wind resource resolution while keeping solar.
3. Explore the other published studies in
   [PowerGenome-examples](https://github.com/PowerGenome/PowerGenome-examples).
4. For your own study, generate starting settings with the
   [System Design tool](https://gschivley.github.io/PowerGenome-tools/web/) and
   [supply the referenced data tables](../how-to/configure-data-tables.md).

Do not change regions or planning years without checking data coverage. The
example's dataset is deliberately small.

## Troubleshooting

### Common Issues

**"Table not found" error**
: Check `data_location` in settings points to correct folder

**"Region not found" error**
: Verify region names in `model_regions` match data tables

**Import errors**
: Activate the environment in `powergenome-tutorial/.venv/`. If installation did
  not complete, rerun `uv pip install powergenome==0.8.0` in that environment.

**Resource-group directory not found**
: Run from `powergenome-tutorial/`, not from inside the Texas study. Check that
  `example_systems/TX-3-zone/resource_groups/` exists.

See [Debugging](../how-to/debugging.md) for more solutions.

## Learning Resources

- **Settings reference**: [Model Definition](../reference/settings/model-definition.md)
- **How settings work**: [Configure Settings](../how-to/configure-settings.md)
- **Understanding the pipeline**: [Data Pipeline Flow](../explanation/pipeline.md)
- **System Design tool**: [Generate settings files for your study area](https://gschivley.github.io/PowerGenome-tools/web/)

## Get Help

- Questions: [groups.io/g/powergenome](https://groups.io/g/powergenome)
- Issues: [GitHub Issues](https://github.com/PowerGenome/PowerGenome/issues)
