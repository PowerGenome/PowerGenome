# Getting Started with PowerGenome

This tutorial will guide you through installing PowerGenome, understanding its structure, running your first model, and examining the outputs.

**Time**: ~30 minutes
**Prerequisites**: Basic command-line knowledge, Python 3.8+

!!! note "Data Sources"
    PowerGenome's examples use US data sources (EIA, NREL ATB, IPM regions), but **the tool works with any geographic region**. Simply provide your own generation data, demand profiles, technology costs, and regional definitions in the expected table formats.

## Installation

Choose one of the following installation methods:

=== "PyPI (Recommended for Users)"

    ```bash
    pip install powergenome
    ```

=== "uv (Recommended for Development)"

    ```bash
    # Install uv
    curl -LsSf https://astral.sh/uv/install.sh | sh  # macOS/Linux
    # or: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

    # Clone and install
    git clone https://github.com/PowerGenome/PowerGenome.git
    cd PowerGenome
    uv venv
    source .venv/bin/activate  # Windows: .venv\Scripts\activate
    uv pip install -e ".[dev]"
    ```

=== "conda"

    ```bash
    git clone https://github.com/PowerGenome/PowerGenome.git
    cd PowerGenome
    conda env create -f environment.yml
    conda activate powergenome
    pip install -e .
    ```

Verify installation:

```bash
run_powergenome_multiple --help
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
    └── case_2030/
        ├── Generators_data.csv
        ├── Load_data.csv
        ├── Fuels_data.csv
        └── ...
```

!!! tip "Keep Projects Separate"
    Create project folders **outside** the PowerGenome repository to avoid git conflicts and make updates easier.

## Running Your First Model

### 1. Use the Test System

PowerGenome includes a minimal test system that's perfect for learning:

```bash
cd /path/to/PowerGenome
run_powergenome_multiple \
    --settings_file tests/test_system/settings \
    --results_folder test_output
```

This will:

1. Load settings from `tests/test_system/settings/`
2. Read data from `tests/test_system/test_data/`
3. Generate output files in `test_output/`

### 2. Understanding Command-Line Flags

```bash
run_powergenome_multiple \
    --settings_file settings \      # Path to settings folder
    --results_folder output \       # Where to save results
    --no-current-gens \            # Skip existing generators (optional)
    --no-load \                     # Skip demand profiles (optional)
    --sort-gens                     # Sort output by resource name (optional)
```

Common flags:

| Flag | Short | Description |
|------|-------|-------------|
| `--settings_file` | `-sf` | Path to settings folder |
| `--results_folder` | `-rf` | Output directory |
| `--no-current-gens` | | Skip existing generator clustering |
| `--no-load` | | Skip demand profile generation |
| `--sort-gens` | | Sort generators by name in output |

See the [CLI Reference](../reference/cli.md) for all options.

### 3. Monitor Progress

PowerGenome provides detailed logging:

```
INFO:powergenome.run_powergenome:Loading settings from tests/test_system/settings
INFO:powergenome.database:Initializing DataManager with data from tests/test_system/test_data
INFO:powergenome.generators:Clustering existing generators
INFO:powergenome.generators:Adding CCS pipeline costs
INFO:powergenome.GenX:Creating load profiles for 1 regions
INFO:powergenome.GenX:Reducing time domain from 8760 to 168 hours
Writing output files to test_output/p1/Inputs/Inputs_p1/
```

## Examining Outputs

PowerGenome generates multiple CSV files formatted for GenX:

### Key Output Files

**Generators_data.csv**
: Resource characteristics, costs, and operational parameters

**Load_data.csv**
: Hourly demand by region

**Fuels_data.csv**
: Fuel prices by region and time period

**Network.csv**
: Transmission topology and constraints

**Generators_variability.csv**
: Hourly generation profiles for variable resources

### Exploring Generators_data.csv

```python
import pandas as pd

# Load generator data
gens = pd.read_csv("test_output/p1/Inputs/Inputs_p1/Generators_data.csv")

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

Key columns in `Generators_data.csv`:

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

Now that you've run your first model:

1. **Modify settings**: Try changing `model_year` or `model_regions`
2. **Add technologies**: Include new resources in `new_resources`
3. **Multi-region model**: Continue to [Building a Multi-Region Model](multi-region-model.md)
4. **Configure renewables**: Try [Working with Renewable Resources](renewable-resources.md)

## Troubleshooting

### Common Issues

**"Table not found" error**
: Check `data_location` in settings points to correct folder

**"Region not found" error**
: Verify region names in `model_regions` match data tables

**Import errors**
: Reinstall with `pip install -e .` from PowerGenome directory

See [Debugging](../how-to/debugging.md) for more solutions.

## Learning Resources

- **Settings reference**: [Model Definition](../reference/settings/model-definition.md)
- **How settings work**: [Configure Settings](../how-to/configure-settings.md)
- **Understanding the pipeline**: [Data Pipeline Flow](../explanation/pipeline.md)
- **Example systems**: [GitHub examples](https://github.com/PowerGenome/PowerGenome/tree/master/example_systems)

## Get Help

- Questions: [groups.io/g/powergenome](https://groups.io/g/powergenome)
- Issues: [GitHub Issues](https://github.com/PowerGenome/PowerGenome/issues)
