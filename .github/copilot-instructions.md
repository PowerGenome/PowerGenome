# PowerGenome AI Coding Agent Instructions

## Project Overview

PowerGenome is a **capacity expansion modeling data pipeline** that generates input files for power system optimization models (primarily GenX). It transforms raw energy data (EIA, NREL ATB, PUDL) into model-ready datasets for studying electricity system futures across different scenarios, regions, and planning years.

**Key insight**: This is NOT a simulator—it's a sophisticated ETL pipeline that clusters generators, builds transmission constraints, creates demand profiles, and manages multi-scenario configurations through YAML-driven workflows.

## Architecture & Data Flow

### Core Pipeline Pattern

1. **Settings → DataManager → Module-specific Processing → GenX Output Files**
2. All data flows through the **DataManager singleton** (in-memory DuckDB) that provides standardized table access
3. Entry point: `run_powergenome` CLI → `run_powergenome.py:main()`

### Major Components

```
powergenome/
├── database.py          # DataManager singleton - centralizes ALL data access
├── settings.py          # Settings class handles YAML configs + scenario management
├── generators.py        # GeneratorClusters class - clusters existing plants, adds new resources
├── GenX.py             # Formats data for GenX model (resource tags, time reduction)
├── load_profiles.py    # Demand curves, DG subtraction, growth projections
├── transmission.py     # Inter-regional constraints, line loss, expansion costs
├── fuels.py            # Fuel price time series from EIA AEO API
├── new_build.py        # New-build resource costs and configuration
├── resource_clusters.py # ClusterBuilder for renewable resource site selection
└── distributed_gen.py   # Rooftop solar capacity & profiles (recent refactor)
```

### Data Architecture (Critical!)

**Only use the Modern DataManager**:

1. **Legacy**: External SQLite databases referenced via `.env` file (`PUDL_DB`, `PG_DB`)
2. **Modern**: DataManager loads from folder/DB configured in settings YAML

**Table Configuration** in settings supports:
- Simple: `generation_table: "generators.csv"`
- Advanced: Dictionary with `table_name`, `columns`, `filters` (DNF logic), `scenario` (convenience filter)

Example:
```yaml
demand_table:
  table_name: demand_timeseries.parquet
  scenario: HighEV
  filters:
    - - [region, '=', 'CA_N']
      - [year, '>=', 2030]
```

The DataManager normalizes these to standardized names (`generation`, `demand`, etc.) accessible via `get_data()`.

## Critical Conventions

### Settings Management

- **Multi-scenario workflows** use `scenario_definitions_fn` CSV to define case variations
- `settings_management` in YAML dynamically swaps parameter values per scenario
- Settings are **frozen dictionaries** after loading—use `Settings` class methods to update
- Model regions defined via `model_regions` + `region_aggregations` (aggregates IPM regions)
- Planning years: `model_year` + `model_first_planning_year` define period ranges

### Generator Clustering

**Existing generators**:
- Clustered by region + technology using k-means (heat rate, capacity as features)
- `num_clusters` sets default, `alt_num_clusters` overrides per region/tech
- `tech_groups` merges similar technologies (e.g., landfill gas + municipal waste)
- `retirement_ages` filters by plant age calculated from operating year
- **Critical**: In myopic models, set retirement ages high (500+) to avoid cluster membership changes between periods

**New generators**:
- Identified by `<technology>_<tech_detail>_<cost_case>` strings
- `new_resources` lists available technologies with unit sizes
- `resource_modifiers` adjusts costs/parameters in-place
- `modified_new_resources` creates renamed copies with modified params
- Renewables use `renewable_clusters` to define resource sites via pre-computed wind/solar profiles

### Regional Patterns

- **Base regions** are the base geography (from a source like ReEDS or IPM)
- **Model regions** aggregate base regions via `region_aggregations`
- Many settings use model region names as keys: `regional_tag_values`, `new_gen_not_available`
- String matching is case-insensitive for technology names but **exact** for regions

### Data Paths

Put these in a settings YAML file (e.g., `env.yml`):
```yaml
data_location: /path/to/data_folder  # Used by DataManager
RESOURCE_GROUP_PROFILES: /path/to/generation_profiles
DISTRIBUTED_GEN_DATA: /path/to/dg_profiles  # Legacy, prefer settings tables
RESOURCE_GROUPS: /path/to/resource_groups   # Can override in settings YAML
```

## Development Workflows

### Running Tests

```bash
# From repo root, activate environment first
conda activate powergenome
pytest tests/                          # All tests
pytest tests/generators_test.py -v    # Specific module
pytest -k "test_cluster" -v           # Pattern matching
```

Tests use **fixtures** with temporary data files. DataManager must be initialized in test setup:
```python
from powergenome.database import initialize_data_manager
initialize_data_manager(settings, data_location=test_data_path)
```

### Running the Pipeline

```bash
# From project folder containing settings/
run_powergenome --settings_file settings --results_folder output_dir

# Common flags
--no-current-gens    # Skip existing generator clustering
--no-load            # Skip demand profile generation
--sort-gens          # Sort output files by resource name
```

**Project structure** (keep separate from repo):
```
my_study/
├── settings/
│   ├── scenario_management.yml
│   ├── generators.yml
│   ├── transmission.yml
│   └── env.yml (optional)
├── extra_inputs/
│   ├── emission_policies.csv
│   ├── misc_gen_inputs.csv
│   └── ...
└── results/
    └── case_2030_high/
        ├── Generators_data.csv
        ├── Load_data.csv
        └── ...
```

### Adding New Technologies

1. Map EIA name to ATB in `eia_atb_tech_map` (for startup costs)
2. Add to `tech_fuel_map` for fuel assignment
3. Set model tags in `model_tag_values` (THERM, VRE, STOR, etc.)
4. Add financial params in `atb_modifiers` or `modified_atb_new_gen`
5. Include in `atb_new_gen` list with size (MW)

### Debugging Common Issues

**"Table not found" errors**:
- Check DataManager initialization in `database.py:_setup_tables()`
- Verify table name mapping in `STANDARD_TABLE_MAPPING`
- Use `list_tables()` to see available tables

**Settings not found**:
- Ensure settings YAML is in a folder, not a single file
- Settings load recursively from folder, later files override earlier ones
- Use `load_settings(Path("settings"))` not `load_settings("settings/file.yml")`

**Regional mismatches**:
- IPM region names must match between settings aggregations and data tables
- Use `find_region_col()` utility to auto-detect region column names
- Check `plant_region_table` for plant→region mappings

## Code Patterns to Follow

### DataFrames Use Snake Case

```python
# Utility functions automatically convert
from powergenome.util import snake_case_col, snake_case_str

df.columns = df.columns.map(snake_case_str)
df["capacity_mw"] = snake_case_col(df["Capacity MW"])
```

### Settings Access Pattern

```python
from powergenome.settings import auto_fill_settings

@auto_fill_settings(settings="SETTINGS")
def my_function(settings, other_param):
    # settings is auto-injected from global/local context
    model_regions = settings.get("model_regions", [])
```

### Cluster Resource Tags (GenX-specific)

Resources must have tags: `THERM`, `VRE`, `MUST_RUN`, `STOR`, `FLEX`, `HYDRO`
```python
from powergenome.GenX import check_resource_tags, RESOURCE_TAGS

# After building generators dataframe
check_resource_tags(gen_df, RESOURCE_TAGS)
```

### Logging Pattern

```python
import logging
logger = logging.getLogger(__name__)

# Use structured messages
logger.info(f"Clustering {len(df)} generators in {region}")
logger.warning(f"No data for region {region} in year {year}")
```

## Files to Watch

- `example_systems/settings_documentation.md` - **Complete settings parameter reference**
- `DISTRIBUTED_GEN_REFACTOR.md` - Explains recent DG architecture changes
- `example_systems/*/settings/` - Working configuration examples
- `tests/test_system/test_data/` - Minimal test datasets showing expected schemas

## GenX Output Format

Final output CSVs follow GenX schema:
- `Generators_data.csv` - Resource attributes, costs, tags
- `Fuels_data.csv` - Fuel prices by region/period
- `Load_data.csv` - Hourly/representative period demand
- `Network.csv` - Transmission topology
- `Minimum_capacity_requirement.csv` - Policy constraints
- `Energy_share_requirement.csv` - RPS/CES policies

**Column naming**: GenX expects specific names like `Existing_Cap_MW`, `Heat_Rate_MMBTU_per_MWh`—do not rename without updating `GenX.py:rename_gen_cols()`

## Testing Strategy

- **Unit tests** for data transformations (see `tests/price_test.py`, `tests/fuel_test.py`)
- **Integration tests** mock full pipeline runs (see `tests/cli_test.py`)
- Use **pytest fixtures** for test data setup
- DataManager must be initialized/torn down in test lifecycle
- Compare outputs to "golden" CSV files in `tests/test_system/`

## Recent Changes (Check CHANGELOG.md)

- **Distributed generation refactor**: Now uses DataManager with separate capacity/profile tables
- **Database abstraction**: Moving from direct SQL to DataManager for all data access
- **Settings flexibility**: Table configs now support dictionary format with filters

---

**When in doubt**: Check `example_systems/` for working configurations, use `load_settings()` to validate YAML syntax, and initialize DataManager before any data operations.
