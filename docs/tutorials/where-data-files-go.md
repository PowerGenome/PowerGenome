# Where Data Files Go

PowerGenome does not ship a dataset. A study is a folder of YAML settings plus
the data files those settings point at. This tutorial takes the Texas example
you ran in [Getting Started](getting-started.md) and gives every input file a
home: which folder it belongs in, which setting finds it, and which published
data deposit supplies the full version.

**Time:** about 15 minutes.
**Prerequisites:** complete [Getting Started](getting-started.md), activate that
Python environment, and work from `powergenome-tutorial/`. This lesson only
inspects files; it does not need another model run.

## Step 1: See the study as a folder

The downloaded example separates configuration, source data, and outputs:

```text
powergenome-tutorial/
├── example_systems/TX-3-zone/       # the study
│   ├── settings/                    # YAML configuration (--settings_file)
│   ├── data/                        # data_location: tabular source tables
│   ├── extra_inputs/                # input_folder: policies, scenarios, segments
│   ├── resource_groups/             # RESOURCE_GROUPS: JSON + site metadata
│   └── resource_profiles/           # RESOURCE_GROUP_PROFILES: hourly profiles
└── tx_output/                       # results from Getting Started
    ├── log.txt
    ├── pg_settings/                 # copy of the run's settings
    └── base/Inputs/Inputs_p1|p2/    # GenX inputs, one folder per period
```

| Folder | Selected by | Holds |
| --- | --- | --- |
| `settings/` | `--settings_file` | Every `.yml` file for the study. A directory loads its immediate `*.yml` files only. |
| `data/` | `data_location` | The tabular tables: generators, costs, demand, fuels, transmission, and so on. |
| `extra_inputs/` | `input_folder` | Scenario definitions, policy CSVs, and other files read outside DataManager. |
| `resource_groups/` | `RESOURCE_GROUPS` | One JSON definition per renewable group, plus its site metadata. |
| `resource_profiles/` | `RESOURCE_GROUP_PROFILES` | Hourly capacity-factor profiles referenced by the group JSON. |
| `tx_output/` | `--results_folder` | Generated model inputs, logs, and the resolved settings. |

The settings folder is not itself a path setting. You pass it on the command
line, and it determines where the relative `data_location` and `input_folder`
paths start (see Step 3).

The example tables are deliberately slimmed and the renewable profiles are
synthetic stand-ins. They demonstrate placement and the workflow, not a
calibrated dataset.

## Step 2: Get the full data for a real study

For a real study, replace the example's slimmed files with the full
[PowerGenome-data](https://github.com/gschivley/PowerGenome-data) releases
published on Zenodo. Three deposits cover the data that the settings reference:

| Deposit | Feeds | Folder in the tools guide | Record |
| --- | --- | --- | --- |
| PowerGenome Input Data (core) | `data_location` | `~/PowerGenome-data/data` | [Zenodo 22233228](https://zenodo.org/records/22233228) |
| PowerGenome Renewable Resource Profiles | `RESOURCE_GROUP_PROFILES` | `~/PowerGenome-data/resource_profiles` | [Zenodo 22235855](https://zenodo.org/records/22235855) |
| PowerGenome Existing Renewable Resource Groups | second `RESOURCE_GROUPS` entry | `~/PowerGenome-data/existing_resource_groups` | [Zenodo 22235662](https://zenodo.org/records/22235662) |

You do not have to fetch these by hand. PowerGenome installs a
`download_zenodo_data` command that downloads a record concurrently, resumes
interrupted transfers, and verifies the checksums Zenodo publishes:

```bash
# Core tables: -sf reads data_location from your settings
download_zenodo_data 22233228 -sf settings

# Profiles and existing groups live in their own folders, so name them with -d
download_zenodo_data 22235855 -d /absolute/path/to/PowerGenome-data/resource_profiles
download_zenodo_data 22235662 -d /absolute/path/to/PowerGenome-data/existing_resource_groups
```

`-sf` writes into `data_location` from the settings you pass, so the core tables
land where PowerGenome already looks for them. If `data_location` is a list, the
first entry is used with a warning. Use `-d` for the profiles and existing-group
deposits, because neither is under `data_location` and `-d` wins when both flags
are given.

Inspect before committing to a multi-GB download:

```bash
download_zenodo_data 22233228 --list-files        # key, size, checksum
download_zenodo_data 22233228 -sf settings --dry-run
download_zenodo_data 22233228 -sf settings --file '*.parquet'
```

`--list-files` downloads nothing and creates no directory. `--file` takes glob
patterns and errors if a pattern matches nothing, so a typo cannot leave you with
a silently empty data folder. Partial transfers stay as `.part` files and resume
when you re-run the same command. See
[Download Data from Zenodo](../how-to/download-data.md) for the full set of
flags, checksum options, and the matching Python API.

Records are versioned, and both DOI forms work with the command:

| Deposit | Version DOI (exact snapshot) | Concept DOI (latest version) |
| --- | --- | --- |
| PowerGenome Input Data | `10.5281/zenodo.22233228` | `10.5281/zenodo.22233227` |
| Renewable Resource Profiles | `10.5281/zenodo.22235855` | `10.5281/zenodo.22235854` |
| Existing Renewable Resource Groups | `10.5281/zenodo.22235662` | `10.5281/zenodo.22235661` |

Use a version DOI to pin a reproducible study and a concept DOI while you are
still setting one up.

The `~/...` values are the tools guide's suggested local layout. In PowerGenome
settings, replace `~` with your absolute home path (see the warning below).

The [System Design tool](https://gschivley.github.io/PowerGenome-tools/web/)
generates settings that reference these deposits. Its
[Downloading Input Data](https://powergenome.github.io/PowerGenome-tools/web_app/#downloading-input-data)
section lists which files each deposit contains, the suggested local folder, and
the exact settings key it feeds. The **Download All** ZIP also includes that
guidance as `DATA_SOURCES.md`, next to the generated `settings/` folder.

With the tools guide's `~` placeholders replaced by absolute paths, the three
settings from the exported `data.yml` look like this:

```yaml
data_location: ["/absolute/path/to/PowerGenome-data/data"]
RESOURCE_GROUPS:
  - "resource_groups_10r_eastern-western_nercr"  # new-build groups, in the ZIP
  - "/absolute/path/to/PowerGenome-data/existing_resource_groups"
RESOURCE_GROUP_PROFILES: "/absolute/path/to/PowerGenome-data/resource_profiles"
```

The first `RESOURCE_GROUPS` entry is the region-specific folder the web app
generates in its Interconnection step and includes in the export ZIP. You do not
download that one separately; the other two rows are the `-d` downloads above.
The profiles deposit also has a
[Google Drive mirror](https://drive.google.com/drive/folders/16PL6HIdYchgyxGwtU6gkXwztBNclBpo1)
if Zenodo is slow.

!!! warning "`~` is not expanded"
    PowerGenome does not call `expanduser` on path settings. A literal
    `~/PowerGenome-data/data` is treated as a relative folder named `~`, so the
    table lookup fails. Use an absolute path, or a path relative to the base
    that setting uses in Step 3. `download_zenodo_data` does expand `~` for its
    `-d` argument, so a path that works on the command line still has to be
    written out in full in YAML.

Data releases are versioned by date (for example `2026.08.14.v1`) and each file
carries its own element version. Zenodo keeps every version; check the release
notes for files that changed before reusing an older download with newer
settings.

## Step 3: Understand the four path settings

PowerGenome deliberately uses different bases for these settings:

| Setting | Relative paths start at | Accepts | Used for tables? |
| --- | --- | --- | --- |
| `data_location` | The settings folder's parent, or a single settings file's directory | One or more folder or database paths | Yes |
| `input_folder` | The settings parent, when you set it | A folder path | Yes |
| `RESOURCE_GROUPS` | The process **working directory** | One or more folders of group JSON | No |
| `RESOURCE_GROUP_PROFILES` | The process **working directory** | One or more folders of profile files | No |

That difference is why the Texas instructions run from `powergenome-tutorial/`:
`data_location: data` resolves to `example_systems/TX-3-zone/data`, while
`RESOURCE_GROUPS: example_systems/TX-3-zone/resource_groups` resolves from your
current directory. Changing into the study folder leaves the first path valid
and breaks the second.

!!! warning "Configure `input_folder` explicitly"
    If you omit `input_folder`, PowerGenome falls back to `extra_inputs` only
    after loading settings, so that default is **not** re-based on the settings
    folder: it resolves from your working directory. A run started outside the
    study then silently looks for a different `extra_inputs/`. Set
    `input_folder: extra_inputs` in your settings to get the settings-parent
    behaviour and remove the dependence on where you invoke the command.

Configured tables are searched in **both** `data_location` and `input_folder`.
Each table must resolve in exactly one place; a second copy with the same name
stops the run with a "found in multiple data locations" error.

Other path rules:

- For a folder of flat files, include `.csv` or `.parquet` in the table setting.
- For SQLite (`.db`, `.sqlite`) or DuckDB (`.duckdb`) sources, use the database
  table name without a file extension.
- `data_location` may be a list, mixing folders and databases.
- A settings directory does not load nested folders or `*.yaml` files, and
  duplicate top-level keys across files replace one another rather than merge.

## Step 4: Put each DataManager table in a search location

Most inputs are DataManager tables. The setting key names the file (or database
table), and DataManager exposes it to the pipeline under a fixed internal name.
DataManager looks for each table in **both** `data_location` and `input_folder`,
so a table setting may point at either place. These are the Texas settings and
the files they load:

| Settings key | DataManager table | Supplies | Texas file |
| --- | --- | --- | --- |
| `generation_table` | `generation` | Existing generating units | `reeds_generators_tx_3_zone.csv` |
| `plant_region_table` | `plant_region` | Plant-to-base-region map | `plant_region_map_tx_3_zone.csv` |
| `resource_cost_table` | `resource_cost` | New-build capital and operating costs | `technology_costs_atb_tx_3_zone.csv` |
| `resource_heat_rate_table` | `resource_heat_rate` | New thermal heat rates | `technology_heat_rates_nrelatb.csv` |
| `operational_constraints_table` | `operational_constraints` | GenX operating overrides | `operational_constraints_tx_3_zone.csv` |
| `transmission_constraints_table` | `transmission_constraints` | Existing transfer capacity | `transmission_capacity_tx_3_zone.csv` |
| `transmission_cost_table` | `transmission_cost` | Line build costs and losses | `network_costs_tx_3_zone.csv` |
| `fuel_price_table` | `fuel_price` | Fuel prices by region and year | `fuel_prices_tx_3_zone.csv` |
| `dollar_year_table` | `dollar_year` | CPI values for dollar-year conversion | `cpi_data.csv` |
| `regional_cost_factor_table` | `regional_cost_factor` | Regional cost multipliers | `regional_cost_multipliers_tx_3_zone.csv` |
| `demand_table` | `demand` | Hourly load by region and year | `reeds_load_tx_3_zone.csv` |
| `supplemental_demand_table` | `supplemental_demand` | Additional hourly load | Not configured |
| `emission_policies_table` | `emission_policies` | CO2 caps and policy requirements | `extra_inputs/emission_policies.csv` |
| `demand_segments_table` | `demand_segments` | Demand curtailment segments | `extra_inputs/demand_segments_voll.csv` |
| `distributed_capacity_table` | `distributed_capacity` | Installed distributed capacity | `distributed_capacity_tx_3_zone.csv` |
| `distributed_profiles_table` | `distributed_profiles` | Distributed generation profiles | `distributed_profiles_tx_3_zone.csv` |

Notes on this table:

- The two rows with an `extra_inputs/` prefix are `input_folder` files, not
  `data_location` files. `emission_policies` and `demand_segments` are ordinary
  DataManager tables that happen to live with the policy inputs; leave them where
  the settings point and do not move them into `data/`. Every other row here
  resolves under `data_location`.
- The Texas `distributed_gen.yml` uses the accepted singular alias
  `distributed_profile_table`; both keys map to `distributed_profiles`.
- The Texas `extra_inputs.yml` configures the policy and demand-segment tables
  with the legacy aliases `emission_policies_fn` and `demand_segments_fn`. Those
  files still resolve under `input_folder`, because DataManager searches both
  locations. Prefer the `*_table` keys in new settings.
- The core Zenodo deposit contains the standard full-data versions of several of
  these files, such as `reeds_generators_transformed.csv`,
  `technology_costs_atb.parquet`, and `reeds_load_transformed.parquet`. Your
  filenames may differ; the setting key is what connects a file to the pipeline.
- Every table also accepts the advanced form with `scenario`, `filters`, and
  `columns`, and a list of sources. See
  [Configure Data Tables](../how-to/configure-data-tables.md) and
  [Data Tables Settings](../reference/settings/data-tables.md).

The columns each table requires are covered by
[Input Table Schemas](../reference/schemas/input-tables.md). Placement and
schema are separate problems: a file in the right folder can still fail the
pipeline if a required column is missing.

## Step 5: Put the other files under `input_folder`

Some settings name a file that is read directly, not through DataManager. These
resolve under `input_folder`:

| Setting | File role | Required when |
| --- | --- | --- |
| `scenario_definitions_fn` | Case and planning-year matrix | Running multiple scenarios |
| `reserves_fn` | Operating reserve requirements | Reserves are modelled from a file |
| `demand_response_fn` | Hourly demand-response profiles | Flexible demand resources are used |
| `regional_load_fn` | User-supplied hourly load curves | You bypass the demand table |
| `user_region_geodata_fn` | User region polygons | Custom region mapping is needed |
| `co2_pipeline_cost_fn` | CO2 pipeline and storage cost percentiles | `co2_pipeline_filters` is set |
| `additional_technologies_fn` | Wide-format custom technology table | Legacy alternative to a cost table |
| `capacity_limit_spur_fn` | Maximum capacity and spur distance | Legacy alternative to `interconnect_capex_mw` |
| `region_wind_pv_cap_fn` | Replacement existing wind/PV capacities | Source fleet needs overriding |

`electrification_stock_fn` is the exception: it names a file relative to the
legacy `EFS_DATA` path (from the environment or settings), not `input_folder`.
`EFS_DATA` and `DISTRIBUTED_GEN_DATA` are legacy roots; distributed generation now
uses the `distributed_*` tables from Step 4, and `EFS_DATA` is only needed for the
electrification-stock workflow.

!!! warning "A present file is not a configured file"
    The Texas `extra_inputs/` folder contains `Reserves.csv`, but
    `reserves_fn` is commented out in `extra_inputs.yml`, so the file is ignored.
    Only a setting brings a file into the run. Add the key intentionally and
    check the log for the file being read.

PowerGenome also writes two caches under `input_folder`:

- `validation_cache/` stores validation fingerprints and results.
- `cluster_assignments/` stores renewable cluster assignments and a hash map.

Both are generated. Do not edit them by hand; deleting either folder is safe and
forces the corresponding work to be recomputed. Disable the validation cache
with `use_validation_cache: false` or `--no-validation-cache`.

## Step 6: Put renewable site data in `resource_groups/` and `resource_profiles/`

Renewable resources start from a JSON definition, not from a table setting. The
Texas solar definition is:

```json
{
    "technology": "utilitypv",
    "metadata": "solar_lcoe_tx_3_zone.csv",
    "profiles": "solar_profiles_tx_3_zone.csv",
    "site_map": "solar_site_map_tx_3_zone.csv"
}
```

Resolution works outwards from the JSON:

- Every `.json` file under each `RESOURCE_GROUPS` directory is discovered
  recursively.
- `metadata` resolves relative to the JSON file's folder.
- `profiles` is searched in each `RESOURCE_GROUP_PROFILES` folder in order, then
  next to the JSON file.
- `site_map` is optional and resolves next to the **resolved profiles file**, so
  keep the map with the profiles if you move them.
- Existing (in-place) resources use separate definitions with
  `"existing": true`; they are matched to existing generator clusters and may
  live in their own group folder.

Adding CSV files without a JSON definition does nothing. The JSON is the entry
point that ties a technology name to its metadata and hourly profiles. See
[Configure Renewable Clusters](../how-to/configure-renewable-clusters.md) for the
required columns and cluster recipes.

## Step 7: Verify which file backs each table

Load the settings and DataManager as the pipeline does, then inspect the resolved
paths and table names. From `powergenome-tutorial/` with the environment active:

```python
from pathlib import Path

from powergenome.database import get_data, initialize_data_manager, list_tables
from powergenome.settings import load_settings

settings = load_settings(Path("example_systems/TX-3-zone/settings"))
print("data_location:", settings["data_location"])
print("input_folder:", settings["input_folder"])

for key in ("generation_table", "demand_table", "fuel_price_table"):
    print(f"{key}: {Path(settings['data_location']) / settings[key]}")

initialize_data_manager(settings, settings["data_location"])
print("tables:", sorted(list_tables()))
print("generation rows:", get_data("generation").shape[0])
```

The printed locations should point at `example_systems/TX-3-zone/data` and
`example_systems/TX-3-zone/extra_inputs` (shown relative to your working
directory because the settings path was relative). The file lines should end in
the filenames from Step 4, and the table list should include names such as
`generation`, `demand`, `fuel_price`, and `distributed_profiles`. The generation
row count should be nonzero.

This snippet only loads tables. It does not generate model inputs.

## Step 8: Know where outputs and caches go

The `--results_folder` path is resolved from your current working directory, not
from the settings folder. A run with the Texas scenario file writes:

```text
tx_output/
├── log.txt                       # full DEBUG log
├── pg_settings/                  # copy of the settings directory used
└── base/                         # case_id
    └── Inputs/
        ├── Inputs_p1/            # planning period 1 (2030)
        │   ├── system/           # Demand, Fuels, Network, variability
        │   ├── resources/        # Thermal, Vre, Storage, ...
        │   ├── policies/         # Caps, reserves, energy-share requirements
        │   ├── extra_outputs/    # cluster assignments and diagnostics
        │   └── powergenome_case_settings.yml
        └── Inputs_p2/            # planning period 2 (2045)
```

Without a scenario-definitions file, the case level is omitted and the period
folders sit directly under `tx_output/Inputs/`. The saved
`powergenome_case_settings.yml` records the resolved settings and source paths for
that period, which is the fastest way to confirm what a run actually used. See
[Output File Schemas](../reference/schemas/output-files.md) for every output
file.

Outputs are snapshots. Editing a source CSV after a run does not change the
inputs already written; rerun `run_powergenome` to regenerate them.

## Step 9: Exercise — one table, one home

Confirm the search rule yourself. Copy the CPI table into the other configured
location:

```bash
cp example_systems/TX-3-zone/data/cpi_data.csv \
  example_systems/TX-3-zone/extra_inputs/cpi_data.csv
```

Run validation without the cache so the table lookup actually happens:

```bash
validate_powergenome \
  --settings_file example_systems/TX-3-zone/settings \
  --no-validation-cache
```

The run stops with an error naming the `dollar_year` table source
`cpi_data.csv` and both locations. DataManager requires exactly one match.

Remove the copy and validate again:

```bash
rm example_systems/TX-3-zone/extra_inputs/cpi_data.csv
validate_powergenome \
  --settings_file example_systems/TX-3-zone/settings \
  --no-validation-cache
```

Validation should complete again. The example can still print its normal
warnings; what matters is that the duplicate-location error is gone.

## Step 10: Check a study before you run it

Use this list when arranging your own files:

- Every `*_table` key names a source that resolves exactly once, in
  `data_location` or `input_folder`. A flat file includes `.csv` or `.parquet`;
  a SQLite or DuckDB source uses the table name without an extension.
- `data_location` and `input_folder` resolve where you expect from the settings
  folder's parent. Remember that `RESOURCE_GROUPS` and
  `RESOURCE_GROUP_PROFILES` resolve from your working directory, and that a
  defaulted `input_folder` resolves from it too.
- No file is duplicated across `data_location` and `input_folder`.
- Every optional `*_fn` setting you rely on is actually present, uncommented, and
  pointing at an existing file.
- Each renewable JSON has a metadata file, a profiles file whose `site_id` values
  match the metadata, and a site map when the IDs differ.
- Source regions and planning years cover `model_regions` and every period.
- The full data deposits are downloaded for real studies, into the folder your
  settings actually name; example files are not a substitute.
- After a download, `download_zenodo_data` reports any failed checksums. Re-run
  the same command to retry only the files that did not land.

Then run `validate_powergenome --settings_file settings` and read its errors and
warnings. [Validate Settings](../how-to/validate-settings.md) explains what the
checks do and do not cover.

## Next steps

- [Building a Multi-Region Model](multi-region-model.md): use the Texas sources
  to change transmission expansion limits.
- [Configure Data Tables](../how-to/configure-data-tables.md): filters, column
  projection, combined tables, and database sources.
- [Input Table Schemas](../reference/schemas/input-tables.md): required columns
  for each table.
- [Debugging](../how-to/debugging.md): what to check when a path or table is not
  found.
- [Download Data from Zenodo](../how-to/download-data.md): every
  `download_zenodo_data` flag, checksum verification, resuming, and the Python
  API.
- [PowerGenome-tools Downloading Input Data](https://powergenome.github.io/PowerGenome-tools/web_app/#downloading-input-data):
  the current Zenodo deposits and their file lists.
