# PowerGenome

[![The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![code style black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4426097.svg)](https://doi.org/10.5281/zenodo.4426096)
[![pytest](https://github.com/PowerGenome/PowerGenome/actions/workflows/pytest.yml/badge.svg)](https://github.com/PowerGenome/PowerGenome/actions/workflows/pytest.yml)
[![codecov](https://codecov.io/gh/PowerGenome/PowerGenome/branch/master/graph/badge.svg?token=7KJYLE3jOW)](https://codecov.io/gh/PowerGenome/PowerGenome)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/PowerGenome/PowerGenome/master.svg)](https://results.pre-commit.ci/latest/github/PowerGenome/PowerGenome/master)

Power system optimization models can be used to explore the cost and emission implications of different regulations in future energy systems. One of the most difficult parts of running these models is assembling all the data. A typical model will define several regions, each of which need data such as:

- All existing generating units (perhaps grouped into a few discrete clusters within each region)
- Transmission constraints between regions
- Hourly load profiles (including new loads from vehicle and building electrification)
- Hourly generation profiles for wind & solar
- Cost estimates for new generating units

Because computational complexity and run times increase as the number of regions and generating unit clusters increases, a user might want only want to disaggregate regions and generating units close to the primary region of interest. For example, a study focused on clean electricity regulations in New Mexico might combine several states in the Pacific Northwest into a single region while also splitting Arizona combined cycle units into multiple clusters.

The goal of PowerGenome is to let a user make all of these choices in a settings file and then run a single script that generates input files for the power system model. PowerGenome currently generates input files for [GenX](https://energy.mit.edu/wp-content/uploads/2017/10/Enhanced-Decision-Support-for-a-Changing-Electricity-Landscape.pdf).

## Data

PowerGenome uses data from a number of different sources, including EIA, NREL, and EPA. The data inputs consist of two main components:

1. **Tabular data** - Database tables or files (CSV/Parquet) containing:
   - Existing generating units
   - New resource costs
   - Resource operational parameters
   - Transmission constraints between regions
   - Cost of increasing transmission capacity between regions
   - Hourly demand profiles by region
   - Fuel prices
   - Distributed generation capacity and profiles
   - Value of currency over time

2. **Renewable Resource Data** - Generation profiles and resource group mapping files:
   - `generation_profiles` - Hourly wind/solar generation profiles, either by site or mapped to resource sites
   - `metadata` - Files that assign interconnection costs to specific renewable sites, and maps each to a model region

## Installation

### Option 1: Install from PyPI (Recommended)

The easiest way to install PowerGenome is from PyPI:

```sh
pip install powergenome
```

If you are installing a packaged version of PowerGenome you won't be able to easily use a .env file. Instead, add the environment parameters (`PUDL_DB`, `PG_DB`, etc) to a YAML file in the same folder as the rest of your settings. It doesn't really matter which file these parameters are included in but creating a new file such as `env.yml` will help keep them separate from other settings parameters that might be shared with other PowerGenome users.

After installation, skip to step 4 below to download the required data files.

### Option 2: Install from GitHub with uv (Recommended for Development)

[uv](https://github.com/astral-sh/uv) is a fast Python package and project manager. If you're developing PowerGenome or need the latest features:

1. Clone this repository to your local machine and navigate to the top level (PowerGenome) folder.

2. Install uv if you haven't already. You can use the standalone installer (recommended):

**On macOS and Linux:**

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**On Windows:**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Or install via pip:**

```sh
pip install uv
```

3. Create a virtual environment and install PowerGenome in editable mode:

```sh
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

For development work with additional tools (black, pytest, etc.):

```sh
uv pip install -e ".[dev]"
```

### Option 3: Install from GitHub with conda

If you prefer conda for environment management:

1. Clone this repository to your local machine and navigate to the top level (PowerGenome) folder.

2. Use the provided `environment.yml` file to create a conda environment named `powergenome`. If you don't already use conda it is easiest to download and install [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install).

```sh
conda env create -f environment.yml
```

3. Activate the `powergenome` environment.

```sh
conda activate powergenome
```

4. pip-install an editable version of this project

```sh
pip install -e .
```

## Running code

### Suggested folder structure

It is best practice to set up project folders outside of the cloned repository so that git doesn't track any new/changed files within the upper-level `PowerGenome` folder. Keeping project folders separate from the cloned `PowerGenome` folder will also make it easier to pull changes as they are released.

### Example systems

A few example systems are included at <https://github.com/PowerGenome/PowerGenome-examples>. Each system has settings files in a folder (`settings`) and a folder with extra user inputs (`extra_inputs`). The different example systems are not meant to be accurate for real-world analysis, so please do not blindly use the external data files included with them in your own studies!

### Settings

Settings are controlled in a set of YAML files within a folder or combined into a single file. An example folder of settings files (`settings`) and folder with extra user inputs (`extra_inputs`) are included in each of the example systems. Scenario options across different planning years are defined in the file `test_scenario_inputs.csv`. Documentation on extra inputs is included in the folder of each example system.

### Example notebooks

A series of example notebooks are included in [`PowerGenome/notebooks`](/notebooks) describe how to access different functions within PowerGenome to create resource clusters, variable generation profiles, fuel costs, hourly demand, and transmission constraints. They include a description of how the data are compiled and the settings parameters that are required for each type of data.

### Command line interface

The outputs are all formatted for GenX we hope to make the data formatting code more module to allow users to easily switch between outputs for different power system models.

Functions from each module can be imported and used in an interactive environment (e.g. JupyterLab). Examples of how to load data in this way are included in `PowerGenome/notebooks`. To run from the command line, navigate to a project folder that contains a settings file and extra inputs (e.g. `myproject/powergenome`), activate the  `powergenome` conda environment, and use the command `run_powergenome` with flags for the settings file name and where the results should be saved. Since the `powergenome` package is installed in the `powergenome` conda environment, you can run the command line function from anywhere on your computer (not just within the cloned `PowerGenome` folder).

```sh
run_powergenome --settings_file settings --results_folder test_system
```

The command line arguments `--settings_file` and `--results_folder` can be shortened to `-sf` and `-rf` respectively. For all options, run:

```sh
run_powergenome --help
```

A folder with extra user inputs is required when using the `run_powergenome` command. The name of this folder is defined in the settings YAML file with the `input_folder` parameter. Look at the files in each example system for test cases to follow.

If you have previously installed PowerGenome and the `run_powergenome` command doesn't work, try reinstalling it using `pip install -e .` as described above. If you downloaded the custom PUDL database before May of 2020, some errors may be resolved by downloading a new version.

## Licensing

PowerGenome is released under the [MIT License](https://opensource.org/licenses/MIT). Most data inputs are from US government sources (EIA, EPA, FERC, etc), which should not be [subject to copyright in the US](https://www.usa.gov/government-works). Hourly FERC demand data has been cleaned using [techniques](https://github.com/truggles/EIA_Cleaned_Hourly_Electricity_Demand_Code) developed by Tyler Ruggles and David Farnham, and allocated to IPM regions using [methods developed](https://github.com/catalyst-cooperative/electricity-demand-mapping) by Catalyst Cooperative. Hourly generation profiles for wind and solar resources were created by [Vibrant Clean Energy](https://www.vibrantcleanenergy.com/) and provided without usage restrictions. All PowerGenome data outputs are released under the [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/legalcode) license.

## Contributing

Contributions are welcome! There is significant work to do on this project and additional perspective on user needs will help make it better. If you see something that needs to be improved, [open an issue](https://github.com/gschivley/PowerGenome/issues). If you have questions or need assistance, join [PowerGenome on groups.io](https://groups.io/g/powergenome) and post a message there.

Pull requests are always welcome. To start modifying/adding code, make a fork of this repository, create a new branch, and [submit a pull request](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request-from-a-fork).

All code added to the project should be formatted with [black](https://black.readthedocs.io/en/stable/). After making a fork and cloning it to your own computer, run `pre-commit install` to [install the git hook scripts](https://pre-commit.com/#3-install-the-git-hook-scripts) that will run every time you make a commit. These hooks will automatically run `black` (in case you forgot), fix trailing whitespace, check yaml formatting, etc.
