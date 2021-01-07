# PowerGenome

[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
[![Build Status](https://travis-ci.com/gschivley/PowerGenome.svg?token=yTGQ4JcCGLW2GZpmvXHw&branch=master)](https://travis-ci.com/gschivley/PowerGenome)
[![codecov](https://codecov.io/gh/gschivley/PowerGenome/branch/master/graph/badge.svg?token=7KJYLE3jOW)](https://codecov.io/gh/gschivley/PowerGenome)
[![code style black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![DOI](https://zenodo.org/badge/197630673.svg)](https://zenodo.org/badge/latestdoi/197630673)

**Note:** The code and data for PowerGenome are under active development and some changes may break existing functions. Keep up to date with major code and data releases by joining [PowerGenome on groups.io](https://groups.io/g/powergenome).

Power system optimization models can be used to explore the cost and emission implications of different regulations in future energy systems. One of the most difficult parts of running these models is assembling all the data. A typical model will define several regions, each of which need data such as:

- All existing generating units (perhaps grouped into a few discrete clusters within each region)
- Transmission constraints between regions
- Hourly load profiles (including new loads from vehicle and building electrification)
- Hourly generation profiles for wind & solar
- Cost estimates for new generating units

Because computational complexity and run times increase as the number of regions and generating unit clusters increases, a user might want only want to disaggregate regions and generating units close to the primary region of interest. For example, a study focused on clean electricity regulations in New Mexico might combine several states in the Pacific Northwest into a single region while also splitting Arizona combined cycle units into multiple clusters.

The goal of PowerGenome is to let a user make all of these choices in a settings file and then run a single script that generates input files for the power system model. PowerGenome currently generates input files for [GenX](https://energy.mit.edu/wp-content/uploads/2017/10/Enhanced-Decision-Support-for-a-Changing-Electricity-Landscape.pdf), and we hope to expand to other models in the near future.

## Data

PowerGenome uses data from a number of different sources, including EIA, NREL, and EPA. Most of the data are already compiled into a [single sqlite database](https://drive.google.com/open?id=17hTZUKweDMqUi2wvBdubaqVhMRgnN5o5) (see instructions for using it below). There are also a few data files stored in this repository:

- Regional cost multipliers for individual technologies developed by EIA (`data/cost_multipliers/AEO_2020_regional_cost_corrections.csv`).
- A simplified geojson version of EPA's shapefile for IPM regions (`data/ipm_regions_simple.geojson`).
- Information on user-defined technologies, which can be included in outputs. This can be used to define a custom cost case (e.g. $500/kW PV) or a new technology such as natural gas with 100% carbon capture. The CSV files are stored in `data/additional_technologies` and there is a documentation file in that folder describing what to include in the file.

## PUDL Dependency

This project pulls data from [PUDL](https://github.com/catalyst-cooperative/pudl). As such, it requires installation of PUDL to access a normalized sqlite database and some of the convienience PUDL functions.

`catalystcoop.pudl` is included in the `environment.yml` file and will be installed automatically in the conda environment (see instructions below). The data used by PowerGenome have outstripped what is available in the public version of PUDL, so download a modifed version of the [PUDL sqlite database here](https://drive.google.com/file/d/1DamR83bR9DyY-gdXac6xYnp5iFhhzgO4/view?usp=sharing). The package `catalystcoop.pudl` must be version 0.3.0 or above to work with this version of the database.

## Installation

1. Clone this repository to your local machine and navigate to the top level (PowerGenome) folder.

2. Create a conda environment named `powergenome` using the provided `environment.yml` file.

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

5. Download a [modifed version of the PUDL database](https://drive.google.com/file/d/1DamR83bR9DyY-gdXac6xYnp5iFhhzgO4/view?usp=sharing) that includes NREL ATB cost data and is not yet included in PUDL. **NOTE: this database was updated for PowerGenome v0.4.0 and is not compatable with earlier versions.** 

6. Download the [renewable resource data](https://drive.google.com/file/d/1g0Q6TdNp4C12HQJy6pAURzp_oVg0Q7ly/view?usp=sharing) containing generation profiles and capacity for existing and new-build renewable resources. Save and unzip this file. The suggested location for all of the unzipped files is `PowerGenome/data/resource_groups/`. These files will eventually be provided through a data repository with citation information.

7. Get an [API key for EIA's OpenData portal](https://www.eia.gov/opendata/register.php). This key is needed to download projected fuel prices from the 2019 Annual Energy Outlook.

8. Create the file `PowerGenome/powergenome/.env`. To this file, add `PUDL_DB=YOUR_PATH_HERE` (your path to the PUDL database), `EIA_API_KEY=YOUR_KEY_HERE` (your EIA API key) and `RESOURCE_GROUPS=YOUR_PATH_HERE` (your path to where the resource groups data from Step 6 are saved). Quotation marks are only needed if your values contain spaces. The `.env` file is included in `.gitignore` and will not be synced with the repository. See the [SQLAlchemy documentation](https://docs.sqlalchemy.org/en/13/dialects/sqlite.html#connect-strings) for examples of how to format the `PUDL_DB` path (e.g. `sqlite:////<entire path to the folder containing pudl file>/pudl_updated.sqlite`, or `sqlite:///C:/path/to/folder/pudl_updated.sqlite` on Windows). If you get any errors when trying to initite the PUDL database, go back and check your path formatting against the SQLAlchemy documentation examples.

9. Update the Consumer Price Index (CPI) data used to adjust U.S. dollars for inflation (see https://github.com/datadesk/cpi#updating-the-cpi). Because the orignial [cpi](https://github.com/datadesk/cpi) package takes ~30 seconds to load, PowerGenome includes a modified version that only stores a handfull of the BLS CPI tables. Update these tables by starting a `python` session and running:

```python
from powergenome.externals.cpi import cpi as cpi
cpi.update()
```

## Running code

### Settings

Settings are controlled in a YAML file. An example settings file (`test_settings.yml`) and folder with extra user inputs (`extra_inputs`), which set up a small 3-zone model of California and Arizona, are included in the folder `example_system`. Scenario options across different planning years are defined in the files `test_scenario_inputs_short.csv` and `test_scenario_inputs.csv` - the "short" version only includes a subset of the full scenario list covered in the settings file and takes much less time to run.

### Example notebooks

A series of example notebooks are included in [`PowerGenome/notebooks`](/notebooks) describe how to access different functions within PowerGenome to create resource clusters, variable generation profiles, fuel costs, hourly demand, and transmission constraints. They include a description of how the data are compiled and the settings parameters that are required for each type of data.

### Command line interface

The outputs are all formatted for GenX we hope to make the data formatting code more module to allow users to easily switch between outputs for different power system models.

Functions from each module can be imported and used in an interactive environment (e.g. JupyterLab). Examples of how to load data in this way are included in `PowerGenome/notebooks`. To run from the command line, navigate to a project folder that contains a settings file and extra inputs (e.g. `myproject/powergenome`), activate the  `powergenome` conda environment, and use the command `run_powergenome_multiple` with flags for the settings file name and where the results should be saved:

```sh
run_powergenome_multiple --settings_file test_settings.yml --results_folder test_system
```

The command line arguments `--settings_file` and `--results_folder` can be shortened to `-sf` and `-rf` respectively. For all options, run:

```sh
run_powergenome_multiple --help
```

A folder with extra user inputs is required when using the `run_powergenome_multiple` command. The name of this folder is defined in the settings YAML file with the `input_folder` parameter. Look at the files in `PowerGenome/example_system` for a working test case to follow.

If you have previously installed PowerGenome and the `run_powergenome_multiple` command doesn't work, try reinstalling it using `pip install -e .` as described above. If you downloaded the custom PUDL database before May of 2020, some errors may be resolved by downloading a new version.

## Licensing

PowerGenome is released under the [MIT License](https://opensource.org/licenses/MIT). Most data inputs are from US government sources (EIA, EPA, FERC, etc), which should not be [subject to copyright in the US](https://www.usa.gov/government-works). Hourly FERC demand data has been cleaned using [techniques](https://github.com/truggles/EIA_Cleaned_Hourly_Electricity_Demand_Code) developed by Tyler Ruggles and David Farnham, and allocated to IPM regions using [methods developed](https://github.com/catalyst-cooperative/electricity-demand-mapping) by Catalyst Cooperative. Hourly generation profiles for wind and solar resources were created by [Vibrant Clean Energy](https://www.vibrantcleanenergy.com/) and provided without usage restrictions. All PowerGenome data outputs are released under the [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/legalcode) license.

## Contributing

Contributions are welcome! There is significant work to do on this project and additional perspective on user needs will help make it better. If you see something that needs to be improved, [open an issue](https://github.com/gschivley/PowerGenome/issues). If you have questions or need assistance, join [PowerGenome on groups.io](https://groups.io/g/powergenome) and post a message there.

Pull requests are always welcome. To start modifying/adding code, make a fork of this repository, create a new branch, and [submit a pull request](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request-from-a-fork).

All code added to the project should be formatted with [black](https://black.readthedocs.io/en/stable/). After making a fork and cloning it to your own computer, run `pre-commit install` to [install the git hook scripts](https://pre-commit.com/#3-install-the-git-hook-scripts) that will run every time you make a commit. These hooks will automatically run `black` (in case you forgot), fix trailing whitespace, check yaml formatting, etc.
