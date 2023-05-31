# PowerGenome

[![The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![code style black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4426097.svg)](https://doi.org/10.5281/zenodo.4426096)
[![pytest](https://github.com/PowerGenome/PowerGenome/actions/workflows/pytest.yml/badge.svg)](https://github.com/PowerGenome/PowerGenome/actions/workflows/pytest.yml)
[![codecov](https://codecov.io/gh/PowerGenome/PowerGenome/branch/master/graph/badge.svg?token=7KJYLE3jOW)](https://codecov.io/gh/PowerGenome/PowerGenome)

**Note:** The code and data for PowerGenome are under active development and some changes may break existing functions. Keep up to date with major code and data releases by joining [PowerGenome on groups.io](https://groups.io/g/powergenome). And **check out the growing documentation on the [Wiki](https://github.com/PowerGenome/PowerGenome/wiki)** for helpful background information.

Power system optimization models can be used to explore the cost and emission implications of different regulations in future energy systems. One of the most difficult parts of running these models is assembling all the data. A typical model will define several regions, each of which need data such as:

- All existing generating units (perhaps grouped into a few discrete clusters within each region)
- Transmission constraints between regions
- Hourly load profiles (including new loads from vehicle and building electrification)
- Hourly generation profiles for wind & solar
- Cost estimates for new generating units

Because computational complexity and run times increase as the number of regions and generating unit clusters increases, a user might want only want to disaggregate regions and generating units close to the primary region of interest. For example, a study focused on clean electricity regulations in New Mexico might combine several states in the Pacific Northwest into a single region while also splitting Arizona combined cycle units into multiple clusters.

The goal of PowerGenome is to let a user make all of these choices in a settings file and then run a single script that generates input files for the power system model. PowerGenome currently generates input files for [GenX](https://energy.mit.edu/wp-content/uploads/2017/10/Enhanced-Decision-Support-for-a-Changing-Electricity-Landscape.pdf), and we hope to expand to other models in the near future.

## Data

PowerGenome uses data from a number of different sources, including EIA, NREL, and EPA. The data are accessed through a combination of sqlite databases, CSV files, and parquet data files.

1. EIA data on existing generating units are already compiled into a [single sqlite database (PUDL)](https://doi.org/10.5281/zenodo.3653158) (see instructions for using it below).
2. A [second sqlite database (pg_misc_efs)](https://drive.google.com/file/d/1LCB0uwnx6VHrmHQDPH2huLHU6fKXb7kG/view?usp=sharing) has tables with new resource costs from NREL ATB, transmission constraints between IPM regions from EIA, and hourly demand within each IPM region derived from NREL or FERC data.
3. The hourly incremental demand for different flexible demand technologies, and stock values across a range of projection scenarios ([efs_files_utc](https://drive.google.com/file/d/1bS-5LycImdp1AYoS_0uK7tXRHo8TWKCD/view?usp=share_link)).

There are also a few data files stored in this repository:

- Regional cost multipliers for individual technologies developed by EIA (`data/cost_multipliers/AEO_2020_regional_cost_corrections.csv`).
- A simplified geojson version of EPA's shapefile for IPM regions (`data/ipm_regions_simple.geojson`).
- Information on user-defined technologies, which can be included in outputs. This can be used to define a custom cost case (e.g. $500/kW PV) or a new technology such as natural gas with 100% carbon capture. The CSV files are stored in the `extra_inputs` subfolders of each example system. A documentation file in that folder describes what to include in the file.

## PUDL Dependency

This project pulls data from [PUDL](https://github.com/catalyst-cooperative/pudl). As such, it requires installation of PUDL to access a normalized sqlite database and some of the convienience PUDL functions.

`catalystcoop.pudl` is included in the `environment.yml` file and will be installed automatically in the conda environment (see instructions below). Catalyst Cooperative will be creating versioned data releases of PUDL, which can be [accessed on Zenodo](https://doi.org/10.5281/zenodo.3653158). Download the zip file from Zenodo, unzip it, and find the sqlite database under `pudl_data/sqlite/pudl.sqlite`. Note that the version of `catalystcoop.pudl` software may change based on the database version you use. Look on the right-hand side of the zenodo archive to see what software version was used to compile the data. If the version in your conda environment does not match the version used to compile the data, you can change it in the `environment.yml` file or install a [different version](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html#installing-packages) using `conda install catalystcoop.pudl=<your_version>`.

![PUDL software version for database](/docs/_static/pudl_version.png)

**IMPORTANT UPDATE:** As of December 2021, our pinned `catalystcoop.pudl` dependency has bumped from 0.3.* to 0.5.* This version bump is associated with some changes in the PUDL database structure and an increase in the Pandas dependency from 0.25.* to 1.* If you are running an older version of PowerGenome it may be easiest to [remove the existing `powergenome` conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#removing-an-environment) and reinstall it.

```sh
conda remove --name powergenome --all
```

Alternatively, you can [update your existing environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#updating-an-environment).

```sh
conda env update --file environment.yml  --prune
```

Either way, you will need to download the new database files in steps 5/6 below and update your `.env` file.

## Installation

1. Clone this repository to your local machine and navigate to the top level (PowerGenome) folder.

2. Create a conda environment named `powergenome` using the provided `environment.yml` file. If you don't already use conda, [download and install miniconda](https://docs.conda.io/en/latest/miniconda.html). Note that resolving all the dependencies can be slow with conda, so I highly recommend that you [install mamba](https://mamba.readthedocs.io/en/latest/installation.html#existing-conda-install) and use it instead (just sub `mamba` for `conda` below). Mamba installation is easy and will probably take less time than sitting around while conda resolves dependencies.

```sh
conda env create -f environment.yml
```

or if you installed mamba:

```sh
mamba env create -f environment.yml
```

3. Activate the `powergenome` environment.

```sh
conda activate powergenome
```

4. pip-install an editable version of this project

```sh
pip install -e .
```

5. Download [the PUDL database](https://doi.org/10.5281/zenodo.3653158), unzip it, and copy the `/pudl_data/sqlite/pudl.sqlite` to wherever you would like to store PowerGenome data on your computer. The zip file contains other data sets that aren't needed for PowerGenome and can be deleted.  Note that as of December 2021 the most recent version of this database (Data Release v3.0.0) is compatible with `catalystcoop.pudl` version 0.5.* and will not work if an earlier version is included in your conda environment.

6. Download [additional PowerGenome data](https://drive.google.com/file/d/1LCB0uwnx6VHrmHQDPH2huLHU6fKXb7kG/view?usp=sharing) that includes NREL ATB cost data, transmission constraints between IPM regions, and hourly demand for each IPM region. Hourly demand is based on a 2012 weather year and was constructed either directly from FERC 714 data (`load_curves_ferc`) or from NREL EFS data (`load_curves_nrel_efs`) that also sources back to FERC 714. The NREL load curves, which separate hourly demand by sector and subsector, are now the default source for load curves in PowerGenome. See [the wiki](https://github.com/PowerGenome/PowerGenome/wiki/Settings-files#demand) for more information. These files will eventually be provided through a data repository with citation information.

7. Download the [renewable resource data](https://drive.google.com/file/d/1g0Q6TdNp4C12HQJy6pAURzp_oVg0Q7ly/view?usp=sharing) containing generation profiles and capacity for existing and new-build renewable resources. Save and unzip this file. The suggested location for all of the unzipped files is `PowerGenome/data/resource_groups/`. These files will eventually be provided through a data repository with citation information.

8. Download and unzip [data files derived from NREL's EFS](https://drive.google.com/file/d/1bS-5LycImdp1AYoS_0uK7tXRHo8TWKCD/view?usp=sharing)

9. Create the file `PowerGenome/powergenome/.env`. In this file, add:

- `PUDL_DB=YOUR_PATH_HERE` (your path to the PUDL database downloaded in step 5)
- `PG_DB=YOUR_PATH_HERE` (your path to the additional PowerGenome data downloaded in step 6)
- `RESOURCE_GROUPS=YOUR_PATH_HERE` (your path to where the resource groups data from Step 6 are saved)
- `EFS_DATA=YOUR_PATH_HERE` (your path to the folder with EFS derived data files)
Quotation marks are only needed if your values contain spaces. The `.env` file is included in `.gitignore` and will not be synced with the repository. See the [SQLAlchemy documentation](https://docs.sqlalchemy.org/en/13/dialects/sqlite.html#connect-strings) for examples of how to format the `PUDL_DB` and `PG_DB` paths (e.g. `sqlite:////<entire path to the folder containing pudl file>/pudl.sqlite`, or `sqlite:///C:/path/to/folder/pudl.sqlite` on Windows). If you get any errors when trying to initite the PUDL database, go back and check your path formatting against the SQLAlchemy documentation examples.

## Running code

### Suggested folder structure

It is best practice to set up project folders outside of the cloned repository so that git doesn't track any new/changed files within the upper-level `PowerGenome` folder. Try copying one of the example systems (settings file and extra inputs) and modifying it. Copy the `notebooks` folder into your project folder, change the path to the settings file as needed, and run code in the notebooks. This can also be a good way to learn how data are created in PowerGenome and debug problem.

Keeping project folders separate from the cloned `PowerGenome` folder will also make it easier to pull changes as they are released.

### Example systems

A few example systems are included under `PowerGenome/example_systems`. Each system has settings files in a folder (`settings`) and a folder with extra user inputs (`extra_inputs`). The different example systems are not meant to be accurate for real-world analysis, so please do not blindly use the external data files included with them in your own studies!

### Settings

Settings are controlled in a set of YAML files within a folder or combined into a single file. An example folder of settings files (`settings`) and folder with extra user inputs (`extra_inputs`) are included in each of the example systems. Scenario options across different planning years are defined in the file `test_scenario_inputs.csv`. Documentation on extra inputs is included in the folder of each example system.

### Example notebooks

A series of example notebooks are included in [`PowerGenome/notebooks`](/notebooks) describe how to access different functions within PowerGenome to create resource clusters, variable generation profiles, fuel costs, hourly demand, and transmission constraints. They include a description of how the data are compiled and the settings parameters that are required for each type of data.

### Command line interface

The outputs are all formatted for GenX we hope to make the data formatting code more module to allow users to easily switch between outputs for different power system models.

Functions from each module can be imported and used in an interactive environment (e.g. JupyterLab). Examples of how to load data in this way are included in `PowerGenome/notebooks`. To run from the command line, navigate to a project folder that contains a settings file and extra inputs (e.g. `myproject/powergenome`), activate the  `powergenome` conda environment, and use the command `run_powergenome_multiple` with flags for the settings file name and where the results should be saved. Since the `powergenome` package is installed in the `powergenome` conda environment, you can run the command line function from anywhere on your computer (not just within the cloned `PowerGenome` folder).

```sh
run_powergenome_multiple --settings_file settings --results_folder test_system
```

The command line arguments `--settings_file` and `--results_folder` can be shortened to `-sf` and `-rf` respectively. For all options, run:

```sh
run_powergenome_multiple --help
```

A folder with extra user inputs is required when using the `run_powergenome_multiple` command. The name of this folder is defined in the settings YAML file with the `input_folder` parameter. Look at the files in each example system for test cases to follow.

If you have previously installed PowerGenome and the `run_powergenome_multiple` command doesn't work, try reinstalling it using `pip install -e .` as described above. If you downloaded the custom PUDL database before May of 2020, some errors may be resolved by downloading a new version.

## Licensing

PowerGenome is released under the [MIT License](https://opensource.org/licenses/MIT). Most data inputs are from US government sources (EIA, EPA, FERC, etc), which should not be [subject to copyright in the US](https://www.usa.gov/government-works). Hourly FERC demand data has been cleaned using [techniques](https://github.com/truggles/EIA_Cleaned_Hourly_Electricity_Demand_Code) developed by Tyler Ruggles and David Farnham, and allocated to IPM regions using [methods developed](https://github.com/catalyst-cooperative/electricity-demand-mapping) by Catalyst Cooperative. Hourly generation profiles for wind and solar resources were created by [Vibrant Clean Energy](https://www.vibrantcleanenergy.com/) and provided without usage restrictions. All PowerGenome data outputs are released under the [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/legalcode) license.

## Contributing

Contributions are welcome! There is significant work to do on this project and additional perspective on user needs will help make it better. If you see something that needs to be improved, [open an issue](https://github.com/gschivley/PowerGenome/issues). If you have questions or need assistance, join [PowerGenome on groups.io](https://groups.io/g/powergenome) and post a message there.

Pull requests are always welcome. To start modifying/adding code, make a fork of this repository, create a new branch, and [submit a pull request](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request-from-a-fork).

All code added to the project should be formatted with [black](https://black.readthedocs.io/en/stable/). After making a fork and cloning it to your own computer, run `pre-commit install` to [install the git hook scripts](https://pre-commit.com/#3-install-the-git-hook-scripts) that will run every time you make a commit. These hooks will automatically run `black` (in case you forgot), fix trailing whitespace, check yaml formatting, etc.
