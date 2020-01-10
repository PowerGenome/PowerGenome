# PowerGenome

[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
[![Build Status](https://travis-ci.com/gschivley/PowerGenome.svg?token=yTGQ4JcCGLW2GZpmvXHw&branch=master)](https://travis-ci.com/gschivley/PowerGenome)
[![codecov](https://codecov.io/gh/gschivley/PowerGenome/branch/master/graph/badge.svg?token=7KJYLE3jOW)](https://codecov.io/gh/gschivley/PowerGenome)

Power system optimization models can be used to explore the cost and emission implications of different regulations in future energy systems. One of the most difficult parts of running these models is assembling all the data. A typical model will define several regions, each of which need data such as:

- All existing generating units (perhaps grouped into a few discrete clusters within each region)
- Transmission constraints between regions
- Hourly load profiles (including new loads from vehicle and building electrification)
- Hourly generation profiles for wind & solar
- Cost estimates for new generating units

Because computational complexity and run times increase as the number of regions and generating unit clusters increases, a user might want only want to disaggregate regions and generating units close to the primary region of interest. For example, a study focused on clean electricity regulations in New Mexico might combine several states in the Pacific Northwest into a single region while also splitting Arizona combined cycle units into multiple clusters.

The goal of PowerGenome is to let a user make all of these choices in a settings file and then run a single script that generates input files for the power system model. PowerGenome currently generates input files for [GenX](https://energy.mit.edu/wp-content/uploads/2017/10/Enhanced-Decision-Support-for-a-Changing-Electricity-Landscape.pdf), and we hope to expand to other models in the near future.

## PUDL Dependency

This project pulls data from [PUDL](https://github.com/catalyst-cooperative/pudl). As such, it requires installation of PUDL to access a normalized sqlite database and some of the convienience PUDL functions.

`catalystcoop.pudl` is included in the `environment.yml` file and will be installed automatically in the conda environment (see instructions below). The data used by PowerGenome have outstripped what is available in the public version of PULD, so download a modifed version of the [PUDL sqlite database here](https://drive.google.com/open?id=18tLKbok1-me81SkfWAhSLXmy5HW6RdvI).

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

5. Download a [modifed version of the PULD database](https://drive.google.com/open?id=18tLKbok1-me81SkfWAhSLXmy5HW6RdvI) that includes NREL ATB cost data and is not yet included in PUDL.

6. Once you have downloaded the sqlite database, change the `SETTINGS["pudl_db"]` parameter in `powergenome/params.py` to match the path on your computer.

7. Get an [API key for EIA's OpenData portal](https://www.eia.gov/opendata/register.php). This key is needed to download projected fuel prices from the 2019 Annual Energy Outlook. Create the file `PowerGenome/powergenome/.env` and save the key in this file using the format `EIA_API_KEY=YOUR_KEY_HERE`. No quotation marks are needed around the API string. The `.env` file is included in `.gitignore` and will not be synced with the repository.


## Running code

Settings are controlled in a YAML file. The default is `pudl_data_extraction.yml`.

The code is currently structured in a series of modules - `load_data.py`, `generators.py`, `transmission.py`, `nrelatb.py`, `eia_opendata.py`, `load_profiles.py`, and a couple others. The code and architecture is under active development. While the outputs are all formatted for GenX we hope to make the data formatting code more module to allow users to easily switch between outputs for different power system models.

Functions from each module can be imported and used in an interactive environment (e.g. JupyterLab). To run from the command line, activate the  `pudl` conda environment, navigate to the `powergenome` folder, and run

```sh
python extract_pudl_data.py
```

There are currently 3 arguments that can be used after the script name:

- --settings_file (-sf), the name of an alternative settings YAML file.
- --results_folder (-rf), the name of a results subfolder to save files in. If no subfolder is specified the default is to create one named for the current datetime.
- --no-current-gens, do not load and cluster existing generators.
- --no-gens, do not create the generators file.
- --no-load, do not calcualte hourly load profiles.
- --no-transmission, do not calculate transmission constraints.
- --no-fuel, do not create a fuels file.
- --sort-gens, sort by generator name within a region (existing generators always show up above new generators)


The following example would use settings in the file `pudl_data_extraction_AZ_2030.yml` to create results for only new-build resources and save the files in the subfolder `results/AZ-future-resources`.

```sh
python extract_pudl_data.py -sf pudl_data_extraction_AZ_2030.yml -rf AZ-future-resources --no-current-gens
```
