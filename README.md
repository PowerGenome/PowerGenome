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

This project pulls data from [PUDL](https://github.com/catalyst-cooperative/pudl). As such, it requires prior installation of PUDL to access some of the convienience functions and the current SQL database. As PUDL transitions away from SQL and becomes pip-installable I will put together an environment.yml file for this project.

## Installation

**Installation instructions are limited at the moment, and will be expanded on in the near future**
Once you have set up a PUDL database, fork and/or clone this repository and pip install an editable version.

## Running code

Settings are controlled in a YAML file. The default is `pudl_data_extraction.yml`.

The code is currently structured in three main modules - `generators.py`, `transmission.py`, and `load_profiles.py`. Functions from each can be imported and used in an interactive environment (e.g. JupyterLab). To run from the command line, activate the  `pudl` conda environment, navigate to the `powergenome` folder, and run

```sh
python extract_pudl_data.py
```

There are currently 3 arguments that can be used after the script name:

- -sf (--settings_file), the name of an alternative settings YAML file.
- -rf (--results_folder), the name of a results subfolder to save files in. If no subfolder is specified the default is to create one named for the current datetime.
- -a (--all_units), if information on all units should be saved in a separate CSV file. The default value is `True`.

An example using all three options would be:

```sh
python extract_pudl_data.py -sf pudl_data_extraction_CA_present.yml -rf CA-present -a True
```
