# PowerGenome
[![Build Status](https://travis-ci.com/gschivley/PowerGenome.svg?token=yTGQ4JcCGLW2GZpmvXHw&branch=master)](https://travis-ci.com/gschivley/PowerGenome)

This project pulls data from PUDL. As such, it requires prior installation of PUDL to access some of the convienience functions and the current SQL database. As PUDL transitions away from SQL and becomes pip-installable I will put together an environment.yml file for this project.

## Running code
Settings are controlled in a YAML file. The default is `pudl_data_extraction.yml`. 

The code is currently structured in three main modules - `generators.py`, `transmission.py`, and `load_profiles.py`. Functions from each can be imported and used in an interactive environment (e.g. JupyterLab). To run from the command line, activate the  `pudl` conda environment, navigate to `src`, and run 

```sh
python extract_pudl_data.py
```

There are currently 2 arguments that can be used after the script name: 1) -sf, the name of an alternative settings YAML file, and 2) -rf, the name of a results subfolder to save files in. If no subfolder is specified the default is to create one named for the current datetime. An example using both options would be

```sh
python extract_pudl_data.py -sf pudl_data_extraction_CA_present.yml -rf CA-present
```
