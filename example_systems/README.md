# Example Systems
This folder contains examples of how to set up PowerGenome settings and input files for different systems. Each system has a YAML settings file and an `extra_inputs` folder where additional user input CSV files are stored. They also have a folder with YAML files that contain settings for running GenX.

Refer to the `settings_documentation.md` file in this folder for a description of parameters in each YAML settings file.

## CA_AZ
This 3-zone system groups the California IPM regions into 2 zones (CA_N and CA_S) and connects them with Arizona (WECC_AZ). Both states have strong transmission connections to southern Nevada in real life.

There are two settings file for this case. The only substantive difference is that `test_settings.yml` is set up to use ATB 2021 and `test_settings_atb2020.yml` uses technology case names from ATB 2020.

Both settings files create cases that explore different natural gas prices, capex for natural gas with CCS and renewables, allowed transmission expansion, and more.

## ISONE
This is another 3-zone system for New England. The only difference in cases is that one creates a full 8760 time series and the other reduces it to a sample 4 periods of 5 days each. To accomplish this the parameter `reduce_time_domain` is changed from `true` to `false` and the name of the GenX settings folder is modified.