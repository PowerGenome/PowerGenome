# Example Systems
This folder contains examples of how to set up PowerGenome settings and input files for different systems. Each system has a folder with YAML settings files and an `extra_inputs` folder where additional user input CSV files are stored.

Refer to the `settings_documentation.md` file in this folder for a description of parameters in each YAML settings file.

## ISONE
This is a 3-zone system for New England with two configurations. One reducing the time representation to 20 days (4 periods of 5 days) and the other is the full 8760 hours. To accomplish this the parameter `reduce_time_domain` is changed from `true` to `false` using the `settings_management` parameter in `scenario_management.yml`.

To run this system download [all data files](https://drive.google.com/drive/folders/1K5GWF5lbe-mKSTUSuJxnFdYGCdyDJ7iE?usp=sharing) listed in the repository README, plus the resource group files and [resource generation profiles](https://drive.google.com/drive/folders/1ZYxnl4U_3HXlYPxm8qlmqyWB8NyC3PpG?usp=share_link). This system uses the [`ipm_regions`](https://drive.google.com/drive/folders/1AjcSM6iwUvCzoZk82oUCsdT1hDqhFIRb?usp=sharing) resource groups.
