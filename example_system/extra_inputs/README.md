# Description of extra input files

## case_id_description_fn

Contains the two columns `case_id` and `case_description`, which are a short ID and a longer description of each case. The ID is used as a foreign key in other files and the description is used in subfolder names.

## scenario_definitions_fn

Starts with mandatory columns `case_id`, which should use `case_id` from the file `case_id_description.csv` as a foreign key, and `year`, which should match the `model_year` values from the settings file. All other columns are user-defined names that refer to the types of parameters in the settings file that will being changed across cases. The values in each column are also user-defined strings describing the scenario (e.g. "high", "mid", "low").

The column names and values in each column are used in the settings parameter `settings_management` to define how the default settings values should be changed across each case.

## distributed_gen_profiles_fn

Normalized hourly generation profiles for distributed generation in all regions listed in the settings file under `distributed_gen_method` and `distributed_gen_values`.

## demand_response_fn

Hourly (not normalized) profiles for demand response resources in each region/year/scenario. The top four rows are the name of the DR resource (matching key values in the settings parameter `demand_response_resources`), the model year, the scenario name (matching names from `scenario_definitions_fn`), and the region.

## emission_policies_fn

Describes the emission policies in each case. The first two columns are `case_id` and `year`. Next, the `region` column can either contain the name of a model region or the string "all" (when identical policies are applied to all regions). The column `copy_case_id` indicates if policies from another case should be used (slightly more clear and hopefully fewer errors than just using copy/paste across multiple rows). Other column names should match the columns used in output policy files (e.g. `RPS`, `CES`, etc) and contain numeric values or the string `None`.

## capacity_limit_spur_fn

Provides the maximum capacity and spur-line construction distance for new resources. Starts with the required columns `region` and `technology`. `cluster` can be omitted, but is required when more than one resource of the same name is used within a region.

The data columns in this file are `spur_miles` and `max_capacity`.

## demand_segments_fn
