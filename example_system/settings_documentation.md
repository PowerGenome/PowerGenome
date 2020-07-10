# PowerGenome settings file documentation

This file provides documentation of each parameter in the current version of PowerGenome. It attempts to provide a comprehensive description of the data type and purpose of every parameter.

## Multi-scenario/period parameters

<!-- Combine model_year and model_first_planning_year into a list of tuples -->
### model_year

type: list

description: Integer values. The final year in each planning period. Can be considered the planning year for capacity expansion.

### model_first_planning_year

type: list

description: Integer values. The first year in each planning period. These are combined with the values from `model_year` to define the range of years for each planning period. Cost values (capex, fuel cost, etc) are the average of values from all years within a planning period.

### input_folder

type: str

description: Name of a subfolder - directly below the settings file - with user-supplied scenario information and extra input data.

### case_id_description_fn

type: str

description: Pointer to a csv file with columns `case_id` and `case_description`. `case_id` is a shorthand id for longer case description names.

### scenario_definitions_fn

type: str

description: Pointer to a csv file that starts with mandatory columns `case_id`, which should use `case_id` from the file `case_id_description.csv` as a foreign key, and `year`, which should match the `model_year` values from the settings file. Every `case_id` should have a row for each value of `year` - make sure that the number of rows is equal to the number of unique case ids multiplied by the number of model years. All other columns are user-defined names that refer to the types of parameters in the settings file that will being changed across cases. The values in each column are also user-defined strings describing the scenario (e.g. "high", "mid", "low").

The column names and values in each column are used in the settings parameter `settings_management` to define how the default settings values should be changed across each case.

### distributed_gen_profiles_fn

type: str

description: Pointer to a csv file with normalized hourly generation profiles for distributed generation in all model regions listed in the settings file under `distributed_gen_method` and `distributed_gen_values`.

### demand_response_fn

type: str

description: Pointer to a csv file with hourly (not normalized) profiles for demand response resources in each region/year/scenario. The top four rows are the name of the DR resource (matching key values in the settings parameter `demand_response_resources`), the model year, the scenario name (matching names from `scenario_definitions_fn`), and the model region.

### emission_policies_fn

type: str

description: Pointer to a csv file that describes the emission policies in each case. The first two columns are `case_id` and `year`. Next, the `region` column can either contain the name of a model region or the string "all" (when identical policies are applied to all regions). The column `copy_case_id` indicates if policies from another case should be used (slightly more clear and hopefully fewer errors than just using copy/paste across multiple rows). Other column names should match the columns used in output policy files (e.g. `RPS`, `CES`, etc) and contain numeric values or the string `None`.

**Come back to other external data files**

### reduce_time_domain

type: bool

description: If the load and generation profiles should be reduced from 8760 hourly values to a representative subset of hours.

### time_domain_periods

type: int

description: The number of periods (or clusters) of days. As an example, a system with 10 representative weeks would have 10 periods.

### time_domain_days_per_period

type: int

description: The number of days (24 consecutive hours) to include in each period. If a model uses operational constraints like start-up or shutdown times then multiple days might be appropriate.

### include_peak_day

type: bool

description: If the system-wide peak demand day should be included in one of periods.

### demand_weight_factor

type: int

description: Demand and variable generation profiles are scaled from 0-1 before calculating clusters. Demand profiles are then multiplied by this parameter. Values greater than 1 will weight demand more heavily than variable generation profiles.

### distributed_gen_method

type: dict

description: This is a dictionary with keys that can be any value from `model_regions`. Values define the method by which distributed generation profiles are calculated in each region. Available values are `capacity` and `fraction_load`. This method is used with `distributed_gen_values` and `distributed_gen_profiles_fn`.

### distributed_gen_values

type: dict

description: This dictionary has top-level keys of model years from `model_year`, then regions from `model_region`, then a numeric value representing either `capacity` (MW) or `fraction_load` (0-1).
