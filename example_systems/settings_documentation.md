# PowerGenome settings file documentation

This file provides documentation of each parameter in the current version of PowerGenome. It attempts to provide a comprehensive description of the data type and purpose of every parameter.

## Model regions and planning periods
These parameters are used to define the regions that will be included in a model - including if/how IPM regions should be aggregated into model regions - and the planning period years.

### model_regions

type: list(str)

description: The names of model regions that will be used in output files. These can either be single IPM regions or the name assigned to a group of (one or more) IPM regions in `region_aggregations`.

### region_aggregations

type: dict(list(str))

description: A dictionary with list values, used to aggregate IPM regions into groups. The keys, which are names of the aggregated regions, should be used in `model_regions`. These region names should also be used in the following parameters:

- `regional_no_grouping`
- `alt_num_clusters`
- `regional_tag_values`
- `new_gen_not_available`
- `new_wind_solar_regional_bins` (depreciated)
- `cost_multiplier_region_map`
- `load_region_map`
- `future_load_region_map`
- `alt_growth_rate`
- `aeo_fuel_region_map`

### regional_capacity_reserves

type: Dict[str, Dict[str, float]]

description: A nested dictionary of capacity reserve constraints for model regions. The top-level keys are of form `CapRes_<num>`. The next level of keys are model regions, with values equal to the capacity reserve requirements.


### cap_res_network_derate_default

type: float

description: The derating of transmission imports used to meet capacity reserve requirements.

<!-- Combine model_year and model_first_planning_year into a list of tuples -->
### model_year

type: List[int]

description: Integer values. The final year in each planning period. Can be considered the planning year for capacity expansion.

### model_first_planning_year

type: List[int]

description: Integer values. The first year in each planning period. These are combined with the values from `model_year` to define the range of years for each planning period. Cost values (capex, fuel cost, etc) are the average of values from all years within a planning period.

## Time reduction
PowerGenome can reduce hourly demand/generation data to a series of time periods/slices. These parameters control how many periods will be used, how many days each period should be, if the day with peak demand should be included, and if/how much demand should be weighted relative to generation profiles when selecting representative time periods.

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

## Multi-scenario/period and user-input parameters
Users need to supply additional information about scenarios and some data that are not yet included in PowerGenome. These parameters point to where those files are located (relative to the settings file).

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

description: Pointer to a csv file with hourly (not normalized) profiles for demand response resources in each region/year/scenario. The top four rows are 1) the name of the DR resource (matching key values in the settings parameter `demand_response_resources`), 2) the model year, 3) the scenario name (scenarios are selected using the `demand_response` settings parameter), and 4) the model region from `model_regions`.

### emission_policies_fn

type: str

description: Pointer to a csv file that describes the emission policies in each case. The first two columns are `case_id` and `year`. Next, the `region` column can either contain the name of a model region or the string "all" (when identical policies are applied to all regions). The column `copy_case_id` indicates if policies from another case should be used (slightly more clear and hopefully fewer errors than just using copy/paste across multiple rows). Other column names should match the columns used in output policy files (e.g. `RPS`, `CES`, etc) and contain numeric values or the string `None`.

### capacity_limit_spur_fn

type: str

description: Provides the maximum capacity and spur-line construction distance for new (non-renewable) resources. Starts with the required columns `region` and `technology`. `cluster` can be omitted, but is required when more than one resource of the same name is used within a region.

The data columns in this file are `spur_miles` and `max_capacity`.

### demand_segments_fn

type: str

description: Describes segments of demand, and the cost for not meeting demand within that segment.

**expand**

### misc_gen_inputs_fn

type: str

description: This file is where users can add extra or miscellaneous inputs for generators. These inputs can include operating constraints like startup/shutdown times, power-to-energy ratios, and any other inputs that are not covered by PowerGenome.

### genx_settings_fn

type: str

description: This is a YAML file used to record input parameters for GenX model runs. A version is copied into each of the final case folders. *This parameter is depreciated in preference of `genx_settings_folder` now that GenX expects a folder of settings files.

### genx_settings_folder

type: str

description: This is a folder of YAML files used to record input parameters for GenX model runs. A version is copied into each of the final case folders.

### regional_load_fn

type: str

description: An optional input file with hourly demand for each region. This file is only required if you don't want to use the data included with PowerGenome.

### regional_load_includes_demand_response

type: bool

description: Optional input, only required if using `regional_load_fn`. Do the user-supplied demand values already include flexible loads?

### distributed_gen_method

type: dict

description: This is a dictionary with keys that can be any value from `model_regions`. Values define the method by which distributed generation profiles are calculated in each region. Available values are `capacity` and `fraction_load`. This method is used with `distributed_gen_values` and `distributed_gen_profiles_fn`.

### distributed_gen_values

type: dict

description: This dictionary has top-level keys of model years from `model_year`, then regions from `model_region`, then a numeric value representing either `capacity` (MW) or `fraction_load` (0-1).

### avg_distribution_loss

type: float

description: Distribution level line-loss is used when subtracting distributed generation from total load. Total load is load at the transmission network, so it includes distribution line loss.

### demand_response_resources

type: dict

description: This nested dictionary has top-level keys of model years from `model_year`. The second level is names of demand response resources. Below the name are they keys:

- `fraction_shiftable` (float)
- `parameter_values` (dict) with key: value pairs for specific columns in the GenX file `Generators_data.csv`.

### demand_response

type: str

description: The scenario name associated with columns in the `demand_response_fn` parameter. This value determines which scenario in the `demand_response_fn` file is used to modify demand curves and create flexible resources in the generators data.

### transmission_investment_cost

type: dict

#### transmission_investment_cost.use_total

type: bool

description: If true, use precalculated interconnection_annuity from resource clusters. If false, calculate interconnection costs using the distances instead and capex provided in `transmission_investment_cost.spur`, `transmission_investment_cost.offshore_spur`, and `transmission_investment_cost.tx`.

#### transmission_investment_cost.[spur, offshore_spur, tx]

type: dict

description: These three dictionaries have keys `capex_mw_mile`, `wacc`, and `investment_years`. Capex values are provided for each model region, and used in conjuction with the weighted average cost of capital (`wacc`) and investment years to calculate annuities for transmission expansion/reinforcement. All three types can be used when calculating interconnection costs for new power plants. `tx` is used to calculate the cost of inter-regional transmission expansion.

### tx_expansion_per_period

type: float, int

description: How much inter-regional transmission can be expanded/increased within a model period. A value of 1.0 allows transmission to double; 0.5 allows for a 50% increase.

### tx_line_loss_100_miles

type: float, int

description: The fraction of electricity lost during each 100 miles of transmission between regions due to line loss. The default value of 0.01 represents a 1% loss per 100 miles.

### partial_ces

type: bool

description: If true, resources are assigned a clean energy standard (CES) credit equal to the difference between their emission rate (in tons/MWh) and a coal plant (assumed to be 1 ton/MWh). Coal plants are not eligible for this credit even if they have emission rates below 1 ton/MWh. Note that units are imperial, not metric tonnes.

### data_years

type: list(int)

description: The years of EIA 923 and 860 data to use when calculating heat rates of generators.

### target_usd_year

type: int

description: The dollar year that all costs will be converted to.

### capacity_col

type: str

description: The capacity of a generator is given in terms of `capacity_mw`, `winter_capacity_mw`, or `summer_capacity_mw`, representing the reported nameplate, winter, and summer capacity. Use one of these three values to determine the total capacity available in each resource cluster. Summer is usually the lowest value, and nameplate the highest.

## Classify some hydro units as "small"
Some regions treat small hydroelectric generators differently for RPS eligibility, or you may want to model them as run-of-river. These parameters are used to rename some hydro resources as "Small Hydroelectic".

### small_hydro

type: bool

description: If PG should separate small hydroelectric dams into their own resource type. Some states treat small hydro different from large hydro for RPS qualification.

### small_hydro_mw

type: int

description: The generator size (MW) below which hydroelectric units are considered "small".

### small_hydro_regions

type: list(str)

description: Regions from `model_regions` that will have hydroelectric generators split into small and conventional. Regions not listed here will not have small hydro split out.

## Clustering existing generators
PowerGenome is set up to cluster existing generating units within regions. These parameters determine how units are clustered within each region.

In addition to clustering units within a technology, users can group several technologies together. This is most useful to combine several technologies with only a few units and little capacity.

### cluster_method

type: str

description: Not implemented yet, leave as `kmeans`

### num_clusters

type: Dict[str, int]

description: The default number of clusters that resources will be split into in every region. More than one cluster might be appropriate if there are a large number of resources with varied ages/sizes/heat rates. Use `alt_num_clusters` to specify a different number of resource clusters in specific regions.

### alt_num_clusters

type: Dict[str, Dict[str, int]]

description: A nested dictionary with keys from `model_regions`, and values that are a key: value pair of the resource name (from EIA) and the number of clusters to create within that region. This parameter lets you set a different number of clusters for a resource within a specific region. You can specify a value of 0 to drop a resource from within a region, which is useful when only a few generators exist and they have extremely high heat rates according to EIA data.

### alt_cluster_method

type: str

description: Not currently in use. Designed to specify different algorithms for clustering existing generators.

### group_technologies

type: bool

description: If `True`, group different technologies together as specified in `tech_groups`. This can be used to combine multiple small capacity technologies that serve a similar purpose or have similar fuel inputs.

### tech_groups

type: dict

description: Key values are the name for a grouping of technologies, values are a list of the EIA technology names to include in the group. This renames all of the listed technologies to the group name.

### regional_no_grouping

type: dict

description: Keys are model regions, values are a list with names of EIA technologies that should not be grouped into the `tech_groups` categories within that region. Exclude or set as None (~) if not used.

example:
```
regional_no_grouping:
  CA_S:
    - Landfill Gas
    - Municipal Solid Waste
```

### capacity_factor_techs

type: list

description: (Not used anymore, fix in code.) Existing technologies that should have their capacity discounted by their average capacity factor (calculated using generation data from `capacity_factor_default_year_filter`).

### capacity_factor_default_year_filter

type: list

description: The years of data to use when calculating capacity factors for each technology cluster.

### alt_year_filters:

type: dict

description: Keys are EIA technology names, values are a list of years to use when calculating the capacity factor of that technology.

### derate_capacity

type: bool

description: If calculated capacity factors should be used to derate the total capacity of a technology.

### energy_storage_duration

type: Dict[str, float]

description: Energy storge duration for existing technologies (e.g. pumped hydro). Keys are the technology name, values are the length of storage duration in hours.

### retirement_ages

type: dict

description: Keys are EIA technology names, values are the maximum age of a generator that will be included in PowerGenome outputs. Generator age is calculated as the difference between `model_year` and the "operating date" year specified in EIA 860. If you want a capacity expansion model to control all retirements for a technology, set the retirement age to some very high value like 500.

**IMPORTANT**
If you are running a myopic model with multiple planning periods, age-based retirements between planning periods can change the units assigned to each cluster. In this situation the heat rates and O&M of a cluster will change because of the units it contains. Economic retirements of capacity from a cluster may not accurately represent the units that should be retired. To avoid this, set all retirement ages to a large value (e.g. 500).

## Model tags

### model_tag_names

type: list

description: This parameter was designed specifically for GenX outputs. The file `Generators_data.csv` has several "model tag" columns that identify attributes of resources (e.g. if they can be committed, if they are hydro, if they are thermal, etc). The function `generators.add_genx_model_tags` (Note: need to rename or move to the `GenX` module) adds columns to a dataframe with the default value from settings parameter `default_model_tag`. Values in each row are then set by technology/resource name according to the settings parameter `model_tag_values`.

### default_model_tag

type: int, float, str

description: This is the default value that each of the `model_tag_names` columns starts with.

### model_tag_values

type: Dict[str, Dict[str, Union[int, float, str]]]

description: This nested dictionary should have top-level keys equal to the `model_tag_names` values. The second level keys will be string matched against EIA, ATB, or user-added technologies/resources. The values are "tags" or other identification infomation populated in the column for each matched resource.

Case insensitive string matching is used to identify technologies, so is not necessary to use full resource names. This is helpful when switching between ATB cost cases, because the resource name will include the cost case. Be careful though, because some resources may be matched against two different tag values.

### regional_tag_values

type: Dict[str, Dict[str, Dict[str, Union[int, float, str]]]]

description: This parameter lets you set different tag values in specific regions. Potential use cases may include changing RPS eligibility or disallowing retirement (or new build) of a technology for a single region.

### MinCapReq

type: Dict[str, Dict[str, Union[int, float, str]]]

description: This parameter specifies minimum capacity requirements. The top-level key (of format `MinCapTag_<*>`) is a model tag linking individual resources to a requirement. The next level of the nested dictionary has keys `description` and `min_mw`, specifying a short description of the capacity requirement and how many MWs are needed to satisfy it.

## New generating resources from NREL ATB
ATB resources are identified using the *technology*, *tech detail*, and *cost case* with a string format of `<technology>_<tech detail>_<cost case>`.

### atb_data_year

type: int

description: The year of ATB data to use (e.g. 2020, 2021, etc.). Note that the `<tech_detail>` and `<cost_case>` names can vary between ATB years.

### atb_financial_case

type: str

description: NREL's ATB provides financial costs for both "Market" and "R&D" scenarios. Select one of them to filter the ATB data.

### atb_cap_recovery_years

type: int

description: The default number of years for capital recovery of new-build generators, used when calculating investment costs.

### alt_atb_cap_recovery_years

type: Dict[str, int]

description: Alternate capital recovery timeframes for specific ATB technologies. The tech names are string matched against ATB string names (identified above).

### atb_existing_year

type: int

description: In some cases (e.g. conbustion turbine variable O&M), ATB data are used to populate costs for existing resources. This parameter sets the year of ATB data to use in these cases.

### atb_modifiers

type: Dict[str, Dict[str, Union[str, list]]]

description: This parameter modifies parameters for ATB technologies in-place (keeping the same name). Top-level keys are user names for each resource and are not used by PowerGenome. Below the top level, a dictionary with the ATB `technology` and `tech_detail` will also include keys of column names that should be modified. The values for each of these keys is a list, where the first value is a string operator name (`add`, `mul`, `truediv`, or `sub`) and the second value is the numeric value.

Valid column names are
- `Var_OM_Cost_per_MWh`
- `Fixed_OM_Cost_per_MWyr`
- `Fixed_OM_Cost_per_MWhyr`
- `Heat_Rate_MMBTU_per_MWh`
- `capex_mw`
- `capex_mwh`
- `wacc_real`

### modified_atb_new_gen

type: Dict[str, Dict[str, Union[str, list]]]

description: Similar to `atb_modifiers`, but this parameter creates a new and modified copy of a technology in ATB. In addition to the keys listed in `atb_modifiers`, this should have `new_technology`, `new_tech_detail`, `new_cost_case`, and `size_mw`.

### atb_battery_wacc

type: Union[str, float]

description: ATB doesn't have a weighted average cost of capital (WACC) for battery storage. Either include a numeric value or `UtilityPV` to use the same WACC as Utility PV.

### eia_atb_tech_map

type: Dict[str, Union[str, List[str]]]

description: This is a mapping of EIA technology names to ATB technology strings (without the cost case). Key values can be a single ATB technology or a list of technologies. It is used to map start-up costs, so be sure to include all custom thermal technologies as part of the dictionary values.

### atb_new_gen

type: List[list]

description: This controls the types of ATB new generation that are included in the generators dataframe. Each resource is specified as a list:
- technology
- tech detail
- cost case
- size of each unit

### new_gen_not_available

type: Dict[str, list]

description: Not all resources are available in all regions. The top-level keys here are model regions, and values are a list of resources (string matched) that should not be included for that region.

### renewable_clusters

type: List[dict]

description: Specify the type of new-build resource (`utilitypv`, `landbasedwind`, or `offshorewind`), maximum capacity (MW), number of clusters, and maximum LCOE (optional) in a model region. The required keys in each dictionary are:
- `region`
- `technology`
- `max_clusters`
- `min_capacity`

For `utilitypv`, users should also include a key `cap_multiplier`. The original capacities were calculated using a resource density of 45 MW/km^2. The default parameter of 0.2 reduces this by 80%.

The optional key `max_lcoe` can be used as a rough cost-cutoff. It uses a pre-calculated LCOE based on 2030 mid-range costs from ATB 2019.

Technologies that have additional characteristics need to have those parameters specified in the dictionary. An example of this is `offshorewind`, where the parameters `turbine_type` (`fixed` or `floating`) and `pref_site` (`0` or `1`) must be included. `pref_site` is a boolean variable indicating if the project site is included in a BOEM lease area or an NREL study of Pacific floating wind.

### cost_multiplier_fn:

type: str

description: The file name containing AEO regional multipliers for different technology types.

### cost_multiplier_region_map

type: Dict[str, list]

description: ATB resource costs do not reflect cost differences across the country. This dictionary maps model regions to regional cost differences reported in EIA AEO 2020. IPM regions have been pre-populated in the example settings file, but all regions from `model_regions` must be included.

### cost_multiplier_technology_map

type: Dict[str, list]

description: ATB technologies need to be matched against the names included in AEO's regional cost multiplier table. All of the ATB technologies are already included in the example file, but any new or modified technologies should be added.

## Load growth

### default_load_year

type: int

description: The year that default load/demand curves are from.

### regular_load_growth_start_year

type: int

description: Historical demand/load curves are inflated from `default_load_year` to this value using historical AEO data.

### historical_load_region_maps

type: Dict[str, list]

description: This parameter matches IPM regions to AEO regions for inflating demand from the historical starting point to current values. Because AEO changed the electricity regions between AEO 2019 and AEO 2020, the keys in this dictionary are different from those in `future_load_region_map`.

### future_load_region_map

type: Dict[str, list]

description: This parameter matches IPM regions to AEO regions for inflating demand from current values to a future planning year. Because AEO changed the electricity regions between AEO 2019 and AEO 2020, the keys in this dictionary are different from those in `future_load_region_map`.

### alt_growth_rate

type: Dict[str, float]

description: This parameter lets you set future growth rates for individual IPM regions (**not** model regions).

## Fuel prices

### aeo_fuel_region_map

type: Dict[str, list]

description: Keys are EIA fuel region names (from `eia_series_region_names`), values are a list of model regions that correspond to each region. Some regional may not fit fully within the EIA regions - this is which region you want to use AEO fuel price data from.

### eia_series_region_names

type: Dict[str, str]

description: A mapping of model names for each AEO fuel region to the string code used in EIA's API.

### eia_series_fuel_names

type: Dict[str, str]

description: A mapping of model names for each AEO fuel type to the string code used in EIA's API.

### eia_aeo_year

type: int

description: The year of EIA AEO data to use for fuel prices. Note that different years have different scenario names so those may need to be modified accordingly in `eia_series_scenario_names`.

### eia_series_scenario_names

type: Dict[str, str]

description: A mapping of the model name for each AEO scenario to the string code used in EIA's API. These may change based on the AEO year used, and not all scenarios are included in the example file. For a full list, look at the eia [open data page](https://www.eia.gov/opendata/qb.php?category=3604304).

### aeo_fuel_scenarios

type: Dict[str, str]

description: The AEO scenario (names from `eia_series_scenario_names`) to use for each fuel type.

### aeo_fuel_usd_year

type: int

description: The dollar year of AEO fuel price data.

### tech_fuel_map

type: Dict[str, str]

description: A mapping of fuel types (from `aeo_fuel_scenarios`) to EIA technology names. ATB technologies are mapped to the EIA names in `eia_atb_tech_map`. Both technologies are assigned a fuel type based on this parameter.

### ccs_fuel_map

type: Dict[str, str]

description: A mapping ATB or user CCS technology names to CCS fuel names (key values on right have to be in the format `<fuel>_<ccslevel>`) where the fuel matches something from `aeo_fuel_scenarios`.

### ccs_capture_rate

type: Dict[str, float]

description: The name of each CCS fuel (from `ccs_fuel_map`) and the capture rate associated with that fuel. Emission rates for each fuel will be based on the base fuel name and be adjusted for the capture rate.

### ccs_disposal_cost

type: Union[int, float]

description: Pipeline and other costs for CCS disposal operations that are added to the fuel price for CCS fuels. Units are USD/ton.

### carbon_tax

type: Union[int, float]

description: This parameter adds a carbon tax cost to fuel costs.

### fuel_emission_factors

type: Dict[str, float]

description: Emission factors provided in the example settings file are from EIA. Coal emission factors are average for the electric power sector.

## Generator startup costs

### startup_vom_costs_mw

type: Dict[str, float]

description: Variable O&M startup costs for different power plant types. The values provided are from the NREL Western wind/solar integration study. Plant types should match the values in `existing_startup_costs_tech_map`, and `new_build_startup_costs`.

### startup_vom_costs_usd_year

type: int

description: Dollar year of costs in `startup_vom_costs_mw` parameter.

### startup_costs_type

type: str

description: Name of the paramter to use for startup costs. The default value in the example documenation is `startup_costs_per_cold_start_mw`, but users can change this parameter (and add other cost sources) if you want.

### startup_costs_per_cold_start_mw

type: Dict[str, Union[int, float]]

description: Costs per cold start for different power plant types. The values provided in the example file are median cold start costs from NREL 2012.

### startup_costs_per_cold_start_usd_year

type: int

description: Dollar year of costs in the dictionary specified by `startup_costs_type`.

**Need to rename the startup costs - maybe remove the "cold" descriptor and just keep as a single dictionary**

### existing_startup_costs_tech_map

type: Dict[str, str]

description: Mapping of EIA technology names (existing generators) to plant types from `startup_vom_costs_mw` and `startup_costs_per_cold_start_mw`.

### new_build_startup_costs

type: Dict[str, str]

description: Mapping of NREL ATB and user-defined technology names (new-build generators) to plant types from `startup_vom_costs_mw` and `startup_costs_per_cold_start_mw`.

### generator_columns

type: List[str]

description: Column names from the new and existing generators dataframes to keep.
