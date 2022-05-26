# Creating your settings file(s)

Parameters for PowerGenome are defined in one or more YAML files. The sections below follow the multi-file format provided as an example. Many parameters are independent of each other, but some will need to be modified depending on the technologies or model regions that you choose.

## Model definition

The first decisions to make when settings up a system are (1) define the geographic model regions and (2) set the planning periods. Model regions (`model_regions`) consist of one or more [IPM Regions](https://github.com/PowerGenome/PowerGenome/wiki/Geospatial-Mappings#ipm-regions). If a model region is composed of multiple IPM regions, it should be defined in `region_aggregations`.

> **NOTE:** The `model_regions` are used in many settings parameters. Most are (hopefully) obvious, but it is important to include any aggregated regions in the parameters `cost_multiplier_region_map` and `aeo_fuel_region_map`.

Planning periods are defined by the parameters `model_year` and `model_first_planning_year`. The model year is used to calculate hourly demand, fuel prices, and existing generators (if any are retired due to age). Costs for new-build resources in each planning period are determined using the full span from the first year of a planning period through the model year.

Additional parameters that are important when defining a model include the dollar year that all costs should be converted to (`target_usd_year`) and the timezone that generation/demand will be presented in (`utc_offset`). For reference, the Eastern time zone is UTC -5 and Pacific is UTC -8.


## Extra inputs
Some inputs for PowerGenome are supplied in extra CSV files, stored in the `extra_inputs` folder (at the same directory level as the settings).

### Scenarios
PowerGenome is set up to provide inputs for multiple scenarios/cases. `case_id_description_fn` supplies a short ID and full name for each case. `scenario_descriptions_fn` defines the scenarios that are used to vary parameter values.

> **NOTE:** The scenario description file has a "year" column that corresponds to values in the `model_year` parameter, and a "case_id" column that should match the values in the case ID description file.`

### User supplied time series
Users can supply time series data for distributed generation profiles(`distributed_gen_profiles_fn`), flexible demand resources (`demand_response_fn`), and load (`regional_load_fn`) for each region. PowerGenome can supply all inputs except for distributed generation profiles.

> **NOTE:** Any user supplied time series data will need to align with the model regions specified in `model_regions`.

If regional demand is supplied by the user and it already includes the flexible/demand response load, then the parameter `regional_load_includes_demand_response` should be set to `true`.

### Generators
PowerGenome doesn't supply all inputs for generator operations. Parameters such as the minimum power for new resources, ramp rates, and minimum up/down time are provided in `misc_gen_inputs_fn`. Any capacity limits for resources within each region and any interconnection distance/cost is in `capacity_limit_spur_fn`.

> **NOTE:** These files depend on the generator technologies listed in `atb_new_gen`, `modified_atb_new_gen` and `additional_technologies`.

### Other
- `emission_policies_fn`: A combination of energy share requirements (ESR) -- a generic category that covers RPS/CES type policies -- and emission limits.
- `demand_segments_fn`: Includes the value of lost load and segmentation of demand based on willingness to curtail.
- `genx_settings_folder`: The location of settings files for GenX.
- `reserves_fn`: A file with regulation and reserve requirements already formatted for GenX.

> **NOTE:** The emission policies file depends on the `model_year` parameter, the case IDs in `case_id_description_fn`, and the `model_region` parameter.

## Resources

### Existing generators

Users should start by selecting the EIA 923/860 data year(s) (`data_years`) to use -- it's fine to only use the most recent year of data available -- and the type of capacity (`capacity_col`). Capacity types match the column names in PUDL (capacity_mw, winter_capacity_mw, or summer_capacity_mw).

Existing generating units are clustered within each region, with the default number of clusters specified in `num_clusters`. Technologies not included in this list will not be included in the outputs. If you want all units included in the outputs, list None (`~`) instead of a number. The number of clusters in individual model regions is specified in `alt_num_clusters`.

If you want to combine technologies (maybe each individual technology has very little capacity), technology groups can be defined in `tech_groups`. Be sure to set `group_technologies` to `true`. The grouping can be disabled in some regions using the parameter `regional_no_grouping`.

> **NOTE:** Custom named groupings of EIA technologies created in `tech_groups` will need to be added to other parameters such as `tech_fuel_map`, `eia_atb_tech_map`, and `model_tag_values`.

If you want to de-rate the capacity of a technology by its historical capacity factor (e.g. as part of creating a must-run resource), list the technologies under both `capacity_factor_tech` and `derate_techs`, set `derate_capacity` to `true`, and select the data years for calculating capacity factor under `capacity_factor_default_year_filter`. Alternative years can be specified for technologies using `alt_year_filters`.

#### Operation and maintenance costs

Most O&M costs are assigned using [data from NEMS](https://www.eia.gov/analysis/studies/powerplants/generationcost/pdf/full_report.pdf). O&M costs for technologies that are not included in the NEMS report are calculated using NREL ATB data. ATB O&M costs are multiplied by the ratio of heat rates between the generating unit and the ATB technology from the year `atb_existing_year`. EIA technologies are mapped to ATB technologies in `eia_atb_tech_map` (if a list of technologies are mapped, the first technology is used).

#### Hydro

Some regions treat large hydroelectric generators differently from smaller hydro. The boolean parameter `small_hydro` can be used to label plants with capacity less than or equal to `small_hydro_mw` in `small_hydro_regions` as small hydro rather than conventional hydro.

GenX uses a parameter for the ratio of energy to power at hydro resources. The parameters `hydro_factor` and `regional_hydro_factor` are used to calculate this ratio. The average inflow rate (0-1) in each region is multiplied by the hydro factor to determine the rated number of hours of reservoir hydro storage at peak discharge power output.

The storage duration for hydroelectric pumped storage is entered in the more generic `energy_storage_duration` paramter, which can be used for any existing technology.

#### Proposed generators and other updated information

PowerGenome uses annual EIA data for information on existing generators, which can be out of date in some cases. EIA 860m (monthly) data can be used to identify recently announced retirements and newly proposed generators. The version of 860m is specified using `eia_860m_fn`. Only proposed plants with status codes listed under `proposed_status_included` with be added to existing generators. The heat rate and minimum load for proposed technologies can be listed under `proposed_gen_heat_rates` and `proposed_min_load`.

If a user knows of unit retirements that are not listed in EIA 860 or 860m they can be listed under `additional_retirements`.


#### Mapping existing and new technologies

The parameter `eia_atb_tech_map` links existing EIA technology names with ATB (or user) names. It is used when calculating O&M costs, assigning fuels, and assigning fuel startup costs.
## New-build generators

### NREL ATB
NREL's ATB serves as the primary data source for new-build generators. The ATB data year is specified using `atb_data_year`. ATB has both "Market" and "R&D" financial cases -- specify which one to select with `atb_financial_case`. The "Market" financial case will generally give higher weighted average cost of capital (WACC) values. To calculate annuities from the ATB capex and WACC, specify the capital recover period length using `atb_cap_recovery_years` and `alt_atb_cap_recovery_years`.

Users should select the ATB resources for their model using `atb_new_gen`. Note that ATB resources have the format <technology>, <tech_detail>, <cost_case> (e.g. NaturalGas, CCAvgCF, Moderate). The items in `atb_new_gen` are a list of these three elements plus the size (MW) of a single plant.

A user can modify one of the ATB resources (in-place) using `atb_modifiers`. One possible reason would be to modify the capex or O&M costs to represent federal ITC/PTC incentives. Modified copies of ATB resources can also created using `modified_atb_new_gen`.

> **NOTE:** Modfied versions of ATB resources have their own names that need to be included in other parts of the settings file(s). This will include `cost_multiplier_technology_map`, `eia_atb_tech_map`, `new_build_startup_costs`, and `model_tag_values`.

ATB doesn't provide a WACC for battery technologies. Users can either provide the name of a different ATB technology to look up a value or a numeric value with the parameter `atb_battery_wacc`.

Technologies that should not be available in one or more model regions can be specified in `new_gen_not_available`. Note that it isn't necessary to list wind or solar technologies here -- if they aren't included with a region in `renewables_clusters` they won't be in the outputs.

### User technologies

Users can supply their own cost and performance characteristics for other resources in the file `additional_technologies_fn`, which should be located in the `extra_inputs` folder. Only technologies listed in the parameter `additional_new_gen` will be included in a case.

> **NOTE:** User technologies should have values for each planning year in `model_year`. The names of user technologies may need to be included in other settings parameters such as `cost_multiplier_technology_map`, `eia_atb_tech_map`, `new_build_startup_costs`, and `model_tag_values`.

### Regional cost variation

PowerGenome uses regional cost multipliers from EIA to adjust ATB and user technology costs in different model regions. Model regions are mapped to EIA NEMS electricity market module (EMM) regions in `cost_multiplier_region_map`. Technology mappings from EIA's names should be included in `cost_multiplier_technology_map`. As EIA publishes new reports the regional modifiers may change. `cost_multiplier_fn` gives the name of a file located in "PowerGenome/data/cost_multipliers" that should be used.

Users can include their own (additional) version of the regional cost multiplier file with other technology names in their extra inputs folder. The name of this file is given with the parameter `user_regional_cost_multiplier_fn`.

## Startup costs

**NOTE: If you are using a technology that is not from NREL ATB then you may need to modify `new_build_startup_costs`**

The default fuel and O&M costs associated with generator startup events are from NREL documents and should only be modified if you really think you have better data. The only startup costs parameter that users may need to modify is `new_build_startup_costs`, which maps new-build technology names to technologies from NREL reports.

Fuel use in startups (mmbtu/MW) by technology is provided in `startup_fuel_use`. These technologies correspond directly to technologies in the model. Values listed here will be mapped to further technologies using `eia_atb_tech_map`. Variable O&M and monetary costs are provided in `startup_vom_costs_mw` and `startup_costs_per_cost_start_mw`, mapping to technology names in NREL reports. the names provided in these parameters are then mapped to existing and new-build technologies using `existing_startup_costs_tech_map` and `new_build_startup_costs`.

The dollar year of startup costs are given in the parameters `startup_vom_costs_usd_year` and `startup_costs_per_cold_start_usd_year`.

The parameter `startup_costs_type` can be used to select something other than cold start costs included in `startup_costs_per_cold_start_mw`.

## Resource tags

Resources can be assigned values in categories (boolean or other values) that are listed under `model_tag_names`. The default value for these tags is `default_model_tag`, and values for each technology are assigned in `model_tag_values`. If tag values vary by region they can be included in `regional_tag_values`.

This section also included the parameter `MinCapReq`, which has minimum capacity requirements associated with different technologies.

## Fuels

Fuel prices are either taken from AEO scenarios or provided by the user. The AEO year is given by `fuel_eia_aeo_year`, and the dollar year of the AEO fuels is given by `aeo_fuel_usd_year`. The dollar year of AEO prices is usually one year less than the data year (e.g. 2021 dollar year for 2022 data).

AEO fuel prices are collected through the EIA Open Data API. To do this we need to assemble an identifier string using EIA's abbreviations for fuels ("STC", "NG", "DFO", and "U") and scenarios. The PowerGenome parameter `eia_series_fuel_names` maps the fuel codes to more natural names, and scenario codes are mapped in `eia_series_scenario_names`. Because the fuel prices aren't linked to the underlying capacity expansion behavior it is probably sufficient to use a few bounding scenarios like reference, high resource, and low resource.

AEO provides fuel price data for Census Divisions, which are mapped to model regions in `aeo_fuel_region_map`. Users need to make informed choices about the best assignment for their model regions. The parameter `eia_series_region_names` maps AEO fuel region codes to the names provided in `aeo_fuel_region_map`.

Fuels are mapped to generating technologies using `tech_fuel_map`. Further (nested) mappings are performed using `eia_atb_tech_map` -- an ATB technology that is mapped to the EIA technology listed in `tech_fuel_map` will also be assigned the same fuel.

> **NOTE:** If you defined a custom grouping of EIA technologies in `tech_groups`, be sure to assign it a fuel here.

Emission factors for each fuel are listed in `fuel_emission_factors`.

The parameter `carbon_tax` implements a simple cost increase in fuels based on their emission factor.

### CCS fuels

PowerGenome treats CCS fuels as a version of normal fuels with different emission factors per MMBUT. These fuels are named and mapped by using a non-ccs fuel name followed by an underscore and a unique suffix that identifies the fuel type (e.g. `naturalgas_ccs90` could identify a version of natural gas used at facilities that capture 90% of CO2 emissions). The emission factor of normal fuels is modified using the rate specified for each CCS fuel in `ccs_capture_rate`.

The parameter `ccs_disposal_cost` can provide a rough disposal cost (USD/tonne) for captured CO2.

### User defined fuels

EIA data only includes price projections for coal, natural gas, fuel oil, and uranium. Users can provide their own fuel prices, either globally or by region, using the parameter `user_fuel_prices`. The dollar year of user fuel prices can be specified in `user_fuel_usd_year`. Emission factors for user fuels should be included in `fuel_emission_factors`.


## Demand

PowerGenome uses historical demand for each IPM region, constructed from FERC 714 data, as the starting point to calculate future demand. IPM regions are mapped to both the current EIA EMM regions (`future_load_region_map`) and the EMM regions that were used through 2019 (`historical_load_region_maps`).

Historical load is inflated through 2018 using the `historical_load_region_maps` EMM regions. The AEO scenario for AEO year `load_eia_aeo_year` specified in `growth_scenario` is used to calculate future load growth. Alternative growth rates for individual model regions can be listed under `alt_growth_rate`.

Users can provide the full hourly demand for model regions using `regional_load_fn` in "extra_inputs".

## Distributed generation

Distributed generation can be included if users provide normalized generation profiles for each region in `distributed_gen_profiles_fn`, a method for scaling the profiles in `distributed gen_method`, and a numeric scaling value in `distributed_gen_values`. The scaling methods are either "capacity" (MW) or "fraction_load" (the fraction of total load that is met using distributed generation).

If distributed generation profiles are provided, the boolean parameter `dg_as_resource` determines if the profiles are subtracted from demand or included as resources with their own generation profiles. The parameter `avg_distribution_loss` is used to scale up distributed generation when subtracting from demand.

## Flexible load
TODO


## Time clustering

PowerGenome can reduce the full timeseries of generation and demand profiles to a representative subset. A boolean parameter (`reduce_time_domain`) turns this functionality on/off. If it is used, the number of periods is determined by `time_domain_periods` and the number of days (24 hour segments) is specified in `time_domain_days_per_period`. The boolean parameter `include_peak_day` forces the peak demand day into one of the periods if it is true. By default, all generation and demand periods are normalized before the clustering selection method. Demand profiles can be given additional weight  using the parameter `demand_weight_factor`. A user might choose this option because some (or many) of the generation profiles used in the clustering method don't exist yet and may not be selected by the model.


## Scenario management

The parameter `settings_management` is a nested dictionary that controls alternate values for all other settings parameters. The first level should be integer values corresponding to model planning years. Below the model planning years are the column labels from the `scenario_definitions_fn` file. The next level can contain different values from each row in the column. Below this are actual settings parameters, which may have multiple nested levels of their own.

> **NOTE:** If a settings parameter is a list or a dictionary with multiple keys you must include all elements of the parameter in `settings_management`, even ones that do not change. The "value" part of the key:value pair will entirely replace your original parameter value.

For example, consider a scenario definitions file with a column named "solar_cost". The rows of this column have values of "high", "mid", and "low". If the original parameter `atb_new_gen` looks like this:

```
atb_new_gen:
  - [NaturalGas, CTAvgCF, Moderate, 100]
  - [LandbasedWind, Class3, Moderate, 1]
  - [OffShoreWind, Class10, Moderate, 1]
  - [UtilityPV, Class1, Moderate, 1]
```

and your `settings_management` parameter looks like this:

```
settings_management:
  2030:
    solar_cost:
      high:
        atb_new_gen:
          - [UtilityPV, Class1, Conservative, 1]
      mid:
        atb_new_gen:
            - [UtilityPV, Class1, Moderate, 1]
      low:
        atb_new_gen:
            - [UtilityPV, Class1, Advanced, 1]
```

then "UtilityPV" will be the only ATB resource availible in 2030. To keep all resources and only modify the solar cost case, the settings should look like this:

```
settings_management:
  2030:
    solar_cost:
      high:
        atb_new_gen:
          - [NaturalGas, CTAvgCF, Moderate, 100]
          - [LandbasedWind, Class3, Moderate, 1]
          - [OffShoreWind, Class10, Moderate, 1]
          - [UtilityPV, Class1, Conservative, 1]
      mid:
        atb_new_gen:
          - [NaturalGas, CTAvgCF, Moderate, 100]
          - [LandbasedWind, Class3, Moderate, 1]
          - [OffShoreWind, Class10, Moderate, 1]
          - [UtilityPV, Class1, Moderate, 1]
      low:
        atb_new_gen:
          - [NaturalGas, CTAvgCF, Moderate, 100]
          - [LandbasedWind, Class3, Moderate, 1]
          - [OffShoreWind, Class10, Moderate, 1]
          - [UtilityPV, Class1, Advanced, 1]
```

This can lead to conflicts if you want to modify elements of a single parameter using different columns in your scenario definitions file (e.g. one column has solar cost cases and another has offshore wind cost cases). In this situation you should create a single column that controls ATB cost cases (e.g. "atb_cost") and have options for each permutation of values (e.g. "mid_all", "low_solar", "high_solar_low_offshorewind").

```
settings_management:
  2030:
    atb_cost:
      mid_all:
        atb_new_gen:
          - [NaturalGas, CTAvgCF, Moderate, 100]
          - [LandbasedWind, Class3, Moderate, 1]
          - [OffShoreWind, Class10, Moderate, 1]
          - [UtilityPV, Class1, Moderate, 1]
      high_solar_low_offshorewind:
        atb_new_gen:
          - [NaturalGas, CTAvgCF, Moderate, 100]
          - [LandbasedWind, Class3, Moderate, 1]
          - [OffShoreWind, Class10, Advanced, 1]
          - [UtilityPV, Class1, Conservative, 1]
      low_solar:
        atb_new_gen:
          - [NaturalGas, CTAvgCF, Moderate, 100]
          - [LandbasedWind, Class3, Moderate, 1]
          - [OffShoreWind, Class10, Moderate, 1]
          - [UtilityPV, Class1, Advanced, 1]
```

Since the "mid_all" case has the same values as the default parameter it can either be omitted or included in `settings_management` for completeness.