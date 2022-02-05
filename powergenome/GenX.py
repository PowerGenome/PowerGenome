"Functions specific to GenX outputs"

from itertools import product
import logging
from pathlib import Path
from typing import Dict, List
import pandas as pd

from powergenome.external_data import (
    load_policy_scenarios,
    load_demand_segments,
    load_user_genx_settings,
)
from powergenome.load_profiles import make_distributed_gen_profiles
from powergenome.time_reduction import kmeans_time_clustering
from powergenome.util import load_settings
from powergenome.nrelatb import investment_cost_calculator

logger = logging.getLogger(__name__)

INT_COLS = [
    "Inv_Cost_per_MWyr",
    "Fixed_OM_Cost_per_MWyr",
    "Inv_Cost_per_MWhyr",
    "Fixed_OM_Cost_per_MWhyr",
    "Line_Reinforcement_Cost_per_MW_yr",
    "Up_Time",
    "Down_Time",
]

COL_ROUND_VALUES = {
    "Var_OM_Cost_per_MWh": 2,
    "Var_OM_Cost_per_MWh_in": 2,
    "Start_Cost_per_MW": 0,
    "Cost_per_MMBtu": 2,
    "CO2_content_tons_per_MMBtu": 5,
    "Cap_size": 2,
    "Heat_Rate_MMBTU_per_MWh": 2,
    "distance_mile": 4,
    "Line_Max_Reinforcement_MW": 0,
    "distance_miles": 1,
    "distance_km": 1,
    "Existing_Cap_MW": 1,
    "Existing_Cap_MWh": 1,
    "Max_Cap_MW": 1,
    "Max_Cap_MWh": 1,
    "Min_Cap_MW": 1,
    "Min_Cap_MWh": 1,
    "Line_Loss_Percentage": 4,
}

# RESOURCE_TAGS = ["THERM", "VRE", "MUST_RUN", "STOR", "FLEX", "HYDRO", "LDS"]
RESOURCE_TAGS = ["THERM", "VRE", "MUST_RUN", "STOR", "FLEX", "HYDRO"]

def create_policy_req(settings: dict, col_str_match: str) -> pd.DataFrame:
    model_year = settings["model_year"]
    case_id = settings["case_id"]

    policies = load_policy_scenarios(settings)
    policy_cols = [c for c in policies.columns if col_str_match in c]
    if len(policy_cols) == 0:
        return None

    year_case_policy = policies.loc[(case_id, model_year), ["region"] + policy_cols]
    if isinstance(year_case_policy, pd.DataFrame):
        year_case_policy = year_case_policy.dropna(subset=policy_cols)
    elif isinstance(year_case_policy, pd.Series):
        year_case_policy = year_case_policy.dropna()
    else:
        raise TypeError(
            "Somehow the object 'year_case_policy' is not a dataframe of a series."
            "Please submit this as an issue in the PowerGenome repository."
            f"{year_case_policy}"
        )
    # Bug where multiple regions for a case will return this as a df, even if the policy
    # for this case applies to all regions (code below expects a Series)
    ycp_shape = year_case_policy.shape
    if ycp_shape[0] == 1 and len(ycp_shape) > 1:
        year_case_policy = year_case_policy.squeeze()  # convert to series

    zones = settings["model_regions"]
    zone_num_map = {
        zone: f"z{number + 1}" for zone, number in zip(zones, range(len(zones)))
    }

    zone_cols = ["Region_description", "Network_zones"] + policy_cols
    zone_df = pd.DataFrame(columns=zone_cols, dtype=float)
    zone_df["Region_description"] = zones
    zone_df["Network_zones"] = zone_df["Region_description"].map(zone_num_map)
    # If there is only one region, assume that the policy is applied across all regions.
    if isinstance(year_case_policy, pd.Series):
        logger.info(
            "Only one zone was found in the emissions policy file."
            " The same emission policies are being applied to all zones."
        )
        for col, value in year_case_policy[policy_cols].iteritems():
            if "CO_2_Max_Mtons" in col:
                zone_df.loc[:, col] = 0
                if value > 0:
                    zone_df.loc[0, col] = value
            else:
                zone_df.loc[:, col] = value
    else:
        for region, col in product(
            year_case_policy["region"].unique(), year_case_policy[policy_cols].columns
        ):
            zone_df.loc[
                zone_df["Region_description"] == region, col
            ] = year_case_policy.loc[year_case_policy.region == region, col].values[0]

    # zone_df = zone_df.drop(columns="region")

    return zone_df


def create_regional_cap_res(settings: dict) -> pd.DataFrame:
    """Create a dataframe of regional capacity reserve constraints from settings params

    Parameters
    ----------
    settings : dict
        PowerGenome settings dictionary with parameters 'model_regions' and
        'regional_capacity_reserves'

    Returns
    -------
    pd.DataFrame
        A dataframe with a 'Network_zones' column and one 'CapRes_*' for each capacity
        reserve constraint

    Raises
    ------
    KeyError
        A region listed in a capacity reserve constraint is not a valid model region
    """

    if not settings.get("regional_capacity_reserves"):
        return None
    else:
        zones = settings["model_regions"]
        zone_num_map = {
            zone: f"z{number + 1}" for zone, number in zip(zones, range(len(zones)))
        }
        cap_res_cols = list(settings["regional_capacity_reserves"])
        cap_res_df = pd.DataFrame(index=zones, columns=["Network_zones"] + cap_res_cols)
        cap_res_df["Network_zones"] = cap_res_df.index.map(zone_num_map)

        for col, cap_res_dict in settings["regional_capacity_reserves"].items():
            for region, val in cap_res_dict.items():
                if region not in zones:
                    raise KeyError(
                        f"The region {region} in 'regional_capacity_reserves', {col} "
                        "is not a valid model region."
                    )
                cap_res_df.loc[region, col] = val

        cap_res_df = cap_res_df.fillna(0)

        return cap_res_df


def label_cap_res_lines(path_names: List[str], dest_regions: List[str]) -> List[int]:
    """Label if each transmission line is part of a capacity reserve constraint and
    if line flow is into or out of the constraint region.

    Parameters
    ----------
    path_names : List[str]
        String names of transmission lines, with format <region>_to_<region>
    dest_regions : List[str]
        Names of model regions, corresponding to region names used in 'path_names'

    Returns
    -------
    List[int]
        Same length as 'path_names'. Values of 1 mean the line connects a region within
        a capacity reserve constraint to a region outside the constraint. Values of -1
        mean it connects a region outside a capacity reserve constraint to a region
        within the constraint. Values of 0 mean it connects two regions that are both
        within or outside the constraint.
    """
    cap_res_list = []
    for name in path_names:
        s_r = name.split("_to_")[0]
        e_r = name.split("_to_")[-1]
        if (s_r in dest_regions) and not (e_r in dest_regions):
            cap_res_list.append(1)
        elif (e_r in dest_regions) and not (s_r in dest_regions):
            cap_res_list.append(-1)
        else:
            cap_res_list.append(0)

    assert len(cap_res_list) == len(path_names)

    return cap_res_list


def add_cap_res_network(tx_df: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """Add capacity reserve colums to the transmission dataframe (Network.csv)

    Parameters
    ----------
    tx_df : pd.DataFrame
        Transmission dataframe with rows for each transmission line, a column
        'transmission_path_name', and columns 'z1', 'z2', etc for each model region. The
        'z*' columns have values of -1 (line is outbound from region), 0 (not connected),
        or 1 (inbound to region).
    settings : dict
        PowerGenome settings, with parameters 'model_regions', 'regional_capacity_reserves',
        and 'cap_res_network_derate_default'.


    Returns
    -------
    pd.DataFrame
        A copy of the input dataframe with additional columns 'CapRes_*', 'DerateCapRes_*',
        and 'CapRes_Excl_*' for each capacity reserve constraint
    """
    if (
        "Transmission Path Name" in tx_df.columns
        and "transmission_path_name" not in tx_df.columns
    ):
        tx_df["transmission_path_name"] = tx_df["Transmission Path Name"]
    original_cols = tx_df.columns.to_list()

    zones = settings["model_regions"]
    zone_num_map = {
        zone: f"z{number + 1}" for zone, number in zip(zones, range(len(zones)))
    }
    path_names = tx_df["transmission_path_name"].to_list()
    policy_nums = []

    # Loop through capacity reserve constraints (CapRes_*) and determine network
    # parameters for each
    for cap_res in settings.get("regional_capacity_reserves", {}):
        cap_res_num = int(cap_res.split("_")[-1])  # the number of the capres constraint
        policy_nums.append(cap_res_num)
        dest_regions = list(
            settings["regional_capacity_reserves"][cap_res].keys()
        )  # list of regions in the CapRes
        # May add ability to have different values by CapRes and line in the future
        tx_df[f"DerateCapRes_{cap_res_num}"] = settings.get(
            "cap_res_network_derate_default", 0.95
        )
        tx_df[f"CapRes_Excl_{cap_res_num}"] = label_cap_res_lines(
            path_names, dest_regions
        )

    policy_nums.sort()
    derate_cols = [f"DerateCapRes_{n}" for n in policy_nums]
    excl_cols = [f"CapRes_Excl_{n}" for n in policy_nums]

    return tx_df[original_cols + derate_cols + excl_cols].fillna(0)


def add_emission_policies(transmission_df, settings):
    """Add emission policies to the transmission dataframe

    Parameters
    ----------
    transmission_df : DataFrame
        Zone to zone transmission constraints
    settings : dict
        User-defined parameters from a settings file. Should have keys of `input_folder`
        (a Path object of where to find user-supplied data) and
        `emission_policies_fn` (the file to load).
    DistrZones : [type], optional
        Placeholder setting, by default None

    Returns
    -------
    DataFrame
        The emission policies provided by user next to the transmission constraints.
    """

    model_year = settings["model_year"]
    case_id = settings["case_id"]

    policies = load_policy_scenarios(settings)
    year_case_policy = policies.loc[(case_id, model_year), :]

    # Bug where multiple regions for a case will return this as a df, even if the policy
    # for this case applies to all regions (code below expects a Series)
    ycp_shape = year_case_policy.shape
    if ycp_shape[0] == 1 and len(ycp_shape) > 1:
        year_case_policy = year_case_policy.squeeze()  # convert to series

    zones = settings["model_regions"]
    zone_num_map = {
        zone: f"z{number + 1}" for zone, number in zip(zones, range(len(zones)))
    }

    zone_cols = ["Region description", "Network_zones"] + list(policies.columns)
    zone_df = pd.DataFrame(columns=zone_cols)
    zone_df["Region description"] = zones
    zone_df["Network_zones"] = zone_df["Region description"].map(zone_num_map)

    # Add code here to make DistrZones something else!
    # If there is only one region, assume that the policy is applied across all regions.
    if isinstance(year_case_policy, pd.Series):
        logger.info(
            "Only one zone was found in the emissions policy file."
            " The same emission policies are being applied to all zones."
        )
        for col, value in year_case_policy.iteritems():
            if col == "CO_2_Max_Mtons":
                zone_df.loc[:, col] = 0
                if value > 0:
                    zone_df.loc[0, col] = value
            else:
                zone_df.loc[:, col] = value
    else:
        for region, col in product(
            year_case_policy["region"].unique(), year_case_policy.columns
        ):
            zone_df.loc[
                zone_df["Region description"] == region, col
            ] = year_case_policy.loc[year_case_policy.region == region, col].values[0]

    zone_df = zone_df.drop(columns="region")

    network_df = pd.concat([zone_df, transmission_df], axis=1)

    return network_df


def add_misc_gen_values(gen_clusters, settings):
    path = Path(settings["input_folder"]) / settings["misc_gen_inputs_fn"]
    misc_values = pd.read_csv(path)
    misc_values = misc_values.fillna("skip")

    for resource in misc_values["Resource"].unique():
        # resource_misc_values = misc_values.loc[misc_values["Resource"] == resource, :].dropna()

        for col in misc_values.columns:
            if col == "Resource":
                continue
            value = misc_values.loc[misc_values["Resource"] == resource, col].values[0]
            if value != "skip":
                gen_clusters.loc[
                    gen_clusters["Resource"].str.contains(resource, case=False), col
                ] = value

    return gen_clusters


def reduce_time_domain(
    resource_profiles, load_profiles, settings, variable_resources_only=True
):

    demand_segments = load_demand_segments(settings)

    if settings.get("reduce_time_domain"):
        days = settings["time_domain_days_per_period"]
        time_periods = settings["time_domain_periods"]
        include_peak_day = settings["include_peak_day"]
        load_weight = settings["demand_weight_factor"]

        results, representative_point, _ = kmeans_time_clustering(
            resource_profiles=resource_profiles,
            load_profiles=load_profiles,
            days_in_group=days,
            num_clusters=time_periods,
            include_peak_day=include_peak_day,
            load_weight=load_weight,
            variable_resources_only=variable_resources_only,
        )

        reduced_resource_profile = results["resource_profiles"]
        reduced_resource_profile.index.name = "Resource"
        reduced_resource_profile.index = range(1, len(reduced_resource_profile) + 1)
        reduced_load_profile = results["load_profiles"]
        time_series_mapping = results["time_series_mapping"]

        time_index = pd.Series(data=reduced_load_profile.index + 1, name="Time_Index")
        sub_weights = pd.Series(
            data=[x * (days * 24) for x in results["ClusterWeights"]],
            name="Sub_Weights",
        )
        hours_per_period = pd.Series(data=[days * 24], name="Timesteps_per_Rep_Period")
        subperiods = pd.Series(data=[time_periods], name="Rep_Periods")
        reduced_load_output = pd.concat(
            [
                demand_segments,
                subperiods,
                hours_per_period,
                sub_weights,
                time_index,
                reduced_load_profile.round(0),
            ],
            axis=1,
        )

        return (
            reduced_resource_profile,
            reduced_load_output,
            time_series_mapping,
            representative_point,
        )

    else:
        time_index = pd.Series(data=range(1, 8761), name="Time_Index")
        sub_weights = pd.Series(data=[1], name="Sub_Weights")
        hours_per_period = pd.Series(data=[168], name="Timesteps_per_Rep_Period")
        subperiods = pd.Series(data=[1], name="Rep_Periods")

        # Not actually reduced
        load_output = pd.concat(
            [
                demand_segments,
                subperiods,
                hours_per_period,
                sub_weights,
                time_index,
                load_profiles.reset_index(drop=True),
            ],
            axis=1,
        )

        return resource_profiles, load_output, None, None


def network_line_loss(transmission: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """Add line loss percentage for each network line between regions.

    Parameters
    ----------
    transmission : pd.DataFrame
        One network line per row with a column "distance_mile"
    settings : dict
        User-defined settings with a parameter "tx_line_loss_100_miles"

    Returns
    -------
    pd.DataFrame
        Same as input but with the new column 'Line_Loss_Percentage'
    """
    if "tx_line_loss_100_miles" not in settings:
        raise KeyError(
            "The parameter 'tx_line_loss_100_miles' is required in your settings file."
        )
    loss_per_100_miles = settings["tx_line_loss_100_miles"]
    if "distance_mile" in transmission.columns:
        distance_col = "distance_mile"
    elif "distance_km" in transmission.columns:
        distance_col = "distance_km"
        loss_per_100_miles *= 0.62137
        logger.info("Line loss per 100 miles was converted to km.")
    else:
        raise KeyError("No distance column is available in the transmission dataframe")

    transmission["Line_Loss_Percentage"] = (
        transmission[distance_col] / 100 * loss_per_100_miles
    )

    return transmission


def network_reinforcement_cost(
    transmission: pd.DataFrame, settings: dict
) -> pd.DataFrame:
    """Add transmission line reinforcement investment costs (per MW-mile-year)

    Parameters
    ----------
    transmission : pd.DataFrame
        One network line per row with columns "transmission_path_name" and
        "distance_mile".
    settings : dict
        User-defined settings with dictionary "transmission_investment_cost.tx" with
        keys "capex_mw_mile", "wacc", and "investment_years".

    Returns
    -------
    pd.DataFrame
        Same as input but with the new column 'Line_Reinforcement_Cost_per_MW_yr'
    """

    cost_dict = (
        settings.get("transmission_investment_cost", {})
        .get("tx", {})
        .get("capex_mw_mile")
    )
    if not cost_dict:
        raise KeyError(
            "No value for transmission reinforcement costs is included in the settings."
            " These costs are included under transmission_investment_costs.tx."
            "capex_mw_mile.<model_region>. See the `test_settings.yml` file for an "
            "example."
        )
    origin_region_cost = (
        transmission["transmission_path_name"].str.split("_to_").str[0].map(cost_dict)
    )
    dest_region_cost = (
        transmission["transmission_path_name"].str.split("_to_").str[-1].map(cost_dict)
    )

    # Average the costs per mile between origin and destination regions
    line_capex = (origin_region_cost + dest_region_cost) / 2
    line_wacc = (
        settings.get("transmission_investment_cost", {}).get("tx", {}).get("wacc")
    )
    if not line_wacc:
        raise KeyError(
            "No value for the transmission weighted average cost of capital (wacc) is "
            "included in the settings."
            "This numeric value is included under transmission_investment_costs.tx."
            "wacc. See the `test_settings.yml` file for an "
            "example."
        )
    line_inv_period = (
        settings.get("transmission_investment_cost", {})
        .get("tx", {})
        .get("investment_years")
    )
    if not line_inv_period:
        raise KeyError(
            "No value for the transmission investment period is included in the settings."
            "This numeric value is included under transmission_investment_costs.tx."
            "investment_years. See the `test_settings.yml` file for an example."
        )
    line_inv_cost = (
        investment_cost_calculator(line_capex, line_wacc, line_inv_period)
        * transmission["distance_mile"]
    )

    transmission["Line_Reinforcement_Cost_per_MWyr"] = line_inv_cost.round(0)

    return transmission


def network_max_reinforcement(
    transmission: pd.DataFrame, settings: dict
) -> pd.DataFrame:
    """Add the maximum amount that transmission lines between regions can be reinforced
    in a planning period.

    Parameters
    ----------
    transmission : pd.DataFrame
        One network line per row with the column "Line_Max_Flow_MW"
    settings : dict
        User-defined settings with the parameter "Line_Max_Reinforcement_MW"

    Returns
    -------
    pd.DataFrame
        A copy of the input transmission constraint dataframe with a new column
        `Line_Max_Reinforcement_MW`.
    """

    max_expansion = settings.get("tx_expansion_per_period")

    if not max_expansion and max_expansion != 0:
        raise KeyError(
            "No value for the transmission expansion allowed in this model period is "
            "included in the settings."
            "This numeric value is included under tx_expansion_per_period. See the "
            "`test_settings.yml` file for an example."
        )
    # if isinstance(max_expansion, dict):
    #     expansion_method = settings.get("tx_expansion_method")
    #     if not expansion_method:
    #         raise KeyError(
    #         "The transmission expansion parameter 'tx_expansion_per_period' is a "
    #         "dictionary. There should also be a settings parameter 'tx_expansion_method' "
    #         "with a dictionary of model region: <type> (either 'capacity' or 'fraction' "
    #         "but it isn't in your settings file."
    #     )
    #     for region, value in max_expansion.items():
    #         existing_tx = transmission.query("Region description == @region")["Line_Max_Flow_MW"]
    #         if expansion_method[region].lower() == "capacity":
    #             transmission.loc[transmission["Region description"] == region,
    #             "Line_Max_Reinforcement_MW"] = value
    #         elif expansion_method[region].lower() == "fraction":
    #             transmission.loc[transmission["Region description"] == region,
    #             "Line_Max_Reinforcement_MW"] = value * existing_tx
    #         else:
    #             raise KeyError(
    #             "The transmission expansion method parameter (tx_expansion_method) "
    #             "should have values of 'capacity' or 'fraction' for each model region. "
    #             f"The value provided was '{expansion_method[region]}'."
    #         )

    # else:
    transmission.loc[:, "Line_Max_Reinforcement_MW"] = (
        transmission.loc[:, "Line_Max_Flow_MW"] * max_expansion
    )
    transmission["Line_Max_Reinforcement_MW"] = transmission[
        "Line_Max_Reinforcement_MW"
    ].round(0)

    return transmission


def set_int_cols(df: pd.DataFrame, cols: list = None) -> pd.DataFrame:
    """Set values of some dataframe columns to integers.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    cols : list, optional
        Columns to set as integer, by default None. If none, will use
        `powergenome.GenX.INT_COLS`.

    Returns
    -------
    pd.DataFrame
        Input dataframe with some columns set as integer.
    """
    if not cols:
        cols = INT_COLS

    cols = [c for c in cols if c in df.columns]

    for col in cols:
        df[col] = df[col].fillna(0).astype(int)
    return df


def round_col_values(
    df: pd.DataFrame, col_round_val: Dict[str, int] = None
) -> pd.DataFrame:
    """Round values in columns to a specific sigfig.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    col_round_val : Dict, optional
        Dictionary with key values of column labels and integer values of the number of
        sigfigs, by default None.

    Returns
    -------
    pd.DataFrame
        Same dataframe as input but with rounded values in specified columns.
    """
    if not col_round_val:
        col_round_val = COL_ROUND_VALUES

    col_round_val = {k: v for k, v in col_round_val.items() if k in df.columns}

    for col, value in col_round_val.items():
        df[col] = df[col].fillna(0).round(value)
    return df


def calculate_partial_CES_values(gen_clusters, fuels, settings):
    gens = gen_clusters.copy()
    if settings.get("partial_ces"):
        assert set(fuels["Fuel"]) == set(gens["Fuel"])
        fuel_emission_map = fuels.copy()
        fuel_emission_map = fuel_emission_map.set_index("Fuel")

        gens["co2_emission_rate"] = gens["Heat_Rate_MMBTU_per_MWh"] * gens["Fuel"].map(
            fuel_emission_map["CO2_content_tons_per_MMBtu"]
        )

        # Make the partial CES credit equal to 1 ton minus the emissions rate, but
        # don't include coal plants

        partial_ces = 1 - gens["co2_emission_rate"]

        gens.loc[
            ~(gens["Resource"].str.contains("coal"))
            & (gens["STOR"] == 0)
            & (gens["FLEX"] == 0),
            # & ~(gens["Resource"].str.contains("battery"))
            # & ~(gens["Resource"].str.contains("load_shifting")),
            "CES",
        ] = partial_ces.round(3)
    # else:
    #     gen_clusters = add_genx_model_tags(gen_clusters, settings)

    gens["Zone"] = gens["Zone"]
    gens["Cap_Size"] = gens["Cap_size"]
    gens["Fixed_OM_Cost_per_MWyr"] = gens["Fixed_OM_Cost_per_MWyr"]
    gens["Fixed_OM_Cost_per_MWhyr"] = gens["Fixed_OM_Cost_per_MWhyr"]
    gens["Inv_Cost_per_MWyr"] = gens["Inv_Cost_per_MWyr"]
    gens["Inv_Cost_per_MWhyr"] = gens["Inv_Cost_per_MWhyr"]
    gens["Var_OM_Cost_per_MWh"] = gens["Var_OM_Cost_per_MWh"]
    # gens["Var_OM_Cost_per_MWh_In"] = gens["Var_OM_Cost_per_MWh_in"]
    gens["Start_Cost_per_MW"] = gens["Start_Cost_per_MW"]
    gens["Start_Fuel_MMBTU_per_MW"] = gens["Start_fuel_MMBTU_per_MW"]
    gens["Heat_Rate_MMBTU_per_MWh"] = gens["Heat_Rate_MMBTU_per_MWh"]
    gens["Min_Power"] = gens["Min_Power"]
    # gens["Self_Disch"] = gens["Self_disch"]
    # gens["Eff_Up"] = gens["Eff_up"]
    # gens["Eff_Down"] = gens["Eff_down"]
    # gens["Ramp_Up_Percentage"] = gens["Ramp_Up_percentage"]
    # gens["Ramp_Dn_Percentage"] = gens["Ramp_Dn_percentage"]
    # gens["Up_Time"] = gens["Up_time"]
    # gens["Down_Time"] = gens["Down_time"]
    # gens["Max_Flexible_Demand_Delay"] = gens["Max_DSM_delay"]
    gens["technology"] = gens["Resource"]
    gens["Resource"] = (
        gens["region"] + "_" + gens["technology"] + "_" + gens["cluster"].astype(str)
    )

    return gens


def check_min_power_against_variability(gen_clusters, resource_profile):

    min_gen_levels = resource_profile.min()

    assert len(min_gen_levels) == len(
        gen_clusters
    ), "The number of hourly resource profiles does not match the number of resources"
    # assert (
    #     gen_clusters["Min_Power"].isna().any() is False
    # ), (
    #     "At least one Min_Power value in 'gen_clusters' is null before checking against"
    #     " resource variability"
    # )

    min_gen_levels.index = gen_clusters.index

    gen_clusters["Min_Power"] = gen_clusters["Min_Power"].combine(min_gen_levels, min)

    # assert (
    #     gen_clusters["Min_Power"].isna().any() is False
    # ), (
    #     "At least one Min_Power value in 'gen_clusters' is null. Values were fine "
    #     "before checking against resoruce variability"
    # )

    return gen_clusters


def calc_emissions_ces_level(network_df, load_df, settings):

    # load_cols = [col for col in load_df.columns if "Load" in col]
    total_load = load_df.sum().sum()

    emissions_limit = settings["emissions_ces_limit"]

    if emissions_limit is not None:
        try:
            ces_value = 1 - (emissions_limit / total_load)
        except TypeError:
            print(emissions_limit, total_load)
            raise TypeError
        network_df["CES"] = ces_value
        network_df["CES"] = network_df["CES"].round(3)

        return network_df
    else:
        return network_df


def fix_min_power_values(
    resource_df: pd.DataFrame,
    gen_profile_df: pd.DataFrame,
    min_power_col: str = "Min_Power",
) -> pd.DataFrame:
    """Fix potentially erroneous min power values for resources with variable generation
    profiles. Any min power values that are higher than the lowest hourly generation
    will be adjusted down to match the lowest hourly generation.


    Parameters
    ----------
    resource_df : pd.DataFrame
        Records of generators/resources. Row order should match column order of
        `gen_profile_df`.
    gen_profile_df : pd.DataFrame
        Hourly generation values for all generators/resources. Column order should match
        row order in `resource_df`.
    min_power_col : str
        Column in `resource_df` that stores the minimum generation power of each
        resource. Default value is "Min_Power".

    Returns
    -------
    pd.DataFrame
        A modified version of `resource_df`. Any rows with minimum power larger than
        hourly generation are adjusted down to match the smallest hourly generation.
    """
    if min_power_col not in resource_df.columns:
        raise ValueError(
            f"When variable generation values against resource min power, the column "
            f"{min_power_col} was not found in the resource dataframe."
        )

    if resource_df.shape[0] != gen_profile_df.shape[1]:
        raise ValueError(
            "When trying to fix min power values, the number of resource dataframe rows"
            f" ({resource_df.shape[0]} rows) does not match the number of variable "
            f"profiles columns ({gen_profile_df.shape[1]} columns)."
        )

    resource_df = resource_df.reset_index(drop=True)
    resource_df.loc[:, "unadjusted_min_power"] = resource_df.loc[:, min_power_col]
    _gen_profile = gen_profile_df.copy(deep=True)
    _gen_profile
    gen_profile_min = _gen_profile.min().reset_index(drop=True)
    mask = (resource_df[min_power_col].fillna(0) > gen_profile_min).values

    logger.info(
        f"{sum(mask)} resources have {min_power_col} larger than hourly generation."
    )

    resource_df.loc[mask, min_power_col] = gen_profile_min[mask].round(3)

    return resource_df


def min_cap_req(settings: dict) -> pd.DataFrame:
    """Create a dataframe of minimum capacity requirements for GenX

    Parameters
    ----------
    settings : dict
        Dictionary with user settings. Should include the key `MinCapReg` with nested
        keys of `MinCapTag_*`, then further nested keys `description` and `min_mw`. The
        `MinCapTag_*` should also be listed as values under `model_tag_names`. Any
        technologies eligible for each of the `MinCapTag_*` should have `model_tag_values`
        of 1.

    Returns
    -------
    pd.DataFrame
        A dataframe with minimum capacity constraints formatted for GenX. If `MinCapReq`
        is not included in the settings dictionary it will return None.

    Raises
    ------
    KeyError
        If a `MinCapTag_*` is included under `MinCapReq` but not included in `model_tag_names`
        the function will raise an error.
    """

    c_num = []
    description = []
    min_mw = []

    # if settings.get("MinCapReq"):
    for cap_tag, values in settings.get("MinCapReq", {}).items():
        if cap_tag not in settings.get("model_tag_names", []):
            raise KeyError(
                f"The minimum capacity tag {cap_tag} is listed in the settings "
                "'MinCapReq' but not under 'model_tag_names'. You must add it to "
                "'model_tag_names' for the column to appear in Generators_data.csv."
            )

        # It's easy to forget to add all the necessary column names to the
        # generators_columns list in settings.
        if cap_tag not in settings.get("generator_columns", []) and isinstance(
            settings.get("generator_columns"), list
        ):
            settings["generator_columns"].append(cap_tag)

        c_num.append(cap_tag.split("_")[1])
        description.append(values.get("description"))
        min_mw.append(values.get("min_mw"))

    min_cap_df = pd.DataFrame()
    min_cap_df["MinCapReqConstraint"] = c_num
    min_cap_df["Constraint_Description"] = description
    min_cap_df["Min_MW"] = min_mw

    if not min_cap_df.empty:
        return min_cap_df
    else:
        return None


def max_cap_req(settings: dict) -> pd.DataFrame:
    """Create a dataframe of maximum capacity requirements for GenX

    Parameters
    ----------
    settings : dict
        Dictionary with user settings. Should include the key `MinCapReg` with nested
        keys of `MaxCapTag_*`, then further nested keys `description` and `min_mw`. The
        `MaxCapTag_*` should also be listed as values under `model_tag_names`. Any
        technologies eligible for each of the `MaxCapTag_*` should have `model_tag_values`
        of 1.

    Returns
    -------
    pd.DataFrame
        A dataframe with maximum capacity constraints formatted for GenX. If `MaxCapReq`
        is not included in the settings dictionary it will return None.

    Raises
    ------
    KeyError
        If a `MaxCapTag_*` is included under `MaxCapReq` but not included in `model_tag_names`
        the function will raise an error.
    """

    c_num = []
    description = []
    max_mw = []

    # if settings.get("MaxCapReq"):
    for cap_tag, values in settings.get("MaxCapReq", {}).items():
        if cap_tag not in settings.get("model_tag_names", []):
            raise KeyError(
                f"The maximum capacity tag {cap_tag} is listed in the settings "
                "'MaxCapReq' but not under 'model_tag_names'. You must add it to "
                "'model_tag_names' for the column to appear in Generators_data.csv."
            )

        # It's easy to forget to add all the necessary column names to the
        # generators_columns list in settings.
        if cap_tag not in settings.get("generator_columns", []) and isinstance(
            settings.get("generator_columns"), list
        ):
            settings["generator_columns"].append(cap_tag)

        c_num.append(cap_tag.split("_")[1])
        description.append(values.get("description"))
        max_mw.append(values.get("max_mw"))

    max_cap_df = pd.DataFrame()
    max_cap_df["MaxCapReqConstraint"] = c_num
    max_cap_df["Constraint_Description"] = description
    max_cap_df["Max_MW"] = max_mw

    if not max_cap_df.empty:
        return max_cap_df
    else:
        return None


def check_resource_tags(df: pd.DataFrame) -> pd.DataFrame:
    """Check complete generators dataframe to make sure each resource is assigned one,
    and only one, resource tag.

    Parameters
    ----------
    df : pd.DataFrame
        Resource clusters. Should have columns "technology" and "region" in addition
        to the resource tag columns expected by GenX.

    Returns
    -------
    pd.DataFrame
        An unaltered version of the input dataframe.
    """
    tags = [t for t in RESOURCE_TAGS if t in df.columns]
    if not (df[tags].sum(axis=1) == 1).all():
        for idx, row in df.iterrows():
            num_tags = row[tags].sum()
            if num_tags == 0:
                logger.warning(
                    "\n*************************\n"
                    f"The resource {row['technology']} in region {row['region']} does "
                    "not have any assigned resource tags. Check the 'model_tag_values' and "
                    "'regional_tag_values' parameters in your settings file to make sure"
                    "it is assigned one resource tag type from this list:\n\n"
                    f"{RESOURCE_TAGS}\n"
                )
            if num_tags > 1:
                s = row[RESOURCE_TAGS]
                tags = list(s[s == 1].index)
                logger.warning(
                    "\n*************************\n"
                    f"The resource {row['technology']} in region {row['region']} is "
                    f"assigned {num_tags} resource tags ({tags}). Check the 'model_tag_values'"
                    " and 'regional_tag_values' parameters in your settings file to make"
                    " sure it is assigned only one resource tag type from this list:\n\n"
                    f"{RESOURCE_TAGS}\n"
                )

        raise ValueError(
            "Use the warnings above to fix the resource tags in your settings file."
        )
    return df
