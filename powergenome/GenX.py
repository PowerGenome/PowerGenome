"Functions specific to GenX outputs"

from itertools import product
import logging
from pathlib import Path
from typing import Dict
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
    "Inv_cost_per_MWyr",
    "Fixed_OM_cost_per_MWyr",
    "Inv_cost_per_MWhyr",
    "Fixed_OM_cost_per_MWhyr",
    "Line_Reinforcement_Cost_per_MW_yr",
]

COL_ROUND_VALUES = {
    "Var_OM_cost_per_MWh": 2,
    "Var_OM_cost_per_MWh_in": 2,
    "Start_cost_per_MW": 0,
    "Cost_per_MMBtu": 2,
    "CO2_content_tons_per_MMBtu": 5,
    "Cap_size": 2,
    "Heat_rate_MMBTU_per_MWh": 2,
    "distance_mile": 4,
    "Line_Max_Reinforcement_MW": 0,
    "distance_miles": 1,
    "distance_km": 1,
}


def create_policy_req(settings: dict, col_str_match: str) -> pd.DataFrame:
    model_year = settings["model_year"]
    case_id = settings["case_id"]

    policies = load_policy_scenarios(settings)
    policy_cols = [c for c in policies.columns if col_str_match in c]
    if len(policy_cols) == 0:
        return None

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

    zone_cols = ["Region_description", "Network_zones"] + policy_cols
    zone_df = pd.DataFrame(columns=zone_cols)
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


def make_genx_settings_file(pudl_engine, settings):
    """Make a copy of the GenX settings file for a specific case.

    This function tries to make some intellegent choices about parameter values like
    the RPS/CES type and can also read values from a file.

    There should be a base-level GenX settings file with parameters like the solver and
    solver-specific settings that stay constant across all cases.

    Parameters
    ----------
    pudl_engine : sqlalchemy.Engine
        A sqlalchemy connection for use by pandas to access IPM load profiles. These
        load profiles are needed when DG is calculated as a fraction of load.
    settings : dict
        User-defined parameters from a settings file. Should have keys of `model_year`
        `case_id`, 'case_name', `input_folder` (a Path object of where to find
        user-supplied data), `emission_policies_fn`, 'distributed_gen_profiles_fn'
        (the files to load in other functions), and 'genx_settings_fn'.

    Returns
    -------
    dict
        Dictionary of settings for a GenX run
    """

    model_year = settings["model_year"]
    case_id = settings["case_id"]
    case_name = settings["case_name"]

    genx_settings = load_settings(settings["genx_settings_fn"])
    policies = load_policy_scenarios(settings)
    year_case_policy = policies.loc[(case_id, model_year), :]

    # Bug where multiple regions for a case will return this as a df, even if the policy
    # for this case applies to all regions (code below expects a Series)
    ycp_shape = year_case_policy.shape
    if ycp_shape[0] == 1 and len(ycp_shape) > 1:
        year_case_policy = year_case_policy.squeeze()  # convert to series

    if settings.get("distributed_gen_profiles_fn"):
        dg_generation = make_distributed_gen_profiles(pudl_engine, settings)
        total_dg_gen = dg_generation.sum().sum()
    else:
        total_dg_gen = 0

    if isinstance(year_case_policy, pd.DataFrame):
        year_case_policy = year_case_policy.sum()

    # Don't wrap when time domain isn't reduced
    if not settings.get("reduce_time_domain"):
        genx_settings["OperationWrapping"] = 0

    genx_settings["case_id"] = case_id
    genx_settings["case_name"] = case_name
    genx_settings["year"] = str(model_year)

    # This is a new setting, will need to have a way to change.
    genx_settings["CapacityReserveMargin"] = 0
    genx_settings["LDS"] = 0

    # Load user defined values for the genx settigns file. This overrides the
    # complicated logic above.
    if settings.get("case_genx_settings_fn"):
        user_genx_settings = load_user_genx_settings(settings)
        user_case_settings = user_genx_settings.loc[(case_id, model_year), :]
        for key, value in user_case_settings.items():
            if not pd.isna(value):
                genx_settings[key] = value

    return genx_settings


def reduce_time_domain(
    resource_profiles, load_profiles, settings, variable_resources_only=True
):

    demand_segments = load_demand_segments(settings)

    if settings.get("reduce_time_domain"):
        days = settings["time_domain_days_per_period"]
        time_periods = settings["time_domain_periods"]
        include_peak_day = settings["include_peak_day"]
        load_weight = settings["demand_weight_factor"]

        results, _, _ = kmeans_time_clustering(
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

        return reduced_resource_profile, reduced_load_output, time_series_mapping

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

        return resource_profiles, load_output, None


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
    if "distance_mile" in transmission.columns:
        distance_col = "distance_mile"
    elif "distance_km" in transmission.columns:
        distance_col = "distance_km"
        loss_per_100_miles *= 0.62137
        logger.info("Line loss per 100 miles was converted to km.")
    else:
        raise KeyError("No distance column is available in the transmission dataframe")
    loss_per_100_miles = settings["tx_line_loss_100_miles"]
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
        One network line per row with columns "Transmission Path Name" and
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
        transmission["Transmission Path Name"].str.split("_to_").str[0].map(cost_dict)
    )
    dest_region_cost = (
        transmission["Transmission Path Name"].str.split("_to_").str[-1].map(cost_dict)
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

    if not max_expansion:
        raise KeyError(
            "No value for the transmission expansion allowed in this model period is "
            "included in the settings."
            "This numeric value is included under tx_expansion_per_period. See the "
            "`test_settings.yml` file for an example."
        )

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

        gens["co2_emission_rate"] = gens["Heat_rate_MMBTU_per_MWh"] * gens["Fuel"].map(
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

    gens["Zone"] = gens["zone"]
    gens["Cap_Size"] = gens["Cap_size"]
    gens["Fixed_OM_Cost_per_MWyr"] = gens["Fixed_OM_cost_per_MWyr"]
    gens["Fixed_OM_Cost_per_MWhyr"] = gens["Fixed_OM_cost_per_MWhyr"]
    gens["Inv_Cost_per_MWyr"] = gens["Inv_cost_per_MWyr"]
    gens["Inv_Cost_per_MWhyr"] = gens["Inv_cost_per_MWhyr"]
    gens["Var_OM_Cost_per_MWh"] = gens["Var_OM_cost_per_MWh"]
    # gens["Var_OM_Cost_per_MWh_In"] = gens["Var_OM_cost_per_MWh_in"]
    gens["Start_Cost_per_MW"] = gens["Start_cost_per_MW"]
    gens["Start_Fuel_MMBTU_per_MW"] = gens["Start_fuel_MMBTU_per_MW"]
    gens["Heat_Rate_MMBTU_per_MWh"] = gens["Heat_rate_MMBTU_per_MWh"]
    gens["Min_Power"] = gens["Min_power"]
    gens["Self_Disch"] = gens["Self_disch"]
    gens["Eff_Up"] = gens["Eff_up"]
    gens["Eff_Down"] = gens["Eff_down"]
    gens["Ramp_Up_Percentage"] = gens["Ramp_Up_percentage"]
    gens["Ramp_Dn_Percentage"] = gens["Ramp_Dn_percentage"]
    gens["Up_Time"] = gens["Up_time"]
    gens["Down_Time"] = gens["Down_time"]
    gens["Max_Flexible_Demand_Delay"] = gens["Max_DSM_delay"]
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
    #     gen_clusters["Min_power"].isna().any() is False
    # ), (
    #     "At least one Min_power value in 'gen_clusters' is null before checking against"
    #     " resource variability"
    # )

    min_gen_levels.index = gen_clusters.index

    gen_clusters["Min_power"] = gen_clusters["Min_power"].combine(min_gen_levels, min)

    # assert (
    #     gen_clusters["Min_power"].isna().any() is False
    # ), (
    #     "At least one Min_power value in 'gen_clusters' is null. Values were fine "
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
    min_power_col: str = "Min_power",
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
        resource. Default value is "Min_power".

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
