"Functions specific to GenX outputs"

from itertools import product
import logging
import pandas as pd

from powergenome.external_data import load_policy_scenarios, load_demand_segments
from powergenome.load_profiles import make_distributed_gen_profiles
from powergenome.time_reduction import kmeans_time_clustering
from powergenome.util import load_settings
from powergenome.nrelatb import investment_cost_calculator

logger = logging.getLogger(__name__)


def add_emission_policies(transmission_df, settings, DistrZones=None):
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

    zones = settings["model_regions"]
    zone_num_map = {
        zone: f"z{number + 1}" for zone, number in zip(zones, range(len(zones)))
    }

    zone_cols = ["Region description", "Network_zones", "DistrZones"] + list(
        policies.columns
    )
    zone_df = pd.DataFrame(columns=zone_cols)
    zone_df["Region description"] = zones
    zone_df["Network_zones"] = zone_df["Region description"].map(zone_num_map)

    if DistrZones is None:
        zone_df["DistrZones"] = 0

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
    path = settings["input_folder"] / settings["misc_gen_inputs_fn"]
    misc_values = pd.read_csv(path)
    misc_values = misc_values.fillna("skip")

    for resource in misc_values["Resource"].unique():
        # resource_misc_values = misc_values.loc[misc_values["Resource"] == resource, :].dropna()

        for col in misc_values.columns:
            value = misc_values.loc[misc_values["Resource"] == resource, col].values[0]
            if value != "skip":
                gen_clusters.loc[
                    gen_clusters["Resource"].str.contains(resource), col
                ] = value

    return gen_clusters


def make_genx_settings_file(pudl_engine, settings):
    """Make a copy of the GenX settings file for a specific case.

    This assumes that there is a base-level GenX settings file with parameters that
    stay constant across all cases.

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
    dg_generation = make_distributed_gen_profiles(pudl_engine, settings)
    total_dg_gen = dg_generation.sum().sum()

    if isinstance(year_case_policy, pd.DataFrame):
        year_case_policy = year_case_policy.sum()

    if not float(year_case_policy["CO_2_Max_Mtons"]) > 0:
        genx_settings["CO2Cap"] = 0
    else:
        genx_settings["CO2Cap"] = 2

    if float(year_case_policy["RPS"]) > 0:
        # print(total_dg_gen)
        # print(year_case_policy["RPS"])
        if policies.loc[(case_id, model_year), "region"] == "all":
            genx_settings["RPS"] = 3
            genx_settings["RPS_Adjustment"] = float(
                (1 - year_case_policy["RPS"]) * total_dg_gen
            )
        else:
            genx_settings["RPS"] = 2
            genx_settings["RPS_Adjustment"] = 0
    else:
        genx_settings["RPS"] = 0
        genx_settings["RPS_Adjustment"] = 0

    if float(year_case_policy["CES"]) > 0:
        if policies.loc[(case_id, model_year), "region"] == "all":
            genx_settings["CES"] = 3
            genx_settings["CES_Adjustment"] = float(
                (1 - year_case_policy["CES"]) * total_dg_gen
            )
        else:
            genx_settings["CES"] = 2
            genx_settings["CES_Adjustment"] = 0
    else:
        genx_settings["CES"] = 0
        genx_settings["CES_Adjustment"] = 0

    genx_settings["case_id"] = case_id
    genx_settings["case_name"] = case_name
    genx_settings["year"] = str(model_year)

    return genx_settings


def reduce_time_domain(
    resource_profiles, load_profiles, settings, variable_resources_only=True
):

    demand_segments = load_demand_segments(settings)

    if settings["reduce_time_domain"]:
        days = settings["time_domain_days_per_period"]
        time_periods = settings["time_domain_periods"]
        include_peak_day = settings["include_peak_day"]
        load_weight = settings["demand_weight_factor"]

        results, rep_point, cluster_weight = kmeans_time_clustering(
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

        time_index = pd.Series(data=reduced_load_profile.index + 1, name="Time_index")
        sub_weights = pd.Series(
            data=[x * (days * 24) for x in results["ClusterWeights"]],
            name="Sub_Weights",
        )
        hours_per_period = pd.Series(data=[days * 24], name="Hours_per_period")
        subperiods = pd.Series(data=[time_periods], name="Subperiods")
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

        return reduced_resource_profile, reduced_load_output

    else:
        time_index = pd.Series(data=range(1, 8761), name="Time_index")
        sub_weights = pd.Series(data=[1], name="Sub_Weights")
        hours_per_period = pd.Series(data=[168], name="Hours_per_period")
        subperiods = pd.Series(data=[1], name="Subperiods")

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

        return resource_profiles, load_output


def network_line_loss(transmission, settings):
    """Add line loss percentage for each network line between regions.

    Parameters
    ----------
    transmission : DataFrame
        One network line per row with a column "distance_mile"
    settings : dict
        User-defined settings with a parameter "tx_line_loss_100_miles"

    Returns
    -------
    DataFrame
        Same as input but with the new column 'Line_Loss_Percentage'
    """
    if "tx_line_loss_100_miles" not in settings:
        raise KeyError(
            "The parameter 'tx_line_loss_100_miles' is required in your settings file."
        )
    loss_per_100_miles = settings["tx_line_loss_100_miles"]
    transmission["Line_Loss_Percentage"] = (
        transmission["distance_mile"] / 100 * loss_per_100_miles
    ).round(4)

    return transmission


def network_reinforcement_cost(transmission, settings):
    """Add transmission line reinforcement investment costs (per MW-mile-year)

    Parameters
    ----------
    transmission : DataFrame
        One network line per row with columns "Transmission Path Name" and
        "distance_mile"
    settings : dict
        User-defined settings with parameters "tx_reinforcement_cost_mw_mile",
        "tx_reinforcement_wacc", and "tx_reinforcement_investment_years"

    Returns
    -------
    DataFrame
        Same as input but with the new column 'Line_Reinforcement_Cost_per_MW_yr'
    """

    cost_dict = settings["tx_reinforcement_cost_mw_mile"]
    origin_region_cost = (
        transmission["Transmission Path Name"].str.split("_to_").str[0].map(cost_dict)
    )
    dest_region_cost = (
        transmission["Transmission Path Name"].str.split("_to_").str[-1].map(cost_dict)
    )

    # Average the costs per mile between origin and destination regions
    line_capex = (origin_region_cost + dest_region_cost) / 2
    line_wacc = settings["tx_reinforcement_wacc"]
    line_inv_period = settings["tx_reinforcement_investment_years"]

    line_inv_cost = (
        investment_cost_calculator(line_capex, line_wacc, line_inv_period)
        * transmission["distance_mile"]
    )

    transmission["Line_Reinforcement_Cost_per_MW_yr"] = line_inv_cost.round(0)

    return transmission


def network_max_reinforcement(transmission, settings):
    """Add the maximum amount that transmission lines between regions can be reinforced
    in a planning period.

    Parameters
    ----------
    transmission : DataFrame
        One network line per row with the column "Line_Max_Flow_MW"
    settings : dict
        User-defined settings with the parameter "Line_Max_Reinforcement_MW"

    Returns
    -------
    [type]
        [description]
    """

    max_expansion = settings["tx_expansion_per_period"]

    transmission.loc[:, "Line_Max_Reinforcement_MW"] = (
        transmission.loc[:, "Line_Max_Flow_MW"] * max_expansion
    )
    transmission["Line_Max_Reinforcement_MW"] = transmission[
        "Line_Max_Reinforcement_MW"
    ].round(0)

    return transmission


def set_int_cols(df):

    df["Up_time"] = df["Up_time"].astype(int)
    df["Down_time"] = df["Down_time"].astype(int)
    df["Max_DSM_delay"] = df["Max_DSM_delay"].astype(int)

    return df


def calculate_partial_CES_values(gen_clusters, fuels, settings):
    gens = gen_clusters.copy()
    if "partial_ces" in settings:
        if settings["partial_ces"]:
            fuel_emission_map = fuels.copy()
            fuel_emission_map = fuel_emission_map.set_index("Fuel")

            gens["co2_emission_rate"] = gens["Heat_rate_MMBTU_per_MWh"] * gens[
                "Fuel"
            ].map(fuel_emission_map["CO2_content_tons_per_MMBtu"])

            # Make the partial CES credit equal to 1 ton minus the emissions rate, but
            # don't include coal plants

            partial_ces = 1 - gens["co2_emission_rate"]

            gens.loc[
                ~(gens["Resource"].str.contains("coal"))
                & ~(gens["Resource"].str.contains("battery"))
                & ~(gens["Resource"].str.contains("load_shifting")),
                "CES",
            ] = partial_ces.round(3)
    # else:
    #     gen_clusters = add_genx_model_tags(gen_clusters, settings)

    return gens


def check_min_power_against_variability(gen_clusters, resource_profile):

    min_gen_levels = resource_profile.min()
    original_min_power = gen_clusters["Min_power"].copy()

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
        ces_value = 1 - (emissions_limit / total_load)
        network_df["CES"] = ces_value
        network_df["CES"] = network_df["CES"].round(3)

        return network_df
    else:
        return network_df
