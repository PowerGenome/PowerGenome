"""
Hourly demand profiles
"""

import logging

import pandas as pd

from powergenome.util import reverse_dict_of_lists, shift_wrap_profiles
from powergenome.external_data import make_demand_response_profiles

logger = logging.getLogger(__name__)


def make_load_curves(
    pudl_engine,
    settings,
    pudl_table="load_curves_epaipm",
    settings_agg_key="region_aggregations",
):

    # Settings has a dictionary of lists for regional aggregations. Need
    # to reverse this to use in a map method.
    region_agg_map = reverse_dict_of_lists(settings[settings_agg_key])

    # IPM regions to keep. Regions not in this list will be dropped from the
    # dataframe
    keep_regions = [
        x
        for x in settings["model_regions"] + list(region_agg_map)
        if x not in region_agg_map.values()
    ]

    # I'd rather use a sql query and only pull the regions of interest but
    # sqlalchemy doesn't allow table names to be parameterized.
    logger.info("Loading load curves from PUDL")
    load_curves = pd.read_sql_table(
        pudl_table, pudl_engine, columns=["region_id_epaipm", "time_index", "load_mw"]
    )

    load_curves = load_curves.loc[load_curves.region_id_epaipm.isin(keep_regions)]

    # Increase demand to account for load growth
    load_curves = add_load_growth(load_curves, settings)

    # Set a new column "region" to the old column values. Then replace values for any
    # regions that are being aggregated
    load_curves.loc[:, "region"] = load_curves.loc[:, "region_id_epaipm"]

    load_curves.loc[
        load_curves.region_id_epaipm.isin(region_agg_map.keys()), "region"
    ] = load_curves.loc[
        load_curves.region_id_epaipm.isin(region_agg_map.keys()), "region_id_epaipm"
    ].map(
        region_agg_map
    )

    logger.info("Aggregating load curves in grouped regions")
    load_curves_agg = load_curves.groupby(["region", "time_index"]).sum()

    lc_wide = load_curves_agg.unstack(level=0)
    lc_wide.columns = lc_wide.columns.droplevel()

    pst_offset = settings["target_region_pst_offset"]

    lc_wide = shift_wrap_profiles(lc_wide, pst_offset)

    lc_wide.index.name = "time_index"
    lc_wide.index = lc_wide.index + 1

    return lc_wide


def add_load_growth(load_curves, settings):

    load_map = reverse_dict_of_lists(settings["load_region_map"])

    load_growth_map = {
        ipm_region: settings["default_growth_rates"][load_region]
        for ipm_region, load_region in load_map.items()
    }

    if settings["alt_growth_rate"] is not None:
        for region, rate in settings["alt_growth_rate"].items():
            load_growth_map[region] = rate

    if "regular_load_growth_start_year" in settings.keys():
        # historical load growth

        demand_start = settings["aeo_hist_start_elec_demand"]
        demand_end = settings["aeo_hist_end_elec_demand"]

        if not all([key in demand_end.keys() for key in demand_start.keys()]):
            raise KeyError(
                "Error in keys for historical electricity demand. /n"
                "Not all keys in 'aeo_hist_start_elec_demand' are also in "
                "'aeo_hist_end_elec_demand'"
            )

        historic_growth_ratio = {
            region: demand_end[region] / demand_start[region] for region in demand_start
        }
        historic_growth_map = {
            ipm_region: historic_growth_ratio[load_region]
            for ipm_region, load_region in load_map.items()
        }

        for region in load_curves["region_id_epaipm"].unique():
            hist_growth_factor = historic_growth_map[region]
            load_curves.loc[
                load_curves["region_id_epaipm"] == region, "load_mw"
            ] *= hist_growth_factor

        # Don't grow load over years where we already have historical data
        years_growth = (
            settings["model_year"] - settings["regular_load_growth_start_year"]
        )

    else:
        years_growth = settings["model_year"] - settings["default_load_year"]

    load_growth_factor = {
        region: (1 + rate) ** years_growth for region, rate in load_growth_map.items()
    }

    for region in load_curves["region_id_epaipm"].unique():
        growth_factor = load_growth_factor[region]
        load_curves.loc[
            load_curves["region_id_epaipm"] == region, "load_mw"
        ] *= growth_factor

    return load_curves


def add_demand_response_resource_load(load_curves, settings):

    dr_path = settings["input_folder"] / settings["demand_response_fn"]
    dr_types = settings["demand_response_resources"][settings["model_year"]].keys()

    dr_curves = make_demand_response_profiles(dr_path, list(dr_types)[0], settings)

    if len(dr_types) > 1:
        for dr in dr_types[1:]:
            _dr_curves = make_demand_response_profiles(dr_path, dr, settings)
            dr_curves = dr_curves + _dr_curves

    for col in dr_curves.columns:
        try:
            load_curves.loc[:, col] += dr_curves[col].values
        except KeyError:
            pass

    return load_curves


def subtract_distributed_generation(load_curves, pudl_engine, settings):

    dg_profiles = make_distributed_gen_profiles(pudl_engine, settings)
    dg_profiles.index = dg_profiles.index + 1

    for col in dg_profiles.columns:
        load_curves.loc[:, col] = load_curves.loc[:, col] - (
            dg_profiles[col].values * 1 + settings["avg_distribution_loss"]
        )

    return load_curves


def load_usr_demand_profiles(settings):
    "Temp function to load user-generated demand profiles"
    from powergenome.external_data import make_usr_demand_profiles

    lp_path = settings["input_folder"] / settings["regional_load_fn"]
    hourly_load_profiles = make_usr_demand_profiles(lp_path, settings)

    return hourly_load_profiles


def make_final_load_curves(
    pudl_engine,
    settings,
    pudl_table="load_curves_epaipm",
    settings_agg_key="region_aggregations",
):
    # Check if regional loads are supplied by the user
    if settings.get("regional_load_fn"):
        logger.info("Loading regional demand profiles from user")
        load_curves_dr = load_usr_demand_profiles(settings)
        if not settings.get("regional_load_includes_demand_response"):
            if settings.get("demand_response_fn"):
                logger.info("Adding DR profiles to user regional demand")
                load_curves_dr = add_demand_response_resource_load(
                    load_curves_dr, settings
                )
            else:
                logger.warning(
                    "The settings parameter 'regional_load_includes_demand_response' "
                    f"is {settings.get('regional_load_includes_demand_response')}, so "
                    "a filename is expected for 'demand_response_fn' in the settings "
                    "file. No filename has been provided."
                )
    else:
        load_curves_before_dg = make_load_curves(
            pudl_engine, settings, pudl_table, settings_agg_key
        )

        if settings.get("demand_response_fn"):
            load_curves_dr = add_demand_response_resource_load(
                load_curves_before_dg, settings
            )
        else:
            load_curves_dr = load_curves_before_dg

    if settings.get("distributed_gen_profiles_fn"):
        final_load_curves = subtract_distributed_generation(
            load_curves_dr, pudl_engine, settings
        )
    else:
        final_load_curves = load_curves_dr

    final_load_curves = final_load_curves.astype(int)

    return final_load_curves


def make_distributed_gen_profiles(pudl_engine, settings):
    """Create 8760 annual generation profiles for distributed generation in regions.
    Uses a distribution loss parameter in the settings file when DG generation is
    defined a fraction of delivered load.

    Parameters
    ----------
    dg_profiles_path : path-like
        Where to load the file from
    pudl_engine : sqlalchemy.Engine
        A sqlalchemy connection for use by pandas. Needed to create base load profiles.
    settings : dict
        User-defined parameters from a settings file

    Returns
    -------
    DataFrame
        Hourly generation profiles for DG resources in each region. Not all regions
        need to be accounted for.

    Raises
    ------
    KeyError
        If the calculation method specified in settings is not 'capacity' or 'fraction_load'
    """

    year = settings["model_year"]
    dg_profiles_path = (
        settings["input_folder"] / settings["distributed_gen_profiles_fn"]
    )

    hourly_norm_profiles = pd.read_csv(dg_profiles_path)
    profile_regions = hourly_norm_profiles.columns

    dg_calc_methods = settings["distributed_gen_method"]
    dg_calc_values = settings["distributed_gen_values"]

    assert (
        year in dg_calc_values.keys()
    ), f"The years in settings parameter 'distributed_gen_values' do not match the model years."

    for region, values in dg_calc_values[year].items():
        assert region in set(profile_regions), (
            "The profile regions in settings parameter 'distributed_gen_values' do not\n"
            f"match the regions in {settings['distributed_gen_profiles_fn']} for year {year}"
        )

    if "fraction_load" in dg_calc_methods.values():
        regional_load = make_load_curves(pudl_engine, settings)

    dg_hourly_gen = pd.DataFrame(columns=dg_calc_methods.keys())

    for region, method in dg_calc_methods.items():
        region_norm_profile = hourly_norm_profiles[region]
        region_calc_value = dg_calc_values[year][region]

        if method == "capacity":
            dg_hourly_gen[region] = calc_dg_capacity_method(
                region_norm_profile, region_calc_value
            )
        elif method == "fraction_load":
            region_load = regional_load[region]
            dg_hourly_gen[region] = calc_dg_frac_load_method(
                region_norm_profile, region_calc_value, region_load, settings
            )
        else:
            raise KeyError(
                "The settings parameter 'distributed_gen_method' can only have key "
                "values of 'capapacity' or 'fraction_load' for each region.\n"
                f"The value in your settings file is {method}"
            )

    return dg_hourly_gen


def calc_dg_capacity_method(dg_profile, dg_capacity):
    """Calculate the hourly distributed generation in a single region when given
    installed capacity.

    Parameters
    ----------
    dg_profile : Series
        Hourly normalized generation profile
    dg_capacity : float
        Total installed DG capacity

    Returns
    -------
    Series
        8760 hourly generation
    """

    hourly_gen = dg_profile * dg_capacity

    return hourly_gen.values


def calc_dg_frac_load_method(dg_profile, dg_requirement, regional_load, settings):
    """Calculate the hourly distributed generation in a single region where generation
    required to be a fraction of total sales.

    Parameters
    ----------
    dg_profile : Series
        Hourly normalized generation profile
    dg_requirement : float
        The fraction of total sales that DG must constitute
    regional_load : Series
        Hourly load for a given region
    settings : dict
        User-defined parameters from a settings file

    Returns
    -------
    Series
        8760 hourly generation
    """

    annual_load = regional_load.sum()
    dg_capacity_factor = dg_profile.mean()
    distribution_loss = settings["avg_distribution_loss"]

    required_dg_gen = annual_load * dg_requirement * (1 - distribution_loss)
    dg_capacity = required_dg_gen / 8760 / dg_capacity_factor

    hourly_gen = dg_profile * dg_capacity

    return hourly_gen
