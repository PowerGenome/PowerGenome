"""
Hourly demand profiles
"""

import logging
from pathlib import Path
from typing import Dict, Union
import pandas as pd
import numpy as np
from joblib import Memory
import sqlalchemy
from powergenome.load_construction import build_total_load

from powergenome.util import regions_to_keep, reverse_dict_of_lists, remove_feb_29
from powergenome.external_data import make_demand_response_profiles
from powergenome.eia_opendata import get_aeo_load
from powergenome.params import DATA_PATHS

logger = logging.getLogger(__name__)


memory = Memory(location=DATA_PATHS["cache"], verbose=0)


def make_load_curves(
    pg_engine: sqlalchemy.engine.Engine,
    settings: dict,
    pg_table: str = "load_curves_ferc",
) -> pd.DataFrame:
    """Make wide-form load (demand) curves for model regionsl in a future year using
    historical IPM region data from a sql table and load growth parameters in the
    settings dictionary.

    Parameters
    ----------
    pg_engine : sqlalchemy.engine.Engine
        Connection to a sql database
    settings : dict
        Dictionary with settings parameters. Should have "model_regions", and "region_aggregations"
        (optional). To add load growth, the additional parameters "future_load_region_map",
        "historical_load_region_map", "growth_scenario", "eia_aeo_year", "model_year",
        "regular_load_growth_start_year" (optional), and "alt_growth_rate" (optional)
        are needed.
    pudl_table : str, optional
        Name of the sql table with historical demand data, by default "load_curves_ferc"

    Returns
    -------
    pd.DataFrame
        [description]
    """
    # IPM regions to keep. Regions not in this list will be dropped from the
    # dataframe
    keep_regions, region_agg_map = regions_to_keep(
        settings["model_regions"], settings.get("region_aggregations")
    )

    # I'd rather use a sql query and only pull the regions of interest but
    # sqlalchemy doesn't allow table names to be parameterized.
    logger.info("Loading load curves from PUDL")
    load_curves = pd.read_sql_table(
        pg_table, pg_engine, columns=["region_id_epaipm", "time_index", "load_mw"]
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

    if len(lc_wide) == 8784:
        lc_wide = remove_feb_29(lc_wide)

    # Shift load from UTC
    for col in lc_wide:
        lc_wide[col] = np.roll(lc_wide[col].values, settings.get("utc_offset", 0))

    lc_wide.index.name = "time_index"
    if lc_wide.index.min() == 0:
        lc_wide.index = lc_wide.index + 1

    return lc_wide


def add_load_growth(load_curves: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """Add load (demand) growth to regions using EIA AEO data for historical and future
    years.

    Parameters
    ----------
    load_curves : pd.DataFrame
        Long-form dataframe of demand for model regions, with columns "region_id_epaipm"
        and "load_mw"
    settings : dict
        Settings dictionary with parameters "future_load_region_map",
        "historical_load_region_map", "growth_scenario", "eia_aeo_year", "model_year",
        "regular_load_growth_start_year" (optional), and "alt_growth_rate" (optional).

    Returns
    -------
    pd.DataFrame
        [description]
    """
    keep_regions, region_agg_map = regions_to_keep(
        settings["model_regions"], settings.get("region_aggregations", {}) or {}
    )

    future_growth_factor = calc_growth_factors(
        keep_regions,
        load_region_map=settings["future_load_region_map"],
        growth_scenario=settings.get(
            "growth_scenario", f"REF{settings.get('eia_aeo_year', 2020)}"
        ),
        eia_aeo_year=settings.get("eia_aeo_year", 2020),
        start_year=settings.get("regular_load_growth_start_year", 2019),
        end_year=settings["model_year"],
        alt_growth_rate=settings.get("alt_growth_rate", {}) or {},
    )

    hist_region_map = reverse_dict_of_lists(settings["historical_load_region_map"])
    hist_demand_start = {
        ipm_region: get_aeo_load(
            region=hist_region_map[ipm_region], aeo_year=2014, scenario_series="REF2014"
        )
        .set_index("year")
        .loc[2012, "demand"]
        for ipm_region in keep_regions
    }
    hist_demand_end = {
        ipm_region: get_aeo_load(
            region=hist_region_map[ipm_region], aeo_year=2019, scenario_series="REF2019"
        )
        .set_index("year")
        .loc[2018, "demand"]
        for ipm_region in keep_regions
    }
    hist_growth_factor = {
        ipm_region: hist_demand_end[ipm_region] / hist_demand_start[ipm_region]
        for ipm_region in keep_regions
    }

    for region in keep_regions:
        load_curves.loc[load_curves["region_id_epaipm"] == region, "load_mw"] *= (
            hist_growth_factor[region] * future_growth_factor[region]
        )

    return load_curves


@memory.cache
def calc_growth_factors(
    keep_regions: list,
    load_region_map: dict,
    growth_scenario: str,
    eia_aeo_year: int,
    start_year: int,
    end_year: int,
    alt_growth_rate: dict = {},
) -> Dict[str, float]:
    """Calculate future demand growth factors for each region using EIA AEO data for
    electricity market regions. Growth factors for each region are determined based on a
    mapping of IPM regions to EMM regions.

    Growth factors are calculated as a ratio of end year demand over start year demand
    rather than as an annualized growth rate.


    Parameters
    ----------
    keep_regions : list
        List of IPM regions
    load_region_map : dict
        Mapping of IPM regions to AEO EMM regions
    growth_scenario : str
        Name of the AEO growth scenario from EIA's open data portal that should be used
        in the API call. Examples from AEO2020 include "REF2020", "HIGHMACRO", and
        "LOWMACRO" for Reference, High growth, and Low growth scenarios.
    eia_aeo_year : int
        Year of AEO data to use
    start_year : int
        First year of demand to use when calculating growth.
    end_year : int
        Final year of demand to use when calculating growth.
    alt_growth_rate : dict, optional
        Alternate growth rates (percentages) for regions, by default {}

    Returns
    -------
    Dict[str: float]
        Key, value pairs of a growth factor for each IPM region.
    """
    region_map = reverse_dict_of_lists(load_region_map)
    # growth_scenario = settings.get(
    #     "growth_scenario", f"REF{settings.get('eia_aeo_year', 2020)}"
    # )
    load_growth_dict = {
        ipm_region: get_aeo_load(
            region=region_map[ipm_region],
            aeo_year=eia_aeo_year,
            scenario_series=growth_scenario,
        ).set_index("year")
        for ipm_region in keep_regions
    }

    # Check to make sure start_year is in the list of data years. If not, use data from
    # the year after start_year
    if start_year not in list(load_growth_dict.values())[0].index:
        data_year = start_year + 1
        for r in load_growth_dict.keys():
            start_scenario = growth_scenario
            if "REF" in start_scenario:
                start_scenario = f"REF{data_year}"

            start_load = (
                get_aeo_load(
                    region=region_map[r],
                    aeo_year=data_year,
                    scenario_series=start_scenario,
                )
                .set_index("year")
                .loc[start_year, "demand"]
            )
            load_growth_dict[r].loc[start_year, "demand"] = start_load

    load_growth_start_map = {
        ipm_region: _df.loc[start_year, "demand"]
        for ipm_region, _df in load_growth_dict.items()
    }

    load_growth_end_map = {
        ipm_region: _df.loc[end_year, "demand"]
        for ipm_region, _df in load_growth_dict.items()
    }
    growth_factor = {
        ipm_region: load_growth_end_map[ipm_region] / load_growth_start_map[ipm_region]
        for ipm_region in keep_regions
    }
    years_growth = end_year - start_year

    for region, rate in alt_growth_rate.items():
        growth_factor[region] = (1 + rate) ** years_growth

    return growth_factor


def add_demand_response_resource_load(load_curves, settings):

    dr_path = Path(settings["input_folder"]) / settings["demand_response_fn"]
    dr_types = settings["flexible_demand_resources"][settings["model_year"]].keys()

    dr_curves = make_demand_response_profiles(
        dr_path, list(dr_types)[0], settings["model_year"], settings["demand_response"]
    )

    if len(dr_types) > 1:
        for dr in dr_types[1:]:
            _dr_curves = make_demand_response_profiles(
                dr_path, dr, settings["model_year"], settings["demand_response"]
            )
            dr_curves = dr_curves + _dr_curves

    for col in dr_curves.columns:
        try:
            load_curves.loc[:, col] += dr_curves[col].values
        except KeyError:
            pass

    return load_curves


def subtract_distributed_generation(load_curves, pg_engine, settings):

    dg_profiles = make_distributed_gen_profiles(pg_engine, settings)
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

    if len(hourly_load_profiles) == 8784:
        remove_feb_29(hourly_load_profiles)

    return hourly_load_profiles


def make_final_load_curves(
    pg_engine, settings, pudl_table="load_curves_ferc"
) -> pd.DataFrame:
    """Make wide-form dataframe of load (demand) curves for each model region. Includes
    load growth, demand response (flexible loads), and distributed generation.

    Regional demand profiles can be provided by the user with the setting parameter
    "regional_load_fn". Flexible loads can be provided with the parameter "demand_response_fn".
    Distributed generation profiles can be provided with the parameter "distributed_gen_profiles_fn".

    Parameters
    ----------
    pudl_engine : sqlalchemy.engine.Engine
        Connection to sql database
    settings : dict
        Dictionary of settings parameters needed to calculate demand, demand growth, etc.
    pudl_table : str, optional
        SQL table name, by default "load_curves_ferc"

    Returns
    -------
    pd.DataFrame
        [description]
    """
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
        load_curves_before_dg = make_load_curves(pg_engine, settings, pudl_table)

        if settings.get("demand_response_fn"):
            load_curves_dr = add_demand_response_resource_load(
                load_curves_before_dg, settings
            )
        elif settings.get("electrification_stock_fn") and settings.get(
            "electrification_scenario"
        ):
            from powergenome.load_construction import (
                AddElectrification,
                FilterTotalProfile,
            )

            keep_regions, region_agg_map = regions_to_keep(
                settings["model_regions"], settings.get("region_aggregations", {}) or {}
            )
            elec_kwargs = {
                "future_load_region_map": settings["future_load_region_map"],
                "eia_aeo_year": settings["eia_aeo_year"],
                "growth_scenario": settings["growth_scenario"],
                "path_in": settings.get("efs_path"),
            }

            total_load = build_total_load(
                settings.get("electrification_stock_fn"),
                settings["model_year"],
                settings.get("electrification_scenario"),
                keep_regions,
                settings.get("extra_outputs"),
                **elec_kwargs,
            )
            load_curves_dr = FilterTotalProfile(settings, total_load)
        else:
            load_curves_dr = load_curves_before_dg

    if settings.get("distributed_gen_profiles_fn") and not settings.get(
        "dg_as_resource"
    ):
        final_load_curves = subtract_distributed_generation(
            load_curves_dr, pg_engine, settings
        )
    else:
        final_load_curves = load_curves_dr

    final_load_curves = final_load_curves.astype(int)

    return final_load_curves


def make_distributed_gen_profiles(pg_engine, settings):
    """Create 8760 annual generation profiles for distributed generation in regions.
    Uses a distribution loss parameter in the settings file when DG generation is
    defined a fraction of delivered load.

    Parameters
    ----------
    dg_profiles_path : path-like
        Where to load the file from
    pg_engine : sqlalchemy.Engine
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
        Path(settings["input_folder"]) / settings["distributed_gen_profiles_fn"]
    )

    hourly_norm_profiles = pd.read_csv(dg_profiles_path)
    profile_regions = hourly_norm_profiles.columns

    dg_calc_methods = settings["distributed_gen_method"]
    dg_calc_values = settings["distributed_gen_values"]

    assert (
        year in dg_calc_values
    ), f"The years in settings parameter 'distributed_gen_values' do not match the model years."

    for region in dg_calc_values[year]:
        assert region in set(profile_regions), (
            "The profile regions in settings parameter 'distributed_gen_values' do not\n"
            f"match the regions in {settings['distributed_gen_profiles_fn']} for year {year}"
        )

    if "fraction_load" in dg_calc_methods.values():
        regional_load = make_load_curves(pg_engine, settings)

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

    if len(dg_hourly_gen) == 8784:
        remove_feb_29(dg_hourly_gen)

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
