"""
Hourly demand profiles
"""

import logging
from inspect import signature
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import sqlalchemy as sa

from powergenome.distributed_gen import distributed_gen_profiles
from powergenome.eia_opendata import get_aeo_load
from powergenome.external_data import make_demand_response_profiles
from powergenome.load_construction import (
    electrification_profiles,
    load_region_pop_frac,
    state_demand_to_region,
)
from powergenome.params import SETTINGS
from powergenome.util import (
    find_region_col,
    map_agg_region_names,
    regions_to_keep,
    remove_feb_29,
    reverse_dict_of_lists,
)

logger = logging.getLogger(__name__)


def filter_load_by_region(load_source):  # "decorator factory"
    """If regional load options are given, return the columns listed in
    settings["regional_load_source"][load_source].

    If settings["regional_load_source"] exists and settings["regional_load_source"][load_source]
    is null, return None.

    If settings["regional_load_source"] DNE, return the load profile if the load_source is EFS,
    else return None. This makes EFS the default load type/source.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            ## retrieve settings:
            # if kwarg:
            settings = kwargs.get("settings", None)
            # if arg:
            if settings is None:
                settings_arg_position = list(signature(func).parameters).index(
                    "settings"
                )
                settings = args[settings_arg_position]

            regional_load_sources = settings.get("regional_load_source")

            if regional_load_sources is not None:
                regions = None
                if load_source == regional_load_sources:
                    # if only one load profile sources are specified, use for all regions
                    regions = settings.get("model_regions")
                elif (
                    isinstance(regional_load_sources, dict)
                    and load_source in regional_load_sources.keys()
                ):
                    # if multiple load profiles sources are specified, find the proper regions
                    regions = regional_load_sources.get(load_source)

                # pd.reindex will return the entire DataFrame if regions=None,
                # We want the opposite; return None if regions = None
                if regions is not None:
                    load_profile = func(*args, **kwargs)
                    load_profile = load_profile.reindex(columns=regions)
                else:
                    load_profile = None
            else:
                load_profile = None
                if load_source == "EFS":
                    load_profile = func(*args, **kwargs)

            return load_profile

        return wrapper

    return decorator


def make_load_curves(
    pg_engine: sa.engine.base.Engine,
    settings: dict,
    pg_table: str = "load_curves_nrel_efs",
) -> pd.DataFrame:
    """Read base load profiles from database and grow the load to a future year.

    Parameters
    ----------
    pg_engine : sa.engine.base.Engine
        Engine to connect to a PowerGenome database
    settings : dict
        User parameter settings. Required keys are "model_regions", "future_load_region_map",
        and "historical_load_region_maps" (if load data represent a year before
        2019). Optional keys include "region_aggregations", "electrification_stock_fn",
        "electrification_scenario", and "alt_growth_rate".
    pg_table : str, optional
        Name of the database table with load profiles, by default "load_curves_nrel_efs"

    Returns
    -------
    pd.DataFrame
        Wide dataframe of load profiles for each model region. The only change from
        base year data in the database is load growth.

    Raises
    ------
    KeyError
        No table in the database with the supplied name.
    """
    # IPM regions to keep. Regions not in this list will be dropped from the
    # dataframe
    keep_regions, region_agg_map = regions_to_keep(
        settings["model_regions"], settings.get("region_aggregations")
    )

    # I'd rather use a sql query and only pull the regions of interest but
    # sqlalchemy doesn't allow table names to be parameterized.
    logger.info("Loading load curves from PUDL")
    inst = sa.inspect(pg_engine)
    if not inst.has_table(pg_table):
        raise KeyError(
            f"There is no load curves table with the name {pg_table} in the 'PG_DB' "
            "database specified in your .env file."
        )
    table_cols = [c["name"] for c in inst.get_columns(pg_table)]
    context = f"Load curves table ({pg_table} in database {pg_engine}."
    try:
        region_col = find_region_col(table_cols, context)
    except ValueError as e:
        if "state" in table_cols:
            path_in = Path(SETTINGS["EFS_DATA"])
            pop_files = path_in.glob("*pop_weight*")
            newest_pop_file = max(pop_files, key=os.path.getmtime)
            pop = load_region_pop_frac(path_in=path_in, fn=newest_pop_file.name)
            pop = pop.rename(columns={"region": "demand_region"})
            pop = pop.loc[pop["demand_region"].isin(keep_regions), :]
            keep_regions = list(pop["state"].unique())
            region_col = "state"
        else:
            raise ValueError(e)

    if "sector" in table_cols or "subsector" in table_cols:
        if settings.get("electrification_stock_fn") and settings.get(
            "electrification_scenario"
        ):
            # This is a default list of sector/subsectors that are considered "base" demand
            # and are not affected by stock levels of electric technologies (e.g. EVs and heat pumps)
            # NOTE: This should be parameratized so it can be changed by the user, especially
            # if load data is from a source other than NREL EFS
            base_sector_subsectors = [
                ("commercial", "other"),
                ("residential", "other"),
                ("residential", "clothes and dish washing/drying"),
                ("industrial", "machine drives"),
                ("industrial", "process heat"),
                ("industrial", "other"),
            ]
            s = f"""
                    SELECT year, {region_col} as demand_region, time_index, sector, sum(load_mw) as load_mw
                    FROM {pg_table}
                    WHERE {region_col} in ({','.join(['?']*len(keep_regions))})
                    AND
                    ({' OR '.join(["(sector=? and subsector=?)"]*len(base_sector_subsectors))})
                    GROUP BY year, demand_region, sector, time_index
                    """
            params = keep_regions + [
                item for sublist in base_sector_subsectors for item in sublist
            ]
            load_curves = pd.read_sql_query(sql=s, con=pg_engine, params=params)
        else:
            s = f"""
                    SELECT year, {region_col} as demand_region, time_index, sector, sum(load_mw) as load_mw
                    FROM {pg_table}
                    WHERE {region_col} in ({','.join(['?']*len(keep_regions))})
                    GROUP BY year, region, sector, time_index
                    """
            params = keep_regions
            load_curves = pd.read_sql_query(sql=s, con=pg_engine, params=params)
    else:
        # With no sector or subsector columns, assume that table has total load in each hour
        s = f"""
            SELECT year, {region_col} as demand_region, time_index, load_mw
            FROM {pg_table}
            WHERE {region_col} in ({','.join(['?']*len(keep_regions))})
            """
        params = keep_regions
        load_curves = pd.read_sql_query(sql=s, con=pg_engine, params=params)

    if region_col == "state":
        load_curves = load_curves.rename(columns={"demand_region": "state"})
        load_curves = state_demand_to_region(
            load_curves,
            pop.rename(columns={"demand_region": "region"}),
            by=["time_index", "region", "year"],
        )
        load_curves = load_curves.rename(columns={"region": "demand_region"})
    # Increase demand to account for load growth
    load_curves = add_load_growth(load_curves, settings)

    load_curves.loc[
        load_curves.demand_region.isin(region_agg_map), "demand_region"
    ] = load_curves.demand_region.map(region_agg_map)

    logger.info("Aggregating load curves in grouped regions")
    load_curves_agg = load_curves.groupby(["demand_region", "time_index"])[
        "load_mw"
    ].sum()

    lc_wide = load_curves_agg.unstack(level=0)
    if lc_wide.columns.nlevels > 1:
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
    """Multiply hourly load profiles by AEO or user growth factors.

    If the base data year is from more than one year before the AEO data year then load
    is first grown to that point, then the AEO data year is used to calculate load growth
    to the model planning year.

    Parameters
    ----------
    load_curves : pd.DataFrame
        Tidy dataframe of load curves with columns "region", "load_mw", and optionally
        "sector".
    settings : dict
        User settings parameters. Should include "historical_load_region_map",
        "future_load_region_map", and "model_year". Optional parameters include
        "aeo_sector_map" (mapping load sectors to AEO API sector names), and
        "alt_growth_rate" (either single growth rates for each region or sector-level
        growth rates within each region, where the sector names match those in the load
        profile).

    Returns
    -------
    pd.DataFrame
        Modified version of input dataframe to account for load growth from base year
        to model planning year.
    """
    keep_regions, region_agg_map = regions_to_keep(
        settings["model_regions"], settings.get("region_aggregations")
    )
    hist_region_map = reverse_dict_of_lists(settings["historical_load_region_map"])
    future_region_map = reverse_dict_of_lists(settings["future_load_region_map"])
    aeo_sector_map = settings.get("aeo_sector_map")
    if not settings.get("aeo_sector_map"):
        aeo_sector_map = {
            "commercial": "COMM",
            "industrial": "IDAL",
            "residential": "RESD",
            "transportation": "TRN",
        }
    if "sector" in load_curves.columns:
        load_sectors = set(load_curves.sector.unique())
        aeo_sectors = set(aeo_sector_map)
        if not all([s in aeo_sectors for s in load_sectors]):
            missing_sectors = list(load_sectors - aeo_sectors)
            logger.warning(
                "*********************\n"
                f"The load sectors {missing_sectors} are in your load data but are not "
                "mapped to EIA AEO sectors. The hourly values for these sectors will not "
                "be changed unless you added a growth rate for this sector to all regions "
                "in the settings parameter 'alt_growth_rate'."
                "*********************\n"
            )

    outer_list = []
    for year, df in load_curves.groupby("year"):
        growth_scenario = settings.get("growth_scenario", "REF2020")
        load_aeo_year = settings.get("load_eia_aeo_year") or settings.get(
            "eia_aeo_year", 2020
        )
        while year < load_aeo_year - 1:
            year, df = grow_historical_load(
                df=df,
                year=year,
                aeo_data_year=load_aeo_year,
                keep_regions=keep_regions,
                hist_region_map=hist_region_map,
                future_region_map=future_region_map,
                aeo_sector_map=aeo_sector_map,
                alt_growth_rate=settings.get("alt_growth_rate"),
            )

        df_list = []
        if "sector" in df.columns:
            for sector, _df in df.groupby("sector"):
                load_growth_dict = {
                    region: get_aeo_load(
                        region=future_region_map[region],
                        aeo_year=load_aeo_year,
                        scenario_series=growth_scenario,
                        sector=aeo_sector_map[sector],
                    ).set_index("year")
                    for region in keep_regions
                }

                load_growth_start_map = {
                    region: _df.loc[year, "demand"]
                    for region, _df in load_growth_dict.items()
                }

                load_growth_end_map = {
                    region: _df.loc[settings["model_year"], "demand"]
                    for region, _df in load_growth_dict.items()
                }

                growth_factor = {
                    region: load_growth_end_map[region] / load_growth_start_map[region]
                    for region in keep_regions
                }

                years_growth = settings["model_year"] - year
                for region, rate in (settings.get("alt_growth_rate") or {}).items():
                    if isinstance(rate, dict):
                        if rate.get(sector):
                            growth_factor[region] = (1 + rate["sector"]) ** years_growth
                        else:
                            raise KeyError(
                                f"You specified a sector specific alt_growth_rate for the "
                                f"region '{region}'. The demand data has a sector {sector}, "
                                f"which you did not specify a rate for. Without a sector "
                                "specific growth rate the demand will not be increased."
                            )
                for region in keep_regions:
                    _df.loc[_df["demand_region"] == region, "load_mw"] *= growth_factor[
                        region
                    ]
                df_list.append(_df)
        else:
            load_growth_dict = {
                region: get_aeo_load(
                    region=future_region_map[region],
                    aeo_year=load_aeo_year,
                    scenario_series=growth_scenario,
                ).set_index("year")
                for region in keep_regions
            }

            load_growth_start_map = {
                region: _df.loc[year, "demand"]
                for region, _df in load_growth_dict.items()
            }

            load_growth_end_map = {
                region: _df.loc[settings["model_year"], "demand"]
                for region, _df in load_growth_dict.items()
            }

            growth_factor = {
                region: load_growth_end_map[region] / load_growth_start_map[region]
                for region in keep_regions
            }

            years_growth = settings["model_year"] - year
            for region, rate in (settings.get("alt_growth_rate") or {}).items():
                if isinstance(rate, float):
                    growth_factor[region] = (1 + rate) ** years_growth
            for region in keep_regions:
                df.loc[df["demand_region"] == region, "load_mw"] *= growth_factor[
                    region
                ]
            df_list.append(df)

        annual_load = pd.concat(df_list, ignore_index=True)
        outer_list.append(annual_load)

    load_curves = pd.concat(outer_list, ignore_index=True)

    return load_curves


def grow_historical_load(
    df: pd.DataFrame,
    year: int,
    aeo_data_year: int,
    keep_regions: List[str],
    hist_region_map: Dict[str, str],
    future_region_map: Dict[str, str],
    aeo_sector_map: Dict[str, str] = None,
    alt_growth_rate: Dict[str, float] = None,
) -> Tuple[int, pd.DataFrame]:
    """Grow historical load up to either 2019 or the year before "aeo_data_year".

    If the data year is less than 2019 than use "hist_region_map" to grow load to 2019.
    Otherwise use "future_region_map" to grow load from the data year to the the year
    before "aeo_data_year". This two step process is needed ecause the AEO EMM regions
    changed from AEO2019 to AEO2020.

    The function returns both the modified input dataframe and an updated "year"
    parameter.

    Parameters
    ----------
    df : pd.DataFrame
        Tidy dataframe of hourly load. Should include the columns "region" and "load_mw".
        Can also include the column "sector" with load disaggregated by sector.
    year : int
        Basis year of the load data.
    aeo_data_year : int
        Year of AEO data that will be used to grow load out to future planning periods.
    keep_regions : List[str]
        Regions that are included in the model. Only load from these regions is modified.
    hist_region_map : Dict[str, str]
        A map of load regions to AEO EMM regions from AEO2019 and earlier.
    future_region_map : Dict[str, str]
        A map of load regions to AEO EMM regions from AEO2020 and later.
    aeo_sector_map : Dict[str, str]
        A mapping of sector names from the load data to names used by EIA, by default
        None.
    alt_growth_rate : Dict[str, float], optional
        Alternative growth rates provided by the user, by default None.

    Returns
    -------
    int
        Updated data year
    pd.Dataframe
        Updated load data
    """

    old_aeo_list = []
    if year < 2019:
        region_map = hist_region_map
        start_aeo_year = year + 2
        end_aeo_year = 2019
        end_data_year = 2019
    else:
        region_map = future_region_map
        start_aeo_year = year + 1
        end_aeo_year = aeo_data_year
        end_data_year = end_aeo_year - 1
    if "sector" in df.columns:
        if not aeo_sector_map:
            raise KeyError(
                "The load data provided has the column 'sector' but no mapping of sectors "
                "to AEO sector names was provided."
            )
        for sector, _df in df.groupby("sector"):
            hist_demand_start = {
                region: get_aeo_load(
                    region=region_map[region],
                    aeo_year=start_aeo_year,
                    scenario_series=f"REF{start_aeo_year}",
                    sector=aeo_sector_map[sector],
                )
                .set_index("year")
                .loc[year, "demand"]
                for region in keep_regions
            }

            hist_demand_end = {
                region: get_aeo_load(
                    region=region_map[region],
                    aeo_year=end_aeo_year,
                    scenario_series=f"REF{end_aeo_year}",
                    sector=aeo_sector_map[sector],
                )
                .set_index("year")
                .loc[end_data_year, "demand"]
                for region in keep_regions
            }
            growth_factor = {
                region: hist_demand_end[region] / hist_demand_start[region]
                for region in keep_regions
            }

            years_growth = 1
            for region, rate in (alt_growth_rate or {}).items():
                if isinstance(rate, dict) and rate.get(sector):
                    growth_factor[region] = (1 + rate["sector"]) ** years_growth
            for region in keep_regions:
                _df.loc[_df["demand_region"] == region, "load_mw"] *= growth_factor[
                    region
                ]
            old_aeo_list.append(_df)

    else:
        hist_demand_start = {
            region: get_aeo_load(
                region=region_map[region],
                aeo_year=start_aeo_year,
                scenario_series=f"REF{start_aeo_year}",
            )
            .set_index("year")
            .loc[year, "demand"]
            for region in keep_regions
        }
        hist_demand_end = {
            region: get_aeo_load(
                region=region_map[region],
                aeo_year=end_aeo_year,
                scenario_series=f"REF{end_aeo_year}",
            )
            .set_index("year")
            .loc[end_data_year, "demand"]
            for region in keep_regions
        }
        growth_factor = {
            region: hist_demand_end[region] / hist_demand_start[region]
            for region in keep_regions
        }

        years_growth = 1
        for region, rate in (alt_growth_rate or {}).items():
            if isinstance(rate, float):
                growth_factor[region] = (1 + rate) ** years_growth
        for region in keep_regions:
            df.loc[df["demand_region"] == region, "load_mw"] *= growth_factor[region]
        old_aeo_list.append(df)

    df = pd.concat(old_aeo_list, ignore_index=True)

    year = end_data_year
    return year, df


def add_demand_response_resource_load(load_curves, settings):
    dr_path = Path(settings["input_folder"]) / settings["demand_response_fn"]
    dr_types = list(
        settings["flexible_demand_resources"][settings["model_year"]].keys()
    )

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

    load_curves.index.name = "time_index"
    load_curves.index = load_curves.index + 1

    return load_curves


def subtract_distributed_generation(load_curves, pg_engine, settings):
    dg_profiles = make_distributed_gen_profiles(pg_engine, settings)
    dg_profiles.index = dg_profiles.index + 1

    for col in dg_profiles.columns:
        load_curves.loc[:, col] = load_curves.loc[:, col] - (
            dg_profiles[col].values * 1 + settings["avg_distribution_loss"]
        )

    return load_curves


@filter_load_by_region(load_source="USER")
def load_usr_demand_profiles(settings):
    """Temp function. Loads user demand profiles if the file name is provided, else returns None.
    If only specified regions are to be used (settings["regional_load_source"]["USER"]), then
    reindex to use only those regions. Else, returns all regions in the regional load file.
    """
    logger.info("Loading user supplied load profile.")
    regional_load_fn = settings.get("regional_load_fn")

    if regional_load_fn is not None:
        from powergenome.external_data import make_usr_demand_profiles

        lp_path = settings["input_folder"] / regional_load_fn
        hourly_load_profiles = make_usr_demand_profiles(lp_path, settings)

        if len(hourly_load_profiles) == 8784:
            remove_feb_29(hourly_load_profiles)

        hourly_load_profiles.index.name = "time_index"
        hourly_load_profiles.index = pd.RangeIndex(
            start=1, stop=len(hourly_load_profiles) + 1, step=1
        )

        regional_load_sources = settings.get("regional_load_source")
        if regional_load_sources is not None:
            if regional_load_sources == "USER":
                cols = settings.get("model_regions")
            else:
                cols = regional_load_sources.get("USER")
            if not all([col in hourly_load_profiles.columns for col in cols]):
                raise KeyError(
                    f"One or more of the regions {cols} is not included in your "
                    f"user-supplied load curves file {regional_load_fn}."
                )
            hourly_load_profiles = hourly_load_profiles.reindex(columns=cols)

        return hourly_load_profiles

    else:
        logger.info("User supplied load profile not found.")
        return None


def make_final_load_curves(
    pg_engine: sa.engine.base.Engine,
    settings: dict,
):
    """Create final load profiles from base year including growth, dg, and flexible loads

    Parameters
    ----------
    pg_engine : sa.engine.base.Engine
        Engine to connect to a PowerGenome database
    settings : dict
        User parameter settings. Required keys are "model_regions", "future_load_region_map",
        and "historical_load_region_maps" (if load data represent a year before
        2019). Optional keys include "load_source_table_name", "demand_response_fn",
        "distributed_gen_profiles_fn", "dg_as_resource", "region_aggregations",
        "electrification_stock_fn", "electrification_scenario", and "alt_growth_rate".

    Returns
    -------
    pd.DataFrame
        Wide dataframe with one column of load profiles for each model region

    Raises
    ------
    ValueError
        When all load curves are null.
    """

    logger.info("Loading load curves")
    user_load_curves = load_usr_demand_profiles(settings)

    if user_load_curves is not None and all(
        [r in user_load_curves.columns for r in settings["model_regions"]]
    ):
        load_curves_before_dr = user_load_curves

    else:
        load_sources = settings.get("load_source_table_name")
        if load_sources is None:
            s = """
            *****************************
            Regional load data sources have not been specified. Defaulting to EFS load data.
            Check your settings file, and please specify the preferred source for load data
            (FERC, EFS, USER) either for each region or for the entire system with the setting
            "regional_load_source".
            *****************************
            """
            logger.warning(s)
            load_sources = {"EFS": "load_curves_nrel_efs_state"}

        # `filter_load_by_region` is a decorator factory that generates a decorator
        # when given the parameter `load_source`. This decorator creates a wrapper
        # for the function `make_load_curves`, which is passed the args from the final
        # parentheses.
        load_curves_before_dr = [
            filter_load_by_region(load_source)(make_load_curves)(
                pg_engine, settings, load_table
            )
            for load_source, load_table in load_sources.items()
        ]
        load_curves_before_dr.append(user_load_curves)
        load_curves_before_dr = [df for df in load_curves_before_dr if df is not None]
        if not all(
            [
                len(load_curves_before_dr[0].index.intersection(df.index))
                == load_curves_before_dr[0].shape[0]
                for df in load_curves_before_dr
                if df is not None
            ]
        ):
            raise ValueError(
                "One or more of your load curve data sources does not have a matching time index."
            )

        try:
            load_curves_before_dr = pd.concat(load_curves_before_dr, axis=1)
        except ValueError:
            raise ValueError("All load curves are null.")

    if settings.get("demand_response_fn"):
        if settings.get("regional_load_includes_demand_response"):
            load_curves_before_dg = load_curves_before_dr
        else:
            load_curves_before_dg = add_demand_response_resource_load(
                load_curves_before_dr, settings
            )
    elif settings.get("electrification_stock_fn") and settings.get(
        "electrification_scenario"
    ):
        load_curves_before_dg = load_curves_before_dr.copy()
        keep_regions, region_agg_map = regions_to_keep(
            settings["model_regions"], settings.get("region_aggregations", {}) or {}
        )

        flex_profiles = electrification_profiles(
            settings.get("electrification_stock_fn"),
            settings["model_year"],
            settings.get("electrification_scenario"),
            keep_regions,
            settings.get("utc_offset", 0),
            settings.get("EFS_DATA"),
        )
        flex_profiles = map_agg_region_names(
            flex_profiles, region_agg_map, "region", "model_region"
        )
        for region in load_curves_before_dg.columns:
            region_flex_load = (
                flex_profiles.query("model_region==@region")
                .groupby("time_index")["load_mw"]
                .sum()
            )
            if not region_flex_load.empty:
                load_curves_before_dg[region] += region_flex_load
    else:
        load_curves_before_dg = load_curves_before_dr

    if (
        settings.get("distributed_gen_profiles_fn")
        or settings.get("distributed_gen_fn")
    ) and not settings.get("dg_as_resource"):
        final_load_curves = subtract_distributed_generation(
            load_curves_before_dg, pg_engine, settings
        )
    else:
        final_load_curves = load_curves_before_dg

    final_load_curves = final_load_curves.astype(int)

    # change order to match model regions
    model_regions = settings.get("model_regions")
    if not all(r in final_load_curves.columns for r in model_regions):
        missing_regions = set(final_load_curves.columns) - set(model_regions)
        logger.warning(
            "You have supplied regional load in an external file, but the load for some "
            f"regions is missing. The regions {missing_regions} are not included in the file. "
            "The load for these regions will not be included in output files."
        )
    final_load_curves = final_load_curves.reindex(columns=model_regions)

    return final_load_curves


def make_distributed_gen_profiles(pg_engine, settings):
    """Create 8760 annual generation profiles for distributed generation in regions.
    Uses a distribution loss parameter in the settings file when DG generation is
    defined a fraction of delivered load.

    Parameters
    ----------
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

    if settings.get("distributed_gen_fn"):
        scenario = settings.get("distributed_gen_scenario")
        path_in = settings.get("")
        if settings.get("region_aggregations"):
            regions = [
                r
                for r in settings["model_regions"]
                if r not in settings["region_aggregations"].keys()
            ]
            regions.extend(
                list(reverse_dict_of_lists(settings["region_aggregations"]).keys())
            )
        else:
            regions = settings["model_regions"]

        dg_profiles = distributed_gen_profiles(
            settings.get("distributed_gen_fn"),
            settings["model_year"],
            scenario,
            regions,
            settings.get("DISTRIBUTED_GEN_DATA"),
            settings.get("region_aggregations"),
            settings.get("utc_offset"),
        )
        return dg_profiles

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
