"""
Hourly demand profiles
"""

import logging
from functools import lru_cache
from inspect import signature
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from powergenome.database import get_data, list_tables
from powergenome.distributed_gen import get_distributed_gen_hourly_generation
from powergenome.eia_opendata import get_aeo_load
from powergenome.external_data import make_demand_response_profiles
from powergenome.load_construction import electrification_profiles
from powergenome.settings import auto_fill_settings
from powergenome.util import (
    deep_freeze_args,
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


# @deep_freeze_args
# @lru_cache
# def read_subsector_demand(
#     pg_engine_str: str, keep_regions: List[str], pg_table: str, region_col: str
# ) -> pd.DataFrame:
#     pg_engine = sa.create_engine(pg_engine_str)
#     # This is a default list of sector/subsectors that are considered "base" demand
#     # and are not affected by stock levels of electric technologies (e.g. EVs and heat pumps)
#     # NOTE: This should be parameratized so it can be changed by the user, especially
#     # if load data is from a source other than NREL EFS
#     base_sector_subsectors = [
#         ("commercial", "other"),
#         ("residential", "other"),
#         ("residential", "clothes and dish washing/drying"),
#         ("industrial", "machine drives"),
#         ("industrial", "process heat"),
#         ("industrial", "other"),
#     ]
#     s = f"""
#             SELECT year, {region_col} as region, time_index, sector, sum(load_mw) as load_mw
#             FROM {pg_table}
#             WHERE {region_col} in ({','.join(['?']*len(keep_regions))})
#             AND
#             ({' OR '.join(["(sector=? and subsector=?)"]*len(base_sector_subsectors))})
#             GROUP BY year, region, sector, time_index
#             """
#     params = list(keep_regions) + [
#         item for sublist in base_sector_subsectors for item in sublist
#     ]
#     load_curves = pd.read_sql_query(sql=s, con=pg_engine, params=params)

#     return load_curves


def make_load_curves(
    settings: dict,
) -> pd.DataFrame:
    """Read base load profiles from DataManager and grow the load to a future year.

    Parameters
    ----------
    settings : dict
        User parameter settings. Required keys are "model_regions", "future_load_region_map",
        and "historical_load_region_maps" (if load data represent a year before
        2019). Optional keys include "region_aggregations", "electrification_stock_fn",
        "electrification_scenario", and "alt_growth_rate".

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
    logger.debug("Loading demand profiles from the database")
    # inst = sa.inspect(pg_engine)
    # if not inst.has_table(pg_table):
    #     raise KeyError(
    #         f"There is no load curves table with the name {pg_table} in the 'PG_DB' "
    #         "database specified in your .env file."
    #     )
    # table_cols = [c["name"] for c in inst.get_columns(pg_table)]
    # context = f"Load curves table ({pg_table} in database {pg_engine}."
    # region_col = find_region_col(table_cols, context)
    # if "sector" in table_cols or "subsector" in table_cols:
    #     if settings.get("electrification_stock_fn") and settings.get(
    #         "electrification_scenario"
    #     ):
    #         load_curves = read_subsector_demand(
    #             pg_engine_str=str(pg_engine)[7:-1],
    #             keep_regions=keep_regions,
    #             pg_table=pg_table,
    #             region_col=region_col,
    #         )
    #     else:
    #         s = f"""
    #                 SELECT year, {region_col} as region, time_index, sector, sum(load_mw) as load_mw
    #                 FROM {pg_table}
    #                 WHERE {region_col} in ({','.join(['?']*len(keep_regions))})
    #                 GROUP BY year, region, sector, time_index
    #                 """
    #         params = keep_regions
    #         load_curves = pd.read_sql_query(sql=s, con=pg_engine, params=params)
    #         load_curves = add_load_growth(load_curves, settings)
    # else:
    # With no sector or subsector columns, assume that table has total load in each hour
    # s = f"SELECT DISTINCT year from {pg_table}"
    # # demand_years = pd.read_sql_query(sql=s, con=pg_engine)
    # demand_years = load_data(data_location, pg_table, query=s)
    # region_params = ", ".join(f"'{r}'" for r in keep_regions)

    load_curve_columns = get_data(
        table_name="demand", query="PRAGMA table_info('demand')"
    ).name.to_list()
    if "weather_year" in load_curve_columns:
        if settings.get("weather_year"):
            weather_years = (
                settings["weather_year"]
                if isinstance(settings["weather_year"], list)
                else [settings["weather_year"]]
            )
        else:
            # find available weather years for the model year in the demand table
            s = f"""
                SELECT DISTINCT weather_year
                FROM demand
                WHERE year = {settings["model_year"]}
                """
            data_weather_years = get_data(
                "demand",
                query=s,
            ).weather_year.to_list()
            weather_years = data_weather_years

    filters = [
        [
            ("year", "=", settings["model_year"]),
            ("region", "in", keep_regions),
        ]
        + (
            [("weather_year", "in", weather_years)]
            if settings.get("weather_year")
            else []
        )
    ]
    get_cols = ["year", "region", "time_index", "load_mw", "weather_year"]
    load_curves = get_data(
        "demand",
        columns=[c for c in get_cols if c in load_curve_columns],
        filters=filters,
    )
    if "weather_year" in load_curves.columns:
        if load_curves.weather_year.nunique() < len(weather_years):
            missing_years = set(weather_years) - set(load_curves.weather_year.unique())
            raise ValueError(
                "*********************\n"
                f"The following weather_years were requested in the settings but are not "
                f"available in the demand profiles table: {missing_years}. "
                "*********************\n"
            )
        for r in keep_regions:
            region_mask = load_curves["region"] == r
            region_data = load_curves.loc[region_mask]
            if not region_data.empty:
                load_curves.loc[region_mask, "time_index"] = np.arange(
                    1, len(region_data) + 1
                )
    # if settings["model_year"] in demand_years["year"].to_list():
    #     s = f"""
    #         SELECT year, region, time_index, load_mw
    #         FROM {pg_table}
    #         WHERE region in ({region_params})
    #         AND year = {settings['model_year']}
    #         """
    #     params = keep_regions
    #     # load_curves = pd.read_sql_query(sql=s, con=pg_engine, params=params)
    #     load_curves = load_data(data_location, pg_table, query=s)

    # else:
    #     s = f"""
    #         SELECT year, region, time_index, load_mw
    #         FROM {pg_table}
    #         WHERE region in ({','.join(['?']*len(keep_regions))})
    #         """
    #     params = keep_regions
    #     # load_curves = pd.read_sql_query(sql=s, con=pg_engine, params=params)
    #     load_curves = load_data(data_location, pg_table, query=s)

    #     # Increase demand to account for load growth
    #     load_curves = add_load_growth(load_curves, settings)

    load_curves.loc[load_curves.region.isin(region_agg_map), "region"] = (
        load_curves.region.map(region_agg_map)
    )

    logger.debug("Aggregating load curves in grouped regions")
    load_curves_agg = load_curves.groupby(["region", "time_index"])["load_mw"].sum()

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
                    _df.loc[_df["region"] == region, "load_mw"] *= growth_factor[region]
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
                df.loc[df["region"] == region, "load_mw"] *= growth_factor[region]
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
                _df.loc[_df["region"] == region, "load_mw"] *= growth_factor[region]
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
            df.loc[df["region"] == region, "load_mw"] *= growth_factor[region]
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


@auto_fill_settings()
def subtract_distributed_generation(
    load_curves: pd.DataFrame,
    model_year: int,
    weather_year: Optional[int] = None,
    model_regions: List[str] = None,
    region_aggregations: Optional[Dict[str, List[str]]] = None,
    utc_offset: Optional[int] = None,
    avg_distribution_loss: Optional[float] = None,
):
    """Subtract distributed generation from load curves.

    Parameters
    ----------
    load_curves : pd.DataFrame
        Wide dataframe of hourly load profiles by region
    model_year : int
        Model planning year
    weather_year : int, optional
        Weather year selection filter for distributed generation profiles, by default None
    model_regions : List[str], optional
        List of model regions, by default None
    region_aggregations : Dict[str, List[str]], optional
        Mapping of aggregated regions to list of model regions, by default None
    utc_offset : int, optional
        UTC offset for time-shifting profiles, by default None
    avg_distribution_loss : float, optional
        Average distribution loss to account for when subtracting distributed generation,
        by default None

    Returns
    -------
    pd.DataFrame
        Modified load curves with distributed generation subtracted
    """

    # Get hourly distributed generation
    dg_hourly_gen = get_distributed_gen_hourly_generation(
        year=model_year,
        weather_year=weather_year,
        regions=model_regions,
        region_aggregations=region_aggregations,
        tz_offset=utc_offset,
    )

    if dg_hourly_gen.empty:
        logger.info("No distributed generation data found, load curves unchanged")
        return load_curves

    # Ensure indices match
    dg_hourly_gen.index = dg_hourly_gen.index

    # Account for distribution losses
    dist_loss_factor = 1 + (avg_distribution_loss or 0)

    # Subtract DG generation from load (with loss adjustment)
    for col in dg_hourly_gen.columns:
        if col in load_curves.columns:
            load_curves.loc[:, col] = load_curves.loc[:, col] - (
                dg_hourly_gen[col].values * dist_loss_factor
            )
        else:
            logger.warning(
                f"Region {col} has distributed generation but is not in load curves"
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
        logger.warning("User supplied load profile not found.")
        return None


def add_supplemental_demand(
    load_curves: pd.DataFrame,
    model_year: int,
    model_regions: List[str],
    hours_per_year: int = 8760,
) -> pd.DataFrame:
    """Add supplemental hourly demand from the DataManager ``supplemental_demand`` table.

    The supplemental demand table should have at minimum the columns:

    * ``region`` – model region name
    * ``time_index`` – integer hour index (1-based) **or** the string ``"all_hours"``
    * ``load_mw`` – MW of demand to add

    Optional columns:

    * ``year`` – when present, rows are filtered to ``model_year``
    * ``weather_year`` – when present, rows with NULL/NaN ``weather_year`` are tiled
      across all weather-year blocks (see ``hours_per_year``); rows with a specific
      ``weather_year`` value are applied directly to matching ``time_index`` values.

    A ``time_index`` value of ``"all_hours"`` is expanded to every hour present in
    ``load_curves`` (i.e., every row in the index).  When a specific integer is given,
    tiling behaviour depends on whether a ``weather_year`` column is present:

    * **No** ``weather_year`` column – rows with specific ``time_index`` values are
      applied directly (no tiling).  Use ``"all_hours"`` if the supplement should
      apply across all weather years.
    * **With** ``weather_year`` column – rows whose ``weather_year`` is NULL/NaN are
      tiled across weather-year blocks of ``hours_per_year`` hours each.

    If the ``supplemental_demand`` table is not registered in the DataManager this
    function is a no-op and returns ``load_curves`` unchanged.

    Parameters
    ----------
    load_curves : pd.DataFrame
        Wide dataframe with one column per model region and ``time_index`` as the index.
        This is the same format returned by :func:`make_load_curves`.
    model_year : int
        Planning year used to filter the supplemental demand table (applied when the
        table contains a ``year`` column).
    model_regions : List[str]
        Model region names; only regions present in both ``load_curves.columns`` and
        the supplemental demand table are modified.
    hours_per_year : int, optional
        Number of hours in a single weather year, used to tile supplemental demand
        across multiple weather years when ``weather_year`` is NULL/NaN.  Defaults to
        8760 (standard non-leap year).

    Returns
    -------
    pd.DataFrame
        A copy of ``load_curves`` with the supplemental demand added.
    """
    if "supplemental_demand" not in list_tables():
        return load_curves

    # Discover available columns in the supplemental demand table.
    try:
        col_info = get_data(
            "supplemental_demand",
            query="PRAGMA table_info('supplemental_demand')",
        )
        supp_cols = col_info["name"].tolist()
    except Exception:
        col_info = get_data(
            "supplemental_demand",
            query="DESCRIBE supplemental_demand",
        )
        supp_cols = col_info.iloc[:, 0].tolist()

    # Build filters for the model year.
    filters = None
    if "year" in supp_cols:
        filters = [
            [("year", "=", model_year)],
        ]

    supp_df = get_data("supplemental_demand", filters=filters)

    if supp_df.empty:
        return load_curves

    load_curves = load_curves.copy()
    all_time_indices = load_curves.index.tolist()
    total_hours = len(all_time_indices)
    has_weather_year_col = "weather_year" in supp_cols

    # Determine how many weather-year blocks fit in load_curves.
    if total_hours > hours_per_year and total_hours % hours_per_year == 0:
        num_blocks = total_hours // hours_per_year
    else:
        num_blocks = 1

    # Separate rows with "all_hours" from rows with specific time indices.
    all_hours_mask = supp_df["time_index"].astype(str).str.strip() == "all_hours"
    all_hours_df = supp_df[all_hours_mask].copy()
    specific_df = supp_df[~all_hours_mask].copy()

    # --- Helper: add a load_mw Series (indexed by time_index) to load_curves ---
    def _apply(region: str, by_time: "pd.Series") -> None:
        if region not in load_curves.columns:
            logger.debug(
                "Supplemental demand region '%s' not in model regions; skipping.", region
            )
            return
        common = load_curves.index.intersection(by_time.index)
        if common.empty:
            return
        load_curves.loc[common, region] = (
            load_curves.loc[common, region] + by_time.loc[common]
        )

    # --- Process "all_hours" rows ---
    for region, region_df in all_hours_df.groupby("region"):
        total_mw = region_df["load_mw"].sum()
        by_time = pd.Series(total_mw, index=all_time_indices, dtype=float)
        _apply(region, by_time)

    # --- Process specific time_index rows ---
    if not specific_df.empty:
        specific_df = specific_df.copy()
        specific_df["time_index"] = pd.to_numeric(
            specific_df["time_index"], errors="coerce"
        )
        specific_df = specific_df.dropna(subset=["time_index"])
        specific_df["time_index"] = specific_df["time_index"].astype(int)

        if has_weather_year_col and num_blocks > 1:
            # Tile rows with NULL weather_year across all weather-year blocks; apply
            # rows with a specific weather_year directly.
            null_wy = specific_df["weather_year"].isna()
            null_wy_df = specific_df[null_wy]
            valid_wy_df = specific_df[~null_wy]

            if not null_wy_df.empty:
                expanded_rows = [null_wy_df]
                for offset in range(1, num_blocks):
                    shifted = null_wy_df.copy()
                    shifted["time_index"] = (
                        shifted["time_index"] + offset * hours_per_year
                    )
                    expanded_rows.append(shifted)
                null_wy_df = pd.concat(expanded_rows, ignore_index=True)
                for region, region_df in null_wy_df.groupby("region"):
                    by_time = region_df.groupby("time_index")["load_mw"].sum()
                    _apply(region, by_time)

            if not valid_wy_df.empty:
                for region, region_df in valid_wy_df.groupby("region"):
                    by_time = region_df.groupby("time_index")["load_mw"].sum()
                    _apply(region, by_time)
        else:
            # No weather_year column or single block: apply rows directly.
            for region, region_df in specific_df.groupby("region"):
                by_time = region_df.groupby("time_index")["load_mw"].sum()
                _apply(region, by_time)

    return load_curves


@auto_fill_settings()
def make_final_load_curves(
    data_location: Path | str = None,
    settings: dict = None,
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

    logger.info("Creating hourly demand profiles")
    user_load_curves = load_usr_demand_profiles(settings)

    if user_load_curves is not None and all(
        [r in user_load_curves.columns for r in settings["model_regions"]]
    ):
        load_curves_before_dr = user_load_curves

    else:
        load_sources = settings.get("load_source_table_name")
        if load_sources is None:
            s = (
                "Regional load data sources have not been specified. Defaulting to EFS load data. "
                "See documentation of the parameter 'regional_load_source' to use other data."
            )
            logger.info(s)
            load_sources = {"EFS": "load_curves_nrel_efs"}

        # `filter_load_by_region` is a decorator factory that generates a decorator
        # when given the parameter `load_source`. This decorator creates a wrapper
        # for the function `make_load_curves`, which is passed the args from the final
        # parentheses.
        load_curves_before_dr = [
            filter_load_by_region(load_source)(make_load_curves)(settings)
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

    if not settings.get("dg_as_resource"):
        final_load_curves = subtract_distributed_generation(
            load_curves_before_dg,
            model_year=settings["model_year"],
            weather_year=settings.get("weather_year"),
            model_regions=settings["model_regions"],
            region_aggregations=settings.get("region_aggregations"),
            utc_offset=settings.get("utc_offset"),
            avg_distribution_loss=settings.get("avg_distribution_loss"),
        )
    else:
        final_load_curves = load_curves_before_dg

    # Add supplemental demand (e.g. data center forecasts) if the table is available.
    final_load_curves = add_supplemental_demand(
        final_load_curves,
        model_year=settings["model_year"],
        model_regions=settings["model_regions"],
    )

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


@auto_fill_settings(regions="model_regions", region_aggregations="region_aggregations")
def make_distributed_gen_profiles(
    regions: List[str],
    region_aggregations: Optional[Dict[str, List[str]]] = None,
    weather_year: Optional[int] = None,
    utc_offset: Optional[int] = None,
) -> pd.DataFrame:
    """Create 8760 annual generation profiles for distributed generation in regions.

    This function retrieves normalized profiles from DataManager and returns them
    for use in creating generator resources.

    Parameters
    ----------
    regions : List[str]
        List of model regions (after aggregation)
    region_aggregations : Optional[Dict[str, List[str]]], optional
        Mapping of aggregated regions to their component regions, by default None
    weather_year : Optional[int], optional
        Weather year for profiles, by default None
    utc_offset : Optional[int], optional
        UTC offset for time zone adjustments, by default None
    Returns
    -------
    DataFrame
        Hourly normalized generation profiles (0-1 range) for DG resources in each region.
        Not all regions need to be accounted for.
    """
    from powergenome.distributed_gen import get_distributed_gen_profiles

    logger.info("Creating distributed generation profiles")

    dg_profiles = get_distributed_gen_profiles(
        weather_year=weather_year,
        regions=regions,
        region_aggregations=region_aggregations,
        tz_offset=utc_offset,
    )

    return dg_profiles


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
