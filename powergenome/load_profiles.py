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
    weather_years = None
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
    has_weather_year = "weather_year" in load_curves.columns
    if has_weather_year:
        if load_curves.weather_year.nunique() < len(weather_years):
            missing_years = set(weather_years) - set(load_curves.weather_year.unique())
            raise ValueError(
                "*********************\n"
                f"The following weather_years were requested in the settings but are not "
                f"available in the demand profiles table: {missing_years}. "
                "*********************\n"
            )

    # Merge supplemental demand (e.g. data-center forecasts) into the LONG load
    # frame NOW, before hours are renumbered and base regions are aggregated.
    # Supplemental rows carry the same (region, weather_year, time_index) as the
    # base load hour they augment, so the renumbering + aggregation below handles
    # them with no block-tiling and no fixed weather-year length.
    load_curves = add_supplemental_demand_long(
        load_curves,
        model_year=settings["model_year"],
        keep_regions=keep_regions,
        region_agg_map=region_agg_map,
        region_aggregations=settings.get("region_aggregations", {}) or {},
        weather_years=weather_years if has_weather_year else None,
    )

    # Renumber hours sequentially 1..N per region. Base and supplemental rows
    # that share (region, weather_year, time_index) receive the SAME renumbered
    # hour, and weather-year block lengths are derived from the base-load rows so
    # variable-length (e.g. leap vs non-leap) weather years both work. The frame
    # is sorted by (region, weather_year, time_index) first so the numbering is
    # deterministic regardless of the order rows arrive from the database.
    if has_weather_year:
        load_curves = _renumber_load_hours(load_curves, weather_years)
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


def _norm_weather_year(value):
    """Normalize a ``weather_year`` value for matching.

    ``None``/``NaN``/empty string -> ``None``, ``"all"`` (any case) -> ``"all"``,
    numeric strings and numbers -> ``int``, anything else is returned unchanged.
    """
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        if s.lower() == "all":
            return "all"
        try:
            return int(s)
        except ValueError:
            return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def _load_supplemental_demand(model_year: int) -> Optional[pd.DataFrame]:
    """Load and normalize the ``supplemental_demand`` table from DataManager.

    Returns a long-format DataFrame with at least the columns ``region``,
    ``time_index``, ``load_mw``, plus the normalized helper column ``_time``
    (either the string ``"all"`` or a single integer hour) and, when the table has
    a ``weather_year`` column, the normalized ``_norm_wy`` column (a concrete
    year, ``"all"``, or ``None`` for blank).  Returns ``None`` when the table is
    not registered or is empty after filtering.

    Because the result is long format, ``time_index`` here is the *per-weather
    year* hour number (e.g. 1..8760) rather than a global hour number — the caller
    is responsible for arranging the rows into weather-year blocks.

    Parameters
    ----------
    model_year : int
        Planning year used to filter the table (only applied when the table
        contains a ``year`` column).
    """
    if "supplemental_demand" not in list_tables():
        return None

    # Discover the columns available in the supplemental demand table.
    try:
        col_info = get_data(
            "supplemental_demand", query="PRAGMA table_info('supplemental_demand')"
        )
        supp_cols = col_info["name"].tolist()
    except Exception:
        col_info = get_data("supplemental_demand", query="DESCRIBE supplemental_demand")
        supp_cols = col_info.iloc[:, 0].tolist()

    # Validate that the table has the required columns so users get a helpful
    # error instead of a raw KeyError later.
    missing_cols = [
        c for c in ("region", "time_index", "load_mw") if c not in supp_cols
    ]
    if missing_cols:
        raise ValueError(
            "The supplemental_demand table is missing required column(s): "
            f"{', '.join(missing_cols)}. It must include 'region', 'time_index', "
            "and 'load_mw'. Check the table/file configured in your settings "
            "file, for example:\n\n"
            "    supplemental_demand_table:\n"
            "      table_name: <your_table_or_file>"
        )

    # Only keep rows for the current model year when the table has a year column;
    # the rest of the filtering is left to the DataManager config.
    filters = None
    if "year" in supp_cols:
        filters = [
            [("year", "=", model_year)],
        ]

    supp_df = get_data("supplemental_demand", filters=filters)
    if supp_df.empty:
        return None

    # A `scenario` column must resolve to exactly one scenario after any
    # DataManager-level filtering; otherwise the user must select one in settings.
    if "scenario" in supp_df.columns:
        scenarios = supp_df["scenario"].dropna().unique().tolist()
        if len(scenarios) > 1:
            scenario_list = ", ".join(
                f"'{s}'" for s in sorted(str(s) for s in scenarios)
            )
            raise ValueError(
                "The supplemental_demand table contains multiple scenarios "
                f"({scenario_list}) but no scenario has been selected. "
                "Specify which scenario to use in your settings file, for "
                "example:\n\n"
                "    supplemental_demand_table:\n"
                f"      table_name: <your_table_or_file>\n"
                f"      scenario: {sorted(str(s) for s in scenarios)[0]}"
            )

    supp_df = supp_df.copy()

    # Normalize time_index: "all" / "all_hours" -> "all"; anything else must be a
    # single integer hour (rows with other values are dropped with a warning).
    def _norm_time_index(value):
        if isinstance(value, str):
            s = value.strip().lower()
            if s in ("all", "all_hours"):
                return "all"
        return value

    supp_df["_time"] = supp_df["time_index"].map(_norm_time_index)
    time_as_str = supp_df["_time"].astype(str).str.strip().str.lower()
    not_all = ~time_as_str.eq("all")
    supp_df.loc[not_all, "_time"] = pd.to_numeric(
        supp_df.loc[not_all, "_time"], errors="coerce"
    )
    n_dropped = int(supp_df["_time"].isna().sum())
    if n_dropped:
        logger.warning(
            "Dropping %d supplemental demand row(s) with a non-numeric, non-all "
            "time_index.",
            n_dropped,
        )
    supp_df = supp_df.dropna(subset=["_time"])
    if not supp_df.empty:
        time_as_str = supp_df["_time"].astype(str).str.strip().str.lower()
        numeric_mask = ~time_as_str.eq("all")
        supp_df.loc[numeric_mask, "_time"] = supp_df.loc[numeric_mask, "_time"].astype(
            int
        )

    if "weather_year" in supp_cols:
        # Normalize weather_year: blank -> None (skipped), "all" -> "all",
        # numeric strings -> int.
        supp_df["_norm_wy"] = supp_df["weather_year"].map(_norm_weather_year)
        blank_mask = supp_df["_norm_wy"].isna()
        n_blank = int(blank_mask.sum())
        if n_blank:
            logger.info(
                "Skipping %d supplemental demand row(s) with a blank weather_year; "
                "use weather_year='all' to apply them across every weather year.",
                n_blank,
            )
        supp_df = supp_df[~blank_mask]

    if supp_df.empty:
        return None

    return supp_df


def _resolve_supp_region(
    region: str,
    region_agg_map: Dict[str, str],
    region_aggregations: Dict[str, List[str]],
    keep_regions: List[str],
) -> Optional[str]:
    """Resolve a supplemental demand region name to a base load-region name.

    Supplemental demand rows may name a region either by its **base** (IPM
    sub-region) name or by the **model** region it is aggregated into.  Returns
    the base region whose rows the supplement should be added to, or ``None`` if
    the name does not resolve to any base or model region.

    Resolution order:

      1. A base region that is aggregated into a model region (a key of
         ``region_agg_map``, e.g. ``p1``) is used as-is.
      2. A model region name (a key of ``region_aggregations``, e.g. ``p1_2``) is
         mapped to its *first* base region.  Because base regions are summed when
         aggregated, adding the supplement to any one member of the group adds
         exactly the supplemental amount to the aggregated model-region profile.
      3. A standalone model region (present in ``keep_regions``, e.g. ``p3``) has
         the same name as its base region and is used as-is.

    A name matching none of these is logged as a warning and skipped.
    """
    if region in region_agg_map:
        return region
    if region in region_aggregations:
        return region_aggregations[region][0]
    if region in keep_regions:
        return region
    logger.warning(
        "Supplemental demand region '%s' does not match any base or model region; "
        "its rows will be skipped. Check the 'region' values in your "
        "supplemental_demand table.",
        region,
    )
    return None


def add_supplemental_demand_long(
    load_curves: pd.DataFrame,
    model_year: int,
    keep_regions: List[str],
    region_agg_map: Dict[str, str],
    region_aggregations: Dict[str, List[str]],
    weather_years: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Add supplemental demand to a long (one row per region x hour) load frame.

    Called from :func:`make_load_curves` after the base demand rows are loaded but
    BEFORE hours are renumbered to 1..N and before base regions are aggregated to
    model regions.  Supplemental rows are emitted in the same long format as the
    base load rows (one row per region x weather year x hour) and concatenated
    onto ``load_curves``; the caller's renumbering + groupby/sum aggregation then
    merges the two sets of rows.  There is no block-tiling and no assumption that
    every weather year has the same number of hours.

    Each row is tagged with ``_is_supp`` (0 = base load, 1 = supplemental) so the
    renumbering step can size weather-year blocks from the base-load rows alone.

    Parameters
    ----------
    load_curves : pd.DataFrame
        Long dataframe of base load with columns ``region``, ``time_index``,
        ``load_mw`` and (optionally) ``weather_year``.
    model_year : int
        Planning year passed to :func:`_load_supplemental_demand`.
    keep_regions, region_agg_map, region_aggregations
        Region resolution inputs from :func:`powergenome.util.regions_to_keep`.
    weather_years : Optional[List[int]], optional
        Ordered list of weather years known to the load data (usually
        ``settings["weather_year"]``).  Used to expand ``"all"`` weather_year rows
        and to validate coverage and specific-year references.
    """
    load_curves = load_curves.copy()
    load_curves["_is_supp"] = 0

    supp = _load_supplemental_demand(model_year)
    if supp is None:
        return load_curves

    has_wy_col = "weather_year" in supp.columns
    has_base_wy = "weather_year" in load_curves.columns
    present_wys = (
        sorted(load_curves["weather_year"].dropna().unique().tolist())
        if has_base_wy
        else []
    )

    # Coverage check: run only when the supplemental table has a weather_year
    # column AND the load data's weather years are known.  Every weather year in
    # the load data must be covered by a specific-year row or an "all" row.
    if has_wy_col and weather_years:
        covered_years = set(
            supp.loc[~supp["_norm_wy"].astype(str).eq("all"), "_norm_wy"]
        )
        if (supp["_norm_wy"] == "all").any():
            covered_years = set(weather_years)
        missing_years = sorted(set(weather_years) - covered_years)
        if missing_years:
            raise ValueError(
                "The supplemental_demand table does not cover weather year(s) "
                f"{missing_years} present in the load data. Add rows with "
                "weather_year='all' or a row for each specific weather year so "
                "the supplemental demand can be applied to every weather year."
            )

    # Base hours present in the load data per (region, weather_year).  Used to
    # expand "all" rows to the real hours and to skip hours that do not exist.
    if has_base_wy:
        base_hours = {
            (reg, wy): sorted(grp["time_index"].dropna().unique().tolist())
            for (reg, wy), grp in load_curves.groupby(
                ["region", "weather_year"], dropna=False, sort=False
            )
        }
    else:
        base_hours = {
            reg: sorted(grp["time_index"].dropna().unique().tolist())
            for reg, grp in load_curves.groupby("region", dropna=False, sort=False)
        }

    def _target_wys(row) -> List:
        """Weather-year values a row should be applied to."""
        if not has_wy_col:
            # No weather_year column: apply to every weather year in the load
            # data (or, if the load data has no weather-year structure at all,
            # to every hour of the region).
            return present_wys if has_base_wy else [None]
        wy = row["_norm_wy"]
        if wy == "all":
            return present_wys
        if not has_base_wy or wy not in present_wys:
            raise ValueError(
                f"A supplemental demand row cites weather_year {wy}, but the "
                "weather years of the load curves are not known or do not include "
                f"this year for model year {model_year}. Set `weather_year` in "
                "your settings file (the same value used to load the demand table) "
                "so weather-year-specific supplemental demand can be placed in the "
                "correct hours."
            )
        return [wy]

    supp_rows = []
    for _, row in supp.iterrows():
        base = _resolve_supp_region(
            row["region"], region_agg_map, region_aggregations, keep_regions
        )
        if base is None:
            continue

        time_idx = row["_time"]
        if time_idx == "all":
            # Expand to every hour of each selected weather-year block.
            for wy_i in _target_wys(row):
                key = (base, wy_i) if has_base_wy else base
                for h in base_hours.get(key, []):
                    supp_rows.append(
                        {
                            "region": base,
                            "weather_year": wy_i,
                            "time_index": h,
                            "load_mw": row["load_mw"],
                            "_is_supp": 1,
                        }
                    )
        else:
            # A single integer hour.
            hour = int(time_idx)
            for wy_i in _target_wys(row):
                key = (base, wy_i) if has_base_wy else base
                if hour not in base_hours.get(key, []):
                    logger.warning(
                        "Supplemental demand row for region '%s', weather year %s, "
                        "hour %d targets an hour that does not exist in the base "
                        "load data; skipping.",
                        base,
                        wy_i,
                        hour,
                    )
                    continue
                supp_rows.append(
                    {
                        "region": base,
                        "weather_year": wy_i,
                        "time_index": hour,
                        "load_mw": row["load_mw"],
                        "_is_supp": 1,
                    }
                )

    if not supp_rows:
        return load_curves

    supp_long = pd.DataFrame(supp_rows)
    if not has_base_wy:
        supp_long = supp_long.drop(columns=["weather_year"])

    return pd.concat([load_curves, supp_long], ignore_index=True)


def _renumber_load_hours(
    load_curves: pd.DataFrame,
    weather_years: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Renumber hours 1..N within each region, merging base and supplemental rows.

    The load frame may contain several weather years, each with an independent
    (possibly different-length) hour index.  This assigns a single sequential
    1..N hour number per region.  Weather-year block lengths are measured from the
    base-load rows only (tagged ``_is_supp == 0``), so a leap year (8784 hours)
    and a standard year (8760 hours) both work with no fixed ``hours_per_year``
    assumption.

    Base and supplemental rows that share ``(region, weather_year, time_index)``
    describe the SAME hour (the supplemental row augments that hour's load), so
    they receive the same renumbered hour.  The frame is explicitly ordered by
    ``(region, weather_year, time_index)`` first so the numbering is deterministic
    regardless of the order rows arrived from the database.
    """
    if weather_years is not None:
        if not isinstance(weather_years, list):
            weather_years = [_norm_weather_year(weather_years)]
        else:
            weather_years = [_norm_weather_year(w) for w in weather_years]

    load_curves = load_curves.copy()
    if "_is_supp" not in load_curves.columns:
        load_curves["_is_supp"] = 0

    # Normalize weather_year values so ints/floats/numeric strings compare equal.
    if "weather_year" in load_curves.columns:
        load_curves["weather_year"] = load_curves["weather_year"].map(
            _norm_weather_year
        )

    present_wys = sorted(
        load_curves.loc[load_curves["weather_year"].notna(), "weather_year"]
        .unique()
        .tolist()
    )
    ordered_wys = weather_years if weather_years else present_wys
    wy_rank = {_norm_weather_year(w): i for i, w in enumerate(ordered_wys)}
    stragglers = [w for w in present_wys if _norm_weather_year(w) not in wy_rank]
    for i, w in enumerate(stragglers):
        wy_rank[_norm_weather_year(w)] = len(ordered_wys) + i

    load_curves["_wy_rank"] = (
        load_curves["weather_year"]
        .map(lambda v: wy_rank.get(_norm_weather_year(v), len(wy_rank)))
        .astype(int)
    )

    # Deterministic ordering: region -> weather year -> hour -> base/supp.  Sorting
    # by hour within a weather year puts each supplemental row directly after the
    # base row for the same (region, weather_year, time_index).
    load_curves = load_curves.sort_values(
        ["region", "_wy_rank", "time_index", "_is_supp"], kind="stable"
    ).reset_index(drop=True)

    base = load_curves.loc[load_curves["_is_supp"] == 0].copy()
    # 1-based rank of each base hour within its (region, weather_year) block.
    base["_block_rank"] = (
        base.groupby(["region", "weather_year"], sort=False)["time_index"]
        .rank(method="first", ascending=True)
        .astype(int)
    )
    # Weather-year block length (number of distinct base hours) per
    # (region, weather_year) -- never a fixed hours_per_year.
    block_sizes = base.groupby(["region", "weather_year"], sort=False)[
        "time_index"
    ].nunique()
    # 0-based starting hour of each block within its region.
    block_starts = block_sizes.groupby(level=0, sort=False).cumsum() - block_sizes

    merged = load_curves.merge(
        base[["region", "weather_year", "time_index", "_block_rank"]],
        on=["region", "weather_year", "time_index"],
        how="left",
    )
    missing = merged["_block_rank"].isna()
    if missing.any():
        bad = list(
            zip(
                merged.loc[missing, "region"],
                merged.loc[missing, "weather_year"],
                merged.loc[missing, "time_index"],
            )
        )[:5]
        raise ValueError(
            "Could not map every (region, weather_year, time_index) to a base load "
            "hour while renumbering hours. Supplemental demand rows must target "
            f"hours that exist in the base load data. Problematic keys include: {bad}"
        )

    merged = merged.merge(
        block_starts.rename("_block_start").reset_index(),
        on=["region", "weather_year"],
        how="left",
    )
    merged["time_index"] = (merged["_block_start"] + merged["_block_rank"]).astype(
        "int64"
    )
    merged = merged.drop(
        columns=["_block_start", "_block_rank", "_wy_rank", "_is_supp"]
    )
    return merged


def add_supplemental_demand(
    load_curves: pd.DataFrame,
    model_year: int,
    model_regions: List[str],
    keep_regions: Optional[List[str]] = None,
    region_agg_map: Optional[Dict[str, str]] = None,
    region_aggregations: Optional[Dict[str, List[str]]] = None,
    weather_years: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Add supplemental demand to a WIDE load frame (one column per model region).

    This is a slim variant kept for the user-supplied WIDE load path in
    :func:`make_final_load_curves` (when ``load_usr_demand_profiles`` supplies the
    load and :func:`make_load_curves` is never called).  Normal pipelines apply
    supplemental demand inside :func:`make_load_curves` (via
    :func:`add_supplemental_demand_long`) in long format.

    A wide frame has no weather-year structure, so no tiling is performed and no
    fixed ``hours_per_year`` block size is assumed: ``"all"``/``"all_hours"`` rows
    add to every hour and integer ``time_index`` rows add to just that hour.
    Weather-year-specific rows are not supported here and raise a ``ValueError``.

    Parameters
    ----------
    load_curves : pd.DataFrame
        Wide dataframe with one column per model region and ``time_index`` as the
        index.
    model_year : int
        Planning year passed to :func:`_load_supplemental_demand`.
    model_regions : List[str]
        Model region names; used as a fallback for ``keep_regions`` when not given.
    keep_regions, region_agg_map, region_aggregations
        Optional region resolution inputs from :func:`powergenome.util.regions_to_keep`.
    weather_years : Optional[List[int]], optional
        Unused in this wide path (no weather-year structure); accepted for
        signature compatibility.
    """
    if "supplemental_demand" not in list_tables():
        return load_curves

    keep_regions = keep_regions if keep_regions is not None else model_regions
    region_agg_map = region_agg_map if region_agg_map is not None else {}
    region_aggregations = region_aggregations if region_aggregations is not None else {}

    supp = _load_supplemental_demand(model_year)
    if supp is None:
        return load_curves

    # The wide frame cannot represent per-weather-year hours.
    if "weather_year" in supp.columns:
        specific = supp.loc[
            ~supp["_norm_wy"].astype(str).eq("all"), "_norm_wy"
        ].dropna()
        if not specific.empty:
            years = sorted({int(s) for s in specific})
            raise ValueError(
                "The wide (user-supplied) load path does not support weather-year-"
                f"specific supplemental demand rows (found weather_year values "
                f"{years}). Use weather_year='all' to apply a row across every "
                "hour, or apply supplemental demand through the standard load "
                "pipeline (make_load_curves) instead."
            )

    load_curves = load_curves.copy()
    all_indices = load_curves.index.tolist()

    def _apply(region: str, by_time: "pd.Series") -> None:
        if region not in load_curves.columns:
            logger.debug(
                "Supplemental demand region '%s' not in model regions; skipping.",
                region,
            )
            return
        common = load_curves.index.intersection(by_time.index)
        if common.empty:
            return
        load_curves.loc[common, region] = (
            load_curves.loc[common, region] + by_time.loc[common]
        )

    for region, region_df in supp.groupby("region", sort=False):
        base = _resolve_supp_region(
            region, region_agg_map, region_aggregations, keep_regions
        )
        if base is None:
            continue
        # Map base -> model so we mutate the correct wide column.  A base region
        # resolves to its aggregated model region; a standalone model region maps
        # to itself.
        model = region_agg_map.get(base, base)

        all_mask = region_df["_time"].astype(str).str.strip().str.lower().eq("all")
        all_df = region_df[all_mask]
        if not all_df.empty:
            _apply(
                model,
                pd.Series(all_df["load_mw"].sum(), index=all_indices, dtype=float),
            )
        specific_df = region_df[~all_mask]
        if not specific_df.empty:
            by_time = specific_df.groupby("_time")["load_mw"].sum()
            by_time.index = by_time.index.astype("int64")
            _apply(model, by_time)

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

    # Region resolution inputs, used both by the electrification branch and the
    # wide-format supplemental-demand application for the user-supplied load path.
    keep_regions, region_agg_map = regions_to_keep(
        settings["model_regions"], settings.get("region_aggregations", {}) or {}
    )
    region_aggregations = settings.get("region_aggregations", {}) or {}

    # Whether the load came entirely from a user-supplied WIDE file. Only that
    # path needs the wide-format `add_supplemental_demand` below; all other paths
    # apply supplemental demand inside make_load_curves (long format).
    used_user_load = False
    if user_load_curves is not None and all(
        [r in user_load_curves.columns for r in settings["model_regions"]]
    ):
        load_curves_before_dr = user_load_curves
        used_user_load = True

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

    # Supplemental demand is normally added INSIDE make_load_curves (long format,
    # before hours are renumbered), so it is applied once per load source. With a
    # single load source (the common default) that applies it exactly once. With
    # multiple sources in `load_source_table_name` it is applied per source — if
    # your supplemental table is scoped to a specific region that only exists in
    # one source, or you keep each source's regions disjoint, this is still
    # correct; a table that targets the same hours in overlapping sources would
    # be counted once per source. See also the user-supplied WIDE load path below.
    # The user-supplied WIDE load path (load_usr_demand_profiles) bypasses
    # make_load_curves entirely, so supplemental demand is applied here on the
    # final wide frame to preserve that coverage:
    if used_user_load:
        final_load_curves = add_supplemental_demand(
            final_load_curves,
            model_year=settings["model_year"],
            model_regions=settings["model_regions"],
            keep_regions=keep_regions,
            region_agg_map=region_agg_map,
            region_aggregations=region_aggregations,
            weather_years=settings.get("weather_year"),
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
