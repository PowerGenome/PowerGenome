import collections
import logging
import operator
import os
import re
from functools import reduce
from numbers import Number
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

os.environ["USE_PYGEOS"] = "0"

import numpy as np
import pandas as pd
from flatten_dict import flatten
from scipy.stats import iqr
from sklearn import cluster, preprocessing

from powergenome.co2_pipeline_cost import merge_co2_pipeline_costs
from powergenome.database import get_data
from powergenome.eia_opendata import fetch_fuel_prices, modify_fuel_prices
from powergenome.external_data import (
    add_resource_max_cap_spur,
    demand_response_resource_capacity,
    make_demand_response_profiles,
)
from powergenome.financials import investment_cost_calculator
from powergenome.GenX import (
    add_co2_costs_to_o_m,
    add_misc_gen_values,
    cap_retire_within_period,
    check_resource_tags,
    hydro_energy_to_power,
    rename_gen_cols,
    round_col_values,
    set_int_cols,
)
from powergenome.load_profiles import make_distributed_gen_profiles
from powergenome.nrelatb import (
    atb_new_generators,
    fetch_heat_rates,
    fetch_resource_costs,
)
from powergenome.params import DATA_PATHS, build_resource_clusters
from powergenome.price_adjustment import inflation_price_adjustment
from powergenome.resource_clusters import map_eia_technology
from powergenome.util import (
    find_region_col,
    map_agg_region_names,
    regions_to_keep,
    remove_fuel_gen_scenario_name,
    reverse_dict_of_lists,
    snake_case_col,
    snake_case_str,
    sort_nested_dict,
)

logger = logging.getLogger(__name__)


planned_col_map = {
    "Entity ID": "utility_id_eia",
    "Entity Name": "utility_name",
    "Plant ID": "plant_id_eia",
    "Plant Name": "plant_name",
    "Sector": "sector_name",
    "Plant State": "state",
    "Generator ID": "generator_id",
    "Unit Code": "unit_code",
    "Nameplate Capacity (MW)": "capacity_mw",
    "Net Summer Capacity (MW)": "summer_capacity_mw",
    "Net Winter Capacity (MW)": "winter_capacity_mw",
    "Technology": "technology_description",
    "Energy Source Code": "energy_source_code_1",
    "Prime Mover Code": "prime_mover_code",
    "Planned Operation Month": "planned_operating_month",
    "Planned Operation Year": "planned_operating_year",
    "Status": "operational_status",
    "Nameplate Energy Capacity (MWh)": "capacity_mwh",
    "DC Net Capacity (MW)": "dc_net_capacity_mw",
    "County": "county",
    "Latitude": "latitude",
    "Longitude": "longitude",
    "Google Map": "google_map",
    "Bing Map": "bing_map",
    "Balancing Authority Code": "balancing_authority_code",
}

op_status_map = {
    "(V) Under construction, more than 50 percent complete": "V",
    "(TS) Construction complete, but not yet in commercial operation": "TS",
    "(U) Under construction, less than or equal to 50 percent complete": "U",
    "(T) Regulatory approvals received. Not under construction": "T",
    "(P) Planned for installation, but regulatory approvals not initiated": "P",
    "(L) Regulatory approvals pending. Not under construction": "L",
    "(OT) Other": "OT",
    "(SB) Standby/Backup: available for service but not normally used": "SB",
    "(OP) Operating": "OP",
    "(OA) Out of service but expected to return to service in next calendar year": "OA",
    "(OS) Out of service and NOT expected to return to service in next calendar year": "OS",
}

TRANSMISSION_TYPES = ["spur", "offshore_spur", "tx"]


# def fill_missing_tech_descriptions(
#     df: pd.DataFrame, date_col: str = "report_date"
# ) -> pd.DataFrame:
#     """
#     EIA 860 records before 2014 don't have a technology description. If we want to
#     include any of this data in the historical record (e.g. heat rates or capacity
#     factors) then they need to be filled in.

#     Parameters
#     ----------
#     df : dataframe
#         A pandas dataframe with columns plant_id_eia, generator_id, and
#         technology_description.
#     date_col: str
#         The column with date information, used to sort values from oldest to newest.
#         Assumes that newer records will have a valid technology description for the
#         generator.

#     Returns
#     -------
#     dataframe
#         Same data that came in, but with missing technology_description values filled
#         in.
#     """
#     if (
#         date_col not in df.columns
#         and not df.loc[df["technology_description"].isnull(), :].empty
#     ):
#         logger.warning(
#             "A dataframe with missing technology descriptions does not have the date column "
#             f"{date_col}. The rows with missing technology descriptions look like:\n\n"
#             f"{df.loc[df['technology_description'].isnull(), :]}\n\n"
#         )
#     start_len = len(df)
#     df = df.sort_values(by=date_col)
#     df_list = []
#     missing_tech_plants = df.loc[df["technology_description"].isna(), :]
#     if missing_tech_plants.empty:
#         return df

#     df = df.drop(index=missing_tech_plants.index)
#     for _, _df in missing_tech_plants.groupby(
#         ["plant_id_eia", "generator_id"], as_index=False
#     ):
#         _df["technology_description"].fillna(method="bfill", inplace=True)
#         df_list.append(_df)
#     results = pd.concat([df, pd.concat(df_list, ignore_index=True, sort=False)])

#     if df.loc[df["technology_description"].isnull(), :].empty is False:
#         logger.warning("Failed to fill some technology names.")

#     end_len = len(results)
#     assert (
#         start_len == end_len
#     ), "Somehow records were dropped when filling tech_descriptions"
#     return results


# def group_generators_at_plant(df, by=["plant_id_eia"], agg_fn={"capacity_mw": "sum"}):
#     """
#     Group generators at a plant. This is a flexible function that lets a user group
#     by the desired attributes (e.g. plant id) and perform aggregated operations on each
#     group.

#     This function also might be a bit unnecessary given how simple it is.

#     Parameters
#     ----------
#     df : dataframe
#         Pandas dataframe with information on power plants.
#     by : list, optional
#         Columns to use for the groupby, by default ["plant_id_eia"]
#     agg_fn : dict, optional
#         Aggregation function to pass to groupby, by default {"capacity_mw": "sum"}

#     Returns
#     -------
#     dataframe
#         The grouped dataframe with aggregation functions applied.
#     """

#     df_grouped = df.groupby(by, as_index=False).agg(agg_fn)

#     return df_grouped


def startup_fuel(df: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """Add startup fuel consumption for generators

    Parameters
    ----------
    df : DataFrame
        All generator clusters. Must have a column "technology". Can include both EIA
        and NRELATB technology names.
    settings : dictionary
        User-defined settings loaded from a YAML file. Keys in "startup_fuel_use"
        must match those in "eia_atb_tech_map".

    Returns
    -------
    DataFrame
        Modified dataframe with the new column "Start_Fuel_MMBTU_per_MW".
    """
    df["Start_Fuel_MMBTU_per_MW"] = 0
    for eia_tech, fuel_use in (settings.get("startup_fuel_use") or {}).items():
        if not isinstance(settings.get("eia_atb_tech_map", {}).get(eia_tech), list):
            settings["eia_atb_tech_map"][eia_tech] = [
                settings["eia_atb_tech_map"][eia_tech]
            ]

        atb_tech = settings["eia_atb_tech_map"][eia_tech]
        atb_tech.append(eia_tech)
        for tech in atb_tech:
            df.loc[df["technology"] == tech, "Start_Fuel_MMBTU_per_MW"] = fuel_use
            df.loc[
                df["technology"].str.contains(tech, case=False, regex=False),
                "Start_Fuel_MMBTU_per_MW",
            ] = fuel_use

    return df


def startup_nonfuel_costs(df: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """Add inflation adjusted startup nonfuel costs per MW for generators

    Parameters
    ----------
    df : DataFrame
        Must contain a column "technology" with the names of each technology type.
    settings : dict
        Dictionary based on YAML settings file. Must contain the keys
        "startup_costs_type", "startup_vom_costs_mw", "existing_startup_costs_tech_map",
        etc.

    Returns
    -------
    DataFrame
        Modified df with new column "Start_Cost_per_MW"
    """
    logger.debug("Adding non-fuel startup costs")
    target_usd_year = settings.get("target_usd_year")

    vom_costs = settings.get("startup_vom_costs_mw", {})
    vom_usd_year = settings.get("startup_vom_costs_usd_year")

    if target_usd_year and vom_usd_year:
        logger.debug(
            f"Changing non-fuel VOM costs from {vom_usd_year} to " f"{target_usd_year}"
        )
        for key, cost in vom_costs.items():
            vom_costs[key] = inflation_price_adjustment(
                price=cost,
                base_year=vom_usd_year,
                target_year=target_usd_year,
                data_location=settings["data_location"],
                table_name=settings["dollar_year_table"],
            )

    startup_type = settings.get("startup_costs_type")
    startup_costs = settings.get(startup_type, {})
    startup_costs_usd_year = settings.get("startup_costs_per_cold_start_usd_year")

    if target_usd_year and startup_costs_usd_year:
        logger.debug(
            f"Changing non-fuel startup costs from {vom_usd_year} to {target_usd_year}"
        )
        for key, cost in startup_costs.items():
            startup_costs[key] = inflation_price_adjustment(
                price=cost,
                base_year=startup_costs_usd_year,
                target_year=target_usd_year,
                data_location=settings["data_location"],
                table_name=settings["dollar_year_table"],
            )

    df["Start_Cost_per_MW"] = 0

    for existing_tech, cost_tech in settings.get(
        "existing_startup_costs_tech_map", {}
    ).items():
        total_startup_costs = vom_costs[cost_tech] + startup_costs[cost_tech]
        df.loc[
            df["technology"].str.contains(existing_tech, case=False, regex=False),
            "Start_Cost_per_MW",
        ] = total_startup_costs

    for new_tech, cost_tech in settings.get("new_build_startup_costs", {}).items():
        total_startup_costs = vom_costs[cost_tech] + startup_costs[cost_tech]
        df.loc[df["technology"].str.contains(new_tech), "Start_Cost_per_MW"] = (
            total_startup_costs
        )
    df.loc[:, "Start_Cost_per_MW"] = df.loc[:, "Start_Cost_per_MW"]

    # df.loc[df["technology"].str.contains("Nuclear"), "Start_Cost_per_MW"] = "FILL VALUE"

    return df


def group_technologies(
    df: pd.DataFrame,
    tech_groups: Dict[str, list] = None,
    regional_no_grouping: Dict[str, list] = None,
) -> pd.DataFrame:
    """Group different technologies together based on parameters in the settings file.
    An example would be to put a bunch of different technologies under the umbrella
    category of "biomass" or "peaker".

    Parameters
    ----------
    df : pd.DataFrame
        Each row represents one resource with a technology described in "technology_description"
    tech_groups : Dict[str, list], optional
        A mapping of existing technology names to group names, by default None
    regional_no_grouping : Dict[str, list], optional
        For each model region listed, the aggregated technologies to ignore, by default None

    Returns
    -------
    pd.DataFrame
        Same as incoming dataframe but with grouped technology types
    """
    if tech_groups is None:
        return df
    df["_technology"] = df["technology"]
    for tech, group in tech_groups.items():
        df.loc[df["technology"].isin(group), "_technology"] = tech

        for region, tech_list in (regional_no_grouping or {}).items():
            df.loc[
                (df["model_region"] == region)
                & (df["technology_description"].isin(tech_list)),
                "_technology",
            ] = df.loc[
                (df["model_region"] == region)
                & (df["technology_description"].isin(tech_list)),
                "technology_description",
            ]

    df.loc[:, "technology"] = df.loc[:, "_technology"]
    df = df.drop(columns=["_technology"])

    return df


# def label_hydro_region(gens_860, pudl_engine, model_regions_gdf):
#     """
#     Label hydro facilities that don't have a region by default.

#     Parameters
#     ----------
#     gens_860 : dataframe
#         Infomation on all generators from PUDL
#     pudl_engine : sqlalchemy.Engine
#         A sqlalchemy connection for use by pandas
#     model_regions_gdf : dataframe
#         Geodataframe of the model regions

#     Returns
#     -------
#     dataframe
#         Plant id and region for any hydro that didn't originally have a region label.
#     """

#     plant_entity = pd.read_sql_table("plants_entity_eia", pudl_engine)

#     model_hydro = gens_860.loc[
#         gens_860["technology_description"] == "Conventional Hydroelectric"
#     ].merge(plant_entity[["plant_id_eia", "latitude", "longitude"]], on="plant_id_eia")

#     no_lat_lon = model_hydro.loc[
#         (model_hydro["latitude"].isnull()) | (model_hydro["longitude"].isnull()), :
#     ]
#     if not no_lat_lon.empty:
#         logger.debug(
#             f"{no_lat_lon['summer_capacity_mw'].sum().round(1)}, MW without lat/lon"
#         )
#     model_hydro = model_hydro.dropna(subset=["latitude", "longitude"])

#     # Convert the lon/lat values to geo points. Need to add an initial CRS and then
#     # change it to align with the IPM regions
#     model_hydro_gdf = gpd.GeoDataFrame(
#         model_hydro,
#         geometry=gpd.points_from_xy(model_hydro.longitude, model_hydro.latitude),
#         crs="EPSG:4326",
#     )

#     if model_hydro_gdf.crs != model_regions_gdf.crs:
#         model_hydro_gdf = model_hydro_gdf.to_crs(model_regions_gdf.crs)

#     model_hydro_gdf = gpd.sjoin(model_regions_gdf, model_hydro_gdf)

#     keep_cols = ["plant_id_eia", "region"]
#     return model_hydro_gdf.loc[:, keep_cols]


# def load_plant_region_map(
#     gens_860,
#     pudl_engine,
#     pg_engine,
#     settings,
#     model_regions_gdf,
#     table="plant_region_map_epaipm",
# ):
#     """
#     Load the region that each plant is located in.

#     Parameters
#     ----------
#     pudl_engine : sqlalchemy.Engine
#         A sqlalchemy connection for use by pandas
#     settings : dictionary
#         The dictionary of settings with a dictionary of region aggregations
#     table : str, optional
#         The SQL table to load, by default "plant_region_map_epaipm"

#     Returns
#     -------
#     dataframe
#         A dataframe where each plant has an associated "model_region" mapped
#         from the original region labels.
#     """
#     # Load dataframe of region labels for each EIA plant id
#     region_map_df = pd.read_sql_table(table, con=pg_engine)

#     if settings.get("plant_region_map_fn"):
#         user_region_map_df = pd.read_csv(
#             Path(settings["input_folder"]) / settings["plant_region_map_fn"]
#         )
#         assert (
#             "region" in user_region_map_df.columns
#         ), f"The column 'region' must appear in {settings['plant_region_map_fn']}"
#         assert (
#             "plant_id_eia" in user_region_map_df.columns
#         ), f"The column 'plant_id_eia' must appear in {settings['plant_region_map_fn']}"

#         user_region_map_df = user_region_map_df.set_index("plant_id_eia")

#         region_map_df.loc[
#             region_map_df["plant_id_eia"].isin(user_region_map_df.index), "region"
#         ] = region_map_df["plant_id_eia"].map(user_region_map_df["region"])

#     # Label hydro using the IPM shapefile because NEEDS seems to drop some hydro
#     all_hydro_regions = label_hydro_region(gens_860, pudl_engine, model_regions_gdf)

#     region_map_df = pd.concat(
#         [region_map_df, all_hydro_regions], ignore_index=True, sort=False
#     ).drop_duplicates(subset=["plant_id_eia"], keep="first")

#     # Settings has a dictionary of lists for regional aggregations. Need
#     # to reverse this to use in a map method.
#     keep_regions, region_agg_map = regions_to_keep(
#         settings["model_regions"], settings.get("region_aggregations")
#     )

#     # Create a new column "model_region" with labels that we're using for aggregated regions

#     model_region_map_df = region_map_df.loc[
#         region_map_df.region.isin(keep_regions), :
#     ].drop(columns="id", errors="ignore")

#     model_region_map_df = map_agg_region_names(
#         df=model_region_map_df,
#         region_agg_map=region_agg_map,
#         original_col_name="region",
#         new_col_name="model_region",
#     )

#     # There are some cases of plants with generators assigned to different IPM regions.
#     # If regions are aggregated there may be some duplicates in the results.
#     model_region_map_df = model_region_map_df.drop_duplicates(
#         subset=["plant_id_eia", "model_region"]
#     )

#     return model_region_map_df


# def label_retirement_year(
#     df: pd.DataFrame,
#     model_year: int,
#     capacity_col: str = "capacity_mw",
#     retirement_ages: Dict[str, int] = None,
#     additional_retirements: List[Tuple[str, str, int]] = None,
#     age_col: str = "operating_date",
# ):
#     """
#     Add a retirement year column to the dataframe based on the year each generator
#     started operating.

#     Parameters
#     ----------
#     df : dataframe
#         Dataframe of generators
#     model_year : int
#         The model year, used to check how much capacity will have retired
#     capacity_col : str, optional
#         The dataframe column to use when calculating unit capacity, by default
#         "capacity_mw"
#     age_col : str, optional
#         The dataframe column to use when calculating the retirement year, by default
#         "operating_date"
#     retirement_ages : Dict[str, int], optional
#         The age at which different technologies will retire, by default None. If no
#         values are given, technology retirement ages are set to 500 years.
#     additional_retirements : List[Tuple[str, str, int]], optional
#         A list of tuples with plant, generator, and a new retirement year to use, by
#         default None.
#     """
#     if age_col not in df.columns:
#         age_col = age_col.replace("operating_date", "generator_operating_date")
#     if age_col not in df.columns:
#         return df
#     start_len = len(df)
#     retirement_ages = retirement_ages or {}
#     if "retirement_year" not in df.columns:
#         df["retirement_year"] = np.nan

#     df["retirement_age"] = df["technology_description"].map(retirement_ages).fillna(500)
#     try:
#         df.loc[df["retirement_year"].isna(), "retirement_year"] = (
#             df.loc[df["retirement_year"].isna(), age_col].dt.year
#             + df.loc[df["retirement_year"].isna(), "retirement_age"]
#         )
#     except AttributeError:
#         df.loc[df["retirement_year"].isna(), "retirement_year"] = (
#             df.loc[df["retirement_year"].isna(), age_col]
#             + df.loc[df["retirement_year"].isna(), "retirement_age"]
#         )

#     try:
#         df.loc[~df["planned_retirement_date"].isnull(), "retirement_year"] = df.loc[
#             ~df["planned_retirement_date"].isnull(), "planned_retirement_date"
#         ].dt.year
#     except KeyError:
#         pass

#     # Add additonal retirements from settings file
#     if additional_retirements:
#         logger.debug("Changing retirement dates based on settings file")
#         start_ret_cap = df.loc[df["retirement_year"] <= model_year, capacity_col].sum()
#         logger.debug(f"Starting retirement capacity is {start_ret_cap} MW")
#         i = 0
#         ret_cap = 0
#         for record in additional_retirements:
#             plant_id, gen_id, ret_year = record
#             # gen ids are strings, not integers
#             gen_id = str(gen_id)

#             df.loc[
#                 (df["plant_id_eia"] == plant_id) & (df["generator_id"] == gen_id),
#                 "retirement_year",
#             ] = ret_year

#             i += 1
#             ret_cap += df.loc[
#                 (df["plant_id_eia"] == plant_id) & (df["generator_id"] == gen_id),
#                 capacity_col,
#             ].sum()

#         end_ret_cap = df.loc[df["retirement_year"] <= model_year, capacity_col].sum()
#         logger.debug(f"Ending retirement capacity is {end_ret_cap} MW")
#         if not end_ret_cap > start_ret_cap:
#             logger.debug(
#                 "Adding retirements from settings didn't change the retiring capacity."
#             )
#         if end_ret_cap - start_ret_cap != ret_cap:
#             logger.debug(
#                 f"Retirement diff is {end_ret_cap - start_ret_cap}, adding retirements "
#                 f"yields {ret_cap} MW"
#             )
#         logger.debug(
#             f"The retirement year for {i} plants, totaling {ret_cap} MW, was changed "
#             "based on settings file parameters"
#         )
#     else:
#         logger.debug("No retirement dates changed based on the settings file")

#     end_len = len(df)

#     assert start_len == end_len

#     return df


def label_small_hydro(df, settings, by=["plant_id_eia"]):
    """
    Use rules from the settings file to label plants below a certain size as small
    hydroelectric rather than conventional hydroelectric.

    Parameters
    ----------
    df : dataframe
        EIA 860 data on generators
    settings : dict
        User-defined parameters from a settings file
    by : list, optional
        What columns to use in the groupby function when summing capacity, by default
        ["plant_id_eia"]

    Returns
    -------
    dataframe
        If the user wants to label small hydro plants, some of the conventional
        hydro facilities will have their technology type changed to small hydro.
    """
    if not settings.get("small_hydro"):
        return df
    if "report_date" not in by and "report_date" in df.columns:
        # by.append("report_date")
        logger.warning("'report_date' is in the df but not used in the groupby")
    region_agg_map = reverse_dict_of_lists(settings.get("region_aggregations", {}))
    keep_regions = [
        x
        for x in settings["model_regions"] + list(region_agg_map)
        if x in settings["small_hydro_regions"]
    ]
    start_len = len(df)
    size_cap = settings["small_hydro_mw"]
    cap_col = settings.get("capacity_col")
    if cap_col not in df:
        cap_col = "capacity_mw"

    start_hydro_capacity = df.query(
        "technology_description=='Conventional Hydroelectric'"
    )[cap_col].sum()

    plant_capacity = (
        df.loc[
            (df["technology_description"] == "Conventional Hydroelectric")
            & (df["model_region"].isin(keep_regions))
        ]
        .groupby(by, as_index=False)[cap_col]
        .sum()
    )

    small_hydro_plants = plant_capacity.loc[
        plant_capacity[cap_col] <= size_cap, "plant_id_eia"
    ]

    df.loc[
        (df["technology_description"] == "Conventional Hydroelectric")
        & (df["plant_id_eia"].isin(small_hydro_plants)),
        "technology_description",
    ] = "Small Hydroelectric"

    end_len = len(df)
    small_hydro_capacity = df.query("technology_description=='Small Hydroelectric'")[
        cap_col
    ].sum()
    end_conv_hydro_capacity = df.query(
        "technology_description=='Conventional Hydroelectric'"
    )[cap_col].sum()

    assert start_len == end_len
    assert np.allclose(
        start_hydro_capacity, small_hydro_capacity + end_conv_hydro_capacity
    )

    return df


# def load_generator_860_data(pudl_engine, data_years=[2017]):
#     """
#     Load EIA 860 generator data from the PUDL database

#     Parameters
#     ----------
#     pudl_engine : sqlalchemy.Engine
#         A sqlalchemy connection for use by pandas
#     data_years : list, optional
#         Years of data to load, by default [2017]

#     Returns
#     -------
#     dataframe
#         All of the generating units from PUDL
#     """
#     data_years = [str(y) for y in data_years]
#     sql = f"""
#         SELECT * FROM generators_eia860
#         WHERE operational_status_code NOT IN ('RE', 'OS', 'IP', 'CN')
#         AND strftime('%Y',report_date) in ({','.join(['?']*len(data_years))})
#     """
#     gens_860 = pd.read_sql_query(
#         sql=sql,
#         con=pudl_engine,
#         params=data_years,
#         parse_dates=["report_date", "planned_retirement_date"],
#     )

#     return gens_860


# def supplement_generator_860_data(
#     gens_860: pd.DataFrame,
#     gens_entity: pd.DataFrame,
#     bga: pd.DataFrame,
#     model_region_map: pd.DataFrame,
#     settings: dict,
# ):
#     """
#     Load data about each generating unit in the model area.

#     Parameters
#     ----------
#     gens_860 : dataframe
#         Information on all generating units for the given data years.
#     pudl_engine : sqlalchemy.Engine
#         A sqlalchemy connection for use by pandas
#     settings : dictionary
#         The dictionary of settings with a dictionary of region aggregations
#     pudl_out : pudl.PudlTabl
#         A PudlTabl object for loading pre-calculated PUDL analysis data
#     model_region_map : dataframe
#         A dataframe with columns 'plant_id_eia' and 'model_region' (aggregated regions)
#     data_years : list, optional
#         Years of data to include, by default [2017]

#     Returns
#     -------
#     dataframe
#         Data about each generator and generation unit that will be included in the
#         model. Columns include:

#         ['plant_id_eia', 'generator_id',
#        'capacity_mw', 'energy_source_code_1',
#        'energy_source_code_2', 'minimum_load_mw', 'operational_status_code',
#        'planned_new_capacity_mw', 'switch_oil_gas', 'technology_description',
#        'time_cold_shutdown_full_load_code', 'model_region', 'prime_mover_code',
#        'operating_date', 'boiler_id', 'unit_id_eia', 'unit_id_pudl', 'unit_id_pg,
#        'retirement_year']
#     """

#     initial_capacity = (
#         gens_860.loc[gens_860["plant_id_eia"].isin(model_region_map["plant_id_eia"])]
#         .groupby("technology_description")[settings["capacity_col"]]
#         .sum()
#     )

#     # Add pudl unit ids, only include specified data years

#     # Combine generator data that can change over time with static entity data
#     # and only keep generators that are in a region of interest

#     gen_cols = set(
#         [
#             # "report_date",
#             "plant_id_eia",
#             # "plant_name",
#             "generator_id",
#             # "balancing_authority_code",
#             settings["capacity_col"],
#             "capacity_mw",
#             "energy_source_code_1",
#             "energy_source_code_2",
#             "minimum_load_mw",
#             "operational_status_code",
#             "planned_new_capacity_mw",
#             "switch_oil_gas",
#             "technology_description",
#             "time_cold_shutdown_full_load_code",
#             "planned_retirement_date",
#             "prime_mover_code",
#         ]
#     )
#     gen_cols = [c for c in gen_cols if c in gens_860]

#     entity_cols = [
#         "plant_id_eia",
#         "generator_id",
#         "prime_mover_code",
#         "operating_date",
#         "generator_operating_date",
#         "original_planned_operating_date",
#         "original_planned_generator_operating_date",
#     ]
#     entity_cols = [c for c in entity_cols if c in gens_entity]

#     bga_cols = [
#         "plant_id_eia",
#         "generator_id",
#         "boiler_id",
#         "unit_id_eia",
#         "unit_id_pudl",
#     ]

#     # In this merge of the three dataframes we're trying to label each generator with
#     # the model region it is part of, the prime mover and operating date, and the
#     # PUDL unit codes (where they exist).

#     # drop duplicate rows of model_region_map
#     mapping = model_region_map.drop(columns="region").drop_duplicates()
#     # drop duplicate mappings for the same ID
#     mapping = mapping.loc[~mapping.plant_id_eia.duplicated(), :]

#     gens_860_model = (
#         pd.merge(
#             gens_860[gen_cols],
#             mapping,
#             on="plant_id_eia",
#             how="inner",
#         )
#         .merge(
#             gens_entity[entity_cols], on=["plant_id_eia", "generator_id"], how="inner"
#         )
#         .merge(bga[bga_cols], on=["plant_id_eia", "generator_id"], how="left")
#     )
#     gens_860_model["unit_id_pg"] = gens_860_model.loc[:, "unit_id_pudl"]
#     gens_860_model.loc[gens_860_model.unit_id_pg.isnull(), "unit_id_pg"] = (
#         gens_860_model.loc[gens_860_model.unit_id_pg.isnull(), "plant_id_eia"].astype(
#             str
#         )
#         + "_"
#         + gens_860_model.loc[gens_860_model.unit_id_pg.isnull(), "generator_id"].astype(
#             str
#         )
#     ).to_numpy()

#     # Where summer/winter capacity values are missing set equal to nameplate capacity,
#     # but only if all generators within a unit are missing the capacity value
#     check_units = gens_860_model.loc[
#         gens_860_model[settings["capacity_col"]].isna()
#     ].groupby(["plant_id_eia", "unit_id_pg"])
#     for (plant_id, unit_id), _df in check_units:
#         if _df[settings["capacity_col"]].isna().all():
#             gens_860_model.loc[
#                 (gens_860_model["plant_id_eia"] == plant_id)
#                 & (gens_860_model["unit_id_pg"] == unit_id),
#                 settings["capacity_col"],
#             ] = gens_860_model.loc[
#                 (gens_860_model["plant_id_eia"] == plant_id)
#                 & (gens_860_model["unit_id_pg"] == unit_id),
#                 "capacity_mw",
#             ]

#     merged_capacity = gens_860_model.groupby("technology_description")[
#         settings["capacity_col"]
#     ].sum()
#     if not np.allclose(initial_capacity.sum(), merged_capacity.sum()):
#         for i_idx, i_row in initial_capacity.iteritems():
#             if abs(i_row - merged_capacity[i_idx]) / i_row > 0.05:
#                 logger.info(
#                     "When adding plant entity/boiler info to generators and filling missing"
#                     " seasonal capacity values, "
#                     f"{i_idx} changed capacity from {i_row} to {merged_capacity[i_idx]}"
#                 )
#             if not np.allclose(i_row, merged_capacity[i_idx]):
#                 logger.debug(
#                     "When adding plant entity/boiler info to generators and filling missing"
#                     " seasonal capacity values, "
#                     f"{i_idx} changed capacity from {i_row} to {merged_capacity[i_idx]}"
#                 )

#     return gens_860_model


# def create_plant_gen_id(df):
#     """Combine the plant id and generator id to form a unique combination

#     Parameters
#     ----------
#     df : dataframe
#         Must contain columns plant_id_eia and generator_id

#     Returns
#     -------
#     dataframe
#         Same as input but with the additional column plant_gen_id
#     """

#     df["plant_gen_id"] = (
#         df["plant_id_eia"].astype(str) + "_" + df["generator_id"].astype(str)
#     )

#     return df


# def remove_canceled_860m(df, canceled_860m):
#     """Remove generators that 860m shows as having been canceled

#     Parameters
#     ----------
#     df : dataframe
#         All of the EIA 860 generators
#     canceled_860m : dataframe
#         From the 860m Canceled or Postponed sheet

#     Returns
#     -------
#     dataframe
#         Same as input, but possibly without generators that were proposed
#     """
#     df = create_plant_gen_id(df)
#     canceled_860m = create_plant_gen_id(canceled_860m)

#     canceled = df.loc[df["plant_gen_id"].isin(canceled_860m["plant_gen_id"]), :]

#     not_canceled_df = df.loc[~df["plant_gen_id"].isin(canceled_860m["plant_gen_id"]), :]

#     not_canceled_df = not_canceled_df.drop(columns="plant_gen_id")

#     if not canceled.empty:
#         assert len(df) == len(canceled) + len(not_canceled_df)

#     return not_canceled_df.reset_index(drop=True)


# def remove_retired_860m(df, retired_860m):
#     """Remove generators that 860m shows as having been retired

#     Parameters
#     ----------
#     df : dataframe
#         All of the EIA 860 generators
#     retired_860m : dataframe
#         From the 860m Retired sheet

#     Returns
#     -------
#     dataframe
#         Same as input, but possibly without generators that have retired
#     """

#     df = create_plant_gen_id(df)
#     retired_860m = create_plant_gen_id(retired_860m)

#     retired = df.loc[df["plant_gen_id"].isin(retired_860m["plant_gen_id"]), :]

#     not_retired_df = df.loc[~df["plant_gen_id"].isin(retired_860m["plant_gen_id"]), :]

#     not_retired_df = not_retired_df.drop(columns="plant_gen_id")

#     if not retired.empty:
#         assert len(df) == len(retired) + len(not_retired_df)

#     return not_retired_df.reset_index(drop=True)


# def update_planned_retirement_date_860m(
#     df: pd.DataFrame, operating_860m: pd.DataFrame
# ) -> pd.DataFrame:
#     """Update the planned retirement date in the main dataframe using the planned
#     retirement year column from 860m existing generators.

#     Parameters
#     ----------
#     df : pd.DataFrame
#         Main dataframe of generators from EIA 860. Must have columns "plant_id_eia",
#         "generator_id", and "planned_retirement_date", which should be in a datatime format.
#     operating_860m : pd.DataFrame
#         Dataframe of operating generators from EIA 860m. Must have columns "plant_id_eia",
#         "generator_id", and "planned_retirement_year"

#     Returns
#     -------
#     pd.DataFrame
#         Modified version of "df" dataframe, with updated values in column
#         "planned_retirement_date" based on EIA-860m.
#     """
#     if "planned_retirement_date" not in df.columns:
#         logger.warning(
#             "The main generators dataframe from EIA 860 does not have a column "
#             "'planned_retirement_date'. If this column is missing all retirement dates "
#             "will be based on plant age."
#         )
#         return df
#     if "planned_retirement_year" not in operating_860m.columns:
#         logger.warning(
#             "The EIA-860m existing dataframe does not have a column 'planned_retirement_year'."
#             "Check the 860m file to see if it has been renamed. No values from the original "
#             "860 data will be changed."
#         )
#         return df
#     if df.empty or operating_860m.empty:
#         return df
#     _df = df.set_index(["plant_id_eia", "generator_id"])
#     _operating_860m = operating_860m.set_index(["plant_id_eia", "generator_id"])
#     _operating_860m["planned_retirement_date_860m"] = pd.to_datetime(
#         _operating_860m["planned_retirement_year"], format="%Y"
#     )
#     update_df = pd.merge(
#         _df,
#         _operating_860m[["planned_retirement_date_860m"]],
#         how="inner",
#         left_index=True,
#         right_index=True,
#     )
#     mask = update_df.loc[
#         update_df["planned_retirement_date"].dt.year.fillna(9999)
#         != update_df["planned_retirement_date_860m"].dt.year.fillna(9999),
#         :,
#     ].index
#     _df.loc[mask, "planned_retirement_date"] = update_df.loc[
#         mask, "planned_retirement_date_860m"
#     ]

#     return _df.reset_index()


# def remove_future_retirements_860m(df, retired_860m):
#     """Remove generators that 860m shows as having been retired

#     Parameters
#     ----------
#     df : dataframe
#         All of the EIA 860 generators
#     retired_860m : dataframe
#         From the 860m Retired sheet

#     Returns
#     -------
#     dataframe
#         Same as input, but possibly without generators that have retired
#     """

#     df = create_plant_gen_id(df)
#     retired_860m = create_plant_gen_id(retired_860m)

#     retired = df.loc[df["plant_gen_id"].isin(retired_860m["plant_gen_id"]), :]

#     not_retired_df = df.loc[~df["plant_gen_id"].isin(retired_860m["plant_gen_id"]), :]

#     not_retired_df = not_retired_df.drop(columns="plant_gen_id")

#     if not retired.empty:
#         assert len(df) == len(retired) + len(not_retired_df)

#     return not_retired_df


# def update_operating_date_860m(
#     df: pd.DataFrame, operating_860m: pd.DataFrame
# ) -> pd.DataFrame:
#     """Update the operating date of EIA generators using data from 860m.

#     When the "operating_date" of a generator is nan, fill with operating year data
#     from 860m.

#     Parameters
#     ----------
#     df : pd.DataFrame
#         Data on existing EIA generating units. Must have columns "plant_id_eia",
#         "generator_id", and "operating_date".
#     operating_860m : pd.DataFrame
#         A dataframe of operating generating units from EIA 860m. Must have columns
#         "plant_id_eia", "generator_id", and "operating_year".

#     Returns
#     -------
#     pd.DataFrame
#         The original "df" dataframe with missing operating dates filled using the operating
#         year from 860m.
#     """
#     _df = df.set_index(["plant_id_eia", "generator_id"])
#     _operating_860m = operating_860m.set_index(["plant_id_eia", "generator_id"])
#     no_op_date = _df.loc[_df["operating_date"].isna(), :]
#     no_op_date = pd.merge(
#         no_op_date,
#         _operating_860m["operating_year"],
#         left_index=True,
#         right_index=True,
#         how="left",
#         validate="1:1",
#     )

#     _df.loc[no_op_date.index, "operating_date"] = pd.to_datetime(
#         no_op_date.loc[no_op_date["operating_year"].notna(), "operating_year"],
#         format="%Y",
#     )

#     return _df.reset_index()


# def load_923_gen_fuel_data(pudl_engine, pudl_out, model_region_map, data_years=[2017]):
#     """
#     Load generation and fuel data for each plant. EIA-923 provides these values for
#     each prime mover/fuel combination at every generator. This data can be used to
#     calculate the heat rate of generators at a single plant. Generators sharing a prime
#     mover (e.g. multiple combustion turbines) will end up sharing the same heat rate.

#     Parameters
#     ----------
#     pudl_engine : sqlalchemy.Engine
#         A sqlalchemy connection for use by pandas
#     pudl_out : pudl.PudlTabl
#         A PudlTabl object for loading pre-calculated PUDL analysis data
#     model_region_map : dataframe
#         A dataframe with columns 'plant_id_eia' and 'model_region' (aggregated regions)
#     data_years : list, optional
#         Years of data to include, by default [2017]

#     Returns
#     -------
#     dataframe
#         Generation, fuel use, and heat rates of prime mover/fuel combos over all data
#         years. Columns are:

#         ['plant_id_eia', 'fuel_type', 'fuel_type_code_pudl',
#        'fuel_type_code_aer', 'prime_mover_code', 'fuel_consumed_units',
#        'fuel_consumed_for_electricity_units', 'fuel_consumed_mmbtu',
#        'fuel_consumed_for_electricity_mmbtu', 'net_generation_mwh',
#        'heat_rate_mmbtu_mwh']
#     """
#     if isinstance(data_years, (int, float)):
#         data_years = [str(data_years)]
#     data_years = [str(y) for y in data_years]

#     # Load 923 generation and fuel data for one or more years.
#     # Only load plants in the model regions.
#     sql = f"""
#         SELECT * FROM generation_fuel_eia923
#         WHERE strftime('%Y',report_date) in ({','.join(['?']*len(data_years))})
#     """
#     gen_fuel_923 = pd.read_sql_query(
#         sql, pudl_engine, params=data_years, parse_dates=["report_date"]
#     )
#     gen_fuel_923 = gen_fuel_923.loc[
#         gen_fuel_923["plant_id_eia"].isin(model_region_map.plant_id_eia),
#         :,
#     ]

#     insp = sqlalchemy.inspect(pudl_engine)
#     if insp.has_table("generation_fuel_nuclear_eia923"):
#         sql = f"""
#             SELECT * FROM generation_fuel_nuclear_eia923
#             WHERE strftime('%Y',report_date) in ({','.join(['?']*len(data_years))})
#         """
#         gen_fuel_nuclear_923 = pd.read_sql_query(
#             sql, pudl_engine, params=data_years, parse_dates=["report_date"]
#         )
#         gen_fuel_nuclear_923 = gen_fuel_nuclear_923.loc[
#             gen_fuel_nuclear_923["plant_id_eia"].isin(model_region_map.plant_id_eia),
#             :,
#         ]

#         gen_fuel_923 = pd.concat(
#             [gen_fuel_923, gen_fuel_nuclear_923], ignore_index=True
#         )

#     return gen_fuel_923


# def modify_cc_prime_mover_code(df, gens_860):
#     """Change combined cycle prime movers from CA and CT to CC.

#     The heat rate of combined cycle plants that aren't included in PUDL heat rate by
#     unit should probably be done with the combustion and steam turbines combined. This
#     modifies the prime mover code of those two generator types so that they match. It
#     doesn't touch the CS code, which is for single shaft combined units.

#     Parameters
#     ----------
#     df : dataframe
#         A dataframe with columns prime_mover_code, and plant_id_eia.
#     gens_860 : dataframe
#         EIA860 dataframe with technology_description, unit_id_pudl, plant_id_eia
#         columns.

#     Returns
#     -------
#     dataframe
#         Modified 923 dataframe where prime mover codes at CC generators that don't have
#         a PUDL unit id are modified from CA and CT to CC.
#     """
#     df.loc[
#         (df["prime_mover_code"].isin(["CA", "CT"])),
#         "prime_mover_code",
#     ] = "CC"

#     return df


# def group_gen_by_year_fuel_primemover(df):
#     """
#     Group generation and fuel consumption by plant, prime mover, and fuel type. Only
#     matters where multiple years of data are used, otherwise output should be the same
#     as input.

#     Parameters
#     ----------
#     df : dataframe
#         Generation and fuel consumption data from EIA 923 for each plant, prime mover,
#         and fuel type

#     Returns
#     -------
#     dataframe
#         Sum of generation and fuel consumption data (if multiple years).
#     """

#     # Group the data by plant, fuel type, and prime mover
#     by = [
#         "plant_id_eia",
#         "fuel_type",
#         "energy_source_code",
#         "fuel_type_code_pudl",
#         "fuel_type_code_aer",
#         "prime_mover_code",
#     ]
#     by = [c for c in by if c in df.columns]
#     sort = ["plant_id_eia", "fuel_type", "energy_source_code", "prime_mover_code"]
#     sort = [c for c in sort if c in df.columns]

#     annual_gen_fuel_923 = (
#         (
#             df.groupby(  # .drop(columns=["id", "nuclear_unit_id"])
#                 by=by, as_index=False
#             )[
#                 [
#                     "fuel_consumed_units",
#                     "fuel_consumed_for_electricity_units",
#                     "fuel_consumed_mmbtu",
#                     "fuel_consumed_for_electricity_mmbtu",
#                     "net_generation_mwh",
#                 ]
#             ].sum()
#         )
#         .reset_index()
#         .drop(columns="index")
#         .sort_values(sort)
#     )

#     return annual_gen_fuel_923


# def add_923_heat_rate(df):
#     """
#     Small function to calculate the heat rate of records with fuel consumption and net
#     generation.

#     Parameters
#     ----------
#     df : dataframe
#         Must contain the columns net_generation_mwh and
#         fuel_consumed_for_electricity_mmbtu

#     Returns
#     -------
#     dataframe
#         Same dataframe with new column of heat_rate_mmbtu_mwh
#     """

#     # Calculate the heat rate for each prime mover/fuel combination
#     df["heat_rate_mmbtu_mwh"] = (
#         df["fuel_consumed_for_electricity_mmbtu"] / df["net_generation_mwh"]
#     )

#     return df


# def calculate_weighted_heat_rate(heat_rate_df):
#     """
#     Calculate the weighed heat rate when multiple years of data are used. Net generation
#     in each year is used as the weights.

#     Parameters
#     ----------
#     heat_rate_df : dataframe
#         Currently the PudlTabl unit_hr method.

#     Returns
#     -------
#     dataframe
#         Heat rate weighted by annual generation for each plant and PUDL unit
#     """

#     def w_hr(df):
#         weighted_hr = np.average(
#             df["heat_rate_mmbtu_mwh"], weights=df["net_generation_mwh"]
#         )
#         return weighted_hr

#     weighted_unit_hr = heat_rate_df.groupby(["plant_id_eia", "unit_id_pudl"]).apply(
#         w_hr
#     )
#     weighted_unit_hr.name = "heat_rate_mmbtu_mwh"
#     weighted_unit_hr = weighted_unit_hr.reset_index()

#     return weighted_unit_hr


# def plant_pm_heat_rates(annual_gen_fuel_923):
#     """
#     Calculate the heat rate by plant, prime mover, and fuel type. Values are saved
#     as a dictionary.

#     Parameters
#     ----------
#     annual_gen_fuel_923 : dataframe
#         Data from the 923 generation and fuel use table. Heat rate for each row should
#         already be calculated.

#     Returns
#     -------
#     dict
#         Keys are a tuple of plant id, prime mover, and fuel type. Values are the heat
#         rate.
#     """

#     by = ["plant_id_eia", "prime_mover_code", "fuel_type", "energy_source_code"]
#     by = [c for c in by if c in annual_gen_fuel_923.columns]
#     annual_gen_fuel_923_groups = annual_gen_fuel_923.groupby(by)

#     prime_mover_hr_map = {
#         _: df["heat_rate_mmbtu_mwh"].values[0] for _, df in annual_gen_fuel_923_groups
#     }

#     return prime_mover_hr_map


# def unit_generator_heat_rates(pudl_out, data_years):
#     """
#     Calculate the heat rate for each PUDL unit and generators that don't have a PUDL
#     unit id.

#     Parameters
#     ----------
#     pudl_out : pudl.PudlTabl
#         A PudlTabl object for loading pre-calculated PUDL analysis data
#     data_years : list
#         Years of data to use

#     Returns
#     -------
#     dataframe, dict
#         A dataframe of heat rates for each pudl unit (columsn are ['plant_id_eia',
#         'unit_id_pg', 'heat_rate_mmbtu_mwh']).
#     """

#     # Load the pre-calculated PUDL unit heat rates for selected years.
#     # Remove rows without generation or with null values.
#     unit_hr = pudl_out.hr_by_unit()
#     unit_hr = unit_hr.loc[
#         (unit_hr.report_date.dt.year.isin(data_years))
#         & (unit_hr.net_generation_mwh > 0),
#         :,
#     ].dropna()

#     weighted_unit_hr = calculate_weighted_heat_rate(unit_hr)

#     return weighted_unit_hr


# def group_units(df, settings):
#     """
#     Group by units within a region/technology/cluster. Add a unique unit code
#     (plant plus generator) for any generators that aren't part of a unit.


#     Returns
#     -------
#     dataframe
#         Grouped generators with the total capacity, minimum load, and average heat
#         rate for each.
#     """

#     by = ["plant_id_eia", "unit_id_pg"]
#     # add a unit code (plant plus generator code) in cases where one doesn't exist
#     df_copy = df.reset_index()

#     # All units should have the same heat rate so taking the mean will just keep the
#     # same value.
#     grouped_units = df_copy.groupby(by).agg(
#         {
#             settings["capacity_col"]: "sum",
#             "capacity_mwh": "sum",
#             "minimum_load_mw": "sum",
#             "heat_rate_mmbtu_mwh": "mean",
#             "Fixed_OM_Cost_per_MWyr": "mean",
#             "Var_OM_Cost_per_MWh": "mean",
#         }
#     )
#     grouped_units = grouped_units.replace([np.inf, -np.inf], np.nan)
#     grouped_units = grouped_units.fillna(grouped_units.mean())

#     return grouped_units


def calc_unit_cluster_values(
    df: pd.DataFrame,
    capacity_col: str = "capacity_mw",
    technology: str = None,
    clustered: bool = True,
):
    """
    Calculate the total capacity, minimum load, weighted heat rate, and number of
    units/generators in a technology cluster.

    Parameters
    ----------
    df : dataframe
        A dataframe with units/generators of a single technology. One column should be
        'cluster', to label units as belonging to a specific cluster grouping.
    capacity_col: str
        Name of the column with capacity values (e.g. capacity_mw, summer_capacity_mw or
        winter_capacity_mw).
    technology : str, optional
        Name of the generating technology, by default None
    clustered : bool, optional
        If units are clustered or only a single unit is being passed, by default True

    Returns
    -------
    dataframe
        Aggragate values for generators in a technology cluster
    """
    # Make a copy and set capacity no NaN if not operating
    gen_df = df.copy()
    cap_cols = ["capacity_mw", "capacity_mwh"]
    gen_df.loc[gen_df["operating"] != True, cap_cols] = np.nan

    # if not clustering units no need to calulate cluster average values
    if len(gen_df) == 1:
        clustered = False
    elif gen_df["cluster"].nunique() == len(gen_df):
        clustered = False

    if not clustered:
        # df["Min_Power"] = df["minimum_load_mw"] / df[capacity_col]
        gen_df = gen_df[
            [
                "cluster",
                capacity_col,
                "capacity_mwh",
                "heat_rate_mmbtu_mwh",
                "fom_per_mwyr",
                "vom_per_mwh",
            ]
        ]
        gen_df["num_units"] = 1
        if technology:
            gen_df["technology"] = technology

        return gen_df.replace(np.inf, 0)

    # Define a function to compute the weighted mean.
    # The issue here is that the df name needs to be used in the function.
    # So this will need to be within a function that takes df as an input
    def wm(x):
        try:
            return np.average(
                x, weights=gen_df.loc[x.index, capacity_col].replace(np.nan, 0)
            )
        except ZeroDivisionError:
            return x.mean()

    # if df["heat_rate_mmbtu_mwh"].isnull().values.any():
    #     # mean =
    #     # df["heat_rate_mmbtu_mwh"] = df["heat_rate_mmbtu_mwh"].fillna(
    #     #     df["heat_rate_mmbtu_mwh"].median()
    #     # )
    #     start_cap = df[capacity_col].sum()
    #     df = df.loc[~df["heat_rate_mmbtu_mwh"].isnull(), :]
    #     end_cap = df[capacity_col].sum()
    #     cap_diff = start_cap - end_cap
    #     logger.warning(f"dropped {cap_diff}MW because of null heat rate values")

    df_values = gen_df.groupby("cluster", as_index=False).agg(
        {
            capacity_col: "sum",
            "capacity_mwh": "sum",
            # "minimum_load_mw": "mean",
            "heat_rate_mmbtu_mwh": wm,
            # "Fixed_OM_Cost_per_MWyr": wm,
            # "Var_OM_Cost_per_MWh": wm,
            "fom_per_mwyr": wm,
            "vom_per_mwh": wm,
        }
    )
    df_values.index = df_values["cluster"].values
    df_values["heat_rate_mmbtu_mwh_iqr"] = gen_df.groupby("cluster").agg(
        {"heat_rate_mmbtu_mwh": iqr}
    )
    df_values["heat_rate_mmbtu_mwh_std"] = gen_df.groupby("cluster").agg(
        {"heat_rate_mmbtu_mwh": "std"}
    )
    df_values["fom_per_mwyr_std"] = gen_df.groupby("cluster").agg(
        {"fom_per_mwyr": "std"}
    )

    # df_values["Min_Power"] = df_values["minimum_load_mw"] / df_values[capacity_col]

    df_values["num_units"] = (
        gen_df.dropna(subset=capacity_col).groupby("cluster")["cluster"].count()
    )

    if technology:
        df_values["technology"] = technology

    return df_values


def add_resource_tags(
    df: pd.DataFrame,
    model_tag_values: Dict[str, Dict[str, int]],
    regional_tag_values: Dict[str, Dict[str, Dict[str, int]]] = None,
    model_tag_names: List[str] = None,
    default_model_tag: Union[int, float, str] = 0,
) -> pd.DataFrame:
    """Add columns to the dataframe of resources and assign values based on technology
    names. Can be boolean-type integers, floats, or strings. Different values can be
    assigned by region.

    Each generator type needs to have certain tags for use by the GenX model. Each tag
    is a column, e.g. THERM for thermal generators. These columns and tag values are
    defined in the settings file and applied here.

    Keys representing technology names are sorted by length so that shorter names are
    applied first. This prevents short or generic names like "nuclear" from overriding
    more specific names like "nuclear_nuclear".

    Parameters
    ----------
    df : pd.DataFrame
        Each row represents one resource. Should have the column "technology",
        and the column "region" if `regional_tag_values` is used.
    model_tag_values : Dict[str, Dict[str, int]]
        Mapping of values to technology names that will be assigned to a new column using
        the key as the column name.
    regional_tag_values : Dict[str, Dict[str, Dict[str, int]]]
        Mapping of values to technology names within a given region that will be assigned
        to a new column using the key as the column name.
    model_tag_names : List[str], optional
        List of tag names to either assign specific values or the default value, by default
        None. Only necessary if a column is required but not specified within
        `model_tag_values` or `regional_tag_values`.
    default_model_tag : Union[int, float, str], optional
        The default value used to fill within a column before assigning specific values to
        each technology, by default 0

    Returns
    -------
    pd.DataFrame
        Modified version of the input df with new columns from the dictionary keys.
    """
    if "technology" not in df.columns:
        raise KeyError(
            "The column 'techology' is required when adding model tag values."
        )
    if regional_tag_values is not None and "region" not in df.columns:
        raise KeyError(
            "When assigning regional model tags, the column 'region' is required."
        )
    _df = df.copy()
    if model_tag_names is None:
        model_tag_names = []
    ignored = r"_"
    technology = _df["technology"].str.replace(ignored, "")

    global_keys = list((model_tag_values or {}).keys())
    regional_keys = []
    for region, regional_tags in (regional_tag_values or {}).items():
        regional_keys.extend(list(regional_tags.keys()))

    # global_keys = list(set(global_keys + regional_keys))
    # Create a new dataframe with the same index
    tags_no_value = set(model_tag_names) - set(global_keys + regional_keys)
    if tags_no_value:
        logger.warning(
            f"The model resource tags {tags_no_value} are listed in the settings parameter "
            "'model_tags_name' but are not assigned values for any resources"
        )
    for tag_col in set(model_tag_names + global_keys + regional_keys):
        _df.loc[:, tag_col] = default_model_tag
    for tag_col in global_keys:
        try:
            for tech, tag_value in sorted(
                model_tag_values[tag_col].items(),
                key=lambda item: len(str(item[0])),
            ):
                tech = re.sub(ignored, "", tech)
                mask = technology.str.contains(rf"{tech}", case=False, regex=False)
                _df.loc[mask, tag_col] = tag_value
        except (KeyError, AttributeError) as e:
            logger.warning(f"No model tag values found for {tag_col} ({e})")

    # Change tags with specific regional values for a technology
    flat_regional_tags = flatten(sort_nested_dict(regional_tag_values or {}))

    for tag_tuple, tag_value in flat_regional_tags.items():
        region, tag_col, tech = tag_tuple
        tech = re.sub(ignored, "", tech)
        mask = technology.str.contains(rf"{tech}", case=False, regex=False)
        _df.loc[(_df["region"] == region) & mask, tag_col] = tag_value

    return _df


# def download_860m(settings: dict) -> pd.ExcelFile:
#     """Load the entire 860m file into memory as an ExcelFile object.

#     Parameters
#     ----------
#     settings : dict
#         User-defined settings loaded from a YAML file. This is where the EIA860m
#         filename is defined as the parameter "eia_860m_fn".

#     Returns
#     -------
#     pd.ExcelFile
#         The ExcelFile object with all sheets from 860m.
#     """
#     fn = settings.get("eia_860m_fn")
#     if not fn:
#         logger.info(
#             "Trying to determine the most recent EIA860m file. For reproducible results "
#             "use the settings parameter 'eia_860m_fn'."
#         )
#         fn = find_newest_860m()

#     engine = None
#     ext = fn.split(".")[-1]
#     if ext == "xlsx":
#         engine = "openpyxl"
#     elif ext == "xls":
#         engine = "xlrd"

#     # Only the most recent file will not have archive in the url
#     url = f"https://www.eia.gov/electricity/data/eia860m/xls/{fn}"
#     archive_url = f"https://www.eia.gov/electricity/data/eia860m/archive/xls/{fn}"

#     local_file = DATA_PATHS["eia_860m"] / fn
#     if local_file.exists():
#         logger.debug(f"Reading a local copy of the EIA860m file {fn}")
#         eia_860m = pd.ExcelFile(local_file)
#     else:
#         logger.debug(f"Downloading the EIA860m file {fn}")
#         try:
#             download_save(url, local_file)
#             eia_860m = pd.ExcelFile(local_file, engine=engine)
#         except (XLRDError, ValueError, BadZipFile) as e:
#             logger.warning(
#                 f"There was an error when downloading the EIA-860m file. Trying again {e}"
#             )
#             download_save(archive_url, local_file)
#             eia_860m = pd.ExcelFile(local_file, engine=engine)
#         # write the file to disk

#     return eia_860m


# def find_newest_860m() -> str:
#     """Scrape the EIA 860m page to find the most recently posted file.

#     Returns
#     -------
#     str
#         Name of most recently posted file
#     """
#     site_url = "https://www.eia.gov/electricity/data/eia860m/"
#     r = requests.get(site_url)
#     soup = BeautifulSoup(r.content, "lxml")
#     table = soup.find("table", attrs={"class": "basic-table"})
#     if not table:
#         raise ValueError(
#             "Could not determine the most recently posted EIA 860m file. EIA may have "
#             "changed their HTML format, please post this as an issue on the PowerGenome "
#             "github repository (https://github.com/PowerGenome/PowerGenome/issues/new)."
#         )
#     href = table.find("a")["href"]
#     fn = href.split("/")[-1]

#     return fn


# def clean_860m_sheet(
#     eia_860m: pd.ExcelFile, sheet_name: str, settings: dict
# ) -> pd.DataFrame:
#     """Load a sheet from the 860m ExcelFile object and clean it.

#     Parameters
#     ----------
#     eia_860m : ExcelFile
#         Entire 860m file loaded into memory
#     sheet_name : str
#         Name of the sheet to load as a dataframe
#     settings : dict
#         User-defined settings loaded from a YAML file.

#     Returns
#     -------
#     pd.DataFrame
#         One of the sheets from 860m
#     """

#     df = eia_860m.parse(sheet_name=sheet_name, na_values=[" "])

#     # Find skiprows and skipfooters, which changes across 860m versions.
#     # NEW: drop rows with all NaN because EIA added a blank row before the footer.
#     sr = 0

#     for idx, row in df.iterrows():
#         if row.iloc[0] == "Entity ID":
#             sr = idx + 1
#             break

#     sf = 0

#     for idx in list(range(-10, 0)):
#         if isinstance(df.iloc[idx, 0], str):
#             sf = -idx
#             break
#     df = eia_860m.parse(
#         sheet_name=sheet_name, skiprows=sr, skipfooter=sf, na_values=[" "]
#     )
#     df = df.dropna(how="all")
#     df = df.rename(columns=planned_col_map)
#     df["plant_id_eia"] = df["plant_id_eia"].astype("Int64")

#     if sheet_name in ["Operating", "Planned"]:
#         df.loc[:, "operational_status_code"] = df.loc[:, "operational_status"].map(
#             op_status_map
#         )

#     df.columns = snake_case_col(df.columns)

#     return df


# def load_860m(settings: dict) -> Dict[str, pd.DataFrame]:
#     """Load the planned, canceled, and retired sheets from an EIA 860m file.

#     Parameters
#     ----------
#     settings : dict
#         User-defined settings loaded from a YAML file. This is where the EIA860m
#         filename is defined.

#     Returns
#     -------
#     Dict[str, pd.DataFrame]
#         The 860m dataframes, with the keys 'planned', 'canceled', and 'retired'.
#     """
#     sheet_map = {
#         "operating": "Operating",
#         "planned": "Planned",
#         "canceled": "Canceled or Postponed",
#         "retired": "Retired",
#     }

#     fn = settings.get("eia_860m_fn")
#     if not fn:
#         fn = find_newest_860m()

#     fn_name = Path(fn).stem

#     data_dict = {}
#     eia_860m_excelfile = None
#     for name, sheet in sheet_map.items():
#         pkl_path = DATA_PATHS["eia_860m"] / f"{fn_name}_{name}.pkl"
#         if pkl_path.exists():
#             data_dict[name] = pd.read_pickle(pkl_path)
#             if sheet == "Planned":
#                 data_dict[name] = filter_op_status_codes(
#                     data_dict[name], settings.get("proposed_status_included")
#                 )
#             data_dict[name]["plant_id_eia"] = data_dict[name]["plant_id_eia"].astype(
#                 "Int64"
#             )
#             data_dict[name].columns = snake_case_col(data_dict[name].columns)
#         else:
#             if eia_860m_excelfile is None:
#                 eia_860m_excelfile = download_860m(settings)
#             data_dict[name] = clean_860m_sheet(eia_860m_excelfile, sheet, settings)
#             if sheet == "planned":
#                 data_dict[name] = filter_op_status_codes(
#                     data_dict[name], settings.get("proposed_status_included")
#                 )
#             data_dict[name].to_pickle(pkl_path)

#     return data_dict


# def filter_op_status_codes(
#     df: pd.DataFrame, proposed_status_included: Union[List[str], None]
# ) -> pd.DataFrame:
#     """Filter a planned 860m sheet to only include desired operational status codes.
#     Used to filter out projects that are still early in the pipeline and might not be built.

#     If proposed_status_included is None, included all proposed plants. Will warn user
#     in logs if invalid codes are included.

#     Parameters
#     ----------
#     df : pd.DataFrame
#         EIA860m planned generators. Includes the column "operational_status_code".
#     proposed_status_included : Union[List[str], None]
#         List of status codes for proposed generators that should be included in the model.
#         Examples include "V" (Under construction, more than 50 percent complete),
#         "TS" (Construction complete, but not yet in commercial operation), "U", (Under
#         construction, less than or equal to 50 percent complete), etc.

#     Returns
#     -------
#     pd.DataFrame
#         Filtered input dataframe.
#     """
#     if proposed_status_included is None:
#         return df

#     valid_status_codes = df["operational_status_code"].unique()
#     invalid_user_codes = [
#         c for c in proposed_status_included if c not in valid_status_codes
#     ]
#     if invalid_user_codes:
#         logger.warning(
#             f"The operational status codes {invalid_user_codes} included in the "
#             "settings parameter 'proposed_status_included' do not appear in EIA 860m.\n"
#             f"Valid status codes from 'operational_status_code' are {valid_status_codes}"
#         )
#     return df.loc[df["operational_status_code"].isin(proposed_status_included), :]


# def label_gen_region(
#     df: pd.DataFrame,
#     model_regions_gdf: gpd.GeoDataFrame,
#     capacity_col: str = "capacity_mw",
# ) -> pd.DataFrame:
#     """Label the region that generators in a dataframe belong to based on their
#     geographic location. This is done via geospaital join and may not always be accurate
#     based on actual utility connections.

#     Parameters
#     ----------
#     df : pd.DataFrame
#         Generators that are not assigned to a model region.
#     model_regions_gdf : gpd.GeoDataFrame
#         Contains the name and geometry of each region being used in the study
#     capacity_col : str, optional
#         The column of `df` that gives the capacity of each plant, by default "capacity_mw"

#     Returns
#     -------
#     pd.DataFrame
#         [description]
#     """

#     no_lat_lon = df.loc[
#         (df["latitude"].isnull()) | (df["longitude"].isnull()), :
#     ].copy()
#     if not no_lat_lon.empty:
#         no_lat_lon_cap = no_lat_lon[capacity_col].sum()
#         logger.warning(
#             "Some generators do not have lon/lat data. Check the source "
#             "file to determine if they should be included in results. "
#             f"\nThe affected generators account for {no_lat_lon_cap} MW in "
#             "these balancing authorities: "
#             f"\n{no_lat_lon['balancing_authority_code'].tolist()}"
#         )

#     df = df.dropna(subset=["latitude", "longitude"])

#     # Convert the lon/lat values to geo points. Need to add an initial CRS and then
#     # change it to align with the IPM regions
#     gdf = gpd.GeoDataFrame(
#         df.copy(),
#         geometry=gpd.points_from_xy(df.longitude.copy(), df.latitude.copy()),
#         crs="EPSG:4326",
#     )
#     if gdf.crs != model_regions_gdf.crs:
#         gdf = gdf.to_crs(model_regions_gdf.crs)

#     gdf = gpd.sjoin(model_regions_gdf.drop(columns="region"), gdf)

#     return gdf


# def import_new_generators(
#     operating_860m: pd.DataFrame,
#     gens_860: pd.DataFrame,
#     model_regions_gdf: gpd.GeoDataFrame,
#     proposed_gen_heat_rates: Dict[str, float],
#     proposed_min_load: Dict[str, float],
#     capacity_col: str = "capacity_mw",
# ) -> pd.DataFrame:
#     """Find the set of generating units in 860m that are not in the annual 860 data.

#     This is especially important for new wind, solar, and battery units, which are built
#     on short timelines. Format the data for inclusion with other existing existing units.


#     Parameters
#     ----------
#     operating_860m : pd.DataFrame
#         Operating generators from EIA 860m.
#     gens_860 : pd.DataFrame
#         The set of operating units from other sources (e.g. annual 860 data in PUDL).
#     model_regions_gdf : gpd.GeoDataFrame
#         Geospatial representation of the model regions. Used to assign generators to
#         model regions.
#     proposed_gen_heat_rates : Dict[str, float]
#         Heat rates to use for proposed technology types
#     proposed_min_load : Dict[str, float]
#         Min load to use for proposed technology types
#     capacity_col : str, optional
#         Column with capacity value, by default "capacity_mw"

#     Returns
#     -------
#     pd.DataFrame
#         Set of operating generators that were not already in the gens_860 dataframe
#     """
#     _operating_860m = operating_860m.copy()
#     _operating_860m["generator_id"] = _operating_860m["generator_id"].apply(
#         remove_leading_zero
#     )
#     gens_860_id = list(zip(gens_860["plant_id_eia"], gens_860["generator_id"]))
#     operating_860m_id = zip(
#         _operating_860m["plant_id_eia"], _operating_860m["generator_id"]
#     )

#     new_mask = [g not in gens_860_id for g in operating_860m_id]
#     new_operating = label_gen_region(
#         _operating_860m.loc[new_mask, :], model_regions_gdf, capacity_col
#     )
#     new_operating = new_operating.drop_duplicates(
#         subset=["plant_id_eia", "generator_id"]
#     )
#     new_operating.loc[:, "heat_rate_mmbtu_mwh"] = new_operating.loc[
#         :, "technology_description"
#     ].map(proposed_gen_heat_rates or {})

#     # The default EIA heat rate for non-thermal technologies is 9.21
#     new_operating.loc[
#         new_operating["heat_rate_mmbtu_mwh"].isnull(), "heat_rate_mmbtu_mwh"
#     ] = 9.21

#     new_operating.loc[:, "minimum_load_mw"] = (
#         new_operating["technology_description"].map(proposed_min_load or {})
#         * new_operating[capacity_col]
#     )

#     # Assume anything else being built at scale is wind/solar and will have a Min_power
#     # of 0
#     new_operating.loc[new_operating["minimum_load_mw"].isnull(), "minimum_load_mw"] = 0

#     new_operating = new_operating.set_index(
#         ["plant_id_eia", "prime_mover_code", "energy_source_code_1"]
#     )
#     if (
#         new_operating.loc[new_operating["technology_description"].isnull(), :].empty
#         is False
#     ):
#         plant_ids = list(
#             new_operating.loc[new_operating["technology_description"].isnull(), :]
#             .index.get_level_values("plant_id_eia")
#             .to_numpy()
#         )
#         plant_capacity = new_operating.loc[
#             new_operating["technology_description"].isnull(),
#             capacity_col,
#         ].sum()

#         logger.debug(
#             f"The EIA860 file has {len(plant_ids)} operating generator(s) without a technology "
#             f"description. The plant IDs are {plant_ids}, and they have a combined "
#             f"capacity of {plant_capacity} MW."
#         )

#     keep_cols = [
#         "model_region",
#         "technology_description",
#         "generator_id",
#         capacity_col,
#         "capacity_mwh",
#         "minimum_load_mw",
#         "operational_status_code",
#         "heat_rate_mmbtu_mwh",
#         "retirement_year",
#         "operating_year",
#         "state",
#     ]

#     return new_operating.reindex(columns=keep_cols)


# def import_proposed_generators(
#     planned: pd.DataFrame,
#     model_year: int,
#     model_regions_gdf: gpd.GeoDataFrame,
#     proposed_gen_heat_rates: Dict[str, float],
#     proposed_min_load: Dict[str, float],
#     capacity_col: str = "capacity_mw",
# ) -> pd.DataFrame:
#     """Load the most recent proposed generating units from EIA860m. Add model region,
#     heat rate, and min load data for generators that will be built by the model year.

#     Parameters
#     ----------
#     planned : pd.DataFrame
#         Planned generators that are not assigned to a model region. Should contain
#         columns "planned_operating_year", "generator_id", "technology_description",
#         "plant_id_eia", "prime_mover_code", "energy_source_code_1", and
#         "operational_status_code",
#     model_year : int
#         Year of the model, used to filter which proposed generators will be included
#     model_regions_gdf : gpd.GeoDataFrame
#         GeoDataframe of the model regions
#     proposed_gen_heat_rates : Dict[str, float]
#         Heat rates to use for proposed technology types
#     proposed_min_load : Dict[str, float]
#         Min load to use for proposed technology types
#     capacity_col : str, optional
#         Column with capacity value, by default "capacity_mw"

#     Returns
#     -------
#     pd.DataFrame
#         All proposed generators that will be built before or in the model year.
#     """
#     _planned = planned.loc[planned["planned_operating_year"] <= model_year, :]
#     _planned["generator_id"] = _planned["generator_id"].fillna("no_gen_id")
#     _planned["generator_id"] = _planned["generator_id"].apply(remove_leading_zero)
#     planned_gdf = label_gen_region(_planned, model_regions_gdf, capacity_col)
#     planned_gdf = planned_gdf.drop_duplicates(subset=["plant_id_eia", "generator_id"])

#     planned_gdf.loc[:, "heat_rate_mmbtu_mwh"] = planned_gdf.loc[
#         :, "technology_description"
#     ].map(proposed_gen_heat_rates)

#     # The default EIA heat rate for non-thermal technologies is 9.21
#     planned_gdf.loc[
#         planned_gdf["heat_rate_mmbtu_mwh"].isnull(), "heat_rate_mmbtu_mwh"
#     ] = 9.21

#     planned_gdf.loc[:, "minimum_load_mw"] = (
#         planned_gdf["technology_description"].map(proposed_min_load)
#         * planned_gdf[capacity_col]
#     )

#     # Assume anything else being built at scale is wind/solar and will have a Min_Power
#     # of 0
#     planned_gdf.loc[planned_gdf["minimum_load_mw"].isnull(), "minimum_load_mw"] = 0

#     planned_gdf = planned_gdf.set_index(
#         ["plant_id_eia", "prime_mover_code", "energy_source_code_1"]
#     )

#     if (
#         planned_gdf.loc[planned_gdf["technology_description"].isnull(), :].empty
#         is False
#     ):
#         plant_ids = list(
#             planned_gdf.loc[planned_gdf["technology_description"].isnull(), :]
#             .index.get_level_values("plant_id_eia")
#             .to_numpy()
#         )
#         plant_capacity = planned_gdf.loc[
#             planned_gdf["technology_description"].isnull(), capacity_col
#         ].sum()

#         logger.debug(
#             f"The EIA860 file has {len(plant_ids)} proposed generator(s) without a technology "
#             f"description. The plant IDs are {plant_ids}, and they have a combined "
#             f"capacity of {plant_capacity} MW."
#         )

#     keep_cols = [
#         "model_region",
#         "technology_description",
#         "generator_id",
#         capacity_col,
#         "minimum_load_mw",
#         "operational_status_code",
#         "heat_rate_mmbtu_mwh",
#         "planned_operating_year",
#     ]

#     return planned_gdf.loc[:, keep_cols]


# def gentype_region_capacity_factor(
#     pudl_engine, plant_region_map, settings, years_filter=None
# ):
#     """
#     Calculate the average capacity factor for all generators of a type/region. This
#     uses all years of available data unless otherwise specified. The potential
#     generation is calculated for every year a plant is in operation using the capacity
#     type specified in settings (nameplate, summer, or winter) and the number of hours
#     in each year.

#     As of this time PUDL only has generation data back to 2011.

#     Parameters
#     ----------
#     pudl_engine : sqlalchemy.Engine
#         A sqlalchemy connection for use by pandas
#     plant_region_map : dataframe
#         A dataframe with the region for every plant
#     settings : dictionary
#         The dictionary of settings with a dictionary of region aggregations

#     Returns
#     -------
#     DataFrame
#         A dataframe with the capacity factor of every selected technology
#     """
#     data_years = settings[
#         "eia_data_years"
#     ].copy()  # [str(y) for y in settings["eia_data_years"]]
#     data_years.extend(settings.get("capacity_factor_default_year_filter", []))

#     cap_col = settings["capacity_col"]

#     # Include standby (SB) generators since they are in our capacity totals
#     sql = f"""
#         SELECT
#             G.report_date,
#             G.plant_id_eia,
#             G.generator_id,
#             SUM(G.capacity_mw) AS capacity_mw,
#             SUM(G.summer_capacity_mw) as summer_capacity_mw,
#             SUM(G.winter_capacity_mw) as winter_capacity_mw,
#             G.technology_description,
#             G.fuel_type_code_pudl
#         FROM
#             generators_eia860 G
#         WHERE operational_status_code NOT IN ('RE', 'OS', 'IP', 'CN')
#         AND strftime('%Y',report_date) in ({','.join(['?']*len(data_years))})
#         GROUP BY
#             G.report_date,
#             G.plant_id_eia,
#             G.technology_description,
#             G.fuel_type_code_pudl,
#             G.generator_id
#         ORDER by G.plant_id_eia, G.report_date
#     """

#     plant_gen_tech_cap = pd.read_sql_query(
#         sql,
#         pudl_engine,
#         params=[str(y) for y in data_years],
#         parse_dates=["report_date"],
#     )
#     plant_gen_tech_cap = plant_gen_tech_cap.loc[
#         plant_gen_tech_cap["plant_id_eia"].isin(plant_region_map["plant_id_eia"]), :
#     ]

#     plant_gen_tech_cap = fill_missing_tech_descriptions(plant_gen_tech_cap)
#     plant_tech_cap = group_generators_at_plant(
#         df=plant_gen_tech_cap,
#         by=["plant_id_eia", "report_date", "technology_description"],
#         agg_fn={cap_col: "sum"},
#     )

#     plant_tech_cap = plant_tech_cap.merge(
#         plant_region_map, on="plant_id_eia", how="left"
#     )

#     label_small_hydro(plant_tech_cap, settings, by=["plant_id_eia", "report_date"])

#     sql = """
#         SELECT
#             strftime('%Y', GF.report_date) AS report_date,
#             GF.plant_id_eia,
#             SUM(GF.net_generation_mwh) AS net_generation_mwh,
#             GF.fuel_type_code_pudl
#         FROM
#             generation_fuel_eia923 GF
#         GROUP BY
#             strftime('%Y', GF.report_date),
#             GF.plant_id_eia,
#             GF.fuel_type_code_pudl
#         ORDER by GF.plant_id_eia, strftime('%Y', GF.report_date)
#     """
#     generation = pd.read_sql_query(sql, pudl_engine, parse_dates={"report_date": "%Y"})

#     if pudl.__version__ > "0.5.0":
#         by = ["plant_id_eia"]
#     else:
#         by = {"plant_id_eia": "eia"}
#     if pudl.__version__ < "2022.11.30":
#         capacity_factor = pudl.helpers.clean_merge_asof(
#             generation,
#             plant_tech_cap,
#             left_on="report_date",
#             right_on="report_date",
#             by=by,
#         )
#     else:
#         capacity_factor = pudl.helpers.date_merge(
#             generation,
#             plant_tech_cap,
#             on=by,
#         )

#     if settings.get("group_technologies"):
#         capacity_factor = group_technologies(
#             capacity_factor,
#             settings.get("tech_groups", {}) or {},
#             settings.get("regional_no_grouping", {}) or {},
#         )

#     cf_years = settings.get("capacity_factor_default_year_filter", data_years)
#     capacity_factor = capacity_factor.loc[
#         capacity_factor["report_date"].dt.year.isin(cf_years)
#     ]

#     # get a unique set of dates to generate the number of hours
#     dates = capacity_factor["report_date"].drop_duplicates()
#     dates_to_hours = pd.DataFrame(
#         data={
#             "report_date": dates,
#             "hours": dates.apply(
#                 lambda d: (
#                     pd.date_range(d, periods=2, freq="YS")[1]
#                     - pd.date_range(d, periods=2, freq="YS")[0]
#                 )
#                 / pd.Timedelta(hours=1)
#             ),
#         }
#     )

#     # merge in the hours for the calculation
#     capacity_factor = capacity_factor.merge(dates_to_hours, on=["report_date"])
#     capacity_factor["potential_generation_mwh"] = (
#         capacity_factor[cap_col] * capacity_factor["hours"]
#     )
#     plant_tech_capacity_factor = capacity_factor.groupby(
#         ["plant_id_eia", "technology_description"], as_index=False
#     )[["potential_generation_mwh", "net_generation_mwh"]].sum()

#     plant_tech_capacity_factor["capacity_factor"] = (
#         plant_tech_capacity_factor["net_generation_mwh"]
#         / plant_tech_capacity_factor["potential_generation_mwh"]
#     )
#     plant_tech_capacity_factor.rename(
#         columns={"technology_description": "technology"},
#         inplace=True,
#     )

#     capacity_factor_tech_region = capacity_factor.groupby(
#         ["model_region", "technology_description"], as_index=False
#     )[["potential_generation_mwh", "net_generation_mwh"]].sum()

#     # actually calculate capacity factor wooo!
#     capacity_factor_tech_region["capacity_factor"] = (
#         capacity_factor_tech_region["net_generation_mwh"]
#         / capacity_factor_tech_region["potential_generation_mwh"]
#     )

#     capacity_factor_tech_region.rename(
#         columns={"model_region": "region", "technology_description": "technology"},
#         inplace=True,
#     )

#     logger.debug(capacity_factor_tech_region)

#     return plant_tech_capacity_factor, capacity_factor_tech_region


def add_fuel_labels(df, fuel_prices, settings):
    """Add a Fuel column with the approproriate regional fuel for each generator type

    Parameters
    ----------
    df : DataFrame
        Generator clusters dataframe with all existing and proposed technologies
    fuel_prices : DataFrame
        Prices of fuels from EIA AEO scenarios in each census region. Columns include
        ['year', 'price', 'fuel', 'region', 'scenario', 'full_fuel_name']
    settings : dictionary
        The dictionary of settings with fuel price variables

    Returns
    -------
    DataFrame
        Same as input, but with a new column "Fuel" that is either the name of the
        corresponding fuel (coal, natural_gas, uranium, or distillate) or "No_fuel".

    Raises
    ------
    KeyError
        The model region is not mapped to a fuel region in 'fuel_region_map'
    """

    df["Fuel"] = np.nan
    # This variable is called eia_tech but it can be any tech name or a mapping from
    # EIA technologies through to other techs via "eia_atb_tech_map"
    for eia_tech, fuel in (settings.get("tech_fuel_map") or {}).items():
        try:
            if eia_tech == "Natural Gas Steam Turbine":
                # No ATB natural gas steam turbine and I match it with coal for O&M
                # which would screw this up and list natural gas as a fuel for ATB
                # coal plants
                atb_tech = None
            else:
                if not isinstance(settings["eia_atb_tech_map"][eia_tech], list):
                    settings["eia_atb_tech_map"][eia_tech] = [
                        settings["eia_atb_tech_map"][eia_tech]
                    ]
                atb_tech = [
                    tech.split("_")[0] + "_"
                    for tech in settings["eia_atb_tech_map"][eia_tech]
                ]
        except KeyError:
            # No corresponding ATB technology
            atb_tech = None
        scenario = settings.get("fuel_scenarios", {}).get(fuel)
        model_year = settings["model_year"]
        if not scenario:
            if fuel not in settings.get("user_fuel_price", []) or []:
                raise KeyError(
                    f"The fuel type '{fuel}' is not in the settings parameters "
                    "'fuel_scenarios' or 'user_fuel_price'. All fuels listed in "
                    "'tech_fuel_map' must be included in one of these."
                )
            if isinstance(settings["user_fuel_price"][fuel], dict):
                for region, price in settings["user_fuel_price"][fuel].items():
                    fuel_name = f"{region}_{fuel}"
                    df.loc[
                        (
                            df["technology"].str.rstrip("_").str.lower()
                            == eia_tech.lower()
                        )
                        & (df["region"] == region),
                        "Fuel",
                    ] = fuel_name

                    if atb_tech is not None:
                        for tech in atb_tech:
                            df.loc[
                                (
                                    df["technology"].str.contains(
                                        tech, case=False, regex=False
                                    )
                                )
                                & (df["region"] == region)
                                & (df["Fuel"].isna()),
                                "Fuel",
                            ] = fuel_name
            else:
                df.loc[
                    (df["technology"].str.rstrip("_").str.lower() == eia_tech.lower())
                    & (df["Fuel"].isna()),
                    "Fuel",
                ] = fuel

                if atb_tech is not None:
                    for tech in atb_tech:
                        df.loc[
                            (
                                df["technology"].str.contains(
                                    tech, case=False, regex=False
                                )
                            )
                            & (df["Fuel"].isna()),
                            "Fuel",
                        ] = fuel
        else:
            for aeo_region, model_regions in settings["fuel_region_map"].items():
                fuel_name = ("_").join([aeo_region, scenario, fuel])
                assert (
                    fuel_prices.query(
                        "year==@model_year & full_fuel_name==@fuel_name"
                    ).empty
                    is False
                ), f"{fuel_name} doesn't show up in {model_year}"

                df.loc[
                    (df["technology"].str.contains(eia_tech, case=False, regex=False))
                    & df["region"].isin(model_regions),
                    "Fuel",
                ] = fuel_name

                if atb_tech is not None:
                    for tech in atb_tech:
                        df.loc[
                            (
                                df["technology"].str.contains(
                                    tech, case=False, regex=False
                                )
                            )
                            & (df["region"].isin(model_regions))
                            & (df["Fuel"].isna()),
                            "Fuel",
                        ] = fuel_name

    for ccs_tech, ccs_fuel in (settings.get("ccs_fuel_map") or {}).items():
        ccs_base_name = ("_").join(ccs_fuel.split("_")[:-1])
        if ccs_base_name in (settings.get("fuel_scenarios", {}) or {}).keys():
            scenario = settings["fuel_scenarios"][ccs_base_name]
            for aeo_region, model_regions in settings["fuel_region_map"].items():
                ccs_fuel_name = ("_").join([aeo_region, scenario, ccs_fuel])

                df.loc[
                    (df["technology"].str.contains(ccs_tech))
                    & df["region"].isin(model_regions),
                    "Fuel",
                ] = ccs_fuel_name
        elif ccs_base_name in (settings.get("user_fuel_price", {}) or {}).keys():
            if isinstance(settings["user_fuel_price"][ccs_base_name], dict):
                for region in settings["user_fuel_price"][ccs_base_name].keys():
                    ccs_fuel_name = ("_").join([region, ccs_fuel])
                    df.loc[
                        (
                            df["technology"].str.contains(
                                ccs_tech, case=False, regex=False
                            )
                        )
                        & (df["region"] == region),
                        "Fuel",
                    ] = ccs_fuel_name
            else:
                df.loc[
                    (df["technology"].str.contains(ccs_tech, case=False, regex=False)),
                    "Fuel",
                ] = ccs_fuel
        else:
            logger.warning(
                f"The fuel {ccs_fuel} is included in settings parameter `ccs_fuel_map` "
                "but it can't be matched against an AEO or user fuel. CCS fuels should "
                "have the format <fuel name>_ccs<capture rate>, where the capture rate "
                "is optional. The <fuel name> should match a fuel from `fuel_scenarios' "
                "or `user_fuel_prices`."
            )

    # Replace AEO region name with model region in cases where users are modifying AEO price
    model_aeo_region_map = reverse_dict_of_lists(settings.get("fuel_region_map", {}))
    for region, adj in (settings.get("regional_fuel_adjustments", {}) or {}).items():
        aeo_region = model_aeo_region_map.get(region)
        if not aeo_region:
            raise KeyError(
                f"There is no mapping of the model region {region} to an AEO fuel region "
                "in the settings parameter 'fuel_region_map'."
            )
        if isinstance(adj, list):
            # Replace the aeo region name with model region for all resources
            df.loc[
                (df["Fuel"].str.contains(aeo_region))
                & (df["Fuel"].notna())
                & (df["region"].str.lower() == region.lower()),
                "Fuel",
            ] = df.loc[
                (df["Fuel"].str.contains(aeo_region))
                & (df["Fuel"].notna())
                & (df["region"].str.lower() == region.lower()),
                "Fuel",
            ].str.replace(
                aeo_region, region
            )
        if isinstance(adj, dict):
            # Replace the aeo region name with model region only for select fuels
            for fuel, op in adj.items():
                df.loc[
                    (df["Fuel"].str.contains(aeo_region))
                    & (df["Fuel"].str.contains(fuel))
                    & (df["Fuel"].notna())
                    & (df["region"].str.lower() == region.lower()),
                    "Fuel",
                ] = df.loc[
                    (df["Fuel"].str.contains(aeo_region))
                    & (df["Fuel"].str.contains(fuel))
                    & (df["Fuel"].notna())
                    & (df["region"].str.lower() == region.lower()),
                    "Fuel",
                ].str.replace(
                    aeo_region, region
                )

    df.loc[df["Fuel"].isna(), "Fuel"] = "No_fuel"

    return df


def calculate_transmission_inv_cost(resource_df, settings, offshore_spur_costs=None):
    """Calculate the transmission investment cost for each new resource.

    Parameters
    ----------
    resource_df : DataFrame
        Each row represents a single resource within a region. Should have columns
        `region` and `<type>_miles`, where transmission <type> is one of
        'spur', 'offshore_spure', or 'tx'.
    settings : dict
        A dictionary of user-supplied settings. Must have key
        `transmission_investment_cost` with the format:
            - <type>
                - `capex_mw_mile` (float)
                - `wacc` (float)
                - `investment_years` (int)
            - ...
    offshore_spur_costs : DataFrame
        Offshore spur costs per mile in the format
        `technology` ('OffShoreWind'), `tech_detail`, `cost_case`, and `capex_mw_mile`.
        Only used if `settings.transmission_investment_cost.capex_mw_mile` is missing.

    Returns
    -------
    DataFrame
        Modified copy of the input dataframe with new columns '<type>_capex' and
        '<type>_inv_mwyr' for each column `<type>_miles`.

    Raises
    ------
    KeyError
        Settings missing transmission types present in resources.
    KeyError
        Settings missing required keys.
    KeyError
        Setting capex_mw_mile missing regions present in resources.
    TypeError
        Setting capex_mw_mile is neither a dictionary nor a numeric value.
    """
    SETTING = "transmission_investment_cost"
    KEYS = ["wacc", "investment_years", "capex_mw_mile"]
    ttypes = settings.get(SETTING, {})
    # Check coverage of transmission types in resources
    resource_ttypes = [x for x in TRANSMISSION_TYPES if f"{x}_miles" in resource_df]
    missing_ttypes = list(set(resource_ttypes) - set(ttypes))
    if missing_ttypes:
        raise KeyError(f"{SETTING} missing transmission line types {missing_ttypes}")
    # Apply calculation for each transmission type
    regions = resource_df["region"].unique()
    use_offshore_spur_costs = False
    for ttype, params in ttypes.items():
        if ttype not in resource_ttypes:
            continue
        if (
            ttype == "offshore_spur"
            and offshore_spur_costs is not None
            and not params.get("capex_mw_mile")
        ):
            use_offshore_spur_costs = True
            # Build technology: capex_mw_mile map
            params = params.copy()
            params["capex_mw_mile"] = (
                offshore_spur_costs.assign(
                    technology=offshore_spur_costs[
                        ["technology", "tech_detail", "cost_case"]
                    ]
                    .astype(str)
                    .agg("_".join, axis=1)
                )
                .set_index("technology")["capex_mw_mile"]
                .to_dict()
            )
        # Check presence of required keys
        missing_keys = list(set(KEYS) - set(params))
        if missing_keys:
            raise KeyError(f"{SETTING}.{ttype} missing required keys {missing_keys}")
        if isinstance(params["capex_mw_mile"], dict):
            if use_offshore_spur_costs:
                capex_mw_mile = resource_df["technology"].map(params["capex_mw_mile"])
            else:
                # Check coverage of regions in resources
                missing_regions = list(set(regions) - set(params["capex_mw_mile"]))
                if missing_regions:
                    raise KeyError(
                        f"{SETTING}.{ttype}.capex_mw_mile missing regions {missing_regions}"
                    )
                capex_mw_mile = (
                    resource_df["region"].map(params["capex_mw_mile"]).fillna(0)
                )
        elif isinstance(params["capex_mw_mile"], Number):
            capex_mw_mile = params["capex_mw_mile"]
        else:
            raise TypeError(
                f"{SETTING}.{ttype}.capex_mw_mile should be numeric or a dictionary"
                f" of <region>: <capex>, not {params['capex_mw_mile']}"
            )
        resource_df[f"{ttype}_capex"] = capex_mw_mile * resource_df[f"{ttype}_miles"]
        resource_df[f"{ttype}_inv_mwyr"] = investment_cost_calculator(
            resource_df[f"{ttype}_capex"],
            params["wacc"],
            params["investment_years"],
            settings.get("interest_compound_method", "discrete"),
        )
    return resource_df


def add_transmission_inv_cost(
    resource_df: pd.DataFrame, settings: dict
) -> pd.DataFrame:
    """Add tranmission investment costs to plant investment costs

    Parameters
    ----------
    resource_df
        Each row represents a single resource within a region. Should have columns
        `Inv_Cost_per_MWyr` and transmission costs.
            - one or more `<type>_inv_mwyr`,
                where <type> is 'spur', 'offshore_spur', or 'tx'.
            - `interconnect_annuity`
    settings
        User settings. If `transmission_investment_cost.use_total` is present and true,
        `interconnect_annuity` is used over `<type>_inv_mwys` if present, not null,
        and not zero.

    Returns
    -------
    DataFrame
        A modified copy of the input dataframe where 'Inv_Cost_per_MWyr' represents the
        combined plant and transmission investment costs. The new column
        `plant_inv_cost_mwyr` represents just the plant investment costs.
    """
    use_total = (
        settings.get("transmission_investment_cost", {}).get("use_total", False)
        and "interconnect_annuity" in resource_df
    )
    resource_df["plant_inv_cost_mwyr"] = resource_df["Inv_Cost_per_MWyr"]
    columns = [
        c for c in [f"{t}_inv_mwyr" for t in TRANSMISSION_TYPES] if c in resource_df
    ]
    cost = resource_df[columns].sum(axis=1)
    if use_total:
        total = resource_df["interconnect_annuity"]
        has_total = ~total.isna() & total != 0
        cost[has_total] = total[has_total]
    if cost.isna().any() or (cost == 0).any():
        logger.warning(
            "Transmission investment costs are missing or zero for some resources"
            " and will not be included in the total investment costs."
        )
    resource_df["Inv_Cost_per_MWyr"] += cost
    return resource_df


# def save_weighted_hr(weighted_unit_hr, pudl_engine):
#     pass


def add_dg_resources(
    settings: dict,
    gen_df: pd.DataFrame = pd.DataFrame(),
) -> pd.DataFrame:
    """Add distributed generation resources as rows in a generators dataframe

    Parameters
    ----------
    settings : dict
        Settings dictionary with parameters "model_year", "input_folder", "distributed_gen_profiles_fn",
        "distributed_gen_method", "distributed_gen_values", and "avg_distribution_loss".
    gen_df : pd.DataFrame, optional
        A dataframe with other generating resources, by default pd.DataFrame()

    Returns
    -------
        A modified version of the input dataframe with distributed generation resources
        for each region where a generation profile has been supplied in the
        "distributed_gen_profiles_fn" file. Each dg resource is one row and includes
        values for the columns "technology", "region", "capacity_mw", and "profile".
    """
    dg_profiles = make_distributed_gen_profiles(settings)
    df = pd.DataFrame(
        columns=["technology", "region", "cluster", "capacity_mw", "profile"],
        index=range(len(dg_profiles.columns)),
    )

    for idx, (region, s) in enumerate(dg_profiles.items()):
        cap = s.max()
        df.loc[idx, "profile"] = (s / cap).round(3).to_numpy()
        df.loc[idx, "capacity_mw"] = cap.round(0).astype(int)
    df["technology"] = "distributed_generation"
    df["region"] = dg_profiles.columns
    df["cluster"] = 1
    df["Resource"] = create_resource_label(
        df["region"], snake_case_col(df["technology"]), df["cluster"], sep="_"
    )

    return pd.concat([gen_df, df], ignore_index=True)


def energy_storage_mwh(
    df: pd.DataFrame,
    energy_storage_duration: Dict[str, Union[float, Dict[str, float]]],
    tech_col: str,
    cap_col: str,
    energy_col: str,
) -> pd.DataFrame:
    """Convert resource capacity (MW) to MWh using a dictionary with storage duration
    by technology name.

    Parameters
    ----------
    df : pd.DataFrame
        Resource dataframe with columns specified by `tech_col`, `cap_col`, and
        `energy_col`
    energy_storage_duration : Dict[str, Union[float, Dict[str, float]]]
        Keys are technology names, values are either the duration of storage (float) or
        a dictionary with region keys and storage duration values
    tech_col : str
        Dataframe column with technology names
    cap_col : str
        Dataframe column with technology capacity (power)
    energy_col : str
        Dataframe column to fill with technology energy storage

    Returns
    -------
    pd.DataFrame
        Modified dataframe with energy storage values
    """
    context = "Setting energy storage MWh in existing generators."
    region_col = find_region_col(df.columns, context)
    all_regions = df[region_col].unique()

    if energy_col not in df.columns:
        df[energy_col] = 0

    storage_techs = list(df.loc[df[energy_col] > 0, tech_col].unique())
    partial_storage = list(
        df.loc[
            (df[tech_col].isin(storage_techs)) & ~(df[energy_col] > 0),
            tech_col,
        ].unique()
    )
    missing_techs = [t for t in partial_storage if t not in energy_storage_duration]
    if missing_techs:
        logger.warning(
            f"The storage technology(ies) {missing_techs} have some existing generators "
            "with energy capacity (MWh) values and some where the energy capacity is "
            "missing. You have not included these technologies in the settings parameter "
            "'energy_storage_duration', which is used to fill missing energy capacity "
            "data.\n\nNOTE: This is not a comprehensive list of technologies that *should* "
            "be included in 'energy_storage_duration'. Technologies without any existing "
            "energy capacity data might also be missing."
        )
    for tech, val in energy_storage_duration.items():
        if isinstance(val, dict):
            tech_regions = val.keys()
            df.loc[df[tech_col].isna(), "technology_description"] = (
                "missing_tech_description"
            )
            model_tech_regions = df.loc[
                snake_case_col(df[tech_col]).str.contains(snake_case_str(tech)),
                region_col,
            ].to_list()
            if not all(r in tech_regions for r in model_tech_regions):
                missing_regions = [
                    r for r in model_tech_regions if r not in tech_regions
                ]
                logger.warning(
                    f"The regions {missing_regions} are missing from technology {tech} "
                    "in the settings parameter 'energy_storage_duration'. This technology "
                    "will not have any energy storage capacity in these regions."
                )
            for region, v in val.items():
                if region not in all_regions:
                    logger.warning(
                        f"The settings parameter 'energy_storage_duration', technology '{tech}' "
                        f"has the region '{region}', which is not one of your model regions."
                    )
                df.loc[
                    (snake_case_col(df[tech_col]).str.contains(snake_case_str(tech)))
                    & (df[region_col] == region)
                    & ~(df[energy_col] > 0),
                    energy_col,
                ] = (
                    df[cap_col] * v
                )
        else:
            df.loc[
                snake_case_col(df[tech_col]).str.contains(snake_case_str(tech))
                & ~(df[energy_col] > 0),
                energy_col,
            ] = (
                df[cap_col] * val
            )
    df[energy_col] = df[energy_col].fillna(0)
    return df


# def load_plants_860(
#     pudl_engine: sqlalchemy.engine.Engine, data_years: List[int] = [2020]
# ) -> pd.DataFrame:
#     """Load database table with EIA860 information on plants

#     Parameters
#     ----------
#     pudl_engine : sqlalchemy.engine.Engine
#         Connection to PUDL database
#     data_years : List[int], optional
#         Year of data to keep, by default [2020]

#     Returns
#     -------
#     pd.DataFrame
#         Includes all columns from the database table
#     """
#     data_years = [str(y) for y in data_years]
#     s = f"""
#     SELECT * from plants_eia860
#     WHERE strftime('%Y',report_date) in ({','.join(['?']*len(data_years))})
#     """
#     plants = pd.read_sql_query(
#         s, pudl_engine, params=data_years, parse_dates=["report_date"]
#     )

#     return plants


def load_demand_response_efs_profile(
    resource: str,
    electrification_stock_fn: str,
    model_year: int,
    electrification_scenario: str,
    model_regions: list,
    region_aggregations: dict = {},
    path_in: Path = None,
    utc_offset: int = None,
) -> pd.DataFrame:
    """Load the demand profile of a single flexible resource in all model regions.

    Parameters
    ----------
    resource : str
        Name of the flexible resource.
    electrification_stock_fn : str
        Name of the data file with stock values for each year.
    model_year : int
        Planning period or model year. Used to select stock values for flexible resources
        and their demand profiles.
    electrification_scenario : str
        Name of a scenario from the stock data file.
    model_regions : list
        Names of the model regions, including those that are aggregated from multiple
        base regions.
    region_aggregations : dict, optional
        A list of base regions for each aggregated model region, by default {}. For
        example, {"CA_N": ["WEC_BANC", "WEC_CALN"]}.
    path_in : Path, optional
        Folder where stock and incremental factor (profile) data are located, by default
        None.
    utc_offset: int, optional
        Number of hours that should be shifted from the default UTC time that data are
        stored in.

    Returns
    -------
    pd.DataFrame
        Flexible demand profiles for the selected resource in each model region. Column
        names are model regions.
    Raises
    ------
    KeyError
        The resource is not a valid name.
    """
    from powergenome.load_construction import electrification_profiles

    keep_regions, region_agg_map = regions_to_keep(model_regions, region_aggregations)
    elec_profiles = electrification_profiles(
        stock_fn=electrification_stock_fn,
        year=model_year,
        elec_scenario=electrification_scenario,
        regions=keep_regions,
        utc_offset=utc_offset or 0,
        path_in=path_in,
    )

    if resource not in elec_profiles.resource.unique():
        raise KeyError(
            f"No profile was available for the flexible resource '{resource}' specified "
            "in your settings file under 'flexible_demand_resources'. Available "
            f"resources include: {list(elec_profiles.resource.unique())}."
        )
    dr_profile = elec_profiles.loc[
        elec_profiles["resource"] == resource, ["time_index", "region", "load_mw"]
    ].pivot(index="time_index", columns="region")
    dr_profile.columns = dr_profile.columns.droplevel()
    for model_region, base_regs in region_aggregations.items():
        base_regs = [r for r in base_regs if r != model_region]
        dr_profile[model_region] = dr_profile.reindex(
            columns=base_regs, fill_value=0
        ).sum(axis=1)
        dr_profile = dr_profile.drop(columns=base_regs, errors="ignore")

    return dr_profile


# def add_860m_storage_mwh(
#     gen_df: pd.DataFrame, operating_860m: pd.DataFrame, storage_techs: List[str] = None
# ) -> pd.DataFrame:
#     idx_cols = ["plant_id_eia", "generator_id"]
#     gen_df = gen_df.set_index(idx_cols)

#     if not storage_techs:
#         storage_techs = [
#             "Batteries",
#             "Natural Gas with Compressed Air Storage",
#             "Flywheels",
#         ]

#     storage_860m = operating_860m.loc[
#         operating_860m.technology_description.isin(storage_techs), :
#     ].set_index(idx_cols)
#     gen_df.loc[storage_860m.index, "capacity_mwh"] = storage_860m["capacity_mwh"]
#     # gen_df = pd.merge(
#     #     gen_df, storage_860m[idx_cols + ["capacity_mwh"]], how="left", on=idx_cols
#     # )

#     return gen_df.reset_index()


def fill_num_regional_clusters(
    num_clusters: Dict[str, int],
    model_regions: List[str],
    alt_num_clusters: Dict[str, Dict[str, int]] = None,
) -> Dict[str, Dict[str, int]]:
    """
    Use main num_clusters to create a version with number of existing resource clusters
    for each model region.

    Parameters
    ----------
    num_clusters : Dict[str, int]
        Number of clusters for each technology type across all regions
    model_regions : List[str]
        List of model regions
    alt_num_clusters : Dict[str, Dict[str, int]], optional
        Number of clusters for specific technologies in each region, if different from
        those in num_clusters, by default None

    Returns
    -------
    Dict[str, Dict[str, int]]
        Number of existing resource clusters in each model region for every technology.
    """

    _num_clusters = {}
    for region in model_regions:
        _num_clusters[region] = num_clusters.copy()

    if alt_num_clusters:
        for region in alt_num_clusters:
            for tech, cluster_size in alt_num_clusters[region].items():
                _num_clusters[region][tech] = cluster_size

    return _num_clusters


def label_retired_gens(
    gen_df: pd.DataFrame, start_year: int, end_year: int
) -> pd.DataFrame:
    """
    Flag generators as operating or retired within a specified period. Missing
    "operating_year" values will be filled with 1900.

    Parameters
    ----------
    gen_df : pandas.DataFrame
        DataFrame of generator data. Must contain columns
        'operating_year' and 'retirement_year'.
    start_year : int
        Start of the retirement period (inclusive).
    end_year : int
        End of the period for determining current operation and
        retirement status.

    Returns
    -------
    pandas.DataFrame
        The same DataFrame, augmented with two boolean columns:

        - operating
            True for generators operating at end_year, i.e.
            retirement_year > end_year and operating_year ≤ end_year.
        - period_retired
            True for generators that retired during the period
            [start_year, end_year], i.e. retirement_year ≥ start_year
            and retirement_year ≤ end_year.
    """
    gen_df["operating"] = False
    gen_df["period_retired"] = False

    # Fill missing operating year with 1900
    gen_df["operating_year"] = gen_df["operating_year"].fillna(1900)

    gen_df.loc[
        (gen_df["retirement_year"] > end_year) & (gen_df["operating_year"] <= end_year),
        "operating",
    ] = True
    gen_df.loc[
        ~(gen_df["retirement_year"] > end_year)
        & (gen_df["retirement_year"] >= start_year),
        "period_retired",
    ] = True

    return gen_df


def create_resource_label(
    *series: Union[pd.Series, Sequence], sep: str = "_"
) -> pd.Series:
    """
    Concatenate any number of Series (or array-like) elementwise with `sep` between.

    Parameters
    ----------
    *series : pd.Series or array-like
        The columns to join. They'll all be cast to str.
    sep : str, default "_"
        The separator to insert between each element.

    Returns
    -------
    pd.Series
        A new Series where each value is the sep-joined string of the inputs.
    """
    lengths = []
    for seq in series:
        try:
            lengths.append(len(seq))
        except TypeError:
            raise ValueError(
                f"All inputs must be sequence-like with length; got {type(seq)}"
            )
    if len(lengths) > 1 and len({*lengths}) != 1:
        raise ValueError(
            f"All inputs to create_resource_label must have the same length, got lengths {lengths}"
        )
    # turn everything into Series[str] and align indexes
    strs = [pd.Series(s).astype(str) for s in series]
    # reduce via str.cat
    return reduce(lambda a, b: a.str.cat(b, sep=sep), strs)


def cluster_existing_generators(
    gen_df: pd.DataFrame,
    num_clusters: Dict[str, Dict[str, int]],
    cluster_cols=["heat_rate_mmbtu_mwh", "fom_per_mwyr"],
    extra_outputs_path: Path = None,
) -> Tuple[pd.DataFrame]:
    """
    Cluster existing generators within each region based on some characteristics.

    Parameters
    ----------
    gen_df : pd.DataFrame
        Generators with columns "model_region", "technology", "capacity_mw", "capacity_mwh",
        "heat_rate_mmbtu_mwh", "fom_per_mwyr", and "vom_per_mwh".
    num_clusters : Dict[str, Dict[str, int]]
        Number of clusters. First level keys are regions, second level keys are technologies.
    cluster_cols : list, optional
        Values to calculate clusters on, by default ["heat_rate_mmbtu_mwh", "fom_per_mwyr"]
    extra_outputs_path : Path, optional
        Location to save data in individual generators and their cluster assignment, by
        default None

    Returns
    -------
    Tuple[pd.DataFrame]
        Both the clustered results and the individual generator dataframes.
    """
    logger.debug("Starting existing generator clustering")

    gen_list = []
    cluster_list = []
    keep_cols = [
        "capacity_mw",
        "capacity_mwh",
        "heat_rate_mmbtu_mwh",
        "fom_per_mwyr",
        "vom_per_mwh",
    ]
    for _, df in gen_df.groupby(["model_region", "technology"]):
        region, tech = _
        n_clusters = num_clusters.get(region, {}).get(tech, 0)
        if n_clusters == 0 or df.empty:
            cap = df.capacity_mw.sum()
            logger.debug(f"Not including existing {tech} in {region} ({cap} MW)")
            continue

        if n_clusters is None or n_clusters == len(df):
            df["cluster"] = np.arange(len(df)) + 1

            # cluster_list.append(df[keep_cols])

        elif n_clusters > 0:

            if len(df) < num_clusters[region][tech]:
                s = f"""
    The technology {tech} in region {region} has only {len(df)} operating units,
    which is less than the {n_clusters} clusters you specified.
    The number of clusters has been set equal to the number of units.
                            """
                logger.info(s)
                n_clusters = len(df)

            _cluster_cols = check_cluster_cols(df, cluster_cols)
            clusters = cluster.KMeans(n_clusters=n_clusters, random_state=6).fit(
                preprocessing.StandardScaler().fit_transform(df[_cluster_cols])
            )

            df["cluster"] = clusters.labels_ + 1

        df["Resource"] = create_resource_label(
            df["model_region"], snake_case_col(df["technology"]), df["cluster"], sep="_"
        )
        gen_list.append(df)

        _df = calc_unit_cluster_values(df, "capacity_mw")
        _df["region"] = region
        _df["technology"] = tech
        cluster_list.append(_df)

    results = pd.concat(cluster_list)
    all_gens = pd.concat(gen_list)

    if extra_outputs_path:
        all_gens.to_csv(extra_outputs_path / "existing_gen_units.csv", index=False)

    results["cap_size"] = results["capacity_mw"] / results["num_units"]

    results["Resource"] = create_resource_label(
        results["region"],
        snake_case_col(results["technology"]),
        results["cluster"],
        sep="_",
    )

    return results, all_gens


def check_cluster_cols(df: pd.DataFrame, cluster_cols: List[str]) -> List[str]:
    """
    Validate cluster columns for missing values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the cluster columns.
    cluster_cols : List[str]
        List of column names to check.

    Returns
    -------
    List[str]
        Filtered list of column names with valid data.

    Raises
    ------
    KeyError
        If a specified column does not exist in df.
    ValueError
        If a column has some but not all values missing, or if all specified columns are entirely missing.
    """
    valid_cols = list(cluster_cols)
    if (df["operating"] == False).all():
        return valid_cols
    for col in cluster_cols:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in existing generators table")
        # Find rows where the column is NaN for operating generators
        col_op_na = df.loc[df["operating"] == True, col].isna()
        if col_op_na.all():
            # Drop columns that are entirely missing
            valid_cols.remove(col)
        elif col_op_na.any():
            # Some but not all values are missing
            col_op_na = col_op_na.reindex(
                df.index, fill_value=False
            )  # Ensure alignment
            missing_plant_ids = df.loc[col_op_na, "plant_id"].unique()
            raise ValueError(
                f"Column '{col}' of the existing generators table contains some missing "
                f"values for plants {missing_plant_ids}."
            )
    if not valid_cols:
        raise ValueError(
            f"All cluster columns {cluster_cols} are entirely missing from the existing generators table"
        )
    return valid_cols


def add_gen_age_column(
    gen_df: pd.DataFrame, planning_year: int, age_col: str = "age"
) -> pd.DataFrame:
    """
    Add a column with the age of each generator in years.

    Parameters
    ----------
    gen_df : pd.DataFrame
        DataFrame containing generator data with an 'operating_year' column.
    planning_year : int, optional
        The planning year to calculate the age from.
    age_col : str, optional
        Name of the column to store the age in, by default "age".

    Returns
    -------
    pd.DataFrame
        The modified DataFrame with an additional 'age' column.
    """
    gen_df[age_col] = planning_year - gen_df["operating_year"]
    return gen_df


def apply_custom_gen_formula(
    gen_df: pd.DataFrame, formula_dict: Dict[str, List[dict]]
) -> pd.DataFrame:
    """
    Apply a custom modifier formula to costs or other attributes of specific generator
    technologies.

    Parameters
    ----------
    gen_df : pd.DataFrame
        DataFrame containing generator data with a technology column the attribute of each
        generator.
    formula_dict : Dict[str, List[dict]]
        A dictionary where keys are technology names and values are lists of dictionaries
        with 'attribute' and 'factor' keys. The 'attribute' is the column to modify, and
        'factor' is the formula to apply.

    Returns
    -------
    pd.DataFrame
        The modified DataFrame with the specified attributes adjusted according to the
        provided formulas.

    Raises
    ------
    KeyError
        If a technology in the formula_dict is not found in the generator DataFrame.
        If an attribute specified in the formula is not found in the generator DataFrame.
        If a formula does not contain 'attribute' or 'formula' keys.

    Example dictionary:
    formula_dict = {
        "coal": [
            {
                "attribute": "fom_per_mwyr",
                "formula": {
                    "op": "add", # Use "replace" to ignore existing values
                    "rate": 126,
                    "multiplier": "age"}
            },
        ]
    }
    """
    allowed_operators = ["add", "mul", "truediv", "sub"]
    ignored = r"_"
    technology = gen_df["technology"].str.replace(ignored, "")
    for tech, modifiers in formula_dict.items():

        tech = re.sub(ignored, "", tech)
        mask = technology.str.contains(rf"{tech}", case=False)
        for modifier in modifiers:
            if "attribute" not in modifier or "formula" not in modifier:
                raise KeyError(
                    f"Formula for technology '{tech}' is missing 'attribute' or 'factor' keys."
                )
            attr = modifier["attribute"]
            f = modifier["formula"]
            op = f["op"]
            multiplier = f["multiplier"]
            if attr not in gen_df.columns:
                raise KeyError(
                    f"Attribute '{attr}' not found in generator DataFrame columns."
                )
            if op == "replace":
                gen_df.loc[mask, attr] = f["rate"] * gen_df.loc[mask, multiplier]
            else:
                assert op in allowed_operators
                _f = operator.attrgetter(op)
                gen_df.loc[mask, attr] = _f(operator)(
                    gen_df.loc[mask, attr],
                    f["rate"] * gen_df.loc[mask, multiplier],
                )

    return gen_df


class GeneratorClusters:
    """
    This class is used to determine genererating units that will likely be operating
    in a given year, clusters them according to parameters for the settings file,
    and determines the average operating characteristics of each cluster. Structuring
    this as a class isn't strictly necessary but makes it easier to access generator
    data part-way through the process.
    """

    def __init__(
        self,
        settings,
        current_gens=True,
        supplement_with_860m=True,
        sort_gens=False,
        plant_region_map_table="plant_region_map_epaipm",
        settings_agg_key="region_aggregations",
        multi_period=False,
        include_retired_cap=False,
    ):
        """

        Parameters
        ----------
        pudl_engine : sqlalchemy.Engine
            A sqlalchemy connection for use by pandas
        pudl_out : pudl.PudlTabl
            A PudlTabl object for loading pre-calculated PUDL analysis data
        settings : dictionary
            The dictionary of settings with a dictionary of region aggregations
        """
        # TODO: #404 Update GeneratorClusters init docstring
        self.tech_groups = settings.get("tech_groups", {}) or {}
        self.regional_tech_no_grouping = settings.get("regional_no_grouping", {}) or {}
        self.settings = settings
        self.current_gens = current_gens
        self.sort_gens = sort_gens
        self.weighted_unit_hr = None
        self.supplement_with_860m = supplement_with_860m
        self.multi_period = multi_period
        self.include_retired_cap = include_retired_cap
        self.cluster_builder = build_resource_clusters(
            self.settings.get("RESOURCE_GROUPS"),
            self.settings.get("RESOURCE_GROUP_PROFILES"),
        )

        self.fuel_prices = fetch_fuel_prices(
            settings=self.settings,
        ).pipe(
            modify_fuel_prices,
            self.settings.get("fuel_region_map"),
            self.settings.get("regional_fuel_adjustments"),
        )
        # if resource_heat_rate_table:

        # self.coal_fgd = pd.read_csv(DATA_PATHS["coal_fgd"])

    # def fill_na_heat_rates(self, s):
    #     """Fill null heat rate values with the median of the series. Not many null
    #     values are expected.

    #     Parameters
    #     ----------
    #     df : DataFrame
    #         Must contain the column 'heat_rate_mmbtu_mwh'

    #     Returns
    #     -------
    #     Dataframe
    #         Same as input but with any null values replaced by the median.
    #     """
    #     if s.isnull().any():
    #         median_hr = s.median()
    #         return s.fillna(median_hr)
    #     else:
    #         return s
    #     # median_hr = df["heat_rate_mmbtu_mwh"].median()
    #     # df["heat_rate_mmbtu_mwh"].fillna(median_hr, inplace=True)

    #     # return df

    # def remove_860_duplicates(self, df_860: pd.DataFrame) -> pd.DataFrame:
    #     """
    #     Remove rows from an EIA 860 DF that contain a duplicate plant-generator pairing.
    #     """
    #     df = df_860.copy(deep=True)
    #     df = df.set_index(["plant_id_eia", "generator_id"])
    #     df = df.loc[~df.index.duplicated(), :].reset_index()
    #     return df

    def create_demand_response_gen_rows(self):
        """Create rows for demand response/management resources to include in the
        generators file.

        Returns
        -------
        DataFrame
            One row for each region/DSM resource with values in all columns filled.
        """
        logger.info("Creating flexible demand resources")
        year = self.settings["model_year"]
        df_list = []
        self.demand_response_profiles = {}

        if not self.settings.get("flexible_demand_resources"):
            logger.warning(
                "A demand response file is included in extra inputs but the parameter "
                "`flexible_demand_resources` is not in the settings file. No demand "
                "response resources will be included with the generators."
            )
            return pd.DataFrame()
        if year not in self.settings["flexible_demand_resources"].keys():
            logger.warning(
                f"The model year {year} is not included in your 'flexible_demand_resources' "
                "parameter. No flexible demand resources will be included for this "
                "planning period."
            )
        if self.settings["flexible_demand_resources"][year] is None:
            logger.warning(
                "Your 'flexible_demand_resources' settings parameter has the value 'None' "
                "for this planning period. No flexible demand resources will be included."
            )
        for resource, parameters in (
            self.settings["flexible_demand_resources"].get(year, {}) or {}
        ).items():
            _df = pd.DataFrame(
                index=self.settings["model_regions"],
                columns=list(self.settings["generator_columns"]) + ["profile"],
            )
            _df = _df.drop(columns="Resource")
            _df["technology"] = resource
            _df["region"] = self.settings["model_regions"]
            if self.settings.get("demand_response_fn"):
                dr_path = (
                    Path.cwd()
                    / self.settings["input_folder"]
                    / self.settings["demand_response_fn"]
                )
                dr_profile = make_demand_response_profiles(
                    dr_path,
                    resource,
                    self.settings["model_year"],
                    self.settings["demand_response"],
                )
            elif self.settings.get("electrification_stock_fn"):
                if not self.settings.get("electrification_scenario"):
                    logger.warning(
                        "You have provided a parameter value for 'electrification_stock_fn' "
                        "but not 'electrification_scenario'. No flexible demand resources "
                        "can be included without a valid electrification scenario."
                    )
                    pass
                keep_regions, region_agg_map = regions_to_keep(
                    self.settings["model_regions"],
                    self.settings.get("region_aggregations", {}) or {},
                )
                dr_profile = load_demand_response_efs_profile(
                    resource,
                    self.settings.get("electrification_stock_fn"),
                    self.settings["model_year"],
                    self.settings.get("electrification_scenario"),
                    keep_regions,
                    self.settings.get("region_aggregations", {}) or {},
                    self.settings.get("EFS_DATA"),
                    self.settings.get("utc_offset"),
                )
            self.demand_response_profiles[resource] = dr_profile
            # Add hourly profile to demand response rows
            dr_cf = dr_profile / dr_profile.max()
            dr_regions = [r for r in dr_cf.columns if r in _df.index]
            _df = _df.loc[dr_regions, :]
            _df["profile"] = list(dr_cf[dr_regions].values.T)

            dr_capacity = demand_response_resource_capacity(
                dr_profile, resource, self.settings
            )

            # This is to solve a bug with only one region. Need to come back and solve
            # in a better fashion.
            if len(dr_capacity) > 1:
                dr_capacity_scenario = dr_capacity.squeeze()
            else:
                dr_capacity_scenario = dr_capacity
            _df["Existing_Cap_MW"] = _df["region"].map(dr_capacity_scenario)

            if not parameters.get("parameter_values"):
                logger.warning(
                    "No model parameter values are provided in the settings file for "
                    f"the flexible demand resource '{resource}'. If another resource"
                    " has values under "
                    "`flexible_demand_resource.<year>.<resource_type>.parameter_values`, "
                    f"those columns will have a value of 0 for '{resource}'."
                )
            for col, value in parameters.get("parameter_values", {}).items():
                _df[col] = value

            df_list.append(_df)

        dr_rows = pd.concat(df_list)
        dr_rows["New_Build"] = -1
        dr_rows["Fuel"] = "No_fuel"
        dr_rows["cluster"] = 1
        dr_rows = dr_rows.fillna(0)

        return dr_rows

    def create_region_technology_clusters(self, return_retirement_capacity=False):
        """
        Calculation of average unit characteristics within a technology cluster
        (capacity, minimum load, heat rate) and the number of units in the cluster.

        Parameters
        ----------
        plant_region_map_table : str, optional
            Name of the table with region names for each plant, by default
            "plant_region_map_epaipm"
        settings_agg_key : str, optional
            Name of the settings dictionary key with regional aggregations, by default
            "region_aggregations"
        return_retirement_capacity : bool, optional
            If retired generators should be retured as a second dataframe, by default
            False

        Returns
        -------
        dataframe

        """

        keep_regions, region_agg_map = regions_to_keep(
            self.settings["model_regions"], self.settings.get("region_aggregations")
        )
        num_clusters = fill_num_regional_clusters(
            self.settings.get("num_clusters", {}),
            self.settings["model_regions"],
            self.settings.get("alt_num_clusters", {}),
        )
        gen_df = (
            get_data("generation")
            .pipe(add_gen_age_column, self.settings["model_year"])
            .pipe(
                apply_custom_gen_formula,
                self.settings.get("resource_attr_modifiers", {}),
            )
        )
        plant_region_map = get_data("plant_region")
        gen_df = pd.merge(gen_df, plant_region_map, on="plant_id")
        self.results, self.all_gens = (
            map_agg_region_names(
                gen_df,
                region_agg_map,
            )
            .pipe(
                label_retired_gens,
                self.settings["model_first_planning_year"],
                self.settings["model_year"],
            )
            .pipe(group_technologies, self.tech_groups, self.regional_tech_no_grouping)
            .pipe(
                cluster_existing_generators,
                num_clusters,
                self.settings.get(
                    "generator_cluster_columns", ["heat_rate_mmbtu_mwh", "fom_per_mwyr"]
                ),
                self.settings.get("extra_outputs_path"),
            )
        )

        if self.settings.get("region_wind_pv_cap_fn"):
            from powergenome.external_data import overwrite_wind_pv_capacity

            logger.debug("Setting existing wind/pv using external file")
            self.results = overwrite_wind_pv_capacity(self.results, self.settings)

        if self.settings.get("dg_as_resource"):
            logger.debug(
                "\n **************** \nDistributed generation is being added as generating"
                " resources. The capacity of DG in each region is increased by "
                f"{self.settings.get('avg_distribution_loss', 0):%} to account for no "
                "distribution losses.\n"
            )
            self.results = add_dg_resources(self.settings, self.results)
        else:
            self.results["profile"] = None

        # Add fixed/variable O&M based on NREL atb
        self.results = (
            self.results.pipe(startup_fuel, self.settings)
            .pipe(add_fuel_labels, self.fuel_prices, self.settings)
            .pipe(startup_nonfuel_costs, self.settings)
            .pipe(
                add_resource_tags,
                model_tag_values=self.settings.get("model_tag_values", {}),
                regional_tag_values=self.settings.get("regional_tag_values", {}),
                model_tag_names=self.settings.get("model_tag_names", []),
                default_model_tag=self.settings.get("default_model_tag", {}),
            )
        )

        if self.sort_gens:
            logger.debug("Sorting new resources alphabetically.")
            self.results = self.results.sort_values(["region", "technology"])

        if self.results["Resource"].nunique() != len(self.results):
            dup_resources = (
                self.results[self.results["Resource"].duplicated()]
                .drop_duplicates()
                .to_list()
            )
            raise ValueError(
                f"The generator resource names {dup_resources} have duplicates. These "
                "names should be unique. You'll probably need to file an issue for this "
                "at https://github.com/PowerGenome/PowerGenome/issues."
            )

        if self.multi_period:
            retire_cols = [
                "Min_Retired_Cap_MW",
                "Min_Retired_Energy_Cap_MW",
                "Min_Retired_Charge_Cap_MW",
            ]
            for col in retire_cols:
                if col not in self.settings.get("generator_columns", []) and isinstance(
                    self.settings.get("generator_columns"), list
                ):
                    self.settings["generator_columns"].append(col)

            if self.include_retired_cap:
                # Add mimimum retirement amounts
                cap_retired = cap_retire_within_period(
                    self.all_gens,
                    self.settings["model_first_planning_year"],
                    self.settings["model_year"],
                    "capacity_mw",
                )
                self.results = pd.merge(
                    self.results, cap_retired, on="Resource", how="left", validate="1:1"
                )
                self.results[retire_cols].fillna(0, inplace=True)
            else:
                self.results[retire_cols] = 0

        # Add variable resource profiles
        self.results = self.results.reset_index(drop=True)
        for i, row in enumerate(self.results.itertuples()):
            params = map_eia_technology(row.technology)
            if not params:
                # EIA technology not supported
                continue
            params.update({"existing": True})
            groups = self.cluster_builder.find_groups(**params)
            if not groups:
                # No matching resource groups
                continue
            if len(groups) > 1:
                # Multiple matching resource groups
                raise ValueError(
                    "Multiple existing resource groups match EIA technology"
                    + row.technology
                )
            group = groups[0]
            if group.profiles is None:
                # Resource group has no profiles
                continue
            if row.region in (self.settings.get("region_aggregations", {}) or {}):
                regions = self.settings.get("region_aggregations", {})[row.region]
            else:
                regions = [row.region]
            metadata = group.metadata.read().rename(
                columns={"ipm_region": "region"}, errors="ignore"
            )
            if not metadata["region"].isin(regions).any():
                # Resource group has no resources in selected IPM regions
                continue
            clusters = group.get_clusters(
                regions=regions,
                max_clusters=1,
                utc_offset=self.settings.get("utc_offset", 0),
            )
            self.results["profile"][i] = clusters["profile"][0]

        self.results = rename_gen_cols(self.results)

        # Drop old index cols from df
        self.results.drop(columns=["level_0", "index"], errors="ignore", inplace=True)

        logger.info("Finished creating existing generator clusters")

        return self.results

    def create_new_generators(self):
        logger.info("Starting to build new generation resources")
        # self.offshore_spur_costs = fetch_atb_offshore_spur_costs(
        #     self.data_location, self.settings
        # )
        self.resource_hr = fetch_heat_rates(
            self.settings.get("resource_data_year"),
        )
        self.resource_costs = fetch_resource_costs(
            self.settings,
            self.settings.get("resource_data_year"),
        )

        self.new_generators = atb_new_generators(
            self.resource_costs, self.resource_hr, self.settings, self.cluster_builder
        )

        if not self.new_generators.empty:
            self.new_generators = (
                self.new_generators.pipe(startup_fuel, self.settings)
                .pipe(add_fuel_labels, self.fuel_prices, self.settings)
                .pipe(startup_nonfuel_costs, self.settings)
            )

            if self.sort_gens:
                logger.debug("Sorting new resources alphabetically.")
                self.new_generators = self.new_generators.sort_values(
                    ["region", "technology"]
                )

            if self.settings.get("capacity_limit_spur_fn"):
                self.new_generators = self.new_generators.pipe(
                    add_resource_max_cap_spur, self.settings
                )
            else:
                logger.warning("No settings parameter for max capacity/spur file")
            self.new_generators = self.new_generators.pipe(
                calculate_transmission_inv_cost,
                self.settings,
                # None or self.offshore_spur_costs,
            ).pipe(add_transmission_inv_cost, self.settings)

        if self.settings.get("demand_response_fn") or self.settings.get(
            "electrification_stock_fn"
        ):
            dr_rows = self.create_demand_response_gen_rows()
            self.new_generators = pd.concat(
                [self.new_generators, dr_rows], sort=False, ignore_index=True
            )
        self.new_generators = add_resource_tags(
            self.new_generators,
            model_tag_values=self.settings.get("model_tag_values", {}),
            regional_tag_values=self.settings.get("regional_tag_values", {}),
            model_tag_names=self.settings.get("model_tag_names", []),
            default_model_tag=self.settings.get("default_model_tag", {}),
        )
        if "cluster" not in self.new_generators.columns:
            self.new_generators["cluster"] = 1
        self.new_generators["cluster"] = self.new_generators["cluster"].astype(
            "Int64", errors="ignore"
        )
        self.new_generators["Resource"] = (
            self.new_generators["region"]
            + "_"
            + snake_case_col(self.new_generators["technology"])
            + "_"
            + self.new_generators["cluster"].astype(str)
        )

        logger.info("Finished creating new generation resources")

        return self.new_generators

    def adjust_min_power_based_on_profile(self):
        """
        Adjust 'Min_Power' by ensuring it is not greater than the minimum value
        in the corresponding 'profile' column (if 'profile' contains an array).
        Uses np.frompyfunc for improved performance on large datasets.
        """
        # Vectorized function to extract the minimum value from arrays
        get_min_value = np.frompyfunc(
            lambda x: min(x) if isinstance(x, (list, np.ndarray)) else np.nan, 1, 1
        )

        # Compute the minimum values from the 'profile' column
        self.all_resources["profile_min"] = get_min_value(self.all_resources["profile"])

        # Update 'Min_Power' where necessary
        self.all_resources.loc[
            (self.all_resources["profile_min"].notna())
            & (self.all_resources["Min_Power"] > self.all_resources["profile_min"]),
            "Min_Power",
        ] = self.all_resources["profile_min"]

        # Drop the temporary column
        self.all_resources.drop(columns=["profile_min"], inplace=True)

    def remove_fuel_scenario_name(self, gen_fuels: List[str]):
        """
        Keeps fuels for used in by generators in the model. Removes the scenario name
        from the `full_fuel_name` column of of the `fuel_prices` DataFrame.
        """
        self.fuel_prices = self.fuel_prices.loc[
            self.fuel_prices["full_fuel_name"].isin(gen_fuels)
        ]
        scenarios = (self.settings.get("eia_series_scenario_names", {}) or {}).keys()
        for s in scenarios:
            self.fuel_prices["full_fuel_name"] = self.fuel_prices[
                "full_fuel_name"
            ].str.replace(f"_{s}", "")

    def apply_multi_period_transformations(self):
        """
        Applies transformations to self.all_resources, renaming columns and modifying
        specific values to ensure compatibility with multi-period analysis.
        """
        if self.all_resources is not None:
            self.all_resources = self.all_resources.rename(
                columns={
                    "cap_recovery_years": "Capital_Recovery_Period",
                    "wacc_real": "WACC",
                }
            )
            self.all_resources["Lifetime"] = self.all_resources[
                "Capital_Recovery_Period"
            ]
            self.all_resources.loc[
                (self.all_resources["Lifetime"] == 0)
                | (self.all_resources["Lifetime"].isna()),
                "Lifetime",
            ] = 50

            # Apply transformations
            self.all_resources = (
                remove_fuel_gen_scenario_name(self.all_resources, self.settings)
                .pipe(set_int_cols)
                .pipe(round_col_values)
                .pipe(check_resource_tags)
            )

    def create_all_generators(self):
        if self.current_gens:
            self.existing_resources = self.create_region_technology_clusters()

        self.new_resources = self.create_new_generators()

        self.all_resources = pd.concat(
            [self.existing_resources, self.new_resources], ignore_index=True, sort=False
        )

        # Add CO2 pipeline and disposal costs from file
        if self.settings.get("co2_pipeline_filters") and self.settings.get(
            "co2_pipeline_cost_fn"
        ):
            self.all_resources = merge_co2_pipeline_costs(
                df=self.all_resources,
                co2_data_path=self.settings["input_folder"]
                / self.settings.get("co2_pipeline_cost_fn"),
                co2_pipeline_filters=self.settings["co2_pipeline_filters"],
                region_aggregations=self.settings.get("region_aggregations"),
                fuel_emission_factors=self.settings["fuel_emission_factors"],
                target_usd_year=self.settings.get("target_usd_year"),
                extra_ccs_cost_tonne=self.settings.get("ccs_disposal_cost"),
                settings=self.settings,
            )

        self.all_resources = self.all_resources.round(3)
        self.all_resources["Cap_Size"] = self.all_resources["Cap_Size"]
        self.all_resources["Heat_Rate_MMBTU_per_MWh"] = self.all_resources[
            "Heat_Rate_MMBTU_per_MWh"
        ]

        self.all_resources = self.all_resources.reset_index(drop=True)
        self.all_resources["variable_CF"] = 0.0
        for i, p in enumerate(self.all_resources["profile"]):
            if isinstance(p, (collections.abc.Sequence, np.ndarray)):
                self.all_resources.loc[i, "variable_CF"] = np.mean(p)

        # Set Min_Power of wind/solar to 0
        if "VRE" in self.all_resources.columns:
            self.all_resources.loc[self.all_resources["VRE"] == 1, "Min_Power"] = 0

        self.all_resources["R_ID"] = np.arange(len(self.all_resources)) + 1

        if self.current_gens:
            logger.debug(
                f"Capacity of {self.all_resources['Existing_Cap_MW'].sum()} MW in final clusters"
            )

        self.all_resources = (
            add_misc_gen_values(
                self.all_resources,
                # self.settings,
            )
            .pipe(
                hydro_energy_to_power,
                self.settings.get("hydro_factor"),
                self.settings.get("regional_hydro_factor", {}),
            )
            .pipe(add_co2_costs_to_o_m)
        )

        self.remove_fuel_scenario_name(self.all_resources["Fuel"].to_list())
        self.adjust_min_power_based_on_profile()
        self.apply_multi_period_transformations()

        # Fill NaN values with 0
        self.all_resources.fillna(0, inplace=True)

        return self.all_resources
