import collections
import logging
import os
import re
from numbers import Number
from pathlib import Path
from typing import Dict, List, Union
from zipfile import BadZipFile

os.environ["USE_PYGEOS"] = "0"
import geopandas as gpd
import numpy as np
import pandas as pd
import pudl
import requests
import sqlalchemy
from bs4 import BeautifulSoup
from flatten_dict import flatten
from scipy.stats import iqr
from sklearn import cluster, preprocessing
from xlrd import XLRDError

from powergenome.cluster_method import (
    cluster_by_owner,
    cluster_kmeans,
    weighted_ownership_by_unit,
)
from powergenome.co2_pipeline_cost import merge_co2_pipeline_costs
from powergenome.eia_opendata import fetch_fuel_prices, modify_fuel_prices
from powergenome.external_data import (
    add_resource_max_cap_spur,
    demand_response_resource_capacity,
    make_demand_response_profiles,
)
from powergenome.financials import investment_cost_calculator
from powergenome.GenX import cap_retire_within_period, rename_gen_cols
from powergenome.load_profiles import make_distributed_gen_profiles
from powergenome.nrelatb import (
    atb_fixed_var_om_existing,
    atb_new_generators,
    fetch_atb_costs,
    fetch_atb_heat_rates,
    fetch_atb_offshore_spur_costs,
)
from powergenome.params import DATA_PATHS, IPM_GEOJSON_PATH, build_resource_clusters
from powergenome.price_adjustment import inflation_price_adjustment
from powergenome.resource_clusters import map_eia_technology
from powergenome.util import (
    download_save,
    find_region_col,
    load_ipm_shapefile,
    map_agg_region_names,
    regions_to_keep,
    remove_leading_zero,
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


def fill_missing_tech_descriptions(
    df: pd.DataFrame, date_col: str = "report_date"
) -> pd.DataFrame:
    """
    EIA 860 records before 2014 don't have a technology description. If we want to
    include any of this data in the historical record (e.g. heat rates or capacity
    factors) then they need to be filled in.

    Parameters
    ----------
    df : dataframe
        A pandas dataframe with columns plant_id_eia, generator_id, and
        technology_description.
    date_col: str
        The column with date information, used to sort values from oldest to newest.
        Assumes that newer records will have a valid technology description for the
        generator.

    Returns
    -------
    dataframe
        Same data that came in, but with missing technology_description values filled
        in.
    """
    if (
        date_col not in df.columns
        and not df.loc[df["technology_description"].isnull(), :].empty
    ):
        logger.warning(
            "A dataframe with missing technology descriptions does not have the date column "
            f"{date_col}. The rows with missing technology descriptions look like:\n\n"
            f"{df.loc[df['technology_description'].isnull(), :]}\n\n"
        )
    start_len = len(df)
    df = df.sort_values(by=date_col)
    df_list = []
    missing_tech_plants = df.loc[df["technology_description"].isna(), :]
    if missing_tech_plants.empty:
        return df

    df = df.drop(index=missing_tech_plants.index)
    for _, _df in missing_tech_plants.groupby(
        ["plant_id_eia", "generator_id"], as_index=False
    ):
        _df["technology_description"].fillna(method="bfill", inplace=True)
        df_list.append(_df)
    results = pd.concat([df, pd.concat(df_list, ignore_index=True, sort=False)])

    if df.loc[df["technology_description"].isnull(), :].empty is False:
        logger.warning("Failed to fill some technology names.")

    end_len = len(results)
    assert (
        start_len == end_len
    ), "Somehow records were dropped when filling tech_descriptions"
    return results


def group_generators_at_plant(df, by=["plant_id_eia"], agg_fn={"capacity_mw": "sum"}):
    """
    Group generators at a plant. This is a flexible function that lets a user group
    by the desired attributes (e.g. plant id) and perform aggregated operations on each
    group.

    This function also might be a bit unnecessary given how simple it is.

    Parameters
    ----------
    df : dataframe
        Pandas dataframe with information on power plants.
    by : list, optional
        Columns to use for the groupby, by default ["plant_id_eia"]
    agg_fn : dict, optional
        Aggregation function to pass to groupby, by default {"capacity_mw": "sum"}

    Returns
    -------
    dataframe
        The grouped dataframe with aggregation functions applied.
    """

    df_grouped = df.groupby(by, as_index=False).agg(agg_fn)

    return df_grouped


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
                df["technology"].str.contains(tech, case=False),
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
    logger.info("Adding non-fuel startup costs")
    target_usd_year = settings.get("target_usd_year")

    vom_costs = settings.get("startup_vom_costs_mw", {})
    vom_usd_year = settings.get("startup_vom_costs_usd_year")

    if target_usd_year and vom_usd_year:
        logger.info(
            f"Changing non-fuel VOM costs from {vom_usd_year} to " f"{target_usd_year}"
        )
        for key, cost in vom_costs.items():
            vom_costs[key] = inflation_price_adjustment(
                price=cost, base_year=vom_usd_year, target_year=target_usd_year
            )

    startup_type = settings.get("startup_costs_type")
    startup_costs = settings.get(startup_type, {})
    startup_costs_usd_year = settings.get("startup_costs_per_cold_start_usd_year")

    if target_usd_year and startup_costs_usd_year:
        logger.info(
            f"Changing non-fuel startup costs from {vom_usd_year} to {target_usd_year}"
        )
        for key, cost in startup_costs.items():
            startup_costs[key] = inflation_price_adjustment(
                price=cost,
                base_year=startup_costs_usd_year,
                target_year=target_usd_year,
            )

    df["Start_Cost_per_MW"] = 0

    for existing_tech, cost_tech in settings.get(
        "existing_startup_costs_tech_map", {}
    ).items():
        total_startup_costs = vom_costs[cost_tech] + startup_costs[cost_tech]
        df.loc[
            df["technology"].str.contains(existing_tech, case=False),
            "Start_Cost_per_MW",
        ] = total_startup_costs

    for new_tech, cost_tech in settings.get("new_build_startup_costs", {}).items():
        total_startup_costs = vom_costs[cost_tech] + startup_costs[cost_tech]
        df.loc[
            df["technology"].str.contains(new_tech), "Start_Cost_per_MW"
        ] = total_startup_costs
    df.loc[:, "Start_Cost_per_MW"] = df.loc[:, "Start_Cost_per_MW"]

    # df.loc[df["technology"].str.contains("Nuclear"), "Start_Cost_per_MW"] = "FILL VALUE"

    return df


def group_technologies(
    df: pd.DataFrame,
    group_technologies: bool = False,
    tech_groups: Dict[str, list] = {},
    regional_no_grouping: Dict[str, list] = {},
) -> pd.DataFrame:
    """
    Group different technologies together based on parameters in the settings file.
    An example would be to put a bunch of different technologies under the umbrella
    category of "biomass" or "peaker".

    Parameters
    ----------
    df : dataframe
        Pandas dataframe with
    settings : dictionary
        User-defined settings loaded from a YAML file. Must have key tech_groups.

    Returns
    -------
    dataframe
        Same as incoming dataframe but with grouped technology types
    """
    if not group_technologies:
        return df
    else:
        df["_technology"] = df["technology_description"]
        for tech, group in tech_groups.items():
            df.loc[df["technology_description"].isin(group), "_technology"] = tech

        for region, tech_list in regional_no_grouping.items():
            df.loc[
                (df["model_region"] == region)
                & (df["technology_description"].isin(tech_list)),
                "_technology",
            ] = df.loc[
                (df["model_region"] == region)
                & (df["technology_description"].isin(tech_list)),
                "technology_description",
            ]

        df.loc[:, "technology_description"] = df.loc[:, "_technology"]
        df = df.drop(columns=["_technology"])

        return df


def label_hydro_region(gens_860, pudl_engine, model_regions_gdf):
    """
    Label hydro facilities that don't have a region by default.

    Parameters
    ----------
    gens_860 : dataframe
        Infomation on all generators from PUDL
    pudl_engine : sqlalchemy.Engine
        A sqlalchemy connection for use by pandas
    model_regions_gdf : dataframe
        Geodataframe of the model regions

    Returns
    -------
    dataframe
        Plant id and region for any hydro that didn't originally have a region label.
    """

    plant_entity = pd.read_sql_table("plants_entity_eia", pudl_engine)

    model_hydro = gens_860.loc[
        gens_860["technology_description"] == "Conventional Hydroelectric"
    ].merge(plant_entity[["plant_id_eia", "latitude", "longitude"]], on="plant_id_eia")

    no_lat_lon = model_hydro.loc[
        (model_hydro["latitude"].isnull()) | (model_hydro["longitude"].isnull()), :
    ]
    if not no_lat_lon.empty:
        print(no_lat_lon["summer_capacity_mw"].sum(), " MW without lat/lon")
    model_hydro = model_hydro.dropna(subset=["latitude", "longitude"])

    # Convert the lon/lat values to geo points. Need to add an initial CRS and then
    # change it to align with the IPM regions
    model_hydro_gdf = gpd.GeoDataFrame(
        model_hydro,
        geometry=gpd.points_from_xy(model_hydro.longitude, model_hydro.latitude),
        crs="EPSG:4326",
    )

    if model_hydro_gdf.crs != model_regions_gdf.crs:
        model_hydro_gdf = model_hydro_gdf.to_crs(model_regions_gdf.crs)

    model_hydro_gdf = gpd.sjoin(model_regions_gdf, model_hydro_gdf)

    keep_cols = ["plant_id_eia", "region"]
    return model_hydro_gdf.loc[:, keep_cols]


def load_plant_region_map(
    gens_860,
    pudl_engine,
    pg_engine,
    settings,
    model_regions_gdf,
    table="plant_region_map_epaipm",
):
    """
    Load the region that each plant is located in.

    Parameters
    ----------
    pudl_engine : sqlalchemy.Engine
        A sqlalchemy connection for use by pandas
    settings : dictionary
        The dictionary of settings with a dictionary of region aggregations
    table : str, optional
        The SQL table to load, by default "plant_region_map_epaipm"

    Returns
    -------
    dataframe
        A dataframe where each plant has an associated "model_region" mapped
        from the original region labels.
    """
    # Load dataframe of region labels for each EIA plant id
    region_map_df = pd.read_sql_table(table, con=pg_engine)

    if settings.get("plant_region_map_fn"):
        user_region_map_df = pd.read_csv(
            Path(settings["input_folder"]) / settings["plant_region_map_fn"]
        )
        assert (
            "region" in user_region_map_df.columns
        ), f"The column 'region' must appear in {settings['plant_region_map_fn']}"
        assert (
            "plant_id_eia" in user_region_map_df.columns
        ), f"The column 'plant_id_eia' must appear in {settings['plant_region_map_fn']}"

        user_region_map_df = user_region_map_df.set_index("plant_id_eia")

        region_map_df.loc[
            region_map_df["plant_id_eia"].isin(user_region_map_df.index), "region"
        ] = region_map_df["plant_id_eia"].map(user_region_map_df["region"])

    # Label hydro using the IPM shapefile because NEEDS seems to drop some hydro
    all_hydro_regions = label_hydro_region(gens_860, pudl_engine, model_regions_gdf)

    region_map_df = pd.concat(
        [region_map_df, all_hydro_regions], ignore_index=True, sort=False
    ).drop_duplicates(subset=["plant_id_eia"], keep="first")

    # Settings has a dictionary of lists for regional aggregations. Need
    # to reverse this to use in a map method.
    keep_regions, region_agg_map = regions_to_keep(
        settings["model_regions"], settings.get("region_aggregations")
    )

    # Create a new column "model_region" with labels that we're using for aggregated regions

    model_region_map_df = region_map_df.loc[
        region_map_df.region.isin(keep_regions), :
    ].drop(columns="id", errors="ignore")

    model_region_map_df = map_agg_region_names(
        df=model_region_map_df,
        region_agg_map=region_agg_map,
        original_col_name="region",
        new_col_name="model_region",
    )

    # There are some cases of plants with generators assigned to different IPM regions.
    # If regions are aggregated there may be some duplicates in the results.
    model_region_map_df = model_region_map_df.drop_duplicates(
        subset=["plant_id_eia", "model_region"]
    )

    return model_region_map_df


def label_retirement_year(
    df: pd.DataFrame,
    settings: dict,
    age_col: str = "operating_date",
    settings_retirement_table: str = "retirement_ages",
    add_additional_retirements: bool = True,
):
    """
    Add a retirement year column to the dataframe based on the year each generator
    started operating.

    Parameters
    ----------
    df : dataframe
        Dataframe of generators
    settings : dictionary
        The dictionary of settings with a dictionary of generator lifetimes
    age_col : str, optional
        The dataframe column to use when calculating the retirement year, by default
        "operating_date"
    settings_retirement_table : str, optional
        The settings dictionary key for another dictionary of generator retirement
        lifetimes, by default "retirement_ages"
    add_additional_retirements : bool, optional
        Logic to determine if additional retirements from the settings file should
        be checked. For example, this isn't necessary when adding proposed generators
        because we probably won't be setting an artifically early retirement year.
    """
    if age_col not in df.columns:
        age_col = age_col.replace("operating_date", "generator_operating_date")
    if age_col not in df.columns:
        return df
    start_len = len(df)
    retirement_ages = settings.get(settings_retirement_table, {}) or {}
    if "retirement_year" not in df.columns:
        df["retirement_year"] = np.nan

    df["retirement_age"] = df["technology_description"].map(retirement_ages).fillna(500)
    try:
        df.loc[df["retirement_year"].isna(), "retirement_year"] = (
            df.loc[df["retirement_year"].isna(), age_col].dt.year
            + df.loc[df["retirement_year"].isna(), "retirement_age"]
        )
    except AttributeError:
        df.loc[df["retirement_year"].isna(), "retirement_year"] = (
            df.loc[df["retirement_year"].isna(), age_col]
            + df.loc[df["retirement_year"].isna(), "retirement_age"]
        )

    try:
        df.loc[~df["planned_retirement_date"].isnull(), "retirement_year"] = df.loc[
            ~df["planned_retirement_date"].isnull(), "planned_retirement_date"
        ].dt.year
    except KeyError:
        pass

    # Add additonal retirements from settings file
    if settings.get("additional_retirements") and add_additional_retirements:
        logger.info("Changing retirement dates based on settings file")
        model_year = settings["model_year"]
        start_ret_cap = df.loc[
            df["retirement_year"] <= model_year, settings["capacity_col"]
        ].sum()
        logger.info(f"Starting retirement capacity is {start_ret_cap} MW")
        i = 0
        ret_cap = 0
        for record in settings["additional_retirements"]:
            plant_id, gen_id, ret_year = record
            # gen ids are strings, not integers
            gen_id = str(gen_id)

            df.loc[
                (df["plant_id_eia"] == plant_id) & (df["generator_id"] == gen_id),
                "retirement_year",
            ] = ret_year

            i += 1
            ret_cap += df.loc[
                (df["plant_id_eia"] == plant_id) & (df["generator_id"] == gen_id),
                settings["capacity_col"],
            ].sum()

        end_ret_cap = df.loc[
            df["retirement_year"] <= model_year, settings["capacity_col"]
        ].sum()
        logger.info(f"Ending retirement capacity is {end_ret_cap} MW")
        if not end_ret_cap > start_ret_cap:
            logger.debug(
                "Adding retirements from settings didn't change the retiring capacity."
            )
        if end_ret_cap - start_ret_cap != ret_cap:
            logger.debug(
                f"Retirement diff is {end_ret_cap - start_ret_cap}, adding retirements "
                f"yields {ret_cap} MW"
            )
        logger.info(
            f"The retirement year for {i} plants, totaling {ret_cap} MW, was changed "
            "based on settings file parameters"
        )
    else:
        logger.info("No retirement dates changed based on the settings file")

    end_len = len(df)

    assert start_len == end_len

    return df


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
    if not cap_col in df:
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


def load_generator_860_data(pudl_engine, data_years=[2017]):
    """
    Load EIA 860 generator data from the PUDL database

    Parameters
    ----------
    pudl_engine : sqlalchemy.Engine
        A sqlalchemy connection for use by pandas
    data_years : list, optional
        Years of data to load, by default [2017]

    Returns
    -------
    dataframe
        All of the generating units from PUDL
    """
    data_years = [str(y) for y in data_years]
    sql = f"""
        SELECT * FROM generators_eia860
        WHERE operational_status_code NOT IN ('RE', 'OS', 'IP', 'CN')
        AND strftime('%Y',report_date) in ({','.join(['?']*len(data_years))})
    """
    gens_860 = pd.read_sql_query(
        sql=sql,
        con=pudl_engine,
        params=data_years,
        parse_dates=["report_date", "planned_retirement_date"],
    )

    return gens_860


def supplement_generator_860_data(
    gens_860: pd.DataFrame,
    gens_entity: pd.DataFrame,
    bga: pd.DataFrame,
    model_region_map: pd.DataFrame,
    settings: dict,
):
    """
    Load data about each generating unit in the model area.

    Parameters
    ----------
    gens_860 : dataframe
        Information on all generating units for the given data years.
    pudl_engine : sqlalchemy.Engine
        A sqlalchemy connection for use by pandas
    settings : dictionary
        The dictionary of settings with a dictionary of region aggregations
    pudl_out : pudl.PudlTabl
        A PudlTabl object for loading pre-calculated PUDL analysis data
    model_region_map : dataframe
        A dataframe with columns 'plant_id_eia' and 'model_region' (aggregated regions)
    data_years : list, optional
        Years of data to include, by default [2017]

    Returns
    -------
    dataframe
        Data about each generator and generation unit that will be included in the
        model. Columns include:

        ['plant_id_eia', 'generator_id',
       'capacity_mw', 'energy_source_code_1',
       'energy_source_code_2', 'minimum_load_mw', 'operational_status_code',
       'planned_new_capacity_mw', 'switch_oil_gas', 'technology_description',
       'time_cold_shutdown_full_load_code', 'model_region', 'prime_mover_code',
       'operating_date', 'boiler_id', 'unit_id_eia', 'unit_id_pudl', 'unit_id_pg,
       'retirement_year']
    """

    initial_capacity = (
        gens_860.loc[gens_860["plant_id_eia"].isin(model_region_map["plant_id_eia"])]
        .groupby("technology_description")[settings["capacity_col"]]
        .sum()
    )

    # Add pudl unit ids, only include specified data years

    # Combine generator data that can change over time with static entity data
    # and only keep generators that are in a region of interest

    gen_cols = set(
        [
            # "report_date",
            "plant_id_eia",
            # "plant_name",
            "generator_id",
            # "balancing_authority_code",
            settings["capacity_col"],
            "capacity_mw",
            "energy_source_code_1",
            "energy_source_code_2",
            "minimum_load_mw",
            "operational_status_code",
            "planned_new_capacity_mw",
            "switch_oil_gas",
            "technology_description",
            "time_cold_shutdown_full_load_code",
            "planned_retirement_date",
            "prime_mover_code",
        ]
    )
    gen_cols = [c for c in gen_cols if c in gens_860]

    entity_cols = [
        "plant_id_eia",
        "generator_id",
        "prime_mover_code",
        "operating_date",
        "generator_operating_date",
        "original_planned_operating_date",
        "original_planned_generator_operating_date",
    ]
    entity_cols = [c for c in entity_cols if c in gens_entity]

    bga_cols = [
        "plant_id_eia",
        "generator_id",
        "boiler_id",
        "unit_id_eia",
        "unit_id_pudl",
    ]

    # In this merge of the three dataframes we're trying to label each generator with
    # the model region it is part of, the prime mover and operating date, and the
    # PUDL unit codes (where they exist).

    # drop duplicate rows of model_region_map
    mapping = model_region_map.drop(columns="region").drop_duplicates()
    # drop duplicate mappings for the same ID
    mapping = mapping.loc[~mapping.plant_id_eia.duplicated(), :]

    gens_860_model = (
        pd.merge(
            gens_860[gen_cols],
            mapping,
            on="plant_id_eia",
            how="inner",
        )
        .merge(
            gens_entity[entity_cols], on=["plant_id_eia", "generator_id"], how="inner"
        )
        .merge(bga[bga_cols], on=["plant_id_eia", "generator_id"], how="left")
    )
    gens_860_model["unit_id_pg"] = gens_860_model.loc[:, "unit_id_pudl"]
    gens_860_model.loc[gens_860_model.unit_id_pg.isnull(), "unit_id_pg"] = (
        gens_860_model.loc[gens_860_model.unit_id_pg.isnull(), "plant_id_eia"].astype(
            str
        )
        + "_"
        + gens_860_model.loc[gens_860_model.unit_id_pg.isnull(), "generator_id"].astype(
            str
        )
    ).to_numpy()

    # Where summer/winter capacity values are missing set equal to nameplate capacity,
    # but only if all generators within a unit are missing the capacity value
    check_units = gens_860_model.loc[
        gens_860_model[settings["capacity_col"]].isna()
    ].groupby(["plant_id_eia", "unit_id_pg"])
    for (plant_id, unit_id), _df in check_units:
        if _df[settings["capacity_col"]].isna().all():
            gens_860_model.loc[
                (gens_860_model["plant_id_eia"] == plant_id)
                & (gens_860_model["unit_id_pg"] == unit_id),
                settings["capacity_col"],
            ] = gens_860_model.loc[
                (gens_860_model["plant_id_eia"] == plant_id)
                & (gens_860_model["unit_id_pg"] == unit_id),
                "capacity_mw",
            ]

    merged_capacity = gens_860_model.groupby("technology_description")[
        settings["capacity_col"]
    ].sum()
    if not np.allclose(initial_capacity.sum(), merged_capacity.sum()):
        for i_idx, i_row in initial_capacity.iteritems():
            if not np.allclose(i_row, merged_capacity[i_idx]):
                logger.warning(
                    "********************************\n"
                    "When adding plant entity/boiler info to generators and filling missing"
                    " seasonal capacity values, technology"
                    f"{i_idx} changed capacity from {i_row} to {merged_capacity[i_idx]}"
                    "\n********************************"
                )

    return gens_860_model


def create_plant_gen_id(df):
    """Combine the plant id and generator id to form a unique combination

    Parameters
    ----------
    df : dataframe
        Must contain columns plant_id_eia and generator_id

    Returns
    -------
    dataframe
        Same as input but with the additional column plant_gen_id
    """

    df["plant_gen_id"] = (
        df["plant_id_eia"].astype(str) + "_" + df["generator_id"].astype(str)
    )

    return df


def remove_canceled_860m(df, canceled_860m):
    """Remove generators that 860m shows as having been canceled

    Parameters
    ----------
    df : dataframe
        All of the EIA 860 generators
    canceled_860m : dataframe
        From the 860m Canceled or Postponed sheet

    Returns
    -------
    dataframe
        Same as input, but possibly without generators that were proposed
    """
    df = create_plant_gen_id(df)
    canceled_860m = create_plant_gen_id(canceled_860m)

    canceled = df.loc[df["plant_gen_id"].isin(canceled_860m["plant_gen_id"]), :]

    not_canceled_df = df.loc[~df["plant_gen_id"].isin(canceled_860m["plant_gen_id"]), :]

    not_canceled_df = not_canceled_df.drop(columns="plant_gen_id")

    if not canceled.empty:
        assert len(df) == len(canceled) + len(not_canceled_df)

    return not_canceled_df.reset_index(drop=True)


def remove_retired_860m(df, retired_860m):
    """Remove generators that 860m shows as having been retired

    Parameters
    ----------
    df : dataframe
        All of the EIA 860 generators
    retired_860m : dataframe
        From the 860m Retired sheet

    Returns
    -------
    dataframe
        Same as input, but possibly without generators that have retired
    """

    df = create_plant_gen_id(df)
    retired_860m = create_plant_gen_id(retired_860m)

    retired = df.loc[df["plant_gen_id"].isin(retired_860m["plant_gen_id"]), :]

    not_retired_df = df.loc[~df["plant_gen_id"].isin(retired_860m["plant_gen_id"]), :]

    not_retired_df = not_retired_df.drop(columns="plant_gen_id")

    if not retired.empty:
        assert len(df) == len(retired) + len(not_retired_df)

    return not_retired_df.reset_index(drop=True)


def update_planned_retirement_date_860m(
    df: pd.DataFrame, operating_860m: pd.DataFrame
) -> pd.DataFrame:
    """Update the planned retirement date in the main dataframe using the planned
    retirement year column from 860m existing generators.

    Parameters
    ----------
    df : pd.DataFrame
        Main dataframe of generators from EIA 860. Must have columns "plant_id_eia",
        "generator_id", and "planned_retirement_date", which should be in a datatime format.
    operating_860m : pd.DataFrame
        Dataframe of operating generators from EIA 860m. Must have columns "plant_id_eia",
        "generator_id", and "planned_retirement_year"

    Returns
    -------
    pd.DataFrame
        Modified version of "df" dataframe, with updated values in column
        "planned_retirement_date" based on EIA-860m.
    """
    if "planned_retirement_date" not in df.columns:
        logger.warning(
            "The main generators dataframe from EIA 860 does not have a column "
            "'planned_retirement_date'. If this column is missing all retirement dates "
            "will be based on plant age."
        )
        return df
    if "planned_retirement_year" not in operating_860m.columns:
        logger.warning(
            "The EIA-860m existing dataframe does not have a column 'planned_retirement_year'."
            "Check the 860m file to see if it has been renamed. No values from the original "
            "860 data will be changed."
        )
        return df
    if df.empty or operating_860m.empty:
        return df
    _df = df.set_index(["plant_id_eia", "generator_id"])
    _operating_860m = operating_860m.set_index(["plant_id_eia", "generator_id"])
    _operating_860m["planned_retirement_date_860m"] = pd.to_datetime(
        _operating_860m["planned_retirement_year"], format="%Y"
    )
    update_df = pd.merge(
        _df,
        _operating_860m[["planned_retirement_date_860m"]],
        how="inner",
        left_index=True,
        right_index=True,
    )
    mask = update_df.loc[
        update_df["planned_retirement_date"].dt.year.fillna(9999)
        != update_df["planned_retirement_date_860m"].dt.year.fillna(9999),
        :,
    ].index
    _df.loc[mask, "planned_retirement_date"] = update_df.loc[
        mask, "planned_retirement_date_860m"
    ]

    return _df.reset_index()


def remove_future_retirements_860m(df, retired_860m):
    """Remove generators that 860m shows as having been retired

    Parameters
    ----------
    df : dataframe
        All of the EIA 860 generators
    retired_860m : dataframe
        From the 860m Retired sheet

    Returns
    -------
    dataframe
        Same as input, but possibly without generators that have retired
    """

    df = create_plant_gen_id(df)
    retired_860m = create_plant_gen_id(retired_860m)

    retired = df.loc[df["plant_gen_id"].isin(retired_860m["plant_gen_id"]), :]

    not_retired_df = df.loc[~df["plant_gen_id"].isin(retired_860m["plant_gen_id"]), :]

    not_retired_df = not_retired_df.drop(columns="plant_gen_id")

    if not retired.empty:
        assert len(df) == len(retired) + len(not_retired_df)

    return not_retired_df


def update_operating_date_860m(
    df: pd.DataFrame, operating_860m: pd.DataFrame
) -> pd.DataFrame:
    """Update the operating date of EIA generators using data from 860m.

    When the "operating_date" of a generator is nan, fill with operating year data
    from 860m.

    Parameters
    ----------
    df : pd.DataFrame
        Data on existing EIA generating units. Must have columns "plant_id_eia",
        "generator_id", and "operating_date".
    operating_860m : pd.DataFrame
        A dataframe of operating generating units from EIA 860m. Must have columns
        "plant_id_eia", "generator_id", and "operating_year".

    Returns
    -------
    pd.DataFrame
        The original "df" dataframe with missing operating dates filled using the operating
        year from 860m.
    """
    _df = df.set_index(["plant_id_eia", "generator_id"])
    _operating_860m = operating_860m.set_index(["plant_id_eia", "generator_id"])
    no_op_date = _df.loc[_df["operating_date"].isna(), :]
    no_op_date = pd.merge(
        no_op_date,
        _operating_860m["operating_year"],
        left_index=True,
        right_index=True,
        how="left",
        validate="1:1",
    )

    _df.loc[no_op_date.index, "operating_date"] = pd.to_datetime(
        no_op_date.loc[no_op_date["operating_year"].notna(), "operating_year"],
        format="%Y",
    )

    return _df.reset_index()


def load_923_gen_fuel_data(pudl_engine, pudl_out, model_region_map, data_years=[2017]):
    """
    Load generation and fuel data for each plant. EIA-923 provides these values for
    each prime mover/fuel combination at every generator. This data can be used to
    calculate the heat rate of generators at a single plant. Generators sharing a prime
    mover (e.g. multiple combustion turbines) will end up sharing the same heat rate.

    Parameters
    ----------
    pudl_engine : sqlalchemy.Engine
        A sqlalchemy connection for use by pandas
    pudl_out : pudl.PudlTabl
        A PudlTabl object for loading pre-calculated PUDL analysis data
    model_region_map : dataframe
        A dataframe with columns 'plant_id_eia' and 'model_region' (aggregated regions)
    data_years : list, optional
        Years of data to include, by default [2017]

    Returns
    -------
    dataframe
        Generation, fuel use, and heat rates of prime mover/fuel combos over all data
        years. Columns are:

        ['plant_id_eia', 'fuel_type', 'fuel_type_code_pudl',
       'fuel_type_code_aer', 'prime_mover_code', 'fuel_consumed_units',
       'fuel_consumed_for_electricity_units', 'fuel_consumed_mmbtu',
       'fuel_consumed_for_electricity_mmbtu', 'net_generation_mwh',
       'heat_rate_mmbtu_mwh']
    """
    if isinstance(data_years, (int, float)):
        data_years = [str(data_years)]
    data_years = [str(y) for y in data_years]

    # Load 923 generation and fuel data for one or more years.
    # Only load plants in the model regions.
    sql = f"""
        SELECT * FROM generation_fuel_eia923
        WHERE strftime('%Y',report_date) in ({','.join(['?']*len(data_years))})
    """
    gen_fuel_923 = pd.read_sql_query(
        sql, pudl_engine, params=data_years, parse_dates=["report_date"]
    )
    gen_fuel_923 = gen_fuel_923.loc[
        gen_fuel_923["plant_id_eia"].isin(model_region_map.plant_id_eia),
        :,
    ]

    insp = sqlalchemy.inspect(pudl_engine)
    if insp.has_table("generation_fuel_nuclear_eia923"):
        sql = f"""
            SELECT * FROM generation_fuel_nuclear_eia923
            WHERE strftime('%Y',report_date) in ({','.join(['?']*len(data_years))})
        """
        gen_fuel_nuclear_923 = pd.read_sql_query(
            sql, pudl_engine, params=data_years, parse_dates=["report_date"]
        )
        gen_fuel_nuclear_923 = gen_fuel_nuclear_923.loc[
            gen_fuel_nuclear_923["plant_id_eia"].isin(model_region_map.plant_id_eia),
            :,
        ]

        gen_fuel_923 = pd.concat(
            [gen_fuel_923, gen_fuel_nuclear_923], ignore_index=True
        )

    return gen_fuel_923


def modify_cc_prime_mover_code(df, gens_860):
    """Change combined cycle prime movers from CA and CT to CC.

    The heat rate of combined cycle plants that aren't included in PUDL heat rate by
    unit should probably be done with the combustion and steam turbines combined. This
    modifies the prime mover code of those two generator types so that they match. It
    doesn't touch the CS code, which is for single shaft combined units.

    Parameters
    ----------
    df : dataframe
        A dataframe with columns prime_mover_code, and plant_id_eia.
    gens_860 : dataframe
        EIA860 dataframe with technology_description, unit_id_pudl, plant_id_eia
        columns.

    Returns
    -------
    dataframe
        Modified 923 dataframe where prime mover codes at CC generators that don't have
        a PUDL unit id are modified from CA and CT to CC.
    """
    df.loc[
        (df["prime_mover_code"].isin(["CA", "CT"])),
        "prime_mover_code",
    ] = "CC"

    return df


def group_gen_by_year_fuel_primemover(df):
    """
    Group generation and fuel consumption by plant, prime mover, and fuel type. Only
    matters where multiple years of data are used, otherwise output should be the same
    as input.

    Parameters
    ----------
    df : dataframe
        Generation and fuel consumption data from EIA 923 for each plant, prime mover,
        and fuel type

    Returns
    -------
    dataframe
        Sum of generation and fuel consumption data (if multiple years).
    """

    # Group the data by plant, fuel type, and prime mover
    by = [
        "plant_id_eia",
        "fuel_type",
        "energy_source_code",
        "fuel_type_code_pudl",
        "fuel_type_code_aer",
        "prime_mover_code",
    ]
    by = [c for c in by if c in df.columns]
    sort = ["plant_id_eia", "fuel_type", "energy_source_code", "prime_mover_code"]
    sort = [c for c in sort if c in df.columns]

    annual_gen_fuel_923 = (
        (
            df.groupby(  # .drop(columns=["id", "nuclear_unit_id"])
                by=by, as_index=False
            )[
                [
                    "fuel_consumed_units",
                    "fuel_consumed_for_electricity_units",
                    "fuel_consumed_mmbtu",
                    "fuel_consumed_for_electricity_mmbtu",
                    "net_generation_mwh",
                ]
            ].sum()
        )
        .reset_index()
        .drop(columns="index")
        .sort_values(sort)
    )

    return annual_gen_fuel_923


def add_923_heat_rate(df):
    """
    Small function to calculate the heat rate of records with fuel consumption and net
    generation.

    Parameters
    ----------
    df : dataframe
        Must contain the columns net_generation_mwh and
        fuel_consumed_for_electricity_mmbtu

    Returns
    -------
    dataframe
        Same dataframe with new column of heat_rate_mmbtu_mwh
    """

    # Calculate the heat rate for each prime mover/fuel combination
    df["heat_rate_mmbtu_mwh"] = (
        df["fuel_consumed_for_electricity_mmbtu"] / df["net_generation_mwh"]
    )

    return df


def calculate_weighted_heat_rate(heat_rate_df):
    """
    Calculate the weighed heat rate when multiple years of data are used. Net generation
    in each year is used as the weights.

    Parameters
    ----------
    heat_rate_df : dataframe
        Currently the PudlTabl unit_hr method.

    Returns
    -------
    dataframe
        Heat rate weighted by annual generation for each plant and PUDL unit
    """

    def w_hr(df):
        weighted_hr = np.average(
            df["heat_rate_mmbtu_mwh"], weights=df["net_generation_mwh"]
        )
        return weighted_hr

    weighted_unit_hr = heat_rate_df.groupby(["plant_id_eia", "unit_id_pudl"]).apply(
        w_hr
    )
    weighted_unit_hr.name = "heat_rate_mmbtu_mwh"
    weighted_unit_hr = weighted_unit_hr.reset_index()

    return weighted_unit_hr


def plant_pm_heat_rates(annual_gen_fuel_923):
    """
    Calculate the heat rate by plant, prime mover, and fuel type. Values are saved
    as a dictionary.

    Parameters
    ----------
    annual_gen_fuel_923 : dataframe
        Data from the 923 generation and fuel use table. Heat rate for each row should
        already be calculated.

    Returns
    -------
    dict
        Keys are a tuple of plant id, prime mover, and fuel type. Values are the heat
        rate.
    """

    by = ["plant_id_eia", "prime_mover_code", "fuel_type", "energy_source_code"]
    by = [c for c in by if c in annual_gen_fuel_923.columns]
    annual_gen_fuel_923_groups = annual_gen_fuel_923.groupby(by)

    prime_mover_hr_map = {
        _: df["heat_rate_mmbtu_mwh"].values[0] for _, df in annual_gen_fuel_923_groups
    }

    return prime_mover_hr_map


def unit_generator_heat_rates(pudl_out, data_years):
    """
    Calculate the heat rate for each PUDL unit and generators that don't have a PUDL
    unit id.

    Parameters
    ----------
    pudl_out : pudl.PudlTabl
        A PudlTabl object for loading pre-calculated PUDL analysis data
    data_years : list
        Years of data to use

    Returns
    -------
    dataframe, dict
        A dataframe of heat rates for each pudl unit (columsn are ['plant_id_eia',
        'unit_id_pg', 'heat_rate_mmbtu_mwh']).
    """

    # Load the pre-calculated PUDL unit heat rates for selected years.
    # Remove rows without generation or with null values.
    unit_hr = pudl_out.hr_by_unit()
    unit_hr = unit_hr.loc[
        (unit_hr.report_date.dt.year.isin(data_years))
        & (unit_hr.net_generation_mwh > 0),
        :,
    ].dropna()

    weighted_unit_hr = calculate_weighted_heat_rate(unit_hr)

    return weighted_unit_hr


def group_units(df, settings):
    """
    Group by units within a region/technology/cluster. Add a unique unit code
    (plant plus generator) for any generators that aren't part of a unit.


    Returns
    -------
    dataframe
        Grouped generators with the total capacity, minimum load, and average heat
        rate for each.
    """

    by = ["plant_id_eia", "unit_id_pg"]
    # add a unit code (plant plus generator code) in cases where one doesn't exist
    df_copy = df.reset_index()

    # All units should have the same heat rate so taking the mean will just keep the
    # same value.
    grouped_units = df_copy.groupby(by).agg(
        {
            settings["capacity_col"]: "sum",
            "capacity_mwh": "sum",
            "minimum_load_mw": "sum",
            "heat_rate_mmbtu_mwh": "mean",
            "Fixed_OM_Cost_per_MWyr": "mean",
            "Var_OM_Cost_per_MWh": "mean",
        }
    )
    grouped_units = grouped_units.replace([np.inf, -np.inf], np.nan)
    grouped_units = grouped_units.fillna(grouped_units.mean())

    return grouped_units


def calc_unit_cluster_values(
    df: pd.DataFrame, capacity_col: str, technology: str = None, clustered: bool = True
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
    # if not clustering units no need to calulate cluster average values
    if len(df) == 1:
        clustered = False
    elif df["cluster"].nunique() == len(df):
        clustered = False

    if not clustered:
        df["Min_Power"] = df["minimum_load_mw"] / df[capacity_col]
        df["num_units"] = 1
        if technology:
            df["technology"] = technology

        return df.reset_index().replace(np.inf, 0)

    # Define a function to compute the weighted mean.
    # The issue here is that the df name needs to be used in the function.
    # So this will need to be within a function that takes df as an input
    def wm(x):
        try:
            return np.average(
                x, weights=df.loc[x.index, capacity_col].replace(np.nan, 0)
            )
        except ZeroDivisionError:
            return x.mean()

    if df["heat_rate_mmbtu_mwh"].isnull().values.any():
        # mean =
        # df["heat_rate_mmbtu_mwh"] = df["heat_rate_mmbtu_mwh"].fillna(
        #     df["heat_rate_mmbtu_mwh"].median()
        # )
        start_cap = df[capacity_col].sum()
        df = df.loc[~df["heat_rate_mmbtu_mwh"].isnull(), :]
        end_cap = df[capacity_col].sum()
        cap_diff = start_cap - end_cap
        logger.warning(f"dropped {cap_diff}MW because of null heat rate values")

    df_values = df.groupby("cluster", as_index=False).agg(
        {
            capacity_col: "mean",
            "capacity_mwh": "sum",
            "minimum_load_mw": "mean",
            "heat_rate_mmbtu_mwh": wm,
            "Fixed_OM_Cost_per_MWyr": wm,
            "Var_OM_Cost_per_MWh": wm,
        }
    )
    df_values.index = df_values["cluster"].values
    if df_values["heat_rate_mmbtu_mwh"].isnull().values.any():
        print(df)
        print(df_values)
    df_values["heat_rate_mmbtu_mwh_iqr"] = df.groupby("cluster").agg(
        {"heat_rate_mmbtu_mwh": iqr}
    )
    df_values["heat_rate_mmbtu_mwh_std"] = df.groupby("cluster").agg(
        {"heat_rate_mmbtu_mwh": "std"}
    )
    df_values["fixed_o_m_mw_std"] = df.groupby("cluster").agg(
        {"Fixed_OM_Cost_per_MWyr": "std"}
    )

    df_values["Min_Power"] = df_values["minimum_load_mw"] / df_values[capacity_col]

    df_values["num_units"] = (
        df.dropna(subset=capacity_col).groupby("cluster")["cluster"].count()
    )

    if technology:
        df_values["technology"] = technology

    return df_values


def add_genx_model_tags(df, settings):
    """
    Each generator type needs to have certain tags for use by the GenX model. Each tag
    is a column, e.g. THERM for thermal generators. These columns and tag values are
    defined in the settings file and applied here. Tags are (usually?) boolean 0/1
    values.

    Parameters
    ----------
    df : dataframe
        Clusters of generators. The index should have a column 'technology', which
        is used to map tag values.
    settings : dict
        User-defined settings loaded from a YAML file.

    Returns
    -------
    dataframe
        The original generator cluster results with new columns for each model tag.
    """
    ignored = r"_"
    technology = df["technology"].str.replace(ignored, "")
    # Create a new dataframe with the same index
    default = settings.get("default_model_tag", 0)
    for tag_col in settings.get("model_tag_names", []):
        df[tag_col] = default
        if tag_col not in settings.get("generator_columns", []) and isinstance(
            settings.get("generator_columns"), list
        ):
            settings["generator_columns"].append(tag_col)

        try:
            for tech, tag_value in sorted(
                settings["model_tag_values"][tag_col].items(),
                key=lambda item: len(str(item[0])),
            ):
                tech = re.sub(ignored, "", tech)
                mask = technology.str.contains(rf"^{tech}", case=False)
                df.loc[mask, tag_col] = tag_value
        except (KeyError, AttributeError) as e:
            logger.warning(f"No model tag values found for {tag_col} ({e})")

    # Change tags with specific regional values for a technology
    flat_regional_tags = flatten(
        sort_nested_dict(settings.get("regional_tag_values", {}) or {})
    )

    for tag_tuple, tag_value in flat_regional_tags.items():
        region, tag_col, tech = tag_tuple
        tech = re.sub(ignored, "", tech)
        mask = technology.str.contains(rf"^{tech}", case=False)
        df.loc[(df["region"] == region) & mask, tag_col] = tag_value

    return df


def download_860m(settings: dict) -> pd.ExcelFile:
    """Load the entire 860m file into memory as an ExcelFile object.

    Parameters
    ----------
    settings : dict
        User-defined settings loaded from a YAML file. This is where the EIA860m
        filename is defined as the parameter "eia_860m_fn".

    Returns
    -------
    pd.ExcelFile
        The ExcelFile object with all sheets from 860m.
    """
    fn = settings.get("eia_860m_fn")
    if not fn:
        logger.info("Trying to determine the most recent EIA860m file...")
        fn = find_newest_860m()

    engine = None
    ext = fn.split(".")[-1]
    if ext == "xlsx":
        engine = "openpyxl"
    elif ext == "xls":
        engine = "xlrd"

    # Only the most recent file will not have archive in the url
    url = f"https://www.eia.gov/electricity/data/eia860m/xls/{fn}"
    archive_url = f"https://www.eia.gov/electricity/data/eia860m/archive/xls/{fn}"

    local_file = DATA_PATHS["eia_860m"] / fn
    if local_file.exists():
        logger.info(f"Reading a local copy of the EIA860m file {fn}")
        eia_860m = pd.ExcelFile(local_file)
    else:
        logger.info(f"Downloading the EIA860m file {fn}")
        try:
            download_save(url, local_file)
            eia_860m = pd.ExcelFile(local_file, engine=engine)
        except (XLRDError, ValueError, BadZipFile):
            logger.warning("A more recent version of EIA-860m is available")
            download_save(archive_url, local_file)
            eia_860m = pd.ExcelFile(local_file, engine=engine)
        # write the file to disk

    return eia_860m


def find_newest_860m() -> str:
    """Scrape the EIA 860m page to find the most recently posted file.

    Returns
    -------
    str
        Name of most recently posted file
    """
    site_url = "https://www.eia.gov/electricity/data/eia860m/"
    r = requests.get(site_url)
    soup = BeautifulSoup(r.content, "lxml")
    table = soup.find("table", attrs={"class": "basic-table"})
    if not table:
        raise ValueError(
            "Could not determine the most recently posted EIA 860m file. EIA may have "
            "changed their HTML format, please post this as an issue on the PowerGenome "
            "github repository (https://github.com/PowerGenome/PowerGenome/issues/new)."
        )
    href = table.find("a")["href"]
    fn = href.split("/")[-1]

    return fn


def clean_860m_sheet(
    eia_860m: pd.ExcelFile, sheet_name: str, settings: dict
) -> pd.DataFrame:
    """Load a sheet from the 860m ExcelFile object and clean it.

    Parameters
    ----------
    eia_860m : ExcelFile
        Entire 860m file loaded into memory
    sheet_name : str
        Name of the sheet to load as a dataframe
    settings : dict
        User-defined settings loaded from a YAML file.

    Returns
    -------
    pd.DataFrame
        One of the sheets from 860m
    """

    df = eia_860m.parse(sheet_name=sheet_name, na_values=[" "])

    # Find skiprows and skipfooters, which changes across 860m versions.
    # NEW: drop rows with all NaN because EIA added a blank row before the footer.
    sr = 0

    for idx, row in df.iterrows():
        if row.iloc[0] == "Entity ID":
            sr = idx + 1
            break

    sf = 0

    for idx in list(range(-10, 0)):
        if isinstance(df.iloc[idx, 0], str):
            sf = -idx
            break
    df = eia_860m.parse(
        sheet_name=sheet_name, skiprows=sr, skipfooter=sf, na_values=[" "]
    )
    df = df.dropna(how="all")
    df = df.rename(columns=planned_col_map)
    df["plant_id_eia"] = df["plant_id_eia"].astype("Int64")

    if sheet_name in ["Operating", "Planned"]:
        df.loc[:, "operational_status_code"] = df.loc[:, "operational_status"].map(
            op_status_map
        )

    if sheet_name == "Planned":
        df = df.loc[
            df["operational_status_code"].isin(settings["proposed_status_included"]), :
        ]
    df.columns = snake_case_col(df.columns)

    return df


def load_860m(settings: dict) -> Dict[str, pd.DataFrame]:
    """Load the planned, canceled, and retired sheets from an EIA 860m file.

    Parameters
    ----------
    settings : dict
        User-defined settings loaded from a YAML file. This is where the EIA860m
        filename is defined.

    Returns
    -------
    Dict[str, pd.DataFrame]
        The 860m dataframes, with the keys 'planned', 'canceled', and 'retired'.
    """
    sheet_map = {
        "operating": "Operating",
        "planned": "Planned",
        "canceled": "Canceled or Postponed",
        "retired": "Retired",
    }

    fn = settings.get("eia_860m_fn")
    if not fn:
        fn = find_newest_860m()

    fn_name = Path(fn).stem

    data_dict = {}
    eia_860m_excelfile = None
    for name, sheet in sheet_map.items():
        pkl_path = DATA_PATHS["eia_860m"] / f"{fn_name}_{name}.pkl"
        if pkl_path.exists():
            data_dict[name] = pd.read_pickle(pkl_path)
            data_dict[name]["plant_id_eia"] = data_dict[name]["plant_id_eia"].astype(
                "Int64"
            )
            data_dict[name].columns = snake_case_col(data_dict[name].columns)
        else:
            if eia_860m_excelfile is None:
                eia_860m_excelfile = download_860m(settings)
            data_dict[name] = clean_860m_sheet(eia_860m_excelfile, sheet, settings)
            data_dict[name].to_pickle(pkl_path)

    return data_dict


def label_gen_region(
    df: pd.DataFrame, settings: dict, model_regions_gdf: gpd.GeoDataFrame
) -> pd.DataFrame:
    """Label the region that generators in a dataframe belong to based on their
    geographic location. This is done via geospaital join and may not always be accurate
    based on actual utility connections.

    Parameters
    ----------
    df : pd.DataFrame
        Generators that are not assigned to a model region.
    settings : dict
        Need the parameter `capacity_col` to determine which column has capacity.
    model_regions_gdf : gpd.GeoDataFrame
        Contains the name and geometry of each region being used in the study

    Returns
    -------
    pd.DataFrame
        [description]
    """

    no_lat_lon = df.loc[
        (df["latitude"].isnull()) | (df["longitude"].isnull()), :
    ].copy()
    if not no_lat_lon.empty:
        no_lat_lon_cap = no_lat_lon[settings["capacity_col"]].sum()
        logger.warning(
            "Some generators do not have lon/lat data. Check the source "
            "file to determine if they should be included in results. "
            f"\nThe affected generators account for {no_lat_lon_cap} in balancing "
            "authorities: "
            f"\n{no_lat_lon['balancing_authority_code'].tolist()}"
        )

    df = df.dropna(subset=["latitude", "longitude"])

    # Convert the lon/lat values to geo points. Need to add an initial CRS and then
    # change it to align with the IPM regions
    print("Creating gdf")
    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df.longitude.copy(), df.latitude.copy()),
        crs="EPSG:4326",
    )
    if gdf.crs != model_regions_gdf.crs:
        gdf = gdf.to_crs(model_regions_gdf.crs)

    gdf = gpd.sjoin(model_regions_gdf.drop(columns="region"), gdf)

    return gdf


def import_new_generators(
    operating_860m: pd.DataFrame,
    gens_860: pd.DataFrame,
    settings: dict,
    model_regions_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Find the set of generating units in 860m that are not in the annual 860 data.

    This is especially important for new wind, solar, and battery units, which are built
    on short timelines. Format the data for inclusion with other existing existing units.

    Parameters
    ----------
    operating_860m : pd.DataFrame
        Operating generators from EIA 860m.
    gens_860 : pd.DataFrame
        The set of operating units from other sources (e.g. annual 860 data in PUDL).
    settings : dict
        Requires the parameter "model_regions". Optional parameters include
        "proposed_gen_heat_rates", "proposed_min_load", "capacity_col", "group_technologies",
        and "retirement_ages".
    model_regions_gdf : gpd.GeoDataFrame
        Geospatial representation of the model regions. Used to assign generators to
        model regions.

    Returns
    -------
    pd.DataFrame
        Set of operating generators that were not already in the gens_860 dataframe
    """
    _operating_860m = operating_860m.copy()
    _operating_860m["generator_id"] = _operating_860m["generator_id"].apply(
        remove_leading_zero
    )
    gens_860_id = list(zip(gens_860["plant_id_eia"], gens_860["generator_id"]))
    operating_860m_id = zip(
        _operating_860m["plant_id_eia"], _operating_860m["generator_id"]
    )

    new_mask = [g not in gens_860_id for g in operating_860m_id]
    new_operating = label_gen_region(
        _operating_860m.loc[new_mask, :], settings, model_regions_gdf
    )
    new_operating = new_operating.drop_duplicates(
        subset=["plant_id_eia", "generator_id"]
    )
    new_operating.loc[:, "heat_rate_mmbtu_mwh"] = new_operating.loc[
        :, "technology_description"
    ].map(settings.get("proposed_gen_heat_rates", {}) or {})

    # The default EIA heat rate for non-thermal technologies is 9.21
    new_operating.loc[
        new_operating["heat_rate_mmbtu_mwh"].isnull(), "heat_rate_mmbtu_mwh"
    ] = 9.21

    new_operating.loc[:, "minimum_load_mw"] = (
        new_operating["technology_description"].map(
            settings.get("proposed_min_load", {}) or {}
        )
        * new_operating[settings.get("capacity_col", "capacity_mw")]
    )

    # Assume anything else being built at scale is wind/solar and will have a Min_power
    # of 0
    new_operating.loc[new_operating["minimum_load_mw"].isnull(), "minimum_load_mw"] = 0

    new_operating = new_operating.set_index(
        ["plant_id_eia", "prime_mover_code", "energy_source_code_1"]
    )

    # Add a retirement year based on the planned start year
    label_retirement_year(
        df=new_operating,
        settings=settings,
        age_col="operating_year",
        add_additional_retirements=False,
    )
    if (
        new_operating.loc[new_operating["technology_description"].isnull(), :].empty
        is False
    ):
        plant_ids = list(
            new_operating.loc[new_operating["technology_description"].isnull(), :]
            .index.get_level_values("plant_id_eia")
            .to_numpy()
        )
        plant_capacity = new_operating.loc[
            new_operating["technology_description"].isnull(),
            settings.get("capacity_col", "capacity_mw"),
        ].sum()

        logger.warning(
            f"The EIA860 file has {len(plant_ids)} operating generator(s) without a technology "
            f"description. The plant IDs are {plant_ids}, and they have a combined "
            f"capcity of {plant_capacity} MW."
        )

    if settings.get("group_technologies"):
        new_operating = group_technologies(
            new_operating,
            settings["group_technologies"],
            settings.get("tech_groups", {}) or {},
            settings.get("regional_no_grouping", {}) or {},
        )
        print(new_operating["technology_description"].unique().tolist())

    keep_cols = [
        "model_region",
        "technology_description",
        "generator_id",
        settings.get("capacity_col", "capacity_mw"),
        "capacity_mwh",
        "minimum_load_mw",
        "operational_status_code",
        "heat_rate_mmbtu_mwh",
        "retirement_year",
        "operating_year",
        "state",
    ]

    return new_operating.reindex(columns=keep_cols)


def import_proposed_generators(
    planned: pd.DataFrame, settings: dict, model_regions_gdf: gpd.GeoDataFrame
) -> pd.DataFrame:
    """
    Load the most recent proposed generating units from EIA860m. Will also add
    any planned generators that are included in the settings file.

    Parameters
    ----------
    planned : pd.DataFrame
        Generators that are not assigned to a model region.
    settings : dict
        User defined parameters from a settings YAML file
    model_regions_gdf : gpd.GeoDataFrame
        Contains the name and geometry of each region being used in the study

    Returns
    -------
    pd.DataFrame
        All proposed generators.
    """

    # Some plants don't have lat/lon data. Log this now to determine if any action is
    # needed, then drop them from the dataframe.
    # no_lat_lon = planned.loc[
    #     (planned["latitude"].isnull()) | (planned["longitude"].isnull()), :
    # ].copy()
    # if not no_lat_lon.empty:
    #     no_lat_lon_cap = no_lat_lon[settings["capacity_col"]].sum()
    #     logger.warning(
    #         "Some generators do not have lon/lat data. Check the source "
    #         "file to determine if they should be included in results. "
    #         f"\nThe affected generators account for {no_lat_lon_cap} in balancing "
    #         "authorities: "
    #         f"\n{no_lat_lon['balancing_authority_code'].tolist()}"
    #     )

    # planned = planned.dropna(subset=["latitude", "longitude"])

    # # Convert the lon/lat values to geo points. Need to add an initial CRS and then
    # # change it to align with the IPM regions
    # print("Creating gdf")
    # planned_gdf = gpd.GeoDataFrame(
    #     planned.copy(),
    #     geometry=gpd.points_from_xy(planned.longitude.copy(), planned.latitude.copy()),
    #     crs="EPSG:4326",
    # )
    # if planned_gdf.crs != model_regions_gdf.crs:
    #     planned_gdf = planned_gdf.to_crs(model_regions_gdf.crs)

    # planned_gdf = gpd.sjoin(model_regions_gdf.drop(columns="IPM_Region"), planned_gdf)
    _planned = planned.copy()
    _planned["generator_id"] = _planned["generator_id"].apply(remove_leading_zero)
    planned_gdf = label_gen_region(_planned, settings, model_regions_gdf)
    planned_gdf = planned_gdf.drop_duplicates(subset=["plant_id_eia", "generator_id"])

    # Add planned additions from the settings file
    additional_planned = settings.get("additional_planned") or []
    for record in additional_planned:
        plant_id, gen_id, model_region = record
        plant_record = planned.loc[
            (planned["plant_id_eia"] == plant_id) & (planned["generator_id"] == gen_id),
            :,
        ]
        plant_record["model_region"] = model_region

        planned_gdf = planned_gdf.append(plant_record, sort=False)

    logger.info(
        f"{len(additional_planned)} generators were added to the planned list based on settings"
    )

    planned_gdf.loc[:, "heat_rate_mmbtu_mwh"] = planned_gdf.loc[
        :, "technology_description"
    ].map(settings["proposed_gen_heat_rates"])

    # The default EIA heat rate for non-thermal technologies is 9.21
    planned_gdf.loc[
        planned_gdf["heat_rate_mmbtu_mwh"].isnull(), "heat_rate_mmbtu_mwh"
    ] = 9.21

    planned_gdf.loc[:, "minimum_load_mw"] = (
        planned_gdf["technology_description"].map(settings["proposed_min_load"])
        * planned_gdf[settings["capacity_col"]]
    )

    # Assume anything else being built at scale is wind/solar and will have a Min_Power
    # of 0
    planned_gdf.loc[planned_gdf["minimum_load_mw"].isnull(), "minimum_load_mw"] = 0

    planned_gdf = planned_gdf.set_index(
        ["plant_id_eia", "prime_mover_code", "energy_source_code_1"]
    )

    if (
        planned_gdf.loc[planned_gdf["technology_description"].isnull(), :].empty
        is False
    ):
        plant_ids = list(
            planned_gdf.loc[planned_gdf["technology_description"].isnull(), :]
            .index.get_level_values("plant_id_eia")
            .to_numpy()
        )
        plant_capacity = planned_gdf.loc[
            planned_gdf["technology_description"].isnull(), settings["capacity_col"]
        ].sum()

        logger.warning(
            f"The EIA860 file has {len(plant_ids)} proposed generator(s) without a technology "
            f"description. The plant IDs are {plant_ids}, and they have a combined "
            f"capcity of {plant_capacity} MW."
        )

    # Add a retirement year based on the planned start year
    label_retirement_year(
        df=planned_gdf,
        settings=settings,
        age_col="planned_operating_year",
        add_additional_retirements=False,
    )

    if settings.get("group_technologies"):
        planned_gdf = group_technologies(
            planned_gdf,
            settings["group_technologies"],
            settings.get("tech_groups", {}) or {},
            settings.get("regional_no_grouping", {}) or {},
        )
        print(planned_gdf["technology_description"].unique().tolist())

    keep_cols = [
        "model_region",
        "technology_description",
        "generator_id",
        settings["capacity_col"],
        "minimum_load_mw",
        "operational_status_code",
        "heat_rate_mmbtu_mwh",
        "planned_operating_year",
        "retirement_year",
    ]

    return planned_gdf.loc[:, keep_cols]


def gentype_region_capacity_factor(
    pudl_engine, plant_region_map, settings, years_filter=None
):
    """
    Calculate the average capacity factor for all generators of a type/region. This
    uses all years of available data unless otherwise specified. The potential
    generation is calculated for every year a plant is in operation using the capacity
    type specified in settings (nameplate, summer, or winter) and the number of hours
    in each year.

    As of this time PUDL only has generation data back to 2011.

    Parameters
    ----------
    pudl_engine : sqlalchemy.Engine
        A sqlalchemy connection for use by pandas
    plant_region_map : dataframe
        A dataframe with the region for every plant
    settings : dictionary
        The dictionary of settings with a dictionary of region aggregations

    Returns
    -------
    DataFrame
        A dataframe with the capacity factor of every selected technology
    """
    data_years = settings[
        "eia_data_years"
    ].copy()  # [str(y) for y in settings["eia_data_years"]]
    data_years.extend(settings.get("capacity_factor_default_year_filter", []))

    cap_col = settings["capacity_col"]

    # Include standby (SB) generators since they are in our capacity totals
    sql = f"""
        SELECT
            G.report_date,
            G.plant_id_eia,
            G.generator_id,
            SUM(G.capacity_mw) AS capacity_mw,
            SUM(G.summer_capacity_mw) as summer_capacity_mw,
            SUM(G.winter_capacity_mw) as winter_capacity_mw,
            G.technology_description,
            G.fuel_type_code_pudl
        FROM
            generators_eia860 G
        WHERE operational_status_code NOT IN ('RE', 'OS', 'IP', 'CN')
        AND strftime('%Y',report_date) in ({','.join(['?']*len(data_years))})
        GROUP BY
            G.report_date,
            G.plant_id_eia,
            G.technology_description,
            G.fuel_type_code_pudl,
            G.generator_id
        ORDER by G.plant_id_eia, G.report_date
    """

    plant_gen_tech_cap = pd.read_sql_query(
        sql,
        pudl_engine,
        params=[str(y) for y in data_years],
        parse_dates=["report_date"],
    )
    plant_gen_tech_cap = plant_gen_tech_cap.loc[
        plant_gen_tech_cap["plant_id_eia"].isin(plant_region_map["plant_id_eia"]), :
    ]

    plant_gen_tech_cap = fill_missing_tech_descriptions(plant_gen_tech_cap)
    plant_tech_cap = group_generators_at_plant(
        df=plant_gen_tech_cap,
        by=["plant_id_eia", "report_date", "technology_description"],
        agg_fn={cap_col: "sum"},
    )

    plant_tech_cap = plant_tech_cap.merge(
        plant_region_map, on="plant_id_eia", how="left"
    )

    label_small_hydro(plant_tech_cap, settings, by=["plant_id_eia", "report_date"])

    sql = """
        SELECT
            strftime('%Y', GF.report_date) AS report_date,
            GF.plant_id_eia,
            SUM(GF.net_generation_mwh) AS net_generation_mwh,
            GF.fuel_type_code_pudl
        FROM
            generation_fuel_eia923 GF
        GROUP BY
            strftime('%Y', GF.report_date),
            GF.plant_id_eia,
            GF.fuel_type_code_pudl
        ORDER by GF.plant_id_eia, strftime('%Y', GF.report_date)
    """
    generation = pd.read_sql_query(sql, pudl_engine, parse_dates={"report_date": "%Y"})

    if pudl.__version__ > "0.5.0":
        by = ["plant_id_eia"]
    else:
        by = {"plant_id_eia": "eia"}
    if pudl.__version__ < "2022.11.30":
        capacity_factor = pudl.helpers.clean_merge_asof(
            generation,
            plant_tech_cap,
            left_on="report_date",
            right_on="report_date",
            by=by,
        )
    else:
        capacity_factor = pudl.helpers.date_merge(
            generation,
            plant_tech_cap,
            on=by,
        )

    if settings.get("group_technologies"):
        capacity_factor = group_technologies(
            capacity_factor,
            settings["group_technologies"],
            settings.get("tech_groups", {}) or {},
            settings.get("regional_no_grouping", {}) or {},
        )

    cf_years = settings.get("capacity_factor_default_year_filter", data_years)
    capacity_factor = capacity_factor.loc[
        capacity_factor["report_date"].dt.year.isin(cf_years)
    ]

    # get a unique set of dates to generate the number of hours
    dates = capacity_factor["report_date"].drop_duplicates()
    dates_to_hours = pd.DataFrame(
        data={
            "report_date": dates,
            "hours": dates.apply(
                lambda d: (
                    pd.date_range(d, periods=2, freq="YS")[1]
                    - pd.date_range(d, periods=2, freq="YS")[0]
                )
                / pd.Timedelta(hours=1)
            ),
        }
    )

    # merge in the hours for the calculation
    capacity_factor = capacity_factor.merge(dates_to_hours, on=["report_date"])
    capacity_factor["potential_generation_mwh"] = (
        capacity_factor[cap_col] * capacity_factor["hours"]
    )
    plant_tech_capacity_factor = capacity_factor.groupby(
        ["plant_id_eia", "technology_description"], as_index=False
    )[["potential_generation_mwh", "net_generation_mwh"]].sum()

    plant_tech_capacity_factor["capacity_factor"] = (
        plant_tech_capacity_factor["net_generation_mwh"]
        / plant_tech_capacity_factor["potential_generation_mwh"]
    )
    plant_tech_capacity_factor.rename(
        columns={"technology_description": "technology"},
        inplace=True,
    )

    capacity_factor_tech_region = capacity_factor.groupby(
        ["model_region", "technology_description"], as_index=False
    )[["potential_generation_mwh", "net_generation_mwh"]].sum()

    # actually calculate capacity factor wooo!
    capacity_factor_tech_region["capacity_factor"] = (
        capacity_factor_tech_region["net_generation_mwh"]
        / capacity_factor_tech_region["potential_generation_mwh"]
    )

    capacity_factor_tech_region.rename(
        columns={"model_region": "region", "technology_description": "technology"},
        inplace=True,
    )

    logger.debug(capacity_factor_tech_region)

    return plant_tech_capacity_factor, capacity_factor_tech_region


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
        corresponding fuel (coal, natural_gas, uranium, or distillate) or "None".

    Raises
    ------
    KeyError
        The model region is not mapped to a fuel region in 'aeo_fuel_region_map'
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
        scenario = settings.get("aeo_fuel_scenarios", {}).get(fuel)
        model_year = settings["model_year"]
        if not scenario:
            if fuel not in settings.get("user_fuel_price", []) or []:
                raise KeyError(
                    f"The fuel type '{fuel}' is not in the settings parameters "
                    "'aeo_fuel_scenarios' or 'user_fuel_price'. All fuels listed in "
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
                                (df["technology"].str.contains(tech, case=False))
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
                            (df["technology"].str.contains(tech, case=False))
                            & (df["Fuel"].isna()),
                            "Fuel",
                        ] = fuel
        else:
            for aeo_region, model_regions in settings["aeo_fuel_region_map"].items():
                fuel_name = ("_").join([aeo_region, scenario, fuel])
                assert (
                    fuel_prices.query(
                        "year==@model_year & full_fuel_name==@fuel_name"
                    ).empty
                    is False
                ), f"{fuel_name} doesn't show up in {model_year}"

                df.loc[
                    (df["technology"].str.contains(eia_tech, case=False))
                    & df["region"].isin(model_regions),
                    "Fuel",
                ] = fuel_name

                if atb_tech is not None:
                    for tech in atb_tech:
                        df.loc[
                            (df["technology"].str.contains(tech, case=False))
                            & (df["region"].isin(model_regions))
                            & (df["Fuel"].isna()),
                            "Fuel",
                        ] = fuel_name

    for ccs_tech, ccs_fuel in (settings.get("ccs_fuel_map") or {}).items():
        ccs_base_name = ("_").join(ccs_fuel.split("_")[:-1])
        if ccs_base_name in (settings.get("aeo_fuel_scenarios", {}) or {}).keys():
            scenario = settings["aeo_fuel_scenarios"][ccs_base_name]
            for aeo_region, model_regions in settings["aeo_fuel_region_map"].items():
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
                        (df["technology"].str.contains(ccs_tech, case=False))
                        & (df["region"] == region),
                        "Fuel",
                    ] = ccs_fuel_name
            else:
                df.loc[
                    (df["technology"].str.contains(ccs_tech, case=False)),
                    "Fuel",
                ] = ccs_fuel
        else:
            logger.warning(
                f"The fuel {ccs_fuel} is included in settings parameter `ccs_fuel_map` "
                "but it can't be matched against an AEO or user fuel. CCS fuels should "
                "have the format <fuel name>_ccs<capture rate>, where the capture rate "
                "is optional. The <fuel name> should match a fuel from `aeo_fuel_scenarios' "
                "or `user_fuel_prices`."
            )

    # Replace AEO region name with model region in cases where users are modifying AEO price
    model_aeo_region_map = reverse_dict_of_lists(
        settings.get("aeo_fuel_region_map", {})
    )
    for region, adj in (settings.get("regional_fuel_adjustments", {}) or {}).items():
        aeo_region = model_aeo_region_map.get(region)
        if not aeo_region:
            raise KeyError(
                f"There is no mapping of the model region {region} to an AEO fuel region "
                "in the settings parameter 'aeo_fuel_region_map'."
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

    df.loc[df["Fuel"].isna(), "Fuel"] = "None"

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
                capex_mw_mile = resource_df["region"].map(params["capex_mw_mile"])
        elif isinstance(params["capex_mw_mile"], Number):
            capex_mw_mile = params["capex_mw_mile"]
        else:
            raise TypeError(
                f"{SETTING}.{ttype}.capex_mw_mile should be numeric or a dictionary"
                f" of <region>: <capex>, not {params['capex_mw_mile']}"
            )
        resource_df[f"{ttype}_capex"] = (
            capex_mw_mile.fillna(0) * resource_df[f"{ttype}_miles"]
        )
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


def save_weighted_hr(weighted_unit_hr, pudl_engine):
    pass


def add_dg_resources(
    pg_engine: sqlalchemy.engine.Engine,
    settings: dict,
    gen_df: pd.DataFrame = pd.DataFrame(),
) -> pd.DataFrame:
    """Add distributed generation resources as rows in a generators dataframe

    Parameters
    ----------
    pg_engine : sqlalchemy.engine.Engine
        Connection to database with hourly generation values. Needed if installed DG
        capacity is calculated as a fraction of load.
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
        values for the columns "technology", "region", "Existing_Cap_MW", and "profile".
    """
    dg_profiles = make_distributed_gen_profiles(pg_engine, settings)
    df = pd.DataFrame(
        columns=["technology", "region", "cluster", "Existing_Cap_MW", "profile"],
        index=range(len(dg_profiles.columns)),
    )

    for idx, (region, s) in enumerate(dg_profiles.iteritems()):
        cap = s.max()
        df.loc[idx, "profile"] = (s / cap).round(3).to_numpy()
        df.loc[idx, "Existing_Cap_MW"] = cap.round(0).astype(int)
    df["technology"] = "distributed_generation"
    df["region"] = dg_profiles.columns
    df["cluster"] = 1

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
            "\n\n**************************\n"
            f"The storage technology(ies) {missing_techs} have some existing generators "
            "with energy capacity (MWh) values and some where the energy capacity is "
            "missing. You have not included these technologies in the settings parameter "
            "'energy_storage_duration', which is used to fill missing energy capacity "
            "data.\n\nNOTE: This is not a comprehensive list of technologies that *should* "
            "be included in 'energy_storage_duration'. Technologies without any existing "
            "energy capacity data might also be missing.\n"
            "**************************\n"
        )
    for tech, val in energy_storage_duration.items():
        if isinstance(val, dict):
            tech_regions = val.keys()
            df.loc[
                df[tech_col].isna(), "technology_description"
            ] = "missing_tech_description"
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


def load_plants_860(
    pudl_engine: sqlalchemy.engine.Engine, data_years: List[int] = [2020]
) -> pd.DataFrame:
    """Load database table with EIA860 information on plants

    Parameters
    ----------
    pudl_engine : sqlalchemy.engine.Engine
        Connection to PUDL database
    data_years : List[int], optional
        Year of data to keep, by default [2020]

    Returns
    -------
    pd.DataFrame
        Includes all columns from the database table
    """
    data_years = [str(y) for y in data_years]
    s = f"""
    SELECT * from plants_eia860
    WHERE strftime('%Y',report_date) in ({','.join(['?']*len(data_years))})
    """
    plants = pd.read_sql_query(
        s, pudl_engine, params=data_years, parse_dates=["report_date"]
    )

    return plants


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


def add_860m_storage_mwh(
    gen_df: pd.DataFrame, operating_860m: pd.DataFrame, storage_techs: List[str] = None
) -> pd.DataFrame:
    idx_cols = ["plant_id_eia", "generator_id"]
    gen_df = gen_df.set_index(idx_cols)

    if not storage_techs:
        storage_techs = [
            "Batteries",
            "Natural Gas with Compressed Air Storage",
            "Flywheels",
        ]

    storage_860m = operating_860m.loc[
        operating_860m.technology_description.isin(storage_techs), :
    ].set_index(idx_cols)
    gen_df.loc[storage_860m.index, "capacity_mwh"] = storage_860m["capacity_mwh"]
    # gen_df = pd.merge(
    #     gen_df, storage_860m[idx_cols + ["capacity_mwh"]], how="left", on=idx_cols
    # )

    return gen_df.reset_index()


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
        pudl_engine,
        pudl_out,
        pg_engine,
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
        self.pudl_engine = pudl_engine
        self.pudl_out = pudl_out
        self.pg_engine = pg_engine
        self.settings = settings
        self.current_gens = current_gens
        if self.settings.get("num_clusters") is None:
            self.current_gens = False
        self.sort_gens = sort_gens
        self.model_regions_gdf = load_ipm_shapefile(self.settings)
        self.weighted_unit_hr = None
        self.supplement_with_860m = supplement_with_860m
        self.multi_period = multi_period
        self.include_retired_cap = include_retired_cap
        self.cluster_builder = build_resource_clusters(
            self.settings.get("RESOURCE_GROUPS"),
            self.settings.get("RESOURCE_GROUP_PROFILES"),
        )

        if self.current_gens:
            self.data_years = self.settings.get("eia_data_years") or []

            self.gens_860 = load_generator_860_data(self.pudl_engine, self.data_years)
            self.gens_entity = pd.read_sql_table(
                "generators_entity_eia", self.pudl_engine
            )

            bga = self.pudl_out.bga_eia860()
            self.bga = bga.loc[
                bga.report_date.dt.year.isin(self.data_years), :
            ].drop_duplicates(["plant_id_eia", "generator_id"])

            logger.info("Loading map of plants to IPM regions")
            self.plant_region_map = load_plant_region_map(
                self.gens_860,
                self.pudl_engine,
                self.pg_engine,
                self.settings,
                self.model_regions_gdf,
                table=plant_region_map_table,
            )

            self.gen_923 = load_923_gen_fuel_data(
                self.pudl_engine,
                self.pudl_out,
                model_region_map=self.plant_region_map,
                data_years=self.data_years,
            )

            self.eia_860m = load_860m(self.settings)
            self.operating_860m = self.eia_860m["operating"]
            self.operating_860m = self.remove_860_duplicates(self.operating_860m)
            self.planned_860m = self.eia_860m["planned"]
            self.planned_860m = self.remove_860_duplicates(self.planned_860m)
            self.canceled_860m = self.eia_860m["canceled"]
            self.canceled_860m = self.remove_860_duplicates(self.canceled_860m)
            self.retired_860m = self.eia_860m["retired"]
            self.retired_860m = self.remove_860_duplicates(self.retired_860m)

            # self.ownership = load_ownership_eia860(self.pudl_engine, self.data_years)
            self.plants_860 = load_plants_860(self.pudl_engine, self.data_years)
            # self.utilities_eia = load_utilities_eia(self.pudl_engine)
        else:
            self.existing_resources = pd.DataFrame()
        self.fuel_prices = fetch_fuel_prices(self.settings).pipe(
            modify_fuel_prices,
            self.settings.get("aeo_fuel_region_map"),
            self.settings.get("regional_fuel_adjustments"),
        )
        self.atb_hr = fetch_atb_heat_rates(self.pg_engine, self.settings)
        self.coal_fgd = pd.read_csv(DATA_PATHS["coal_fgd"])

    def fill_na_heat_rates(self, s):
        """Fill null heat rate values with the median of the series. Not many null
        values are expected.

        Parameters
        ----------
        df : DataFrame
            Must contain the column 'heat_rate_mmbtu_mwh'

        Returns
        -------
        Dataframe
            Same as input but with any null values replaced by the median.
        """
        if s.isnull().any():
            median_hr = s.median()
            return s.fillna(median_hr)
        else:
            return s
        # median_hr = df["heat_rate_mmbtu_mwh"].median()
        # df["heat_rate_mmbtu_mwh"].fillna(median_hr, inplace=True)

        # return df

    def remove_860_duplicates(self, df_860: pd.DataFrame()) -> pd.DataFrame():
        """
        Remove rows from an EIA 860 DF that contain a duplicate plant-generator pairing.
        """
        df = df_860.copy(deep=True)
        df = df.set_index(["plant_id_eia", "generator_id"])
        df = df.loc[~df.index.duplicated(), :].reset_index()
        return df

    def create_demand_response_gen_rows(self):
        """Create rows for demand response/management resources to include in the
        generators file.

        Returns
        -------
        DataFrame
            One row for each region/DSM resource with values in all columns filled.
        """
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
                f"Your 'flexible_demand_resources' settings parameter has the value 'None' "
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
        dr_rows["Fuel"] = "None"
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
        if self.gens_860.technology_description.isna().any():
            self.gens_860 = fill_missing_tech_descriptions(self.gens_860)
        self.gens_860_model = (
            self.gens_860.pipe(
                supplement_generator_860_data,
                self.gens_entity,
                self.bga,
                self.plant_region_map,
                self.settings,
            )
            .pipe(remove_canceled_860m, self.canceled_860m)
            .pipe(remove_retired_860m, self.retired_860m)
            .pipe(update_planned_retirement_date_860m, self.operating_860m.copy())
            .pipe(update_operating_date_860m, self.operating_860m.copy())
            .pipe(label_retirement_year, self.settings, add_additional_retirements=True)
            .pipe(
                label_retirement_year,
                self.settings,
                age_col="original_planned_operating_date",
            )
            .pipe(label_small_hydro, self.settings, by=["plant_id_eia"])
            .pipe(
                group_technologies,
                self.settings.get("group_technologies"),
                self.settings.get("tech_groups", {}) or {},
                self.settings.get("regional_no_grouping", {}) or {},
            )
        )
        self.gens_860_model = self.gens_860_model.pipe(
            modify_cc_prime_mover_code, self.gens_860_model
        )
        self.gens_860_model.drop_duplicates(inplace=True)

        self.annual_gen_hr_923 = (
            self.gen_923.pipe(modify_cc_prime_mover_code, self.gens_860_model)
            .pipe(group_gen_by_year_fuel_primemover)
            .pipe(add_923_heat_rate)
        )

        # Add heat rates to the data we already have from 860
        logger.info("Loading heat rate data for units and generator/fuel combinations")
        self.prime_mover_hr_map = plant_pm_heat_rates(self.annual_gen_hr_923)
        if self.weighted_unit_hr is None:
            self.weighted_unit_hr = unit_generator_heat_rates(
                self.pudl_out, self.data_years
            )
        else:
            logger.info("Using unit heat rates from previous round.")
        self.weighted_unit_hr["unit_id_pg"] = self.weighted_unit_hr[
            "unit_id_pudl"
        ].astype("object")

        # Merge the PUDL calculated heat rate data and set the index for easy
        # mapping using plant/prime mover heat rates from 923
        hr_cols = ["plant_id_eia", "unit_id_pg", "heat_rate_mmbtu_mwh"]
        idx = ["plant_id_eia", "prime_mover_code", "energy_source_code_1"]
        self.units_model = self.gens_860_model.merge(
            self.weighted_unit_hr[hr_cols],
            on=["plant_id_eia", "unit_id_pg"],
            how="left",
        ).set_index(idx)

        logger.info(
            f"Units model technologies are "
            f"{self.units_model.technology_description.unique().tolist()}"
        )
        # print(units_model.head())

        logger.info(
            "Assigning technology/fuel heat rates where unit heat rates are not "
            "available"
        )
        self.units_model.loc[
            self.units_model.heat_rate_mmbtu_mwh.isnull(), "heat_rate_mmbtu_mwh"
        ] = self.units_model.loc[
            self.units_model.heat_rate_mmbtu_mwh.isnull()
        ].index.map(
            self.prime_mover_hr_map
        )

        self.units_model.loc[
            self.units_model.heat_rate_mmbtu_mwh > 35, "heat_rate_mmbtu_mwh"
        ] = self.units_model.loc[self.units_model.heat_rate_mmbtu_mwh > 35].index.map(
            self.prime_mover_hr_map
        )

        # Set heat rates < 6.36 (ATB NGCC) or > 35 mmbtu/MWh to nan. Don't change heat rates of 0,
        # which is when there is positive generation and no fuel use (pumped storage)
        self.units_model.loc[
            (
                (self.units_model.heat_rate_mmbtu_mwh < 6.36)
                & ~(
                    self.units_model.technology_description.str.contains(
                        "pumped", case=False
                    )
                )
            )
            | (self.units_model.heat_rate_mmbtu_mwh > 35),
            "heat_rate_mmbtu_mwh",
        ] = np.nan

        # Fill any null heat rate values for each tech
        for tech in self.units_model["technology_description"].unique():
            self.units_model.loc[
                self.units_model.technology_description == tech, "heat_rate_mmbtu_mwh"
            ] = self.fill_na_heat_rates(
                self.units_model.loc[
                    self.units_model.technology_description == tech,
                    "heat_rate_mmbtu_mwh",
                ]
            )
        # assert (
        #     self.units_model["heat_rate_mmbtu_mwh"].isnull().any() is False
        # ), "There are still some null heat rate values"
        # from IPython import embed

        # embed()
        logger.info(
            f"Units model technologies are "
            f"{self.units_model.technology_description.unique().tolist()}"
        )
        if self.supplement_with_860m:
            logger.info(
                f"Before adding proposed generators, {len(self.units_model)} units with "
                f"{self.units_model[self.settings['capacity_col']].sum()} MW capacity"
            )
            self.proposed_gens = import_proposed_generators(
                planned=self.planned_860m,
                settings=self.settings,
                model_regions_gdf=self.model_regions_gdf,
            )
            self.new_860m_gens = import_new_generators(
                operating_860m=self.operating_860m,
                gens_860=self.gens_860_model,
                settings=self.settings,
                model_regions_gdf=self.model_regions_gdf,
            )
            # Add new/proposed generators to plant_region_map
            self.plant_region_map = pd.concat(
                [
                    self.plant_region_map,
                    self.proposed_gens.reset_index()[
                        ["plant_id_eia", "model_region"]
                    ].drop_duplicates(),
                    self.new_860m_gens.reset_index()[
                        ["plant_id_eia", "model_region"]
                    ].drop_duplicates(),
                ]
            ).drop_duplicates(subset=["plant_id_eia", "model_region"])
            # embed()
            logger.info(
                f"Proposed gen technologies are "
                f"{self.proposed_gens.technology_description.unique().tolist()}"
            )
            logger.info(
                f"{self.proposed_gens[self.settings['capacity_col']].sum()} MW proposed"
            )
            self.units_model = pd.concat(
                [self.proposed_gens, self.units_model, self.new_860m_gens], sort=False
            )

        # Create a pudl unit id based on plant and generator id where one doesn't exist.
        # This is used later to match the cluster numbers to plants
        self.units_model.reset_index(inplace=True)
        self.units_model = self.units_model.drop_duplicates(
            subset=["plant_id_eia", "generator_id"]
        )
        if self.supplement_with_860m:
            self.units_model = add_860m_storage_mwh(
                self.units_model,
                self.new_860m_gens.reset_index(),
                storage_techs=["Batteries"],
            )
        else:
            self.units_model["capacity_mwh"] = 0
        if self.settings.get("energy_storage_duration"):
            self.units_model = energy_storage_mwh(
                self.units_model,
                self.settings["energy_storage_duration"],
                "technology_description",
                self.settings["capacity_col"],
                "capacity_mwh",
            )
        else:
            logger.warning(
                "\n\n*******************************\n"
                "The parameter 'energy_storage_duration' is not included in your settings "
                "file. This parameter converts MW to energy capacity (MWh) for existing "
                "generators where data is otherwise not available. For example, EIA-860m "
                "generally does not have MWh for battery storage units installed in the "
                "current calendar year. The energy capacity of existing storage clusters "
                "may be incorrect if you do not include this parameter!\n"
                "*******************************\n"
            )
        self.units_model["plant_id_eia"] = self.units_model["plant_id_eia"].astype(
            "Int64"
        )
        self.units_model.loc[self.units_model.unit_id_pg.isnull(), "unit_id_pg"] = (
            self.units_model.loc[
                self.units_model.unit_id_pg.isnull(), "plant_id_eia"
            ].astype(str)
            + "_"
            + self.units_model.loc[
                self.units_model.unit_id_pg.isnull(), "generator_id"
            ].astype(str)
        ).values
        self.units_model.set_index(idx, inplace=True)

        logger.info("Calculating plant O&M costs")
        techs = self.settings["num_clusters"].keys()
        self.units_model = (
            self.units_model.rename(columns={"technology_description": "technology"})
            .query("technology.isin(@techs).values")
            .pipe(
                atb_fixed_var_om_existing,
                self.atb_hr,
                self.settings,
                self.pg_engine,
                self.coal_fgd,
            )
        )

        # logger.info(
        #     f"After adding proposed, units model technologies are "
        #     f"{self.units_model.technology_description.unique().tolist()}"
        # )
        logger.info(
            f"After adding proposed generators, {len(self.units_model)} units with "
            f"{self.units_model[self.settings['capacity_col']].sum()} MW capacity"
        )

        techs = list(self.settings["num_clusters"])

        num_clusters = {}
        for region in self.settings["model_regions"]:
            num_clusters[region] = self.settings["num_clusters"].copy()

        if self.settings.get("alt_num_clusters"):
            for region in self.settings["alt_num_clusters"]:
                for tech, cluster_size in self.settings["alt_num_clusters"][
                    region
                ].items():
                    num_clusters[region][tech] = cluster_size

        self.retired = self.units_model.loc[
            ~(self.units_model.retirement_year > self.settings["model_year"]), :
        ]
        self.period_retired = self.units_model.loc[
            ~(self.units_model.retirement_year > self.settings["model_year"])
            & (
                self.units_model.retirement_year
                >= self.settings["model_first_planning_year"]
            ),
            :,
        ]
        self.retired_index = self.retired.set_index(
            ["plant_id_eia", "unit_id_pg"]
        ).index
        if self.settings.get("cluster_with_retired_gens", False) is True:
            logger.info(
                "\n\nAge-retired gens are included for clustering to keep consistent "
                "cluster assignments across periods. \n\n"
            )
            region_tech_grouped = self.units_model.loc[
                self.units_model.technology.isin(techs), :
            ].groupby(["model_region", "technology"])
        else:
            logger.info(
                "\n\nAge-retired gens are NOT included for clustering. This may lead to "
                "inconsistent cluster assignments across periods. If you want to change "
                "this behavior, add 'cluster_with_retired_gens: true' to your settings.\n\n"
            )
            region_tech_grouped = self.units_model.loc[
                (self.units_model.technology.isin(techs))
                & ~(self.units_model.retirement_year <= self.settings["model_year"]),
                :,
            ].groupby(["model_region", "technology"])

        # gens_860 lost the ownership code... refactor this!
        # self.all_gens_860 = load_generator_860_data(self.pudl_engine, self.data_years)
        # Getting weighted ownership for each unit, which will be used below.
        # self.weighted_ownership = weighted_ownership_by_unit(
        #     self.units_model, self.all_gens_860, self.ownership, self.settings
        # )

        # For each group, cluster and calculate the average size/min load/heat rate
        # logger.info("Creating technology clusters by region")
        logger.info("Creating technology clusters by region")
        unit_list = []
        self.cluster_list = []
        alt_cluster_method = self.settings.get("alt_cluster_method") or {}

        for _, df in region_tech_grouped:
            region, tech = _
            grouped = group_units(df, self.settings)
            grouped["technology"] = tech

            # This is bad. Should be setting up a dictionary of objects that picks the
            # correct clustering method. Can't keep doing if statements as the number of
            # methods grows. CHANGE LATER.
            if not alt_cluster_method:
                # Allow users to set value as None and not cluster units, or when the
                # num clusters is equal to the number of units.
                if num_clusters[region][tech] is None or num_clusters[region][
                    tech
                ] == len(grouped):
                    if self.multi_period:
                        grouped.loc[
                            grouped.index.isin(self.retired_index),
                            [
                                self.settings["capacity_col"],
                                "minimum_load_mw",
                                "capacity_mwh",
                            ],
                        ] = np.nan
                    else:
                        grouped = grouped.loc[
                            ~grouped.index.isin(self.retired_index), :
                        ]

                    grouped["cluster"] = np.arange(len(grouped)) + 1
                    unit_list.append(grouped)
                    if not grouped.empty:
                        _df = calc_unit_cluster_values(
                            grouped, self.settings["capacity_col"], tech
                        )
                        _df["region"] = region

                        self.cluster_list.append(_df)
                        continue
                elif num_clusters[region][tech] > 0:
                    cluster_cols = [
                        "Fixed_OM_Cost_per_MWyr",
                        # "Var_OM_Cost_per_MWh",
                        # "minimum_load_mw",
                        "heat_rate_mmbtu_mwh",
                    ]
                    if len(grouped) < num_clusters[region][tech]:
                        s = f"""
    *****************************
    The technology {tech} in region {region} has only {len(grouped)} operating units,
    which is less than the {num_clusters[region][tech]} clusters you specified.
    The number of clusters has been set equal to the number of units.
    *****************************
                            """
                        logger.info(s)
                        num_clusters[region][tech] = len(grouped)
                    clusters = cluster.KMeans(
                        n_clusters=num_clusters[region][tech], random_state=6
                    ).fit(
                        preprocessing.StandardScaler().fit_transform(
                            grouped[cluster_cols]
                        )
                    )

                    grouped["cluster"] = (
                        clusters.labels_ + 1
                    )  # Change to 1-index for julia
                else:
                    continue
            else:
                if (
                    region in alt_cluster_method
                    and tech in alt_cluster_method[region]["technology_description"]
                ):
                    grouped = cluster_by_owner(
                        df,
                        self.weighted_ownership,
                        # self.ownership,
                        self.plants_860,
                        region,
                        tech,
                        self.settings,
                    )

                elif num_clusters[region][tech] > 0:
                    clusters = cluster.KMeans(
                        n_clusters=num_clusters[region][tech], random_state=6
                    ).fit(preprocessing.StandardScaler().fit_transform(grouped))

                    grouped["cluster"] = (
                        clusters.labels_ + 1
                    )  # Change to 1-index for julia
                else:
                    continue

            # Saving individual unit data for later analysis (if needed)
            unit_list.append(grouped)

            # Don't add technologies with specified 0 clusters
            if num_clusters[region][tech] != 0:
                # Remove retired units before calculating cluster operating values
                if self.multi_period:
                    grouped.loc[
                        grouped.index.isin(self.retired_index),
                        [
                            self.settings["capacity_col"],
                            "minimum_load_mw",
                            "capacity_mwh",
                        ],
                    ] = np.nan
                else:
                    grouped = grouped.loc[~grouped.index.isin(self.retired_index), :]
                if not grouped.empty:
                    _df = calc_unit_cluster_values(
                        grouped, self.settings["capacity_col"], tech
                    )
                    _df["region"] = region
                    _df["plant_id_eia"] = (
                        grouped.reset_index()
                        .groupby("cluster")["plant_id_eia"]
                        .apply(list)
                    )
                    _df["unit_id_pg"] = (
                        grouped.reset_index()
                        .groupby("cluster")["unit_id_pg"]
                        .apply(list)
                    )

                    self.cluster_list.append(_df)

        # Save some data about individual units for easy access
        self.all_units = pd.concat(unit_list, sort=False)
        self.all_units = pd.merge(
            self.units_model,  # .reset_index(),
            self.all_units.reset_index()[
                ["plant_id_eia", "unit_id_pg", "cluster", "technology"]
            ],
            on=["plant_id_eia", "unit_id_pg", "technology"],
            how="inner",
        ).merge(
            self.plants_860[["plant_id_eia", "utility_id_eia"]],
            on=["plant_id_eia"],
            how="left",
        )

        logger.info("Finalizing generation clusters")
        self.results = pd.concat(self.cluster_list)
        logger.info(
            f"Results technologies are {self.results.technology.unique().tolist()}"
        )

        # if self.settings.get("region_wind_pv_cap_fn"):
        #     from powergenome.external_data import overwrite_wind_pv_capacity

        #     logger.info("Setting existing wind/pv using external file")
        #     self.results = overwrite_wind_pv_capacity(self.results, self.settings)
        for col in ["region", "technology", "cluster"]:
            if col in self.results.index.names:
                self.results = self.results.reset_index()

        self.results = self.results.set_index(["region", "technology", "cluster"])
        self.results.rename(
            columns={
                self.settings["capacity_col"]: "Cap_Size",
                "heat_rate_mmbtu_mwh": "Heat_Rate_MMBTU_per_MWh",
            },
            inplace=True,
        )

        # Calculate average capacity factors
        if self.settings.get("derate_techs"):
            (
                plant_tech_cf,
                self.region_tech_capacity_factors,
            ) = gentype_region_capacity_factor(
                self.pudl_engine,
                self.units_model[["plant_id_eia", "model_region"]],
                self.settings,
            )
            self.all_units = pd.merge(
                self.all_units,
                plant_tech_cf,
                how="left",
                on=["plant_id_eia", "technology"],
            )

            self.results = pd.merge(
                self.results.reset_index(),
                self.region_tech_capacity_factors[
                    ["region", "technology", "capacity_factor"]
                ],
                on=["region", "technology"],
                how="left",
            )

            derate_techs = self.settings["derate_techs"]
            self.results.loc[:, "unmodified_cap_size"] = self.results.loc[
                :, "Cap_Size"
            ].copy()
            self.results.loc[
                self.results["technology"].isin(derate_techs), "Cap_Size"
            ] = (
                self.results.loc[
                    self.results["technology"].isin(derate_techs),
                    "unmodified_cap_size",
                ]
                * self.results.loc[
                    self.results["technology"].isin(derate_techs), "capacity_factor"
                ]
            )

        self.all_units = self.all_units.rename(columns={"model_region": "region"})
        self.all_units["Resource"] = (
            self.all_units["region"]
            + "_"
            + snake_case_col(self.all_units["technology"])
            + "_"
            + self.all_units["cluster"].astype(int).astype(str)
        )

        if self.settings.get("extra_outputs"):
            self.all_units.to_csv(
                Path(self.settings["extra_outputs"]) / "existing_gen_units.csv",
                index=False,
            )
        # Round Cap_size to prevent GenX error.
        self.results = self.results.round(3)
        self.results["Cap_Size"] = self.results["Cap_Size"]
        self.results["Existing_Cap_MW"] = self.results.Cap_Size * self.results.num_units

        # A cap size of 0 causes issues in GenX with thermal commitment.
        self.results["Cap_Size"] = self.results["Cap_Size"].fillna(1).replace(0, 1)
        if self.settings.get("derate_capacity"):
            self.results["unmodified_existing_cap_mw"] = (
                self.results["unmodified_cap_size"] * self.results["num_units"]
            )

        if self.settings.get("region_wind_pv_cap_fn"):
            from powergenome.external_data import overwrite_wind_pv_capacity

            logger.info("Setting existing wind/pv using external file")
            self.results = overwrite_wind_pv_capacity(self.results, self.settings)

        if self.settings.get("dg_as_resource"):
            logger.info(
                "\n **************** \nDistributed generation is being added as generating"
                " resources. The capacity of DG in each region is increased by "
                f"{self.settings.get('avg_distribution_loss', 0):%} to account for no "
                "distribution losses.\n"
            )
            self.results = add_dg_resources(
                self.pg_engine, self.settings, self.results.reset_index()
            )
        else:
            self.results["profile"] = None

        # Add fixed/variable O&M based on NREL atb
        self.results = (
            self.results.reset_index()
            # .pipe(
            #     atb_fixed_var_om_existing, self.atb_costs, self.atb_hr, self.settings
            # )
            # .pipe(atb_new_generators, self.atb_costs, self.atb_hr, self.settings)
            .pipe(startup_fuel, self.settings)
            .pipe(add_fuel_labels, self.fuel_prices, self.settings)
            .pipe(startup_nonfuel_costs, self.settings)
            .pipe(add_genx_model_tags, self.settings)
        )

        if self.sort_gens:
            logger.info("Sorting new resources alphabetically.")
            self.results = self.results.sort_values(["region", "technology"])

        # self.results = self.results.rename(columns={"technology": "Resource"})
        self.results["Resource"] = (
            self.results["region"]
            + "_"
            + snake_case_col(self.results["technology"])
            + "_"
            + self.results["cluster"].astype(str)
        )
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
                    self.all_units,
                    self.settings["model_first_planning_year"],
                    self.settings["model_year"],
                    self.settings.get("capacity_col", "capacity_mw"),
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
                    f"Multiple existing resource groups match EIA technology"
                    + row.technology
                )
            group = groups[0]
            if group.profiles is None:
                # Resource group has no profiles
                continue
            if row.region in (self.settings.get("region_aggregations", {}) or {}):
                ipm_regions = self.settings.get("region_aggregations", {})[row.region]
            else:
                ipm_regions = [row.region]
            metadata = group.metadata.read()
            if not metadata["ipm_region"].isin(ipm_regions).any():
                # Resource group has no resources in selected IPM regions
                continue
            clusters = group.get_clusters(
                ipm_regions=ipm_regions,
                max_clusters=1,
                utc_offset=self.settings.get("utc_offset", 0),
            )
            self.results["profile"][i] = clusters["profile"][0]

        self.results = rename_gen_cols(self.results)

        # Drop old index cols from df
        self.results.drop(columns=["level_0", "index"], errors="ignore", inplace=True)

        return self.results

    def create_new_generators(self):
        self.offshore_spur_costs = fetch_atb_offshore_spur_costs(
            self.pg_engine, self.settings
        )
        self.atb_costs = fetch_atb_costs(
            self.pg_engine, self.settings, self.offshore_spur_costs
        )

        self.new_generators = atb_new_generators(
            self.atb_costs, self.atb_hr, self.settings, self.cluster_builder
        )

        if not self.new_generators.empty:
            self.new_generators = (
                self.new_generators.pipe(startup_fuel, self.settings)
                .pipe(add_fuel_labels, self.fuel_prices, self.settings)
                .pipe(startup_nonfuel_costs, self.settings)
            )

            if self.sort_gens:
                logger.info("Sorting new resources alphabetically.")
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
                calculate_transmission_inv_cost, self.settings, self.offshore_spur_costs
            ).pipe(add_transmission_inv_cost, self.settings)

        if self.settings.get("demand_response_fn") or self.settings.get(
            "electrification_stock_fn"
        ):
            dr_rows = self.create_demand_response_gen_rows()
            self.new_generators = pd.concat(
                [self.new_generators, dr_rows], sort=False, ignore_index=True
            )
        self.new_generators = add_genx_model_tags(self.new_generators, self.settings)
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

        return self.new_generators

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
            logger.info(
                f"Capacity of {self.all_resources['Existing_Cap_MW'].sum()} MW in final clusters"
            )

        return self.all_resources
