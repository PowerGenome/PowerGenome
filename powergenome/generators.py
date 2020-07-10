import collections
import logging
from numbers import Number
import re

import requests

import geopandas as gpd
import numpy as np
import pandas as pd
from pathlib import Path
import pudl
from bs4 import BeautifulSoup
from flatten_dict import flatten
from powergenome.cluster_method import cluster_by_owner, weighted_ownership_by_unit
from powergenome.eia_opendata import fetch_fuel_prices
from powergenome.external_data import (
    make_demand_response_profiles,
    demand_response_resource_capacity,
    add_resource_max_cap_spur,
)
from powergenome.load_data import (
    load_ipm_plant_region_map,
    load_ownership_eia860,
    load_plants_860,
    load_utilities_eia,
)
from powergenome.nrelatb import (
    atb_fixed_var_om_existing,
    atb_new_generators,
    fetch_atb_costs,
    fetch_atb_heat_rates,
    investment_cost_calculator,
)
from powergenome.params import DATA_PATHS, IPM_GEOJSON_PATH
from powergenome.price_adjustment import inflation_price_adjustment
from powergenome.util import (
    download_save,
    map_agg_region_names,
    reverse_dict_of_lists,
    snake_case_col,
)
from scipy.stats import iqr
from sklearn import cluster, preprocessing
from xlrd import XLRDError

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
}

TRANSMISSION_TYPES = ["spur", "offshore_spur", "tx"]


def fill_missing_tech_descriptions(df):
    """
    EIA 860 records before 2014 don't have a technology description. If we want to
    include any of this data in the historical record (e.g. heat rates or capacity
    factors) then they need to be filled in.

    Parameters
    ----------
    df : dataframe
        A pandas dataframe with columns plant_id_eia, generator_id, and
        technology_description.

    Returns
    -------
    dataframe
        Same data that came in, but with missing technology_description values filled
        in.
    """
    start_len = len(df)
    df = df.sort_values(by="report_date")
    df_list = []
    for _, _df in df.groupby(["plant_id_eia", "generator_id"], as_index=False):
        _df["technology_description"].fillna(method="bfill", inplace=True)
        df_list.append(_df)
    results = pd.concat(df_list, ignore_index=True, sort=False)

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


def startup_fuel(df, settings):
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
        Modified dataframe with the new column "Start_fuel_MMBTU_per_MW".
    """
    df["Start_fuel_MMBTU_per_MW"] = 0
    for eia_tech, fuel_use in (settings.get("startup_fuel_use") or {}).items():
        atb_tech = settings["eia_atb_tech_map"][eia_tech].split("_")[0]

        df.loc[df["technology"] == eia_tech, "Start_fuel_MMBTU_per_MW"] = fuel_use
        df.loc[
            df["technology"].str.contains(atb_tech), "Start_fuel_MMBTU_per_MW"
        ] = fuel_use

    return df


def startup_nonfuel_costs(df, settings):
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
        Modified df with new column "Start_cost_per_MW"
    """
    logger.info("Adding non-fuel startup costs")
    target_usd_year = settings["target_usd_year"]

    vom_costs = settings["startup_vom_costs_mw"]
    vom_usd_year = settings["startup_vom_costs_usd_year"]

    logger.info(
        f"Changing non-fuel VOM costs from {vom_usd_year} to " f"{target_usd_year}"
    )
    for key, cost in vom_costs.items():
        vom_costs[key] = inflation_price_adjustment(
            price=cost, base_year=vom_usd_year, target_year=target_usd_year
        )

    startup_type = settings["startup_costs_type"]
    startup_costs = settings[startup_type]
    startup_costs_usd_year = settings["startup_costs_per_cold_start_usd_year"]

    logger.info(
        f"Changing non-fuel startup costs from {vom_usd_year} to {target_usd_year}"
    )
    for key, cost in startup_costs.items():
        startup_costs[key] = inflation_price_adjustment(
            price=cost, base_year=startup_costs_usd_year, target_year=target_usd_year
        )

    df["Start_cost_per_MW"] = 0

    for existing_tech, cost_tech in settings["existing_startup_costs_tech_map"].items():
        total_startup_costs = vom_costs[cost_tech] + startup_costs[cost_tech]
        df.loc[
            df["technology"].str.contains(existing_tech), "Start_cost_per_MW"
        ] = total_startup_costs

    for new_tech, cost_tech in settings["new_build_startup_costs"].items():
        total_startup_costs = vom_costs[cost_tech] + startup_costs[cost_tech]
        df.loc[
            df["technology"].str.contains(new_tech), "Start_cost_per_MW"
        ] = total_startup_costs
    df.loc[:, "Start_cost_per_MW"] = df.loc[:, "Start_cost_per_MW"].round(0)

    # df.loc[df["technology"].str.contains("Nuclear"), "Start_cost_per_MW"] = "FILL VALUE"

    return df


def group_technologies(df, settings):
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
    if settings.get("group_technologies"):

        df["_technology"] = df["technology_description"]
        for tech, group in settings["tech_groups"].items():
            df.loc[df["technology_description"].isin(group), "_technology"] = tech

        for region, tech_list in (settings.get("regional_no_grouping") or {}).items():
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
        crs={"init": "epsg:4326"},
    )

    if model_hydro_gdf.crs != model_regions_gdf.crs:
        model_hydro_gdf = model_hydro_gdf.to_crs(model_regions_gdf.crs)

    model_hydro_gdf = gpd.sjoin(model_regions_gdf, model_hydro_gdf)
    model_hydro_gdf = model_hydro_gdf.rename(columns={"IPM_Region": "region"})

    keep_cols = ["plant_id_eia", "region"]
    return model_hydro_gdf.loc[:, keep_cols]


def load_plant_region_map(
    gens_860,
    pudl_engine,
    settings,
    model_regions_gdf,
    table="plant_region_map_epaipm",
    settings_agg_key="region_aggregations",
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
    settings_agg_key : str, optional
        The name of a dictionary of lists aggregatign regions in the settings
        object, by default "region_aggregations"

    Returns
    -------
    dataframe
        A dataframe where each plant has an associated "model_region" mapped
        from the original region labels.
    """
    # Load dataframe of region labels for each EIA plant id
    region_map_df = pd.read_sql_table(table, con=pudl_engine)

    # Label hydro using the IPM shapefile because NEEDS seems to drop some hydro
    all_hydro_regions = label_hydro_region(gens_860, pudl_engine, model_regions_gdf)

    region_map_df = pd.concat(
        [region_map_df, all_hydro_regions], ignore_index=True, sort=False
    ).drop_duplicates(subset=["plant_id_eia"], keep="first")

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

    # Create a new column "model_region" with labels that we're using for aggregated
    # regions

    model_region_map_df = region_map_df.loc[
        region_map_df.region.isin(keep_regions), :
    ].drop(columns="id")

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
    df,
    settings,
    age_col="operating_date",
    settings_retirement_table="retirement_ages",
    add_additional_retirements=True,
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

    start_len = len(df)
    retirement_ages = settings[settings_retirement_table]

    if df.loc[df["technology_description"].isnull(), :].empty is False:
        df = fill_missing_tech_descriptions(df)

    for tech, life in retirement_ages.items():
        try:
            df.loc[df.technology_description == tech, "retirement_year"] = (
                df.loc[df.technology_description == tech, age_col].dt.year + life
            )
        except AttributeError:
            # This is a bit hacky but for the proposed plants I have an int column
            df.loc[df.technology_description == tech, "retirement_year"] = (
                df.loc[df.technology_description == tech, age_col] + life
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
            df["retirement_year"] < model_year, settings["capacity_col"]
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
            df["retirement_year"] < model_year, settings["capacity_col"]
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
        return None
    if "report_date" not in by and "report_date" in df.columns:
        # by.append("report_date")
        logger.warning("'report_date' is in the df but not used in the groupby")
    region_agg_map = reverse_dict_of_lists(settings["region_aggregations"])
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

    sql = """
        SELECT * FROM generators_eia860
        WHERE operational_status_code NOT IN ('RE', 'OS', 'IP', 'CN')
    """
    gens_860 = pd.read_sql_query(
        sql=sql, con=pudl_engine, parse_dates=["planned_retirement_date", "report_date"]
    )
    gens_860 = gens_860.loc[gens_860["report_date"].dt.year.isin(data_years), :]

    return gens_860


def supplement_generator_860_data(
    gens_860, gens_entity, bga, model_region_map, settings
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
       'operating_date', 'boiler_id', 'unit_id_eia', 'unit_id_pudl',
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

    gen_cols = [
        # "report_date",
        "plant_id_eia",
        # "plant_name",
        "generator_id",
        # "balancing_authority_code",
        settings["capacity_col"],
        "energy_source_code_1",
        "energy_source_code_2",
        "minimum_load_mw",
        "operational_status_code",
        "planned_new_capacity_mw",
        "switch_oil_gas",
        "technology_description",
        "time_cold_shutdown_full_load_code",
        "planned_retirement_date",
    ]

    entity_cols = ["plant_id_eia", "generator_id", "prime_mover_code", "operating_date"]

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
    gens_860_model = (
        pd.merge(
            gens_860[gen_cols],
            model_region_map.drop(columns="region"),
            on="plant_id_eia",
            how="inner",
        )
        .merge(
            gens_entity[entity_cols], on=["plant_id_eia", "generator_id"], how="inner"
        )
        .merge(bga[bga_cols], on=["plant_id_eia", "generator_id"], how="left")
    )

    merged_capacity = gens_860_model.groupby("technology_description")[
        settings["capacity_col"]
    ].sum()
    if not np.allclose(initial_capacity.sum(), merged_capacity.sum()):
        logger.warning(
            f"Capacity changed from {initial_capacity} \nto \n{merged_capacity}"
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

    return not_canceled_df


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

    return not_retired_df


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

    # Load 923 generation and fuel data for one or more years.
    # Only load plants in the model regions.
    sql = """
        SELECT * FROM generation_fuel_eia923
    """
    gen_fuel_923 = pd.read_sql_query(sql, pudl_engine, parse_dates=["report_date"])
    gen_fuel_923 = gen_fuel_923.loc[
        (gen_fuel_923["report_date"].dt.year.isin(data_years))
        & (gen_fuel_923["plant_id_eia"].isin(model_region_map.plant_id_eia)),
        :,
    ]

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
    cc_without_pudl_id = gens_860.loc[
        (gens_860["unit_id_pudl"].isnull())
        & (gens_860["technology_description"] == "Natural Gas Fired Combined Cycle"),
        "plant_id_eia",
    ]
    df.loc[
        (df["plant_id_eia"].isin(cc_without_pudl_id))
        & (df["prime_mover_code"].isin(["CA", "CT"])),
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
        "fuel_type_code_pudl",
        "fuel_type_code_aer",
        "prime_mover_code",
    ]

    annual_gen_fuel_923 = (
        (
            df.drop(columns=["id", "nuclear_unit_id"])
            .groupby(by=by, as_index=False)[
                "fuel_consumed_units",
                "fuel_consumed_for_electricity_units",
                "fuel_consumed_mmbtu",
                "fuel_consumed_for_electricity_mmbtu",
                "net_generation_mwh",
            ]
            .sum()
        )
        .reset_index()
        .drop(columns="index")
        .sort_values(["plant_id_eia", "fuel_type", "prime_mover_code"])
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

    weighted_unit_hr = (
        heat_rate_df.groupby(["plant_id_eia", "unit_id_pudl"], as_index=False)
        .apply(w_hr)
        .reset_index()
    )

    weighted_unit_hr = weighted_unit_hr.rename(columns={0: "heat_rate_mmbtu_mwh"})

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

    by = ["plant_id_eia", "prime_mover_code", "fuel_type"]
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
        'unit_id_pudl', 'heat_rate_mmbtu_mwh']).
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

    by = ["plant_id_eia", "unit_id_pudl"]
    # add a unit code (plant plus generator code) in cases where one doesn't exist
    df_copy = df.reset_index()

    # All units should have the same heat rate so taking the mean will just keep the
    # same value.
    grouped_units = df_copy.groupby(by).agg(
        {
            settings["capacity_col"]: "sum",
            "minimum_load_mw": "sum",
            "heat_rate_mmbtu_mwh": "mean",
        }
    )
    grouped_units = grouped_units.replace([np.inf, -np.inf], np.nan)
    grouped_units = grouped_units.fillna(grouped_units.mean())

    return grouped_units


def calc_unit_cluster_values(df, settings, technology=None):
    """
    Calculate the total capacity, minimum load, weighted heat rate, and number of
    units/generators in a technology cluster.

    Parameters
    ----------
    df : dataframe
        A dataframe with units/generators of a single technology. One column should be
        'cluster', to label units as belonging to a specific cluster grouping.
    technology : str, optional
        Name of the generating technology, by default None

    Returns
    -------
    dataframe
        Aggragate values for generators in a technology cluster
    """

    # Define a function to compute the weighted mean.
    # The issue here is that the df name needs to be used in the function.
    # So this will need to be within a function that takes df as an input
    def wm(x):
        return np.average(x, weights=df.loc[x.index, settings["capacity_col"]])

    if df["heat_rate_mmbtu_mwh"].isnull().values.any():
        # mean =
        # df["heat_rate_mmbtu_mwh"] = df["heat_rate_mmbtu_mwh"].fillna(
        #     df["heat_rate_mmbtu_mwh"].median()
        # )
        start_cap = df[settings["capacity_col"]].sum()
        df = df.loc[~df["heat_rate_mmbtu_mwh"].isnull(), :]
        end_cap = df[settings["capacity_col"]].sum()
        cap_diff = start_cap - end_cap
        logger.warning(f"dropped {cap_diff}MW because of null heat rate values")

    df_values = df.groupby("cluster").agg(
        {
            settings["capacity_col"]: "mean",
            "minimum_load_mw": "mean",
            "heat_rate_mmbtu_mwh": wm,
        }
    )
    if df_values["heat_rate_mmbtu_mwh"].isnull().values.any():
        print(df)
        print(df_values)
    df_values["heat_rate_mmbtu_mwh_iqr"] = df.groupby("cluster").agg(
        {"heat_rate_mmbtu_mwh": iqr}
    )
    df_values["heat_rate_mmbtu_mwh_std"] = df.groupby("cluster").agg(
        {"heat_rate_mmbtu_mwh": "std"}
    )

    df_values["Min_power"] = (
        df_values["minimum_load_mw"] / df_values[settings["capacity_col"]]
    )

    df_values["num_units"] = df.groupby("cluster")["cluster"].count()

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
    ignored = r"\s+|_"
    technology = df["technology"].str.replace(ignored, "")
    # Create a new dataframe with the same index
    default = settings.get("default_model_tag", 0)
    for tag_col in settings.get("model_tag_names", []):
        df[tag_col] = default

        try:
            for tech, tag_value in settings["model_tag_values"][tag_col].items():
                tech = re.sub(ignored, "", tech)
                mask = technology.str.contains(fr"^{tech}", case=False)
                df.loc[mask, tag_col] = tag_value
        except (KeyError, AttributeError) as e:
            logger.warning(f"No model tag values found for {tag_col} ({e})")

    # Change tags with specific regional values for a technology
    flat_regional_tags = flatten(settings.get("regional_tag_values", {}))

    for tag_tuple, tag_value in flat_regional_tags.items():
        region, tag_col, tech = tag_tuple
        tech = re.sub(ignored, "", tech)
        mask = technology.str.contains(fr"^{tech}", case=False)
        df.loc[(df["region"] == region) & mask, tag_col,] = tag_value

    return df


def load_ipm_shapefile(settings, path=IPM_GEOJSON_PATH):
    """
    Load the shapefile of IPM regions

    Parameters
    ----------
    settings : dict
        User-defined parameters from a settings YAML file. This is where any region
        aggregations would be defined.

    Returns
    -------
    geodataframe
        Regions to use in the study with the matching geometry for each.
    """

    region_agg_map = reverse_dict_of_lists(settings["region_aggregations"])

    # IPM regions to keep. Regions not in this list will be dropped
    keep_regions = [
        x
        for x in settings["model_regions"] + list(region_agg_map)
        if x not in region_agg_map.values()
    ]

    ipm_regions = gpd.read_file(IPM_GEOJSON_PATH)
    # ipm_regions = gpd.read_file(IPM_SHAPEFILE_PATH)

    model_regions_gdf = ipm_regions.loc[ipm_regions["IPM_Region"].isin(keep_regions)]
    model_regions_gdf = map_agg_region_names(
        model_regions_gdf, region_agg_map, "IPM_Region", "model_region"
    )

    return model_regions_gdf


def download_860m(settings):
    """Load the entire 860m file into memory as an ExcelFile object.

    Parameters
    ----------
    settings : dict
        User-defined settings loaded from a YAML file. This is where the EIA860m
        filename is defined.

    Returns
    -------
    [type]
        [description]
    """
    try:
        fn = settings["eia_860m_fn"]
    except KeyError:
        # No key in the settings file
        logger.info("Trying to determine the most recent EIA860m file...")
        fn = find_newest_860m()

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
            eia_860m = pd.ExcelFile(local_file)
        except XLRDError:
            logger.warning("A more recent version of EIA-860m is available")
            download_save(archive_url, local_file)
            eia_860m = pd.ExcelFile(local_file)
        # write the file to disk

    return eia_860m


def find_newest_860m():
    """Scrape the EIA 860m page to find the most recently posted file.

    Returns
    -------
    str
        Name of most recently posted file
    """
    site_url = "https://www.eia.gov/electricity/data/eia860m/"
    r = requests.get(site_url)
    soup = BeautifulSoup(r.content, "lxml")
    table = soup.find("table", attrs={"class": "simpletable"})
    href = table.find("a")["href"]
    fn = href.split("/")[-1]

    return fn


def clean_860m_sheet(eia_860m, sheet_name, settings):
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
    dataframe
        One of the sheets from 860m
    """

    df = eia_860m.parse(
        sheet_name=sheet_name, skiprows=1, skipfooter=1, na_values=[" "]
    )
    df = df.rename(columns=planned_col_map)

    if sheet_name in ["Operating", "Planned"]:
        df.loc[:, "operational_status_code"] = df.loc[:, "operational_status"].map(
            op_status_map
        )

    if sheet_name == "Planned":
        df = df.loc[
            df["operational_status_code"].isin(settings["proposed_status_included"]), :
        ]

    return df


def import_proposed_generators(planned, settings, model_regions_gdf):
    """
    Load the most recent proposed generating units from EIA860m. Will also add
    any planned generators that are included in the settings file.

    Parameters
    ----------
    settings : dict
        User defined parameters from a settings YAML file
    model_regions_gdf : geodataframe
        Contains the name and geometry of each region being used in the study

    Returns
    -------
    dataframe
        All proposed generators.
    """

    # Some plants don't have lat/lon data. Log this now to determine if any action is
    # needed, then drop them from the dataframe.
    no_lat_lon = planned.loc[
        (planned["latitude"].isnull()) | (planned["longitude"].isnull()), :
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

    planned = planned.dropna(subset=["latitude", "longitude"])

    # Convert the lon/lat values to geo points. Need to add an initial CRS and then
    # change it to align with the IPM regions
    print("Creating gdf")
    planned_gdf = gpd.GeoDataFrame(
        planned.copy(),
        geometry=gpd.points_from_xy(planned.longitude.copy(), planned.latitude.copy()),
        crs={"init": "epsg:4326"},
    )
    # planned_gdf.crs = {"init": "epsg:4326"}
    if planned_gdf.crs != model_regions_gdf.crs:
        planned_gdf = planned_gdf.to_crs(model_regions_gdf.crs)

    planned_gdf = gpd.sjoin(model_regions_gdf.drop(columns="IPM_Region"), planned_gdf)

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

    # Assume anything else being built at scale is wind/solar and will have a Min_power
    # of 0
    planned_gdf.loc[planned_gdf["minimum_load_mw"].isnull(), "minimum_load_mw"] = 0

    planned_gdf = planned_gdf.set_index(
        ["plant_id_eia", "prime_mover_code", "energy_source_code_1"]
    )

    # Add a retirement year based on the planned start year
    label_retirement_year(
        df=planned_gdf,
        settings=settings,
        age_col="planned_operating_year",
        add_additional_retirements=False,
    )

    if settings.get("group_technologies"):
        planned_gdf = group_technologies(planned_gdf, settings)
        print(planned_gdf["technology_description"].unique().tolist())

    keep_cols = [
        "model_region",
        "technology_description",
        "generator_id",
        settings["capacity_col"],
        "minimum_load_mw",
        "operational_status_code",
        "heat_rate_mmbtu_mwh",
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

    cap_col = settings["capacity_col"]

    # Include standby (SB) generators since they are in our capacity totals
    sql = """
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
        GROUP BY
            G.report_date,
            G.plant_id_eia,
            G.technology_description,
            G.fuel_type_code_pudl,
            G.generator_id
        ORDER by G.plant_id_eia, G.report_date
    """

    plant_gen_tech_cap = pd.read_sql_query(
        sql, pudl_engine, parse_dates=["report_date"]
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

    capacity_factor = pudl.helpers.merge_on_date_year(
        plant_tech_cap, generation, on=["plant_id_eia"], how="left"
    )

    if settings.get("group_technologies"):
        capacity_factor = group_technologies(capacity_factor, settings)

    if years_filter is None:
        years_filter = {
            tech: settings["capacity_factor_default_year_filter"]
            for tech in plant_gen_tech_cap["technology_description"].unique()
        }
        if type(settings["alt_year_filters"]) is dict:
            for tech, value in settings["alt_year_filters"].items():
                years_filter[tech] = value

        data_years = plant_gen_tech_cap["report_date"].dt.year.unique()

        # Use all years where the value is None

        for tech, value in years_filter.items():
            if value is None:
                years_filter[tech] = data_years

    df_list = []
    for tech, years in years_filter.items():
        _df = capacity_factor.loc[
            (capacity_factor["technology_description"] == tech)
            & (capacity_factor["report_date"].dt.year.isin(years)),
            :,
        ]
        df_list.append(_df)
    capacity_factor = pd.concat(df_list, sort=False)

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

    return capacity_factor_tech_region


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
    """

    df["Fuel"] = "None"
    for eia_tech, fuel in (settings.get("tech_fuel_map") or {}).items():
        try:
            if eia_tech == "Natural Gas Steam Turbine":
                # No ATB natural gas steam turbine and I match it with coal for O&M
                # which would screw this up and list natural gas as a fuel for ATB
                # coal plants
                atb_tech = None
            else:
                atb_tech = settings["eia_atb_tech_map"][eia_tech].split("_")[0]
        except KeyError:
            # No corresponding ATB technology
            atb_tech = None
        scenario = settings["aeo_fuel_scenarios"][fuel]
        model_year = settings["model_year"]

        for aeo_region, model_regions in settings["aeo_fuel_region_map"].items():
            fuel_name = ("_").join([aeo_region, scenario, fuel])
            assert (
                fuel_prices.query(
                    "year==@model_year & full_fuel_name==@fuel_name"
                ).empty
                is False
            ), f"{fuel_name} doesn't show up in {model_year}"

            df.loc[
                (df["technology"] == eia_tech) & df["region"].isin(model_regions),
                "Fuel",
            ] = fuel_name

            if atb_tech is not None:
                df.loc[
                    (df["technology"].str.contains(atb_tech))
                    & df["region"].isin(model_regions),
                    "Fuel",
                ] = fuel_name

    for ccs_tech, ccs_fuel in (settings.get("ccs_fuel_map") or {}).items():
        scenario = settings["aeo_fuel_scenarios"][ccs_fuel.split("_")[0]]
        for aeo_region, model_regions in settings["aeo_fuel_region_map"].items():
            ccs_fuel_name = ("_").join([aeo_region, scenario, ccs_fuel])

            df.loc[
                (df["technology"].str.contains(ccs_tech))
                & df["region"].isin(model_regions),
                "Fuel",
            ] = ccs_fuel_name

    return df


def calculate_transmission_inv_cost(resource_df, settings):
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
    # Apply calculation for
    regions = resource_df["region"].unique()
    for ttype, params in ttypes.items():
        if ttype not in resource_ttypes:
            continue
        # Check presence of required keys
        missing_keys = list(set(KEYS) - set(params))
        if missing_keys:
            raise KeyError(f"{SETTING}.{ttype} missing required keys {missing_keys}")
        if isinstance(params["capex_mw_mile"], dict):
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
        resource_df[f"{ttype}_capex"] = capex_mw_mile * resource_df[f"{ttype}_miles"]
        resource_df[f"{ttype}_inv_mwyr"] = investment_cost_calculator(
            resource_df[f"{ttype}_capex"], params["wacc"], params["investment_years"],
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
        `Inv_cost_per_MWyr` and transmission costs.
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
        A modified copy of the input dataframe where 'Inv_cost_per_MWyr' represents the
        combined plant and transmission investment costs. The new column
        `plant_inv_cost_mwyr` represents just the plant investment costs.
    """
    use_total = (
        settings.get("transmission_investment_cost", {}).get("use_total", False)
        and "interconnect_annuity" in resource_df
    )
    resource_df["plant_inv_cost_mwyr"] = resource_df["Inv_cost_per_MWyr"]
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
    resource_df["Inv_cost_per_MWyr"] += cost
    return resource_df


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
        settings,
        current_gens=True,
        sort_gens=False,
        plant_region_map_table="plant_region_map_epaipm",
        settings_agg_key="region_aggregations",
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
        self.settings = settings
        self.current_gens = current_gens
        self.sort_gens = sort_gens
        self.model_regions_gdf = load_ipm_shapefile(self.settings)

        if self.current_gens:
            self.data_years = self.settings["data_years"]

            self.gens_860 = load_generator_860_data(self.pudl_engine, self.data_years)
            self.gens_entity = pd.read_sql_table(
                "generators_entity_eia", self.pudl_engine
            )

            bga = self.pudl_out.bga()
            self.bga = bga.loc[
                bga.report_date.dt.year.isin(self.data_years), :
            ].drop_duplicates(["plant_id_eia", "generator_id"])

            logger.info("Loading map of plants to IPM regions")
            self.plant_region_map = load_plant_region_map(
                self.gens_860,
                self.pudl_engine,
                self.settings,
                self.model_regions_gdf,
                table=plant_region_map_table,
                settings_agg_key=settings_agg_key,
            )

            self.gen_923 = load_923_gen_fuel_data(
                self.pudl_engine,
                self.pudl_out,
                model_region_map=self.plant_region_map,
                data_years=self.data_years,
            )

            self.eia_860m = download_860m(self.settings)
            self.planned_860m = clean_860m_sheet(
                self.eia_860m, sheet_name="Planned", settings=self.settings
            )
            self.canceled_860m = clean_860m_sheet(
                self.eia_860m,
                sheet_name="Canceled or Postponed",
                settings=self.settings,
            )
            self.retired_860m = clean_860m_sheet(
                self.eia_860m, sheet_name="Retired", settings=self.settings
            )

            self.ownership = load_ownership_eia860(self.pudl_engine, self.data_years)
            self.plants_860 = load_plants_860(self.pudl_engine, self.data_years)
            self.utilities_eia = load_utilities_eia(self.pudl_engine)
        else:
            self.existing_resources = pd.DataFrame()

        self.atb_costs = fetch_atb_costs(self.pudl_engine, self.settings)
        self.atb_hr = fetch_atb_heat_rates(self.pudl_engine)

        self.fuel_prices = fetch_fuel_prices(self.settings)

    def fill_na_heat_rates(self, df):
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

        median_hr = df["heat_rate_mmbtu_mwh"].median()
        df["heat_rate_mmbtu_mwh"].fillna(median_hr, inplace=True)

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

        if self.settings.get("demand_response_resources"):
            for resource, parameters in self.settings["demand_response_resources"][
                year
            ].items():

                _df = pd.DataFrame(
                    index=self.settings["model_regions"],
                    columns=self.settings["generator_columns"],
                )
                _df = _df.drop(columns="Resource")
                _df["technology"] = resource
                _df["region"] = self.settings["model_regions"]

                dr_path = (
                    Path.cwd()
                    / self.settings["input_folder"]
                    / self.settings["demand_response_fn"]
                )
                dr_profile = make_demand_response_profiles(
                    dr_path, resource, self.settings
                )
                self.demand_response_profiles[resource] = dr_profile

                dr_capacity = demand_response_resource_capacity(
                    dr_profile, resource, self.settings
                )
                dr_capacity_scenario = dr_capacity.squeeze()
                _df["Existing_Cap_MW"] = _df["region"].map(dr_capacity_scenario)

                if isinstance(parameters["parameter_values"], dict):
                    for col, value in parameters["parameter_values"].items():
                        _df[col] = value

                df_list.append(_df)

            dr_rows = pd.concat(df_list)
            dr_rows["New_Build"] = -1
            dr_rows["Fuel"] = "None"
            dr_rows["cluster"] = 1
            dr_rows = dr_rows.fillna(0)
        else:
            dr_rows = pd.DataFrame()

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
            .pipe(label_retirement_year, self.settings, add_additional_retirements=True)
            .pipe(label_small_hydro, self.settings, by=["plant_id_eia"])
            .pipe(group_technologies, self.settings)
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
        self.weighted_unit_hr = unit_generator_heat_rates(
            self.pudl_out, self.data_years
        )

        # Merge the PUDL calculated heat rate data and set the index for easy
        # mapping using plant/prime mover heat rates from 923
        hr_cols = ["plant_id_eia", "unit_id_pudl", "heat_rate_mmbtu_mwh"]
        idx = ["plant_id_eia", "prime_mover_code", "energy_source_code_1"]
        self.units_model = self.gens_860_model.merge(
            self.weighted_unit_hr[hr_cols],
            on=["plant_id_eia", "unit_id_pudl"],
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

        # Fill any null heat rate values for each tech
        for tech in self.units_model["technology_description"]:
            self.units_model.loc[
                self.units_model.technology_description == tech, :
            ] = self.fill_na_heat_rates(
                self.units_model.loc[self.units_model.technology_description == tech, :]
            )
        # assert (
        #     self.units_model["heat_rate_mmbtu_mwh"].isnull().any() is False
        # ), "There are still some null heat rate values"

        logger.info(
            f"Units model technologies are "
            f"{self.units_model.technology_description.unique().tolist()}"
        )
        logger.info(
            f"Before adding proposed generators, {len(self.units_model)} units with "
            f"{self.units_model[self.settings['capacity_col']].sum()} MW capacity"
        )
        proposed_gens = import_proposed_generators(
            planned=self.planned_860m,
            settings=self.settings,
            model_regions_gdf=self.model_regions_gdf,
        )
        logger.info(
            f"Proposed gen technologies are "
            f"{proposed_gens.technology_description.unique().tolist()}"
        )
        logger.info(f"{proposed_gens[self.settings['capacity_col']].sum()} MW proposed")
        self.units_model = pd.concat([proposed_gens, self.units_model], sort=False)

        # Create a pudl unit id based on plant and generator id where one doesn't exist.
        # This is used later to match the cluster numbers to plants
        self.units_model.reset_index(inplace=True)
        self.units_model.loc[self.units_model.unit_id_pudl.isnull(), "unit_id_pudl"] = (
            self.units_model.loc[
                self.units_model.unit_id_pudl.isnull(), "plant_id_eia"
            ].astype(str)
            + "_"
            + self.units_model.loc[
                self.units_model.unit_id_pudl.isnull(), "generator_id"
            ].astype(str)
        ).values
        self.units_model.set_index(idx, inplace=True)

        logger.info(
            f"After adding proposed, units model technologies are "
            f"{self.units_model.technology_description.unique().tolist()}"
        )
        logger.info(
            f"After adding proposed generators, {len(self.units_model)} units with "
            f"{self.units_model[self.settings['capacity_col']].sum()} MW capacity"
        )

        techs = list(self.settings["num_clusters"])

        num_clusters = {}
        for region in self.settings["model_regions"]:
            num_clusters[region] = self.settings["num_clusters"].copy()

        for region in self.settings["alt_num_clusters"]:
            for tech, cluster_size in self.settings["alt_num_clusters"][region].items():
                num_clusters[region][tech] = cluster_size

        region_tech_grouped = self.units_model.loc[
            (self.units_model.technology_description.isin(techs))
            & (self.units_model.retirement_year >= self.settings["model_year"]),
            :,
        ].groupby(["model_region", "technology_description"])

        self.retired = self.units_model.loc[
            self.units_model.retirement_year < self.settings["model_year"], :
        ]

        # gens_860 lost the ownership code... refactor this!
        self.all_gens_860 = load_generator_860_data(self.pudl_engine, self.data_years)
        # Getting weighted ownership for each unit, which will be used below.
        self.weighted_ownership = weighted_ownership_by_unit(
            self.units_model, self.all_gens_860, self.ownership, self.settings
        )

        # For each group, cluster and calculate the average size/min load/heat rate
        # logger.info("Creating technology clusters by region")
        logger.info("Creating technology clusters by region")
        unit_list = []
        self.cluster_list = []
        alt_cluster_method = self.settings.get("alt_cluster_method") or {}

        for _, df in region_tech_grouped:
            region, tech = _
            grouped = group_units(df, self.settings)

            # This is bad. Should be setting up a dictionary of objects that picks the
            # correct clustering method. Can't keep doing if statements as the number of
            # methods grows. CHANGE LATER.
            if not alt_cluster_method:
                if num_clusters[region][tech] > 0:
                    clusters = cluster.KMeans(
                        n_clusters=num_clusters[region][tech], random_state=6
                    ).fit(preprocessing.StandardScaler().fit_transform(grouped))

                    grouped["cluster"] = (
                        clusters.labels_ + 1
                    )  # Change to 1-index for julia

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

            # Saving individual unit data for later analysis (if needed)
            unit_list.append(grouped)

            # Don't add technologies with specified 0 clusters
            if num_clusters[region][tech] != 0:
                _df = calc_unit_cluster_values(grouped, self.settings, tech)
                _df["region"] = region
                self.cluster_list.append(_df)

        # Save some data about individual units for easy access
        self.all_units = pd.concat(unit_list, sort=False)
        self.all_units = pd.merge(
            self.units_model.reset_index(),
            self.all_units,
            on=["plant_id_eia", "unit_id_pudl"],
            how="left",
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

        self.results = self.results.reset_index().set_index(
            ["region", "technology", "cluster"]
        )
        self.results.rename(
            columns={
                self.settings["capacity_col"]: "Cap_size",
                "heat_rate_mmbtu_mwh": "Heat_rate_MMBTU_per_MWh",
            },
            inplace=True,
        )

        # Calculate average capacity factors
        if type(self.settings["capacity_factor_techs"]) is list:
            self.capacity_factors = gentype_region_capacity_factor(
                self.pudl_engine, self.plant_region_map, self.settings
            )

            self.results = pd.merge(
                self.results.reset_index(),
                self.capacity_factors[["region", "technology", "capacity_factor"]],
                on=["region", "technology"],
                how="left",
            )

            if self.settings.get("derate_capacity"):
                derate_techs = self.settings["derate_techs"]
                self.results.loc[:, "unmodified_cap_size"] = self.results.loc[
                    :, "Cap_size"
                ].copy()
                self.results.loc[
                    self.results["technology"].isin(derate_techs), "Cap_size"
                ] = (
                    self.results.loc[
                        self.results["technology"].isin(derate_techs),
                        "unmodified_cap_size",
                    ]
                    * self.results.loc[
                        self.results["technology"].isin(derate_techs), "capacity_factor"
                    ]
                )

        # Round Cap_size to prevent GenX error.
        self.results = self.results.round(3)
        self.results["Cap_size"] = self.results["Cap_size"].round(2)
        self.results["Existing_Cap_MW"] = self.results.Cap_size * self.results.num_units
        self.results["unmodified_existing_cap_mw"] = (
            self.results["unmodified_cap_size"] * self.results["num_units"]
        )

        # Add fixed/variable O&M based on NREL atb
        self.results = (
            self.results.pipe(
                atb_fixed_var_om_existing, self.atb_costs, self.atb_hr, self.settings
            )
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
        self.results["Resource"] = snake_case_col(self.results["technology"])

        return self.results

    def create_new_generators(self):

        self.new_generators = atb_new_generators(
            self.atb_costs, self.atb_hr, self.settings
        )

        self.new_generators = (
            self.new_generators.pipe(startup_fuel, self.settings)
            .pipe(add_fuel_labels, self.fuel_prices, self.settings)
            .pipe(startup_nonfuel_costs, self.settings)
            .pipe(add_genx_model_tags, self.settings)
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
            calculate_transmission_inv_cost, self.settings
        ).pipe(add_transmission_inv_cost, self.settings)

        if self.settings.get("demand_response_fn"):
            dr_rows = self.create_demand_response_gen_rows()

        self.new_generators = pd.concat([self.new_generators, dr_rows], sort=False)

        self.new_generators["Resource"] = snake_case_col(
            self.new_generators["technology"]
        )

        return self.new_generators

    def create_all_generators(self):

        if self.current_gens:
            self.existing_resources = self.create_region_technology_clusters()

        self.new_resources = self.create_new_generators()

        self.all_resources = pd.concat(
            [self.existing_resources, self.new_resources], ignore_index=True, sort=False
        )

        self.all_resources = self.all_resources.round(3)
        self.all_resources["Cap_size"] = self.all_resources["Cap_size"].round(2)
        self.all_resources["Heat_rate_MMBTU_per_MWh"] = self.all_resources[
            "Heat_rate_MMBTU_per_MWh"
        ].round(2)

        # Set Min_power of wind/solar to 0
        self.all_resources.loc[self.all_resources["DISP"] == 1, "Min_power"] = 0

        self.all_resources["R_ID"] = np.arange(len(self.all_resources)) + 1

        if self.current_gens:
            logger.info(
                f"Capacity of {self.all_resources['Existing_Cap_MW'].sum()} MW in final clusters"
            )

        return self.all_resources
