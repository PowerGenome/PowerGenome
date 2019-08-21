import logging

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn import cluster, preprocessing
from xlrd import XLRDError
import pudl

from src.params import IPM_SHAPEFILE_PATH
from src.util import map_agg_region_names, reverse_dict_of_lists, snake_case_col

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
    start_len = len(df)

    df["_technology"] = df["technology_description"]
    for tech, group in settings["tech_groups"].items():
        df.loc[df["technology_description"].isin(group), "_technology"] = tech

    df.loc[:, "technology_description"] = df.loc[:, "_technology"]
    df = df.drop(columns=["_technology"])

    end_len = len(df)
    assert (
        start_len == end_len,
        "One or more records was dropped when grouping technology descriptions",
    )

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
    table="plant_region_map_ipm",
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
        The SQL table to load, by default "plant_region_map_ipm"
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
    all_hydro_regions = label_hydro_region(
        gens_860, pudl_engine, settings, model_regions_gdf
    )
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
    if settings["additional_retirements"] and add_additional_retirements:
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
            # print(plant_id, gen_id, ret_year)
            df.loc[
                (df["plant_id_eia"] == plant_id) & (df["generator_id"] == gen_id),
                "retirement_year",
            ] = ret_year
            # print(df.loc[
            #     (df["plant_id_eia"] == plant_id) & (df["generator_id"] == gen_id),
            #     ['plant_id_eia', 'generator_id', 'summer_capacity_mw', "retirement_year"]
            # ])
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


def label_small_hydro(df, settings, by=["plant_id_eia"]):
    region_agg_map = reverse_dict_of_lists(settings["region_aggregations"])
    keep_regions = [
        x
        for x in settings["model_regions"] + list(region_agg_map)
        if x in settings["small_hydro_regions"]
    ]
    start_len = len(df)
    size_cap = settings["small_hydro_mw"]
    if settings["capacity_col"] in df:
        cap_col = settings["capacity_col"]
    else:
        cap_col = "capacity_mw"

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

    assert start_len == end_len


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
        WHERE DATE_PART('year', report_date) IN %(data_years)s
        AND operational_status_code NOT IN ('RE', 'OS', 'IP', 'CN')
    """
    gens_860 = pd.read_sql_query(
        sql=sql,
        con=pudl_engine,
        params={"data_years": tuple(data_years)},
        parse_dates=["planned_retirement_date"],
    )

    return gens_860


def supplement_generator_860_data(
    gens_860, pudl_engine, settings, pudl_out, model_region_map, data_years=[2017]
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
    initial_record_count = len(initial_capacity)
    gen_entity = pd.read_sql_table("generators_entity_eia", pudl_engine)

    # Add pudl unit ids, only include specified data years
    bga = pudl_out.bga()
    bga = bga.loc[bga.report_date.dt.year.isin(data_years), :].drop_duplicates(
        ["plant_id_eia", "generator_id"]
    )

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
            gen_entity[entity_cols], on=["plant_id_eia", "generator_id"], how="inner"
        )
        .merge(bga[bga_cols], on=["plant_id_eia", "generator_id"], how="left")
    )

    # merged_record_count = len(gens_860_model)
    # if initial_record_count != merged_record_count:
    #     dropped_records = pd.concat(
    #         [gens_860[['plant_id_eia', 'generator_id']],
    #         gens_860_model[['plant_id_eia', 'generator_id']]], sort=False
    #     )
    #     dropped_records = dropped_records.loc[
    #         :, ['plant_id_eia', 'generator_id']
    #     ].drop_duplicates(keep=False)

    #     logger.error(f'{initial_record_count - merged_record_count} records dropped', dropped_records)
    merged_capacity = gens_860_model.groupby("technology_description")[
        settings["capacity_col"]
    ].sum()
    if not np.allclose(initial_capacity.sum(), merged_capacity.sum()):
        logger.warning(
            f"Capacity changed from {initial_capacity} \nto \n{merged_capacity}"
        )

    # Label retirement years for each generator
    label_retirement_year(gens_860_model, settings, add_additional_retirements=True)

    # Label small hydro and group technologies
    if settings["small_hydro"] is True:
        label_small_hydro(gens_860_model, settings, by=["plant_id_eia"])
    if settings["group_technologies"] is True:
        logger.info("Grouping some technology categories together")
        gens_860_model = group_technologies(gens_860_model, settings)
        print(gens_860_model["technology_description"].unique())

    final_capacity = gens_860_model.groupby("technology_description")[
        settings["capacity_col"]
    ].sum()
    if not np.allclose(merged_capacity.sum(), final_capacity.sum()):
        logger.warning(
            f"Capacity changed from {merged_capacity} \nto \n{final_capacity}"
        )

    logger.info(f"Capacity of {final_capacity.sum()} MW loaded from 860")

    final_not_ret_capacity = (
        gens_860_model.loc[
            (gens_860_model["retirement_year"] >= settings["model_year"])
            & (
                gens_860_model["technology_description"].isin(
                    settings["num_clusters"].keys()
                )
            ),
            :,
        ]
        .groupby("technology_description")[settings["capacity_col"]]
        .sum()
    )
    # included_tech_capacity = final_capacity[settings["num_clusters"].keys()]
    logger.info(
        f"Capacity of {final_not_ret_capacity.sum()} MW is not retired and should be in final clusters"
    )
    # print(final_not_ret_capacity)

    return gens_860_model


def load_generator_923_data(pudl_engine, pudl_out, model_region_map, data_years=[2017]):
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
        WHERE DATE_PART('year', report_date) IN %(data_years)s
        AND plant_id_eia IN %(plant_ids)s

    """

    gen_fuel_923 = pd.read_sql_query(
        sql,
        pudl_engine,
        params={
            "data_years": tuple(data_years),
            "plant_ids": tuple(model_region_map.plant_id_eia),
        },
    )

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
            gen_fuel_923.drop(columns=["id", "nuclear_unit_id"])
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

    # Calculate the heat rate for each prime mover/fuel combination
    annual_gen_fuel_923["heat_rate_mmbtu_mwh"] = (
        annual_gen_fuel_923["fuel_consumed_for_electricity_mmbtu"]
        / annual_gen_fuel_923["net_generation_mwh"]
    )

    return annual_gen_fuel_923


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


def unit_generator_heat_rates(pudl_out, annual_gen_fuel_923, data_years):
    """
    Calculate the heat rate for each PUDL unit and generators that don't have a PUDL
    unit id.

    Parameters
    ----------
    pudl_out : pudl.PudlTabl
        A PudlTabl object for loading pre-calculated PUDL analysis data
    annual_gen_fuel_923 : dataframe
        Data from EIA-923 with generation and fuel use for each plant/prime mover/fuel
        combo, and the heat rate calculated using this data.
    data_years : list
        Years of data to use

    Returns
    -------
    dataframe, dict
        A dataframe of heat rates for each pudl unit (columsn are ['plant_id_eia',
        'unit_id_pudl', 'heat_rate_mmbtu_mwh']), and a dictionary mapping keys of
        (plant_id_eia, prime_mover_code, fuel_type) to a heat rate value.
    """

    # Create groupings of plant/prime mover/fuel type to use the calculated
    # heat rate in cases where PUDL doesn't have a unit heat rate
    by = ["plant_id_eia", "prime_mover_code", "fuel_type"]
    annual_gen_fuel_923_groups = annual_gen_fuel_923.groupby(by)

    prime_mover_hr_map = {
        _: df["heat_rate_mmbtu_mwh"].values[0] for _, df in annual_gen_fuel_923_groups
    }

    # Load the pre-calculated PUDL unit heat rates for selected years.
    # Remove rows without generation or with null values.
    unit_hr = pudl_out.hr_by_unit()
    unit_hr = unit_hr.loc[
        (unit_hr.report_date.dt.year.isin(data_years))
        & (unit_hr.net_generation_mwh > 0),
        :,
    ].dropna()

    weighted_unit_hr = calculate_weighted_heat_rate(unit_hr)

    return weighted_unit_hr, prime_mover_hr_map


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

    # df_copy.loc[df_copy.unit_id_pudl.isnull(), "unit_id_pudl"] = (
    #     df_copy.loc[df_copy.unit_id_pudl.isnull(), "plant_id_eia"].astype(str)
    #     + "_"
    #     + df_copy.loc[df_copy.unit_id_pudl.isnull(), "generator_id"].astype(str)
    # ).values

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

    df_values = df.groupby("cluster").agg(
        {
            settings["capacity_col"]: "mean",
            "minimum_load_mw": "mean",
            "heat_rate_mmbtu_mwh": wm,
        }
    )

    df_values["Min_power"] = (
        df_values["minimum_load_mw"] / df_values[settings["capacity_col"]]
    )

    df_values["num_units"] = df.groupby("cluster")["cluster"].count()

    if technology:
        df_values["technology"] = technology

    return df_values


def add_model_tags(df, settings):

    model_tag_cols = settings["model_tag_names"]

    # Create a new dataframe with the same index
    tag_df = pd.DataFrame(
        index=df.index, columns=model_tag_cols, data=settings["default_model_tag"]
    )
    model_tag_dict = settings["model_tag_values"]
    for col, value_map in model_tag_dict.items():
        tag_df[col] = tag_df.index.get_level_values("technology").map(value_map)

    tag_df.fillna(settings["default_model_tag"], inplace=True)

    combined_df = pd.concat([df, tag_df], axis=1)

    return combined_df


def load_ipm_shapefile(settings):

    region_agg_map = reverse_dict_of_lists(settings["region_aggregations"])

    # IPM regions to keep. Regions not in this list will be dropped
    keep_regions = [
        x
        for x in settings["model_regions"] + list(region_agg_map)
        if x not in region_agg_map.values()
    ]

    ipm_regions = gpd.read_file(IPM_SHAPEFILE_PATH)

    model_regions_gdf = ipm_regions.loc[ipm_regions["IPM_Region"].isin(keep_regions)]
    model_regions_gdf = map_agg_region_names(
        model_regions_gdf, region_agg_map, "IPM_Region", "model_region"
    )

    return model_regions_gdf


def import_proposed_generators(settings, model_regions_gdf):

    # model_regions_gdf = load_ipm_shapefile(settings)

    fn = settings["proposed_gen_860_fn"]
    # Only the most recent file will not have archive in the url
    url = f"https://www.eia.gov/electricity/data/eia860m/xls/{fn}"
    archive_url = f"https://www.eia.gov/electricity/data/eia860m/archive/xls/{fn}"

    try:
        planned = pd.read_excel(
            url, sheet_name="Planned", skiprows=1, skipfooter=1, na_values=[" "]
        )
    except XLRDError:
        logger.warning("A more recent version of EIA-860m is available")
        planned = pd.read_excel(
            archive_url, sheet_name="Planned", skiprows=1, skipfooter=1, na_values=[" "]
        )

    planned = planned.rename(columns=planned_col_map)
    planned.loc[:, "operational_status_code"] = planned.loc[
        :, "operational_status"
    ].map(op_status_map)
    planned = planned.loc[
        planned["operational_status_code"].isin(settings["proposed_status_included"]), :
    ]

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
    #     print(planned.columns)

    # Convert the lon/lat values to geo points. Need to add an initial CRS and then
    # change it to align with the IPM regions
    print("Creating gdf")
    planned_gdf = gpd.GeoDataFrame(
        planned.copy(),
        geometry=gpd.points_from_xy(planned.longitude.copy(), planned.latitude.copy()),
        crs={"init": "epsg:4326"},
    )
    planned_gdf.crs = {"init": "epsg:4326"}
    planned_gdf = planned_gdf.to_crs(model_regions_gdf.crs)

    planned_gdf = gpd.sjoin(model_regions_gdf.drop(columns="IPM_Region"), planned_gdf)

    # Add planned additions from the settings file
    if settings["additional_planned"]:
        i = 0
        for record in settings["additional_planned"]:
            plant_id, gen_id, model_region = record
            plant_record = planned.loc[
                (planned["plant_id_eia"] == plant_id)
                & (planned["generator_id"] == gen_id),
                :,
            ]
            plant_record["model_region"] = model_region

            planned_gdf = planned_gdf.append(plant_record, sort=False)
            i += 1
        logger.info(f"{i} generators were added to the planned list based on settings")

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

    if settings["group_technologies"] is True:
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

    # cf_techs = []
    # for tech in settings["capacity_factor_techs"]:
    #     if tech in settings["tech_groups"]:
    #         cf_techs += settings["tech_groups"][tech]
    #     else:
    #         cf_techs.append(tech)

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
            AND G.plant_id_eia IN %(plant_ids)s
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
        parse_dates=["report_date"],
        params={"plant_ids": tuple(plant_region_map["plant_id_eia"].tolist())},
    )

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
    # plant_tech_cap = plant_tech_cap.loc[
    #     plant_tech_cap["technology_description"].isin(cf_techs), :
    # ]

    sql = """
        SELECT
            DATE_PART('year', GF.report_date) AS report_date,
            GF.plant_id_eia,
            SUM(GF.net_generation_mwh) AS net_generation_mwh,
            GF.fuel_type_code_pudl
        FROM
            generation_fuel_eia923 GF
        GROUP BY DATE_PART('year', GF.report_date), GF.plant_id_eia, GF.fuel_type_code_pudl
        ORDER by GF.plant_id_eia, DATE_PART('year', GF.report_date)
    """
    generation = pd.read_sql_query(sql, pudl_engine, parse_dates={"report_date": "%Y"})

    capacity_factor = pudl.helpers.merge_on_date_year(
        plant_tech_cap, generation, on=["plant_id_eia"], how="left"
    )

    if settings["group_technologies"] is True:
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
    # capacity_factor_tech_region["technology"] = snake_case_col(
    #     capacity_factor_tech_region["technology"]
    # )
    # capacity_factor_tech_region.set_index(
    #     ["model_region", "technology_description"], inplace=True
    # )
    logger.debug(capacity_factor_tech_region)

    return capacity_factor_tech_region


class GeneratorClusters:
    """
    This class is used to determine genererating units that will likely be operating
    in a given year, clusters them according to parameters for the settings file,
    and determines the average operating characteristics of each cluster. Structuring
    this as a class isn't strictly necessary but makes it easier to access generator
    data part-way through the process.
    """

    def __init__(self, pudl_engine, pudl_out, settings):
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
        self.model_regions_gdf = load_ipm_shapefile(self.settings)
        self.data_years = self.settings["data_years"]
        self.gens_860 = load_generator_860_data(self.pudl_engine, self.data_years)

    def create_region_technology_clusters(
        self,
        plant_region_map_table="plant_region_map_ipm",
        settings_agg_key="region_aggregations",
        return_retirement_capacity=False,
    ):
        """
        Calculation of average unit characteristics within a technology cluster (capacity,
        minimum load, heat rate) and the number of units in the cluster.

        Parameters
        ----------
        plant_region_map_table : str, optional
            Name of the table with region names for each plant, by default
            "plant_region_map_ipm"
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
        # start = dt.now()

        logger.info("Loading map of plants to IPM regions")
        self.plant_region_map = load_plant_region_map(
            self.gens_860,
            self.pudl_engine,
            self.settings,
            self.model_regions_gdf,
            table=plant_region_map_table,
            settings_agg_key=settings_agg_key,
        )
        # check1 = dt.now()
        # check1_diff = (check1 - start).total_seconds()
        # print(f"{check1_diff} seconds to load plant_region_map")

        # logger.info("Loading EIA860 generator data")

        self.gens_860_model = supplement_generator_860_data(
            self.gens_860,
            self.pudl_engine,
            self.settings,
            self.pudl_out,
            self.plant_region_map,
            data_years=self.data_years,
        )
        # check2 = dt.now()
        # check2_diff = (check2 - check1).total_seconds()
        # print(f"{check2_diff} seconds to load 860 generator data")

        # logger.info(f"Loading EIA923 fuel and generation data for {settings['data_years']}")
        self.annual_gen_923 = load_generator_923_data(
            self.pudl_engine,
            self.pudl_out,
            model_region_map=self.plant_region_map,
            data_years=self.data_years,
        )
        # check3 = dt.now()
        # check3_diff = (check3 - check2).total_seconds()
        # print(f"{check3_diff} seconds to load 923 generator data")

        # Add heat rates to the data we already have from 860
        logger.info("Loading heat rate data for units and generator/fuel combinations")
        weighted_unit_hr, prime_mover_hr_map = unit_generator_heat_rates(
            self.pudl_out, self.annual_gen_923, self.data_years
        )
        # check4 = dt.now()
        # check4_diff = (check4 - check3).total_seconds()
        # print(f"{check4_diff} seconds to load heat rate data")

        # Merge the PUDL calculated heat rate data and set the index for easy
        # mapping using plant/prime mover heat rates from 923
        hr_cols = ["plant_id_eia", "unit_id_pudl", "heat_rate_mmbtu_mwh"]
        idx = ["plant_id_eia", "prime_mover_code", "energy_source_code_1"]
        units_model = self.gens_860_model.merge(
            weighted_unit_hr[hr_cols], on=["plant_id_eia", "unit_id_pudl"], how="left"
        ).set_index(idx)

        logger.info(
            f"Units model technologies are "
            f"{units_model.technology_description.unique().tolist()}"
        )
        # print(units_model.head())

        logger.info(
            "Assigning technology/fuel heat rates where unit heat rates are not "
            "available"
        )
        units_model.loc[
            units_model.heat_rate_mmbtu_mwh.isnull(), "heat_rate_mmbtu_mwh"
        ] = units_model.loc[units_model.heat_rate_mmbtu_mwh.isnull()].index.map(
            prime_mover_hr_map
        )

        logger.info(
            f"Units model technologies are "
            f"{units_model.technology_description.unique().tolist()}"
        )
        logger.info(
            f"Before adding proposed generators, {len(units_model)} units with "
            f"{units_model[self.settings['capacity_col']].sum()} MW capacity"
        )
        proposed_gens = import_proposed_generators(
            settings=self.settings, model_regions_gdf=self.model_regions_gdf
        )
        logger.info(
            f"Proposed gen technologies are "
            f"{proposed_gens.technology_description.unique().tolist()}"
        )
        logger.info(f"{proposed_gens[self.settings['capacity_col']].sum()} MW proposed")
        units_model = pd.concat([proposed_gens, units_model], sort=False)

        # Create a pudl unit id based on plant and generator id where one doesn't exist.
        # This is used later to match the cluster numbers to plants
        units_model.reset_index(inplace=True)
        units_model.loc[units_model.unit_id_pudl.isnull(), "unit_id_pudl"] = (
            units_model.loc[units_model.unit_id_pudl.isnull(), "plant_id_eia"].astype(
                str
            )
            + "_"
            + units_model.loc[units_model.unit_id_pudl.isnull(), "generator_id"].astype(
                str
            )
        ).values
        units_model.set_index(idx, inplace=True)

        logger.info(
            f"After adding proposed, units model technologies are "
            f"{units_model.technology_description.unique().tolist()}"
        )
        # logger.info(units_model['technology_description'].unique().tolist())
        logger.info(
            f"After adding proposed generators, {len(units_model)} units with "
            f"{units_model[self.settings['capacity_col']].sum()} MW capacity"
        )
        # return units_model

        techs = list(self.settings["num_clusters"])
        # logger.info(f"Technology clusters include {', '.join(techs)}")

        num_clusters = {}
        for region in self.settings["model_regions"]:
            num_clusters[region] = self.settings["num_clusters"].copy()

        for region in self.settings["alt_clusters"]:
            for tech, cluster_size in self.settings["alt_clusters"][region].items():
                num_clusters[region][tech] = cluster_size

        region_tech_grouped = units_model.loc[
            (units_model.technology_description.isin(techs))
            & (units_model.retirement_year >= self.settings["model_year"]),
            :,
        ].groupby(["model_region", "technology_description"])

        # if return_retirement_capacity:
        self.retired = units_model.loc[
            units_model.retirement_year < self.settings["model_year"], :
        ]

        # For each group, cluster and calculate the average size, min load, and heat rate
        # logger.info("Creating technology clusters by region")
        print("Creating technology clusters by region")
        unit_list = []
        cluster_list = []
        for _, df in region_tech_grouped:
            region, tech = _
            # logger.info(f"{region}, {tech}")

            grouped = group_units(df, self.settings)

            # try:
            clusters = cluster.KMeans(
                n_clusters=num_clusters[region][tech], random_state=6
            ).fit(preprocessing.StandardScaler().fit_transform(grouped))
            # except:
            #     print(grouped)

            grouped["cluster"] = clusters.labels_ + 1  # Change to 1-index for julia

            # Saving individual unit data for later analysis (if needed)
            unit_list.append(grouped)

            _df = calc_unit_cluster_values(grouped, self.settings, tech)
            _df["region"] = region
            cluster_list.append(_df)

        # Save some data about individual units for easy access
        self.units_model = units_model
        self.all_units = pd.concat(unit_list, sort=False)
        self.all_units = pd.merge(
            self.units_model,
            self.all_units["cluster"],
            on=["plant_id_eia", "unit_id_pudl"],
            how="left",
        )

        logger.info("Finalizing generation clusters")
        self.results = pd.concat(cluster_list)
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
            capacity_factors = gentype_region_capacity_factor(
                self.pudl_engine, self.plant_region_map, self.settings
            )

            self.results = pd.merge(
                self.results.reset_index(),
                capacity_factors[["region", "technology", "capacity_factor"]],
                on=["region", "technology"],
                how="left",
            )

            if self.settings["derate_capacity"] is True:
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

        # Set the index to region, tech, cluster again
        self.results.set_index(["region", "technology", "cluster"], inplace=True)
        self.results["Existing_Cap_MW"] = self.results.Cap_size * self.results.num_units
        self.results["unmodified_existing_cap_mw"] = (
            self.results["unmodified_cap_size"] * self.results["num_units"]
        )

        self.results = add_model_tags(self.results, self.settings)

        # Convert technology names to snake_case and add a 1-indexed column R_ID
        self.results = self.results.reset_index()
        self.results["technology"] = snake_case_col(self.results["technology"])

        # Set Min_power of wind/solar to 0
        self.results.loc[self.results["DISP"] == 1, "Min_power"] = 0

        self.results.set_index(["region", "technology"], inplace=True)
        self.results["R_ID"] = np.array(range(len(self.results))) + 1

        logger.info(
            f"Capacity of {self.results['Existing_Cap_MW'].sum()} MW in final clusters"
        )

        # logger.info(f"{(dt.now() - start).total_seconds()} seconds total")

        if return_retirement_capacity:
            return self.results, retired
        else:
            return self.results
