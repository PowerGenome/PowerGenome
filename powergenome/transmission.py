"""
Transmission constraints between regions and line distance
"""

import itertools
import logging
from math import asin, cos, radians, sin, sqrt
from pathlib import Path

import pandas as pd
import sqlalchemy as sa

from powergenome.util import map_agg_region_names, reverse_dict_of_lists, find_centroid

logger = logging.getLogger(__name__)


def agg_transmission_constraints(
    pg_engine: sa.engine.base.Engine,
    settings: dict,
    pg_table: str = "transmission_single_epaipm",
    settings_agg_key: str = "region_aggregations",
) -> pd.DataFrame:
    """Aggregate transmission constraints/capacity between model regions

    Model regions can consist of one or more individual regions. When two or more regions
    are in a model region, the transmission capacity between individual regions is
    combined.

    Values in a user transmission table will override the database transmission table.

    Parameters
    ----------
    pg_engine : sa.engine.base.Engine
        Engine to conect with a database
    settings : dict
        Dictionary of settings parameters. Must include "model_regions". Optional parameters
        include "tx_value_col" (name of the data column in `pg_table`, default value is
        "firm_ttc_mw"), "user_transmission_constraints_fn" (name of user data file, must
        be combined with the parameter "input_folder"), and the value of `settings_agg_key`
        if any regions are aggregated.
    pg_table : str, optional
        Name of the database table with transmission capacity, by default "transmission_single_epaipm"
    settings_agg_key : str, optional
        Name of the settings parameter where regions are aggregated, by default "region_aggregations"

    Returns
    -------
    pd.DataFrame
        Network lines connecting regions and the min/max flow of each line (if different
        based on direction)

    Raises
    ------
    KeyError
        The specified data column is not in the database transmission table
    KeyError
        The database transmission table has duplicate lines in the same direction
    KeyError
        The specified data column is not in the user supplied transmission table
    KeyError
        The user transmission table has duplicate lines in the same direction
    """
    tx_value_col = settings.get("tx_value_col")
    if not tx_value_col:
        logger.warning(
            "No transmission value column (e.g. firm vs non-firm) was specified in the "
            "settings. The column 'firm_ttc_mw' will be used as a default. This is a change "
            "from previous versions of PG, where 'nonfirm_ttc_mw' was used. Firm transmission "
            "capacity is lower or equal to non-firm capacity."
        )
        tx_value_col = "firm_ttc_mw"
    zones = settings["model_regions"]
    zone_num_map = {
        zone: f"z{number + 1}" for zone, number in zip(zones, range(len(zones)))
    }

    combos = list(itertools.combinations(zones, 2))
    reverse_combos = [(combo[-1], combo[0]) for combo in combos]

    logger.info("Loading transmission constraints from PUDL")
    transmission_constraints_table = pd.read_sql_table(pg_table, con=pg_engine)

    if tx_value_col not in transmission_constraints_table.columns:
        raise KeyError(
            f"There is no column {tx_value_col} in the transmission capacity table '{pg_table}'"
        )
    if transmission_constraints_table.duplicated(
        subset=["region_from", "region_to"]
    ).any():
        dup_lines = transmission_constraints_table.loc[
            transmission_constraints_table.duplicated(
                subset=["region_from", "region_to"]
            ),
            ["region_from", "region_to"],
        ]

        raise KeyError(
            "The transmission table has duplicate lines. This table should only have unique lines.\n",
            dup_lines,
        )
    if settings.get("user_transmission_constraints_fn"):
        user_tx_constraints = pd.read_csv(
            Path(settings["input_folder"])
            / settings["user_transmission_constraints_fn"]
        )
        if tx_value_col not in user_tx_constraints.columns:
            raise KeyError(
                f"There is no column {tx_value_col} in the user supplied transmission capacity table"
            )
        if user_tx_constraints.duplicated(subset=["region_from", "region_to"]).any():
            dup_lines = user_tx_constraints.loc[
                user_tx_constraints.duplicated(subset=["region_from", "region_to"]),
                ["region_from", "region_to"],
            ]

            raise KeyError(
                "The user transmission table has duplicate lines. This table should only have unique lines.\n",
                dup_lines,
            )

        # user constraints are needed bidirectionaly
        transmission_constraints_table = pd.concat(
            [
                transmission_constraints_table,
                user_tx_constraints,
                user_tx_constraints.rename(
                    columns={"region_from": "region_to", "region_to": "region_from"}
                ),
            ]
        )

        if transmission_constraints_table.duplicated(
            subset=["region_from", "region_to"]
        ).any():
            logger.warning(
                "The user transmission capacity table duplicates values from the "
                "database. Database values will be discarded in favor of user values."
            )

        transmission_constraints_table = transmission_constraints_table.drop_duplicates(
            keep="last"
        )

    # Settings has a dictionary of lists for regional aggregations. Need
    # to reverse this to use in a map method.
    region_agg_map = reverse_dict_of_lists(settings.get(settings_agg_key))

    # IPM regions to keep. Regions not in this list will be dropped from the
    # dataframe
    keep_regions = [
        x
        for x in settings["model_regions"] + list(region_agg_map)
        if x not in region_agg_map.values()
    ]

    # Create new column "model_region_from"  and "model_region_to" with labels that
    # we're using for aggregated regions
    transmission_constraints_table = transmission_constraints_table.loc[
        (transmission_constraints_table.region_from.isin(keep_regions))
        & (transmission_constraints_table.region_to.isin(keep_regions)),
        :,
    ].drop(columns="id", errors="ignore")

    logger.info("Map and aggregate region names for transmission constraints")
    for col in ["region_from", "region_to"]:
        model_col = "model_" + col

        transmission_constraints_table = map_agg_region_names(
            df=transmission_constraints_table,
            region_agg_map=region_agg_map,
            original_col_name=col,
            new_col_name=model_col,
        )

    keep_cols = ["model_region_from", "model_region_to", tx_value_col]
    drop_cols = [
        c for c in transmission_constraints_table.columns if c not in keep_cols
    ]
    transmission_constraints_table.drop(columns=drop_cols, inplace=True)
    transmission_constraints_table = transmission_constraints_table.groupby(
        ["model_region_from", "model_region_to"]
    ).sum()

    # Build the final output dataframe
    logger.info(
        "Build a new transmission constraints dataframe with a single line between "
        "regions"
    )
    tc_joined = pd.DataFrame(
        columns=["Network_Lines"] + zones + ["Line_Max_Flow_MW", "Line_Min_Flow_MW"],
        index=transmission_constraints_table.reindex(combos).dropna().index,
        data=0,
    )

    if tc_joined.empty:
        logger.info(f"No transmission lines exist between model regions {combos}")
        tc_joined["transmission_path_name"] = None
        tc_joined.rename(columns=zone_num_map, inplace=True)
        return tc_joined.reset_index(drop=True)

    tc_joined["Network_Lines"] = range(1, len(tc_joined) + 1)
    tc_joined["Line_Max_Flow_MW"] = transmission_constraints_table.reindex(
        combos
    ).dropna()

    reverse_tc = transmission_constraints_table.reindex(reverse_combos).dropna() * -1
    reverse_tc.index = tc_joined.index
    tc_joined["Line_Min_Flow_MW"] = reverse_tc

    for idx, _ in tc_joined.iterrows():
        tc_joined.loc[idx, idx[0]] = 1
        tc_joined.loc[idx, idx[-1]] = -1

    tc_joined.rename(columns=zone_num_map, inplace=True)
    tc_joined = tc_joined.reset_index()
    tc_joined["transmission_path_name"] = (
        tc_joined["model_region_from"] + "_to_" + tc_joined["model_region_to"]
    )
    # tc_joined = tc_joined.set_index("Transmission Path Name")
    tc_joined.drop(columns=["model_region_from", "model_region_to"], inplace=True)

    return tc_joined


def haversine(lon1, lat1, lon2, lat2, units="mile"):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    https://gis.stackexchange.com/questions/166820/geopandas-return-lat-and-long-of-a-centroid-point
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))

    if units == "mile":
        r = 3956  # Radius of earth in miles. Use 6371 for kilometers, 3956 for miles
    elif units == "km":
        r = 6371
    else:
        raise ValueError(f"Units are {units}, but should be 'mile' or 'km'")

    return c * r


def getXY(pt):
    "Return the X and Y parts of a coordinate point"
    return (pt.x, pt.y)


def single_line_distance(line_name, region_centroids, units):
    """Calculate the transmission line distance between centroids of two regions.

    Parameters
    ----------
    line_name : str
        Two region names in the format <start>_to_<end>
    region_centroids : geoseries
        Centroid points of each region with region names as the index
    units : str
        Name of the distance units to use. Options are 'mile' or 'km'.

    Returns
    -------
    float
        The distance
    """

    start, end = line_name.split("_to_")
    start_lon, start_lat = getXY(region_centroids[start])
    end_lon, end_lat = getXY(region_centroids[end])
    distance = haversine(start_lon, start_lat, end_lon, end_lat, units=units)

    return distance


def transmission_line_distance(
    trans_constraints_df, ipm_shapefile, settings, units="mile"
):
    logger.info("Calculating transmission line distance")
    ipm_shapefile["geometry"] = ipm_shapefile.buffer(0.01)
    model_polygons = ipm_shapefile.dissolve(by="model_region")
    model_polygons = model_polygons.to_crs(epsg=4326)
    region_centroids = find_centroid(model_polygons)

    distances = [
        single_line_distance(line_name, region_centroids, units=units)
        for line_name in trans_constraints_df["transmission_path_name"]
    ]
    trans_constraints_df[f"distance_{units}"] = distances
    trans_constraints_df[f"distance_{units}"] = trans_constraints_df[
        f"distance_{units}"
    ]

    return trans_constraints_df
