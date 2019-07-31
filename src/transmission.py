import itertools
import logging
import numpy as np
import pandas as pd
from src.util import reverse_dict_of_lists, map_agg_region_names

logger = logging.getLogger(__name__)


def agg_transmission_constraints(
    pudl_engine,
    settings,
    pudl_table="transmission_single_ipm",
    settings_agg_key="region_aggregations",
):

    zones = settings["model_regions"]
    zone_num_map = {
        zone: f"z{number + 1}" for zone, number in zip(zones, range(len(zones)))
    }

    combos = list(itertools.combinations(zones, 2))
    reverse_combos = [(combo[-1], combo[0]) for combo in combos]

    logger.info("Loading transmission constraints from PUDL")
    transmission_constraints_table = pd.read_sql_table(pudl_table, con=pudl_engine)
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

    # Create new column "model_region_from"  and "model_region_to" with labels that
    # we're using for aggregated regions
    transmission_constraints_table = transmission_constraints_table.loc[
        (transmission_constraints_table.region_from.isin(keep_regions))
        & (transmission_constraints_table.region_to.isin(keep_regions)),
        :,
    ].drop(columns="id")

    logger.info("Map and aggregate region names for transmission constraints")
    for col in ["region_from", "region_to"]:
        model_col = "model_" + col

        transmission_constraints_table = map_agg_region_names(
            df=transmission_constraints_table,
            region_agg_map=region_agg_map,
            original_col_name=col,
            new_col_name=model_col,
        )

    transmission_constraints_table.drop(
        columns=["firm_ttc_mw", "tariff_mills_kwh"], inplace=True
    )
    transmission_constraints_table = transmission_constraints_table.groupby(
        ["model_region_from", "model_region_to"]
    ).sum()

    # Build the final output dataframe
    logger.info(
        "Build a new transmission constraints dataframe with a single line between"
        "regions"
    )
    tc_joined = pd.DataFrame(
        columns=["Network_lines"] + zones + ["Line_Max_Flow_MW", "Line_Min_Flow_MW"],
        index=transmission_constraints_table.reindex(combos).dropna().index,
        data=0,
    )
    tc_joined["Network_lines"] = range(1, len(tc_joined) + 1)
    tc_joined["Line_Max_Flow_MW"] = transmission_constraints_table.reindex(
        combos
    ).dropna()

    reverse_tc = transmission_constraints_table.reindex(reverse_combos).dropna() * -1
    reverse_tc.index = tc_joined.index
    tc_joined["Line_Min_Flow_MW"] = reverse_tc

    for idx, row in tc_joined.iterrows():
        tc_joined.loc[idx, idx[0]] = 1
        tc_joined.loc[idx, idx[-1]] = -1

    tc_joined.rename(columns=zone_num_map, inplace=True)
    tc_joined = tc_joined.reset_index()
    tc_joined["Transmission Path Name"] = (
        tc_joined["model_region_from"] + "_to_" + tc_joined["model_region_to"]
    )
    tc_joined = tc_joined.set_index("Transmission Path Name")
    tc_joined.drop(columns=["model_region_from", "model_region_to"], inplace=True)

    return tc_joined
