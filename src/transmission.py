import numpy as np
import pandas as pd
from src.util import reverse_dict_of_lists, map_agg_region_names


def agg_transmission_constraints(
    pudl_engine,
    settings,
    pudl_table="transmission_single_ipm",
    settings_agg_key="region_aggregations",
):

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

    # Create a new column "model_region" with labels that we're using for aggregated
    # regions
    transmission_constraints_table = transmission_constraints_table.loc[
        (transmission_constraints_table.region_from.isin(keep_regions))
        & (transmission_constraints_table.region_to.isin(keep_regions)),
        :,
    ].drop(columns="id")

    for col in ["region_from", "region_to"]:
        model_col = "model_" + col

        transmission_constraints_table = map_agg_region_names(
            df=transmission_constraints_table,
            region_agg_map=region_agg_map,
            original_col_name=col,
            new_col_name=model_col
        )

        # transmission_constraints_table.loc[
        #     :, model_col
        # ] = transmission_constraints_table.loc[:, col]
        # transmission_constraints_table.loc[
        #     transmission_constraints_table[col].isin(region_agg_map.keys()), model_col
        # ] = transmission_constraints_table.loc[
        #     transmission_constraints_table[col].isin(region_agg_map.keys()), col
        # ].map(
        #     region_agg_map
        # )

    tc_square = transmission_constraints_table.pivot_table(
        index="model_region_from",
        columns="model_region_to",
        values="nonfirm_ttc_mw",
        aggfunc="sum",
    ).fillna(0)
    np.fill_diagonal(tc_square.values, -1)

    return tc_square
