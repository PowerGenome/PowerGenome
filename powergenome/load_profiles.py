import pandas as pd
import logging
from powergenome.util import reverse_dict_of_lists

logger = logging.getLogger(__name__)


def load_curves(
    pudl_engine,
    settings,
    pudl_table="load_curves_epaipm",
    settings_agg_key="region_aggregations",
):

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

    # I'd rather use a sql query and only pull the regions of interest but
    # sqlalchemy doesn't allow table names to be parameterized.
    logger.info("Loading load curves from PUDL")
    load_curves = pd.read_sql_table(
        pudl_table, pudl_engine, columns=["region_id_epaipm", "time_index", "load_mw"]
    )

    load_curves = load_curves.loc[load_curves.region_id_epaipm.isin(keep_regions)]

    # Increase demand to account for load growth
    load_curves = add_load_growth(load_curves, settings)

    # Set a new column "region" to the old column values. Then replace values for any
    # regions that are being aggregated
    load_curves.loc[:, "region"] = load_curves.loc[:, "region_id_epaipm"]

    load_curves.loc[
        load_curves.region_id_epaipm.isin(region_agg_map.keys()), "region"
    ] = load_curves.loc[
        load_curves.region_id_epaipm.isin(region_agg_map.keys()), "region_id_epaipm"
    ].map(
        region_agg_map
    )

    logger.info("Aggregating load curves in grouped regions")
    load_curves_agg = load_curves.groupby(["region", "time_index"]).sum()

    lc_wide = load_curves_agg.unstack(level=0)
    lc_wide.columns = lc_wide.columns.droplevel()

    return lc_wide


def add_load_growth(load_curves, settings):

    load_map = reverse_dict_of_lists(settings["load_region_map"])
    load_growth_map = {
        ipm_region: settings["default_growth_rates"][load_region]
        for ipm_region, load_region in load_map.items()
    }

    if settings["alt_growth_rate"] is not None:
        for region, rate in settings["alt_growth_rate"].items():
            print(region, rate)
            load_growth_map[region] = rate

    years_growth = settings["model_year"] - settings["default_load_year"]
    load_growth_factor = {
        region: (1 + rate) ** years_growth for region, rate in load_growth_map.items()
    }

    for region in load_curves["region_id_epaipm"].unique():
        growth_factor = load_growth_factor[region]
        print(region, growth_factor)
        load_curves.loc[
            load_curves["region_id_epaipm"] == region, "load_mw"
        ] *= growth_factor

    return load_curves
