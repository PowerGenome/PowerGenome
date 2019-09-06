"Load and download data for use in other modules"

import sqlite3

import geopandas as gpd
import pandas as pd
import requests
import sqlalchemy as sa
from bs4 import BeautifulSoup
from xlrd import XLRDError

from powergenome.params import IPM_GEOJSON_PATH


def load_ipm_plant_region_map(pudl_engine):
    """Load the table associating each power plant to an IPM region

    Parameters
    ----------
    pudl_engine : sqlalchemy.Engine
        A sqlalchemy connection for use by pandas

    Returns
    -------
    dataframe
        All plants in the NEEDS database and their associated IPM region. Columns are
        plant_id_eia and region.
    """
    region_map_df = pd.read_sql_table(
        "plant_region_map_ipm", con=pudl_engine, columns=["plant_id_eia", "region"]
    )

    return region_map_df


def load_ownership_eia860(pudl_engine, data_years=[2017]):

    cols = [
        "report_date",
        "utility_id_eia",
        "plant_id_eia",
        "generator_id",
        "operational_status_code",
        "owner_utility_id_eia",
        "owner_name",
        "owner_state",
        "fraction_owned",
    ]
    ownership = pd.read_sql_table(
        "ownership_eia860", pudl_engine, columns=cols, parse_dates=["report_date"]
    )
    ownership = ownership.loc[ownership["report_date"].dt.year.isin(data_years)]

    return ownership
