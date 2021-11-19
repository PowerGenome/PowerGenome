"""
Load data from EIA's Open Data API. Requires an api key, which should be included in a
.env file (/powergenome/.env) with the format EIA_API_KEY=YOUR_API_KEY
"""

from itertools import product
from typing import Union

import pandas as pd
import requests

from powergenome.params import SETTINGS, DATA_PATHS
from powergenome.price_adjustment import inflation_price_adjustment

numeric = Union[int, float]


def load_aeo_series(series_id: str, api_key: str, columns: list = None) -> pd.DataFrame:
    """Load EIA AEO data either from file (if it exists) or from the API.

    Parameters
    ----------
    series_id : str
        The AEO API series ID that uniquely identifies the data request.
    api_key : str
        A valid API key for EIA's open data portal
    columns : list
        The expected output dataframe columns

    Returns
    -------
    pd.DataFrame
        [description]
    """
    data_dir = DATA_PATHS["eia"] / "open_data"
    if not (data_dir / f"{series_id}.csv").exists():
        url = f"http://api.eia.gov/series/?series_id={series_id}&api_key={api_key}&out=json"
        r = requests.get(url)
        df = pd.DataFrame(r.json()["series"][0]["data"], columns=columns, dtype=float)
        df.to_csv(data_dir / f"{series_id}.csv", index=False)
    else:
        df = pd.read_csv(data_dir / f"{series_id}.csv")

    return df


def fetch_fuel_prices(settings):
    API_KEY = SETTINGS["EIA_API_KEY"]

    aeo_year = settings["eia_aeo_year"]

    fuel_price_cases = product(
        settings["eia_series_region_names"].items(),
        settings["eia_series_fuel_names"].items(),
        settings["eia_series_scenario_names"].items(),
    )

    df_list = []
    for region, fuel, scenario in fuel_price_cases:
        region_name, region_series = region
        fuel_name, fuel_series = fuel
        scenario_name, scenario_series = scenario

        SERIES_ID = f"AEO.{aeo_year}.{scenario_series}.PRCE_REAL_ELEP_NA_{fuel_series}_NA_{region_series}_Y13DLRPMMBTU.A"

        df = load_aeo_series(
            series_id=SERIES_ID, api_key=API_KEY, columns=["year", "price"]
        )
        df["fuel"] = fuel_name
        df["region"] = region_name
        df["scenario"] = scenario_name
        df["full_fuel_name"] = df.region + "_" + df.scenario + "_" + df.fuel
        df["year"] = df["year"].astype(int)

        df_list.append(df)

    final = pd.concat(df_list, ignore_index=True)

    fuel_price_base_year = settings["aeo_fuel_usd_year"]
    fuel_price_target_year = settings["target_usd_year"]
    final.loc[:, "price"] = inflation_price_adjustment(
        price=final.loc[:, "price"],
        base_year=fuel_price_base_year,
        target_year=fuel_price_target_year,
    )

    return final


def get_aeo_load(
    region: str, aeo_year: Union[str, numeric], scenario_series: str
) -> pd.DataFrame:
    """Find the electricity demand in a single AEO region. Use EIA API if data has not
    been previously saved.

    Parameters
    ----------
    region : str
        Short name of the AEO region
    aeo_year : Union[str, numeric]
        AEO data year
    scenario_series : str
        Short name of the AEO scenario

    Returns
    -------
    pd.DataFrame
        The demand data for a single region.
    """

    data_dir = DATA_PATHS["eia"] / "open_data"
    data_dir.mkdir(exist_ok=True)

    API_KEY = SETTINGS["EIA_API_KEY"]

    SERIES_ID = (
        f"AEO.{aeo_year}.{scenario_series}.CNSM_NA_ELEP_NA_ELC_NA_{region}_BLNKWH.A"
    )

    if not (data_dir / f"{SERIES_ID}.csv").exists():
        url = f"http://api.eia.gov/series/?series_id={SERIES_ID}&api_key={API_KEY}&out=json"
        r = requests.get(url)
        df = pd.DataFrame(
            r.json()["series"][0]["data"], columns=["year", "demand"], dtype=float
        )
        df.to_csv(data_dir / f"{SERIES_ID}.csv", index=False)
    else:
        df = pd.read_csv(data_dir / f"{SERIES_ID}.csv")

    return df
