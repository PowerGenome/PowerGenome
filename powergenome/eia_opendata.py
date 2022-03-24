"""
Load data from EIA's Open Data API. Requires an api key, which should be included in a
.env file (/powergenome/.env) with the format EIA_API_KEY=YOUR_API_KEY
"""

from itertools import product
import logging
from typing import Union

import pandas as pd
import requests

from powergenome.params import SETTINGS, DATA_PATHS
from powergenome.price_adjustment import inflation_price_adjustment

logger = logging.getLogger(__name__)

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
        Data from EIA's AEO via their open data API.
    """
    data_dir = DATA_PATHS["eia"] / "open_data"
    data_dir.mkdir(exist_ok=True)
    if not (data_dir / f"{series_id}.csv").exists():
        url = f"https://api.eia.gov/series/?series_id={series_id}&api_key={api_key}&out=json"
        r = requests.get(url)

        try:
            df = pd.DataFrame(
                r.json()["series"][0]["data"], columns=columns, dtype=float
            )
        except KeyError:
            print(
                "There was an error creating a dataframe from your EIA AEO data request. "
                f"The constructed series ID is {series_id}. Check to make sure it looks "
                "correct. The data returned from EIA's API is: \n"
                f"{r.json()}"
            )
        df.to_csv(data_dir / f"{series_id}.csv", index=False)
    else:
        df = pd.read_csv(data_dir / f"{series_id}.csv")

    return df


def fetch_fuel_prices(settings: dict, inflate_price: bool = True) -> pd.DataFrame:
    """
    Get EIA AEO fuel prices for all regions, fuel types, and scenarios (series IDs)
    included in the settings.

    Parameters
    ----------
    settings : dict
        Should include the following keys:
            eia_aeo_year (int)
            eia_series_region_names (list)
            eia_series_fuel_names (list)
            eia_series_scenario_names (list)
    inflate_price: bool
        If True, adjust the AEO prices to the year "target_usd_year" from the settings.
        If False, do not adjust the AEO prices. Requires the additional settings keys
        "target_usd_year" and "aeo_fuel_usd_year".

    Returns
    -------
    pd.DataFrame
        All fuel price data from AEO for the product of regions, fuels, and scenarios
        included in the settings dictionary.

    Examples
    --------
    Prepare the settings dictionary
    >>> settings = {}
    >>> settings["eia_aeo_year"] = 2020
    >>> settings["aeo_fuel_usd_year"] = 2019
    >>> settings["eia_series_scenario_names"] = {"reference": "REF2020"}
    >>> settings["eia_series_fuel_names"] = {"coal": "STC"}
    >>> settings["eia_series_region_names"] = {"mountain": "MTN"}

    Find the fuel cost with inflating costs.

    >>> fuel_price = fetch_fuel_prices(settings, inflate_price=False)
    >>> print(fuel_price.head())
       year     price  fuel    region   scenario           full_fuel_name
    0  2050  1.501850  coal  mountain  reference  mountain_reference_coal
    1  2049  1.488098  coal  mountain  reference  mountain_reference_coal
    2  2048  1.508208  coal  mountain  reference  mountain_reference_coal
    3  2047  1.506809  coal  mountain  reference  mountain_reference_coal
    4  2046  1.497366  coal  mountain  reference  mountain_reference_coal

    If either of the keys "target_usd_year" or "aeo_fuel_usd_year" is missing, fuel prices
    cannot be inflated.

    >>> fuel_price = fetch_fuel_prices(settings)
    ************
    Unable to inflate fuel prices. Check your settings file to ensure the keys
    "target_usd_year" and "aeo_fuel_usd_year" exist and are valid integers.
    ************
    """
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

    if inflate_price:
        try:
            fuel_price_base_year = settings["aeo_fuel_usd_year"]
            fuel_price_target_year = settings["target_usd_year"]
            final.loc[:, "price"] = inflation_price_adjustment(
                price=final.loc[:, "price"],
                base_year=fuel_price_base_year,
                target_year=fuel_price_target_year,
            )
        except (KeyError, TypeError):
            logger.warning(
                """
    ************
    Unable to inflate fuel prices. Check your settings file to ensure the keys
    "target_usd_year" and "aeo_fuel_usd_year" are valid integers.
    ************
                """
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

    Examples
    --------
    >>> texas_load = get_aeo_load("TRE", 2020, "REF2020")
    >>> print(texas_load.head())
       year      demand
    0  2050  489.009247
    1  2049  483.176544
    2  2048  477.624481
    3  2047  472.314972
    4  2046  466.875671
    """

    data_dir = DATA_PATHS["eia"] / "open_data"
    data_dir.mkdir(exist_ok=True)

    API_KEY = SETTINGS["EIA_API_KEY"]

    SERIES_ID = (
        f"AEO.{aeo_year}.{scenario_series}.CNSM_NA_ELEP_NA_ELC_NA_{region}_BLNKWH.A"
    )

    df = load_aeo_series(SERIES_ID, API_KEY, columns=["year", "demand"])
    df["year"] = df["year"].astype(int)

    return df
