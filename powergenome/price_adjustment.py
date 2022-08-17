"""
Adjust price/cost from one year to another
"""

import logging

import requests
import json
from typing import NamedTuple, Union
from datetime import date
import pandas as pd
import numpy as np
from pathlib import Path
from powergenome.params import DATA_PATHS


class MonthlyCPI(NamedTuple):
    year: int
    period: int
    value: float


logger = logging.getLogger(__name__)


def get_cpi_data(start_year: int = 1980, end_year: int = None) -> pd.DataFrame:
    """Get monthly consumer price index data from the US BLS API, create annual averages.

    BLS has 2 versions of their API. V1 does not require an API key but it restricts the
    number of API queries per day and only returns monthly data. Since V2 requires a key
    I'm using V1, even though it requires a little extra work to go from monthly to
    annual data.

    According to BLS documentation (https://www.bls.gov/cpi/factsheets/cpi-math-calculations.pdf),
    "Annual averages are the sum of the 12 monthly data points (i.e. indexes), divided
    by 12."

    Parameters
    ----------
    start_year : int, optional
        First year of CPI data in the timeseries, by default 1980
    end_year : int, optional
        Final year of CPI data in the timeseries, by default None. If value is None,
        the current year will be used.

    Returns
    -------
    pd.DataFrame
        Annual averages of BLS CPI for all years from `start_year` to `end_year` (inclusive)
        that have 12 months of data.

    Examples
    --------
    >>> get_cpi_data(start_year=2015, end_year=2020)
        year    period  value
    0   2015    12      237.017
    1   2016    12      240.007167
    2   2017    12      245.119583
    3   2018    12      251.106833
    4   2019    12      255.657417
    5   2020    12      258.811167
    """
    if end_year is None:
        todays_date = date.today()
        end_year = todays_date.year
    headers = {"Content-type": "application/json"}

    df_list = []
    e_y = start_year + 10
    while start_year <= end_year:
        data = json.dumps(
            {
                "seriesid": ["CUUR0000SA0"],
                "startyear": str(start_year),
                "endyear": str(e_y),
            }
        )
        p = requests.post(
            "https://api.bls.gov/publicAPI/v1/timeseries/data/",
            data=data,
            headers=headers,
        )
        json_data = json.loads(p.text)

        data_list = []
        for m_data in json_data["Results"]["series"][0]["data"]:
            monthly_cpi = MonthlyCPI(
                int(m_data["year"]),
                int(m_data["period"].lstrip("M")),
                float(m_data["value"]),
            )
            data_list.append(monthly_cpi)

        m_cpi_df = pd.DataFrame(data_list)
        a_cpi_df = m_cpi_df.groupby("year", as_index=False).agg(
            {"period": "count", "value": "mean"}
        )
        a_cpi_df = a_cpi_df.query("period == 12")
        df_list.append(a_cpi_df)
        start_year = e_y
        e_y = start_year + 10

    annual_cpi = pd.concat(df_list)

    return annual_cpi


def load_cpi_data(reload_data: bool = False, **kwargs) -> pd.DataFrame:
    """Load BLS CPI data from CSV file if it exists and `reload_data` is False, otherwise
    get it from the API and write results to file.

    Parameters
    ----------
    reload_data : bool, optional
        If data should be reloaded from the BLS API, by default False
    **kwargs: optional
        Optional keyword arguments for the function `get_cpi_data`. Only used if
        `reload_data` is True or the CPI data file doesn't already exist.

    Returns
    -------
    pd.DataFrame
        Annual averages of BLS CPI. If data are reloaded then the results will be
    """

    if reload_data or not DATA_PATHS["cpi_data"].exists():
        DATA_PATHS["cpi_data"].parent.mkdir(exist_ok=True)
        start_year = 1980
        end_year = None
        for k, v in kwargs.items():
            if k == "start_year":
                start_year = v
            if k == "end_year":
                end_year = v
        cpi_data = get_cpi_data(start_year, end_year)
        cpi_data.to_csv(DATA_PATHS["cpi_data"], index=False, float_format="%g")
    else:
        cpi_data = pd.read_csv(DATA_PATHS["cpi_data"])

    return cpi_data


def inflation_price_adjustment(
    price: Union[int, float, pd.Series, pd.DataFrame, np.ndarray],
    base_year: int,
    target_year: int,
) -> float:
    """Convert costs from one dollar-year to another dollar-year using BLS annual CPI data.

    Parameters
    ----------
    price : Union[int, float, pd.Series, pd.DataFrame, np.ndarray]
        The cost to adjust. Can be a single float or an object compatible with broadcast
        multiplication
    base_year : int
        The original data dollar-year
    target_year : int
        The target dollar-year

    Returns
    -------
    Union[int, float, pd.Series, pd.DataFrame, np.ndarray]
        Cost data transformed from the base dollar-year to the target dollar_year

    Raises
    ------
    ValueError
        The target dollar year is greater than years available with 12 months of data.
    ValueError
        The base dollar year is lower than the minimum year available.

    Examples
    --------
    >>> p = 10.0
    >>> inflation_price_adjustment(p, 2000, 2010)
    1.2662921022067364
    >>> p = pd.Series([1, 10, 100])
    >>> inflation_price_adjustment(p, 2000, 2010)
    0      1.266292
    1     12.662921
    2    126.629210
    >>> p = pd.DataFrame(data=[[1, 2], [3, 4]], columns=["a", "b"])
    >>> inflation_price_adjustment(p, 2000, 2010)
        a	        b
    0	1.266292	2.532584
    1	3.798876	5.065168
    >>> inflation_price_adjustment(p, 2020, 2050)
    ValueError: CPI data are only available through 2020. Your target year is 2050
    """

    base_year = int(base_year)
    target_year = int(target_year)

    cpi_data = load_cpi_data()
    if cpi_data["year"].max() < max(target_year, base_year):
        logger.info("Updating CPI data")
        cpi_data = load_cpi_data(reload_data=True, kwargs={"end_year": target_year})
        if cpi_data["year"].max() < target_year:
            raise ValueError(
                f"CPI data are only available through {cpi_data['year'].max()}. Your target year is "
                f"{target_year}"
            )
    if cpi_data["year"].min() > base_year:
        logger.info("Updating CPI data")
        cpi_data = load_cpi_data(reload_data=True, kwargs={"start_year": base_year})
        if cpi_data["year"].min() > base_year:
            raise ValueError(
                f"CPI data only start in year {cpi_data['year'].min()}. Your base year is "
                f"{base_year}"
            )
    cpi_data = cpi_data.set_index("year")
    price = price * (
        cpi_data.loc[target_year, "value"] / cpi_data.loc[base_year, "value"]
    )

    return price
