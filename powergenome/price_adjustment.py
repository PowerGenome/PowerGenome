"""
Adjust price/cost from one year to another
"""

import logging

import requests
import json
from typing import NamedTuple
from datetime import date
import pandas as pd
from pathlib import Path
from powergenome.params import DATA_PATHS


class MonthlyCPI(NamedTuple):
    year: int
    period: int
    value: float


logger = logging.getLogger(__name__)


def get_cpi_data(start_year: int = 1980, end_year: int = None) -> pd.DataFrame:
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
            "https://api.bls.gov/publicAPI/v2/timeseries/data/",
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
        start_year = e_y + 1
        e_y = start_year + 10

    annual_cpi = pd.concat(df_list)

    return annual_cpi


def load_cpi_data(reload_data: bool = False) -> pd.DataFrame:

    if reload_data or not DATA_PATHS["cpi_data"].exists():
        DATA_PATHS["cpi_data"].parent.mkdir(exist_ok=True)
        cpi_data = get_cpi_data()
        cpi_data.to_clipboard(DATA_PATHS["cpi_data"], index=False)
    else:
        cpi_data = pd.read_csv(DATA_PATHS["cpi_data"])

    return cpi_data


def inflation_price_adjustment(price: float, base_year: int, target_year: int) -> float:
    base_year = int(base_year)
    target_year = int(target_year)

    cpi_data = load_cpi_data()
    if cpi_data["year"].max() < target_year:
        logger.info("Updating CPI data")
        cpi_data = load_cpi_data(reload_data=True)
        if cpi_data["year"].max() < target_year:
            raise ValueError(
                f"CPI data are only available through {cpi_data['year'].max()}. Your target year is "
                f"{target_year}"
            )
    cpi_data = cpi_data.set_index("year")
    price = price * (
        cpi_data.loc[target_year, "value"] / cpi_data.loc[base_year, "value"]
    )

    return price
