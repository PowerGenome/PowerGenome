"""
Load data from EIA's bulk data
"""

import logging
import operator
import zipfile
from itertools import product
from pathlib import Path
from typing import Union

import pandas as pd

from powergenome.database import get_data
from powergenome.params import DATA_PATHS
from powergenome.price_adjustment import inflation_price_adjustment
from powergenome.util import download_save, reverse_dict_of_lists

logger = logging.getLogger(__name__)

numeric = Union[int, float]


def download_bulk_file(fn: str, retry: bool = False):
    """Download and unzip a bulk data file from EIA

    Parameters
    ----------
    fn : str
        Name of the bulk data file
    retry : bool
        If True, redo the download/save
    """
    if ".zip" not in fn:
        fn = f"{fn}.zip"
    url = f"https://api.eia.gov/bulk/{fn}"

    save_path = DATA_PATHS["eia"] / "bulk_files"
    if not (save_path / fn).exists():
        download_save(url, save_path / fn)
    elif retry:
        download_save(url, save_path / fn)


def extract_bulk_series(aeo_year: int, name_str: str, columns: list = None):
    """Extract relevant data series from an AEO bulk data file.

    Parameters
    ----------
    aeo_year : int
        Year of AEO data
    name_str : str
        String match for something in the "name" field of the JSON
    columns : list, optional
        Names for columns in the dataframe written to file, by default None
    """
    bulk_data_dir = DATA_PATHS["eia"] / "bulk_files"
    save_data_dir = DATA_PATHS["eia"] / "open_data"
    save_data_dir.mkdir(exist_ok=True)
    fn = f"AEO{aeo_year}.zip"
    df = pd.read_json(bulk_data_dir / fn, lines=True)
    df = df.dropna(subset=["series_id"])
    filtered_df = df.query("name.str.contains(@name_str)")
    for row in filtered_df.itertuples():
        series = row.series_id
        _df = pd.DataFrame(row.data, columns=columns)
        _df.to_csv(save_data_dir / f"{series}.csv", index=False, float_format="%g")


def load_aeo_series(series_id: str) -> pd.DataFrame:
    """Load EIA AEO data either from file (if it exists) from a bulk download file or
    from the API.

    Parameters
    ----------
    series_id : str
        The AEO API series ID that uniquely identifies the data request.

    Returns
    -------
    pd.DataFrame
        Data from EIA's AEO via their open data API.
    """
    data_dir = DATA_PATHS["eia"] / "open_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    if not (data_dir / f"{series_id}.csv").exists():
        try:
            year = series_id.split(".")[1]
            fn = f"AEO{year}.zip"
            download_bulk_file(fn)
            extract_bulk_series(
                aeo_year=year, name_str="Electricity Demand", columns=["year", "demand"]
            )
            extract_bulk_series(
                aeo_year=year,
                name_str="Energy Prices : Electric Power",
                columns=["year", "price"],
            )
        except (FileNotFoundError, zipfile.BadZipFile):
            download_bulk_file(fn, retry=True)
            extract_bulk_series(
                aeo_year=year,
                name_str="Electricity Demand",
                columns=["year", "demand"],
            )
            extract_bulk_series(
                aeo_year=year,
                name_str="Energy Prices : Electric Power",
                columns=["year", "price"],
            )

    df = pd.read_csv(data_dir / f"{series_id}.csv")

    return df


def get_aeo_load(
    region: str,
    aeo_year: Union[str, numeric],
    scenario_series: str,
    sector: str = "ELEP",
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
    SERIES_ID = (
        f"AEO.{aeo_year}.{scenario_series}.CNSM_NA_{sector}_NA_ELC_NA_{region}_BLNKWH.A"
    )

    df = load_aeo_series(SERIES_ID)
    df["year"] = df["year"].astype(int)

    return df
