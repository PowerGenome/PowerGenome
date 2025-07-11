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

from powergenome.params import DATA_PATHS
from powergenome.price_adjustment import inflation_price_adjustment
from powergenome.util import download_save, load_data, reverse_dict_of_lists

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


def fetch_fuel_prices(
    data_location: Path, table_name: str, settings: dict, inflate_price: bool = True
) -> pd.DataFrame:
    """
    Get fuel prices for all regions, fuel types, and scenarios (series IDs)
    included in the settings. Combine the region, scenario, and fuel name into a full
    fuel name column. Optionally, adjust the prices to the target dollar year.

    Parameters
    ----------
    data_location : Path
        Path to the directory containing the saved fuel price data table.
    table_name : str
        Name of the CSV file or table containing the fuel price data.
    settings : dict
        Should include the following keys:
            fuel_data_year (int)
            fuel_series_region_names (dict)
            fuel_series_names (dict)
            fuel_series_scenario_names (dict)
    inflate_price : bool, optional
        If True, adjust the fuel prices to the year "target_usd_year" from the settings.
        If False, do not adjust the prices. Requires the column "dollar_year" in the
        fuel price data table.

    Returns
    -------
    pd.DataFrame
        All fuel price data for the specified data year, with columns:
        ['year', 'price', 'fuel', 'region', 'scenario', 'full_fuel_name'].

    Raises
    ------
    KeyError
        If 'fuel_data_year' is missing from settings and the data file contains multiple data years.
        If the specified 'fuel_data_year' is not found in the 'data_year' column of the data file.
    FileNotFoundError
        If the specified data file is empty.
    """

    data_year = settings.get("fuel_data_year")

    all_fuel_data = load_data(data_location, table_name)
    if all_fuel_data.empty:
        raise FileNotFoundError(
            f"The file {data_location / table_name} does is empty. "
            "Please check the data location and table name."
        )
    if data_year is not None and "data_year" in all_fuel_data.columns:
        if data_year not in all_fuel_data["data_year"].unique():
            raise KeyError(
                f"The data year specified in parameter 'fuel_data_year' ({data_year}) is "
                "not in the 'data_year' column of your fuel price data table. "
                "Please check the data file and the settings parameter."
            )
        fuel_data = all_fuel_data.query("data_year == @data_year")
    elif (
        data_year is None
        and "year" in all_fuel_data.columns
        and all_fuel_data["data_year"].nunique() > 1
    ):
        raise KeyError(
            "The parameter 'fuel_data_year' is not in your settings files and the fuel price "
            "data table contains multiple data years."
        )
    else:
        fuel_data = all_fuel_data.copy()

    if settings.get("fuel_series_names"):
        fuel_data["fuel"] = (
            fuel_data["fuel"]
            .str.lower()
            .map({v.lower(): k for k, v in settings["fuel_series_names"].items()})
        )
    if settings.get("fuel_series_scenario_names"):
        fuel_data["scenario"] = (
            fuel_data["scenario"]
            .str.lower()
            .map(
                {
                    v.lower(): k
                    for k, v in settings["fuel_series_scenario_names"].items()
                }
            )
        )
    if settings.get("fuel_series_region_names"):
        fuel_data["region"] = (
            fuel_data["region"]
            .str.lower()
            .map(
                {v.lower(): k for k, v in settings["fuel_series_region_names"].items()}
            )
        )
    fuel_data["full_fuel_name"] = (
        fuel_data["region"] + "_" + fuel_data["scenario"] + "_" + fuel_data["fuel"]
    )
    fuel_data = fuel_data.dropna(subset=["full_fuel_name"])

    if inflate_price:
        try:
            df_list = []
            for year, _df in fuel_data.groupby("dollar_year"):
                _df.loc[:, "price"] = inflation_price_adjustment(
                    price=_df.loc[:, "price"],
                    base_year=year,
                    target_year=settings["target_usd_year"],
                )
                df_list.append(_df)
            fuel_data = pd.concat(df_list, ignore_index=True, sort=False)
        except (KeyError, TypeError):
            logger.warning(
                """
    ************
    Unable to inflate fuel prices. Check your settings file to ensure the keys
    "target_usd_year" and "aeo_fuel_usd_year" are valid integers.
    ************
                """
            )

    return fuel_data


def modify_fuel_prices(
    prices: pd.DataFrame,
    fuel_region_map: dict,
    regional_fuel_adjustments: dict = None,
) -> pd.DataFrame:
    """Modify the AEO fuel prices by model region or fuel within a model region.

    Parameters
    ----------
    prices : pd.DataFrame
        Fuel prices from AEO, with columns ['year', 'price', 'fuel', 'region', 'scenario',
        'full_fuel_name']
    fuel_region_map : dict
        Mapping of AEO census division fuel names to lists of model regions
    regional_fuel_adjustments : dict, optional
        Modifications of fuel prices by region or fuel within region, by default None

    Returns
    -------
    pd.DataFrame
        Full input dataframe with modified copies for model regions and fuels specified
        in `regional_fuel_adjustments`.

    Raises
    ------
    KeyError
        The required parameter 'fuel_region_map' is missing
    KeyError
        One or more model regions having fuel prices modified is not in `fuel_region_map`
    KeyError
        Invalid operator type
    KeyError
        Invalid fuel name
    KeyError
        Invalid operator type
    TypeError
        Fuel price modifiers are not a list or a dictionary of lists
    """

    if not regional_fuel_adjustments:
        return prices

    if not fuel_region_map:
        raise KeyError("The required parameter 'fuel_region_map' is missing.")

    allowed_operators = ["add", "mul", "truediv", "sub"]
    model_regions = list(regional_fuel_adjustments)
    model_aeo_region_map = reverse_dict_of_lists(fuel_region_map)
    if not all(r in model_aeo_region_map for r in model_regions):
        raise KeyError(
            "All model regions listed in the settings parameter 'regional_fuel_adjustments' "
            "should also be included in `fuel_region_map`. One or more regions was "
            "not found."
        )

    df_list = []
    for region, adj in regional_fuel_adjustments.items():
        aeo_region = model_aeo_region_map[region]
        if isinstance(adj, list):
            op, op_value = adj
            if op not in allowed_operators:
                raise KeyError(
                    f"The regional fuel price adjustment for {region} needs a valid "
                    f"operator from the list\n{allowed_operators}\n"
                    "in the format [<operator>, <value>].\n"
                )
            f = operator.attrgetter(op)
            df = prices.loc[prices["region"] == aeo_region, :]
            df.loc[:, "region"] = region
            df.loc[:, "price"] = f(operator)(df["price"], op_value)
            df.loc[:, "full_fuel_name"] = df["full_fuel_name"].str.replace(
                aeo_region, region
            )
            df_list.append(df)
        elif isinstance(adj, dict):
            for fuel, op_list in adj.items():
                if fuel not in prices["fuel"].unique():
                    raise KeyError(
                        f"The fuel '{fuel}' is listed under the region {region} in your settings "
                        "parameter 'regional_fuel_adjustments'. There was no AEO fuel "
                        "price fetched for this fuel so it cannot be modified."
                    )
                op, op_value = op_list
                if op not in allowed_operators:
                    raise KeyError(
                        f"The regional fuel price adjustment for '{fuel}' in {region} "
                        f"needs to be an operator from the list {allowed_operators}. "
                        f"You supplied '{op}', which is not a valid operator."
                    )
                f = operator.attrgetter(op)
                df = prices.loc[
                    (prices["region"] == aeo_region) & (prices["fuel"] == fuel.lower()),
                    :,
                ]
                df.loc[:, "region"] = region
                df.loc[:, "price"] = f(operator)(df["price"], op_value)
                df.loc[:, "full_fuel_name"] = df["full_fuel_name"].str.replace(
                    aeo_region, region
                )
                df_list.append(df)
        else:
            raise TypeError(
                "Fuel price modifiers in the settings parameter 'regional_fuel_adjustments' "
                "must be a list of the form '[<op>, <value>]', or a similar list for a "
                "specific fuel. "
                f"Your value look like '{adj}' for region '{region}'."
            )

    mod_prices = pd.concat([prices] + df_list, ignore_index=True, sort=False)

    return mod_prices


def add_user_fuel_prices(settings: dict, df: pd.DataFrame = None) -> pd.DataFrame:
    """Add user fuel prices to a dataframe of user prices from AEO (or elsewhere)

    Parameters
    ----------
    settings : dict
        If adding user prices, should have the key "user_fuel_price" with value of a
        dictionary matching user fuel names and prices. Prices can either be a single
        price for all regions or a price per region. For example this shows biomass with
        different prices in two regions and ZCF with the same price in all regions:

        settings["user_fuel_price"] = {
            "biomass": {"SC_VACA": 10, "PJM_DOM": 5},
            "ZCF": 15
        }

        If the keys "target_usd_year" and "user_fuel_usd_year" are also included, fuel
        prices will be corrected to the correct USD year. "user_fuel_usd_year" should
        be a dictionary with fuel name: USD year pairings. Only fuels included in this
        dictionary will have their prices changed to the target USD year.
    df : pd.DataFrame, optional
        A dataframe with fuel prices from AEO (or elsewhere), by default None. Should
        have columns ["year", "price", "fuel", "region", "scenario", "full_fuel_name"]

    Returns
    -------
    pd.DataFrame
        The combined dataframes of user prices and the other price dataframe provided
        as input. Columns are ["year", "price", "fuel", "region", "scenario", "full_fuel_name"].
    """

    if not settings.get("user_fuel_price"):
        if df is not None:
            return df
    cols = ["year", "price", "fuel", "region", "scenario", "full_fuel_name"]
    if df is not None and not df.empty:
        years = df["year"].unique()
    else:
        years = range(2020, 2051)
    fuel_data = {c: [] for c in cols}

    for fuel, val in settings["user_fuel_price"].items():
        if isinstance(val, dict):
            for region, price in val.items():
                fuel_name = f"{region}_{fuel}"
                fuel_data["year"].extend(years)
                fuel_data["price"].extend([price] * len(years))
                fuel_data["fuel"].extend([fuel] * len(years))
                fuel_data["region"].extend([region] * len(years))
                fuel_data["scenario"].extend(["user"] * len(years))
                fuel_data["full_fuel_name"].extend([fuel_name] * len(years))
        else:
            fuel_data["year"].extend(years)
            fuel_data["price"].extend([val] * len(years))
            fuel_data["fuel"].extend([fuel] * len(years))
            fuel_data["region"].extend([""] * len(years))
            fuel_data["scenario"].extend(["user"] * len(years))
            fuel_data["full_fuel_name"].extend([fuel] * len(years))

    user_fuel_price = pd.DataFrame(fuel_data)
    if settings.get("target_usd_year"):
        for fuel, year in (settings.get("user_fuel_usd_year", {}) or {}).items():
            user_fuel_price.loc[user_fuel_price["fuel"] == fuel, "price"] = (
                inflation_price_adjustment(
                    user_fuel_price.loc[user_fuel_price["fuel"] == fuel, "price"],
                    year,
                    settings["target_usd_year"],
                )
            )
    if df is not None:
        user_fuel_price = pd.concat([df, user_fuel_price])
    return user_fuel_price


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
