"Functions for financial calculation of investment costs from capex and WACC"

import json
import logging
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple, Union

import numpy as np
import pandas as pd
import requests

from powergenome.params import DATA_PATHS
from powergenome.util import load_data

logger = logging.getLogger(__name__)

ListLike = Union[list, set, pd.Series, np.array]


class MonthlyCPI(NamedTuple):
    year: int
    period: int
    value: float


def investment_cost_calculator(
    capex: Union[list, pd.Series, np.array, float],
    wacc: Union[list, pd.Series, np.array, float],
    cap_rec_years: Union[list, pd.Series, np.array, int],
    compound_method: str = "discrete",
) -> np.array:
    """Calculate annualized investment cost using either discrete or continuous compounding.

    Parameters
    ----------
    capex : Union[LIST_LIKE, float]
        Single or list-like capital costs for one or more resources
    wacc : Union[LIST_LIKE, float]
        Weighted average cost of capital. Can be a single value or one value for each resource.
        Should be the same length as capex or a single value.
    cap_rec_years : Union[LIST_LIKE, int]
        Capital recovery years or the financial lifetime of each asset. Should be the same
        length as capex or a single value.
    compound_method : str, optional
        The method to compound interest. Either "discrete" or "continuous", by default
        "discrete"

    Returns
    -------
    np.array
        An annual investment cost for each capital cost

    Raises
    ------
    TypeError
        A list-like type of WACC or capital recovery years was provided for only a single
        capex
    ValueError
        The capex and WACC or capital recovery years are both list-like but not the same
        length
    ValueError
        One of the inputs contains a nan value
    ValueError
        The compounding_method argument must be either "discrete" or "continuous"
    """
    # Data checks
    for var, name in zip([wacc, cap_rec_years], ["wacc", "capital recovery years"]):
        if np.isscalar(capex):
            if not np.isscalar(var):
                raise TypeError(
                    f"Multiple {name} values were provided for only a single resource capex "
                    "when calculating annualized inventment costs. Only a single value "
                    "should be provided with only a single resource capex."
                )
        else:
            if not np.isscalar(var) and len(var) != len(capex):
                raise ValueError(
                    f"The number of {name} values ({len(var)}) and the number of resource "
                    f"capex values ({len(capex)}) should be the same but they are not."
                )

    # Convert everything to arrays and do the calculations.
    vars = [capex, wacc, cap_rec_years]
    dtypes = [float, float, int]
    for idx, (var, dtype) in enumerate(zip(vars, dtypes)):
        vars[idx] = np.asarray(var, dtype=dtype)
    capex, wacc, cap_rec_years = vars
    # capex = np.asarray(capex, dtype=float)
    # wacc = np.asarray(wacc, dtype=float)
    # cap_rec_years = np.asarray(cap_rec_years, dtype=int)

    for var, name in zip(
        [capex, wacc, cap_rec_years], ["capex", "wacc", "capital recovery years"]
    ):
        if np.isnan(var).any() or pd.isnull(var).any():
            raise ValueError(f"Investment variable {name} costs contains nan values")

    if compound_method.lower() == "discrete":
        inv_cost = _discrete_inv_cost_calc(
            capex=capex, wacc=wacc, cap_rec_years=cap_rec_years
        )
    elif "cont" in compound_method.lower():
        inv_cost = _continuous_inv_cost_calc(
            capex=capex, wacc=wacc, cap_rec_years=cap_rec_years
        )
    else:
        raise ValueError(
            f"'{compound_method}' is not a valid compounding method for converting capex "
            "into annual investment costs. Valid methods are 'discrete' or 'continuous'."
        )

    return inv_cost


def _continuous_inv_cost_calc(
    capex: Union[np.array, float],
    wacc: Union[np.array, float],
    cap_rec_years: Union[np.array, int],
) -> np.array:
    """Calculate annualized investment cost using continuous compounding.

    Parameters
    ----------
    capex : Union[LIST_LIKE, float]
        Single or list-like capital costs for one or more resources
    wacc : Union[LIST_LIKE, float]
        Weighted average cost of capital. Can be a single value or one value for each resource.
        Should be the same length as capex or a single value.
    cap_rec_years : Union[LIST_LIKE, int]
        Capital recovery years or the financial lifetime of each asset. Should be the same
        length as capex or a single value.

    Returns
    -------
    np.array
        An annual investment cost for each capital cost
    """
    inv_cost = capex * (
        np.exp(wacc * cap_rec_years)
        * (np.exp(wacc) - 1)
        / (np.exp(wacc * cap_rec_years) - 1)
    )

    return inv_cost


def _discrete_inv_cost_calc(
    capex: Union[np.array, float],
    wacc: Union[np.array, float],
    cap_rec_years: Union[np.array, int],
) -> np.array:
    """Calculate annualized investment cost using discrete compounding.

    Parameters
    ----------
    capex : Union[LIST_LIKE, float]
        Single or list-like capital costs for one or more resources
    wacc : Union[LIST_LIKE, float]
        Weighted average cost of capital. Can be a single value or one value for each resource.
        Should be the same length as capex or a single value.
    cap_rec_years : Union[LIST_LIKE, int]
        Capital recovery years or the financial lifetime of each asset. Should be the same
        length as capex or a single value.

    Returns
    -------
    np.array
        An annual investment cost for each capital cost
    """
    inv_cost = capex * wacc / (1 - (1 + wacc) ** -cap_rec_years)

    return inv_cost


def load_dollar_year_data(data_location: Path | str, table_name: str) -> pd.DataFrame:
    """
    Load dollar year data from a specified data location and table.

    This function retrieves a DataFrame containing dollar year information from the
    given data location and table name. It validates that the DataFrame is not empty and
    contains the required 'year' and 'value' columns.

    Parameters
    ----------
    data_location : Path | str
        The path to the data file or directory containing the table. Can be a folder
        or the path to a database.
    table_name : str
        The name of the table to load from the data location. Either the name of a CSV
        or parquet file, or the name of a table in a database.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the dollar year data with 'year' and 'value' columns.

    Raises
    ------
    ValueError
        If the loaded table is empty or does not contain the required columns.
    """

    df = load_data(data_location=data_location, file_or_table_name=table_name)
    if df.empty:
        raise ValueError(
            f"The dollar year table {table_name} is empty. "
            "Please check the data file."
        )
    if "year" not in df.columns or "value" not in df.columns:
        raise ValueError(
            "The dollar year table must contain 'year' and 'value' columns."
        )
    return df


def inflation_price_adjustment(
    price: Union[int, float, pd.Series, pd.DataFrame, np.ndarray],
    base_year: int,
    target_year: int,
    **kwargs,
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
    **kwargs : optional
        Optional keyword arguments for the function `load_cpi_data`.

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
    data_location = kwargs.pop("data_location", None)
    table_name = kwargs.pop("table_name")

    dollar_year_df = load_dollar_year_data(
        data_location=data_location, table_name=table_name
    )

    min_year = dollar_year_df["year"].min()
    max_year = dollar_year_df["year"].max()
    if any(y < min_year or y > max_year for y in [base_year, target_year]):
        raise ValueError(
            f"Dollar-year data are only available from {min_year} to {max_year}. "
            f"Your base year is {base_year} and target year is {target_year}."
        )

    dollar_year_df = interpolate_values(
        dollar_year_df, base_year, target_year
    ).set_index("year")

    price = price * (
        dollar_year_df.loc[target_year, "value"]
        / dollar_year_df.loc[base_year, "value"]
    )

    return price


def interpolate_values(df: pd.DataFrame, *target_years: Any) -> pd.DataFrame:
    """
    Interpolates 'value' at one or more target_years using linear interpolation.
    Adds any missing year(s) into the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns 'year' (numeric) and 'value'.
    *target_years : int, float, or list/tuple of them
        One or more years at which to interpolate the value.

    Returns
    -------
    pd.DataFrame
        Original rows plus any new interpolated rows, sorted by 'year'.
    """
    # 1) sort
    df = df.sort_values("year").reset_index(drop=True)
    min_year, max_year = df["year"].min(), df["year"].max()

    # 2) flatten args into a single list of years
    if not target_years:
        raise ValueError("At least one target year must be provided")
    years = []
    for y in target_years:
        if isinstance(y, (list, tuple)):
            years.extend(y)
        else:
            years.append(y)

    # 3) build rows for any missing years
    new_rows = []
    for year in years:
        if year in df["year"].values:
            continue
        if year < min_year or year > max_year:
            raise ValueError(f"Year {year} is outside range {min_year}–{max_year}")
        val = float(np.interp(year, df["year"], df["value"]))
        new_rows.append({"year": year, "value": val})

    # 4) append + re-sort
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        df = df.sort_values("year").reset_index(drop=True)

    return df
