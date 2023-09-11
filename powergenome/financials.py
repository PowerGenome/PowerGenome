"Functions for financial calculation of investment costs from capex and WACC"

import logging
from typing import Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ListLike = Union[list, set, pd.Series, np.array]




def investment_cost_calculator(
    capex: Union[ListLike, float],
    wacc: Union[ListLike, float],
    cap_rec_years: Union[ListLike, int],
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
