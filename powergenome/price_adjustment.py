"""
Adjust price/cost from one year to another
"""

from pathlib import Path
from typing import Union
from warnings import warn

import numpy as np
import pandas as pd

from powergenome.financials import get_cpi_data as _get_cpi_data
from powergenome.financials import (
    inflation_price_adjustment as _inflation_price_adjustment,
)
from powergenome.financials import load_cpi_data as _load_cpi_data


def get_cpi_data(start_year: int = 1980, end_year: int = None) -> pd.DataFrame:
    warn(
        "The function 'get_cpi_data' has been moved to powergenome.financials -- the "
        "location in powergenome.price_adjustement will be depreciated in a future version. "
        "Update your code to use the correct import.",
        DeprecationWarning,
        stacklevel=2,
    )

    return _get_cpi_data(start_year, end_year)


def load_cpi_data(
    reload_data: bool = False, data_path: Path = None, **kwargs
) -> pd.DataFrame:
    warn(
        "The function 'load_cpi_data' has been moved to powergenome.financials -- the "
        "location in powergenome.price_adjustement will be depreciated in a future version. "
        "Update your code to use the correct import.",
        DeprecationWarning,
        stacklevel=2,
    )

    return _load_cpi_data(reload_data, data_path, **kwargs)


def inflation_price_adjustment(
    price: Union[int, float, pd.Series, pd.DataFrame, np.ndarray],
    base_year: int,
    target_year: int,
    **kwargs,
) -> float:
    warn(
        "The function 'inflation_price_adjustment' has been moved to powergenome.financials -- the "
        "location in powergenome.price_adjustement will be depreciated in a future version. "
        "Update your code to use the correct import.",
        DeprecationWarning,
        stacklevel=2,
    )

    return _inflation_price_adjustment(price, base_year, target_year, **kwargs)
