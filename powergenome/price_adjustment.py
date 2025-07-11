"""
Adjust price/cost from one year to another
"""

from pathlib import Path
from typing import Union
from warnings import warn

import numpy as np
import pandas as pd

from powergenome.financials import (
    inflation_price_adjustment as _inflation_price_adjustment,
)

#     return _get_cpi_data(start_year, end_year)


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
