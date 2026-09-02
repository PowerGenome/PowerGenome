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

# Track whether we've already emitted the deprecation warning so the test suite
# isn't flooded with identical messages.
_inflation_price_adjustment_warned = False

#     return _get_cpi_data(start_year, end_year)


def inflation_price_adjustment(
    price: Union[int, float, pd.Series, pd.DataFrame, np.ndarray],
    base_year: int,
    target_year: int,
    **kwargs,
) -> float:
    global _inflation_price_adjustment_warned
    if not _inflation_price_adjustment_warned:
        warn(
            "The function 'inflation_price_adjustment' has been moved to powergenome.financials. "
            "The location in powergenome.price_adjustment is deprecated and will be removed in a future version. "
            "Update imports to 'from powergenome.financials import inflation_price_adjustment'.",
            DeprecationWarning,
            stacklevel=2,
        )
        _inflation_price_adjustment_warned = True

    return _inflation_price_adjustment(price, base_year, target_year, **kwargs)
