"""
Adjust price/cost from one year to another
"""

import logging

from powergenome.externals.cpi import cpi as cpi

logger = logging.getLogger(__name__)


def inflation_price_adjustment(price, base_year, target_year):

    if cpi.LATEST_YEAR < target_year:
        logger.info("Updating CPI data")
        cpi.update()
        if cpi.LATEST_YEAR < target_year:
            raise ValueError(
                f"CPI data are only available through {cpi.LATEST_YEAR}. Your target year is "
                f"{target_year}"
            )

    price = price * cpi.inflate(1, base_year, to=target_year)

    return price
