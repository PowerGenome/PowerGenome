#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quickly adjust U.S. dollars for inflation using the Consumer Price Index (CPI)
"""
import numbers
import warnings
from datetime import date, datetime

from . import parsers
from .download import Downloader
from .errors import StaleDataWarning
from .defaults import DEFAULT_SERIES_ID, DEFAULTS_SERIES_ATTRS

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# Parse data for use
logger.info("Parsing data files from the BLS")
areas = parsers.ParseArea().parse()
items = parsers.ParseItem().parse()
periods = parsers.ParsePeriod().parse()
periodicities = parsers.ParsePeriodicity().parse()
series = parsers.ParseSeries(
    periods=periods, periodicities=periodicities, areas=areas, items=items
).parse()

# set the default series to the CPI-U
DEFAULT_SERIES = series.get_by_id(DEFAULT_SERIES_ID)

# Establish the range of data available
LATEST_MONTH = DEFAULT_SERIES.latest_month
LATEST_YEAR = DEFAULT_SERIES.latest_year

# Figure out how out of date you are
DAYS_SINCE_LATEST_MONTH = (date.today() - LATEST_MONTH).days
DAYS_SINCE_LATEST_YEAR = (date.today() - date(LATEST_YEAR, 1, 1)).days

# If it's more than two and a half years out of date, raise a warning.
if DAYS_SINCE_LATEST_YEAR > (365 * 2.25) or DAYS_SINCE_LATEST_MONTH > 90:
    warnings.warn(StaleDataWarning())
    logger.warn(
        "CPI data is out of date. To accurately inflate to today's dollars, you must run `cpi.update()`."
    )


def get(
    year_or_month,
    survey=DEFAULTS_SERIES_ATTRS["survey"],
    seasonally_adjusted=DEFAULTS_SERIES_ATTRS["seasonally_adjusted"],
    periodicity=DEFAULTS_SERIES_ATTRS["periodicity"],
    area=DEFAULTS_SERIES_ATTRS["area"],
    items=DEFAULTS_SERIES_ATTRS["items"],
    series_id=None,
):
    """
    Returns the CPI value for a given year.
    """
    # Pull the series
    if series_id:
        # If the user has provided an explicit series id, we are going to ignore the humanized options.
        series_obj = series.get_by_id(series_id)
    else:
        # Otherwise, we build the series id using the more humanized options
        series_obj = series.get(survey, seasonally_adjusted, periodicity, area, items)

    # Prep the lookup value depending on the input type.
    if isinstance(year_or_month, numbers.Integral):
        year_or_month = date(year_or_month, 1, 1)
        period_type = "annual"
    elif isinstance(year_or_month, date):
        period_type = "monthly"
        # If it's not set to the first day of the month, we should do that now.
        if year_or_month.day != 1:
            year_or_month = year_or_month.replace(day=1)
    else:
        raise ValueError("Only integers and date objects are accepted.")

    # Pull the value from the series by date
    return series_obj.get_index_by_date(year_or_month, period_type=period_type).value


def inflate(
    value,
    year_or_month,
    to=None,
    survey=DEFAULTS_SERIES_ATTRS["survey"],
    seasonally_adjusted=DEFAULTS_SERIES_ATTRS["seasonally_adjusted"],
    periodicity=DEFAULTS_SERIES_ATTRS["periodicity"],
    area=DEFAULTS_SERIES_ATTRS["area"],
    items=DEFAULTS_SERIES_ATTRS["items"],
    series_id=None,
):
    """
    Returns a dollar value adjusted for inflation.

    You must submit the value, followed by the year or month its from.

    Years should be submitted as integers. Months as datetime.date objects.

    By default, the input is adjusted to the most recent year or month available from the CPI.

    If you'd like to adjust to a different year or month, submit it to the optional `to` keyword argument.

    Yearly data can only be updated to other years. Monthly data can only be updated to other months.
    """
    # If the two dates match, just return the value unadjusted
    if year_or_month == to:
        return value

    # Figure out the 'to' date if it has not been provided
    if not to:
        if isinstance(year_or_month, (date, datetime)):
            to = LATEST_MONTH
        else:
            to = LATEST_YEAR
    # Otherwise sanitize it
    else:
        if isinstance(to, numbers.Integral):
            to = int(to)
        elif isinstance(to, datetime):
            # We want dates not datetimes
            to = to.date()

    # Sanitize the year_or_month
    if isinstance(year_or_month, numbers.Integral):
        # We need to make sure that int64, int32 and other int-like objects
        # are the same type for the comparison to come.
        year_or_month = int(year_or_month)
    # If a datetime has been provided, shave it down to a date.
    elif isinstance(year_or_month, datetime):
        year_or_month = year_or_month.date()

    # Make sure the two dates are the same type
    if type(year_or_month) != type(to):
        raise TypeError(
            "Years can only be converted to other years. Months only to other months."
        )

    # Otherwise, let's do the math.
    # The input value is multiplied by the CPI of the target year,
    # then divided into the CPI from the source year.
    kwargs = {
        "survey": survey,
        "seasonally_adjusted": seasonally_adjusted,
        "periodicity": periodicity,
        "area": area,
        "items": items,
        "series_id": series_id,
    }
    source_index = get(year_or_month, **kwargs)
    target_index = get(to, **kwargs)
    return (value * target_index) / float(source_index)


def update():
    """
    Updates the Consumer Price Index dataset at the core of this library.

    Requires an Internet connection.
    """
    Downloader().update()
