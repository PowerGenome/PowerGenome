#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Command-line interface.
"""
import cpi
import click
from dateutil.parser import parse as dateparse


@click.command()
@click.argument("value", nargs=1, required=True)
@click.argument("year_or_month", nargs=1, required=True)
@click.option(
    "--to", nargs=1, default=None, help="The year or month to adjust the value to."
)
@click.option(
    "--series_id",
    type=click.STRING,
    nargs=1,
    default=cpi.DEFAULT_SERIES_ID,
    help="The CPI data series used for the conversion. The default is the CPI-U.",
)
def inflate(value, year_or_month, to=None, series_id=cpi.DEFAULT_SERIES):
    """
    Returns a dollar value adjusted for inflation.
    """
    # Sanitize the value
    try:
        value = float(value)
    except ValueError:
        click.ClickException("Dollar value must be an integer or float.")

    # Sanitize the `from` date.
    try:
        year_or_month = _parse_date(year_or_month)
    except ValueError:
        click.ClickException(
            "Source date must be a year as an integer or a month as a parseable date string."
        )

    # Sanitize the `to` date.
    if to:
        try:
            to = _parse_date(to)
        except ValueError:
            click.ClickException(
                "Source date must be a year as an integer or a month as a parseable date string."
            )

    # Run the command
    result = cpi.inflate(value, year_or_month, to=to, series_id=series_id)

    # Print out the result to the terminal
    click.echo(result)


def _parse_date(value):
    """
    Parse a date submitted to the CLIself.

    Returns and integer if its a year. Returns a date object if its a month.
    """
    try:
        return int(value)
    except ValueError:
        return dateparse(value).date()
