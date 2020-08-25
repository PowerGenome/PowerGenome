#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python objects for modeling Consumer Price Index (CPI) data structures.
"""
import collections
from datetime import date
from pandas.io.json import json_normalize

# CPI tools
from .errors import CPIObjectDoesNotExist
from .defaults import DEFAULTS_SERIES_ATTRS

# Logging
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class MappingList(list):
    """
    A custom list that allows for lookups by attribute.
    """

    def __init__(self):
        self._id_dict = {}
        self._name_dict = {}

    def get_by_id(self, value):
        try:
            return self._id_dict[value]
        except KeyError:
            raise CPIObjectDoesNotExist(
                "Object with id {} could not be found".format(value)
            )

    def get_by_name(self, value):
        try:
            return self._name_dict[value]
        except KeyError:
            raise CPIObjectDoesNotExist(
                "Object with id {} could not be found".format(value)
            )

    def append(self, item):
        """
        Override to default append method that allows dictionary-style lookups
        """
        # Add to dictionary lookup
        self._id_dict[item.id] = item
        self._name_dict[item.name] = item

        # Append to list
        super(MappingList, self).append(item)


class SeriesList(list):
    """
    A custom list of indexes in a series.
    """

    SURVEYS = {
        "All urban consumers": "CU",
        "Urban wage earners and clerical workers": "CW",
    }
    SEASONALITIES = {True: "S", False: "U"}

    def __init__(self, periodicities, areas, items):
        self.periodicities = periodicities
        self.areas = areas
        self.items = items
        self._dict = {}

    def to_dataframe(self):
        """
        Returns the list as a pandas DataFrame.
        """
        dict_list = [obj.__dict__() for obj in self]
        return json_normalize(dict_list, sep="_")

    def append(self, item):
        """
        Override to default append method that allows validation and dictionary-style lookups
        """
        # Valid item type
        if not isinstance(item, Series):
            raise TypeError("Only Series objects can be added to this list.")

        # Add to dictionary lookup
        self._dict[item.id] = item

        # Append to list
        super(SeriesList, self).append(item)

    def get_by_id(self, value):
        """
        Returns the CPI series object with the provided identifier code.
        """
        logger.debug("Retrieving series with id {}".format(value))
        try:
            return self._dict[value]
        except KeyError:
            raise CPIObjectDoesNotExist(
                "Object with id {} could not be found".format(value)
            )

    def get(
        self,
        survey=DEFAULTS_SERIES_ATTRS["survey"],
        seasonally_adjusted=DEFAULTS_SERIES_ATTRS["seasonally_adjusted"],
        periodicity=DEFAULTS_SERIES_ATTRS["periodicity"],
        area=DEFAULTS_SERIES_ATTRS["area"],
        items=DEFAULTS_SERIES_ATTRS["items"],
    ):
        """
        Returns a single CPI Series object based on the input.

        The default series is returned if not configuration is made to the keyword arguments.
        """
        # Get all the codes for these humanized input.
        try:
            survey_code = self.SURVEYS[survey]
        except KeyError:
            raise CPIObjectDoesNotExist(
                "Survey with the name {} does not exist".format(survey)
            )

        try:
            seasonality_code = self.SEASONALITIES[seasonally_adjusted]
        except KeyError:
            raise CPIObjectDoesNotExist(
                "Seasonality {} does not exist".format(seasonally_adjusted)
            )

        # Generate the series id
        series_id = "{}{}{}{}{}".format(
            survey_code,
            seasonality_code,
            self.periodicities.get_by_name(periodicity).code,
            self.areas.get_by_name(area).code,
            self.items.get_by_name(items).code,
        )

        # Pull the series
        return self.get_by_id(series_id)


class BaseObject(object):
    """
    An abstract base class for all the models.
    """

    def __repr__(self):
        return "<{}: {}>".format(self.__class__.__name__, self.__str__())

    def __eq__(self, other):
        return self.id == other.id

    def __str__(self):
        return self.name


class Area(BaseObject):
    """
    A geographical area where prices are gathered monthly.
    """

    def __init__(self, code, name):
        self.id = code
        self.code = code
        self.name = name

    def __dict__(self):
        return {"id": self.id, "code": self.code, "name": self.name}


class Item(BaseObject):
    """
    A consumer item that has its price tracked.
    """

    def __init__(self, code, name):
        self.id = code
        self.code = code
        self.name = name

    def __dict__(self):
        return {"id": self.id, "code": self.code, "name": self.name}


class Period(BaseObject):
    """
    A time period tracked by the CPI.
    """

    def __init__(self, code, abbreviation, name):
        self.id = code
        self.code = code
        self.abbreviation = abbreviation
        self.name = name

    def __dict__(self):
        return {
            "id": self.id,
            "code": self.code,
            "abbreviation": self.abbreviation,
            "name": self.name,
            "month": self.month,
            "type": self.type,
        }

    @property
    def month(self):
        """
        Returns the month integer for the period.
        """
        if self.id in ["M13", "S01", "S03"]:
            return 1
        elif self.id == "S02":
            return 7
        else:
            return int(self.id.replace("M", ""))

    @property
    def type(self):
        """
        Returns a string classifying the period.
        """
        if self.id in ["M13", "S03"]:
            return "annual"
        elif self.id in ["S01", "S02"]:
            return "semiannual"
        else:
            return "monthly"


class Periodicity(BaseObject):
    """
    A time interval tracked by the CPI.
    """

    def __init__(self, code, name):
        self.id = code
        self.code = code
        self.name = name

    def __dict__(self):
        return {"id": self.id, "code": self.code, "name": self.name}


class Series(BaseObject):
    """
    A set of CPI data observed over an extended period of time over consistent time intervals ranging from
    a specific consumer item in a specific geographical area whose price is gathered monthly to a category
    of worker in a specific industry whose employment rate is being recorded monthly, etc.

    Yes, that's the offical government definition. I'm not kidding.
    """

    def __init__(
        self, id, title, survey, seasonally_adjusted, periodicity, area, items
    ):
        self.id = id
        self.title = title
        self.survey = survey
        self.seasonally_adjusted = seasonally_adjusted
        self.periodicity = periodicity
        self.area = area
        self.items = items
        self._indexes = {
            "annual": collections.OrderedDict(),
            "monthly": collections.OrderedDict(),
            "semiannual": collections.OrderedDict(),
        }

    def __str__(self):
        return "{}: {}".format(self.id, self.title)

    def __dict__(self):
        return {
            "id": self.id,
            "title": self.title,
            "survey": self.survey,
            "seasonally_adjusted": self.seasonally_adjusted,
            "periodicity": self.periodicity.__dict__(),
            "area": self.area.__dict__(),
            "items": self.items.__dict__(),
        }

    def to_dataframe(self):
        """
        Returns this series and all its indexes as a pandas DataFrame.
        """
        dict_list = [obj.__dict__() for obj in self.indexes]
        return json_normalize(dict_list, sep="_")

    @property
    def indexes(self):
        flat = []
        for l in self._indexes.values():
            flat.extend(l.values())
        return flat

    @property
    def latest_month(self):
        if not self._indexes["monthly"]:
            return None
        return max([i.date for i in self._indexes["monthly"].values()])

    @property
    def latest_year(self):
        if not self._indexes["annual"]:
            return None
        return max([i.year for i in self._indexes["annual"].values()])

    def get_index_by_date(self, date, period_type="annual"):
        try:
            return self._indexes[period_type][date]
        except KeyError:
            raise CPIObjectDoesNotExist(
                "Index of {} type for {} does not exist".format(period_type, date)
            )


class Index(BaseObject):
    """
    A Consumer Price Index value generated by the Bureau of Labor Statistics.
    """

    def __init__(self, series, year, period, value):
        self.series = series
        self.year = year
        self.period = period
        self.value = value

    def __str__(self):
        return "{} ({}): {}".format(self.date, self.period, self.value)

    def __eq__(self, other):
        return (
            self.value == other.value
            and self.series == other.series
            and self.year == other.year
            and self.period == other.period
        )

    def __dict__(self):
        return {
            "series": self.series.__dict__(),
            "year": self.year,
            "date": str(self.date),
            "period": self.period.__dict__(),
            "value": self.value,
        }

    @property
    def date(self):
        """
        Accepts a row from the raw BLS data. Returns a Python date object based on its period.
        """
        return date(self.year, self.period.month, 1)
