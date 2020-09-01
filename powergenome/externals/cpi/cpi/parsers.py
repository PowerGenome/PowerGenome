#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Parse and prepare the Consumer Price Index (CPI) dataset.
"""
import os
import sqlite3
import logging
from .models import MappingList, SeriesList
from .models import Area, Item, Period, Periodicity, Index, Series

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class BaseParser(object):
    THIS_DIR = os.path.dirname(__file__)

    def get_file(self, file):
        """
        Returns the CPI data file provided as a list of dictionaries.
        """
        # Connect to database
        db_path = os.path.join(self.THIS_DIR, "cpi.db")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Query this file
        query = cursor.execute('SELECT * FROM "{}"'.format(file))
        columns = [d[0] for d in query.description]
        result_list = [dict(zip(columns, r)) for r in query.fetchall()]

        # Close database connection
        cursor.close()
        cursor.connection.close()

        # Return data
        return result_list


class ParseArea(BaseParser):
    """
    Parses the raw list of CPI areas.
    """

    def parse(self):
        """
        Returns a list Area objects.
        """
        logger.debug("Parsing area file")
        object_list = MappingList()
        for row in self.get_file("cu.area"):
            obj = Area(row["area_code"], row["area_name"])
            object_list.append(obj)
        return object_list


class ParseItem(BaseParser):
    """
    Parses the raw list of CPI items.
    """

    def parse(self):
        """
        Returns a list Area objects.
        """
        logger.debug("Parsing item file")
        object_list = MappingList()
        for row in self.get_file("cu.item"):
            obj = Item(row["item_code"], row["item_name"])
            object_list.append(obj)
        return object_list


class ParsePeriod(BaseParser):
    """
    Parses the raw list of CPI periods.
    """

    def parse(self):
        """
        Returns a list Area objects.
        """
        logger.debug("Parsing period file")
        object_list = MappingList()
        for row in self.get_file("cu.period"):
            obj = Period(row["period"], row["period_abbr"], row["period_name"])
            object_list.append(obj)
        return object_list


class ParsePeriodicity(BaseParser):
    """
    Parses the raw list of CPI periodicities.
    """

    def parse(self):
        """
        Returns a list Periodicity objects.
        """
        logger.debug("Parsing periodicity file")
        object_list = MappingList()
        for row in self.get_file("cu.periodicity"):
            obj = Periodicity(row["periodicity_code"], row["periodicity_name"])
            object_list.append(obj)
        return object_list


class ParseSeries(BaseParser):
    """
    Parses the raw list of CPI series from the BLS.
    """

    SURVEYS = {
        "CU": "All urban consumers",
        "CW": "Urban wage earners and clerical workers",
    }
    FILE_LIST = [
        "cu.data.0.Current",
        # "cu.data.1.AllItems",
        # "cu.data.2.Summaries",
        # "cu.data.3.AsizeNorthEast",
        # "cu.data.4.AsizeNorthCentral",
        # "cu.data.5.AsizeSouth",
        # "cu.data.6.AsizeWest",
        # "cu.data.7.OtherNorthEast",
        # "cu.data.8.OtherNorthCentral",
        # "cu.data.9.OtherSouth",
        # "cu.data.10.OtherWest",
        # "cu.data.11.USFoodBeverage",
        # "cu.data.12.USHousing",
        # "cu.data.13.USApparel",
        # "cu.data.14.USTransportation",
        # "cu.data.15.USMedical",
        # "cu.data.16.USRecreation",
        # "cu.data.17.USEducationAndCommunication",
        # "cu.data.18.USOtherGoodsAndServices",
        # "cu.data.19.PopulationSize",
        # "cu.data.20.USCommoditiesServicesSpecial"
    ]

    def __init__(self, periods=None, periodicities=None, areas=None, items=None):
        self.periods = periods or ParsePeriod().parse()
        self.periodicities = periodicities or ParsePeriodicity().parse()
        self.areas = areas or ParseArea().parse()
        self.items = items or ParseItem().parse()

    def parse_id(self, id):
        return dict(
            survey_code=id[:2],
            seasonal_code=id[2:3],
            periodicity_code=id[3:4],
            area_code=id[4:8],
            item_code=id[8:],
        )

    def parse(self):
        self.series_list = self.parse_series()
        self.parse_indexes()
        return self.series_list

    def parse_series(self):
        """
        Returns a list Series objects.
        """
        logger.debug("Parsing series file")
        object_list = SeriesList(
            periodicities=self.periodicities, areas=self.areas, items=self.items
        )
        for row in self.get_file("cu.series"):
            parsed_id = self.parse_id(row["series_id"])
            obj = Series(
                row["series_id"],
                row["series_title"],
                self.SURVEYS[parsed_id["survey_code"]],
                row["seasonal"] == "S",
                self.periodicities.get_by_id(row["periodicity_code"]),
                self.areas.get_by_id(row["area_code"]),
                self.items.get_by_id(row["item_code"]),
            )
            object_list.append(obj)
        return object_list

    def parse_indexes(self):
        logger.debug("Parsing index files")
        # Loop through all the files ...
        for file in self.FILE_LIST:
            # ... and for each file ...
            for row in self.get_file(file):
                # Get the series
                series = self.series_list.get_by_id(row["series_id"])

                # Create an object
                index = Index(
                    series,
                    int(row["year"]),
                    self.periods.get_by_id(row["period"]),
                    float(row["value"]),
                )

                # If the value has already been loaded ...
                if index.date in series._indexes[index.period.type]:
                    # ... verify this value matches what we have ...
                    assert index == series._indexes[index.period.type][index.date]
                else:
                    # ... and if the series doesn't have the index yet, add it.
                    series._indexes[index.period.type][index.date] = index
