#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Download the latest annual Consumer Price Index (CPI) dataset.
"""
# Files
import os
import csv
import sqlite3
import requests
import pandas as pd

# Logging
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class Downloader(object):
    THIS_DIR = os.path.dirname(__file__)
    FILE_LIST = [
        "cu.area",
        "cu.item",
        "cu.period",
        "cu.periodicity",
        "cu.series",
        "cu.data.0.Current",
        "cu.data.1.AllItems",
        "cu.data.2.Summaries",
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
        "cu.data.20.USCommoditiesServicesSpecial",
    ]

    def update(self):
        """
        Update the Consumer Price Index dataset that powers this library.
        """
        logger.debug("Downloading {} files from the BLS".format(len(self.FILE_LIST)))
        [self.get_tsv(file) for file in self.FILE_LIST]
        logger.debug("Loading data into SQLite database")
        [self.insert_tsv(file) for file in self.FILE_LIST]

    def insert_tsv(self, file):
        # Connect to db
        db_path = os.path.join(self.THIS_DIR, "cpi.db")
        conn = sqlite3.connect(db_path)

        # Read file
        logger.debug(" - {}".format(file))
        csv_path = os.path.join(self.THIS_DIR, "{}.csv".format(file))
        csv_reader = list(csv.DictReader(open(csv_path, "r")))

        # Convert it to a DataFrame
        df = pd.DataFrame(csv_reader)
        df.drop([None], axis=1, inplace=True, errors="ignore")

        # Write file to db
        df.to_sql(file, conn, if_exists="replace", index=False)

        # Close connection
        conn.close()

    def get_tsv(self, file):
        """
        Download TSV file from the BLS.
        """
        # Download it
        url = "https://download.bls.gov/pub/time.series/cu/{}".format(file)
        logger.debug(" - {}".format(url))
        tsv_path = os.path.join(self.THIS_DIR, "{}.tsv".format(file))
        response = requests.get(url)
        with open(tsv_path, "w") as f:
            f.write(response.text)

        # Convert it to csv
        reader = csv.reader(open(tsv_path, "r"), delimiter="\t")
        csv_path = os.path.join(self.THIS_DIR, "{}.csv".format(file))
        writer = csv.writer(open(csv_path, "w"))
        for row in reader:
            writer.writerow([cell.strip() for cell in row])


if __name__ == "__main__":
    Downloader().update()
