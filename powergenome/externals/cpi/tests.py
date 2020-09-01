#! /usr/bin/env python
# -*- coding: utf-8 -*-
import cpi
import warnings
import unittest
from cpi import cli
import pandas as pd
from datetime import date, datetime
from click.testing import CliRunner
from cpi.errors import CPIObjectDoesNotExist


class BaseCPITest(unittest.TestCase):
    """
    These global variables change with each data update.
    """

    LATEST_YEAR = 2018
    LATEST_YEAR_1950_ALL_ITEMS = 1041.9377593360996
    LATEST_YEAR_1950_CUSR0000SA0 = 1041.9377593360996
    LATEST_MONTH = date(2019, 11, 1)
    LATEST_MONTH_1950_ALL_ITEMS = 1094.5021276595746
    LATEST_MONTH_1950_CUSR0000SA0 = 1097.1331348362398


class CPITest(BaseCPITest):
    def test_latest_year(self):
        self.assertEqual(cpi.LATEST_YEAR, self.LATEST_YEAR)

    def test_latest_month(self):
        self.assertEqual(cpi.LATEST_MONTH, self.LATEST_MONTH)

    def test_get(self):
        self.assertEqual(cpi.get(1950), 24.1)
        self.assertEqual(cpi.get(date(1950, 1, 1)), 23.5)
        self.assertEqual(cpi.get(2000), 172.2)

    def test_get_by_kwargs(self):
        # "CUUR0000SA0"
        self.assertEqual(cpi.get(2000), 172.2)

        # "CUSR0000SA0"
        self.assertEqual(cpi.get(date(2000, 1, 1), seasonally_adjusted=True), 169.30)
        # ... which doesn't have annual values
        with self.assertRaises(CPIObjectDoesNotExist):
            cpi.get(2000, seasonally_adjusted=True)

        # "CUSR0000SA0E"
        # ... which we don't have loaded yet as data
        with self.assertRaises(CPIObjectDoesNotExist):
            cpi.get(2000, seasonally_adjusted=True, items="Energy")

        # "CUURS49ASA0"
        self.assertEqual(
            cpi.get(2000, area="Los Angeles-Long Beach-Anaheim, CA"), 171.6
        )
        self.assertEqual(
            cpi.get(date(2000, 1, 1), area="Los Angeles-Long Beach-Anaheim, CA"), 167.9
        )

        # "CUURS49ASA0E"
        self.assertEqual(
            cpi.get(2000, items="Energy", area="Los Angeles-Long Beach-Anaheim, CA"),
            132.0,
        )

        # "CUURA421SAT"
        self.assertEqual(
            cpi.get(
                2000, items="Transportation", area="Los Angeles-Long Beach-Anaheim, CA"
            ),
            154.2,
        )

        # "CUURA421SA0E"
        self.assertEqual(
            cpi.get(
                2000,
                items="All items - old base",
                area="Los Angeles-Long Beach-Anaheim, CA",
            ),
            506.8,
        )

    def test_get_by_series_id(self):
        self.assertEqual(cpi.get(date(1950, 1, 1), series_id="CUSR0000SA0"), 23.51)

    def test_series_list(self):
        cpi.series.get_by_id("CUSR0000SA0")

    def test_series_indexes(self):
        for series in cpi.series:
            self.assertTrue(len(series.indexes) > 0)
            series.latest_month
            series.latest_year
            series.__str__()
            series.__dict__()
            for index in series.indexes:
                index.__str__()
                index.__dict__()

    def test_get_errors(self):
        with self.assertRaises(CPIObjectDoesNotExist):
            cpi.get(1900)
        with self.assertRaises(CPIObjectDoesNotExist):
            cpi.get(date(1900, 1, 1))
        with self.assertRaises(CPIObjectDoesNotExist):
            cpi.get(1950, series_id="FOOBAR")

    def test_get_value_error(self):
        with self.assertRaises(ValueError):
            cpi.get(1900.1)
            cpi.get(datetime.now())
            cpi.get(3000)

    def test_inflate_years(self):
        self.assertEqual(cpi.inflate(100, 1950), self.LATEST_YEAR_1950_ALL_ITEMS)
        self.assertEqual(
            cpi.inflate(100, 1950, series_id="CUUR0000SA0"),
            self.LATEST_YEAR_1950_CUSR0000SA0,
        )
        self.assertEqual(cpi.inflate(100, 1950, to=2017), 1017.0954356846472)
        self.assertEqual(cpi.inflate(100, 1950, to=1960), 122.82157676348547)
        self.assertEqual(cpi.inflate(100.0, 1950, to=1950), 100)

    def test_inflate_months(self):
        self.assertEqual(
            cpi.inflate(100, date(1950, 1, 1)), self.LATEST_MONTH_1950_ALL_ITEMS
        )
        self.assertEqual(
            cpi.inflate(100, date(1950, 1, 11)), self.LATEST_MONTH_1950_ALL_ITEMS
        )
        self.assertEqual(
            cpi.inflate(100, datetime(1950, 1, 1)), self.LATEST_MONTH_1950_ALL_ITEMS
        )
        self.assertEqual(
            cpi.inflate(100, date(1950, 1, 1), to=date(2018, 1, 1)), 1054.7531914893618
        )
        self.assertEqual(
            cpi.inflate(100, date(1950, 1, 1), to=date(1960, 1, 1)), 124.68085106382979
        )

    def test_inflate_other_series(self):
        self.assertEqual(
            cpi.inflate(100, date(1950, 1, 1), series_id="CUSR0000SA0"),
            self.LATEST_MONTH_1950_CUSR0000SA0,
        )

    def test_deflate(self):
        self.assertEqual(cpi.inflate(1017.0954356846472, 2017, to=1950), 100)
        self.assertEqual(cpi.inflate(122.82157676348547, 1960, to=1950), 100)

    def test_numpy_dtypes(self):
        self.assertEqual(cpi.get(pd.np.int64(1950)), cpi.get(1950))
        self.assertEqual(cpi.inflate(100, pd.np.int32(1950)), cpi.inflate(100, 1950))
        self.assertEqual(
            cpi.inflate(100, pd.np.int64(1950), to=pd.np.int64(1960)),
            cpi.inflate(100, 1950, to=1960),
        )
        self.assertEqual(
            cpi.inflate(100, pd.np.int64(1950), to=pd.np.int32(1960)),
            cpi.inflate(100, 1950, to=1960),
        )
        self.assertEqual(
            cpi.inflate(100, pd.np.int64(1950), to=1960),
            cpi.inflate(100, 1950, to=1960),
        )
        self.assertEqual(
            cpi.inflate(
                100, pd.to_datetime("1950-07-01"), to=pd.to_datetime("1960-07-01")
            ),
            cpi.inflate(100, date(1950, 7, 1), to=date(1960, 7, 1)),
        )

    def test_mismatch(self):
        with self.assertRaises(TypeError):
            cpi.inflate(100, 1950, to=date(2000, 1, 1))
        with self.assertRaises(TypeError):
            cpi.inflate(100, date(2000, 1, 1), to=1950)

    def test_warning(self):
        warnings.warn(cpi.StaleDataWarning())

    def test_pandas(self):
        df = pd.read_csv("test.csv")
        df["ADJUSTED"] = df.apply(
            lambda x: cpi.inflate(x.MEDIAN_HOUSEHOLD_INCOME, x.YEAR), axis=1
        )
        df = df.set_index("YEAR")
        self.assertEqual(
            cpi.inflate(df.at[1984, "MEDIAN_HOUSEHOLD_INCOME"], 1984),
            df.at[1984, "ADJUSTED"],
        )
        cpi.series.to_dataframe()
        cpi.series.get().to_dataframe()


class CliTest(BaseCPITest):
    def invoke(self, *args):
        runner = CliRunner()
        result = runner.invoke(cli.inflate, args)
        self.assertEqual(result.exit_code, 0)
        string_value = result.output.replace("\n", "")
        # Do some rounding to ensure the same results for Python 2 and 3
        return str(round(float(string_value), 7))

    def test_inflate_years(self):
        self.assertEqual(
            self.invoke("100", "1950"), str(round(self.LATEST_YEAR_1950_CUSR0000SA0, 7))
        )
        self.assertEqual(self.invoke("100", "1950", "--to", "1960"), "122.8215768")
        self.assertEqual(self.invoke("100", "1950", "--to", "1950"), "100.0")

    def test_inflate_months(self):
        self.assertEqual(
            self.invoke("100", "1950-01-01"),
            str(round(self.LATEST_MONTH_1950_ALL_ITEMS, 7)),
        )
        self.assertEqual(
            self.invoke("100", "1950-01-11"),
            str(round(self.LATEST_MONTH_1950_ALL_ITEMS, 7)),
        )
        self.assertEqual(
            self.invoke("100", "1950-01-11", "--to", "1960-01-01"), "124.6808511"
        )
        self.assertEqual(
            self.invoke("100", "1950-01-01 00:00:00", "--to", "1950-01-01"), "100.0"
        )
        self.assertEqual(
            self.invoke("100", "1950-01-01", "--to", "2018-01-01"), "1054.7531915"
        )
        self.assertEqual(
            self.invoke("100", "1950-01-01", "--to", "1960-01-01"), "124.6808511"
        )
        self.assertEqual(
            self.invoke("100", "1950-01-01", "--series_id", "CUSR0000SA0"),
            str(round(self.LATEST_MONTH_1950_CUSR0000SA0, 7)),
        )


if __name__ == "__main__":
    unittest.main()
