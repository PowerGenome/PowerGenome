"Test functions from price_adjustment.py"

import pandas as pd
from powergenome.price_adjustment import inflation_price_adjustment, load_cpi_data


def test_load_cpi(tmp_path):
    # Get data the first time
    cpi = load_cpi_data(data_path=tmp_path / "cpi_data.csv")

    # Get data saved to file
    cpi = load_cpi_data(data_path=tmp_path / "cpi_data.csv")

    assert all(cpi["value"].notna())
    assert all(cpi["period"] == 12)


def test_inflate_price(tmp_path):
    p = 10
    p2 = inflation_price_adjustment(p, 2000, 2010, data_path=tmp_path / "cpi_data.csv")
    assert p2 > p

    # remove later years from saved CPI file
    cpi = pd.read_csv(tmp_path / "cpi_data.csv")
    first_year = cpi["year"].min()
    last_year = cpi["year"].max()
    cpi.loc[5:10, :].to_csv(tmp_path / "cpi_data.csv", float_format="%g")

    p = pd.Series([1, 10])
    p2 = inflation_price_adjustment(p, first_year, last_year)
    assert all(p2 > p)


def test_user_cpi(tmp_path):
    cpi_data_path = tmp_path / "cpi_data.csv"
    data = {"year": range(2000, 2022), "value": range(1, 23)}
    cpi_data = pd.DataFrame(data)
    cpi_data.to_csv(cpi_data_path, index=False)

    p = 10
    p2 = inflation_price_adjustment(p, 2000, 2010, data_path=cpi_data_path)
    assert p2 > p
