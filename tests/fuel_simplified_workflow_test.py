"""Tests for simplified fuel price workflow without fuel_region_map, including aggregated
region averaging and direct model-region fuel labels.
"""

import pandas as pd
import pytest

from powergenome.fuels import fetch_fuel_prices
from powergenome.generators import add_fuel_labels


class DummySettings(dict):
    """Lightweight settings container allowing attribute-style updates if needed."""

    pass


@pytest.fixture()
def simplified_settings():
    # Minimal settings mimicking simplified workflow
    s = DummySettings()
    s["fuel_data_year"] = 2025
    s["target_usd_year"] = 2025  # No inflation adjustment edge case
    s["model_year"] = 2030  # Single planning period final year
    s["model_regions"] = ["p1", "p2", "p1_2", "p3"]
    s["region_aggregations"] = {"p1_2": ["p1", "p2"]}
    # Provide fuel_scenarios for mapping scenario->fuel series
    s["fuel_scenarios"] = {"coal": "reference", "naturalgas": "reference"}
    s["tech_fuel_map"] = {"Coal": "coal", "Gas": "naturalgas"}
    # Required by inflation function (won't be used because target matches dollar year)
    s["data_location"] = "tests/test_system/test_data"
    s["dollar_year_table"] = "cpi_data.csv"
    return s


@pytest.fixture()
def monkeypatch_get_data(monkeypatch):
    def _patch(df):
        monkeypatch.setattr(
            "powergenome.fuels.get_data",
            lambda name: df if name == "fuel_price" else pd.DataFrame(),
        )

    return _patch


def build_base_fuel_df():
    # Construct minimal base price table with p1 and p2 only; aggregated p1_2 should be created
    rows = []
    for region, coal_price, gas_price in [
        ("p1", 2.0, 3.0),
        ("p2", 4.0, 5.0),
        ("p3", 6.0, 7.0),  # include unrelated region to ensure not part of aggregation
    ]:
        rows.append(
            {
                "year": 2030,
                "price": coal_price,
                "data_year": 2025,
                "scenario": "reference",
                "fuel": "coal",
                "region": region,
                "dollar_year": 2025,
            }
        )
        rows.append(
            {
                "year": 2030,
                "price": gas_price,
                "data_year": 2025,
                "scenario": "reference",
                "fuel": "naturalgas",
                "region": region,
                "dollar_year": 2025,
            }
        )
    return pd.DataFrame(rows)


def test_aggregated_region_average_and_full_fuel_name(
    simplified_settings, monkeypatch_get_data
):
    base_df = build_base_fuel_df()
    monkeypatch_get_data(base_df)
    prices = fetch_fuel_prices(simplified_settings, inflate_price=False)

    # Aggregated region should exist
    assert "p1_2" in prices["region"].unique(), "Aggregated region p1_2 not created"

    # Average price check: (2.0 + 4.0)/2 = 3.0 for coal; (3.0 + 5.0)/2 = 4.0 for naturalgas
    coal_price_agg = prices.query(
        "region == 'p1_2' and fuel == 'coal' and year == 2030"
    )["price"].iloc[0]
    gas_price_agg = prices.query(
        "region == 'p1_2' and fuel == 'naturalgas' and year == 2030"
    )["price"].iloc[0]
    assert coal_price_agg == pytest.approx(
        3.0
    ), f"Expected coal average 3.0 got {coal_price_agg}"
    assert gas_price_agg == pytest.approx(
        4.0
    ), f"Expected gas average 4.0 got {gas_price_agg}"

    # full_fuel_name format region_scenario_fuel
    assert any(
        prices["full_fuel_name"].str.contains("p1_2_reference_coal")
    ), "Missing expected full fuel name for aggregated region"


def test_add_fuel_labels_simplified_path(simplified_settings, monkeypatch_get_data):
    base_df = build_base_fuel_df()
    monkeypatch_get_data(base_df)
    prices = fetch_fuel_prices(simplified_settings, inflate_price=False)

    # Build minimal generators df
    gens = pd.DataFrame(
        {
            "technology": ["Coal Plant", "Gas Turbine", "Coal Plant"],
            "region": ["p1", "p1_2", "p2"],
        }
    )

    labeled = add_fuel_labels(gens.copy(), prices, simplified_settings)

    # Expect Fuel labels using model region directly: region_scenario_fuel
    expected_labels = {
        "p1_reference_coal",
        "p1_2_reference_naturalgas",  # if Gas Turbine mapped to naturalgas
        "p2_reference_coal",
    }
    assert expected_labels.issubset(
        set(labeled["Fuel"])
    ), f"Fuel labels missing: {expected_labels - set(labeled['Fuel'])}"
    # No legacy aeo region names (e.g., pacific) should appear
    assert (
        not labeled["Fuel"].str.contains("pacific", case=False).any()
    ), "Legacy AEO region name leaked into simplified labels"
