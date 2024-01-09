"Test functions related to fuel assignment, price, and emissions"

import logging
import os
import numpy as np
import pandas as pd
import pytest

import powergenome
from powergenome.eia_opendata import fetch_fuel_prices, modify_fuel_prices
from powergenome.fuels import (
    add_user_fuel_prices,
    add_carbon_tax,
    adjust_ccs_fuels,
    fuel_cost_table,
)
from powergenome.generators import GeneratorClusters
from powergenome.params import DATA_PATHS
from powergenome.util import init_pudl_connection

logger = logging.getLogger(powergenome.__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    # More extensive test-like formatter...
    "%(asctime)s [%(levelname)8s] %(name)s:%(lineno)s %(message)s",
    # This is the datetime format string.
    "%Y-%m-%d %H:%M:%S",
)
handler.setFormatter(formatter)
logger.addHandler(handler)

if os.name == "nt":
    # if user is using a windows system
    sql_prefix = "sqlite:///"
else:
    sql_prefix = "sqlite:////"
pudl_engine, pudl_out, pg_engine = init_pudl_connection(
    start_year=2018,
    end_year=2020,
    pudl_db=sql_prefix + str(DATA_PATHS["test_data"] / "pudl_test_data.db"),
    pg_db=sql_prefix + str(DATA_PATHS["test_data"] / "pg_misc_tables.sqlite3"),
)


@pytest.fixture()
def fuel_settings():
    settings = {
        "modified_atb_new_gen": {
            "ZCF_CombinedCycle1": {
                "new_technology": "ZCF",
                "new_tech_detail": "CCAvgCF",
                "new_cost_case": "Advanced",
                "atb_technology": "NaturalGas",
                "atb_tech_detail": "CCAvgCF",
                "atb_cost_case": "Advanced",
                "size_mw": 500,
            },
            "ZCF_CombinedCycle2": {
                "new_technology": "ZeroCarbon",
                "new_tech_detail": "CCAvgCF",
                "new_cost_case": "Advanced",
                "atb_technology": "NaturalGas",
                "atb_tech_detail": "CCAvgCF",
                "atb_cost_case": "Advanced",
                "size_mw": 500,
            },
            "biopower_ccs": {
                "new_technology": "Biopower",
                "new_tech_detail": "DedicatedCCS",
                "new_cost_case": "Moderate",
                "atb_technology": "Biopower",
                "atb_tech_detail": "Dedicated",
                "atb_cost_case": "Moderate",
                "size_mw": 500,
            },
        },
        "eia_atb_tech_map": {
            "Biomass": ["Biopower_Dedicated"],
            "Zero Carbon": ["ZCF"],
        },
        "atb_new_gen": [
            ["Biopower", "Dedicated", "Moderate", 100],
            ["NaturalGas", "CTAvgCF", "Moderate", 100],
            ["NaturalGas", "CCCCSAvgCF", "Moderate", 100],
        ],
        "atb_data_year": 2020,
        "atb_cap_recovery_years": 20,
        "fuel_eia_aeo_year": 2021,
        "aeo_fuel_usd_year": 2021,
        "eia_series_scenario_names": {"reference": "REF2021"},
        "aeo_fuel_region_map": {"south_atlantic": ["S_VACA", "PJM_Dom"]},
        "eia_series_region_names": {"south_atlantic": "SOATL"},
        "eia_series_fuel_names": {"naturalgas": "NG"},
        "aeo_fuel_scenarios": {"naturalgas": "reference"},
        "user_fuel_price": {
            "zerocarbonfuel1": 14,
            "zerocarbonfuel2": 10,
            "biomass": {"S_VACA": 10, "PJM_Dom": 5},
        },
        "user_fuel_usd_year": {
            "zerocarbonfuel1": 2020,
            "zerocarbonfuel2": 2020,
            "biomass": 2019,
        },
        "tech_fuel_map": {
            "ZeroCarbon_CCAvgCF_Advanced": "zerocarbonfuel1",
            "Zero Carbon": "zerocarbonfuel2",
            "Biomass": "biomass",
            "NaturalGas": "naturalgas",
        },
        "ccs_fuel_map": {
            "biopower_dedicatedccs": "biomass_ccs",
            "naturalgas_ccccs": "naturalgas_ccs90",
        },
        "ccs_capture_rate": {"biomass_ccs": 0.9, "naturalgas_ccs90": 0.9},
        "model_regions": ["S_VACA", "PJM_Dom"],
        "model_year": 2040,
        "model_first_planning_year": 2035,
        "cost_multiplier_region_map": {
            "SRCA": ["S_VACA"],
            "PJMD": ["PJM_Dom"],
        },
        "cost_multiplier_technology_map": {
            "Biomass": ["Biopower_Dedicated"],
            "CC - multi shaft": ["ZeroCarbon_CCAvgCF", "ZCF_CCAvgCF"],
        },
        "fuel_emission_factors": {
            "biomass": 0.1,  # Dummy value for biomass
            "naturalgas": 0.05306,
        },
        "ccs_disposal_cost": 25,
    }
    return settings


def test_fuel_labels_and_prices(fuel_settings):
    df_base = add_user_fuel_prices(fuel_settings)

    for fuel in [
        "S_VACA_biomass",
        "PJM_Dom_biomass",
        "zerocarbonfuel1",
        "zerocarbonfuel2",
    ]:
        assert fuel in df_base["full_fuel_name"].unique()

    fuel_settings["target_usd_year"] = 2020
    df_inflate = add_user_fuel_prices(fuel_settings)
    assert (
        df_base.loc[df_base["fuel"] == "biomass", "price"].mean()
        < df_inflate.loc[df_inflate["fuel"] == "biomass", "price"].mean()
    )
    assert np.allclose(
        df_base.loc[df_base["fuel"] == "zerocarbonfuel1", "price"].mean(),
        df_inflate.loc[df_inflate["fuel"] == "zerocarbonfuel1", "price"].mean(),
    )

    gc = GeneratorClusters(
        pudl_engine, pudl_out, pg_engine, fuel_settings, current_gens=False
    )
    gens = gc.create_new_generators()
    assert gens["Fuel"].isna().any() == False
    assert gens["Fuel"].str.contains("ccs", case=False).any() == True

    fuel_table = fuel_cost_table(gc.fuel_prices, gens, fuel_settings)
    for r in ["PJM_Dom", "S_VACA"]:
        assert (
            fuel_table.loc[0, f"{r}_biomass"]
            == fuel_settings["fuel_emission_factors"]["biomass"]
        )
        assert np.allclose(
            fuel_table.loc[0, f"{r}_biomass_ccs"],
            fuel_settings["fuel_emission_factors"]["biomass"]
            * (1 - fuel_settings["ccs_capture_rate"]["biomass_ccs"]),
        )
    assert (fuel_table.loc[1:, :] == 0).any().any() == False


def test_fetch_fuel_price_errors(fuel_settings, caplog):
    aeo_year = fuel_settings.pop("fuel_eia_aeo_year")
    with pytest.raises(KeyError):
        fetch_fuel_prices(fuel_settings)

    fuel_settings["eia_aeo_year"] = {"bad": "value"}
    with pytest.raises(TypeError):
        fetch_fuel_prices(fuel_settings)

    fuel_settings["eia_aeo_year"] = aeo_year

    eia_series_scenario_names = fuel_settings.pop("eia_series_scenario_names")
    with pytest.raises(KeyError):
        fetch_fuel_prices(fuel_settings)

    fuel_settings["eia_series_scenario_names"] = eia_series_scenario_names
    caplog.set_level(logging.WARNING)
    fetch_fuel_prices(fuel_settings)
    # assert "Unable to inflate fuel prices" in caplog.text

    eia_series_region_names = fuel_settings.pop("eia_series_region_names")
    eia_series_fuel_names = fuel_settings.pop("eia_series_fuel_names")

    fetch_fuel_prices(fuel_settings)
    assert "EIA fuel region names were not found" in caplog.text
    assert "EIA fuel names were not found" in caplog.text


def test_regional_fuel_price_mod(fuel_settings):
    fuel_prices = fetch_fuel_prices(fuel_settings)
    mod_fuel_prices = modify_fuel_prices(
        fuel_prices,
        fuel_settings["aeo_fuel_region_map"],
        fuel_settings.get("regional_fuel_adjustments"),
    )
    assert np.allclose(fuel_prices["price"].values, mod_fuel_prices["price"].values)

    fuel_settings["regional_fuel_adjustments"] = {
        "S_VACA": ["mul", 2],
        "PJM_Dom": {"naturalgas": ["add", 1]},
    }

    mod_fuel_prices = modify_fuel_prices(
        fuel_prices,
        fuel_settings["aeo_fuel_region_map"],
        fuel_settings.get("regional_fuel_adjustments"),
    )
    assert np.isclose(
        fuel_prices.query("region == 'south_atlantic'")["price"].mean(),
        mod_fuel_prices.query("region == 'S_VACA'")["price"].mean() / 2,
    )

    assert np.isclose(
        fuel_prices.query("region == 'south_atlantic' and fuel == 'naturalgas'")[
            "price"
        ].mean(),
        mod_fuel_prices.query("region == 'PJM_Dom' and fuel == 'naturalgas'")[
            "price"
        ].mean()
        - 1,
    )

    fuel_settings["regional_fuel_adjustments"] = {
        "S_VACA": ["mul", 2],
        "PJM_Dom": {"coal": ["add", 1]},
    }
    with pytest.raises(KeyError):
        modify_fuel_prices(
            fuel_prices,
            fuel_settings["aeo_fuel_region_map"],
            fuel_settings.get("regional_fuel_adjustments"),
        )

    fuel_settings["regional_fuel_adjustments"] = {
        "S_VACA": ["mul", 2],
        "PJM_Dom": {"coal": 1},
    }
    with pytest.raises(KeyError):
        modify_fuel_prices(
            fuel_prices,
            fuel_settings["aeo_fuel_region_map"],
            fuel_settings.get("regional_fuel_adjustments"),
        )
    fuel_settings["regional_fuel_adjustments"] = {
        "S_VACA": 1,
        "PJM_Dom": {"coal": 1},
    }
    with pytest.raises(TypeError):
        modify_fuel_prices(
            fuel_prices,
            fuel_settings["aeo_fuel_region_map"],
            fuel_settings.get("regional_fuel_adjustments"),
        )

    fuel_settings["regional_fuel_adjustments"] = {
        "S_VACA": ["mul", 2],
        "PJM_Dom": {"naturalgas": ["add", 1]},
    }
    with pytest.raises(KeyError):
        modify_fuel_prices(
            fuel_prices,
            None,
            fuel_settings.get("regional_fuel_adjustments"),
        )

    fuel_settings["regional_fuel_adjustments"] = {
        "S_VACA": ["mul", 2],
        "PJM_Dom": {"naturalga": ["add", 1]},
    }
    with pytest.raises(KeyError):
        modify_fuel_prices(
            fuel_prices,
            fuel_settings["aeo_fuel_region_map"],
            fuel_settings.get("regional_fuel_adjustments"),
        )

    fuel_settings["regional_fuel_adjustments"] = {
        "S_VACA": ["mul", 2],
        "PJM_Dom": {"naturalgas": ["ad", 1]},
    }
    with pytest.raises(KeyError):
        modify_fuel_prices(
            fuel_prices,
            fuel_settings["aeo_fuel_region_map"],
            fuel_settings.get("regional_fuel_adjustments"),
        )


def test_regional_mod_fuel_labels(fuel_settings):
    fuel_settings["regional_fuel_adjustments"] = {
        "S_VACA": ["mul", 2],
        "PJM_Dom": {"naturalgas": ["add", 1]},
    }
    fuel_settings["target_usd_year"] = 2020
    gc = GeneratorClusters(
        pudl_engine, pudl_out, pg_engine, fuel_settings, current_gens=False
    )
    gens = gc.create_new_generators()
    assert "S_VACA_reference_naturalgas" in gens.Fuel.values
    assert "PJM_Dom_reference_naturalgas" in gens.Fuel.values


class TestAddCarbonTax:
    # returns unaltered dataframe if no carbon tax value is provided
    def test_returns_unaltered_dataframe_if_no_carbon_tax_value_is_provided(self):
        # Arrange
        fuel_df = pd.DataFrame(
            {"Cost_per_MMBtu": [10, 20, 30], "CO2_content_tons_per_MMBtu": [1, 2, 3]}
        )

        # Act
        result = add_carbon_tax(fuel_df)

        # Assert
        assert result.equals(fuel_df)

    # increases fuel prices to reflect carbon tax value
    def test_increases_fuel_prices_to_reflect_carbon_tax_value(self):
        # Arrange
        fuel_df = pd.DataFrame(
            {"Cost_per_MMBtu": [10, 20, 30], "CO2_content_tons_per_MMBtu": [1, 2, 3]}
        )
        carbon_tax_value = 5

        expected_result = pd.DataFrame(
            {"Cost_per_MMBtu": [15, 30, 45], "CO2_content_tons_per_MMBtu": [1, 2, 3]}
        )

        # Act
        result = add_carbon_tax(fuel_df, carbon_tax_value)

        # Assert
        assert result.equals(expected_result)

    # raises KeyError if "Cost_per_MMBtu" column is missing from input dataframe
    def test_raises_KeyError_if_Cost_per_MMBtu_column_is_missing(self):
        # Arrange
        fuel_df = pd.DataFrame({"CO2_content_tons_per_MMBtu": [1, 2, 3]})
        carbon_tax_value = 5

        # Act & Assert
        with pytest.raises(KeyError):
            add_carbon_tax(fuel_df, carbon_tax_value)

    # raises KeyError if "CO2_content_tons_per_MMBtu" column is missing from input dataframe
    def test_raises_KeyError_if_CO2_content_tons_per_MMBtu_column_is_missing(self):
        # Arrange
        fuel_df = pd.DataFrame({"Cost_per_MMBtu": [10, 20, 30]})
        carbon_tax_value = 5

        # Act & Assert
        with pytest.raises(KeyError):
            add_carbon_tax(fuel_df, carbon_tax_value)

    # returns unaltered dataframe if input dataframe is empty
    def test_returns_unaltered_dataframe_if_input_dataframe_is_empty(self):
        # Arrange
        fuel_df = pd.DataFrame(columns=["Cost_per_MMBtu", "CO2_content_tons_per_MMBtu"])
        carbon_tax_value = 5

        # Act
        result = add_carbon_tax(fuel_df, carbon_tax_value)

        # Assert
        assert result.empty


class TestAdjustCCSFuels:
    # If the function is called with a row that does not contain a CCS fuel, it should return the row unmodified.
    def test_no_ccs_fuel(self):
        # Arrange
        row = pd.Series(
            {"Fuel": "coal", "Cost_per_MMBtu": 10, "CO2_content_tons_per_MMBtu": 5}
        )

        # Act
        result = adjust_ccs_fuels(row)

        # Assert
        assert result["Fuel"] == "coal"
        assert result["Cost_per_MMBtu"] == 10
        assert result["CO2_content_tons_per_MMBtu"] == 5

    # If the function is called with a row that contains a CCS fuel, it should adjust the "CO2_content_tons_per_MMBtu" and "Cost_per_MMBtu" values based on the capture rate and disposal cost specified in the settings.
    def test_with_ccs_fuel(self):
        # Arrange
        row = pd.Series(
            {
                "Fuel": "naturalgas_ccs90",
                "Cost_per_MMBtu": 10,
                "CO2_content_tons_per_MMBtu": 5,
            }
        )
        ccs_fuels = ["naturalgas_ccs90"]
        ccs_capture_rate = {"naturalgas_ccs90": 0.9}
        ccs_disposal_cost = 50

        # Act
        result = adjust_ccs_fuels(row, ccs_fuels, ccs_capture_rate, ccs_disposal_cost)

        # Assert
        assert result["Fuel"] == "naturalgas_ccs90"
        assert result["Cost_per_MMBtu"] == 10 + (5 * 0.9 * 50)
        assert result["CO2_content_tons_per_MMBtu"] == 5 - (5 * 0.9)

    # If the function is called with a row that contains a CCS fuel and a disposal cost of 0, it should adjust the "CO2_content_tons_per_MMBtu" value but not the "Cost_per_MMBtu" value.
    def test_with_ccs_fuel_and_zero_disposal_cost(self):
        # Arrange
        row = pd.Series(
            {
                "Fuel": "naturalgas_ccs90",
                "Cost_per_MMBtu": 10,
                "CO2_content_tons_per_MMBtu": 5,
            }
        )
        ccs_fuels = ["naturalgas_ccs90"]
        ccs_capture_rate = {"naturalgas_ccs90": 0.9}
        ccs_disposal_cost = 0

        # Act
        result = adjust_ccs_fuels(row, ccs_fuels, ccs_capture_rate, ccs_disposal_cost)

        # Assert
        assert result["Fuel"] == "naturalgas_ccs90"
        assert result["Cost_per_MMBtu"] == 10
        assert result["CO2_content_tons_per_MMBtu"] == 5 - (5 * 0.9)

    # If the function is called with a row that contains a CCS fuel that is not included in the "ccs_capture_rate" dict, it should raise a KeyError.
    def test_with_ccs_fuel_not_in_capture_rate(self):
        # Arrange
        row = pd.Series(
            {
                "Fuel": "naturalgas_ccs90",
                "Cost_per_MMBtu": 10,
                "CO2_content_tons_per_MMBtu": 5,
            }
        )
        ccs_fuels = ["naturalgas_ccs90"]
        ccs_capture_rate = {}
        ccs_disposal_cost = 50

        # Act & Assert
        with pytest.raises(KeyError):
            adjust_ccs_fuels(row, ccs_fuels, ccs_capture_rate, ccs_disposal_cost)

    # If the function is called with a row that contains a CCS fuel and a disposal cost that is not specified in the settings, it should issue a warning and set the disposal cost to 0.
    def test_ccs_fuel_with_no_disposal_cost_fixed(self, caplog):
        caplog.set_level(logging.WARNING)
        # Arrange
        row = pd.Series(
            {
                "Fuel": "naturalgas_ccs",
                "Cost_per_MMBtu": 10,
                "CO2_content_tons_per_MMBtu": 5,
            }
        )

        # Act
        result = adjust_ccs_fuels(
            row,
            ccs_fuels=["naturalgas_ccs"],
            ccs_capture_rate={"naturalgas_ccs": 0.9},
            ccs_disposal_cost=None,
        )

        # Assert
        assert result["Fuel"] == "naturalgas_ccs"
        assert result["Cost_per_MMBtu"] == 10
        assert result["CO2_content_tons_per_MMBtu"] == 5 - (5 * 0.9)
        assert "You did not specify a CCS disposal cost" in caplog.text


class TestFuelCostTable:
    def test_fuel_cost_table_tdr(self):
        fuel_costs = pd.DataFrame(
            {
                "year": [2022, 2022],
                "price": [10, 20],
                "fuel": ["coal", "naturalgas"],
                "region": ["US", "US"],
                "full_fuel_name": ["US_coal", "US_naturalgas"],
            }
        )

        generators = pd.DataFrame(
            {
                "Fuel": [
                    "US_coal",
                    "US_naturalgas",
                    "US_naturalgas_ccs90",
                    "hydrogen",
                ]
            }
        )

        settings = {
            "model_year": 2022,
            "fuel_emission_factors": {"coal": 2.5, "naturalgas": 1.8, "hydrogen": 0},
            "ccs_fuel_map": {"naturalgas_ccs": "naturalgas_ccs90"},
            "ccs_capture_rate": {"naturalgas_ccs90": 0.9},
            "ccs_disposal_cost": 50,
            "carbon_tax": 20,
            "reduce_time_domain": True,
            "time_domain_days_per_period": 7,
            "time_domain_periods": 52,
            "aeo_fuel_scenarios": {
                "coal": "reference",
                "naturalgas": "reference",
            },
            "user_fuel_price": {"hydrogen": 20},
        }

        result = fuel_cost_table(fuel_costs, generators, settings)

        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == 52 * 7 * 24 + 1
        assert result.shape[1] == 4
        assert result.columns.tolist() == [
            "US_coal",
            "US_naturalgas",
            "US_naturalgas_ccs90",
            "hydrogen",
        ]
        assert np.allclose(result.iloc[0].tolist(), [2.5, 1.8, 1.8 - (1.8 * 0.9), 0])

    def test_fuel_cost_table_no_tdr(self):
        fuel_costs = pd.DataFrame(
            {
                "year": [2022, 2022],
                "price": [10, 20],
                "fuel": ["coal", "naturalgas"],
                "region": ["US", "US"],
                "full_fuel_name": ["US_coal", "US_naturalgas"],
            }
        )

        generators = pd.DataFrame(
            {
                "Fuel": [
                    "US_coal",
                    "US_naturalgas",
                    "hydrogen",
                ]
            }
        )

        settings = {
            "model_year": 2022,
            "fuel_emission_factors": {"coal": 2.5, "naturalgas": 1.8, "hydrogen": 0},
            "aeo_fuel_scenarios": {
                "coal": "reference",
                "naturalgas": "reference",
            },
            "user_fuel_price": {"hydrogen": 20},
        }

        result = fuel_cost_table(fuel_costs, generators, settings)

        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == 8761
        assert result.shape[1] == 3
        assert result.columns.tolist() == [
            "US_coal",
            "US_naturalgas",
            "hydrogen",
        ]
        assert np.allclose(result.iloc[0].tolist(), [2.5, 1.8, 0])

    def test_fuel_cost_table_warning(self, caplog):
        fuel_costs = pd.DataFrame(
            {
                "year": [2022, 2022],
                "price": [10, 20],
                "fuel": ["coal", "naturalgas"],
                "region": ["US", "US"],
                "full_fuel_name": ["US_coal", "US_naturalgas"],
            }
        )

        generators = pd.DataFrame(
            {
                "Fuel": [
                    "US_coal",
                    "US_naturalgas",
                    "hydrogen",
                ]
            }
        )

        settings = {
            "model_year": 2022,
            "fuel_emission_factors": {"coal": 2.5, "naturalgas": 1.8},
            "aeo_fuel_scenarios": {
                "coal": "reference",
                "naturalgas": "reference",
            },
            "user_fuel_price": {"hydrogen": 20},
        }

        result = fuel_cost_table(fuel_costs, generators, settings)

        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == 8761
        assert result.shape[1] == 3
        assert result.columns.tolist() == [
            "US_coal",
            "US_naturalgas",
            "hydrogen",
        ]
        assert np.allclose(result.iloc[0].tolist(), [2.5, 1.8, 0])
        assert "The user fuel" in caplog.text
