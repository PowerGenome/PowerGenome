"Test functions related to fuel assignment, price, and emissions"

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import powergenome
from powergenome.database import initialize_data_manager
from powergenome.fuels import (
    add_carbon_tax,
    add_user_fuel_prices,
    adjust_ccs_fuels,
    fetch_fuel_prices,
    fuel_cost_table,
    modify_fuel_prices,
)
from powergenome.generators import GeneratorClusters
from powergenome.params import DATA_PATHS
from powergenome.settings import Settings

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


@pytest.fixture()
def fuel_settings():
    settings = Settings(config_path="tests/test_system/settings")
    settings["RESOURCE_GROUPS"] = "tests/test_system/test_data/resource_groups"
    settings["data_location"] = "tests/test_system/test_data"
    settings["cache_resource_clusters"] = False
    settings["use_resource_clusters_cache"] = False
    settings_modifications = {
        "modified_new_resources": {
            "ZCF_CombinedCycle1": {
                "new_technology": "ZCF",
                "new_tech_detail": "CCAvgCF",
                "new_cost_case": "Advanced",
                "technology": "NaturalGas",
                "tech_detail": "2-on-1 Combined Cycle (H-Frame)",
                "cost_case": "Advanced",
                "size_mw": 500,
            },
            "ZCF_CombinedCycle2": {
                "new_technology": "ZeroCarbon",
                "new_tech_detail": "CCAvgCF",
                "new_cost_case": "Advanced",
                "technology": "NaturalGas",
                "tech_detail": "2-on-1 Combined Cycle (H-Frame)",
                "cost_case": "Advanced",
                "size_mw": 500,
            },
            "biopower_ccs": {
                "new_technology": "Biopower",
                "new_tech_detail": "DedicatedCCS",
                "new_cost_case": "Moderate",
                "technology": "NaturalGas",
                "tech_detail": "2-on-1 Combined Cycle (F-Frame) 97% CCS",
                "cost_case": "Moderate",
                "size_mw": 500,
            },
        },
        "resource_tech_map": {
            "Biomass": ["Biopower_Dedicated"],
            "Zero Carbon": ["ZCF"],
        },
        "new_resources": [
            ["NaturalGas", "Combustion Turbine (F-Frame)", "Moderate", 100],
            ["NaturalGas", "2-on-1 Combined Cycle (F-Frame) 97% CCS", "Moderate", 100],
        ],
        "user_fuel_price": {
            "zerocarbonfuel1": 14,
            "zerocarbonfuel2": 10,
            "biomass": {"p1_2": 10, "p3": 5, "p4": 5},
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
        # "model_regions": ["S_VACA", "PJM_Dom"],
        "model_year": 2030,
        "model_first_planning_year": 2025,
        # "cost_multiplier_region_map": {
        #     "SRCA": ["S_VACA"],
        #     "PJMD": ["PJM_Dom"],
        # },
        # "cost_multiplier_technology_map": {
        #     "Biomass": ["Biopower_Dedicated"],
        #     "CC - multi shaft": ["ZeroCarbon_CCAvgCF", "ZCF_CCAvgCF"],
        # },
        # "fuel_emission_factors": {
        #     "biomass": 0.1,  # Dummy value for biomass
        #     "naturalgas": 0.05306,
        # },
        "ccs_disposal_cost": 25,
        "data_location": "tests/test_system/test_data",
        "generation_table": "generators_test_data.csv",
        "plant_region_table": "plant_region_map_test_data.csv",
        "resource_heat_rate_table": "technology_heat_rates_test_data.csv",
        "resource_cost_table": "technology_costs_test_data.csv",
        "operational_constraints_table": "operational_constraints_test_data.csv",
    }
    updated_tech_fuel_map = {
        "ZeroCarbon_CCAvgCF_Advanced": "zerocarbonfuel1",
        "Zero Carbon": "zerocarbonfuel2",
        "Biomass": "biomass",
        "NaturalGas": "naturalgas",
    }
    settings.update(settings_modifications)
    settings["tech_fuel_map"].update(updated_tech_fuel_map)

    # Initialize DataManager with the test settings
    # Note: The DataManager is a singleton, so this will either initialize it
    # or reinitialize it with new settings
    initialize_data_manager(settings, settings["data_location"])

    return settings


def test_fuel_labels_and_prices(fuel_settings):
    df_base = add_user_fuel_prices(fuel_settings)

    for fuel in [
        "p1_2_biomass",
        "zerocarbonfuel1",
        "zerocarbonfuel2",
    ]:
        assert fuel in df_base["full_fuel_name"].unique()

    gc = GeneratorClusters(
        settings=fuel_settings,
    )
    gens = gc.create_new_generators()
    assert gens["Fuel"].isna().any() == False
    assert gens["Fuel"].str.contains("ccs", case=False).any() == True
    assert "zerocarbonfuel1" in gens["Fuel"].values

    fuel_table = fuel_cost_table(gc.fuel_prices, gens, fuel_settings)
    assert "zerocarbonfuel1" in fuel_table.columns.to_list()
    assert "p1_2_biomass_ccs" in fuel_table.columns.to_list()


def test_fetch_fuel_price_no_mappings(fuel_settings):
    region_names = fuel_settings.pop("fuel_series_region_names")
    series_names = fuel_settings.pop("fuel_series_names")
    scenario_names = fuel_settings.pop("fuel_series_scenario_names")

    fetch_fuel_prices(
        settings=fuel_settings,
    )


def test_fetch_fuel_price_errors(fuel_settings):
    # data_year = fuel_settings.pop("fuel_data_year")
    # with pytest.raises(KeyError):
    #     fetch_fuel_prices(
    #         data_location=fuel_settings["data_location"],
    #         table_name=fuel_settings["fuel_price_table"],
    #         settings=fuel_settings,
    #     )

    fuel_settings["fuel_data_year"] = 2000
    with pytest.raises(KeyError):
        fetch_fuel_prices(
            settings=fuel_settings,
        )


class TestRegionalFuelPriceMod:
    def test_no_modifications(self, fuel_settings):
        """Test that fuel prices remain unchanged when no adjustments are specified."""
        fuel_prices = fetch_fuel_prices(fuel_settings)
        mod_fuel_prices = modify_fuel_prices(
            fuel_prices,
            fuel_settings["fuel_region_map"],
            fuel_settings.get("regional_fuel_adjustments"),
        )
        assert np.allclose(fuel_prices["price"].values, mod_fuel_prices["price"].values)

    def test_valid_modifications(self, fuel_settings):
        """Test that fuel price modifications work correctly for multiplication and addition."""
        fuel_prices = fetch_fuel_prices(fuel_settings)

        fuel_settings["regional_fuel_adjustments"] = {
            "p1_2": ["mul", 2],
            "p3": {"naturalgas": ["add", 1]},
        }

        mod_fuel_prices = modify_fuel_prices(
            fuel_prices,
            fuel_settings["fuel_region_map"],
            fuel_settings.get("regional_fuel_adjustments"),
        )

        assert np.isclose(
            fuel_prices.query("region == 'pacific'")["price"].mean(),
            mod_fuel_prices.query("region == 'p1_2'")["price"].mean() / 2,
        )

        assert np.isclose(
            fuel_prices.query("region == 'pacific' and fuel == 'naturalgas'")[
                "price"
            ].mean(),
            mod_fuel_prices.query("region == 'p3' and fuel == 'naturalgas'")[
                "price"
            ].mean()
            - 1,
        )

    def test_modifications_without_fuel_region_map(self, fuel_settings):
        """Test that fuel price modifications work without fuel_region_map."""
        fuel_prices = fetch_fuel_prices(fuel_settings)

        # Modify existing regions (pacific) without using fuel_region_map
        regional_fuel_adjustments = {
            "pacific": ["mul", 1.5],
            "mountain": {"coal": ["add", 0.5]},
        }

        original_pacific_price = fuel_prices.query("region == 'pacific'")[
            "price"
        ].mean()
        original_mountain_coal_price = fuel_prices.query(
            "region == 'mountain' and fuel == 'coal'"
        )["price"].mean()

        mod_fuel_prices = modify_fuel_prices(
            fuel_prices,
            None,  # No fuel_region_map
            regional_fuel_adjustments,
        )

        # Check that pacific prices were multiplied by 1.5
        assert np.isclose(
            mod_fuel_prices.query("region == 'pacific'")["price"].mean(),
            original_pacific_price * 1.5,
        )

        # Check that mountain coal price increased by 0.5
        assert np.isclose(
            mod_fuel_prices.query("region == 'mountain' and fuel == 'coal'")[
                "price"
            ].mean(),
            original_mountain_coal_price + 0.5,
        )

        # Check that original dataframe is unchanged (function should return a copy)
        assert np.isclose(
            fuel_prices.query("region == 'pacific'")["price"].mean(),
            original_pacific_price,
        )

    def test_modifications_without_fuel_region_map_invalid_region(self, fuel_settings):
        """Test that modifying non-existent region without fuel_region_map raises KeyError."""
        fuel_prices = fetch_fuel_prices(fuel_settings)

        regional_fuel_adjustments = {
            "nonexistent_region": ["mul", 1.5],
        }

        with pytest.raises(KeyError, match="not found in the fuel price table"):
            modify_fuel_prices(
                fuel_prices,
                None,
                regional_fuel_adjustments,
            )

    def test_invalid_operation_type(self, fuel_settings):
        """Test that invalid operation type raises KeyError."""
        fuel_prices = fetch_fuel_prices(fuel_settings)

        fuel_settings["regional_fuel_adjustments"] = {
            "p3": {"coal": ["ad", 1]},  # Invalid operation "ad"
        }

        with pytest.raises(KeyError):
            modify_fuel_prices(
                fuel_prices,
                fuel_settings["fuel_region_map"],
                fuel_settings.get("regional_fuel_adjustments"),
            )

        fuel_settings["regional_fuel_adjustments"] = {
            "p1_2": ["div", 2],  # Invalid operation "div"
        }

        with pytest.raises(KeyError):
            modify_fuel_prices(
                fuel_prices,
                fuel_settings["fuel_region_map"],
                fuel_settings.get("regional_fuel_adjustments"),
            )

    def test_invalid_data_type(self, fuel_settings):
        """Test that invalid data type raises TypeError."""
        fuel_prices = fetch_fuel_prices(fuel_settings)

        fuel_settings["regional_fuel_adjustments"] = {
            "p1_2": 1,  # Should be a list, not an integer
            "p3": {"coal": 1},
        }

        with pytest.raises(TypeError):
            modify_fuel_prices(
                fuel_prices,
                fuel_settings["fuel_region_map"],
                fuel_settings.get("regional_fuel_adjustments"),
            )

    def test_missing_fuel_region_map(self, fuel_settings):
        """Test that missing fuel_region_map raises KeyError."""
        fuel_prices = fetch_fuel_prices(fuel_settings)

        fuel_settings["regional_fuel_adjustments"] = {
            "p1_2": ["mul", 2],
            "p3": {"naturalgas": ["add", 1]},
        }

        with pytest.raises(KeyError):
            modify_fuel_prices(
                fuel_prices,
                None,
                fuel_settings.get("regional_fuel_adjustments"),
            )

    def test_invalid_fuel_name(self, fuel_settings):
        """Test that invalid fuel name raises KeyError."""
        fuel_prices = fetch_fuel_prices(fuel_settings)

        fuel_settings["regional_fuel_adjustments"] = {
            "p1_2": ["mul", 2],
            "p3": {"naturalga": ["add", 1]},  # Typo in fuel name
        }

        with pytest.raises(KeyError):
            modify_fuel_prices(
                fuel_prices,
                fuel_settings["fuel_region_map"],
                fuel_settings.get("regional_fuel_adjustments"),
            )

    def test_invalid_region_in_adjustments(self, fuel_settings):
        """Test that invalid region in regional_fuel_adjustments raises KeyError."""
        fuel_prices = fetch_fuel_prices(fuel_settings)

        fuel_settings["regional_fuel_adjustments"] = {
            "invalid_region": ["mul", 2],  # Region not in fuel_region_map
            "p3": {"naturalgas": ["add", 1]},
        }

        with pytest.raises(KeyError):
            modify_fuel_prices(
                fuel_prices,
                fuel_settings["fuel_region_map"],
                fuel_settings.get("regional_fuel_adjustments"),
            )

    def test_invalid_operator_in_fuel_specific_adjustment(self, fuel_settings):
        """Test that invalid operator in fuel-specific adjustment raises KeyError."""
        fuel_prices = fetch_fuel_prices(fuel_settings)

        fuel_settings["regional_fuel_adjustments"] = {
            "p1_2": ["mul", 2],
            "p3": {"naturalgas": ["invalid_op", 1]},  # Invalid operator
        }

        with pytest.raises(KeyError):
            modify_fuel_prices(
                fuel_prices,
                fuel_settings["fuel_region_map"],
                fuel_settings.get("regional_fuel_adjustments"),
            )


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
        caplog.set_level(logging.DEBUG)
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
        assert "You did not specify a fuel-modifying CCS disposal cost" in caplog.text


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
            "fuel_scenarios": {
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
            "fuel_scenarios": {
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
            "fuel_scenarios": {
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
        assert result.shape[1] == 3
        assert result.columns.tolist() == [
            "US_coal",
            "US_naturalgas",
            "hydrogen",
        ]
        assert np.allclose(result.iloc[0].tolist(), [2.5, 1.8, 0])
        assert "The user fuel" in caplog.text
