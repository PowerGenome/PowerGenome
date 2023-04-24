"Test functions related to fuel assignment, price, and emissions"

import logging
import os
import numpy as np
import pytest

import powergenome
from powergenome.eia_opendata import fetch_fuel_prices, modify_fuel_prices
from powergenome.fuels import add_user_fuel_prices, add_carbon_tax, fuel_cost_table
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
                "new_tech_detail": "CCS",
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
            "biopower_ccs": "biomass_ccs",
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
            "Biomass": ["Biopower_Dedicated", "biopower_ccs"],
            "CC - multi shaft": [
                "ZeroCarbon_CCAvgCF",
                "ZCF_CCAvgCF",
                "naturalgas_ct",
                "naturalgas_ccccs",
            ],
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
