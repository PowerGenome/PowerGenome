"Test functions related to fuel assignment, price, and emissions"

import os
import numpy as np

from powergenome.fuels import add_user_fuel_prices, add_carbon_tax, fuel_cost_table
from powergenome.generators import GeneratorClusters
from powergenome.params import DATA_PATHS
from powergenome.util import init_pudl_connection

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


def test_fuel_labels_and_prices():
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

    df_base = add_user_fuel_prices(settings)

    for fuel in [
        "S_VACA_biomass",
        "PJM_Dom_biomass",
        "zerocarbonfuel1",
        "zerocarbonfuel2",
    ]:
        assert fuel in df_base["full_fuel_name"].unique()

    settings["target_usd_year"] = 2020
    df_inflate = add_user_fuel_prices(settings)
    assert (
        df_base.loc[df_base["fuel"] == "biomass", "price"].mean()
        < df_inflate.loc[df_inflate["fuel"] == "biomass", "price"].mean()
    )
    assert np.allclose(
        df_base.loc[df_base["fuel"] == "zerocarbonfuel1", "price"].mean(),
        df_inflate.loc[df_inflate["fuel"] == "zerocarbonfuel1", "price"].mean(),
    )

    gc = GeneratorClusters(
        pudl_engine, pudl_out, pg_engine, settings, current_gens=False
    )
    gens = gc.create_new_generators()
    assert gens["Fuel"].isna().any() == False
    assert gens["Fuel"].str.contains("ccs", case=False).any() == True

    fuel_table = fuel_cost_table(gc.fuel_prices, gens, settings)
    for r in ["PJM_Dom", "S_VACA"]:
        assert (
            fuel_table.loc[0, f"{r}_biomass"]
            == settings["fuel_emission_factors"]["biomass"]
        )
        assert np.allclose(
            fuel_table.loc[0, f"{r}_biomass_ccs"],
            settings["fuel_emission_factors"]["biomass"]
            * (1 - settings["ccs_capture_rate"]["biomass_ccs"]),
        )
    assert (fuel_table.loc[1:, :] == 0).any().any() == False
