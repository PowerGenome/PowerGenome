"Test functions for adding CO2 costs to CCS generators"

from pathlib import Path

import numpy as np
import pandas as pd

from powergenome.co2_pipeline_cost import merge_co2_pipeline_costs
from powergenome.GenX import add_co2_costs_to_o_m

CWD = Path.cwd()
DATA_FOLDER = CWD / "tests" / "data" / "co2"


def test_merge_co2_costs():
    data = {
        "region": ["NY_Z_A", "NY_Z_A", "ERCOT"],
        "technology": ["other", "Coal_CCS90AvgCF_Mod", "Coal_CCS90AvgCF_Mod"],
        "Var_OM_Cost_per_MWh": [1, 1, 1],
        "Inv_Cost_per_MWyr": [10, 10000, 10000],
        "Fixed_OM_Cost_per_MWyr": [1, 1000, 1000],
        "Heat_Rate_MMBTU_per_MWh": [5000, 8000, 8000],
        "Fuel": ["No_fuel", "coal", "coal"],
    }
    df = pd.DataFrame(data)
    region_aggregations = {"ERCOT": ["ERC_WEST", "ERC_REST"]}
    co2_pipeline_filters = [
        {
            "technology": "Coal",
            "tech_detail": "CCS90AvgCF",
            "with_backbone": True,
            "percentile": 25,
        }
    ]
    fuel_emission_factors = {"coal": 0.09552}

    merge_df = merge_co2_pipeline_costs(
        df,
        DATA_FOLDER / "co2_pipeline_cost_percentiles.csv",
        co2_pipeline_filters,
        region_aggregations,
        fuel_emission_factors,
        target_usd_year=2020,
    )
    assert len(merge_df) == 3
    assert merge_df.loc[1:, :].notna().all().all()
    assert merge_df.loc[0, :].isna().any()

    co2_pipeline_filters = [
        {
            "technology": "Coal",
            "tech_detail": "CCS90AvgCF",
            "with_backbone": False,
            "percentile": 25,
        }
    ]

    merge_df = merge_co2_pipeline_costs(
        df,
        DATA_FOLDER / "co2_pipeline_cost_percentiles.csv",
        co2_pipeline_filters,
        region_aggregations,
        fuel_emission_factors,
        target_usd_year=2020,
    )
    assert len(merge_df) == 2


def test_add_co2_costs_genx():
    data = {
        "region": ["NY_Z_A", "NY_Z_A", "ERCOT"],
        "technology": ["other", "Coal_CCS90AvgCF_Mod", "Coal_CCS90AvgCF_Mod"],
        "Var_OM_Cost_per_MWh": [1, 1, 1],
        "Inv_Cost_per_MWyr": [10, 10000, 10000],
        "Fixed_OM_Cost_per_MWyr": [1, 1000, 1000],
        "Heat_Rate_MMBTU_per_MWh": [5000, 8000, 8000],
        "Fuel": ["No_fuel", "coal", "coal"],
    }
    co2_cost_data = {
        "co2_o_m_mw": [0, 2814, 471],
        "co2_pipeline_annuity_mw": [0, 8670, 1457],
        "co2_cost_mwh": [0, 19614, 17813],
    }
    df = pd.DataFrame(data)
    merge_df = pd.concat([df, pd.DataFrame(co2_cost_data)], axis=1)
    merge_df = add_co2_costs_to_o_m(merge_df)
    cost_cols = ["Var_OM_Cost_per_MWh", "Inv_Cost_per_MWyr", "Fixed_OM_Cost_per_MWyr"]
    assert np.allclose(df.loc[0, cost_cols].sum(), merge_df.loc[0, cost_cols].sum())
    assert (df.loc[1:, cost_cols].sum() < merge_df.loc[1:, cost_cols].sum()).all()
