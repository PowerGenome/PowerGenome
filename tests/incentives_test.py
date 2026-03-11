import pandas as pd
import pytest

from powergenome.GenX import create_incentive_inputs


def _sample_gen_data():
    return pd.DataFrame(
        {
            "Resource": [
                "Z1_onshore_wind_1",
                "Z1_natural_gas_ccs100_1",
                "Z1_nuclear_1",
            ],
            "technology": [
                "Onshore Wind",
                "Natural Gas CCS100",
                "Nuclear",
            ],
        }
    )


def test_policy_description_override_and_type_normalization():
    gen = _sample_gen_data()

    settings = {
        "investment_incentives": {
            "Inv_Incentive_1": {
                "value": 0.3,
                "description": "ITC for Wind",
                "technologies": ["wind"],
            }
        },
        "production_incentives": {
            "Prod_Incentive_1": {
                "value": 85,
                "type": "tonne co2",
                "description": "45Q-CCS",
                "technologies": ["CCS"],
            }
        },
    }

    out = create_incentive_inputs(gen, settings)

    inv_df = out["investment_incentive"]
    prod_df = out["production_incentive"]
    assert inv_df.loc[0, "PolicyDescription"] == "ITC for Wind"
    assert prod_df.loc[0, "PolicyDescription"] == "45Q-CCS"
    assert prod_df.loc[0, "Production_Type"] == "Tonne_CO2"


def test_resource_filtering_only_qualifying_rows():
    gen = _sample_gen_data()

    settings = {
        "investment_incentives": {
            "Inv_Incentive_1": {"value": 0.2, "technologies": ["wind"]},
            "Inv_Incentive_2": {"value": 0.1, "technologies": ["geothermal"]},
        },
        "production_incentives": {
            "Prod_Incentive_1": {
                "value": 10,
                "type": "MWh",
                "technologies": ["biomass"],
            }
        },
    }

    out = create_incentive_inputs(gen, settings)

    res_inv = out["resource_investment_incentive"]
    # Only wind qualifies for Inv_Incentive_1; others are zero and should be dropped
    assert set(res_inv["Resource"]) == {"Z1_onshore_wind_1"}
    assert (res_inv[["Inv_Incentive_1"]].sum(axis=1) > 0).all()


def test_incentive_numbering_validation():
    gen = _sample_gen_data()

    # Skip a number (1 and 3)
    bad_settings = {
        "investment_incentives": {
            "Inv_Incentive_1": {"value": 0.2, "technologies": ["wind"]},
            "Inv_Incentive_3": {"value": 0.2, "technologies": ["nuclear"]},
        }
    }
    with pytest.raises(ValueError):
        create_incentive_inputs(gen, bad_settings)

    # Duplicate numbers
    dup_settings = {
        "production_incentives": {
            "Prod_Incentive_1": {
                "value": 10,
                "type": "MWh",
                "technologies": ["wind"],
            },
            "Prod_Incentive_1": {
                "value": 12,
                "type": "MWh",
                "technologies": ["nuclear"],
            },
        }
    }
    # Python dict will overwrite the first key, so simulate duplicate by calling validator indirectly
    # Instead, test start at 1 requirement for production
    start_bad = {
        "production_incentives": {
            "Prod_Incentive_2": {
                "value": 10,
                "type": "MWh",
                "technologies": ["wind"],
            }
        }
    }
    with pytest.raises(ValueError):
        create_incentive_inputs(gen, start_bad)


def test_unsupported_production_type_and_tech_list_type():
    gen = _sample_gen_data()

    bad_type = {
        "production_incentives": {
            "Prod_Incentive_1": {
                "value": 10,
                "type": "kWh",  # unsupported
                "technologies": ["wind"],
            }
        }
    }
    with pytest.raises(ValueError):
        create_incentive_inputs(gen, bad_type)

    bad_list = {
        "investment_incentives": {
            "Inv_Incentive_1": {"value": 0.1, "technologies": "wind"}
        }
    }
    with pytest.raises(TypeError):
        create_incentive_inputs(gen, bad_list)


def test_deterministic_ordering_and_substring_match():
    gen = _sample_gen_data()

    settings = {
        # Intentionally swap order in settings; expect IDs 1 then 2 by suffix
        "investment_incentives": {
            "Inv_Incentive_2": {"value": 0.2, "technologies": ["nuclear"]},
            "Inv_Incentive_1": {"value": 0.3, "technologies": ["wind"]},
        },
        "production_incentives": {
            "Prod_Incentive_1": {
                "value": 85,
                "type": "Tonne_CO2",
                "technologies": ["CCS"],
            }
        },
    }

    out = create_incentive_inputs(gen, settings)
    inv_df = out["investment_incentive"].sort_values("Policy_ID").reset_index(drop=True)
    assert inv_df["Policy_ID"].tolist() == [1, 2]
    assert inv_df.loc[0, "Value"] == 0.3  # Inv_1
    assert inv_df.loc[1, "Value"] == 0.2  # Inv_2

    # Substring match: CCS matches "Natural Gas CCS100"
    res_prod = out["resource_production_incentive"]
    assert set(res_prod["Resource"]) == {"Z1_natural_gas_ccs100_1"}
