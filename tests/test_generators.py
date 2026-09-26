import logging

import numpy as np
import pandas as pd
import pytest

from powergenome.database import initialize_data_manager
from powergenome.generators import (
    GeneratorClusters,
    add_dg_resources,
    add_fuel_labels,
    add_gen_age_column,
    add_resource_tags,
    add_transmission_inv_cost,
    apply_custom_gen_formula,
    calc_unit_cluster_values,
    calculate_transmission_inv_cost,
    check_cluster_cols,
    cluster_existing_generators,
    create_resource_label,
    energy_storage_mwh,
    fill_num_regional_clusters,
    group_technologies,
    label_retired_gens,
    label_small_hydro,
    startup_fuel,
    startup_nonfuel_costs,
)
from powergenome.GenX import check_retirement_budget
from powergenome.settings import Settings, resolve_settings_to_year


def test_startup_fuel():
    df = pd.DataFrame({"technology": ["tech1", "tech2", "tech1"]})
    settings = {
        "startup_fuel_use": {"tech1": 5},
    }
    out = startup_fuel(df.copy(), settings)
    assert "Start_Fuel_MMBTU_per_MW" in out.columns
    assert (out.loc[out["technology"] == "tech1", "Start_Fuel_MMBTU_per_MW"] == 5).all()


def test_startup_nonfuel_costs():
    df = pd.DataFrame({"technology": ["a", "b"]})
    settings = {
        "startup_vom_costs_mw": {"x": 1},
        "startup_vom_costs_usd_year": 2020,
        "startup_costs_type": "startup_costs_per_cold_start",
        "startup_costs_per_cold_start": {"x": 2},
        "startup_costs_per_cold_start_usd_year": 2020,
        "existing_startup_costs_tech_map": {"a": "x"},
        "new_build_startup_costs": {"b": "x"},
        "target_usd_year": 2020,
        "data_location": "tests/test_system/test_data",
        "dollar_year_table": "cpi_test_data.csv",
    }
    # Initialize DataManager for accessing dollar year data
    initialize_data_manager(settings, settings["data_location"])
    out = startup_nonfuel_costs(df.copy(), settings)
    assert "Start_Cost_per_MW" in out.columns
    assert (out["Start_Cost_per_MW"] >= 0).all()


def test_group_technologies():
    df = pd.DataFrame(
        {
            "technology": ["a", "b", "c"],
            "technology_description": ["a", "b", "c"],
            "model_region": ["r1", "r1", "r2"],
        }
    )
    tech_groups = {"agg": ["a", "b"]}
    regional_no_grouping = {"r2": ["c"]}
    out = group_technologies(df.copy(), tech_groups, regional_no_grouping)
    assert "technology" in out.columns
    assert (out.loc[out["model_region"] == "r2", "technology"] == "c").all()
    assert (out.loc[out["model_region"] == "r1", "technology"].isin(["agg"])).all()


def test_label_small_hydro():
    df = pd.DataFrame(
        {
            "plant_id_eia": [1, 2, 3],
            "technology_description": ["Conventional Hydroelectric"] * 3,
            "model_region": ["A", "B", "A"],
            "capacity_mw": [5, 15, 3],
        }
    )
    settings = {
        "small_hydro": True,
        "small_hydro_regions": ["A"],
        "small_hydro_mw": 10,
        "model_regions": ["A", "B"],
        "region_aggregations": {},
        "capacity_col": "capacity_mw",
    }
    out = label_small_hydro(df.copy(), settings)
    assert "Small Hydroelectric" in out["technology_description"].values


def test_calc_unit_cluster_values():
    df = pd.DataFrame(
        {
            "cluster": [1, 1, 2],
            "capacity_mw": [10, 20, 30],
            "capacity_mwh": [5, 10, 15],
            "heat_rate_mmbtu_mwh": [8, 9, 10],
            "fom_per_mwyr": [1, 2, 3],
            "vom_per_mwh": [0.1, 0.2, 0.3],
            "operating": [True, True, True],
        }
    )
    out = calc_unit_cluster_values(df, "capacity_mw", technology="tech", clustered=True)
    assert "cluster" in out.columns
    assert "capacity_mw" in out.columns


def test_add_resource_tags():
    df = pd.DataFrame({"technology": ["nuclear", "coal"], "region": ["A", "B"]})
    model_tag_values = {"tag1": {"nuclear": 1, "coal": 2}}
    regional_tag_values = {"A": {"tag2": {"nuclear": 3}}}
    out = add_resource_tags(
        df.copy(),
        model_tag_values,
        regional_tag_values,
        model_tag_names=["tag1", "tag2"],
    )
    assert "tag1" in out.columns and "tag2" in out.columns
    assert out.loc[0, "tag2"] == 3


def test_add_fuel_labels():
    df = pd.DataFrame({"technology": ["coal"], "region": ["A"]})
    fuel_prices = pd.DataFrame(
        {
            "year": [2020],
            "price": [1],
            "fuel": ["coal"],
            "region": ["A"],
            "scenario": ["REF"],
            "full_fuel_name": ["A_REF_coal"],
        }
    )
    settings = {
        "tech_fuel_map": {"coal": "coal"},
        "fuel_scenarios": {"coal": "REF"},
        "fuel_region_map": {"A": ["A"]},
        "model_year": 2020,
        "user_fuel_price": {},
    }
    out = add_fuel_labels(df.copy(), fuel_prices, settings)
    assert "Fuel" in out.columns
    assert out.loc[0, "Fuel"].startswith("A_REF_coal") or out.loc[0, "Fuel"] == "coal"


def test_calculate_transmission_inv_cost():
    df = pd.DataFrame({"region": ["A"], "spur_miles": [10]})
    legacy_spec = {
        "spur": {"wacc": 0.05, "investment_years": 20, "capex_mw_mile": 1000}
    }
    out = calculate_transmission_inv_cost(
        df.copy(),
        interconnect_capex_spec=None,  # trigger legacy path
        legacy_transmission_cost=legacy_spec,
    )
    assert "spur_capex" in out.columns and "spur_inv_mwyr" in out.columns


def test_calculate_transmission_inv_cost_existing_capex_annuity():
    # Existing capex with zero annuity should be annualized even if spec assigns no new capex
    df = pd.DataFrame(
        {
            "region": ["A"],
            "technology": ["battery_storage"],
            "interconnect_capex_mw": [1000],
            "interconnect_annuity": [0],
            "wacc_real": [0.05],
            "cap_recovery_years": [20],
        }
    )
    out = calculate_transmission_inv_cost(
        df.copy(), interconnect_capex_spec={}, legacy_transmission_cost=None
    )
    annuity = (0.05 * 1000) / (1 - (1 + 0.05) ** -20)
    assert np.isclose(out.loc[0, "interconnect_annuity"], annuity)


def test_calculate_transmission_inv_cost_scalar():
    df = pd.DataFrame(
        {
            "region": ["A", "B"],
            "technology": ["onshore_wind", "battery_storage"],
            "wacc_real": [0.05, 0.05],
            "cap_recovery_years": [20, 20],
        }
    )
    out = calculate_transmission_inv_cost(df.copy(), interconnect_capex_spec=50000)
    assert (out["interconnect_capex_mw"] == 50000).all()
    annuity = (0.05 * 50000) / (1 - (1 + 0.05) ** -20)
    assert np.isclose(out.loc[0, "interconnect_annuity"], annuity)


def test_calculate_transmission_inv_cost_region_only():
    spec = {"fallback_capex_mw": 10000, "by_region": {"A": 20000}}
    df = pd.DataFrame(
        {
            "region": ["A", "B"],
            "technology": ["solar_pv", "solar_pv"],
            "wacc_real": [0.05, 0.05],
            "cap_recovery_years": [25, 25],
        }
    )
    out = calculate_transmission_inv_cost(df.copy(), interconnect_capex_spec=spec)
    assert out.loc[0, "interconnect_capex_mw"] == 20000
    assert out.loc[1, "interconnect_capex_mw"] == 10000
    # annuity computed
    annuity_A = (0.05 * 20000) / (1 - (1 + 0.05) ** -25)
    annuity_B = (0.05 * 10000) / (1 - (1 + 0.05) ** -25)
    assert np.isclose(out.loc[0, "interconnect_annuity"], annuity_A)
    assert np.isclose(out.loc[1, "interconnect_annuity"], annuity_B)


def test_calculate_transmission_inv_cost_explicit_schema_precedence():
    spec = {
        "fallback_capex_mw": 9000,
        "by_technology": {"wind": 20000, "offshore_wind": 30000},
        "by_region": {"A": 7000},
        "by_region_technology": {"A": {"offshore_wind": 25000}},
    }
    df = pd.DataFrame(
        {
            "region": ["A", "A", "B", "C"],
            "technology": [
                "offshore_wind_fixed_mid",
                "battery_storage",
                "wind_onshore_mid",
                "solar_pv",
            ],
            "wacc_real": [0.05, 0.05, 0.05, 0.05],
            "cap_recovery_years": [20, 20, 20, 20],
        }
    )

    out = calculate_transmission_inv_cost(df.copy(), interconnect_capex_spec=spec)
    assert out.loc[0, "interconnect_capex_mw"] == 25000
    assert out.loc[1, "interconnect_capex_mw"] == 7000
    assert out.loc[2, "interconnect_capex_mw"] == 20000
    assert out.loc[3, "interconnect_capex_mw"] == 9000


def test_calculate_transmission_inv_cost_explicit_schema_unknown_keys_error():
    spec = {
        "fallback_capex_mw": 10000,
        "default": 5000,
    }
    df = pd.DataFrame(
        {
            "region": ["A"],
            "technology": ["wind_onshore_mid"],
            "wacc_real": [0.05],
            "cap_recovery_years": [20],
        }
    )

    with pytest.raises(ValueError, match="explicit schema only allows keys"):
        calculate_transmission_inv_cost(df.copy(), interconnect_capex_spec=spec)


def test_calculate_transmission_inv_cost_legacy_default_raises():
    spec = {"default": 10000, "A": 20000}
    df = pd.DataFrame(
        {
            "region": ["A", "B"],
            "technology": ["solar_pv", "solar_pv"],
            "wacc_real": [0.05, 0.05],
            "cap_recovery_years": [25, 25],
        }
    )

    with pytest.raises(ValueError, match="explicit schema only allows keys"):
        calculate_transmission_inv_cost(df.copy(), interconnect_capex_spec=spec)


def test_calculate_transmission_inv_cost_tech_only_precedence():
    # Shortest-first precedence: 'wind' applied first then 'offshore_wind' overwrites matching rows
    spec = {
        "fallback_capex_mw": 10000,
        "by_technology": {"wind": 20000, "offshore_wind": 30000},
    }
    df = pd.DataFrame(
        {
            "region": ["A", "A", "A"],
            "technology": [
                "wind_onshore_mid",
                "offshore_wind_fixed_mid",
                "battery_storage",
            ],
            "wacc_real": [0.06, 0.06, 0.06],
            "cap_recovery_years": [30, 30, 30],
        }
    )
    out = calculate_transmission_inv_cost(df.copy(), interconnect_capex_spec=spec)
    # battery gets default
    assert out.loc[2, "interconnect_capex_mw"] == 10000
    # onshore wind matched only 'wind'
    assert out.loc[0, "interconnect_capex_mw"] == 20000
    # offshore wind should be overwritten to 30000
    assert out.loc[1, "interconnect_capex_mw"] == 30000


def test_calculate_transmission_inv_cost_region_to_tech_nested():
    spec = {
        "fallback_capex_mw": 8000,
        "by_region_technology": {
            "A": {"battery": 6000, "offshore_wind": 25000},
            "B": {"battery": 7000},
        },
    }
    df = pd.DataFrame(
        {
            "region": ["A", "A", "B", "C"],
            "technology": [
                "battery_storage",
                "offshore_wind_floating_mid",
                "battery_storage",
                "solar_pv",
            ],
            "wacc_real": [0.05, 0.05, 0.05, 0.05],
            "cap_recovery_years": [15, 15, 15, 15],
        }
    )
    out = calculate_transmission_inv_cost(df.copy(), interconnect_capex_spec=spec)
    assert out.loc[0, "interconnect_capex_mw"] == 6000
    assert out.loc[1, "interconnect_capex_mw"] == 25000
    assert out.loc[2, "interconnect_capex_mw"] == 7000
    # Region C falls back to default
    assert out.loc[3, "interconnect_capex_mw"] == 8000


def test_calculate_transmission_inv_cost_tech_to_region_nested():
    # Technology defaults can be refined by region-technology overrides.
    spec = {
        "fallback_capex_mw": 5000,
        "by_technology": {"wind": 14000, "battery": 5000},
        "by_region_technology": {
            "A": {"wind": 15000},
            "C": {"battery": 5500},
        },
    }
    df = pd.DataFrame(
        {
            "region": ["A", "B", "C", "D"],
            "technology": [
                "offshore_wind_fixed_mid",
                "wind_onshore_mid",
                "battery_storage",
                "battery_storage",
            ],
            "wacc_real": [0.05, 0.05, 0.05, 0.05],
            "cap_recovery_years": [20, 20, 20, 20],
        }
    )
    out = calculate_transmission_inv_cost(df.copy(), interconnect_capex_spec=spec)
    assert out.loc[0, "interconnect_capex_mw"] == 15000  # offshore wind matches wind
    assert out.loc[1, "interconnect_capex_mw"] == 14000
    assert out.loc[2, "interconnect_capex_mw"] == 5500
    assert (
        out.loc[3, "interconnect_capex_mw"] == 5000
    )  # battery row region D falls back to default


def test_calculate_transmission_inv_cost_mixed_error():
    spec = {"A": 10000, "wind": 20000}
    df = pd.DataFrame(
        {
            "region": ["A"],
            "technology": ["wind_onshore_mid"],
            "wacc_real": [0.05],
            "cap_recovery_years": [20],
        }
    )
    with pytest.raises(ValueError, match="explicit schema only allows keys"):
        calculate_transmission_inv_cost(df.copy(), interconnect_capex_spec=spec)


def test_add_transmission_inv_cost():
    df = pd.DataFrame(
        {
            "Inv_Cost_per_MWyr": [100],
            "spur_inv_mwyr": [10],
            "offshore_spur_inv_mwyr": [5],
            "tx_inv_mwyr": [0],
            "interconnect_annuity": [0],
        }
    )
    out = add_transmission_inv_cost(df.copy(), use_total=True)
    assert "Inv_Cost_per_MWyr" in out.columns
    assert out.loc[0, "Inv_Cost_per_MWyr"] == 115  # 100 + 10 + 5


def test_add_transmission_inv_cost_calc_annuity():
    df = pd.DataFrame(
        {
            "Inv_Cost_per_MWyr": [100, 100],
            "spur_inv_mwyr": [10, 0],
            "tx_inv_mwyr": [0, 0],
            "interconnect_capex_mw": [0, 1000],
            "wacc_real": [0.05, 0.05],
            "cap_recovery_years": [20, 20],
        }
    )
    out = add_transmission_inv_cost(df.copy(), use_total=True)
    assert "Inv_Cost_per_MWyr" in out.columns
    assert out.loc[0, "Inv_Cost_per_MWyr"] == 110  # 100 + 10
    # Annuity for 1000 capex at 5% over 20 years
    annuity = (0.05 * 1000) / (1 - (1 + 0.05) ** -20)
    assert np.isclose(out.loc[1, "Inv_Cost_per_MWyr"], 100 + annuity)


def test_add_dg_resources():
    import powergenome.generators as generators_mod

    # Minimal stub for make_distributed_gen_profiles (catch-all signature)
    def fake_make_distributed_gen_profiles(*args, **kwargs):
        return pd.DataFrame({"A": [0.5, 1.0], "B": [0.2, 0.8]})

    old_func = generators_mod.make_distributed_gen_profiles
    generators_mod.make_distributed_gen_profiles = fake_make_distributed_gen_profiles
    try:
        out = add_dg_resources(
            model_regions=["A", "B"],
            region_aggregations=None,
            model_year=2020,
            gen_df=pd.DataFrame(),
        )
        assert "technology" in out.columns and "region" in out.columns
    finally:
        generators_mod.make_distributed_gen_profiles = old_func


def test_add_dg_resources_no_profiles(monkeypatch):
    import powergenome.generators as generators_mod

    # Return empty profiles to simulate skip path
    monkeypatch.setattr(
        generators_mod,
        "make_distributed_gen_profiles",
        lambda *args, **kwargs: pd.DataFrame(),
    )

    out = add_dg_resources(
        model_regions=["A"],
        region_aggregations=None,
        model_year=2020,
        gen_df=pd.DataFrame(),
    )
    assert "profile" in out.columns


def test_add_dg_resources_no_capacity(monkeypatch):
    import powergenome.generators as generators_mod

    # Provide a basic profile but no capacity
    monkeypatch.setattr(
        generators_mod,
        "make_distributed_gen_profiles",
        lambda *args, **kwargs: pd.DataFrame({"A": [0.5, 0.5]}),
    )
    monkeypatch.setattr(
        generators_mod,
        "get_distributed_gen_capacity",
        lambda *args, **kwargs: pd.DataFrame(columns=["region", "capacity_mw"]),
    )

    out = add_dg_resources(
        model_regions=["A"],
        region_aggregations=None,
        model_year=2020,
        gen_df=pd.DataFrame(),
    )
    assert "profile" in out.columns
    # Ensure no DG rows were added
    assert (
        (out.get("technology") != "distributed_generation").all()
        if "technology" in out.columns
        else True
    )


def test_energy_storage_mwh():
    df = pd.DataFrame({"tech": ["bat"], "cap": [10], "energy": [0], "region": ["A"]})
    energy_storage_duration = {"bat": 4}
    out = energy_storage_mwh(
        df.copy(), energy_storage_duration, "tech", "cap", "energy"
    )
    assert out.loc[0, "energy"] == 40


def test_fill_num_regional_clusters():
    num_clusters = {"a": 2, "b": 3}
    model_regions = ["R1", "R2"]
    alt_num_clusters = {"R2": {"b": 5}}
    out = fill_num_regional_clusters(num_clusters, model_regions, alt_num_clusters)
    assert out["R1"]["b"] == 3 and out["R2"]["b"] == 5


def _retirement_year_fixture():
    """Five units covering every retirement_year case: retiring after the period,
    no planned retirement (blank), retired before the period, retired within the
    period, and a blank retirement year with no operating year."""
    return pd.DataFrame(
        {
            "plant_id": [1, 2, 3, 4, 5],
            "generator_id": ["A", "B", "C", "D", "E"],
            "capacity_mw": [100.0, 200.0, 300.0, 400.0, 500.0],
            "capacity_mwh": [0.0] * 5,
            "operating_year": [2000, 2001, 2002, np.nan, 2010],
            "retirement_year": [2050, np.nan, 2020, np.nan, 2028],
            "heat_rate_mmbtu_mwh": [8.0, 9.0, 10.0, 11.0, 12.0],
            "fom_per_mwyr": [50, 60, 70, 80, 90],
            "vom_per_mwh": [0.0] * 5,
            "cluster": [1] * 5,
            "Resource": ["region_tech_1"] * 5,
        }
    )


def test_label_retired_gens():
    df = pd.DataFrame({"operating_year": [2000, 2010], "retirement_year": [2025, 2015]})
    out = label_retired_gens(df.copy(), 2010, 2020)
    assert "operating" in out.columns and "period_retired" in out.columns
    # retiring after the period end keeps the unit operating; retiring inside the
    # period marks it retired and drops it from the operating set
    assert list(out["operating"]) == [True, False]
    assert list(out["period_retired"]) == [False, True]


def test_label_retired_gens_blank_retirement_year_stays_operating():
    """A blank ``retirement_year`` means no planned retirement, not "drop the unit"."""
    out = label_retired_gens(_retirement_year_fixture(), start_year=2025, end_year=2030)

    # plant 3 retired in 2020, before the period; plant 5 retires within it
    assert list(out["operating"]) == [True, True, False, True, False]
    assert list(out["period_retired"]) == [False, False, False, False, True]
    # the blank values are left alone rather than filled with a sentinel year
    assert out.loc[out["plant_id"].isin([2, 4]), "retirement_year"].isna().all()


def test_blank_retirement_year_capacity_is_not_lost():
    """Regression test for units silently disappearing from ``Existing_Cap_MW``."""
    out = label_retired_gens(_retirement_year_fixture(), start_year=2025, end_year=2030)
    rollup = calc_unit_cluster_values(out, "capacity_mw")

    # 1,500 MW across the five units, minus the 300 MW retired before the period
    # and the 500 MW retired within it
    assert rollup["capacity_mw"].sum() == 700
    assert rollup["num_units"].sum() == 3


def test_label_retired_gens_without_retirement_year_column(caplog):
    caplog.set_level(logging.INFO)
    df = _retirement_year_fixture().drop(columns=["retirement_year"])
    out = label_retired_gens(df, start_year=2025, end_year=2030)

    assert out["operating"].all()
    assert not out["period_retired"].any()
    # the column is added as blanks so downstream retirement queries still work
    assert out["retirement_year"].isna().all()
    assert "no 'retirement_year' column" in caplog.text


def test_label_retired_gens_reports_blanks(caplog):
    caplog.set_level(logging.INFO)
    label_retired_gens(_retirement_year_fixture(), start_year=2025, end_year=2030)

    assert "have no 'retirement_year'" in caplog.text


def test_create_resource_label():
    s1 = pd.Series(["A", "B"])
    s2 = pd.Series(["x", "y"])
    out = create_resource_label(s1, s2, sep="-")
    assert list(out) == ["A-x", "B-y"]


def test_cluster_existing_generators():
    df = pd.DataFrame(
        {
            "model_region": ["R1", "R1", "R2"],
            "technology": ["tech", "tech", "tech2"],
            "capacity_mw": [10, 20, 30],
            "capacity_mwh": [5, 10, 15],
            "heat_rate_mmbtu_mwh": [8, 9, 10],
            "fom_per_mwyr": [1, 2, 3],
            "vom_per_mwh": [0.1, 0.2, 0.3],
            "operating": [True, True, True],
        }
    )
    num_clusters = {"R1": {"tech": 2}, "R2": {"tech2": 1}}
    results, all_gens = cluster_existing_generators(df, num_clusters)
    assert isinstance(results, pd.DataFrame)
    assert isinstance(all_gens, pd.DataFrame)


def test_check_cluster_cols():
    df = pd.DataFrame(
        {
            "operating": [True, True, False],
            "col1": [1, 2, np.nan],
            "col2": [np.nan, np.nan, np.nan],
            "plant_id": [1, 2, 3],
        }
    )
    # Should drop col2, keep col1
    out = check_cluster_cols(df, ["col1", "col2"])
    assert out == ["col1"]


def test_check_cluster_cols_missing_column():
    df = pd.DataFrame(
        {
            "operating": [True, True, False],
            "col1": [1, 2, np.nan],
            "plant_id": [1, 2, 3],
        }
    )
    with pytest.raises(KeyError):
        check_cluster_cols(df, ["missing_col"])


def test_check_cluster_cols_some_missing_values():
    df = pd.DataFrame(
        {
            "operating": [True, True, False],
            "col1": [1, 2, np.nan],
            "plant_id": [1, 2, 3],
        }
    )
    df2 = df.copy()
    df2.loc[0, "col1"] = np.nan
    # The function now fills missing values with mean instead of raising an error
    result = check_cluster_cols(df2, ["col1"])
    assert result == ["col1"]
    # Check that the missing value was filled with the mean (2.0)
    assert df2.loc[0, "col1"] == 2.0


def test_add_gen_age_column():
    df = pd.DataFrame({"operating_year": [2000, 2010]})
    out = add_gen_age_column(df.copy(), 2020)
    assert "age" in out.columns
    assert list(out["age"]) == [20, 10]


def test_apply_custom_gen_formula_add():
    df = pd.DataFrame({"technology": ["coal"], "fom_per_mwyr": [100], "age": [10]})
    formula_dict = {
        "coal": [
            {
                "attribute": "fom_per_mwyr",
                "formula": {"op": "add", "rate": 2, "multiplier": "age"},
            }
        ]
    }
    out = apply_custom_gen_formula(df.copy(), formula_dict)
    assert out.loc[0, "fom_per_mwyr"] == 100 + 2 * 10


def test_apply_custom_gen_formula_replace():
    df = pd.DataFrame({"technology": ["coal"], "fom_per_mwyr": [100], "age": [10]})
    formula_dict = {
        "coal": [
            {
                "attribute": "fom_per_mwyr",
                "formula": {"op": "replace", "rate": 2, "multiplier": "age"},
            }
        ]
    }
    out = apply_custom_gen_formula(df.copy(), formula_dict)
    assert out.loc[0, "fom_per_mwyr"] == 2 * 10


def test_apply_custom_gen_formula_missing_attribute():
    # If the DataFrame lacks the named attribute, KeyError should be raised
    import pandas as pd
    import pytest

    from powergenome.generators import apply_custom_gen_formula

    df = pd.DataFrame({"technology": ["coal"], "age": [10]})
    # 'fom_per_mwyr' column does not exist on purpose
    formula_dict = {
        "coal": [
            {
                "attribute": "fom_per_mwyr",
                "formula": {"op": "add", "rate": 2, "multiplier": "age"},
            }
        ]
    }
    with pytest.raises(KeyError):
        apply_custom_gen_formula(df.copy(), formula_dict)


def test_apply_custom_gen_formula_no_attribute():
    # If the DataFrame lacks the named attribute, KeyError should be raised
    import pandas as pd
    import pytest

    from powergenome.generators import apply_custom_gen_formula

    df = pd.DataFrame({"technology": ["coal"], "age": [10]})
    # 'fom_per_mwyr' column does not exist on purpose
    formula_dict = {
        "coal": [
            {
                # "attribute": "fom_per_mwyr",
                "formula": {"op": "add", "rate": 2, "multiplier": "age"},
            }
        ]
    }
    with pytest.raises(KeyError):
        apply_custom_gen_formula(df.copy(), formula_dict)


def test_apply_custom_gen_formula_unknown_operation():
    # If the 'op' value is not recognized, KeyError should be raised
    import pandas as pd
    import pytest

    from powergenome.generators import apply_custom_gen_formula

    df = pd.DataFrame({"technology": ["coal"], "fom_per_mwyr": [100], "age": [10]})
    # 'multiply' is not a supported op (only 'add' or 'replace')
    formula_dict = {
        "coal": [
            {
                "attribute": "fom_per_mwyr",
                "formula": {"op": "multiply", "rate": 2, "multiplier": "age"},
            }
        ]
    }
    with pytest.raises(AssertionError):
        apply_custom_gen_formula(df.copy(), formula_dict)


class TestGeneratorCluster:

    def load_settings(self):
        settings = Settings(config_path="tests/test_system/settings")
        settings["RESOURCE_GROUPS"] = "tests/test_system/test_data/resource_groups"
        settings["data_location"] = "tests/test_system/test_data"
        settings["cache_resource_clusters"] = False
        settings["use_resource_clusters_cache"] = False

        # Initialize DataManager before creating GeneratorClusters
        initialize_data_manager(settings, settings["data_location"])

        if isinstance(settings["model_year"], list):
            settings["model_year"] = settings["model_year"][0]
            settings["model_first_planning_year"] = settings[
                "model_first_planning_year"
            ][0]
        return settings

    def test_cluster_existing_generators(self):
        settings = self.load_settings()
        self.gc = GeneratorClusters(
            settings=settings,
            multi_period=True,
            include_retired_cap=True,
        )
        existing_gen = self.gc.create_region_technology_clusters()

    def test_create_new_generators(self):
        settings = self.load_settings()
        self.gc = GeneratorClusters(
            settings=settings,
            multi_period=True,
            include_retired_cap=True,
        )
        new_gen = self.gc.create_new_generators()
        assert (new_gen.Inv_Cost_per_MWyr > 0).all()

    def test_create_all_generators(self, tmp_path):
        settings = self.load_settings()
        extra_outputs_path = tmp_path / "extra_outputs"
        extra_outputs_path.mkdir(
            parents=True, exist_ok=True
        )  # Ensure the directory exists
        settings["extra_outputs_path"] = extra_outputs_path
        self.gc = GeneratorClusters(
            settings=settings,
            multi_period=False,
            include_retired_cap=False,
            sort_gens=True,
        )
        all_gen = self.gc.create_all_generators()
        assert isinstance(all_gen, pd.DataFrame)
        assert "Resource" in all_gen.columns
        assert "region" in all_gen.columns
        assert "profile" in all_gen.columns
        assert "Existing_Cap_MW" in all_gen.columns
        assert "Max_Cap_MW" in all_gen.columns
        assert "Fuel" in all_gen.columns
        assert "Existing_Cap_MW" in all_gen.columns


RETIRED_CAP_COLUMNS = [
    "Min_Retired_Cap_MW",
    "Min_Retired_Energy_Cap_MW",
    "Min_Retired_Charge_Cap_MW",
]


class TestMultiPeriodRetirement:
    """Retirement requirements written for later planning periods must never
    outweigh the capacity that GenX sees as available in the first period."""

    SETTINGS_PATH = "tests/test_system/settings"

    def load_settings(self):
        settings = Settings(config_path=self.SETTINGS_PATH)
        settings["RESOURCE_GROUPS"] = "tests/test_system/test_data/resource_groups"
        settings["data_location"] = "tests/test_system/test_data"
        settings["cache_resource_clusters"] = False
        settings["use_resource_clusters_cache"] = False
        initialize_data_manager(settings, settings["data_location"])
        return settings

    def _gen_data(self, settings, year, include_retired_cap, extra_outputs_path):
        """Build the generator dataframe for one planning period, the way the
        multi-period pipeline does."""
        year_settings = resolve_settings_to_year(settings, year)
        year_settings["extra_outputs_path"] = extra_outputs_path
        gc = GeneratorClusters(
            settings=year_settings,
            multi_period=True,
            include_retired_cap=include_retired_cap,
        )
        return gc.create_all_generators()

    def test_first_period_writes_no_retirement_requirements(self, tmp_path):
        settings = self.load_settings()
        # GenX compares retirement requirements with the capacity available in the
        # first period, so every case's first period must require nothing to retire
        # even when another case has already written requirements for later periods.
        gen_data = self._gen_data(settings, 2030, False, tmp_path)
        for col in RETIRED_CAP_COLUMNS:
            assert col in gen_data.columns
            assert (gen_data[col] == 0).all(), f"{col} is not zero in the first period"

    def test_later_periods_require_retiring_capacity(self, tmp_path):
        settings = self.load_settings()
        gen_data = self._gen_data(settings, 2040, True, tmp_path)
        assert (gen_data["Min_Retired_Cap_MW"] > 0).any()

    def test_requirements_are_floored_to_capacity_precision(self, tmp_path):
        settings = self.load_settings()
        gen_data = self._gen_data(settings, 2040, True, tmp_path)
        for col in RETIRED_CAP_COLUMNS:
            values = gen_data[col].astype(float)
            floored = np.floor(np.round(values, 7) * 10) / 10
            assert np.allclose(values, floored), f"{col} is finer than Existing_Cap_MW"

    def test_requirements_do_not_exceed_first_period_capacity(self, tmp_path):
        settings = self.load_settings()
        budget = {}
        period_1 = self._gen_data(settings, 2030, False, tmp_path)
        check_retirement_budget(period_1, "test_case", 1, budget)

        later = self._gen_data(settings, 2040, True, tmp_path)
        # Raises if the later period asks GenX to retire more capacity than the
        # first period makes available.
        check_retirement_budget(later, "test_case", 2, budget)

        available = dict(zip(period_1["Resource"], period_1["Existing_Cap_MW"]))
        required = later.groupby("Resource")["Min_Retired_Cap_MW"].sum()
        assert (required > 0).any()
        for resource, total in required.items():
            assert total <= available[resource] + 1e-6
