import numpy as np
import pandas as pd
import pytest

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
from powergenome.util import load_settings


def test_startup_fuel():
    df = pd.DataFrame({"technology": ["tech1", "tech2", "tech1"]})
    settings = {
        "startup_fuel_use": {"tech1": 5},
        "eia_atb_tech_map": {"tech1": ["tech1a"]},
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
        "eia_atb_tech_map": {"coal": ["coal"]},
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
    settings = {
        "transmission_investment_cost": {
            "spur": {"wacc": 0.05, "investment_years": 20, "capex_mw_mile": 1000}
        }
    }
    out = calculate_transmission_inv_cost(df.copy(), settings)
    assert "spur_capex" in out.columns and "spur_inv_mwyr" in out.columns


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
    settings = {"transmission_investment_cost": {"use_total": False}}
    out = add_transmission_inv_cost(df.copy(), settings)
    assert "Inv_Cost_per_MWyr" in out.columns


def test_add_dg_resources():
    import powergenome.generators as generators_mod

    # Minimal stub for make_distributed_gen_profiles (catch-all signature)
    def fake_make_distributed_gen_profiles(*args, **kwargs):
        return pd.DataFrame({"A": [0.5, 1.0], "B": [0.2, 0.8]})

    old_func = generators_mod.make_distributed_gen_profiles
    generators_mod.make_distributed_gen_profiles = fake_make_distributed_gen_profiles
    try:
        out = add_dg_resources({"model_year": 2020}, pd.DataFrame())
        assert "technology" in out.columns and "region" in out.columns
    finally:
        generators_mod.make_distributed_gen_profiles = old_func


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


def test_label_retired_gens():
    df = pd.DataFrame({"operating_year": [2000, 2010], "retirement_year": [2025, 2015]})
    out = label_retired_gens(df.copy(), 2010, 2020)
    assert "operating" in out.columns and "period_retired" in out.columns


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
    with pytest.raises(ValueError):
        check_cluster_cols(df2, ["col1"])


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
        settings = load_settings("tests/test_system/settings")
        settings["RESOURCE_GROUPS"] = "tests/test_system/test_data/resource_groups"
        settings["data_location"] = "tests/test_system/test_data"

        if isinstance(settings["model_year"], list):
            settings["model_year"] = settings["model_year"][0]
            settings["model_first_planning_year"] = settings[
                "model_first_planning_year"
            ][0]
        return settings

    def test_cluster_existing_generators(self):
        settings = self.load_settings()
        self.gc = GeneratorClusters(
            data_location=settings["data_location"],
            generation_table=settings["generation_table"],
            settings=settings,
            resource_heat_rate_table=settings["resource_heat_rate_table"],
            resource_cost_table=settings["resource_cost_table"],
            multi_period=True,
            include_retired_cap=True,
        )
        existing_gen = self.gc.create_region_technology_clusters()

    def test_create_new_generators(self):
        settings = self.load_settings()
        self.gc = GeneratorClusters(
            data_location=settings["data_location"],
            generation_table=settings["generation_table"],
            settings=settings,
            resource_heat_rate_table=settings["resource_heat_rate_table"],
            resource_cost_table=settings["resource_cost_table"],
            multi_period=True,
            include_retired_cap=True,
        )
        new_gen = self.gc.create_new_generators()

    def test_create_all_generators(self, tmp_path):
        settings = self.load_settings()
        extra_outputs_path = tmp_path / "extra_outputs"
        extra_outputs_path.mkdir(
            parents=True, exist_ok=True
        )  # Ensure the directory exists
        settings["extra_outputs_path"] = extra_outputs_path
        self.gc = GeneratorClusters(
            data_location=settings["data_location"],
            generation_table=settings["generation_table"],
            settings=settings,
            resource_heat_rate_table=settings["resource_heat_rate_table"],
            resource_cost_table=settings["resource_cost_table"],
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
