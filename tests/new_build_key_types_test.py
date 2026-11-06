"""Resource key matching between settings and the cost/heat rate tables.

YAML turns unquoted values such as ``1``, ``2030``, or ``1.5`` into numbers.
These tests check that new resources match table rows the same way whether the
settings value is quoted or not, and whatever type DuckDB infers for the column.
"""

import pandas as pd
import pytest

from powergenome import new_build
from powergenome.database import DataManager

COST_PARAMS = {
    "capex_mw": 1000.0,
    "fixed_o_m_mw": 10.0,
    "variable_o_m_mwh": 1.0,
    "wacc_real": 0.05,
}


def _cost_rows(technology, tech_detail, cost_case, scale=1.0, wacc_detail="*"):
    # wacc_detail=None keeps the wacc rows on tech_detail, so an all-numeric
    # tech_detail column is read by DuckDB as an integer column.
    wacc_detail = tech_detail if wacc_detail is None else wacc_detail
    rows = []
    for basis_year in (2030, 2031):
        for parameter, value in COST_PARAMS.items():
            rows.append(
                {
                    "technology": technology,
                    "tech_detail": (
                        tech_detail if parameter != "wacc_real" else wacc_detail
                    ),
                    "cost_case": cost_case,
                    "financial_case": "Market",
                    "basis_year": basis_year,
                    "parameter": parameter,
                    "parameter_value": value * scale,
                    "dollar_year": 2022,
                    "data_year": 2024,
                }
            )
    return rows


def _settings(new_resources, **kwargs):
    settings = {
        "new_resources": new_resources,
        "target_usd_year": 2022,
        "data_location": None,
        "dollar_year_table": "cpi",
        "model_year": 2031,
        "model_first_planning_year": 2030,
    }
    settings.update(kwargs)
    return settings


@pytest.fixture
def make_dm(tmp_path, monkeypatch):
    """Load cost/heat rate rows into a local DataManager used by new_build."""
    monkeypatch.setattr(
        new_build, "inflation_price_adjustment", lambda price, **kwargs: price
    )

    def _make(cost_rows, hr_rows=None):
        pd.DataFrame(cost_rows).to_csv(tmp_path / "costs.csv", index=False)
        tables = {"resource_cost_table": "costs.csv"}
        if hr_rows is not None:
            pd.DataFrame(hr_rows).to_csv(tmp_path / "heat_rates.csv", index=False)
            tables["resource_heat_rate_table"] = "heat_rates.csv"
        dm = DataManager()
        dm.initialize(tables, tmp_path)
        monkeypatch.setattr(new_build, "get_data", dm.get_data)
        return dm

    yield _make


def _row(costs, technology, tech_detail, cost_case, new_gen_type):
    # single_generator_row expects the merged cost + heat rate frame.
    for col in ("fixed_o_m_mwh", "capex_mwh", "heat_rate"):
        if col not in costs:
            costs = costs.assign(**{col: float("nan")})
    gen = new_build.single_generator_row(costs, new_gen_type, range(2030, 2032))
    assert gen.loc[0, "technology"] == technology
    assert gen.loc[0, "tech_detail"] == tech_detail
    assert gen.loc[0, "cost_case"] == cost_case
    return gen


@pytest.mark.parametrize("settings_value", [1, "1"])
def test_unquoted_int_tech_detail_in_mixed_text_column(make_dm, settings_value):
    """An int setting must not make DuckDB cast a mixed text column to numbers."""
    make_dm(
        _cost_rows("Wind", "1", "Moderate") + _cost_rows("Wind", "Class2", "Moderate")
    )
    costs = new_build.fetch_resource_costs(
        _settings([["Wind", settings_value, "Moderate", 1]])
    )

    gen = _row(costs, "Wind", "1", "Moderate", ("Wind", settings_value, "Moderate", 1))
    assert gen.loc[0, "capex_mw"] == pytest.approx(1000.0)
    assert gen.loc[0, "wacc_real"] == pytest.approx(0.05)


@pytest.mark.parametrize(
    "table_value, settings_value",
    [("2030", 2030), ("1.5", 1.5), ("1", 1), ("1", "1")],
)
def test_numeric_looking_text_values_match_unquoted_settings(
    make_dm, table_value, settings_value
):
    """Text table values match the YAML number that renders to the same text."""
    make_dm(
        _cost_rows("Wind", table_value, "Moderate")
        + _cost_rows("Wind", "Class2", "Moderate", scale=2)
    )
    costs = new_build.fetch_resource_costs(
        _settings([["Wind", settings_value, "Moderate", 1]])
    )

    gen = _row(
        costs, "Wind", table_value, "Moderate", ("Wind", settings_value, "Moderate", 1)
    )
    assert gen.loc[0, "capex_mw"] == pytest.approx(1000.0)


@pytest.mark.parametrize("settings_value", [1, "1"])
def test_all_numeric_csv_column_matches_quoted_and_unquoted(make_dm, settings_value):
    """DuckDB reads an all-numeric CSV column as BIGINT; both forms still match."""
    dm = make_dm(
        _cost_rows("Wind", 1, "Moderate", wacc_detail=None)
        + _cost_rows("Wind", 2, "Moderate", 2, wacc_detail=None)
    )
    assert pd.api.types.is_integer_dtype(dm.get_data("resource_cost")["tech_detail"])
    costs = new_build.fetch_resource_costs(
        _settings([["Wind", settings_value, "Moderate", 1]])
    )

    gen = _row(costs, "Wind", "1", "Moderate", ("Wind", settings_value, "Moderate", 1))
    assert gen.loc[0, "capex_mw"] == pytest.approx(1000.0)


@pytest.mark.parametrize("settings_value", [1.5, "1.5"])
def test_text_filter_does_not_round_decimal_settings(make_dm, settings_value):
    """A setting of 1.5 must not match an integer table value of 2."""
    dm = make_dm(_cost_rows("Wind", 2, "Moderate", wacc_detail=None))
    assert pd.api.types.is_integer_dtype(dm.get_data("resource_cost")["tech_detail"])
    costs = new_build.fetch_resource_costs(
        _settings([["Wind", settings_value, "Moderate", 1]])
    )

    assert costs.empty


@pytest.mark.parametrize("settings_value", [True, "True"])
def test_boolean_csv_column_matches_settings(make_dm, settings_value):
    """A boolean key column matches YAML booleans and quoted "True".

    DuckDB infers ``BOOLEAN`` for a column of lowercase ``true``/``false`` and
    casts it to 'true'/'false' in SQL, unlike Python ``str()``.
    """
    dm = make_dm(
        _cost_rows("Wind", "true", "Moderate", wacc_detail=None)
        + _cost_rows("Wind", "false", "Moderate", 2, wacc_detail=None),
        hr_rows=[
            {
                "technology": "Wind",
                "tech_detail": "true",
                "cost_case": "Moderate",
                "basis_year": basis_year,
                "heat_rate": 7.0,
                "data_year": 2024,
            }
            for basis_year in (2030, 2031)
        ],
    )
    assert pd.api.types.is_bool_dtype(dm.get_data("resource_cost")["tech_detail"])
    assert pd.api.types.is_bool_dtype(dm.get_data("resource_heat_rate")["tech_detail"])

    costs = new_build.fetch_resource_costs(
        _settings([["Wind", settings_value, "Moderate", 1]])
    )
    merged = costs.merge(
        new_build.fetch_heat_rates(),
        on=["technology", "tech_detail", "cost_case", "basis_year"],
        how="left",
    )

    gen = _row(
        merged, "Wind", "True", "Moderate", ("Wind", settings_value, "Moderate", 1)
    )
    assert gen.loc[0, "capex_mw"] == pytest.approx(1000.0)
    assert gen.loc[0, "heat_rate"] == pytest.approx(7.0)


def test_build_new_resources_unquoted_keys_end_to_end(make_dm, monkeypatch):
    """Unquoted numeric keys get costs, heat rates, modifiers, and names."""
    monkeypatch.setattr(new_build, "apply_all_tag_to_regions", lambda s: s)
    make_dm(
        _cost_rows("NaturalGas", "1", "2030") + _cost_rows("NaturalGas", "CT", "2030"),
        hr_rows=[
            {
                "technology": "NaturalGas",
                "tech_detail": tech_detail,
                "cost_case": "2030",
                "basis_year": basis_year,
                "heat_rate": 7.0,
                "data_year": 2024,
            }
            for tech_detail in ("1", "CT")
            for basis_year in (2030, 2031)
        ],
    )
    settings = _settings(
        [["NaturalGas", 1, 2030, 100]],
        model_regions=["R1"],
        resource_cap_recovery_years=20,
        clustering_n_jobs=1,
        resource_modifiers={
            "ng": {
                "technology": "NaturalGas",
                "tech_detail": 1,
                "Var_OM_Cost_per_MWh": ["add", 5],
            }
        },
    )
    costs = new_build.fetch_resource_costs(settings)
    heat_rates = new_build.fetch_heat_rates()

    monkeypatch.setattr(
        new_build,
        "get_data",
        lambda table_name, **kwargs: pd.DataFrame(
            {"region": ["R1"], "technology": ["NaturalGas"], "value": [1.0]}
        ),
    )
    result = new_build.build_new_resources(costs, heat_rates, settings)

    assert result["technology"].tolist() == ["NaturalGas_1_2030"]
    assert result["Heat_Rate_MMBTU_per_MWh"].iloc[0] == pytest.approx(7.0)
    assert result["Var_OM_Cost_per_MWh"].iloc[0] == pytest.approx(6.0)
    assert result["Inv_Cost_per_MWyr"].iloc[0] > 0
