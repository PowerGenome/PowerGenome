"""
Tests for powergenome.validate — settings and data validation module.

Phase 1 tests use plain dicts and require no file I/O.
Phase 2 tests use the test_system settings + DataManager.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from powergenome.validate import (
    ValidationLevel,
    ValidationResult,
    _check_aggregation_base_regions,
    _check_data_tables_loaded,
    _check_fuel_consistency,
    _check_fuel_price_coverage,
    _check_model_tag_coverage,
    _check_new_resource_cost_years,
    _check_paths_exist,
    _check_region_consistency,
    _check_required_keys,
    _check_transmission_regions,
    _check_year_list_consistency,
    _extract_planning_periods,
    _make_iterable,
    _parse_validate_args,
    _settings_as_dict,
    _tech_has_required_tag,
    _tech_matches_any_key,
    report_validation_results,
    validate_powergenome,
    validate_settings,
    validate_settings_with_data,
)

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ─────────────────────────────────────────────────────────────────────────────

TEST_SETTINGS_PATH = "tests/test_system/settings"
TEST_DATA_PATH = "tests/test_system/test_data"


def _errors(results):
    return [r for r in results if r.level == ValidationLevel.ERROR]


def _warnings(results):
    return [r for r in results if r.level == ValidationLevel.WARNING]


def _minimal_valid_settings(**overrides):
    """Return a minimal settings dict that passes all Phase 1 checks (no paths)."""
    base = {
        "model_regions": ["RegionA", "RegionB"],
        "target_usd_year": 2020,
        "model_year": [2030],
        "model_first_planning_year": [2025],
        "model_tag_values": {
            "THERM": {"NaturalGas": 1, "Coal": 1},
            "VRE": {"Wind": 1, "Solar": 1},
            "STOR": {"Battery": 1},
            "MUST_RUN": {"Nuclear": 1},
            "FLEX": {},
            "HYDRO": {"Hydro": 1},
        },
        "fuel_scenarios": {"naturalgas": "reference", "coal": "reference"},
        "tech_fuel_map": {"NaturalGas CC": "naturalgas", "Coal ST": "coal"},
        "fuel_emission_factors": {"naturalgas": 0.05306, "coal": 0.09552},
    }
    base.update(overrides)
    return base


# ─────────────────────────────────────────────────────────────────────────────
# Unit helpers
# ─────────────────────────────────────────────────────────────────────────────


def test_extract_planning_periods_model_periods_single():
    s = {"model_periods": [2025, 2030]}
    assert _extract_planning_periods(s) == [(2025, 2030)]


def test_extract_planning_periods_model_periods_multi():
    s = {"model_periods": [[2025, 2030], [2031, 2040]]}
    assert _extract_planning_periods(s) == [(2025, 2030), (2031, 2040)]


def test_extract_planning_periods_legacy():
    s = {"model_year": [2030, 2040], "model_first_planning_year": [2025, 2031]}
    assert _extract_planning_periods(s) == [(2025, 2030), (2031, 2040)]


def test_extract_planning_periods_empty():
    assert _extract_planning_periods({}) == []


def test_extract_planning_periods_empty_model_periods_list():
    assert _extract_planning_periods({"model_periods": []}) == []


def test_extract_planning_periods_skips_invalid_period_entries():
    s = {"model_periods": [["bad", 2030], [2025, 2030], [2025]]}
    assert _extract_planning_periods(s) == [(2025, 2030)]


@pytest.mark.parametrize(
    "value, expected",
    [
        ([1, 2], [1, 2]),
        ((1, 2), [1, 2]),
        (None, []),
        ("x", ["x"]),
    ],
)
def test_make_iterable_variants(value, expected):
    assert _make_iterable(value) == expected


def test_settings_as_dict_from_dict():
    d = {"a": 1}
    assert _settings_as_dict(d) is d


def test_settings_as_dict_from_to_dict_object():
    class HasToDict:
        def to_dict(self):
            return {"k": "v"}

    assert _settings_as_dict(HasToDict()) == {"k": "v"}


def test_settings_as_dict_from_get_data_object():
    class HasGetData:
        def get_data(self):
            return {"x": 2}

    assert _settings_as_dict(HasGetData()) == {"x": 2}


def test_settings_as_dict_rejects_invalid_type():
    with pytest.raises(TypeError, match="settings must be a dict"):
        _settings_as_dict(123)


def test_validation_result_str_with_and_without_detail():
    no_detail = ValidationResult(ValidationLevel.WARNING, "cat", "msg")
    with_detail = ValidationResult(ValidationLevel.ERROR, "cat", "msg", detail="more")
    assert "Detail:" not in str(no_detail)
    assert "Detail: more" in str(with_detail)


def test_tech_matches_any_key_positive():
    tag_dict = {"NaturalGas": 1, "Wind": 1}
    assert _tech_matches_any_key("NaturalGas_CCGT", tag_dict)
    assert _tech_matches_any_key("naturalgas_ccgt", tag_dict)  # case-insensitive


def test_tech_matches_any_key_negative():
    tag_dict = {"NaturalGas": 1}
    assert not _tech_matches_any_key("Battery_Lithium", tag_dict)


def test_tech_has_required_tag_therm():
    model_tag_values = {
        "THERM": {"NaturalGas": 1},
        "VRE": {},
        "STOR": {},
        "MUST_RUN": {},
        "FLEX": {},
        "HYDRO": {},
    }
    assert _tech_has_required_tag("NaturalGas_CCGT", model_tag_values)


def test_tech_has_required_tag_no_match():
    model_tag_values = {
        "THERM": {"NaturalGas": 1},
        "VRE": {},
        "STOR": {},
        "MUST_RUN": {},
        "FLEX": {},
        "HYDRO": {},
    }
    assert not _tech_has_required_tag("UnknownTech_v1", model_tag_values)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — _check_required_keys
# ─────────────────────────────────────────────────────────────────────────────


class TestCheckRequiredKeys:
    def test_valid_settings_no_errors(self):
        s = {
            "model_regions": ["r1"],
            "target_usd_year": 2020,
            "model_year": [2030],
            "model_first_planning_year": [2025],
        }
        assert _errors(_check_required_keys(s)) == []

    def test_valid_with_model_periods(self):
        s = {
            "model_regions": ["r1"],
            "target_usd_year": 2020,
            "model_periods": [[2025, 2030]],
        }
        assert _errors(_check_required_keys(s)) == []

    def test_missing_model_regions(self):
        s = {
            "target_usd_year": 2020,
            "model_year": [2030],
            "model_first_planning_year": [2025],
        }
        results = _check_required_keys(s)
        assert any("model_regions" in r.message for r in _errors(results))

    def test_missing_target_usd_year(self):
        s = {
            "model_regions": ["r1"],
            "model_year": [2030],
            "model_first_planning_year": [2025],
        }
        results = _check_required_keys(s)
        assert any("target_usd_year" in r.message for r in _errors(results))

    def test_missing_all_planning_year_options(self):
        s = {"model_regions": ["r1"], "target_usd_year": 2020}
        results = _check_required_keys(s)
        assert len(_errors(results)) >= 1
        assert any("model_periods" in r.message for r in _errors(results))

    def test_missing_only_model_first_planning_year(self):
        """Has model_year but not model_first_planning_year → error about combined requirement."""
        s = {"model_regions": ["r1"], "target_usd_year": 2020, "model_year": [2030]}
        results = _check_required_keys(s)
        assert len(_errors(results)) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — _check_year_list_consistency
# ─────────────────────────────────────────────────────────────────────────────


class TestCheckYearListConsistency:
    def test_valid_legacy(self):
        s = {"model_year": [2030, 2040], "model_first_planning_year": [2025, 2031]}
        assert _check_year_list_consistency(s) == []

    def test_valid_model_periods(self):
        s = {"model_periods": [[2025, 2030], [2031, 2040]]}
        assert _check_year_list_consistency(s) == []

    def test_length_mismatch(self):
        s = {"model_year": [2030, 2040], "model_first_planning_year": [2025]}
        results = _check_year_list_consistency(s)
        assert len(_errors(results)) == 1

    def test_first_year_after_end_year(self):
        s = {"model_year": [2030], "model_first_planning_year": [2035]}
        results = _check_year_list_consistency(s)
        assert len(_errors(results)) == 1

    def test_model_periods_bad_entries(self):
        s = {"model_periods": [[2025, 2030], [2040]]}  # second entry is length 1
        results = _check_year_list_consistency(s)
        assert len(_errors(results)) == 1

    def test_model_periods_inverted(self):
        s = {"model_periods": [[2040, 2030]]}  # first > last
        results = _check_year_list_consistency(s)
        assert len(_errors(results)) == 1

    def test_valid_single_period(self):
        s = {"model_periods": [2025, 2030]}  # flat single period
        assert _check_year_list_consistency(s) == []

    def test_model_periods_non_numeric_values_are_ignored(self):
        s = {"model_periods": [["bad", 2030]]}
        assert _check_year_list_consistency(s) == []

    def test_legacy_non_numeric_values_are_ignored(self):
        s = {"model_year": ["bad"], "model_first_planning_year": [2025]}
        assert _check_year_list_consistency(s) == []


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — _check_paths_exist
# ─────────────────────────────────────────────────────────────────────────────


class TestCheckPathsExist:
    def test_no_paths_configured(self):
        assert _check_paths_exist({}) == []

    def test_existing_data_location(self, tmp_path):
        s = {"data_location": str(tmp_path)}
        assert _errors(_check_paths_exist(s)) == []

    def test_existing_multiple_data_locations(self, tmp_path):
        locations = [tmp_path / "data_a", tmp_path / "data_b"]
        for location in locations:
            location.mkdir()

        assert _errors(_check_paths_exist({"data_location": locations})) == []

    def test_missing_data_location_in_list(self, tmp_path):
        locations = [tmp_path, tmp_path / "no_such_folder"]
        results = _check_paths_exist({"data_location": locations})

        assert len(_errors(results)) == 1
        assert "no_such_folder" in _errors(results)[0].message

    def test_missing_data_location(self, tmp_path):
        s = {"data_location": str(tmp_path / "no_such_folder")}
        results = _check_paths_exist(s)
        assert any("data_location" in r.message for r in _errors(results))

    def test_missing_resource_groups(self, tmp_path):
        s = {"RESOURCE_GROUPS": str(tmp_path / "does_not_exist")}
        results = _check_paths_exist(s)
        assert any("RESOURCE_GROUPS" in r.message for r in _errors(results))

    def test_existing_input_folder_with_scenario_file(self, tmp_path):
        scenario_file = tmp_path / "scenarios.csv"
        scenario_file.write_text("case_id,year\nbaseline,2030\n")
        s = {"input_folder": str(tmp_path), "scenario_definitions_fn": "scenarios.csv"}
        assert _errors(_check_paths_exist(s)) == []

    def test_missing_scenario_file(self, tmp_path):
        s = {
            "input_folder": str(tmp_path),
            "scenario_definitions_fn": "missing_scenarios.csv",
        }
        results = _check_paths_exist(s)
        assert any("scenario_definitions_fn" in r.message for r in _errors(results))

    def test_missing_input_folder(self, tmp_path):
        s = {"input_folder": str(tmp_path / "no_folder")}
        results = _check_paths_exist(s)
        assert any("input_folder" in r.message for r in _errors(results))


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — _check_region_consistency
# ─────────────────────────────────────────────────────────────────────────────


class TestCheckRegionConsistency:
    def test_no_region_settings(self):
        s = {"model_regions": ["r1", "r2"]}
        assert _check_region_consistency(s) == []

    def test_valid_regional_tag_values(self):
        s = {
            "model_regions": ["r1", "r2"],
            "regional_tag_values": {"r1": {"THERM": {"Coal": 1}}, "r2": {}},
        }
        assert _warnings(_check_region_consistency(s)) == []

    def test_unknown_region_in_regional_tag_values(self):
        s = {
            "model_regions": ["r1", "r2"],
            "regional_tag_values": {"r1": {}, "r_unknown": {}},
        }
        results = _check_region_consistency(s)
        assert any("regional_tag_values" in r.message for r in _warnings(results))
        assert any(
            "r_unknown" in (r.message + (r.detail or "")) for r in _warnings(results)
        )

    def test_unknown_region_in_alt_num_clusters(self):
        s = {
            "model_regions": ["r1"],
            "alt_num_clusters": {"r1": {"Coal": 2}, "r_bad": {"Coal": 1}},
        }
        results = _check_region_consistency(s)
        assert any("alt_num_clusters" in r.message for r in _warnings(results))

    def test_unknown_region_in_regional_capacity_reserves(self):
        s = {
            "model_regions": ["r1", "r2"],
            "regional_capacity_reserves": {"CapRes_1": {"r1": 0.15, "r_unknown": 0.15}},
        }
        results = _check_region_consistency(s)
        assert any(
            "regional_capacity_reserves" in r.message for r in _warnings(results)
        )

    def test_regional_capacity_reserves_non_dict_is_ignored(self):
        s = {
            "model_regions": ["r1"],
            "regional_capacity_reserves": {"CapRes_1": ["r1", 0.15]},
        }
        assert _warnings(_check_region_consistency(s)) == []

    def test_unknown_small_hydro_region(self):
        s = {
            "model_regions": ["r1", "r2"],
            "small_hydro_regions": ["r1", "r_ghost"],
        }
        results = _check_region_consistency(s)
        assert any("small_hydro_regions" in r.message for r in _warnings(results))

    def test_all_regions_valid(self):
        s = {
            "model_regions": ["r1", "r2"],
            "regional_hydro_factor": {"r1": 4, "r2": 4},
            "distributed_gen_method": {"r1": "capacity", "r2": "capacity"},
        }
        assert _warnings(_check_region_consistency(s)) == []


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — _check_model_tag_coverage
# ─────────────────────────────────────────────────────────────────────────────


class TestCheckModelTagCoverage:
    def test_no_new_resources(self):
        s = _minimal_valid_settings(new_resources=None)
        assert _check_model_tag_coverage(s) == []

    def test_with_tagged_resources(self):
        s = _minimal_valid_settings(
            new_resources=[
                ["NaturalGas", "CCGT", "Moderate", 500],
                ["Wind", "Class3", "Moderate", 1],
                ["Battery", "LithiumIon", "Moderate", 1],
            ]
        )
        assert _warnings(_check_model_tag_coverage(s)) == []

    def test_untagged_resource(self):
        s = _minimal_valid_settings(
            new_resources=[["UnknownTech", "v1", "Moderate", 100]]
        )
        results = _check_model_tag_coverage(s)
        assert any("UnknownTech_v1" in (r.detail or "") for r in _warnings(results))

    def test_no_model_tag_values(self):
        """If model_tag_values is absent we skip the check (no false positives)."""
        s = _minimal_valid_settings(model_tag_values=None)
        assert _check_model_tag_coverage(s) == []

    def test_modified_new_resources_untagged(self):
        s = _minimal_valid_settings(
            modified_new_resources={
                "MyCustomTech": {
                    "new_technology": "Zeppelin",
                    "new_tech_detail": "v1",
                }
            }
        )
        results = _check_model_tag_coverage(s)
        assert any("Zeppelin" in (r.detail or "") for r in _warnings(results))

    def test_modified_new_resources_tagged(self):
        s = _minimal_valid_settings(
            modified_new_resources={
                "MyWind": {
                    "new_technology": "Wind",
                    "new_tech_detail": "Class3",
                }
            }
        )
        assert _warnings(_check_model_tag_coverage(s)) == []

    def test_malformed_new_resources_entries_are_ignored(self):
        s = _minimal_valid_settings(
            new_resources=["bad-entry", ["NaturalGas"]],
            modified_new_resources={"bad": "not-a-dict"},
        )
        assert _check_model_tag_coverage(s) == []


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — _check_fuel_consistency
# ─────────────────────────────────────────────────────────────────────────────


class TestCheckFuelConsistency:
    def test_valid(self):
        s = _minimal_valid_settings()
        assert _check_fuel_consistency(s) == []

    def test_fuel_not_in_fuel_scenarios(self):
        s = _minimal_valid_settings(
            tech_fuel_map={"SomeGen": "hydrogen"},
            fuel_scenarios={"naturalgas": "ref"},
        )
        results = _check_fuel_consistency(s)
        assert any("hydrogen" in r.message for r in _warnings(results))

    def test_fuel_in_user_fuel_price_ok(self):
        s = _minimal_valid_settings(
            tech_fuel_map={"SomeGen": "hydrogen"},
            fuel_scenarios={},
            user_fuel_price={"hydrogen": 10.0},
        )
        assert _warnings(_check_fuel_consistency(s)) == []

    def test_ccs_missing_capture_rate(self):
        s = _minimal_valid_settings(
            ccs_fuel_map={"NaturalGas_CCS90": "naturalgas_ccs90"},
            ccs_capture_rate={},  # missing entry
        )
        results = _check_fuel_consistency(s)
        assert any("ccs_capture_rate" in r.message for r in _warnings(results))

    def test_ccs_with_capture_rate_ok(self):
        s = _minimal_valid_settings(
            ccs_fuel_map={"NaturalGas_CCS90": "naturalgas_ccs90"},
            ccs_capture_rate={"naturalgas_ccs90": 0.9},
            fuel_emission_factors={"naturalgas": 0.05306, "coal": 0.09552},
        )
        assert _warnings(_check_fuel_consistency(s)) == []

    def test_ccs_missing_base_fuel_emission_factor(self):
        s = _minimal_valid_settings(
            ccs_fuel_map={"CoalCCS": "coal_ccs90"},
            ccs_capture_rate={"coal_ccs90": 0.9},
            fuel_emission_factors={},  # no 'coal' entry
        )
        results = _check_fuel_consistency(s)
        assert any("fuel_emission_factors" in r.message for r in _warnings(results))

    def test_ccs_base_fuel_split_without_ccs_suffix(self):
        s = _minimal_valid_settings(
            ccs_fuel_map={"GasCCS": "naturalgas_capture90"},
            ccs_capture_rate={"naturalgas_capture90": 0.9},
            fuel_emission_factors={},
        )
        results = _check_fuel_consistency(s)
        assert any("naturalgas" in r.message for r in _warnings(results))


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — validate_settings (integration)
# ─────────────────────────────────────────────────────────────────────────────


class TestValidateSettings:
    def test_minimal_valid_dict_no_issues(self):
        """A well-formed minimal settings dict should produce no results."""
        results = validate_settings(_minimal_valid_settings())
        # Path checks may produce errors if actual paths aren't set, but
        # we haven't set any paths, so there should be no path errors.
        path_errors = [r for r in results if r.category == "paths"]
        assert path_errors == []

    def test_empty_settings_produces_errors(self):
        results = validate_settings({})
        assert len(_errors(results)) >= 1  # at least required-key errors

    def test_accepts_settings_object(self):
        from powergenome.settings import Settings

        s = Settings.from_dict(_minimal_valid_settings())
        results = validate_settings(s)
        path_errors = [r for r in results if r.category == "paths"]
        assert path_errors == []

    def test_unknown_region_produces_warning(self):
        s = _minimal_valid_settings(
            regional_tag_values={"RegionA": {}, "Unknown_Region": {}}
        )
        results = validate_settings(s)
        # RegionA is valid, Unknown_Region is not
        assert any(
            r.level == ValidationLevel.WARNING and "region_consistency" == r.category
            for r in results
        )

    def test_mismatched_year_lengths_error(self):
        s = _minimal_valid_settings(
            model_year=[2030, 2040],
            model_first_planning_year=[2025],  # wrong length
        )
        results = validate_settings(s)
        assert any(
            r.level == ValidationLevel.ERROR and "planning_years" == r.category
            for r in results
        )

    def test_check_exceptions_become_error_results(self, monkeypatch):
        import powergenome.validate as validate_mod

        def bad_check(_):
            raise RuntimeError("boom")

        monkeypatch.setattr(validate_mod, "_PHASE1_CHECKS", [bad_check])
        results = validate_settings(_minimal_valid_settings())
        assert len(results) == 1
        assert results[0].level == ValidationLevel.ERROR
        assert results[0].category == "internal_error"
        assert "bad_check" in results[0].message
        assert "RuntimeError" in results[0].detail
        assert "boom" in results[0].detail


# ─────────────────────────────────────────────────────────────────────────────
# report_validation_results
# ─────────────────────────────────────────────────────────────────────────────


class TestReportValidationResults:
    def test_empty_results_no_raise(self):
        report_validation_results([])  # should not raise

    def test_warnings_only_no_raise(self):
        results = [ValidationResult(ValidationLevel.WARNING, "test", "a warning")]
        report_validation_results(results)  # should not raise

    def test_error_raises_by_default(self):
        results = [ValidationResult(ValidationLevel.ERROR, "test", "an error")]
        with pytest.raises(ValueError, match="1 error"):
            report_validation_results(results)

    def test_error_no_raise_when_disabled(self):
        results = [ValidationResult(ValidationLevel.ERROR, "test", "an error")]
        report_validation_results(results, raise_on_error=False)  # must not raise


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 tests (require DataManager + test data)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def test_settings():
    """Load the test system settings with paths pointing at test data."""
    from powergenome.settings import Settings

    settings = Settings(config_path=TEST_SETTINGS_PATH)
    settings["RESOURCE_GROUPS"] = str(Path(TEST_DATA_PATH) / "resource_groups")
    settings["data_location"] = TEST_DATA_PATH
    settings["cache_resource_clusters"] = False
    settings["use_resource_clusters_cache"] = False
    return settings


@pytest.fixture(scope="module")
def initialized_dm(test_settings):
    """Initialize DataManager with test data once for the module."""
    from powergenome.database import _data_manager, initialize_data_manager

    initialize_data_manager(test_settings, test_settings["data_location"])
    return _data_manager


class TestPhase2WithTestData:
    def test_transmission_regions_no_spurious_warnings(
        self, test_settings, initialized_dm
    ):
        """Test system network_costs_test_data.csv should have no unknown regions."""
        results = _check_transmission_regions(test_settings.to_dict(), initialized_dm)
        assert (
            _warnings(results) == []
        ), f"Unexpected transmission region warnings: {results}"

    def test_fuel_price_coverage_skips_if_table_absent(self):
        """If fuel_price table is not loaded, the check should silently skip."""

        class FakeDM:
            available_tables = set()  # no fuel_price table

        s = _minimal_valid_settings()
        results = _check_fuel_price_coverage(s, FakeDM())
        assert results == []

    def test_new_resource_cost_years_skips_if_table_absent(self):
        """If resource_cost table is not loaded, the check should silently skip."""

        class FakeDM:
            available_tables = set()

        s = _minimal_valid_settings(
            new_resources=[["NaturalGas", "CCGT", "Moderate", 500]]
        )
        results = _check_new_resource_cost_years(s, FakeDM())
        assert results == []

    def test_new_resource_cost_years_warns_on_no_overlap(self):
        """A resource with no basis_year in the planning range should warn."""
        import pandas as pd

        class FakeDM:
            available_tables = {"resource_cost"}

            def get_data(self, table_name, columns=None, **kwargs):
                return pd.DataFrame(
                    {
                        "technology": ["NaturalGas"],
                        "tech_detail": ["CCGT"],
                        "cost_case": ["Moderate"],
                        "basis_year": [2010],  # out of range for 2025-2030 period
                    }
                )

        s = _minimal_valid_settings(
            new_resources=[["NaturalGas", "CCGT", "Moderate", 500]]
        )
        results = _check_new_resource_cost_years(s, FakeDM())
        assert any(
            "NaturalGas/CCGT/Moderate" in (r.detail or "") for r in _warnings(results)
        )

    def test_new_resource_cost_years_no_warning_when_overlap(self):
        """A resource with a matching basis_year should produce no warning."""
        import pandas as pd

        class FakeDM:
            available_tables = {"resource_cost"}

            def get_data(self, table_name, columns=None, **kwargs):
                return pd.DataFrame(
                    {
                        "technology": ["NaturalGas"],
                        "tech_detail": ["CCGT"],
                        "cost_case": ["Moderate"],
                        "basis_year": [2028],  # within 2025-2030
                    }
                )

        s = _minimal_valid_settings(
            new_resources=[["NaturalGas", "CCGT", "Moderate", 500]]
        )
        results = _check_new_resource_cost_years(s, FakeDM())
        assert _warnings(results) == []

    def test_transmission_regions_warns_on_unknown(self):
        """Unknown region in transmission table should produce a WARNING."""
        import pandas as pd

        class FakeDM:
            available_tables = {"transmission_cost"}

            def get_data(self, table_name, columns=None, **kwargs):
                return pd.DataFrame(
                    {
                        "start_region": ["RegionA", "UNKNOWN_REGION"],
                        "dest_region": ["RegionB", "RegionA"],
                    }
                )

        s = _minimal_valid_settings()  # model_regions = ["RegionA", "RegionB"]
        results = _check_transmission_regions(s, FakeDM())
        assert any("UNKNOWN_REGION" in (r.detail or "") for r in _warnings(results))

    def test_transmission_regions_no_warning_for_base_ipm_regions(self):
        """Base IPM regions from region_aggregations should be considered valid."""
        import pandas as pd

        class FakeDM:
            available_tables = {"transmission_cost"}

            def get_data(self, table_name, columns=None, **kwargs):
                # p1 is a base region of p1_2; should not trigger a warning
                return pd.DataFrame({"start_region": ["p1"], "dest_region": ["p2"]})

        s = {
            "model_regions": ["p1_2", "p3", "p4"],
            "region_aggregations": {"p1_2": ["p1", "p2"]},
            "target_usd_year": 2020,
            "model_year": [2030],
            "model_first_planning_year": [2025],
        }
        results = _check_transmission_regions(s, FakeDM())
        assert _warnings(results) == []

    def test_fuel_price_coverage_warns_on_missing(self):
        """Missing fuel/year/region combination should produce a WARNING."""
        import pandas as pd

        class FakeDM:
            available_tables = {"fuel_price"}

            def get_data(self, table_name, columns=None, **kwargs):
                # Only has 'coal' for year 2030 in RegionA — missing naturalgas
                return pd.DataFrame(
                    {
                        "year": [2030],
                        "fuel": ["coal"],
                        "region": ["RegionA"],
                        "scenario": ["reference"],
                    }
                )

        s = _minimal_valid_settings(
            fuel_scenarios={"coal": "reference", "naturalgas": "reference"}
        )
        results = _check_fuel_price_coverage(s, FakeDM())
        warns = _warnings(results)
        assert len(warns) == 1
        assert "fuel_price_coverage" == warns[0].category
        # naturalgas/reference/RegionA/2030 and naturalgas/reference/RegionB/2030 missing
        # coal/reference/RegionB/2030 also missing
        assert "naturalgas" in (warns[0].detail or "")

    def test_fuel_price_coverage_returns_when_no_planning_periods(self):
        class FakeDM:
            available_tables = {"fuel_price"}

            def get_data(self, table_name, columns=None, **kwargs):
                return pd.DataFrame(
                    {
                        "year": [2030],
                        "fuel": ["coal"],
                        "region": ["RegionA"],
                        "scenario": ["reference"],
                    }
                )

        s = _minimal_valid_settings(model_year=None, model_first_planning_year=None)
        assert _check_fuel_price_coverage(s, FakeDM()) == []

    def test_fuel_price_coverage_returns_when_fuels_or_regions_missing(self):
        class FakeDM:
            available_tables = {"fuel_price"}

            def get_data(self, table_name, columns=None, **kwargs):
                return pd.DataFrame({"year": [2030], "fuel": ["coal"]})

        s = _minimal_valid_settings(fuel_scenarios={}, model_regions=[])
        assert _check_fuel_price_coverage(s, FakeDM()) == []

    def test_fuel_price_coverage_warns_when_table_read_fails(self):
        class FakeDM:
            available_tables = {"fuel_price"}

            def get_data(self, table_name, columns=None, **kwargs):
                raise RuntimeError("cannot read fuel table")

        s = _minimal_valid_settings()
        results = _check_fuel_price_coverage(s, FakeDM())
        assert any(
            "Could not load fuel_price table" in r.message for r in _warnings(results)
        )

    def test_fuel_price_coverage_returns_for_empty_or_no_year(self):
        class FakeDM:
            available_tables = {"fuel_price"}

            def get_data(self, table_name, columns=None, **kwargs):
                return pd.DataFrame()

        s = _minimal_valid_settings()
        assert _check_fuel_price_coverage(s, FakeDM()) == []

    def test_fuel_price_coverage_handles_non_numeric_year(self):
        class FakeDM:
            available_tables = {"fuel_price"}

            def get_data(self, table_name, columns=None, **kwargs):
                return pd.DataFrame(
                    {
                        "year": ["bad"],
                        "fuel": ["coal"],
                        "region": ["RegionA"],
                        "scenario": ["reference"],
                    }
                )

        s = _minimal_valid_settings(fuel_scenarios={"coal": "reference"})
        results = _check_fuel_price_coverage(s, FakeDM())
        assert _warnings(results)

    def test_fuel_price_coverage_truncates_long_missing_detail(self):
        class FakeDM:
            available_tables = {"fuel_price"}

            def get_data(self, table_name, columns=None, **kwargs):
                # Keep table non-empty so the check reaches the missing-combo logic.
                return pd.DataFrame(
                    {
                        "year": [1990],
                        "fuel": ["coal"],
                        "region": ["Nowhere"],
                        "scenario": ["reference"],
                    }
                )

        s = _minimal_valid_settings(
            model_year=[2030, 2040],
            model_first_planning_year=[2025, 2031],
            model_regions=["RegionA", "RegionB", "RegionC"],
            fuel_scenarios={
                "coal": "reference",
                "naturalgas": "reference",
            },
        )
        results = _check_fuel_price_coverage(s, FakeDM())
        warn = _warnings(results)[0]
        assert "and" in (warn.detail or "")

    def test_transmission_regions_warns_when_table_read_fails(self):
        class FakeDM:
            available_tables = {"transmission_cost"}

            def get_data(self, table_name, columns=None, **kwargs):
                raise RuntimeError("cannot read tx")

        s = _minimal_valid_settings()
        results = _check_transmission_regions(s, FakeDM())
        assert any(
            "Could not load transmission_cost table" in r.message
            for r in _warnings(results)
        )

    def test_data_tables_loaded_error_on_missing(self):
        """If a configured table is absent from DataManager, flag as ERROR."""

        class FakeDM:
            available_tables = set()  # nothing loaded

        s = {"resource_cost_table": "myfile.csv"}
        results = _check_data_tables_loaded(s, FakeDM())
        assert any("resource_cost_table" in r.message for r in _errors(results))

    def test_data_tables_loaded_no_error_when_present(self):
        """A table present in available_tables should produce no error."""

        class FakeDM:
            available_tables = {"resource_cost"}

        s = {"resource_cost_table": "myfile.csv"}
        results = _check_data_tables_loaded(s, FakeDM())
        assert _errors(results) == []

    def test_data_tables_loaded_graceful_when_mapping_unavailable(self, monkeypatch):
        import powergenome.database as db

        monkeypatch.delattr(db, "DataManager", raising=False)

        class FakeDM:
            available_tables = set()

        assert (
            _check_data_tables_loaded({"resource_cost_table": "x.csv"}, FakeDM()) == []
        )

    def test_validate_settings_with_data_integration(
        self, test_settings, initialized_dm
    ):
        """Full Phase 2 run on test system should not raise (warnings are OK)."""
        results = validate_settings_with_data(test_settings, initialized_dm)
        # Report without raising errors so we can inspect
        report_validation_results(results, raise_on_error=False)
        # The test system should have no ERROR-level issues in Phase 2
        assert _errors(results) == [], f"Unexpected data errors: {_errors(results)}"

    def test_new_resource_cost_years_returns_when_no_new_resources(self):
        class FakeDM:
            available_tables = {"resource_cost"}

            def get_data(self, *a, **kw):
                return pd.DataFrame()

        s = _minimal_valid_settings(new_resources=[])
        assert _check_new_resource_cost_years(s, FakeDM()) == []

    def test_new_resource_cost_years_returns_when_no_planning_periods(self):
        class FakeDM:
            available_tables = {"resource_cost"}

            def get_data(self, *a, **kw):
                return pd.DataFrame()

        s = _minimal_valid_settings(
            new_resources=[["NaturalGas", "CCGT", "Moderate", 500]],
            model_year=None,
            model_first_planning_year=None,
        )
        assert _check_new_resource_cost_years(s, FakeDM()) == []

    def test_new_resource_cost_years_warns_when_table_read_fails(self):
        class FakeDM:
            available_tables = {"resource_cost"}

            def get_data(self, *a, **kw):
                raise RuntimeError("cannot read resource cost")

        s = _minimal_valid_settings(
            new_resources=[["NaturalGas", "CCGT", "Moderate", 500]]
        )
        results = _check_new_resource_cost_years(s, FakeDM())
        assert any(
            "Could not load resource_cost table" in r.message
            for r in _warnings(results)
        )

    def test_new_resource_cost_years_returns_when_empty_table(self):
        class FakeDM:
            available_tables = {"resource_cost"}

            def get_data(self, *a, **kw):
                return pd.DataFrame(
                    columns=["technology", "tech_detail", "cost_case", "basis_year"]
                )

        s = _minimal_valid_settings(
            new_resources=[["NaturalGas", "CCGT", "Moderate", 500]]
        )
        assert _check_new_resource_cost_years(s, FakeDM()) == []

    def test_new_resource_cost_years_handles_bad_basis_year_dtype(self):
        class FakeDM:
            available_tables = {"resource_cost"}

            def get_data(self, *a, **kw):
                return pd.DataFrame(
                    {
                        "technology": ["NaturalGas"],
                        "tech_detail": ["CCGT"],
                        "cost_case": ["Moderate"],
                        "basis_year": ["bad"],
                    }
                )

        s = _minimal_valid_settings(
            new_resources=[["NaturalGas", "CCGT", "Moderate", 500]]
        )
        assert _warnings(_check_new_resource_cost_years(s, FakeDM()))

    def test_new_resource_cost_years_skips_malformed_resource_entries(self):
        class FakeDM:
            available_tables = {"resource_cost"}

            def get_data(self, *a, **kw):
                return pd.DataFrame(
                    {
                        "technology": ["NaturalGas"],
                        "tech_detail": ["CCGT"],
                        "cost_case": ["Moderate"],
                        "basis_year": [2028],
                    }
                )

        s = _minimal_valid_settings(new_resources=[["NaturalGas", "CCGT"], "bad"])
        assert _check_new_resource_cost_years(s, FakeDM()) == []

    def test_new_resource_cost_years_reports_no_matching_rows(self):
        class FakeDM:
            available_tables = {"resource_cost"}

            def get_data(self, *a, **kw):
                return pd.DataFrame(
                    {
                        "technology": ["Wind"],
                        "tech_detail": ["Class3"],
                        "cost_case": ["Moderate"],
                        "basis_year": [2030],
                    }
                )

        s = _minimal_valid_settings(
            new_resources=[["NaturalGas", "CCGT", "Moderate", 500]]
        )
        results = _check_new_resource_cost_years(s, FakeDM())
        assert any(
            "no matching rows found" in (r.detail or "") for r in _warnings(results)
        )

    def test_new_resource_cost_years_ellipsis_for_many_available_years(self):
        class FakeDM:
            available_tables = {"resource_cost"}

            def get_data(self, *a, **kw):
                return pd.DataFrame(
                    {
                        "technology": ["NaturalGas"] * 8,
                        "tech_detail": ["CCGT"] * 8,
                        "cost_case": ["Moderate"] * 8,
                        "basis_year": [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017],
                    }
                )

        s = _minimal_valid_settings(
            new_resources=[["NaturalGas", "CCGT", "Moderate", 500]]
        )
        results = _check_new_resource_cost_years(s, FakeDM())
        assert "…" in (_warnings(results)[0].detail or "")


class TestCheckAggregationBaseRegions:
    """Tests for _check_aggregation_base_regions."""

    def test_no_regions_at_all_skipped(self):
        """Settings with neither model_regions nor region_aggregations → no results."""

        class FakeDM:
            available_tables = {"plant_region"}

            def get_data(self, *a, **kw):
                return pd.DataFrame({"region": ["p1", "p2"]})

        results = _check_aggregation_base_regions({}, FakeDM())
        assert results == []

    def test_passthrough_model_region_valid(self):
        """A pass-through model region present in data → no warning."""

        class FakeDM:
            available_tables = {"plant_region"}

            def get_data(self, *a, **kw):
                return pd.DataFrame({"region": ["p1", "p2", "p3"]})

        # p3 is a pass-through (not in region_aggregations)
        s = {
            "model_regions": ["p1_2", "p3"],
            "region_aggregations": {"p1_2": ["p1", "p2"]},
        }
        results = _check_aggregation_base_regions(s, FakeDM())
        assert results == []

    def test_passthrough_model_region_typo_detected(self):
        """A pass-through model region absent from all tables → WARNING."""

        class FakeDM:
            available_tables = {"plant_region"}

            def get_data(self, *a, **kw):
                return pd.DataFrame({"region": ["p1", "p2", "p3"]})

        # q2 is a typo; listed directly in model_regions without an aggregation entry
        s = {
            "model_regions": ["p1_2", "q2"],
            "region_aggregations": {"p1_2": ["p1", "p2"]},
        }
        results = _check_aggregation_base_regions(s, FakeDM())
        warns = _warnings(results)
        assert len(warns) == 1
        assert warns[0].category == "aggregation_base_regions"
        assert "q2" in (warns[0].detail or "")
        assert "pass-through" in (warns[0].detail or "")

    def test_no_region_aggregations_passthrough_checked(self):
        """With no region_aggregations, pass-through model_regions are still checked."""

        class FakeDM:
            available_tables = {"plant_region"}

            def get_data(self, *a, **kw):
                return pd.DataFrame({"region": ["p1", "p2"]})

        # q3 is a typo in a pure pass-through setup (no aggregations at all)
        s = {"model_regions": ["p1", "q3"]}
        results = _check_aggregation_base_regions(s, FakeDM())
        warns = _warnings(results)
        assert len(warns) == 1
        assert "q3" in (warns[0].detail or "")

    def test_all_base_regions_valid(self):
        """All base regions present in plant_region table → no warning."""

        class FakeDM:
            available_tables = {"plant_region"}

            def get_data(self, *a, **kw):
                return pd.DataFrame({"region": ["p1", "p2", "p3"]})

        s = {
            "model_regions": ["p1_2", "p3"],
            "region_aggregations": {"p1_2": ["p1", "p2"], "p3": ["p3"]},
        }
        results = _check_aggregation_base_regions(s, FakeDM())
        assert results == []

    def test_typo_region_detected(self):
        """Base region 'q2' (typo for 'p2') absent from all tables → WARNING."""

        class FakeDM:
            available_tables = {"plant_region"}

            def get_data(self, *a, **kw):
                return pd.DataFrame({"region": ["p1", "p2", "p3"]})

        s = {
            "model_regions": ["AZ"],
            "region_aggregations": {"AZ": ["p1", "q2"]},  # q2 is a typo
        }
        results = _check_aggregation_base_regions(s, FakeDM())
        warns = _warnings(results)
        assert len(warns) == 1
        assert warns[0].category == "aggregation_base_regions"
        assert "q2" in (warns[0].detail or "")

    def test_region_found_in_demand_not_plant_region(self):
        """Region present in demand but absent from plant_region → no warning."""

        class FakeDM:
            available_tables = {"plant_region", "demand"}

            def get_data(self, table_name, columns=None, **kwargs):
                if table_name == "plant_region":
                    return pd.DataFrame({"region": ["p1"]})
                if table_name == "demand":
                    return pd.DataFrame({"region": ["p1", "p2"]})
                return pd.DataFrame()

        s = {
            "model_regions": ["p1_2"],
            "region_aggregations": {"p1_2": ["p1", "p2"]},
        }
        results = _check_aggregation_base_regions(s, FakeDM())
        assert results == []

    def test_no_available_tables_skipped_gracefully(self):
        """If no relevant data tables are loaded, the check is skipped silently."""

        class FakeDM:
            available_tables = set()

            def get_data(self, *a, **kw):
                return pd.DataFrame()

        s = {
            "model_regions": ["AZ"],
            "region_aggregations": {"AZ": ["p1", "q999"]},
        }
        results = _check_aggregation_base_regions(s, FakeDM())
        assert results == []

    def test_detail_identifies_offending_model_region(self):
        """Detail string should name which model region maps to the bad base region."""

        class FakeDM:
            available_tables = {"plant_region"}

            def get_data(self, *a, **kw):
                return pd.DataFrame({"region": ["p1", "p2"]})

        s = {
            "model_regions": ["AZ", "NM"],
            "region_aggregations": {
                "AZ": ["p1", "p2"],  # valid
                "NM": ["p3", "TYPO_REGION"],  # typo
            },
        }
        results = _check_aggregation_base_regions(s, FakeDM())
        warns = _warnings(results)
        assert len(warns) == 1
        assert "NM" in (warns[0].detail or "")
        assert "TYPO_REGION" in (warns[0].detail or "")
        assert "AZ" not in (warns[0].detail or "")

    def test_region_found_in_fuel_price_table(self):
        """Region present only in fuel_price table → no warning."""

        class FakeDM:
            available_tables = {"fuel_price"}

            def get_data(self, *a, **kw):
                return pd.DataFrame({"region": ["p1", "p2"]})

        s = {
            "model_regions": ["p1_2"],
            "region_aggregations": {"p1_2": ["p1", "p2"]},
        }
        results = _check_aggregation_base_regions(s, FakeDM())
        assert results == []

    def test_region_found_in_transmission_table(self):
        """Region present only in transmission_cost table → no warning."""

        class FakeDM:
            available_tables = {"transmission_cost"}

            def get_data(self, *a, **kw):
                return pd.DataFrame({"start_region": ["p1"], "dest_region": ["p2"]})

        s = {
            "model_regions": ["p1_2"],
            "region_aggregations": {"p1_2": ["p1", "p2"]},
        }
        results = _check_aggregation_base_regions(s, FakeDM())
        assert results == []

    def test_non_list_aggregation_value_is_ignored(self):
        class FakeDM:
            available_tables = {"plant_region"}

            def get_data(self, *a, **kw):
                return pd.DataFrame({"region": ["p1"]})

        s = {"model_regions": ["AZ"], "region_aggregations": {"AZ": "p1"}}
        assert _check_aggregation_base_regions(s, FakeDM()) == []

    def test_aggregation_base_regions_non_list_entry_ignored_when_unknown_exists(self):
        class FakeDM:
            available_tables = {"plant_region"}

            def get_data(self, *a, **kw):
                return pd.DataFrame({"region": ["p1"]})

        s = {
            "model_regions": ["AZ", "p2"],
            "region_aggregations": {"AZ": "p1", "NM": ["q2"]},
        }
        warns = _warnings(_check_aggregation_base_regions(s, FakeDM()))
        assert len(warns) == 1
        assert "q2" in (warns[0].detail or "")

    def test_aggregation_base_regions_handles_table_read_failures(self):
        class FakeDM:
            available_tables = {
                "plant_region",
                "demand",
                "fuel_price",
                "transmission_cost",
            }

            def get_data(self, *a, **kw):
                raise RuntimeError("table read failed")

        s = {"model_regions": ["AZ"], "region_aggregations": {"AZ": ["p1"]}}
        assert _check_aggregation_base_regions(s, FakeDM()) == []


def test_validate_settings_with_data_checker_exception_becomes_error(monkeypatch):
    import powergenome.validate as validate_mod

    def bad_check(_settings, _dm):
        raise RuntimeError("boom")

    class FakeDM:
        available_tables = set()

    monkeypatch.setattr(validate_mod, "_PHASE2_CHECKS", [bad_check])
    results = validate_settings_with_data(_minimal_valid_settings(), FakeDM())
    assert len(results) == 1
    assert results[0].level == ValidationLevel.ERROR
    assert results[0].category == "internal_error"
    assert "bad_check" in results[0].message
    assert "RuntimeError" in results[0].detail
    assert "boom" in results[0].detail


class TestValidateCli:
    def test_parse_validate_args_flags(self):
        args = _parse_validate_args(
            ["-sf", "settings", "--skip-data-checks", "--no-fail"]
        )
        assert args.settings_file == "settings"
        assert args.skip_data_checks is True
        assert args.no_fail is True

    def test_validate_powergenome_exits_on_errors(self, monkeypatch):
        import argparse

        import powergenome.settings as settings_mod

        class FakeSettings(dict):
            def __init__(self, config_path):
                super().__init__({"data_location": None})

        monkeypatch.setattr(settings_mod, "Settings", FakeSettings)
        monkeypatch.setattr(
            "powergenome.validate._parse_validate_args",
            lambda argv=None: argparse.Namespace(
                settings_file="settings",
                skip_data_checks=True,
                no_fail=False,
            ),
        )
        monkeypatch.setattr(
            "powergenome.validate.validate_settings",
            lambda _settings: [ValidationResult(ValidationLevel.ERROR, "x", "err")],
        )
        monkeypatch.setattr(
            "powergenome.validate.report_validation_results",
            lambda *_a, **_kw: None,
        )

        with pytest.raises(SystemExit) as exc:
            validate_powergenome()
        assert exc.value.code == 1

    def test_validate_powergenome_logs_when_phase2_skipped_for_no_data_location(
        self, monkeypatch
    ):
        import argparse

        import powergenome.settings as settings_mod

        class FakeSettings(dict):
            def __init__(self, config_path):
                super().__init__({"data_location": None})

        monkeypatch.setattr(settings_mod, "Settings", FakeSettings)
        monkeypatch.setattr(
            "powergenome.validate._parse_validate_args",
            lambda argv=None: argparse.Namespace(
                settings_file="settings",
                skip_data_checks=False,
                no_fail=True,
            ),
        )
        monkeypatch.setattr(
            "powergenome.validate.validate_settings", lambda _settings: []
        )

        validate_powergenome()

    def test_validate_powergenome_runs_phase2_with_no_phase2_results(self, monkeypatch):
        import argparse
        import types

        import powergenome.database as db_mod
        import powergenome.settings as settings_mod

        class FakeSettings(dict):
            def __init__(self, config_path):
                super().__init__({"data_location": "tests/test_system/test_data"})

        monkeypatch.setattr(settings_mod, "Settings", FakeSettings)
        monkeypatch.setattr(
            "powergenome.validate._parse_validate_args",
            lambda argv=None: argparse.Namespace(
                settings_file="settings",
                skip_data_checks=False,
                no_fail=True,
            ),
        )
        monkeypatch.setattr(
            "powergenome.validate.validate_settings", lambda _settings: []
        )
        monkeypatch.setattr(
            "powergenome.validate.validate_settings_with_data", lambda *_a: []
        )
        monkeypatch.setattr(
            "powergenome.validate.report_validation_results", lambda *_a, **_kw: None
        )
        monkeypatch.setattr(db_mod, "_data_manager", types.SimpleNamespace())
        monkeypatch.setattr(db_mod, "initialize_data_manager", lambda *_a, **_kw: None)

        validate_powergenome()

    def test_validate_powergenome_phase2_exception_sets_error(self, monkeypatch):
        import argparse
        import types

        import powergenome.database as db_mod
        import powergenome.settings as settings_mod

        class FakeSettings(dict):
            def __init__(self, config_path):
                super().__init__({"data_location": "tests/test_system/test_data"})

        monkeypatch.setattr(settings_mod, "Settings", FakeSettings)
        monkeypatch.setattr(
            "powergenome.validate._parse_validate_args",
            lambda argv=None: argparse.Namespace(
                settings_file="settings",
                skip_data_checks=False,
                no_fail=False,
            ),
        )
        monkeypatch.setattr(
            "powergenome.validate.validate_settings", lambda _settings: []
        )
        monkeypatch.setattr(db_mod, "_data_manager", types.SimpleNamespace())

        def _boom(*_a, **_kw):
            raise RuntimeError("init failed")

        monkeypatch.setattr(db_mod, "initialize_data_manager", _boom)

        with pytest.raises(SystemExit) as exc:
            validate_powergenome()
        assert exc.value.code == 1

    def test_validate_powergenome_phase2_results_with_warnings_and_errors(
        self, monkeypatch
    ):
        import argparse
        import types

        import powergenome.database as db_mod
        import powergenome.settings as settings_mod

        class FakeSettings(dict):
            def __init__(self, config_path):
                super().__init__({"data_location": "tests/test_system/test_data"})

        monkeypatch.setattr(settings_mod, "Settings", FakeSettings)
        monkeypatch.setattr(
            "powergenome.validate._parse_validate_args",
            lambda argv=None: argparse.Namespace(
                settings_file="settings",
                skip_data_checks=False,
                no_fail=True,
            ),
        )
        monkeypatch.setattr(
            "powergenome.validate.validate_settings", lambda _settings: []
        )
        monkeypatch.setattr(
            "powergenome.validate.validate_settings_with_data",
            lambda *_a: [
                ValidationResult(ValidationLevel.WARNING, "cat", "warn"),
                ValidationResult(ValidationLevel.ERROR, "cat", "err"),
            ],
        )
        monkeypatch.setattr(
            "powergenome.validate.report_validation_results", lambda *_a, **_kw: None
        )
        monkeypatch.setattr(db_mod, "_data_manager", types.SimpleNamespace())
        monkeypatch.setattr(db_mod, "initialize_data_manager", lambda *_a, **_kw: None)

        validate_powergenome()
