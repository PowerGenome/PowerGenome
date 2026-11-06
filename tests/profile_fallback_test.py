"""Demand-relative safeguards for existing-generator profile coverage."""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from powergenome.external_data import make_generator_variability
from powergenome.generators import (
    PROFILE_FALLBACK_FRACTION,
    GeneratorClusters,
    calc_unit_cluster_values,
    check_profile_fallback_budget,
    label_retired_gens,
    validate_profile_fallback_fraction,
)
from powergenome.resource_clusters import MissingProfileCoverage, ResourceGroup


def affected_rows(capacities, regions=None):
    return pd.DataFrame(
        {
            "Resource": [f"resource_{i}" for i in range(len(capacities))],
            "technology": ["Run of River Hydroelectric"] * len(capacities),
            "region": regions or ["SD"] * len(capacities),
            "capacity_mw": capacities,
            "base_regions": [["p32", "p38"]] * len(capacities),
            "reason": ["Missing profile sites"] * len(capacities),
        }
    )


@pytest.mark.parametrize(
    "fraction",
    [True, False, np.bool_(True), None, "0.005", -0.1, 1.1, np.nan, np.inf, 0.005 + 0j],
)
def test_invalid_fallback_fraction(fraction):
    with pytest.raises(ValueError, match=PROFILE_FALLBACK_FRACTION):
        validate_profile_fallback_fraction(fraction)


@pytest.mark.parametrize("fraction", [0, 0.005, 1, np.float64(0.01)])
def test_valid_fallback_fraction(fraction):
    assert validate_profile_fallback_fraction(fraction) == fraction


@pytest.mark.parametrize("capacity", [4, 5])
def test_fallback_budget_warns_at_or_below_limit(capacity, caplog):
    check_profile_fallback_budget(
        affected_rows([capacity]),
        pd.DataFrame({"SD": [500, 1500]}),
        model_year=2026,
        scenario="IRA_low",
    )
    assert f"Affected operating capacity {capacity} MW" in caplog.text
    assert "allowed capacity 5 MW" in caplog.text
    assert "2026" in caplog.text and "IRA_low" in caplog.text
    assert "availability 1.0" in caplog.text


def test_fallback_budget_combines_clusters_and_technologies():
    affected = affected_rows([3, 3])
    affected.loc[1, "technology"] = "Solar Photovoltaic"
    with pytest.raises(ValueError, match="Affected operating capacity 6 MW"):
        check_profile_fallback_budget(affected, pd.DataFrame({"SD": [1000]}))


def test_fallback_budget_regions_are_independent():
    check_profile_fallback_budget(
        affected_rows([4, 4], ["SD", "ND"]), pd.DataFrame({"SD": [1000], "ND": [1000]})
    )


@pytest.mark.parametrize(
    "demand",
    [
        None,
        pd.DataFrame({"other": [1000]}),
        pd.DataFrame({"SD": []}),
        pd.DataFrame({"SD": [0]}),
        pd.DataFrame({"SD": [1000, np.nan]}),
        pd.DataFrame({"SD": [np.inf]}),
        pd.DataFrame({"SD": [-1, 1000]}),
    ],
)
def test_fallback_budget_fails_closed_without_valid_demand(demand):
    with pytest.raises(ValueError, match="pre-DG regional demand"):
        check_profile_fallback_budget(affected_rows([4]), demand)


def test_strict_budget_and_zero_operating_capacity(caplog):
    with pytest.raises(ValueError, match="exceeds the regional budget"):
        check_profile_fallback_budget(
            affected_rows([0.01]), pd.DataFrame({"SD": [1000]}), fraction=0
        )
    check_profile_fallback_budget(affected_rows([0]), None, fraction=0)
    assert "No operating capacity" in caplog.text


def test_fallback_budget_rejects_rows_without_a_region(caplog):
    with pytest.raises(ValueError, match="no model region") as excinfo:
        check_profile_fallback_budget(
            affected_rows([4], [np.nan]), pd.DataFrame({"SD": [1000]})
        )
    assert "model region <missing>" in str(excinfo.value)
    check_profile_fallback_budget(affected_rows([0], [np.nan]), None)
    assert "No operating capacity" in caplog.text


def make_profile_group(profiles=None, metadata=None, profile_path=None):
    if metadata is None:
        metadata = pd.DataFrame(
            {"id": [0, 1], "region": ["p38", "p38"], "capacity_mw": [1644.4, 100]}
        )
    group = {"technology": "hydro", "existing": True, "small": True}
    if profile_path:
        group["profiles"] = str(profile_path)
    return ResourceGroup(group, metadata=metadata, profiles=profiles)


def tidy_profiles():
    return pd.DataFrame(
        {
            "site_id": np.repeat([0, 1], 24),
            "time_index": np.tile(np.arange(1, 25), 2),
            "value": 1.0,
            "weather_year": 2013,
        }
    )


@pytest.mark.parametrize("storage", ["memory", "csv", "parquet"])
@pytest.mark.parametrize(
    "coverage",
    [
        "complete",
        "missing_site",
        "partial_hours",
        "uniform_gap",
        "absent_sites",
        "empty",
    ],
)
def test_strict_profile_source_coverage(tmp_path, storage, coverage):
    profiles = tidy_profiles()
    if coverage == "missing_site":
        profiles = profiles.loc[profiles.site_id == 0]
    elif coverage == "partial_hours":
        profiles = profiles.drop(index=47)
    elif coverage == "uniform_gap":
        profiles = profiles.loc[profiles.time_index != 12]
    elif coverage == "absent_sites":
        profiles["site_id"] += 10
    elif coverage == "empty":
        profiles = profiles.iloc[:0]
    if storage == "memory":
        group = make_profile_group(profiles)
    else:
        path = tmp_path / f"profiles.{storage}"
        if storage == "csv":
            profiles.to_csv(path, index=False)
        else:
            profiles.to_parquet(path, index=False)
        group = make_profile_group(profile_path=path)
    if coverage == "complete":
        result = group.get_clusters(
            max_clusters=1, weather_year=2013, require_complete_profiles=True
        )
        np.testing.assert_array_equal(result.profile.iloc[0], np.ones(24))
    else:
        with pytest.raises(MissingProfileCoverage):
            group.get_clusters(
                max_clusters=1, weather_year=2013, require_complete_profiles=True
            )


@pytest.mark.parametrize(
    "problem,match",
    [
        ("duplicate", "Duplicate"),
        ("nonfinite", "finite"),
        ("null_time", "non-null"),
        ("null_year", "weather years must be non-null"),
        ("weather_year", "not available"),
        ("schema", "unrecognized format"),
    ],
)
def test_malformed_profiles_are_not_missing_coverage(problem, match):
    profiles = tidy_profiles()
    if problem == "duplicate":
        profiles = pd.concat([profiles, profiles.iloc[[0]]])
    elif problem == "nonfinite":
        profiles.loc[0, "value"] = np.inf
    elif problem == "null_time":
        profiles.loc[0, "time_index"] = np.nan
    elif problem == "null_year":
        profiles.loc[0, "weather_year"] = np.nan
    elif problem == "schema":
        profiles = pd.DataFrame({"bad_column": [1]})
    group = make_profile_group(profiles)
    with pytest.raises(ValueError, match=match) as exc:
        group.get_clusters(
            max_clusters=1,
            weather_year=2014 if problem == "weather_year" else 2013,
            require_complete_profiles=True,
        )
    assert not isinstance(exc.value, MissingProfileCoverage)


def test_integer_site_ids_match_string_profile_ids(tmp_path):
    """Coverage keys follow the requested IDs, not the source column type."""
    profiles = tidy_profiles()
    profiles["site_id"] = profiles["site_id"].astype(str)
    path = tmp_path / "profiles.parquet"
    profiles.to_parquet(path, index=False)
    metadata = pd.DataFrame(
        {"id": [0, 1], "region": ["p38", "p38"], "capacity_mw": [1644.4, 100]}
    )
    group = make_profile_group(profile_path=path, metadata=metadata)

    result = group.get_clusters(
        max_clusters=1, weather_year=2013, require_complete_profiles=True
    )

    np.testing.assert_array_equal(result.profile.iloc[0], np.ones(24))


def test_wide_malformed_column_is_not_masked_by_missing_site(tmp_path):
    """A nonfinite value in a present column outranks a missing site."""
    path = tmp_path / "profiles.csv"
    values = np.ones(24)
    values[-1] = np.inf
    pd.DataFrame({"0": values}).to_csv(path, index=False)
    group = make_profile_group(profile_path=path)

    with pytest.raises(ValueError, match="finite") as exc:
        group.get_clusters(max_clusters=1, require_complete_profiles=True)

    assert not isinstance(exc.value, MissingProfileCoverage)


def test_wide_source_for_other_region_is_missing_coverage(tmp_path):
    """A wide source covering another region is missing coverage, not malformed."""
    path = tmp_path / "profiles.csv"
    pd.DataFrame({"0": np.ones(24)}).to_csv(path, index=False)
    metadata = pd.DataFrame(
        {"id": [0, 1], "region": ["p32", "p38"], "capacity_mw": [1644.4, 100]}
    )
    group = make_profile_group(profile_path=path, metadata=metadata)

    with pytest.raises(MissingProfileCoverage, match="Missing profile sites"):
        group.get_clusters(
            regions=["p38"], max_clusters=1, require_complete_profiles=True
        )


def test_default_loader_preserves_missing_site_behavior():
    profiles = tidy_profiles().query("site_id == 0")
    group = make_profile_group(profiles)
    result = group.get_clusters(max_clusters=1, weather_year=2013)
    np.testing.assert_array_equal(result.profile.iloc[0], np.ones(24))


def test_strict_legacy_wide_missing_and_complete():
    group = make_profile_group(pd.DataFrame({"0": np.ones(24)}))
    with pytest.raises(MissingProfileCoverage, match="Missing profile sites"):
        group.get_clusters(max_clusters=1, require_complete_profiles=True)
    group = make_profile_group(pd.DataFrame({"0": np.ones(24), "1": np.ones(24)}))
    result = group.get_clusters(max_clusters=1, require_complete_profiles=True)
    np.testing.assert_array_equal(result.profile.iloc[0], np.ones(24))


def profile_assigner(region="SD", year=2026, capacity=4, gross_demand=None):
    cls = GeneratorClusters.__wrapped__
    instance = cls.__new__(cls)
    instance.settings = {
        "region_aggregations": {"SD": ["p32", "p38"]},
        "model_year": year,
        "weather_year": 2013,
        "case_id": "IRA_low",
    }
    instance.profile_fallback_fraction = 0.005
    instance.gross_demand = gross_demand
    instance.results = pd.DataFrame(
        {
            "Resource": [f"{region}_ror_1"],
            "technology": ["Run of River Hydroelectric"],
            "region": [region],
            "capacity_mw": [capacity],
            "HYDRO": [1],
            "profile": [None],
        }
    )
    instance.cluster_builder = Mock()
    return instance


@pytest.mark.parametrize(
    "region,year,demand,capacity,errors",
    [
        ("SD", 2026, 1584.86, 4, False),
        ("SD", 2030, 1688.69, 0, False),
        ("p32", 2026, 206.28, 4, True),
        ("p32", 2030, 221.69, 0, False),
    ],
)
def test_sd_p32_profile_fallback_matrix(region, year, demand, capacity, errors, caplog):
    units = pd.DataFrame(
        {
            "cluster": [1, 1],
            "capacity_mw": [2.0, 2.0],
            "capacity_mwh": [0.0, 0.0],
            "operating_year": [1950, 1950],
            "retirement_year": [2030, 2030],
            "heat_rate_mmbtu_mwh": [0.0, 0.0],
            "fom_per_mwyr": [10.0, 10.0],
            "vom_per_mwh": [0.0, 0.0],
        }
    )
    units = label_retired_gens(units, start_year=2026, end_year=year)
    cluster = calc_unit_cluster_values(units)
    assert cluster.capacity_mw.iloc[0] == capacity
    instance = profile_assigner(
        region,
        year,
        cluster.capacity_mw.iloc[0],
        pd.DataFrame({region: np.full(24, demand)}),
    )
    # Metadata describes p38, while the ROR profile source contains p32 only.
    profiles = tidy_profiles().assign(site_id="p32")
    metadata = pd.DataFrame({"id": ["p38"], "region": ["p38"], "capacity_mw": [1644.4]})
    instance.cluster_builder.find_groups.return_value = [
        make_profile_group(profiles, metadata)
    ]
    if errors:
        with pytest.raises(
            ValueError,
            match="Affected operating capacity 4 MW.*allowed capacity 1.0314 MW",
        ):
            instance._assign_existing_profiles()
        assert instance.results.profile.iloc[0] is None
    else:
        instance._assign_existing_profiles()
        np.testing.assert_array_equal(instance.results.profile.iloc[0], np.ones(24))
        assert instance.results.capacity_mw.iloc[0] == capacity
        assert instance.results.HYDRO.iloc[0] == 1
        if capacity:
            assert "Affected operating capacity 4 MW" in caplog.text
            assert "1644.4 MW" not in caplog.text
        else:
            assert "No operating capacity" in caplog.text


@pytest.mark.parametrize(
    "technology",
    ["Run of River Hydroelectric", "Onshore Wind Turbine", "Solar Photovoltaic"],
)
def test_absent_group_checks_budget_before_variability(technology):
    instance = profile_assigner(
        "p32", gross_demand=pd.DataFrame({"p32": [206.28] * 24})
    )
    instance.results["technology"] = technology
    instance.cluster_builder.find_groups.return_value = []
    with pytest.raises(ValueError, match="No matching existing resource group"):
        instance._assign_existing_profiles()


@pytest.mark.parametrize("skip", ["absent_group", "absent_metadata", "empty_clusters"])
def test_small_fallbacks_produce_full_variability_axis(skip):
    instance = profile_assigner(gross_demand=pd.DataFrame({"SD": [1584.86] * 48}))
    group = Mock()
    group.profiles = object()
    group.metadata.read.return_value = pd.DataFrame(
        {"region": ["outside" if skip == "absent_metadata" else "p38"]}
    )
    group.get_clusters.return_value = pd.DataFrame()
    instance.cluster_builder.find_groups.return_value = (
        [] if skip == "absent_group" else [group]
    )
    instance._assign_existing_profiles()
    variability = make_generator_variability(instance.results)
    assert variability.shape == (48, 1)
    np.testing.assert_array_equal(variability.to_numpy(), np.ones((48, 1)))


def test_profile_free_group_and_thermal_are_exempt(monkeypatch):
    instance = profile_assigner()
    group = make_profile_group()
    instance.cluster_builder.find_groups.return_value = [group]
    demand = Mock(side_effect=AssertionError("Exempt resources must not load demand"))
    monkeypatch.setattr("powergenome.generators.make_pre_dg_load_curves", demand)
    instance._assign_existing_profiles()
    assert instance.results.profile.iloc[0] is None
    instance.results["technology"] = "Natural Gas Fired Combined Cycle"
    instance.cluster_builder.find_groups.return_value = []
    instance._assign_existing_profiles()
    demand.assert_not_called()


def test_complete_profiles_do_not_require_demand(monkeypatch):
    instance = profile_assigner()
    profiles = tidy_profiles().assign(value=0.4)
    instance.cluster_builder.find_groups.return_value = [make_profile_group(profiles)]
    demand = Mock(side_effect=AssertionError("Complete profiles must not load demand"))
    monkeypatch.setattr("powergenome.generators.make_pre_dg_load_curves", demand)
    instance._assign_existing_profiles()
    np.testing.assert_allclose(instance.results.profile.iloc[0], np.full(24, 0.4))
    assert instance.results.capacity_mw.iloc[0] == 4


def test_no_load_lazily_loads_gross_demand_once(monkeypatch):
    instance = profile_assigner()
    instance.cluster_builder.find_groups.return_value = []
    demand = Mock(return_value=pd.DataFrame({"SD": [1584.86] * 24}))
    monkeypatch.setattr("powergenome.generators.make_pre_dg_load_curves", demand)
    instance._assign_existing_profiles()
    instance._assign_existing_profiles()
    demand.assert_called_once_with(settings=instance.settings)


def test_no_load_fails_closed_on_demand_error(monkeypatch):
    instance = profile_assigner()
    instance.cluster_builder.find_groups.return_value = []
    monkeypatch.setattr(
        "powergenome.generators.make_pre_dg_load_curves",
        Mock(side_effect=KeyError("demand_table")),
    )
    with pytest.raises(ValueError, match="including with --no-load"):
        instance._assign_existing_profiles()


@pytest.mark.parametrize("operating,raises", [(False, False), (True, True)])
def test_unclustered_nan_capacity_only_exempt_when_nonoperating(operating, raises):
    instance = profile_assigner(year=2030, capacity=np.nan)
    instance.all_gens = pd.DataFrame(
        {"Resource": instance.results.Resource, "operating": [operating]}
    )
    instance.cluster_builder.find_groups.return_value = []
    if raises:
        with pytest.raises(ValueError, match="Invalid existing-generator capacity"):
            instance._assign_existing_profiles()
    else:
        instance._assign_existing_profiles()
        np.testing.assert_array_equal(instance.results.profile.iloc[0], np.ones(8760))


def test_retired_multiweather_fallback_does_not_load_demand(monkeypatch):
    instance = profile_assigner(year=2030, capacity=0)
    instance.settings["weather_year"] = [2013, 2014]
    instance.cluster_builder.find_groups.return_value = []
    demand = Mock(side_effect=AssertionError("Retired resources must not load demand"))
    monkeypatch.setattr("powergenome.generators.make_pre_dg_load_curves", demand)
    instance._assign_existing_profiles()
    variability = make_generator_variability(instance.results)
    assert variability.shape == (17520, 1)
    demand.assert_not_called()


def test_sd_mixed_fleet_preserves_complete_profiles_and_capacity(caplog):
    instance = profile_assigner(gross_demand=pd.DataFrame({"SD": [1584.86] * 24}))
    instance.results = pd.DataFrame(
        {
            "Resource": ["SD_ror_1", "SD_hydro_1", "SD_wind_1", "SD_solar_1"],
            "technology": [
                "Run of River Hydroelectric",
                "Conventional Hydroelectric",
                "Onshore Wind Turbine",
                "Solar Photovoltaic",
            ],
            "region": "SD",
            "capacity_mw": [4, 1594, 3153, 214],
            "HYDRO": [1, 1, 0, 0],
            "VRE": [0, 0, 1, 1],
            "retirement_year": [2030, np.nan, np.nan, np.nan],
            "profile": None,
        }
    )
    original = instance.results.drop(columns="profile").copy()
    metadata = pd.DataFrame({"id": ["p38"], "region": ["p38"], "capacity_mw": [1644.4]})
    ror = make_profile_group(tidy_profiles().assign(site_id="p32"), metadata)
    complete = make_profile_group(tidy_profiles().assign(value=0.4))
    instance.cluster_builder.find_groups.side_effect = lambda **params: [
        ror if params.get("small") else complete
    ]
    instance._assign_existing_profiles()
    pd.testing.assert_frame_equal(instance.results.drop(columns="profile"), original)
    variability = make_generator_variability(instance.results)
    np.testing.assert_allclose(variability.iloc[:, 0], 1.0)
    np.testing.assert_allclose(variability.iloc[:, 1:], 0.4)
    assert "Affected operating capacity 4 MW" in caplog.text
    assert "1644.4 MW" not in caplog.text


def test_regionless_missing_profile_row_is_never_given_ones_profile():
    instance = profile_assigner(gross_demand=pd.DataFrame({"SD": [1584.86] * 24}))
    instance.results["region"] = np.nan
    instance.cluster_builder.find_groups.return_value = []
    with pytest.raises(ValueError, match="no model region"):
        instance._assign_existing_profiles()
    assert instance.results["profile"].isna().all()


def test_assignment_combines_missing_technologies_before_allowing_fallback():
    instance = profile_assigner(gross_demand=pd.DataFrame({"SD": [1000] * 24}))
    second = instance.results.iloc[0].copy()
    second["Resource"] = "SD_solar_1"
    second["technology"] = "Solar Photovoltaic"
    instance.results = pd.concat(
        [instance.results, second.to_frame().T], ignore_index=True
    )
    instance.results["capacity_mw"] = 3.0
    instance.cluster_builder.find_groups.return_value = []
    with pytest.raises(ValueError, match="Affected operating capacity 6 MW"):
        instance._assign_existing_profiles()
    assert instance.results.profile.isna().all()


def test_unexpected_cluster_errors_are_not_swallowed():
    instance = profile_assigner(gross_demand=pd.DataFrame({"SD": [1584.86] * 24}))
    group = Mock()
    group.profiles = object()
    group.metadata.read.return_value = pd.DataFrame({"region": ["p38"]})
    group.get_clusters.side_effect = ValueError("corrupt schema")
    instance.cluster_builder.find_groups.return_value = [group]
    with pytest.raises(ValueError, match="corrupt schema"):
        instance._assign_existing_profiles()


def test_multiweather_coverage_and_hour_weighted_budget():
    profiles = tidy_profiles()
    next_year = profiles.assign(weather_year=2014).query("time_index <= 12")
    group = make_profile_group(pd.concat([profiles, next_year]))
    result = group.get_clusters(
        max_clusters=1, weather_year=[2013, 2014], require_complete_profiles=True
    )
    assert len(result.profile.iloc[0]) == 36
    check_profile_fallback_budget(
        affected_rows([4]), pd.DataFrame({"SD": [1000] * 24 + [500] * 12})
    )
    incomplete = pd.concat([profiles, next_year.query("site_id == 0")])
    group = make_profile_group(incomplete)
    with pytest.raises(MissingProfileCoverage):
        group.get_clusters(
            max_clusters=1, weather_year=[2013, 2014], require_complete_profiles=True
        )
