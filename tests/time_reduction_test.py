import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest

from powergenome.GenX import reduce_time_domain as genx_reduce_time_domain
from powergenome.tdr import reduce_time_domain as tdr_reduce_time_domain
from powergenome.tdr.algorithms import cluster_periods
from powergenome.tdr.matrix import build_candidate_matrix
from powergenome.time_reduction import kmeans_time_clustering, max_rep_periods


@pytest.fixture
def test_data():
    """Fixture for setting up test data."""
    # Create a test DataFrame with hourly profiles
    resource_data = {
        "Resource_1": [i for i in range(400)],
        "Resource_2": [2 * i for i in range(400)],
    }
    load_data = {
        "Load_1": [3 * i for i in range(400)],
        "Load_2": [4 * i for i in range(400)],
    }

    resource_profiles = pd.DataFrame(resource_data)
    load_profiles = pd.DataFrame(load_data)

    return resource_profiles, load_profiles


def test_max_rep_periods(test_data):
    """Test the `max_rep_periods` function."""
    resource_profiles, load_profiles = test_data
    days_in_group = 4
    num_clusters = 4

    result = max_rep_periods(
        resource_profiles, load_profiles, days_in_group, num_clusters
    )
    (
        reduced_load_profiles,
        reduced_resouce_profiles,
        cluster_weights,
        time_series_mapping,
        rep_point_df,
    ) = result

    # Test if the reduced DataFrames have the correct number of hours (weeks * days in group * 24)
    num_hours = num_clusters * days_in_group * 24
    assert (
        len(reduced_load_profiles) == num_hours
    ), "Reduced load profiles length is incorrect"
    assert (
        len(reduced_resouce_profiles) == num_hours
    ), "Reduced resource profiles length is incorrect"

    # Test if cluster_weights have the correct length and are all 1
    assert len(cluster_weights) == num_clusters, "Cluster weights length is incorrect"
    assert all(
        weight == 1 for weight in cluster_weights
    ), "Cluster weights are not all 1"

    # Test if the time_series_mapping has the correct structure and mappings
    assert (
        time_series_mapping.shape[0] == num_clusters
    ), "Time series mapping length is incorrect"
    assert list(time_series_mapping["Period_Index"]) == list(
        range(1, num_clusters + 1)
    ), "Period index mismatch"
    assert list(time_series_mapping["Rep_Period_Index"]) == list(
        range(1, num_clusters + 1)
    ), "Representative period index mismatch"

    # Check if the representative period point dataframe is as expected
    expected_rep_points = [f"p{i}" for i in range(1, num_clusters + 1)]
    assert (
        list(rep_point_df["slot"]) == expected_rep_points
    ), "Representative points mismatch"


def test_kmeans_time_clustering(test_data):
    """Test the `kmeans_time_clustering` function."""
    resource_profiles, load_profiles = test_data
    days_in_group = 2
    num_clusters = 5

    results, representative_point, _ = kmeans_time_clustering(
        resource_profiles, load_profiles, days_in_group, num_clusters, n_init=1
    )

    reduced_resource_profiles = results["resource_profiles"]
    reduced_load_profiles = results["load_profiles"]
    time_series_mapping = results["time_series_mapping"]
    cluster_weights = results["ClusterWeights"]

    # Test if the reduced DataFrames have the correct number of hours (weeks * days in group * 24)
    num_hours = num_clusters * days_in_group * 24
    assert (
        len(reduced_load_profiles) == num_hours
    ), "Reduced load profiles length is incorrect"
    assert (
        len(reduced_resource_profiles) == num_hours
    ), "Reduced resource profiles length is incorrect"

    # Test if cluster_weights have the correct length and are all 1
    assert len(cluster_weights) == num_clusters, "Cluster weights length is incorrect"
    assert not all(
        weight == 1 for weight in cluster_weights
    ), "Cluster weights are all 1"

    # Test if the time_series_mapping has the correct structure and mappings
    assert (
        time_series_mapping["Rep_Period"].nunique() == num_clusters
    ), "Time series mapping length is incorrect"
    assert len(time_series_mapping["Period_Index"]) == int(
        len(load_profiles) / (days_in_group * 24)
    ), "Period index mismatch"


def test_kmeans_time_clustering_distinct_groups():
    # 3 clusters, 2 days per group, 24 hours per day
    days_in_group = 2
    num_clusters = 3
    hours_per_group = days_in_group * 24

    # Create 3 groups with distinct constant values
    group_values = [10, 50, 100]
    resource_profiles = pd.DataFrame(
        {
            "Resource_1": np.concatenate(
                [
                    np.full(hours_per_group, group_values[0]),
                    np.full(hours_per_group, group_values[1]),
                    np.full(hours_per_group, group_values[2]),
                ]
            ),
            "Resource_2": np.concatenate(
                [
                    np.full(hours_per_group, group_values[0] + 1),
                    np.full(hours_per_group, group_values[1] + 1),
                    np.full(hours_per_group, group_values[2] + 1),
                ]
            ),
        }
    )
    load_profiles = pd.DataFrame(
        {
            "Load_1": np.concatenate(
                [
                    np.full(hours_per_group, group_values[0] + 2),
                    np.full(hours_per_group, group_values[1] + 2),
                    np.full(hours_per_group, group_values[2] + 2),
                ]
            )
        }
    )

    results, rep_points, cluster_weights = kmeans_time_clustering(
        resource_profiles, load_profiles, days_in_group, num_clusters, n_init=10
    )

    # The representative points should correspond to the three distinct groups
    rep_indices = [int(s[1:]) for s in rep_points["slot"]]
    rep_indices.sort()
    assert rep_indices == [
        1,
        2,
        3,
    ], "Representative points do not match expected groups"

    # Each cluster weight should be 1 (since each group is unique)
    assert sorted(cluster_weights) == [1, 1, 1], "Cluster weights are not all 1"

    # The reduced profiles should have the correct values for each group
    reduced_resource = results["resource_profiles"]
    reduced_load = results["load_profiles"]
    for i, val in enumerate(group_values):
        start = i * hours_per_group
        end = (i + 1) * hours_per_group
        assert (reduced_resource.iloc[start:end, 0] == val).all()
        assert (reduced_resource.iloc[start:end, 1] == val + 1).all()
        assert (reduced_load.iloc[start:end, 0] == val + 2).all()


# ─────────────────────────────────────────────────────────────────────────────
# Time domain reduction backend (powergenome.tdr)
# ─────────────────────────────────────────────────────────────────────────────

HOURS_PER_DAY = 24


def _period_series(values, hours_per_period=HOURS_PER_DAY):
    """Expand one constant value per period into hourly rows."""
    return np.repeat(np.asarray(values, dtype=float), hours_per_period)


def _separated_frames(group_values=(10, 50, 100), hours_per_period=HOURS_PER_DAY):
    """Three well-separated groups of three equal, constant periods."""
    period_values = np.repeat(np.asarray(group_values, dtype=float), 3)
    resource = pd.DataFrame({"Wind_1": _period_series(period_values, hours_per_period)})
    load = pd.DataFrame({"Load_1": _period_series(period_values + 2, hours_per_period)})
    return resource, load


def test_candidate_matrix_orientation_and_normalize_scaling():
    hours_per_period = HOURS_PER_DAY
    load = pd.DataFrame({"Load_1": _period_series([10, 20, 30], hours_per_period)})
    resource = pd.DataFrame(
        {
            "Wind_1": _period_series([1, 2, 3], hours_per_period),
            "Constant": np.ones(3 * hours_per_period),
        }
    )

    candidate = build_candidate_matrix(
        load,
        resource,
        hours_per_period,
        scaling="normalize",
        demand_weight_factor=4,
    )

    # Rows are (feature, timestep within period) blocks, columns are periods.
    assert candidate.matrix.shape == (2 * hours_per_period, 3)
    assert candidate.load_feature_names == ["Load_1"]
    assert candidate.resource_feature_names == ["Wind_1"]
    assert candidate.excluded_resource_names == ["Constant"]
    assert candidate.n_periods == 3
    assert candidate.trailing_hours == 0

    load_block = candidate.matrix[:hours_per_period]
    assert np.allclose(load_block[:, 0], 0.0)
    # 0.5 min-max normalized, scaled by sqrt(4)
    assert np.allclose(load_block[:, 1], 1.0)
    assert np.allclose(load_block[:, 2], 2.0)

    resource_block = candidate.matrix[hours_per_period:]
    assert np.allclose(resource_block[:, 0], 0.0)
    assert np.allclose(resource_block[:, 1], 0.5)
    assert np.allclose(resource_block[:, 2], 1.0)


def test_candidate_matrix_standardize_scales_each_feature():
    load = pd.DataFrame({"Load_1": _period_series([10, 20, 30])})
    resource = pd.DataFrame({"Wind_1": _period_series([1, 5, 9])})

    candidate = build_candidate_matrix(load, resource, HOURS_PER_DAY, "standardize")
    matrix = candidate.matrix

    assert matrix[:HOURS_PER_DAY].mean() == pytest.approx(0)
    assert matrix[:HOURS_PER_DAY].std() == pytest.approx(1)
    assert matrix[HOURS_PER_DAY:].mean() == pytest.approx(0)
    assert matrix[HOURS_PER_DAY:].std() == pytest.approx(1)
    # Each feature is standardized on its own scale, so the blocks match.
    assert np.allclose(matrix[:HOURS_PER_DAY], matrix[HOURS_PER_DAY:])


def test_variable_resources_only_false_keeps_every_resource_feature():
    load = pd.DataFrame({"Load_1": _period_series([10, 20, 30, 40])})
    resource = pd.DataFrame(
        {
            "Wind_1": _period_series([1, 2, 3, 4]),
            "Constant_5": np.full(4 * HOURS_PER_DAY, 5.0),
        }
    )

    kept = build_candidate_matrix(
        load, resource, HOURS_PER_DAY, variable_resources_only=False
    )
    assert kept.resource_feature_names == ["Wind_1", "Constant_5"]
    assert kept.excluded_resource_names == []

    filtered = build_candidate_matrix(
        load, resource, HOURS_PER_DAY, variable_resources_only=True
    )
    assert filtered.resource_feature_names == ["Wind_1"]
    assert filtered.excluded_resource_names == ["Constant_5"]

    reduced_resource, _, _, _ = tdr_reduce_time_domain(
        resource,
        load,
        pd.DataFrame(),
        days_per_period=1,
        num_periods=2,
        include_peak_day=False,
        variable_resources_only=False,
    )
    assert (reduced_resource["Constant_5"] == 5.0).all()

    filtered_resource, _, _, _ = tdr_reduce_time_domain(
        resource,
        load,
        pd.DataFrame(),
        days_per_period=1,
        num_periods=2,
        include_peak_day=False,
        variable_resources_only=True,
    )
    assert (filtered_resource["Constant_5"] == 1.0).all()


@pytest.mark.parametrize("method", ["kmeans", "kmedoids"])
def test_cluster_periods_recovers_separated_groups(method):
    resource, load = _separated_frames()
    candidate = build_candidate_matrix(load, resource, HOURS_PER_DAY)

    result = cluster_periods(candidate.matrix, 3, method=method, n_restarts=3)

    # Cluster order is arbitrary; the members of each cluster are not.
    assert sorted(result.representatives.tolist()) == [0, 3, 6]
    assert sorted(result.counts.tolist()) == [3, 3, 3]
    assert len(set(result.representatives.tolist())) == 3


@pytest.mark.parametrize("method", ["kmeans", "kmedoids"])
def test_reduce_time_domain_separated_groups(method):
    resource, load = _separated_frames()

    reduced_resource, load_output, mapping, reps = tdr_reduce_time_domain(
        resource,
        load,
        pd.DataFrame(),
        days_per_period=1,
        num_periods=3,
        include_peak_day=False,
        method=method,
    )

    assert reps["slot"].tolist() == ["p1", "p4", "p7"]
    assert mapping["Rep_Period"].nunique() == 3
    assert len(reduced_resource) == 3 * HOURS_PER_DAY
    assert len(load_output) == 3 * HOURS_PER_DAY

    sub_weights = load_output["Sub_Weights"]
    assert sub_weights.sum() == 9 * HOURS_PER_DAY
    # Sub_Weights carries one value per representative period, like GenX expects.
    per_period = sub_weights.dropna()
    assert np.allclose(per_period, per_period.round())
    assert per_period.tolist() == [3 * HOURS_PER_DAY] * 3

    # Raw resource profiles of the representative periods, in sorted order.
    assert (reduced_resource["Wind_1"] == _period_series([10, 50, 100])).all()
    assert (load_output["Load_1"] == _period_series([12, 52, 102])).all()


def test_peak_period_is_forced_representative():
    load = pd.DataFrame(
        {"Load_1": _period_series([10, 20, 30, 99, 40, 50], HOURS_PER_DAY)}
    )
    resource = pd.DataFrame(
        {"Wind_1": _period_series([1, 2, 3, 4, 5, 6], HOURS_PER_DAY)}
    )

    _, load_output, mapping, reps = tdr_reduce_time_domain(
        resource,
        load,
        pd.DataFrame(),
        days_per_period=1,
        num_periods=3,
        include_peak_day=True,
    )

    slots = reps["slot"].tolist()
    assert len(slots) == len(set(slots)) == 3
    assert "p4" in slots
    # The forced period is its own representative and is not clustered as well.
    assert mapping.loc[mapping["Period_Index"] == 4, "Rep_Period"].item() == 4
    assert (mapping["Rep_Period"] == 4).sum() == 1
    assert load_output["Sub_Weights"].sum() == 6 * HOURS_PER_DAY


def test_trailing_hours_preserved_and_warned(test_data, caplog):
    resource_profiles, load_profiles = test_data

    with caplog.at_level(logging.WARNING):
        _, load_output, mapping, _ = tdr_reduce_time_domain(
            resource_profiles,
            load_profiles,
            pd.DataFrame(),
            days_per_period=3,
            num_periods=2,
            include_peak_day=False,
        )

    # 400 hours hold five complete 72-hour periods and 40 trailing hours.
    assert len(mapping) == 5
    sub_weights = load_output["Sub_Weights"]
    assert sub_weights.sum() == 400
    per_period = sub_weights.dropna()
    assert np.allclose(per_period, per_period.round())

    warnings = [
        record for record in caplog.records if record.levelno == logging.WARNING
    ]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "400 hours" in message
    assert "72-hour periods" in message
    assert "trailing 40 hours" in message
    assert "Sub_Weights still sum to 400 hours" in message


def test_evenly_divisible_horizon_does_not_warn(test_data, caplog):
    resource_profiles, load_profiles = test_data

    with caplog.at_level(logging.WARNING):
        _, load_output, _, _ = tdr_reduce_time_domain(
            resource_profiles.iloc[:360],
            load_profiles.iloc[:360],
            pd.DataFrame(),
            days_per_period=3,
            num_periods=2,
            include_peak_day=False,
        )

    assert not [
        record for record in caplog.records if record.levelno == logging.WARNING
    ]
    assert load_output["Sub_Weights"].sum() == 360


def test_period_map_and_reduced_shape_full_year():
    hours = 8760
    hours_per_period = 7 * HOURS_PER_DAY
    daily = np.arange(HOURS_PER_DAY, dtype=float)
    load = pd.DataFrame({"Load_1": np.tile(daily, hours // HOURS_PER_DAY)})
    resource = pd.DataFrame({"Wind_1": np.tile(daily[::-1], hours // HOURS_PER_DAY)})

    reduced_resource, load_output, mapping, reps = tdr_reduce_time_domain(
        resource,
        load,
        pd.DataFrame(),
        days_per_period=7,
        num_periods=52,
    )

    # Row count is part of the GenX contract: periods * days * 24.
    assert len(reduced_resource) == 52 * 7 * HOURS_PER_DAY
    assert len(load_output) == 52 * 7 * HOURS_PER_DAY
    assert list(mapping.columns) == [
        "Period_Index",
        "Rep_Period_Index",
        "Month",
        "Rep_Period",
    ]
    assert mapping["Period_Index"].tolist() == list(range(1, 53))
    assert mapping["Rep_Period"].isin(mapping["Period_Index"]).all()
    assert set(mapping["Rep_Period"]) == set(reps["slot"].str[1:].astype(int))
    assert mapping["Month"].is_monotonic_increasing
    assert list(load_output["Time_Index"]) == list(range(1, 52 * 7 * HOURS_PER_DAY + 1))
    # The 24 trailing hours of the year still carry weight.
    assert load_output["Sub_Weights"].sum() == hours


def _mock_demand_segments(monkeypatch):
    monkeypatch.setattr(
        "powergenome.GenX.load_demand_segments", lambda settings=None: pd.DataFrame()
    )


def _genx_tdr_settings(**overrides):
    settings = {
        "reduce_time_domain": True,
        "time_domain_days_per_period": 2,
        "time_domain_periods": 3,
        "include_peak_day": True,
        "demand_weight_factor": 1,
    }
    settings.update(overrides)
    return settings


def test_legacy_kmeans_method_still_available(monkeypatch):
    _mock_demand_segments(monkeypatch)
    rng = np.random.default_rng(0)
    hours = 400
    load = pd.DataFrame({"Load_1": 100 + rng.normal(size=hours)})
    resource = pd.DataFrame({"Wind_1": rng.uniform(size=hours)})

    reduced_resource, load_output, mapping, reps = genx_reduce_time_domain(
        resource,
        load,
        settings=_genx_tdr_settings(
            time_domain_reduction_method="legacy_kmeans", tdr_n_init=1
        ),
    )

    assert len(reduced_resource) == 3 * 2 * HOURS_PER_DAY
    assert len(load_output) == 3 * 2 * HOURS_PER_DAY
    assert mapping["Rep_Period"].nunique() == 3
    assert len(reps) == 3


def test_default_method_uses_new_backend(monkeypatch):
    _mock_demand_segments(monkeypatch)
    rng = np.random.default_rng(0)
    hours = 400
    load = pd.DataFrame({"Load_1": 100 + rng.normal(size=hours)})
    resource = pd.DataFrame({"Wind_1": rng.uniform(size=hours)})

    _, load_output, _, _ = genx_reduce_time_domain(
        resource, load, settings=_genx_tdr_settings()
    )

    # The legacy backend drops the trailing hours; the new one keeps all 400.
    assert load_output["Sub_Weights"].sum() == hours


# ─────────────────────────────────────────────────────────────────────────────
# VRE profile grouping
# ─────────────────────────────────────────────────────────────────────────────


def _vre_frames():
    """Three wind sites plus one solar site in a single region."""
    hours_per_period = HOURS_PER_DAY
    n_periods = 6
    hours = n_periods * hours_per_period
    steps = np.arange(hours, dtype=float)
    rng = np.random.default_rng(1)

    resource = pd.DataFrame(
        {
            "wind_a": rng.uniform(0.1, 0.9, hours),
            "wind_b": rng.uniform(0.1, 0.9, hours),
            "wind_c": rng.uniform(0.1, 0.9, hours),
            "solar_a": np.maximum(0, np.sin(steps / hours_per_period * 2 * np.pi)),
        }
    )
    load = pd.DataFrame(
        {"Load_1": 100 + 50 * np.sin(steps / hours_per_period * 2 * np.pi)}
    )
    metadata = pd.DataFrame(
        {
            "region": ["R1"] * 4,
            "technology": ["wind", "wind", "wind", "utilitypv"],
            "VRE": [1, 1, 1, 1],
        },
        index=pd.Index(["wind_a", "wind_b", "wind_c", "solar_a"], name="Resource"),
    )
    return resource, load, metadata


def test_vre_profile_grouping_builds_one_feature_per_region_technology():
    resource, load, metadata = _vre_frames()
    ungrouped = build_candidate_matrix(load, resource, HOURS_PER_DAY)
    grouped = build_candidate_matrix(
        load,
        resource,
        HOURS_PER_DAY,
        vre_profile_grouping="region_technology",
        resource_metadata=metadata,
    )

    assert ungrouped.resource_feature_names == [
        "wind_a",
        "wind_b",
        "wind_c",
        "solar_a",
    ]
    assert grouped.resource_feature_names == ["R1|wind", "R1|utilitypv"]
    assert grouped.matrix.shape[1] == ungrouped.matrix.shape[1] == 6

    # The wind group is exactly the normalized mean of its members' raw profiles.
    mean_profile = resource[["wind_a", "wind_b", "wind_c"]].to_numpy().mean(axis=1)
    expected = (mean_profile - mean_profile.min()) / (
        mean_profile.max() - mean_profile.min()
    )
    expected = expected.reshape(6, HOURS_PER_DAY).T
    wind_block = grouped.matrix[HOURS_PER_DAY : 2 * HOURS_PER_DAY]
    assert np.allclose(wind_block, expected)


def test_vre_profile_grouping_keeps_all_resource_columns():
    resource, load, metadata = _vre_frames()

    for grouping in ("none", "region_technology"):
        reduced_resource, _, _, _ = tdr_reduce_time_domain(
            resource,
            load,
            pd.DataFrame(),
            days_per_period=1,
            num_periods=2,
            include_peak_day=False,
            vre_profile_grouping=grouping,
            resource_metadata=metadata,
        )
        assert list(reduced_resource.columns) == [
            "wind_a",
            "wind_b",
            "wind_c",
            "solar_a",
        ]
        assert len(reduced_resource) == 2 * HOURS_PER_DAY


def test_vre_profile_grouping_missing_metadata_falls_back(caplog):
    resource, load, metadata = _vre_frames()
    metadata = metadata.drop(index="wind_b")

    with caplog.at_level(logging.WARNING):
        candidate = build_candidate_matrix(
            load,
            resource,
            HOURS_PER_DAY,
            vre_profile_grouping="region_technology",
            resource_metadata=metadata,
        )

    assert candidate.resource_feature_names == ["R1|wind", "wind_b", "R1|utilitypv"]
    warnings = [
        record for record in caplog.records if record.levelno == logging.WARNING
    ]
    assert len(warnings) == 1
    assert "1 resource(s)" in warnings[0].getMessage()


def test_vre_profile_grouping_none_reproduces_ungrouped_matrix():
    resource, load, metadata = _vre_frames()

    without_metadata = build_candidate_matrix(load, resource, HOURS_PER_DAY)
    explicit_none = build_candidate_matrix(
        load,
        resource,
        HOURS_PER_DAY,
        vre_profile_grouping="none",
        resource_metadata=metadata,
    )

    assert np.array_equal(without_metadata.matrix, explicit_none.matrix)
    assert (
        explicit_none.resource_feature_names == without_metadata.resource_feature_names
    )


def test_vre_profile_grouping_without_metadata_warns(caplog):
    resource, load, _ = _vre_frames()

    with caplog.at_level(logging.WARNING):
        candidate = build_candidate_matrix(
            load,
            resource,
            HOURS_PER_DAY,
            vre_profile_grouping="region_technology",
        )

    assert candidate.resource_feature_names == [
        "wind_a",
        "wind_b",
        "wind_c",
        "solar_a",
    ]
    warnings = [
        record for record in caplog.records if record.levelno == logging.WARNING
    ]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "no resource metadata was supplied" in message
