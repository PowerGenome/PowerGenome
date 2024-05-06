from typing import List, Tuple

import pandas as pd
import pytest

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
    days_in_group = 7
    num_clusters = 2

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
        list(representative_point["slot"]) == expected_rep_points
    ), "Representative points mismatch"
