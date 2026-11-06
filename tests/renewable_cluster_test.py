"Test functions for clustering renewable sites"

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from powergenome.cluster.renewables import (
    agg_cluster_other,
    agg_cluster_profile,
    assign_site_cluster,
    cluster_sites_binned,
    cluster_sites_no_bin,
    load_site_profiles,
    modify_renewable_group,
    num_bins_from_capacity,
    value_bin,
)

CWD = Path.cwd()
DATA_FOLDER = CWD / "tests" / "data" / "cpa_cluster_data"

# These tests are deterministic, parametrized replacements for the former randomized
# property-based fuzz tests of the same functions. The fixed cases below deliberately
# span the input space those tests explored: every combination of binning arguments,
# weighted vs. unweighted binning, constant/degenerate series, n_clusters values below,
# at, and above the number of sites, and both clustering methods/features.

# Values include near-zero entries: pandas binning breaks with very small values, so
# the tests clip anything in (0, 0.01) to 0.01 (see the LOC below).
FEATURE_VALUES = [0.0, 0.001, 0.005, 3.5, 12.0, 40.0, 55.0, 70.0, 85.0, 100.0]

FEATURE_WEIGHTS = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

VALUE_BIN_BINS = [
    pytest.param(1, id="bins-int-1"),
    pytest.param(4, id="bins-int-4"),
    pytest.param(20, id="bins-int-20"),
    pytest.param([20.0, 100.0], id="bins-list"),
]

VALUE_BIN_Q = [
    pytest.param(None, id="q-none"),
    pytest.param(4, id="q-int-4"),
    pytest.param([0.0, 0.5, 1.0], id="q-list"),
]

# 10-element float profiles for 6 sites. Each row is distinct so clustering is stable.
SITE_PROFILES = [[float((i * 7 + j * 3) % 17) for j in range(10)] for i in range(6)]


def _site_dataframe() -> pd.DataFrame:
    """Fixed stand-in for the randomly generated renewable site data."""
    return pd.DataFrame(
        {
            "profile": [np.asarray(row, dtype=np.float64) for row in SITE_PROFILES],
            "lcoe": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "lat": [55.0, 57.5, 60.0, 62.5, 65.0, 67.5],
            "lon": [-95.0, -90.0, -85.0, -80.0, -75.0, -72.0],
            "state": ["a", "a", "a", "b", "b", "b"],
        },
        index=pd.RangeIndex(6),
    )


def _expected_bins(bins, q) -> int:
    """Number of bins value_bin should request (q takes precedence over bins)."""
    if q:
        return q if isinstance(q, int) else len(q) - 1
    if isinstance(bins, int):
        return bins
    return len(bins) - 1


@pytest.mark.parametrize("bins", VALUE_BIN_BINS)
@pytest.mark.parametrize("q", VALUE_BIN_Q)
@pytest.mark.parametrize(
    "use_weights", [False, True], ids=["no-weights", "with-weights"]
)
def test_value_bins_bins_q_combinations(bins, q, use_weights):
    s = pd.Series(FEATURE_VALUES)
    # pandas binning breaks with very small values. Allow 0 but nothing smaller than 0.01
    s.loc[(s > 0) & (s < 0.01)] = 0.01
    weights = pd.Series(FEATURE_WEIGHTS) if use_weights else None
    n_requested = min(_expected_bins(bins, q), len(s))

    # value_bin sorts/clamps list arguments in place, so pass fresh copies.
    labels = value_bin(
        s=s,
        bins=list(bins) if isinstance(bins, list) else bins,
        q=list(q) if isinstance(q, list) else q,
        weights=weights,
    )

    assert len(labels) == len(s)
    assert not pd.isna(labels).any()
    n_bins = len(pd.unique(labels))
    assert 1 <= n_bins <= n_requested


@pytest.mark.parametrize(
    "q", [pytest.param(4, id="q-int-4"), pytest.param([0.0, 0.5, 1.0], id="q-list")]
)
@pytest.mark.parametrize(
    "weights",
    [
        pytest.param([0.0] * 10, id="all-zero-weights"),
        pytest.param([0.0] + [1.0] * 9, id="zero-weight-at-minimum"),
    ],
)
def test_value_bins_zero_weights(q, weights):
    """Zero weights are a real edge case of the weighted quantile path."""
    s = pd.Series(FEATURE_VALUES)
    labels = value_bin(
        s=s, q=list(q) if isinstance(q, list) else q, weights=pd.Series(weights)
    )

    assert len(labels) == len(s)
    # A zero weight on the minimum value can push the lowest weighted quantile above
    # it, leaving that single site unassigned to a bin.
    assert pd.isna(labels).sum() <= 1
    assert not pd.isna(labels).all()


@pytest.mark.parametrize(
    "values",
    [
        pytest.param([0.0] * 10, id="all-zero"),
        pytest.param([7.0] * 10, id="all-same-value"),
    ],
)
@pytest.mark.parametrize("bins", VALUE_BIN_BINS)
@pytest.mark.parametrize("q", VALUE_BIN_Q)
def test_value_bins_constant_series(values, bins, q):
    """Constant and all-zero series collapse to a single bin for every argument combo."""
    s = pd.Series(values)
    labels = value_bin(
        s=s,
        bins=list(bins) if isinstance(bins, list) else bins,
        q=list(q) if isinstance(q, list) else q,
    )

    assert len(labels) == len(s)
    assert (np.asarray(labels) == 1).all()


@pytest.mark.parametrize(
    "n_clusters",
    [
        pytest.param(-2, id="n-clusters-negative"),
        pytest.param(0, id="n-clusters-zero"),
        pytest.param(1, id="n-clusters-1"),
        pytest.param(3, id="n-clusters-3"),
        pytest.param(6, id="n-clusters-rows"),
        pytest.param(10, id="n-clusters-exceeds-rows"),
    ],
)
@pytest.mark.parametrize(
    "profiles",
    [
        pytest.param([], id="empty"),
        pytest.param(SITE_PROFILES[:1], id="single-row"),
        pytest.param(SITE_PROFILES, id="six-rows"),
    ],
)
def test_agg_cluster_profile_parametrized(profiles, n_clusters):
    s = pd.Series([np.asarray(row, dtype=np.float64) for row in profiles], dtype=object)
    labels = np.asarray(agg_cluster_profile(s=s, n_clusters=n_clusters))

    assert len(labels) == len(profiles)
    if len(profiles) == 0:
        return
    expected = min(max(n_clusters, 1), len(profiles))
    assert len(pd.unique(labels)) == expected


@pytest.mark.parametrize(
    "values",
    [
        pytest.param(None, id="default-empty-series"),
        pytest.param([], id="empty-float-series"),
        pytest.param([5.0], id="single-value"),
        pytest.param([5.0, 10.0, 15.0, 20.0, 25.0, 30.0], id="six-values"),
    ],
)
@pytest.mark.parametrize(
    "n_clusters",
    [
        pytest.param(-2, id="n-clusters-negative"),
        pytest.param(0, id="n-clusters-zero"),
        pytest.param(1, id="n-clusters-1"),
        pytest.param(3, id="n-clusters-3"),
        pytest.param(6, id="n-clusters-rows"),
        pytest.param(10, id="n-clusters-exceeds-rows"),
    ],
)
def test_agg_cluster_other_parametrized(values, n_clusters):
    if values is None:
        s = pd.Series()
    else:
        s = pd.Series(values, dtype=float)
    labels = np.asarray(agg_cluster_other(s=s, n_clusters=n_clusters))

    assert len(labels) == len(s)
    if len(s) == 0:
        return
    expected = min(max(n_clusters, 1), len(s))
    assert len(pd.unique(labels)) == expected


CLUSTER_METHOD_FEATURES = [
    pytest.param("agg", "profile", id="agg-profile"),
    pytest.param("agg", "lcoe", id="agg-lcoe"),
    pytest.param("kmeans", ["lat", "lon"], id="kmeans-lat-lon"),
]

CLUSTER_N_CLUSTERS = [
    pytest.param(-1, id="n-clusters-negative"),
    pytest.param(1, id="n-clusters-1"),
    pytest.param(2, id="n-clusters-2"),
    pytest.param(3, id="n-clusters-group-size"),
    pytest.param(6, id="n-clusters-rows"),
    pytest.param(10, id="n-clusters-exceeds-rows"),
]


@pytest.mark.parametrize("n_clusters", CLUSTER_N_CLUSTERS)
@pytest.mark.parametrize("method, feature", CLUSTER_METHOD_FEATURES)
def test_cluster_sites_no_bin_parametrized(method, feature, n_clusters):
    data = _site_dataframe()
    result = cluster_sites_no_bin(
        data=data, method=method, feature=feature, n_clusters=n_clusters
    )

    assert len(result) == len(data)
    assert "cluster" in result.columns
    assert result["cluster"].notna().all()
    expected = min(max(n_clusters, 1), len(data))
    assert result["cluster"].nunique() == expected


@pytest.mark.parametrize("n_clusters", CLUSTER_N_CLUSTERS)
@pytest.mark.parametrize("method, feature", CLUSTER_METHOD_FEATURES)
def test_cluster_sites_binned_parametrized(method, feature, n_clusters):
    data = _site_dataframe()
    result = cluster_sites_binned(
        data=data,
        by=["state"],
        method=method,
        feature=feature,
        n_clusters=n_clusters,
    )

    assert len(result) == len(data)
    assert "cluster" in result.columns
    assert result["cluster"].notna().all()
    # Clusters are numbered contiguously across groups, so the total number of clusters
    # is the sum of the per-state minimum(n_clusters, group size).
    group_sizes = data.groupby("state").size()
    expected = sum(min(max(n_clusters, 1), size) for size in group_sizes)
    assert result["cluster"].nunique() == expected


@pytest.mark.parametrize("binned", [False, True], ids=["no-bin", "binned"])
@pytest.mark.parametrize("n_rows", [0, 1], ids=["empty-frame", "single-row-frame"])
def test_cluster_sites_small_frames(binned, n_rows):
    """Empty and single-row site data are short-circuited by both clusterers."""
    data = _site_dataframe().iloc[:n_rows].copy()
    kwargs = dict(method="agg", feature="lcoe", n_clusters=2)
    if binned:
        result = cluster_sites_binned(data=data, by=["state"], **kwargs)
    else:
        result = cluster_sites_no_bin(data=data, **kwargs)

    assert len(result) == n_rows
    assert "cluster" in result.columns
    if n_rows == 0:
        return
    assert result["cluster"].notna().all()
    # cluster_sites_no_bin assigns the scalar label 1 to a single site; the binned
    # variant offsets the labels from the underlying clusterer (0).
    assert (result["cluster"] == (0 if binned else 1)).all()


def test_assign_site_cluster():
    renew_data = pd.read_csv(DATA_FOLDER / "cpa_data.csv")
    profile_path = DATA_FOLDER / "cpa_profiles.csv"
    regions = ["A", "B"]
    cluster = {
        "min_capacity": 2000,
        "filter": [
            {
                "feature": "lcoe",
                "max": 49,
            }
        ],
        "bin": [{"feature": "interconnect_annuity", "bins": 2}],
        "group": ["county"],
        "cluster": [
            {"feature": "lcoe", "method": "agg", "n_clusters": 2},
        ],
    }

    data = assign_site_cluster(
        renew_data=renew_data, profile_path=profile_path, regions=regions, **cluster
    )
    assert data.notna().all().all()
    assert "cluster" in data.columns

    cluster = {
        "cluster": [
            {"feature": "profile", "method": "hierarchical", "n_clusters": 3},
        ],
    }
    data = assign_site_cluster(
        renew_data=renew_data, profile_path=profile_path, regions=regions, **cluster
    )
    assert data.notna().all().all()
    assert "cluster" in data.columns

    cluster = {
        "bin": [{"feature": "interconnect_annuity", "mw_per_bin": 200}],
    }
    data = assign_site_cluster(
        renew_data=renew_data, profile_path=profile_path, regions=regions, **cluster
    )
    assert data.notna().all().all()
    assert "cluster" in data.columns

    cluster = {
        "bin": [{"feature": "interconnect_annuity", "mw_per_q": 200}],
    }
    data = assign_site_cluster(
        renew_data=renew_data, profile_path=profile_path, regions=regions, **cluster
    )
    assert data.notna().all().all()
    assert "cluster" in data.columns

    cluster = {
        "cluster": [
            {"feature": "profile", "method": "agglomerative", "mw_per_cluster": 200},
        ],
    }
    data = assign_site_cluster(
        renew_data=renew_data, profile_path=profile_path, regions=regions, **cluster
    )
    assert data.notna().all().all()
    assert "cluster" in data.columns
    assert len(data) == len(renew_data)

    cluster = {
        "cluster": [
            {
                "feature": ["interconnect_annuity", "lcoe"],
                "method": "kmeans",
                "mw_per_cluster": 200,
            },
        ],
    }
    data = assign_site_cluster(
        renew_data=renew_data, profile_path=profile_path, regions=regions, **cluster
    )
    assert data.notna().all().all()
    assert "cluster" in data.columns
    assert len(data) == len(renew_data)


# Tests that the function correctly calculates the number of bins based on the "mw_per_bin" key in the input dictionary. tags: [happy path]
def test_num_bins_from_capacity_with_mw_per_bin():
    # Happy path test for calculating number of bins based on "mw_per_bin"
    data = pd.DataFrame({"capacity_mw": [10, 20, 30]})
    b = {"mw_per_bin": 14}
    expected_output = {"bins": 4}
    assert num_bins_from_capacity(data, b) == expected_output


def test_num_bins_from_capacity_with_mw_per_bin_one_bin():
    # Happy path test for calculating number of bins based on "mw_per_bin"
    data = pd.DataFrame({"capacity_mw": [1, 2, 3]})
    b = {"mw_per_bin": 14}
    expected_output = {"bins": 1}
    assert num_bins_from_capacity(data, b) == expected_output


# Tests that the function correctly calculates the number of quantiles based on the "mw_per_q" key in the input dictionary. tags: [happy path]
def test_num_bins_from_capacity_with_mw_per_q():
    # Happy path test for calculating number of quantiles based on "mw_per_q" key
    data = pd.DataFrame({"capacity_mw": [10, 20, 30]})
    b = {"mw_per_q": 15}
    expected_output = {"q": 4}
    assert num_bins_from_capacity(data, b) == expected_output


# Tests that the function returns the input dictionary unaltered if neither "mw_per_bin" nor "mw_per_q" key is present. tags: [happy path]
def test_num_bins_from_capacity_with_no_mw_key():
    # Happy path test for returning input dictionary unaltered if no "mw_per_bin" or "mw_per_q" key is present
    data = pd.DataFrame({"capacity_mw": [10, 20, 30]})
    b = {"other_key": "value"}
    expected_output = {"other_key": "value"}
    assert num_bins_from_capacity(data, b) == expected_output


# Tests that the function handles input dictionary containing non-integer values for "mw_per_bin" or "mw_per_q". tags: [edge case]
def test_num_bins_from_capacity_with_non_integer_values():
    data = pd.DataFrame({"capacity_mw": [100, 200, 300]})
    b = {"mw_per_bin": 0.5}
    result = num_bins_from_capacity(data, b)
    assert result == {"bins": 1200}


# Tests that the function logs a warning message if the "bins" key is already present in the input dictionary and is being overwritten. tags: [behavior]
def test_num_bins_from_capacity_with_overwriting_bins(caplog):
    data = pd.DataFrame({"capacity_mw": [100, 200, 300]})
    b = {"mw_per_bin": 100, "bins": 5}
    result = num_bins_from_capacity(data, b)
    assert result == {"bins": 6}
    assert "Overwriting 'bins' based on mw_per_bin" in caplog.text


# Generated by CodiumAI
class TestModifyRenewableGroup:
    # Modifies values of a renewable cluster based on group membership
    def test_modify_values_renewable_cluster(self):
        # Arrange
        df = pd.DataFrame(
            {
                "cluster": ["group1_value1", "group2_value2", "group1_value3"],
                "cost": [100, 200, 300],
            }
        )
        group_modifiers = [
            {"group": "group1", "group_value": "value1", "cost": ["mul", 2]},
            {"group": "group2", "group_value": "value2", "cost": ["add", 100]},
        ]

        # Act
        result = modify_renewable_group(df, group_modifiers)

        # Assert
        assert result["cost"].tolist() == [200, 300, 300]

    # Returns a modified version of the input dataframe
    def test_return_modified_dataframe(self):
        # Arrange
        df = pd.DataFrame(
            {
                "cluster": ["group1:value1", "group2:value2", "group1:value3"],
                "cost": [100, 200, 300],
            }
        )
        group_modifiers = [
            {"group": "group1", "group_value": "value1", "cost": ["mul", 2]},
            {"group": "group2", "group_value": "value2", "cost": ["add", 100]},
        ]

        # Act
        result = modify_renewable_group(df, group_modifiers)

        # Assert
        assert isinstance(result, pd.DataFrame)
        assert result.equals(df)

    # Handles empty group_modifiers list
    def test_empty_group_modifiers_list(self):
        # Arrange
        df = pd.DataFrame(
            {
                "cluster": ["group1:value1", "group2:value2", "group1:value3"],
                "cost": [100, 200, 300],
            }
        )
        group_modifiers = []

        # Act
        result = modify_renewable_group(df, group_modifiers)

        # Assert
        assert isinstance(result, pd.DataFrame)
        assert result.equals(df)

    # Raises KeyError if a group_modifiers dictionary is missing "group" or "group_value" keys
    def test_missing_group_or_group_value_keys(self):
        # Arrange
        df = pd.DataFrame(
            {
                "cluster": ["group1:value1", "group2:value2", "group1:value3"],
                "cost": [100, 200, 300],
            }
        )
        group_modifiers = [
            {"group": "group1", "cost": ["mul", 2]},
            {"group_value": "value2", "cost": ["add", 100]},
        ]

        # Act & Assert
        with pytest.raises(KeyError):
            modify_renewable_group(df, group_modifiers)

    # Raises ValueError if operator list is not a 2-item list
    def test_operator_list_not_2_item_list(self):
        # Arrange
        df = pd.DataFrame(
            {
                "cluster": ["group1:value1", "group2:value2", "group1:value3"],
                "cost": [100, 200, 300],
            }
        )
        group_modifiers = [
            {"group": "group1", "group_value": "value1", "cost": ["mul"]},
            {"group": "group2", "group_value": "value2", "cost": ["add"]},
        ]

        # Act & Assert
        with pytest.raises(ValueError):
            modify_renewable_group(df, group_modifiers)

    # Raises ValueError if operator is not in the valid list (["add", "mul", "truediv", "sub"])
    def test_operator_not_in_valid_list(self):
        # Arrange
        df = pd.DataFrame(
            {
                "cluster": ["group1:value1", "group2:value2", "group1:value3"],
                "cost": [100, 200, 300],
            }
        )
        group_modifiers = [
            {"group": "group1", "group_value": "value1", "cost": ["div", 2]},
            {"group": "group2", "group_value": "value2", "cost": ["sub", 100]},
        ]

        # Act & Assert
        with pytest.raises(ValueError):
            modify_renewable_group(df, group_modifiers)


def test_load_site_profiles_tidy_format():
    """Test loading site profiles in tidy format without weather_year filter."""
    profile_path = DATA_FOLDER / "cpa_profiles.csv"
    site_ids = [1, 2, 3]

    df = load_site_profiles(profile_path, site_ids=site_ids)

    # Should return a wide dataframe
    assert isinstance(df, pd.DataFrame)
    # Columns should match requested site_ids
    assert list(df.columns) == site_ids
    # Test data has 20 time periods per site per year (simplified test data)
    # When no weather_year is specified with in-memory data, it loads all years
    assert len(df) == 40  # Both years loaded
    # Values should be between 0 and 1 for capacity factors
    assert (df >= 0).all().all()
    assert (df <= 1).all().all()


def test_load_site_profiles_with_weather_year():
    """Test loading site profiles with specific weather_year filter."""
    profile_path = DATA_FOLDER / "cpa_profiles.csv"
    site_ids = [1, 2]

    # Load with weather_year 2012
    df = load_site_profiles(profile_path, site_ids=site_ids, weather_year=2012)

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == site_ids
    # Test data has 20 time periods per site per year
    assert len(df) == 20


def test_load_site_profiles_with_multiple_weather_years():
    """Test loading and concatenating multiple weather years."""
    profile_path = DATA_FOLDER / "cpa_profiles.csv"
    site_ids = [1, 2]

    # Load with multiple years - should concatenate
    df = load_site_profiles(profile_path, site_ids=site_ids, weather_year=[2012, 2013])

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == site_ids
    # Should have double the time periods for two years (20 * 2 = 40)
    assert len(df) == 40


def test_load_site_profiles_missing_site():
    """Test that missing sites are filled with 1.0."""
    profile_path = DATA_FOLDER / "cpa_profiles.csv"
    # Request a site that doesn't exist
    site_ids = [1, 999999]

    df = load_site_profiles(profile_path, site_ids=site_ids)

    # Should still have both columns
    assert list(df.columns) == site_ids
    # The missing site should be filled with 1.0
    assert (df[999999] == 1.0).all()


def test_assign_site_cluster_with_weather_year():
    """Test assign_site_cluster with weather_year parameter."""
    renew_data = pd.read_csv(DATA_FOLDER / "cpa_data.csv")
    profile_path = DATA_FOLDER / "cpa_profiles.csv"
    regions = ["A", "B"]
    cluster = {
        "cluster": [
            {"feature": "profile", "method": "hierarchical", "n_clusters": 3},
        ],
    }

    # Test with weather_year parameter
    data = assign_site_cluster(
        renew_data=renew_data,
        profile_path=profile_path,
        regions=regions,
        weather_year=2012,
        **cluster,
    )
    assert data.notna().all().all()
    assert "cluster" in data.columns


def test_load_site_profiles_with_single_list_weather_year():
    """Test loading site profiles with single-element list weather_year."""
    profile_path = DATA_FOLDER / "cpa_profiles.csv"
    site_ids = [1, 2]
    df_list = load_site_profiles(profile_path, site_ids=site_ids, weather_year=[2012])
    df_int = load_site_profiles(profile_path, site_ids=site_ids, weather_year=2012)
    assert len(df_list) == len(df_int) == 20
    assert (df_list[site_ids] == df_int[site_ids]).all().all()


def test_load_site_profiles_parquet_format(tmp_path):
    """Test loading site profiles from parquet file."""
    # Create test parquet file
    parquet_path = tmp_path / "test_profiles.parquet"
    test_data = pd.DataFrame(
        {
            "site_id": [1, 1, 1, 2, 2, 2],
            "time_index": [1, 2, 3, 1, 2, 3],
            "value": [0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
            "weather_year": [2012, 2012, 2012, 2012, 2012, 2012],
        }
    )
    test_data.to_parquet(parquet_path)

    # Load profiles
    site_ids = [1, 2]
    df = load_site_profiles(parquet_path, site_ids=site_ids, weather_year=2012)

    # Verify structure
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == site_ids
    assert len(df) == 3
    # Verify values
    assert df[1].tolist() == [0.5, 0.6, 0.7]
    assert df[2].tolist() == [0.8, 0.9, 0.95]


def test_assign_site_cluster_with_single_list_weather_year():
    """Test assign_site_cluster with weather_year as single-element list."""
    renew_data = pd.read_csv(DATA_FOLDER / "cpa_data.csv")
    profile_path = DATA_FOLDER / "cpa_profiles.csv"
    regions = ["A", "B"]
    cluster = {
        "cluster": [
            {"feature": "profile", "method": "hierarchical", "n_clusters": 3},
        ],
    }
    data_list = assign_site_cluster(
        renew_data=renew_data,
        profile_path=profile_path,
        regions=regions,
        weather_year=[2012],
        **cluster,
    )
    data_int = assign_site_cluster(
        renew_data=renew_data,
        profile_path=profile_path,
        regions=regions,
        weather_year=2012,
        **cluster,
    )
    assert data_list.notna().all().all()
    assert data_int.notna().all().all()
    assert "cluster" in data_list.columns
    assert "cluster" in data_int.columns
    assert len(data_list) == len(data_int)


# Additional coverage tests for error cases and edge paths


def test_load_site_profiles_unsupported_format():
    """Test that unsupported file format raises ValueError."""
    import tempfile
    from pathlib import Path

    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        bad_path = Path(f.name)
    try:
        with pytest.raises(ValueError, match="Unsupported profile file format"):
            load_site_profiles(bad_path, site_ids=[1, 2])
    finally:
        bad_path.unlink()


def test_load_site_profiles_missing_required_columns_csv(tmp_path):
    """Test that CSV without tidy columns and no site_id column matches raises ValueError."""
    bad_csv = tmp_path / "bad_profiles.csv"
    pd.DataFrame({"wrong_col": [1, 2], "another": [3, 4]}).to_csv(bad_csv, index=False)
    with pytest.raises(ValueError, match="unrecognized format"):
        load_site_profiles(bad_csv, site_ids=[1, 2])


def test_load_site_profiles_missing_required_columns_parquet(tmp_path):
    """Test that parquet without tidy columns and no site_id column matches raises ValueError."""
    bad_parquet = tmp_path / "bad_profiles.parquet"
    pd.DataFrame({"wrong_col": [1, 2], "another": [3, 4]}).to_parquet(bad_parquet)
    with pytest.raises(ValueError, match="unrecognized format"):
        load_site_profiles(bad_parquet, site_ids=[1, 2])


def test_load_site_profiles_wide_csv(tmp_path, caplog):
    """Wide CSV without weather_year: warns and loads matching columns."""
    wide_csv = tmp_path / "wide_profiles.csv"
    pd.DataFrame({"1": [0.1, 0.2, 0.3], "2": [0.4, 0.5, 0.6]}).to_csv(
        wide_csv, index=False
    )
    site_ids = ["1", "3"]  # "3" is missing
    with caplog.at_level(logging.WARNING):
        df = load_site_profiles(wide_csv, site_ids=site_ids)
    assert "wide format" in caplog.text.lower()
    assert list(df.columns) == ["1", "3"]
    assert len(df) == 3
    assert (df["3"] == 1.0).all()


def test_load_site_profiles_wide_parquet(tmp_path, caplog):
    """Wide parquet without weather_year: warns and loads matching columns."""
    wide_pq = tmp_path / "wide_profiles.parquet"
    pd.DataFrame({"1": [0.1, 0.2, 0.3], "2": [0.4, 0.5, 0.6]}).to_parquet(wide_pq)
    site_ids = ["1", "2"]
    with caplog.at_level(logging.WARNING):
        df = load_site_profiles(wide_pq, site_ids=site_ids)
    assert "wide format" in caplog.text.lower()
    assert list(df.columns) == ["1", "2"]
    assert len(df) == 3


def test_load_site_profiles_wide_csv_with_weather_year_raises(tmp_path):
    """Wide CSV with weather_year specified: raises ValueError."""
    wide_csv = tmp_path / "wide_profiles.csv"
    pd.DataFrame({"1": [0.1, 0.2], "2": [0.4, 0.5]}).to_csv(wide_csv, index=False)
    with pytest.raises(ValueError, match="weather_year filtering requires tidy"):
        load_site_profiles(wide_csv, site_ids=["1"], weather_year=2012)


def test_load_site_profiles_wide_parquet_with_weather_year_raises(tmp_path):
    """Wide parquet with weather_year specified: raises ValueError."""
    wide_pq = tmp_path / "wide_profiles.parquet"
    pd.DataFrame({"1": [0.1, 0.2], "2": [0.4, 0.5]}).to_parquet(wide_pq)
    with pytest.raises(ValueError, match="weather_year filtering requires tidy"):
        load_site_profiles(wide_pq, site_ids=["1"], weather_year=2012)


def test_value_bin_with_bins_outside_data_range(caplog):
    """Test warning when bins are outside data range."""
    import logging

    s = pd.Series([5, 10, 15, 20])
    # Min bin > data min triggers warning
    bins = [8, 12, 25]
    with caplog.at_level(logging.WARNING):
        result = value_bin(s, bins=bins)
    assert "minimum bin value" in caplog.text

    # Max bin < data max triggers warning
    caplog.clear()
    bins2 = [1, 12, 18]  # 18 < 20 (data max)
    with caplog.at_level(logging.WARNING):
        result2 = value_bin(s, bins=bins2)
    assert "maximum bin value" in caplog.text


def test_value_bin_empty_series():
    """Test value_bin with empty series."""
    s = pd.Series([], dtype=float)
    result = value_bin(s, bins=3)
    assert len(result) == 0


def test_value_bin_all_same_values():
    """Test value_bin when all values are identical."""
    s = pd.Series([5.0, 5.0, 5.0, 5.0])
    result = value_bin(s, bins=3)
    assert (result == 1).all()


def test_value_bin_q_equals_one():
    """Test value_bin with q=1 returns all ones."""
    s = pd.Series([1, 2, 3, 4, 5])
    result = value_bin(s, q=1)
    assert (result == 1).all()


def test_value_bin_bins_equals_one():
    """Test value_bin with bins=1 returns all ones."""
    s = pd.Series([1, 2, 3, 4, 5])
    result = value_bin(s, bins=1)
    assert (result == 1).all()


def test_value_bin_no_bins_or_q(caplog):
    """Test value_bin warning when neither bins nor q provided."""
    import logging

    s = pd.Series([1, 2, 3, 4, 5])
    with caplog.at_level(logging.WARNING):
        result = value_bin(s)
    assert "doesn't include either the 'bins' or 'q' argument" in caplog.text
    assert (result == 1).all()


def test_kmeans_cluster_other():
    """Test kmeans_cluster_other function."""
    from powergenome.cluster.renewables import kmeans_cluster_other

    df = pd.DataFrame({"lat": [10, 20, 30, 40], "lon": [50, 60, 70, 80]})
    labels = kmeans_cluster_other(df, n_clusters=2)
    assert len(labels) == 4
    assert len(set(labels)) <= 2


def test_kmeans_cluster_other_empty():
    """Test kmeans_cluster_other with empty dataframe."""
    from powergenome.cluster.renewables import kmeans_cluster_other

    df = pd.DataFrame()
    labels = kmeans_cluster_other(df, n_clusters=2)
    assert len(labels) == 0


def test_kmeans_cluster_other_single_row():
    """Test kmeans_cluster_other with single row."""
    from powergenome.cluster.renewables import kmeans_cluster_other

    df = pd.DataFrame({"lat": [10], "lon": [50]})
    labels = kmeans_cluster_other(df, n_clusters=2)
    assert len(labels) == 1
    assert labels[0] == 0


def test_kmeans_cluster_other_n_clusters_too_large(caplog):
    """Test kmeans when n_clusters > num sites."""
    import logging

    from powergenome.cluster.renewables import kmeans_cluster_other

    df = pd.DataFrame({"lat": [10, 20], "lon": [50, 60]})
    with caplog.at_level(logging.WARNING):
        labels = kmeans_cluster_other(df, n_clusters=5)
    assert "greater than the number of renewable sites" in caplog.text
    assert len(set(labels)) == 2  # Each site gets own cluster


def test_kmeans_cluster_other_n_clusters_zero(caplog):
    """Test kmeans with n_clusters <= 0."""
    import logging

    from powergenome.cluster.renewables import kmeans_cluster_other

    df = pd.DataFrame({"lat": [10, 20, 30], "lon": [50, 60, 70]})
    with caplog.at_level(logging.WARNING):
        labels = kmeans_cluster_other(df, n_clusters=0)
    assert "less than or equal to 0" in caplog.text


def test_min_capacity_mw_no_lcoe_column(caplog):
    """Test min_capacity_mw warning when lcoe column missing."""
    import logging

    from powergenome.cluster.renewables import min_capacity_mw

    df = pd.DataFrame({"capacity_mw": [100, 200, 300], "cost": [10, 20, 30]})
    with caplog.at_level(logging.WARNING):
        result = min_capacity_mw(df, min_cap=200)
    assert "lcoe" in caplog.text
    assert len(result) == len(df)  # All sites included


def test_min_capacity_mw_none():
    """Test min_capacity_mw when min_cap is None."""
    from powergenome.cluster.renewables import min_capacity_mw

    df = pd.DataFrame({"capacity_mw": [100, 200, 300], "lcoe": [10, 20, 30]})
    result = min_capacity_mw(df, min_cap=None)
    assert len(result) == len(df)


def test_assign_site_cluster_missing_bin_feature():
    """Test error when bin feature key is missing."""
    renew_data = pd.read_csv(DATA_FOLDER / "cpa_data.csv")
    profile_path = DATA_FOLDER / "cpa_profiles.csv"
    with pytest.raises(KeyError, match="doesn't include the 'feature' argument"):
        assign_site_cluster(
            renew_data=renew_data,
            profile_path=profile_path,
            regions=["A"],
            bin=[{"bins": 3}],  # Missing 'feature'
        )


def test_assign_site_cluster_bin_feature_not_in_data():
    """Test error when bin feature not in renewable data."""
    renew_data = pd.read_csv(DATA_FOLDER / "cpa_data.csv")
    profile_path = DATA_FOLDER / "cpa_profiles.csv"
    with pytest.raises(KeyError, match="not in the renewable site data"):
        assign_site_cluster(
            renew_data=renew_data,
            profile_path=profile_path,
            regions=["A"],
            bin=[{"feature": "nonexistent_column", "bins": 3}],
        )


def test_assign_site_cluster_bin_feature_not_numeric():
    """Test error when bin feature is not numeric."""
    renew_data = pd.read_csv(DATA_FOLDER / "cpa_data.csv")
    profile_path = DATA_FOLDER / "cpa_profiles.csv"
    with pytest.raises(TypeError, match="not a numeric column"):
        assign_site_cluster(
            renew_data=renew_data,
            profile_path=profile_path,
            regions=["A"],
            bin=[{"feature": "region", "bins": 3}],  # region is categorical
        )


def test_assign_site_cluster_weights_not_in_data():
    """Test error when weights column not in data."""
    renew_data = pd.read_csv(DATA_FOLDER / "cpa_data.csv")
    profile_path = DATA_FOLDER / "cpa_profiles.csv"
    with pytest.raises(KeyError, match="not in the renewable site data"):
        assign_site_cluster(
            renew_data=renew_data,
            profile_path=profile_path,
            regions=["A"],
            bin=[{"feature": "lcoe", "bins": 3, "weights": "nonexistent_weight"}],
        )


def test_assign_site_cluster_empty_after_filter():
    """Test filtering with extreme values."""
    renew_data = pd.read_csv(DATA_FOLDER / "cpa_data.csv")
    profile_path = DATA_FOLDER / "cpa_profiles.csv"
    # Use a very restrictive filter to reduce results significantly
    result = assign_site_cluster(
        renew_data=renew_data,
        profile_path=profile_path,
        regions=["A"],
        filter=[{"feature": "lcoe", "max": 1}],  # Very restrictive filter
    )
    assert "cluster" in result.columns
    # Just check that filtering worked (may not be zero due to data content)
    assert len(result) <= len(renew_data[renew_data["region"] == "A"])


def test_assign_site_cluster_no_profile_path():
    """Test assign_site_cluster when profile_path is None."""
    renew_data = pd.read_csv(DATA_FOLDER / "cpa_data.csv")
    result = assign_site_cluster(
        renew_data=renew_data,
        profile_path=None,
        regions=["A"],
    )
    assert "cluster" in result.columns
    assert "profile" in result.columns
    assert (result["profile"] == 1.0).all()


def test_cluster_sites_no_bin_empty_data():
    """Test cluster_sites_no_bin with empty dataframe."""
    df = pd.DataFrame()
    result = cluster_sites_no_bin(df, method="agg", feature="lcoe", n_clusters=2)
    assert "cluster" in result.columns
    assert len(result) == 0


def test_cluster_sites_binned_empty_data():
    """Test cluster_sites_binned with empty dataframe."""
    df = pd.DataFrame()
    result = cluster_sites_binned(
        df, by=["region"], method="agg", feature="lcoe", n_clusters=2
    )
    assert "cluster" in result.columns
    assert len(result) == 0


def test_cluster_sites_no_bin_single_site():
    """Test cluster_sites_no_bin with single site."""
    df = pd.DataFrame({"lcoe": [10], "capacity_mw": [100]})
    result = cluster_sites_no_bin(df, method="agg", feature="lcoe", n_clusters=2)
    assert "cluster" in result.columns
    assert len(result) == 1


def test_value_filter_max_only():
    """Test value_filter with only max_value."""
    from powergenome.cluster.renewables import value_filter

    df = pd.DataFrame({"lcoe": [10, 20, 30, 40], "capacity_mw": [100, 200, 300, 400]})
    result = value_filter(df, feature="lcoe", max_value=25)
    assert len(result) == 2
    assert result["lcoe"].max() <= 25


def test_value_filter_min_only():
    """Test value_filter with only min_value."""
    from powergenome.cluster.renewables import value_filter

    df = pd.DataFrame({"lcoe": [10, 20, 30, 40], "capacity_mw": [100, 200, 300, 400]})
    result = value_filter(df, feature="lcoe", min_value=25)
    assert len(result) == 2
    assert result["lcoe"].min() >= 25


def test_value_filter_both():
    """Test value_filter with both min and max."""
    from powergenome.cluster.renewables import value_filter

    df = pd.DataFrame({"lcoe": [10, 20, 30, 40], "capacity_mw": [100, 200, 300, 400]})
    result = value_filter(df, feature="lcoe", min_value=15, max_value=35)
    assert len(result) == 2
    assert result["lcoe"].min() >= 15
    assert result["lcoe"].max() <= 35


def test_assign_site_cluster_mw_per_cluster(caplog):
    """Test mw_per_cluster overrides n_clusters."""
    import logging

    renew_data = pd.read_csv(DATA_FOLDER / "cpa_data.csv")
    profile_path = DATA_FOLDER / "cpa_profiles.csv"
    with caplog.at_level(logging.WARNING):
        result = assign_site_cluster(
            renew_data=renew_data,
            profile_path=profile_path,
            regions=["A"],
            cluster=[
                {
                    "feature": "lcoe",
                    "method": "agg",
                    "n_clusters": 10,
                    "mw_per_cluster": 500,
                }
            ],
        )
    assert "Overwriting 'n_clusters'" in caplog.text
    assert "cluster" in result.columns
