"Test functions for clustering renewable sites"

from pathlib import Path

import hypothesis
import pandas as pd
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from hypothesis.extra.pandas import column, data_frames, range_indexes, series

from powergenome.cluster.renewables import (
    agg_cluster_other,
    agg_cluster_profile,
    agglomerative_cluster_binned,
    agglomerative_cluster_no_bin,
    assign_site_cluster,
    num_bins_from_capacity,
    value_bin,
)

CWD = Path.cwd()
DATA_FOLDER = CWD / "tests" / "data" / "cpa_cluster_data"

cluster_data = data_frames(
    columns=[
        column(
            name="profile",
            elements=arrays(
                float, (10,), elements=st.floats(min_value=0, max_value=100)
            ),
        ),
        column(
            name="lcoe",
            elements=st.floats(min_value=0, max_value=100, allow_infinity=False),
        ),
        column(name="state", elements=st.sampled_from(["a", "b"])),
    ]
)


@given(
    bins=st.one_of(
        st.integers(min_value=1, max_value=20),
        st.lists(
            st.floats(min_value=0.001, max_value=100),
            min_size=2,
            max_size=5,
            unique=True,
        ),
    ),
    q=st.one_of(
        st.none(),
        st.integers(min_value=1, max_value=20),
        st.lists(
            st.floats(min_value=0, max_value=1), min_size=2, max_size=5, unique=True
        ),
    ),
    data=st.data(),
)
def test_fuzz_value_bins(bins, q, data):
    strategy = series(
        elements=st.floats(min_value=0, max_value=100),
        index=range_indexes(min_size=10, max_size=10),
    )
    s = data.draw(strategy)
    # pandas binning breaks with very small values. Allow 0 but nothing smaller than 0.01
    s.loc[(s > 0) & (s < 0.01)] = 0.01

    # Run separately with and without weights. Tried st.one_of but it causes an error:
    # elif (weights == 0).all():
    # AttributeError: 'bool' object has no attribute 'all'
    value_bin(s=s, bins=bins, q=q)
    w = data.draw(strategy)
    value_bin(s=s, bins=bins, q=q, weights=w)


@given(
    s=series(
        elements=arrays(float, (10,), elements=st.floats(min_value=0, max_value=100))
    ),
    n_clusters=st.integers(),
)
def test_fuzz_agg_cluster_profile(s, n_clusters):
    agg_cluster_profile(s=s, n_clusters=n_clusters)


@given(s=st.builds(pd.Series), n_clusters=st.integers())
def test_fuzz_agg_cluster_other(s, n_clusters):
    agg_cluster_other(s=s, n_clusters=n_clusters)


@given(
    data=cluster_data,
    feature=st.sampled_from(["profile", "lcoe"]),
    n_clusters=st.integers(),
)
def test_fuzz_agglomerative_cluster_no_bin(data, feature, n_clusters):
    agglomerative_cluster_no_bin(data=data, feature=feature, n_clusters=n_clusters)


@given(
    data=cluster_data,
    by=st.just(["state"]),
    feature=st.sampled_from(["profile", "lcoe"]),
    n_clusters=st.integers(),
)
def test_fuzz_agglomerative_cluster_binned(data, feature, by, n_clusters):
    agglomerative_cluster_binned(
        data=data, by=by, feature=feature, n_clusters=n_clusters
    )


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


# Tests that the function correctly calculates the number of bins based on the "mw_per_bin" key in the input dictionary. tags: [happy path]
def test_num_bins_from_capacity_with_mw_per_bin():
    # Happy path test for calculating number of bins based on "mw_per_bin"
    data = pd.DataFrame({"mw": [10, 20, 30]})
    b = {"mw_per_bin": 14}
    expected_output = {"bins": 4}
    assert num_bins_from_capacity(data, b) == expected_output


def test_num_bins_from_capacity_with_mw_per_bin_one_bin():
    # Happy path test for calculating number of bins based on "mw_per_bin"
    data = pd.DataFrame({"mw": [1, 2, 3]})
    b = {"mw_per_bin": 14}
    expected_output = {"bins": 1}
    assert num_bins_from_capacity(data, b) == expected_output


# Tests that the function correctly calculates the number of quantiles based on the "mw_per_q" key in the input dictionary. tags: [happy path]
def test_num_bins_from_capacity_with_mw_per_q():
    # Happy path test for calculating number of quantiles based on "mw_per_q" key
    data = pd.DataFrame({"mw": [10, 20, 30]})
    b = {"mw_per_q": 15}
    expected_output = {"q": 4}
    assert num_bins_from_capacity(data, b) == expected_output


# Tests that the function returns the input dictionary unaltered if neither "mw_per_bin" nor "mw_per_q" key is present. tags: [happy path]
def test_num_bins_from_capacity_with_no_mw_key():
    # Happy path test for returning input dictionary unaltered if no "mw_per_bin" or "mw_per_q" key is present
    data = pd.DataFrame({"mw": [10, 20, 30]})
    b = {"other_key": "value"}
    expected_output = {"other_key": "value"}
    assert num_bins_from_capacity(data, b) == expected_output


# Tests that the function handles input dictionary containing non-integer values for "mw_per_bin" or "mw_per_q". tags: [edge case]
def test_num_bins_from_capacity_with_non_integer_values():
    data = pd.DataFrame({"mw": [100, 200, 300]})
    b = {"mw_per_bin": 0.5}
    result = num_bins_from_capacity(data, b)
    assert result == {"bins": 1200}


# Tests that the function logs a warning message if the "bins" key is already present in the input dictionary and is being overwritten. tags: [behavior]
def test_num_bins_from_capacity_with_overwriting_bins(caplog):
    data = pd.DataFrame({"mw": [100, 200, 300]})
    b = {"mw_per_bin": 100, "bins": 5}
    result = num_bins_from_capacity(data, b)
    assert result == {"bins": 6}
    assert "Overwriting 'bins' based on mw_per_bin" in caplog.text
