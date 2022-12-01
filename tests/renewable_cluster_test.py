"Test functions for clustering renewable sites"

import pandas as pd

from powergenome.cluster.renewables import (
    value_bin,
    agg_cluster_other,
    agg_cluster_profile,
    agglomerative_cluster_binned,
    agglomerative_cluster_no_bin,
)

import hypothesis
from hypothesis import given, strategies as st
from hypothesis.extra.pandas import data_frames, column, range_indexes, series
from hypothesis.extra.numpy import arrays


@given(
    s=st.builds(pd.Series),
    bins=st.one_of(st.none(), st.integers(), st.lists(st.floats())),
    q=st.one_of(st.none(), st.integers(), st.lists(st.floats())),
)
def test_fuzz_value_bin(s, bins, q):
    value_bin(s=s, bins=bins, q=q)


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
