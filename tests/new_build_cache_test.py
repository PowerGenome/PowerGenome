"""Tests for the renewable cluster cache key in powergenome.new_build."""

import pytest

from powergenome import new_build

BASE_KEY_ARGS = dict(
    region="R1",
    technology="landbasedwind_class3",
    detail_suffix="max_clusters_2",
    utc_offset=-5,
    weather_year=2012,
    data_file_hash="abc123",
    precluster=False,
)


def test_cache_key_is_deterministic():
    assert new_build._renewable_cache_key(
        **BASE_KEY_ARGS
    ) == new_build._renewable_cache_key(**BASE_KEY_ARGS)


@pytest.mark.parametrize(
    "field, value",
    [
        ("region", "R2"),
        ("technology", "utilitypv_class1"),
        ("detail_suffix", "max_clusters_3"),
        ("utc_offset", 0),
        ("weather_year", 2013),
        ("weather_year", "all"),
        ("data_file_hash", "def456"),
        ("precluster", True),
    ],
)
def test_cache_key_changes_with_each_input(field, value):
    base_name, base_hash = new_build._renewable_cache_key(**BASE_KEY_ARGS)
    name, unique_hash = new_build._renewable_cache_key(
        **{**BASE_KEY_ARGS, field: value}
    )
    assert name != base_name
    assert unique_hash != base_hash


def test_cache_key_changes_with_profile_transform_version(monkeypatch):
    base_name, base_hash = new_build._renewable_cache_key(**BASE_KEY_ARGS)
    monkeypatch.setattr(
        new_build,
        "PROFILE_TRANSFORM_VERSION",
        new_build.PROFILE_TRANSFORM_VERSION + 1,
    )
    name, unique_hash = new_build._renewable_cache_key(**BASE_KEY_ARGS)
    assert name != base_name
    assert unique_hash != base_hash


def test_cache_name_is_readable():
    name, unique_hash = new_build._renewable_cache_key(**BASE_KEY_ARGS)
    assert name.startswith("R1_landbasedwind_class3_max_clusters_2_UTC-5")
    assert "weather_year2012" in name
    assert "preclusterFalse" in name
    assert f"_v{new_build.PROFILE_TRANSFORM_VERSION}_" in name
    assert name.endswith("file_abc123")
    assert unique_hash == new_build.hash_string_sha256(name)
