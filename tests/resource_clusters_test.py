"""Tests for resource_clusters.py — ResourceGroup._read_profiles and test_profiles."""

import logging

import numpy as np
import pandas as pd
import pytest

from powergenome.resource_clusters import ResourceGroup


@pytest.fixture
def tidy_profiles_df():
    """Tidy-format profiles: 3 sites × 24 hours."""
    np.random.seed(42)
    n_hours = 24
    site_ids = [0, 1, 2]
    records = []
    for sid in site_ids:
        for t in range(1, n_hours + 1):
            records.append(
                {"site_id": sid, "time_index": t, "value": np.random.random()}
            )
    return pd.DataFrame(records)


@pytest.fixture
def tidy_profiles_with_weather_df():
    """Tidy-format profiles with weather_year: 3 sites × 24 hours × 2 years."""
    np.random.seed(42)
    n_hours = 24
    site_ids = [0, 1, 2]
    weather_years = [2012, 2013]
    records = []
    for sid in site_ids:
        for wy in weather_years:
            for t in range(1, n_hours + 1):
                records.append(
                    {
                        "site_id": sid,
                        "time_index": t,
                        "value": np.random.random(),
                        "weather_year": wy,
                    }
                )
    return pd.DataFrame(records)


@pytest.fixture
def tidy_profiles_valid_df():
    """Tidy-format profiles with valid 8760 hours."""
    np.random.seed(42)
    n_hours = 8760
    site_ids = [0, 1]
    records = []
    for sid in site_ids:
        for t in range(1, n_hours + 1):
            records.append(
                {"site_id": sid, "time_index": t, "value": np.random.random()}
            )
    return pd.DataFrame(records)


@pytest.fixture
def wide_profiles_valid_df():
    """Wide-format profiles with valid 8760 hours."""
    np.random.seed(42)
    n_hours = 8760
    return pd.DataFrame(
        {"0": np.random.random(n_hours), "1": np.random.random(n_hours)}
    )


@pytest.fixture
def wide_profiles_df():
    """Wide-format profiles: 3 sites as columns × 24 rows."""
    np.random.seed(42)
    n_hours = 24
    return pd.DataFrame(
        {
            "0": np.random.random(n_hours),
            "1": np.random.random(n_hours),
            "2": np.random.random(n_hours),
        }
    )


@pytest.fixture
def unrecognized_profiles_df():
    """Neither tidy nor wide — completely unrelated columns."""
    np.random.seed(42)
    return pd.DataFrame({"foo": range(10), "bar": np.random.random(10)})


@pytest.fixture
def metadata_df():
    """Minimal metadata for 3 sites."""
    return pd.DataFrame(
        {"id": [0, 1, 2], "region": ["A", "A", "B"], "capacity_mw": [100, 200, 150]}
    )


@pytest.fixture
def metadata_df_2():
    """Minimal metadata for 2 sites (for valid-length tests)."""
    return pd.DataFrame({"id": [0, 1], "region": ["A", "A"], "capacity_mw": [100, 200]})


@pytest.fixture
def resource_group_tidy(metadata_df, tidy_profiles_df):
    """ResourceGroup with tidy-format profiles in memory."""
    group = {"technology": "utilitypv"}
    return ResourceGroup(group, metadata=metadata_df, profiles=tidy_profiles_df)


@pytest.fixture
def resource_group_tidy_weather(metadata_df, tidy_profiles_with_weather_df):
    """ResourceGroup with tidy + weather_year profiles in memory."""
    group = {"technology": "utilitypv"}
    return ResourceGroup(
        group, metadata=metadata_df, profiles=tidy_profiles_with_weather_df
    )


@pytest.fixture
def resource_group_wide(metadata_df, wide_profiles_df):
    """ResourceGroup with wide-format profiles in memory."""
    group = {"technology": "utilitypv"}
    return ResourceGroup(group, metadata=metadata_df, profiles=wide_profiles_df)


@pytest.fixture
def resource_group_unrecognized(metadata_df, unrecognized_profiles_df):
    """ResourceGroup with unrecognized-format profiles in memory."""
    group = {"technology": "utilitypv"}
    return ResourceGroup(group, metadata=metadata_df, profiles=unrecognized_profiles_df)


# ── _read_profiles tests ──────────────────────────────────────────────


class TestReadProfiles:
    """Test ResourceGroup._read_profiles in all format/weather_year combinations."""

    def test_tidy_no_weather_year(self, resource_group_tidy):
        """Tidy format without weather_year: returns wide DataFrame of requested sites."""
        result = resource_group_tidy._read_profiles(site_ids=[0, 1])
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == [0, 1]
        assert len(result) == 24  # n_hours

    def test_tidy_with_weather_year_single(self, resource_group_tidy_weather):
        """Tidy format with weather_year filter: returns only the specified year."""
        result = resource_group_tidy_weather._read_profiles(
            site_ids=[0, 1], weather_year=2012
        )
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == [0, 1]
        assert len(result) == 24  # only 2012

    def test_tidy_with_weather_year_list(self, resource_group_tidy_weather):
        """Tidy format with weather_year list: concatenates years."""
        result = resource_group_tidy_weather._read_profiles(
            site_ids=[0, 1], weather_year=[2012, 2013]
        )
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == [0, 1]
        assert len(result) == 48  # 2 years × 24h each

    def test_tidy_with_weather_year_missing(self, resource_group_tidy_weather):
        """Tidy format: weather_year not in data raises ValueError."""
        with pytest.raises(ValueError, match="None of the requested weather years"):
            resource_group_tidy_weather._read_profiles(site_ids=[0], weather_year=9999)

    def test_tidy_with_weather_year_all_missing(self, resource_group_tidy_weather):
        """Tidy format: requesting nonexistent weather years raises ValueError."""
        with pytest.raises(ValueError, match="None of the requested weather years"):
            resource_group_tidy_weather._read_profiles(
                site_ids=[0], weather_year=[9999, 8888]
            )

    def test_tidy_missing_site_ids_filled(self, resource_group_tidy):
        """Tidy format: missing site IDs are filled with 1.0."""
        result = resource_group_tidy._read_profiles(site_ids=[0, 99])
        assert list(result.columns) == [0, 99]
        assert (result[99] == 1.0).all()

    def test_wide_no_weather_year(self, resource_group_wide, caplog):
        """Wide format without weather_year: warns and loads requested columns."""
        with caplog.at_level(logging.WARNING):
            result = resource_group_wide._read_profiles(site_ids=[0, 1])
        assert "wide format" in caplog.text.lower()
        assert list(result.columns) == [0, 1]
        assert len(result) == 24

    def test_wide_missing_site_ids_filled(self, resource_group_wide, caplog):
        """Wide format: missing site IDs filled with 1.0."""
        with caplog.at_level(logging.WARNING):
            result = resource_group_wide._read_profiles(site_ids=[0, 99])
        assert list(result.columns) == [0, 99]
        assert (result[99] == 1.0).all()

    def test_wide_with_weather_year_raises(self, resource_group_wide):
        """Wide format with weather_year specified: raises ValueError."""
        with pytest.raises(ValueError, match="weather_year filtering requires tidy"):
            resource_group_wide._read_profiles(site_ids=[0], weather_year=2012)

    def test_unrecognized_format_raises(self, resource_group_unrecognized):
        """Neither tidy nor wide (no matching columns): raises ValueError."""
        with pytest.raises(
            ValueError, match="Profiles file has an unrecognized format"
        ):
            resource_group_unrecognized._read_profiles(site_ids=[0, 1])

    def test_unrecognized_format_with_weather_year_raises(
        self, resource_group_unrecognized
    ):
        """Unrecognized format with weather_year: raises ValueError."""
        with pytest.raises(
            ValueError, match="Profiles file has an unrecognized format"
        ):
            resource_group_unrecognized._read_profiles(site_ids=[0], weather_year=2012)

    def test_wide_generates_deprecation_warning(self, resource_group_wide, caplog):
        """Wide format should log a deprecation warning via logger."""
        with caplog.at_level(logging.WARNING):
            resource_group_wide._read_profiles(site_ids=[0])
        assert "wide format" in caplog.text.lower()

    def test_tidy_multiyear_no_weather_year_concatenates(
        self, resource_group_tidy_weather, caplog
    ):
        """Tidy multiyear without explicit weather_year: concatenates all years."""
        with caplog.at_level(logging.DEBUG):
            result = resource_group_tidy_weather._read_profiles(site_ids=[0])
        assert "concatenating all available years" in caplog.text
        assert len(result) == 48  # 2 years × 24h


# ── test_profiles tests ───────────────────────────────────────────────


class TestTestProfiles:
    """Test ResourceGroup.test_profiles validation."""

    def test_tidy_valid_passes(self, metadata_df_2, tidy_profiles_valid_df):
        """Tidy format with valid length (8760h): no error."""
        group = {"technology": "utilitypv"}
        rg = ResourceGroup(
            group, metadata=metadata_df_2, profiles=tidy_profiles_valid_df
        )
        rg.test_profiles()  # should not raise

    def test_wide_valid_passes(self, metadata_df_2, wide_profiles_valid_df):
        """Wide format with site IDs matching metadata IDs and 8760h: no error."""
        group = {"technology": "utilitypv"}
        rg = ResourceGroup(
            group, metadata=metadata_df_2, profiles=wide_profiles_valid_df
        )
        rg.test_profiles()  # should not raise

    def test_unrecognized_format_raises(self, resource_group_unrecognized):
        """Neither tidy nor wide: raises ValueError."""
        with pytest.raises(
            ValueError, match="Profiles file has an unrecognized format"
        ):
            resource_group_unrecognized.test_profiles()

    def test_no_profiles_returns_none(self, metadata_df):
        """No profiles: test_profiles returns None."""
        group = {"technology": "utilitypv"}
        rg = ResourceGroup(group, metadata=metadata_df)
        assert rg.test_profiles() is None

    def test_tidy_bad_length_raises(self, metadata_df):
        """Tidy format with invalid hour count: raises ValueError."""
        bad_profiles = pd.DataFrame(
            {
                "site_id": [0] * 100,
                "time_index": range(1, 101),
                "value": np.random.random(100),
            }
        )
        group = {"technology": "utilitypv"}
        rg = ResourceGroup(group, metadata=metadata_df, profiles=bad_profiles)
        with pytest.raises(ValueError, match="not a multiple of 8760 or 8784"):
            rg.test_profiles()

    def test_wide_bad_length_raises(self, metadata_df):
        """Wide format with invalid hour count: raises ValueError."""
        bad_wide = pd.DataFrame(
            {"0": np.random.random(100), "1": np.random.random(100)}
        )
        group = {"technology": "utilitypv"}
        rg = ResourceGroup(group, metadata=metadata_df, profiles=bad_wide)
        with pytest.raises(ValueError, match="not a multiple of 8760 or 8784"):
            rg.test_profiles()
