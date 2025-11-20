"""
Test functions for the new DataManager-based distributed generation implementation.

This test suite covers:
- get_distributed_gen_capacity with interpolation/extrapolation
- get_distributed_gen_profiles with weather year filtering
- get_distributed_gen_hourly_generation
- Region aggregation logic
- Error handling
"""

import logging
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from powergenome.database import DataManager, initialize_data_manager
from powergenome.distributed_gen import (
    get_distributed_gen_capacity,
    get_distributed_gen_hourly_generation,
    get_distributed_gen_profiles,
)


@pytest.fixture
def sample_capacity_data():
    """Sample distributed generation capacity data."""
    return pd.DataFrame(
        {
            "region": ["A", "A", "A", "B", "B", "B", "C", "C"],
            "capacity_mw": [100.0, 150.0, 200.0, 50.0, 75.0, 100.0, 200.0, 300.0],
            "year": [2020, 2025, 2030, 2020, 2025, 2030, 2020, 2030],
        }
    )


@pytest.fixture
def sample_profile_data():
    """Sample distributed generation profile data (normalized 0-1)."""
    data = []
    for region in ["A", "B", "C"]:
        for weather_year in [2012, 2013]:
            for time_index in range(1, 9):  # 8 hours for simplicity
                # Create different patterns for each region
                if region == "A":
                    value = 0.5 + 0.3 * np.sin(time_index / 8 * np.pi * 2)
                elif region == "B":
                    value = 0.4 + 0.2 * np.sin(time_index / 8 * np.pi * 2)
                else:  # C
                    value = 0.6 + 0.25 * np.sin(time_index / 8 * np.pi * 2)

                data.append(
                    {
                        "region": region,
                        "time_index": time_index,
                        "value": max(0, min(1, value)),  # Clamp to [0, 1]
                        "weather_year": weather_year,
                    }
                )
    return pd.DataFrame(data)


@pytest.fixture
def temp_csv_data_folder(sample_capacity_data, sample_profile_data):
    """Create temporary folder with distributed gen CSV files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create CSV files
        capacity_file = temp_path / "distributed_capacity.csv"
        sample_capacity_data.to_csv(capacity_file, index=False)

        profile_file = temp_path / "distributed_profiles.csv"
        sample_profile_data.to_csv(profile_file, index=False)

        yield temp_path


@pytest.fixture
def initialized_data_manager(temp_csv_data_folder):
    """Initialize DataManager with distributed gen tables."""
    # Clean up any existing DataManager first
    try:
        DataManager().close()
    except Exception:
        # Ignore: DataManager may not have been initialized yet
        pass

    settings = {
        "distributed_capacity_table": "distributed_capacity.csv",
        "distributed_profiles_table": "distributed_profiles.csv",
    }
    initialize_data_manager(settings, temp_csv_data_folder)
    yield
    # Cleanup
    try:
        DataManager().close()
    except Exception:
        # Ignore: DataManager may already be closed or not initialized
        pass


class TestGetDistributedGenCapacity:
    """Tests for get_distributed_gen_capacity function."""

    def test_exact_year_match(self, initialized_data_manager):
        """Test loading capacity when exact year exists in data."""
        capacity = get_distributed_gen_capacity(
            year=2025, regions=["A", "B"], region_aggregations=None
        )

        assert len(capacity) == 2
        assert set(capacity["region"]) == {"A", "B"}
        assert capacity.loc[capacity["region"] == "A", "capacity_mw"].values[0] == 150.0
        assert capacity.loc[capacity["region"] == "B", "capacity_mw"].values[0] == 75.0

    def test_interpolation_between_years(self, initialized_data_manager, caplog):
        """Test linear interpolation when year falls between data points."""
        with caplog.at_level(logging.INFO):
            capacity = get_distributed_gen_capacity(
                year=2027, regions=["A"], region_aggregations=None
            )

        # Year 2027 is between 2025 (150 MW) and 2030 (200 MW)
        # Linear interpolation: 150 + (200-150) * (2027-2025)/(2030-2025) = 150 + 50*0.4 = 170
        assert len(capacity) == 1
        expected_value = 150.0 + (200.0 - 150.0) * (2027 - 2025) / (2030 - 2025)
        actual_value = capacity.loc[capacity["region"] == "A", "capacity_mw"].values[0]
        assert np.isclose(
            actual_value, expected_value, rtol=1e-5
        ), f"Expected {expected_value}, got {actual_value}"

        # Check that interpolation was logged
        assert any("Interpolated" in record.message for record in caplog.records)

    def test_backward_extrapolation(self, initialized_data_manager, caplog):
        """Test extrapolation when year is before all available data."""
        with caplog.at_level(logging.INFO):
            capacity = get_distributed_gen_capacity(
                year=2018, regions=["A"], region_aggregations=None
            )

        # Year 2018 is before 2020 (earliest year), should use 2020 value (100 MW)
        assert len(capacity) == 1
        assert capacity.loc[capacity["region"] == "A", "capacity_mw"].values[0] == 100.0

        # Check that backward extrapolation was logged
        assert any(
            "Backward extrapolated" in record.message for record in caplog.records
        )

    def test_forward_extrapolation(self, initialized_data_manager, caplog):
        """Test extrapolation when year is after all available data."""
        with caplog.at_level(logging.INFO):
            capacity = get_distributed_gen_capacity(
                year=2035, regions=["A"], region_aggregations=None
            )

        # Year 2035 is after 2030 (latest year), should use 2030 value (200 MW)
        assert len(capacity) == 1
        assert capacity.loc[capacity["region"] == "A", "capacity_mw"].values[0] == 200.0

        # Check that forward extrapolation was logged
        assert any(
            "Forward extrapolated" in record.message for record in caplog.records
        )

    def test_region_aggregation_sum(self, initialized_data_manager):
        """Test that capacities are summed when aggregating regions."""
        capacity = get_distributed_gen_capacity(
            year=2025,
            regions=["AB"],
            region_aggregations={"AB": ["A", "B"]},
        )

        # Should sum A (150) + B (75) = 225
        assert len(capacity) == 1
        assert (
            capacity.loc[capacity["region"] == "AB", "capacity_mw"].values[0] == 225.0
        )

    def test_mixed_aggregation_and_base_regions(self, initialized_data_manager):
        """Test with both aggregated and non-aggregated regions."""
        capacity = get_distributed_gen_capacity(
            year=2030,
            regions=["AB", "C"],
            region_aggregations={"AB": ["A", "B"]},
        )

        # Should have AB (200 + 100 = 300) and C (300)
        assert len(capacity) == 2
        assert set(capacity["region"]) == {"AB", "C"}
        assert (
            capacity.loc[capacity["region"] == "AB", "capacity_mw"].values[0] == 300.0
        )
        assert capacity.loc[capacity["region"] == "C", "capacity_mw"].values[0] == 300.0

    def test_missing_region_in_data(self, initialized_data_manager, caplog):
        """Test handling of regions that don't exist in the data."""
        with caplog.at_level(logging.INFO):
            capacity = get_distributed_gen_capacity(
                year=2025, regions=["A", "NonExistent"], region_aggregations=None
            )

        # Should only return region A (NonExistent is silently skipped)
        assert len(capacity) == 1
        assert capacity["region"].values[0] == "A"

    def test_year_none_raises_error(self, initialized_data_manager):
        """Test that year=None raises a ValueError."""
        with pytest.raises(ValueError, match="Model year must be provided"):
            get_distributed_gen_capacity(year=None, regions=["A"])

    def test_empty_result_when_no_data(self, temp_csv_data_folder):
        """Test that empty DataFrame is returned when no data is available."""
        # Initialize with empty data
        empty_capacity = pd.DataFrame(columns=["region", "capacity_mw", "year"])
        empty_capacity.to_csv(temp_csv_data_folder / "empty_capacity.csv", index=False)

        settings = {"distributed_capacity_table": "empty_capacity.csv"}
        initialize_data_manager(settings, temp_csv_data_folder)

        capacity = get_distributed_gen_capacity(
            year=2025, regions=["A"], region_aggregations=None
        )

        assert capacity.empty
        assert list(capacity.columns) == ["region", "capacity_mw"]

        DataManager().close()

    def test_interpolation_summary_logging(self, initialized_data_manager, caplog):
        """Test that interpolation results are summarized in a single log message."""
        with caplog.at_level(logging.INFO):
            capacity = get_distributed_gen_capacity(
                year=2027, regions=["A", "B", "C"], region_aggregations=None
            )

        # Should have exactly one summary log message (not one per region)
        info_messages = [
            record.message
            for record in caplog.records
            if "Distributed generation capacity for year 2027" in record.message
        ]

        assert len(info_messages) == 1
        summary_msg = info_messages[0]

        # Should mention all regions
        assert "A" in summary_msg
        assert "B" in summary_msg
        assert "C" in summary_msg

        # Should indicate interpolation
        assert "Interpolated" in summary_msg or "interpolated" in summary_msg


class TestGetDistributedGenProfiles:
    """Tests for get_distributed_gen_profiles function."""

    def test_load_profiles_single_weather_year(self, initialized_data_manager):
        """Test loading profiles for a single weather year."""
        profiles = get_distributed_gen_profiles(
            weather_year=2012,
            regions=["A", "B"],
            region_aggregations=None,
            tz_offset=None,
            year=2025,
        )

        assert not profiles.empty
        assert set(profiles.columns) == {"A", "B"}
        assert len(profiles) == 8  # 8 time steps in test data
        # Values should be between 0 and 1 (normalized)
        assert profiles.min().min() >= 0
        assert profiles.max().max() <= 1

    def test_load_profiles_multiple_weather_years(self, initialized_data_manager):
        """Test loading profiles for multiple weather years."""
        profiles = get_distributed_gen_profiles(
            weather_year=[2012, 2013],
            regions=["A"],
            region_aggregations=None,
            tz_offset=None,
            year=2025,
        )

        # Should have 16 time steps (8 hours * 2 years)
        assert len(profiles) == 16
        assert "A" in profiles.columns

    def test_load_profiles_all_weather_years(self, initialized_data_manager, caplog):
        """Test loading profiles when weather_year is None (all years)."""
        with caplog.at_level(logging.INFO):
            profiles = get_distributed_gen_profiles(
                weather_year=None,
                regions=["A"],
                region_aggregations=None,
                tz_offset=None,
                year=2025,
            )

        # Should have all available weather years
        assert len(profiles) == 16  # 8 hours * 2 weather years

        # Check log message
        assert any(
            "ALL available weather years" in record.message for record in caplog.records
        )

    def test_profile_timezone_offset(self, initialized_data_manager):
        """Test that timezone offset shifts profiles correctly."""
        profiles_no_offset = get_distributed_gen_profiles(
            weather_year=2012,
            regions=["A"],
            region_aggregations=None,
            tz_offset=None,
            year=2025,
        )

        profiles_with_offset = get_distributed_gen_profiles(
            weather_year=2012,
            regions=["A"],
            region_aggregations=None,
            tz_offset=-2,  # Shift back 2 hours
            year=2025,
        )

        # After shifting back 2 hours, hour 1 should have the value that was at hour 3
        # (values wrap around, so hour 3 -> hour 1 when offset is -2)
        assert np.isclose(
            profiles_with_offset.loc[1, "A"], profiles_no_offset.loc[3, "A"], rtol=1e-5
        )

    def test_profile_region_aggregation_weighted(self, initialized_data_manager):
        """Test that profiles are aggregated using capacity-weighted average."""
        # Get individual profiles
        profiles_individual = get_distributed_gen_profiles(
            weather_year=2012,
            regions=["A", "B"],
            region_aggregations=None,
            tz_offset=None,
            year=2025,
        )

        # Get aggregated profile
        profiles_agg = get_distributed_gen_profiles(
            weather_year=2012,
            regions=["AB"],
            region_aggregations={"AB": ["A", "B"]},
            tz_offset=None,
            year=2025,  # Using year 2025: A=150, B=75
        )

        # At year 2025, capacity A=150, B=75, total=225
        # Weighted average should be (A*150 + B*75) / 225
        total_capacity = 225.0
        expected_profile = (
            profiles_individual["A"] * 150.0 + profiles_individual["B"] * 75.0
        ) / total_capacity

        assert np.allclose(
            profiles_agg["AB"].values, expected_profile.values, rtol=1e-5
        )

    def test_profile_aggregation_requires_year(self, initialized_data_manager):
        """Test that region aggregation requires a model year."""
        with pytest.raises(
            ValueError, match="Cannot aggregate distributed generation profiles"
        ):
            get_distributed_gen_profiles(
                weather_year=2012,
                regions=["AB"],
                region_aggregations={"AB": ["A", "B"]},
                tz_offset=None,
                year=None,  # This should cause an error
            )

    def test_empty_profiles_when_no_data(self, temp_csv_data_folder):
        """Test that empty DataFrame is returned when no profile data exists."""
        empty_profiles = pd.DataFrame(
            columns=["region", "time_index", "value", "weather_year"]
        )
        empty_profiles.to_csv(temp_csv_data_folder / "empty_profiles.csv", index=False)

        settings = {"distributed_profiles_table": "empty_profiles.csv"}
        initialize_data_manager(settings, temp_csv_data_folder)

        profiles = get_distributed_gen_profiles(
            weather_year=2012, regions=["A"], region_aggregations=None, year=2025
        )

        assert profiles.empty

        DataManager().close()

    def test_profile_index_starts_at_one(self, initialized_data_manager):
        """Test that the time_index starts at 1, not 0."""
        profiles = get_distributed_gen_profiles(
            weather_year=2012,
            regions=["A"],
            region_aggregations=None,
            tz_offset=None,
            year=2025,
        )

        assert profiles.index.min() == 1
        assert profiles.index.name == "time_index"


class TestGetDistributedGenHourlyGeneration:
    """Tests for get_distributed_gen_hourly_generation function."""

    def test_hourly_generation_calculation(self, initialized_data_manager):
        """Test that hourly generation = capacity * profile."""
        hourly_gen = get_distributed_gen_hourly_generation(
            year=2025,
            weather_year=2012,
            regions=["A"],
            region_aggregations=None,
            tz_offset=None,
        )

        # Get capacity and profile separately to verify
        capacity = get_distributed_gen_capacity(
            year=2025, regions=["A"], region_aggregations=None
        )
        profiles = get_distributed_gen_profiles(
            weather_year=2012,
            regions=["A"],
            year=2025,
            region_aggregations=None,
            tz_offset=None,
        )

        capacity_mw = capacity.loc[capacity["region"] == "A", "capacity_mw"].values[0]
        expected_gen = profiles["A"] * capacity_mw

        assert np.allclose(hourly_gen["A"].values, expected_gen.values, rtol=1e-5)

    def test_hourly_generation_multiple_regions(self, initialized_data_manager):
        """Test hourly generation for multiple regions."""
        hourly_gen = get_distributed_gen_hourly_generation(
            year=2030,
            weather_year=2012,
            regions=["A", "B", "C"],
            region_aggregations=None,
            tz_offset=None,
        )

        assert set(hourly_gen.columns) == {"A", "B", "C"}
        assert len(hourly_gen) == 8  # 8 time steps

        # Values should be non-negative (MW generation)
        assert hourly_gen.min().min() >= 0

    def test_hourly_generation_with_aggregation(self, initialized_data_manager):
        """Test hourly generation with region aggregation."""
        hourly_gen = get_distributed_gen_hourly_generation(
            year=2025,
            weather_year=2012,
            regions=["AB"],
            region_aggregations={"AB": ["A", "B"]},
            tz_offset=None,
        )

        # Get individual generation
        gen_a = get_distributed_gen_hourly_generation(
            year=2025, weather_year=2012, regions=["A"], region_aggregations=None
        )
        gen_b = get_distributed_gen_hourly_generation(
            year=2025, weather_year=2012, regions=["B"], region_aggregations=None
        )

        # Aggregated generation is (capacity-weighted avg profile)*total capacity
        # which algebraically equals gen_a + gen_b
        capacity = get_distributed_gen_capacity(
            year=2025, regions=["AB"], region_aggregations={"AB": ["A", "B"]}
        )
        total_capacity = capacity.loc[capacity["region"] == "AB", "capacity_mw"].values[
            0
        ]
        assert total_capacity == 225.0  # A=150, B=75

        # Generation bounds
        assert hourly_gen["AB"].max() <= total_capacity
        assert hourly_gen["AB"].min() >= 0

        # Verify equality with sum of individual region generation
        expected_sum = gen_a["A"] + gen_b["B"]
        assert np.allclose(hourly_gen["AB"].values, expected_sum.values, rtol=1e-6)

    def test_hourly_generation_with_timezone_offset(self, initialized_data_manager):
        """Test that timezone offset is applied to hourly generation."""
        gen_no_offset = get_distributed_gen_hourly_generation(
            year=2025,
            weather_year=2012,
            regions=["A"],
            region_aggregations=None,
            tz_offset=None,
        )

        gen_with_offset = get_distributed_gen_hourly_generation(
            year=2025,
            weather_year=2012,
            regions=["A"],
            region_aggregations=None,
            tz_offset=-2,  # Shift back 2 hours
        )

        # After shifting back 2 hours, hour 3 should have value from hour 1
        assert np.isclose(
            gen_with_offset.loc[1, "A"], gen_no_offset.loc[3, "A"], rtol=1e-5
        )

    def test_hourly_generation_multiple_weather_years(self, initialized_data_manager):
        """Test hourly generation with multiple weather years."""
        hourly_gen = get_distributed_gen_hourly_generation(
            year=2025,
            weather_year=[2012, 2013],
            regions=["A"],
            region_aggregations=None,
            tz_offset=None,
        )

        # Should have 16 time steps (8 * 2 years)
        assert len(hourly_gen) == 16
        assert "A" in hourly_gen.columns

    def test_hourly_generation_year_none_raises_error(self, initialized_data_manager):
        """Test that year=None raises an error."""
        with pytest.raises(ValueError, match="Model year must be provided"):
            get_distributed_gen_hourly_generation(
                year=None,
                weather_year=2012,
                regions=["A"],
            )

    def test_hourly_generation_empty_when_no_capacity(
        self, temp_csv_data_folder, sample_profile_data
    ):
        """Test that generation is empty when no capacity data exists."""
        # Setup with profiles but no capacity
        empty_capacity = pd.DataFrame(columns=["region", "capacity_mw", "year"])
        empty_capacity.to_csv(temp_csv_data_folder / "empty_capacity.csv", index=False)
        sample_profile_data.to_csv(
            temp_csv_data_folder / "distributed_profiles.csv", index=False
        )

        settings = {
            "distributed_capacity_table": "empty_capacity.csv",
            "distributed_profiles_table": "distributed_profiles.csv",
        }
        initialize_data_manager(settings, temp_csv_data_folder)

        hourly_gen = get_distributed_gen_hourly_generation(
            year=2025, weather_year=2012, regions=["A"]
        )

        assert hourly_gen.empty

        DataManager().close()

    def test_hourly_generation_empty_when_no_profiles(
        self, temp_csv_data_folder, sample_capacity_data
    ):
        """Test that generation is empty when no profile data exists."""
        # Setup with capacity but no profiles
        sample_capacity_data.to_csv(
            temp_csv_data_folder / "distributed_capacity.csv", index=False
        )
        empty_profiles = pd.DataFrame(
            columns=["region", "time_index", "value", "weather_year"]
        )
        empty_profiles.to_csv(temp_csv_data_folder / "empty_profiles.csv", index=False)

        settings = {
            "distributed_capacity_table": "distributed_capacity.csv",
            "distributed_profiles_table": "empty_profiles.csv",
        }
        initialize_data_manager(settings, temp_csv_data_folder)

        hourly_gen = get_distributed_gen_hourly_generation(
            year=2025, weather_year=2012, regions=["A"]
        )

        assert hourly_gen.empty

        DataManager().close()


class TestAutoFillSettings:
    """Tests for the @auto_fill_settings decorator integration."""

    def test_auto_fill_from_settings(self, initialized_data_manager):
        """Test that parameters are auto-filled from settings when available."""
        # Create a mock Settings object
        mock_settings = Mock()
        mock_settings.get = Mock(
            side_effect=lambda key, default=None: {
                "model_year": 2025,
                "model_regions": ["A", "B"],
                "region_aggregations": {"AB": ["A", "B"]},
                "weather_year": 2012,
                "utc_offset": 0,
            }.get(key, default)
        )

        with patch(
            "powergenome.settings.get_current_settings",
            return_value=mock_settings,
        ):
            # Call without providing parameters (should use settings)
            capacity = get_distributed_gen_capacity()

            # Should use model_year=2025 and model_regions=["A", "B"]
            # but region_aggregations will aggregate them into "AB"
            assert len(capacity) == 1
            assert capacity["region"].values[0] == "AB"

    def test_explicit_params_override_settings(self, initialized_data_manager):
        """Test that explicit parameters override settings values."""
        # Create a mock Settings object
        mock_settings = Mock()
        mock_settings.get = Mock(
            side_effect=lambda key, default=None: {
                "model_year": 2025,
                "model_regions": ["A", "B"],
                "region_aggregations": None,
            }.get(key, default)
        )

        with patch(
            "powergenome.settings.get_current_settings",
            return_value=mock_settings,
        ):
            # Provide explicit year that differs from settings
            capacity = get_distributed_gen_capacity(year=2030, regions=["C"])

            # Should use explicit values, not settings
            assert len(capacity) == 1
            assert capacity["region"].values[0] == "C"
            assert capacity["capacity_mw"].values[0] == 300.0  # 2030 value for C


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_year_in_data_no_interpolation(self, temp_csv_data_folder):
        """Test behavior when only a single year is available."""
        # Create data with only one year
        single_year_data = pd.DataFrame(
            {"region": ["A", "B"], "capacity_mw": [100.0, 50.0], "year": [2025, 2025]}
        )
        single_year_data.to_csv(
            temp_csv_data_folder / "single_year_capacity.csv", index=False
        )

        settings = {"distributed_capacity_table": "single_year_capacity.csv"}
        initialize_data_manager(settings, temp_csv_data_folder)

        # Request a different year - should use the only available year
        capacity = get_distributed_gen_capacity(year=2030, regions=["A"])

        assert len(capacity) == 1
        assert capacity["capacity_mw"].values[0] == 100.0  # Uses 2025 value

        DataManager().close()

    def test_profile_values_clamped_to_valid_range(self, initialized_data_manager):
        """Test that profile values are within [0, 1] range."""
        profiles = get_distributed_gen_profiles(
            weather_year=2012,
            regions=["A", "B", "C"],
            region_aggregations=None,
            year=2025,
        )

        # All values should be between 0 and 1
        assert profiles.min().min() >= 0
        assert profiles.max().max() <= 1

    def test_large_timezone_offset(self, initialized_data_manager):
        """Test that large timezone offsets wrap correctly."""
        profiles_no_offset = get_distributed_gen_profiles(
            weather_year=2012, regions=["A"], tz_offset=None, year=2025
        )

        # Offset equal to length should result in same data
        profiles_full_cycle = get_distributed_gen_profiles(
            weather_year=2012, regions=["A"], tz_offset=8, year=2025
        )

        # Should be identical (wrapped around)
        assert np.allclose(
            profiles_no_offset["A"].values, profiles_full_cycle["A"].values
        )

    def test_mixed_data_availability_across_regions(self, temp_csv_data_folder):
        """
        Test when different regions have data for different years. Should only return
        the region that can provide data for the requested year.
        """
        mixed_data = pd.DataFrame(
            {
                "region": ["A", "A", "B", "B", "C"],
                "capacity_mw": [100.0, 200.0, 50.0, 100.0, 150.0],
                "year": [2020, 2030, 2025, 2030, 2027],
            }
        )
        mixed_data.to_csv(temp_csv_data_folder / "mixed_capacity.csv", index=False)

        # Cleanup and reinitialize
        try:
            DataManager().close()
        except Exception:
            # Ignore: prior tests may not have initialized DataManager
            pass

        settings = {"distributed_capacity_table": "mixed_capacity.csv"}
        initialize_data_manager(settings, temp_csv_data_folder)

        # Request year 2025 - A needs interpolation, B exact match, C needs extrapolation
        capacity = get_distributed_gen_capacity(
            year=2025, regions=["A", "B", "C"], region_aggregations=None
        )

        # All three regions should be returned
        assert len(capacity) == 1
        assert set(capacity["region"]) == {"B"}

        # B should have exact value
        assert capacity.loc[capacity["region"] == "B", "capacity_mw"].values[0] == 50.0

        DataManager().close()


# Additional tests to cover single-element list weather_year and missing weather_year in settings


def test_load_profiles_single_list_weather_year(initialized_data_manager):
    """Profiles loaded with single-element list weather_year should match int case."""
    profiles_list = get_distributed_gen_profiles(
        weather_year=[2012],
        regions=["A"],
        region_aggregations=None,
        tz_offset=None,
        year=2025,
    )
    profiles_int = get_distributed_gen_profiles(
        weather_year=2012,
        regions=["A"],
        region_aggregations=None,
        tz_offset=None,
        year=2025,
    )
    assert len(profiles_list) == len(profiles_int) == 8
    assert np.allclose(profiles_list["A"].values, profiles_int["A"].values)


def test_hourly_generation_single_list_weather_year(initialized_data_manager):
    """Hourly generation with single-element list behaves like int weather_year."""
    gen_list = get_distributed_gen_hourly_generation(
        year=2025,
        weather_year=[2012],
        regions=["A"],
        region_aggregations=None,
        tz_offset=None,
    )
    gen_int = get_distributed_gen_hourly_generation(
        year=2025,
        weather_year=2012,
        regions=["A"],
        region_aggregations=None,
        tz_offset=None,
    )
    assert len(gen_list) == len(gen_int) == 8
    assert np.allclose(gen_list["A"].values, gen_int["A"].values)


def test_auto_fill_missing_weather_year(initialized_data_manager):
    """Auto-filled profiles when weather_year is absent in settings should load all years."""
    mock_settings = Mock()
    mock_settings.get = Mock(
        side_effect=lambda key, default=None: {
            "model_year": 2025,
            "model_regions": ["A"],
            # No weather_year key provided
            "region_aggregations": None,
            "utc_offset": 0,
        }.get(key, default)
    )
    with patch(
        "powergenome.settings.get_current_settings",
        return_value=mock_settings,
    ):
        profiles = get_distributed_gen_profiles(regions=["A"], year=2025)
        # Expect concatenation of all available weather years (2 * 8 = 16)
        assert len(profiles) == 16
        assert "A" in profiles.columns


@pytest.fixture(autouse=True)
def cleanup_data_manager():
    """Ensure DataManager is properly cleaned up after each test."""
    yield
    try:
        DataManager().close()
    except Exception:
        # Ignore: DataManager may already be closed or never initialized in test
        pass
