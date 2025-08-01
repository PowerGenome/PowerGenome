"""
Test the Settings module functionality.

This module tests the Settings class and related functions for managing
PowerGenome configuration parameters.
"""

import logging
import tempfile
from pathlib import Path
from unittest.mock import mock_open, patch

import pandas as pd
import pytest
import yaml

import powergenome
from powergenome.settings import (
    Settings,
    add_model_tags_to_gen_columns,
    apply_all_tag_to_regions,
    assign_model_planning_years,
    build_scenario_settings,
    fix_param_names,
    get_current_settings,
    load_settings,
)

logger = logging.getLogger(powergenome.__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    # More extensive test-like formatter...
    "%(asctime)s [%(levelname)8s] %(name)s:%(lineno)s %(message)s",
    # This is the datetime format string.
    "%Y-%m-%d %H:%M:%S",
)
handler.setFormatter(formatter)
logger.addHandler(handler)


class TestSettings:
    """Test the Settings class functionality."""

    def test_init_with_data(self):
        """Test Settings initialization with data dictionary."""
        data = {"key1": "value1", "key2": "value2"}
        settings = Settings(data=data)
        assert settings["key1"] == "value1"
        assert settings["key2"] == "value2"

    def test_init_with_config_path(self, tmp_path):
        """Test Settings initialization with config file path."""
        # Create a temporary YAML file
        config_data = {
            "model_regions": ["region1", "region2", "region3"],
            "test_key": "test_value",
            "nested": {"key": "value"},
        }
        config_file = tmp_path / "test_config.yml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        settings = Settings(config_path=config_file)
        assert settings["model_regions"] == ["region1", "region2", "region3"]
        assert settings["test_key"] == "test_value"
        assert settings["nested"]["key"] == "value"

    def test_init_with_both_data_and_config(self, tmp_path):
        """Test Settings initialization with both data and config path."""
        # Data should be loaded first, then config can override
        data = {
            "model_regions": ["region1", "region2", "region3"],
            "key1": "data_value",
            "key2": "data_value",
        }
        config_data = {
            "model_regions": ["region1", "region2", "region3"],
            "key1": "config_value",
            "key3": "config_value",
        }

        config_file = tmp_path / "test_config.yml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        settings = Settings(data=data, config_path=config_file)
        # Model regions should be the same as the config data
        assert settings["model_regions"] == ["region1", "region2", "region3"]
        # Config should override data for key1
        assert settings["key1"] == "config_value"
        # Data value should remain for key2
        assert settings["key2"] == "data_value"
        # Config value should be present for key3
        assert settings["key3"] == "config_value"

    def test_from_dict_classmethod(self):
        """Test the from_dict class method."""
        data = {"key1": "value1", "key2": "value2"}
        settings = Settings.from_dict(data)
        assert settings["key1"] == "value1"
        assert settings["key2"] == "value2"

    def test_for_scenario_classmethod(self):
        """Test the for_scenario class method."""
        base_data = {"base_key": "base_value", "common_key": "base_value"}
        scenario_data = {
            "scenario_key": "scenario_value",
            "common_key": "scenario_value",
        }

        base_settings = Settings(data=base_data)
        scenario_settings = Settings.for_scenario(base_settings, scenario_data)

        # Base settings should remain unchanged
        assert base_settings["base_key"] == "base_value"
        assert base_settings["common_key"] == "base_value"

        # Scenario settings should have both base and scenario data
        assert scenario_settings["base_key"] == "base_value"
        assert scenario_settings["scenario_key"] == "scenario_value"
        # Scenario data should override base data
        assert scenario_settings["common_key"] == "scenario_value"

    def test_for_scenario_with_get_current_settings(self):
        """Test that Settings.for_scenario works with get_current_settings() context manager."""
        base_data = {"base_key": "base_value", "common_key": "base_value"}
        scenario_data = {
            "scenario_key": "scenario_value",
            "common_key": "scenario_value",
        }

        base_settings = Settings(data=base_data)
        scenario_settings = Settings.for_scenario(base_settings, scenario_data)

        # Test that scenario settings work as context manager with get_current_settings
        with scenario_settings:
            current = get_current_settings()

            # Should get the scenario settings instance
            assert current is scenario_settings

            # Should have base settings preserved
            assert current["base_key"] == "base_value"

            # Should have scenario data added
            assert current["scenario_key"] == "scenario_value"

            # Should have scenario overrides applied
            assert current["common_key"] == "scenario_value"

        # After context, should raise error again
        with pytest.raises(RuntimeError):
            get_current_settings()

    def test_context_manager(self):
        """Test Settings as a context manager."""
        settings = Settings(data={"test_key": "test_value"})

        # Before context manager
        with pytest.raises(RuntimeError):
            get_current_settings()

        # Inside context manager
        with settings:
            current = get_current_settings()
            assert current is settings
            assert current["test_key"] == "test_value"

        # After context manager
        with pytest.raises(RuntimeError):
            get_current_settings()

    def test_get_method(self):
        """Test the get method with default values."""
        settings = Settings(data={"key1": "value1"})

        assert settings.get("key1") == "value1"
        assert settings.get("missing_key") is None
        assert settings.get("missing_key", "default") == "default"

    def test_getitem_setitem(self):
        """Test dictionary-like access with __getitem__ and __setitem__."""
        settings = Settings()

        # Test setting and getting values
        settings["test_key"] = "test_value"
        assert settings["test_key"] == "test_value"

        # Test KeyError for missing key
        with pytest.raises(KeyError):
            _ = settings["missing_key"]

    def test_pop_method(self):
        """Test the pop method."""
        settings = Settings(
            data={
                "model_regions": ["region1", "region2", "region3"],
                "key1": "value1",
                "key2": "value2",
            }
        )

        # Test popping existing key
        value = settings.pop("key1")
        assert value == "value1"
        assert "key1" not in settings._data

        # Test popping non-existent key with default
        value = settings.pop("missing_key", "default")
        assert value == "default"

        # Test popping non-existent key without default
        with pytest.raises(KeyError):
            settings.pop("missing_key")

    def test_getattr(self):
        """Test attribute-style access."""
        settings = Settings(
            data={
                "model_regions": ["region1", "region2", "region3"],
                "attr_key": "attr_value",
            }
        )

        assert settings.attr_key == "attr_value"
        assert settings.missing_attr is None

    def test_copy_methods(self):
        """Test shallow and deep copy methods."""
        original_data = {
            "model_regions": ["region1", "region2", "region3"],
            "key1": "value1",
            "nested": {"key2": "value2"},
        }
        settings = Settings(data=original_data)

        # Test shallow copy
        shallow_copy = settings.__copy__()
        assert shallow_copy["key1"] == "value1"
        assert shallow_copy["nested"] is settings["nested"]  # Same reference

        # Test deep copy
        deep_copy = settings.__deepcopy__({})
        assert deep_copy["key1"] == "value1"
        assert deep_copy["nested"] is not settings["nested"]  # Different reference

    def test_to_dict(self):
        """Test converting Settings to dictionary."""
        data = {
            "model_regions": ["region1", "region2", "region3"],
            "key1": "value1",
            "key2": "value2",
        }
        settings = Settings(data=data)

        result_dict = settings.to_dict()
        assert result_dict == data
        assert result_dict is not data  # Should be a copy

    def test_update_method(self):
        """Test the update method."""
        settings = Settings(data={"key1": "value1"})

        updates = {"key2": "value2", "key1": "updated_value"}
        settings.update(updates)

        assert settings["key1"] == "updated_value"
        assert settings["key2"] == "value2"

    def test_get_data(self):
        """Test the get_data method returns reference to internal data."""
        data = {"key1": "value1"}
        settings = Settings(data=data)

        internal_data = settings.get_data()
        assert internal_data is settings._data

        # Modifying the returned data should affect the settings
        internal_data["key2"] = "value2"
        assert settings["key2"] == "value2"

    def test_load_settings_method(self, tmp_path):
        """Test the load_settings method."""
        settings = Settings()
        # Create a temporary YAML file
        config_data = {
            "model_regions": ["region1", "region2", "region3"],
            "loaded_key": "loaded_value",
        }
        config_file = tmp_path / "test_config.yml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        settings.load_settings(config_file)
        assert settings["model_regions"] == ["region1", "region2", "region3"]
        assert settings["loaded_key"] == "loaded_value"


class TestGlobalSettings:
    """Test the global settings functionality."""

    def setup_method(self):
        """Clear global settings before each test."""
        Settings.clear_global()

    def teardown_method(self):
        """Clear global settings after each test."""
        Settings.clear_global()

    def test_set_and_get_global_settings(self):
        """Test setting and getting global settings."""
        settings = Settings(data={"model_year": 2030, "test_key": "test_value"})

        # Set global settings
        Settings.set_global(settings)

        # Get global settings
        global_settings = Settings.get_global()
        assert global_settings is settings
        assert global_settings["model_year"] == 2030
        assert global_settings["test_key"] == "test_value"

    def test_get_global_without_setting_raises_error(self):
        """Test that getting global settings without setting them raises RuntimeError."""
        with pytest.raises(RuntimeError, match="No global settings have been set"):
            Settings.get_global()

    def test_clear_global_settings(self):
        """Test clearing global settings."""
        settings = Settings(data={"model_year": 2030})
        Settings.set_global(settings)

        # Verify it was set
        assert Settings.get_global() is settings

        # Clear global settings
        Settings.clear_global()

        # Verify it was cleared
        with pytest.raises(RuntimeError, match="No global settings have been set"):
            Settings.get_global()

    def test_get_current_settings_with_global_fallback(self):
        """Test that get_current_settings falls back to global settings."""
        settings = Settings(data={"model_year": 2030, "global_key": "global_value"})
        Settings.set_global(settings)

        # Should work without context manager
        current = get_current_settings()
        assert current is settings
        assert current["model_year"] == 2030
        assert current["global_key"] == "global_value"

    def test_context_overrides_global_settings(self):
        """Test that context settings override global settings."""
        global_settings = Settings(data={"model_year": 2030, "source": "global"})
        context_settings = Settings(data={"model_year": 2035, "source": "context"})

        Settings.set_global(global_settings)

        # Outside context, should get global settings
        current = get_current_settings()
        assert current is global_settings
        assert current["source"] == "global"

        # Inside context, should get context settings
        with context_settings:
            current = get_current_settings()
            assert current is context_settings
            assert current["source"] == "context"

        # After context, should return to global settings
        current = get_current_settings()
        assert current is global_settings
        assert current["source"] == "global"

    def test_global_settings_update_affects_get_current_settings(self):
        """Test that updates to global settings are reflected in get_current_settings."""
        settings = Settings(data={"model_year": 2030})
        Settings.set_global(settings)

        # Update the global settings
        settings.update({"new_key": "new_value"})

        # Should be reflected in get_current_settings
        current = get_current_settings()
        assert current["new_key"] == "new_value"

    def test_multiple_global_settings_updates(self):
        """Test setting different global settings instances."""
        settings1 = Settings(data={"source": "first", "model_year": 2030})
        settings2 = Settings(data={"source": "second", "model_year": 2035})

        # Set first global settings
        Settings.set_global(settings1)
        current = get_current_settings()
        assert current["source"] == "first"
        assert current["model_year"] == 2030

        # Set second global settings
        Settings.set_global(settings2)
        current = get_current_settings()
        assert current["source"] == "second"
        assert current["model_year"] == 2035

    def test_global_settings_with_for_scenario(self):
        """Test global settings work with Settings.for_scenario."""
        base_settings = Settings(data={"base_key": "base_value", "model_year": 2030})
        scenario_data = {"scenario_key": "scenario_value", "model_year": 2035}

        scenario_settings = Settings.for_scenario(base_settings, scenario_data)
        Settings.set_global(scenario_settings)

        current = get_current_settings()
        assert current["base_key"] == "base_value"
        assert current["scenario_key"] == "scenario_value"
        assert current["model_year"] == 2035  # Overridden by scenario

    def test_global_settings_isolation_from_context(self):
        """Test that global settings don't interfere with context isolation."""
        global_settings = Settings(data={"source": "global"})
        context_settings1 = Settings(data={"source": "context1"})
        context_settings2 = Settings(data={"source": "context2"})

        Settings.set_global(global_settings)

        # Nested contexts should work independently of global
        with context_settings1:
            assert get_current_settings()["source"] == "context1"

            with context_settings2:
                assert get_current_settings()["source"] == "context2"

            # Should return to context1, not global
            assert get_current_settings()["source"] == "context1"

        # Should return to global
        assert get_current_settings()["source"] == "global"


class TestGetCurrentSettings:
    """Test the get_current_settings function."""

    def setup_method(self):
        """Clear global settings before each test."""
        Settings.clear_global()

    def teardown_method(self):
        """Clear global settings after each test."""
        Settings.clear_global()

    def test_get_current_settings_with_context(self):
        """Test getting current settings within context."""
        settings = Settings(data={"test_key": "test_value"})

        with settings:
            current = get_current_settings()
            assert current is settings
            assert current["test_key"] == "test_value"

    def test_get_current_settings_with_global_only(self):
        """Test getting current settings with only global settings."""
        settings = Settings(data={"test_key": "test_value"})
        Settings.set_global(settings)

        current = get_current_settings()
        assert current is settings
        assert current["test_key"] == "test_value"

    def test_get_current_settings_without_any_settings(self):
        """Test that get_current_settings raises error without any settings."""
        with pytest.raises(RuntimeError, match="No settings are currently available"):
            get_current_settings()

    def test_get_current_settings_error_message_mentions_both_options(self):
        """Test that error message mentions both context and global options."""
        with pytest.raises(RuntimeError) as exc_info:
            get_current_settings()

        error_message = str(exc_info.value)
        assert "with settings:" in error_message
        assert "Settings.set_global" in error_message


@pytest.fixture(autouse=True)
def noop_assign_and_tags(monkeypatch):
    # Prevent assign_model_planning_years from altering our settings
    from powergenome import settings as settings_module

    monkeypatch.setattr(
        settings_module, "assign_model_planning_years", lambda settings, year: None
    )


class TestApplyAllTagToRegions:
    """Test the apply_all_tag_to_regions function."""

    def test_apply_all_tag_to_regions_basic(self):
        """Test basic functionality of apply_all_tag_to_regions."""
        settings = {
            "model_regions": ["region1", "region2", "region3"],
            "renewables_clusters": [
                {
                    "region": "all",
                    "technology": "wind",
                    "bin": {"feature": "lcoe", "q": 4},
                },
                {
                    "region": "region1",
                    "technology": "solar",
                    "filter": {"feature": "lcoe", "max": 50},
                },
            ],
        }

        result = apply_all_tag_to_regions(settings)

        # Should have entries for all regions for wind technology
        wind_entries = [
            entry
            for entry in result["renewables_clusters"]
            if entry["technology"] == "wind"
        ]
        assert len(wind_entries) == 3

        # Should have one entry for solar in region1
        solar_entries = [
            entry
            for entry in result["renewables_clusters"]
            if entry["technology"] == "solar"
        ]
        assert len(solar_entries) == 1
        assert solar_entries[0]["region"] == "region1"

    def test_apply_all_tag_to_regions(self, caplog):
        settings = {
            "model_regions": ["a", "b", "c"],
            "renewables_clusters": [
                {
                    "region": "all",
                    "technology": "landbasedwind",
                    "bin": {"feature": "lcoe", "q": 4},
                },
                {
                    "region": "b",
                    "technology": "landbasedwind",
                    "filter": {"feature": "lcoe", "max": 50},
                },
                {"region": "all", "technology": "utilitypv", "group": ["state"]},
                {
                    "region": "all",
                    "technology": "offshorewind",
                    "pref_site": True,
                    "bin": {"feature": "lcoe", "q": 4},
                },
                {
                    "region": "c",
                    "technology": "offshorewind",
                    "pref_site": True,
                    "cluster": {"feature": "lcoe", "n_clusters": 4},
                },
                {
                    "region": "all",
                    "technology": "offshorewind",
                    "pref_site": True,
                    "group": ["metro_id"],
                },
            ],
        }

        # Check for warning that "all" is applied to offshore wind more than once
        with caplog.at_level(logging.WARNING):
            settings = apply_all_tag_to_regions(settings)

        assert "Multiple 'all' tags applied" in caplog.text

        assert len(settings["renewables_clusters"]) == 9
        for d in settings["renewables_clusters"]:
            if d["technology"] == "landbasedwind":
                if d["region"] == "b":
                    assert "filter" in d.keys()
                else:
                    assert "bin" in d.keys()
            if d["technology"] == "utilitypv":
                assert "group" in d.keys()
            if d["technology"] == "offshorewind":
                if d["region"] == "c":
                    assert "cluster" in d.keys()
                else:
                    assert "group" in d.keys()

        # Test two ways to raise a KeyError: no "region" and no "technology" when region is "all"
        d = {"technology": "solarpv"}
        settings["renewables_clusters"].append(d)
        with pytest.raises(KeyError):
            apply_all_tag_to_regions(settings)

        settings["renewables_clusters"].pop()

        d = {"region": "ALL"}
        settings["renewables_clusters"].append(d)
        with pytest.raises(KeyError):
            apply_all_tag_to_regions(settings)

        settings = {"model_regions": ["a", "b", "c"], "renewables_clusters": None}
        apply_all_tag_to_regions(settings)
        settings = {"model_regions": ["a", "b", "c"]}
        apply_all_tag_to_regions(settings)

    def test_apply_all_tag_to_regions_missing_region(self):
        """Test that missing region tag raises KeyError."""
        settings = {
            "model_regions": ["region1"],
            "renewables_clusters": [
                {"technology": "wind", "bin": {"feature": "lcoe", "q": 4}}
            ],
        }

        with pytest.raises(KeyError, match="Entry missing 'region' tag"):
            apply_all_tag_to_regions(settings)

    def test_apply_all_tag_to_regions_missing_technology(self):
        """Test that missing technology tag raises KeyError."""
        settings = {
            "model_regions": ["region1"],
            "renewables_clusters": [
                {"region": "all", "bin": {"feature": "lcoe", "q": 4}}
            ],
        }

        with pytest.raises(KeyError, match="Entry for all missing 'technology' tag"):
            apply_all_tag_to_regions(settings)

    def test_apply_all_tag_to_regions_multiple_all_entries(self, caplog):
        """Test handling of multiple 'all' entries for same technology."""
        settings = {
            "model_regions": ["region1", "region2"],
            "renewables_clusters": [
                {
                    "region": "all",
                    "technology": "wind",
                    "bin": {"feature": "lcoe", "q": 4},
                },
                {
                    "region": "all",
                    "technology": "wind",
                    "filter": {"feature": "lcoe", "max": 50},
                },
            ],
        }

        result = apply_all_tag_to_regions(settings)

        # Should log a warning about multiple 'all' entries
        assert "Multiple 'all' tags applied to technology wind" in caplog.text

        # Should only have entries for the last 'all' configuration
        wind_entries = [
            entry
            for entry in result["renewables_clusters"]
            if entry["technology"] == "wind"
        ]
        assert len(wind_entries) == 2  # One for each region


class TestAssignModelPlanningYears:
    """Test the assign_model_planning_years function."""

    # The function is called with a dictionary containing the key 'model_periods' with a list of tuples as value, and an integer year.
    def test_with_model_periods(self):
        # Prepare input
        _settings = {
            "model_periods": [(2030, 2040), (2041, 2050)],
            "model_year": [2030, 2040],
            "model_first_planning_year": [2030, 2041],
        }
        year = 2040

        # Execute function
        result = assign_model_planning_years(_settings, year)

        # Check output
        assert result["model_first_planning_year"] == 2030
        assert result["model_year"] == 2040

    # The function is called with an empty dictionary.
    def test_with_empty_dictionary(self):
        # Prepare input
        _settings = {}
        year = 2022

        # Execute function
        with pytest.raises(KeyError):
            assign_model_planning_years(_settings, year)

    # The function is called with a dictionary containing the key 'model_first_planning_year' with an integer value, and an integer year.
    def test_with_model_first_planning_year(self):
        # Prepare input
        _settings = {"model_first_planning_year": 2030}
        year = 2030

        # Execute function
        result = assign_model_planning_years(_settings, year)

        # Check output
        assert result["model_first_planning_year"] == 2030
        assert result["model_year"] == 2030

    # The function is called with a dictionary containing the keys 'model_year' and 'model_first_planning_year' with integer values, and an integer year.
    def test_with_model_year_first_planning_year(self):
        # Prepare input
        _settings = {
            "model_year": [2030, 2040],
            "model_first_planning_year": [2030, 2035],
        }
        year = 2040

        # Execute function
        result = assign_model_planning_years(_settings, year)

        # Check output
        assert result["model_first_planning_year"] == 2035
        assert result["model_year"] == 2040

    # The function is called with a dictionary containing the key 'model_periods' with a list of tuples where at least one tuple has length different from 2.
    def test_with_invalid_model_periods_length_tuples(self):
        # Prepare input
        _settings = {
            "model_periods": [(2030, 2040), (2041, 2050), (2051,)],
            "model_year": [2030, 2040],
            "model_first_planning_year": [2030, 2041],
        }
        year = 2030

        # Execute function and assert ValueError is raised
        with pytest.raises(ValueError):
            assign_model_planning_years(_settings, year)

    # The function is called with a dictionary containing the key 'model_periods' with a non-list value.
    def test_with_non_list_model_periods(self):
        # Prepare input
        _settings = {
            "model_periods": "2030-2040",
            "model_year": [2030, 2040],
            "model_first_planning_year": [2030, 2041],
        }
        year = 2030

        # Execute function
        with pytest.raises(ValueError):
            assign_model_planning_years(_settings, year)

    # The function is called with a dictionary containing the keys 'model_year' and 'model_first_planning_year' with values that are not integers or lists of integers.
    def test_invalid_values(self):
        # Prepare input
        _settings = {"model_year": "2040", "model_first_planning_year": "2031"}
        year = 2022

        # Execute function
        with pytest.raises(ValueError):
            assign_model_planning_years(_settings, year)

    def test_with_model_periods_single_tuple(self):
        """Test with single model_periods tuple."""
        settings = {"model_periods": [(2025, 2030), (2035, 2040)]}

        result = assign_model_planning_years(settings, 2030)

        assert result["model_first_planning_year"] == 2025
        assert result["model_year"] == 2030
        assert "model_periods" not in result

    def test_with_model_periods_list_of_tuples(self):
        """Test with list of model_periods tuples."""
        settings = {"model_periods": [(2025, 2030), (2035, 2040)]}

        result = assign_model_planning_years(settings, 2030)

        assert result["model_first_planning_year"] == 2025
        assert result["model_year"] == 2030
        assert "model_periods" not in result

    def test_with_model_periods_different_year(self):
        """Test with model_periods for a different year."""
        settings = {"model_periods": [(2025, 2030), (2035, 2040)]}

        result = assign_model_planning_years(settings, 2040)

        assert result["model_first_planning_year"] == 2035
        assert result["model_year"] == 2040

    def test_with_model_year_and_first_planning_year_scalars(self):
        """Test with scalar model_year and model_first_planning_year."""
        settings = {"model_year": 2030, "model_first_planning_year": 2025}

        result = assign_model_planning_years(settings, 2030)

        assert result["model_first_planning_year"] == 2025
        assert result["model_year"] == 2030
        assert "model_periods" not in result

    def test_with_model_year_and_first_planning_year_lists(self):
        """Test with list model_year and model_first_planning_year."""
        settings = {
            "model_year": [2030, 2040],
            "model_first_planning_year": [2025, 2035],
        }

        result = assign_model_planning_years(settings, 2030)

        assert result["model_first_planning_year"] == 2025
        assert result["model_year"] == 2030

    def test_with_model_year_and_first_planning_year_different_year(self):
        """Test with lists for a different year."""
        settings = {
            "model_year": [2030, 2040],
            "model_first_planning_year": [2025, 2035],
        }

        result = assign_model_planning_years(settings, 2040)

        assert result["model_first_planning_year"] == 2035
        assert result["model_year"] == 2040

    def test_with_model_first_planning_year_only_scalar(self):
        """Test with only scalar model_first_planning_year parameter."""
        settings = {"model_first_planning_year": 2025}

        result = assign_model_planning_years(settings, 2025)

        assert result["model_first_planning_year"] == 2025
        assert result["model_year"] == 2025

    def test_with_model_first_planning_year_only_list(self):
        """Test with only list model_first_planning_year parameter."""
        settings = {"model_first_planning_year": [2025, 2035]}

        result = assign_model_planning_years(settings, 2025)

        assert result["model_first_planning_year"] == [2025, 2035]
        assert result["model_year"] == [2025, 2035]

    def test_with_model_first_planning_year_only_different_year(self):
        """Test with list model_first_planning_year for different year."""
        settings = {"model_first_planning_year": [2025, 2035]}

        result = assign_model_planning_years(settings, 2035)

        assert result["model_first_planning_year"] == [2025, 2035]
        assert result["model_year"] == [2025, 2035]

    def test_with_invalid_model_periods_length(self):
        """Test that invalid model_periods tuple length raises ValueError."""
        settings = {"model_periods": [2025, 2030, 2035]}  # 3 elements instead of 2

        with pytest.raises(
            ValueError,
            match="The settings parameter 'model_periods' must be a list of tuples. It is currently \\[2025, 2030, 2035\\]",
        ):
            assign_model_planning_years(settings, 2030)

    def test_with_mixed_length_tuples(self):
        """Test that mixed length tuples raise ValueError."""
        settings = {
            "model_periods": [(2025, 2030), (2035, 2040, 2045)]  # Mixed lengths
        }

        with pytest.raises(
            ValueError,
            match="The tuples in settings parameter 'model_periods' must all be 2 years",
        ):
            assign_model_planning_years(settings, 2030)

    def test_with_non_tuple_model_periods(self):
        """Test that non-tuple model_periods raises ValueError."""
        settings = {"model_periods": [2025, 2030]}  # Not tuples

        with pytest.raises(ValueError, match="must be a list of tuples"):
            assign_model_planning_years(settings, 2030)

    def test_with_mixed_tuple_and_non_tuple(self):
        """Test that mixed tuple and non-tuple raises ValueError."""
        settings = {"model_periods": [(2025, 2030), 2035]}  # Mixed types

        with pytest.raises(ValueError, match="must be a list of tuples"):
            assign_model_planning_years(settings, 2030)

    def test_with_invalid_year_values_strings(self):
        """Test that string year values raises ValueError."""
        settings = {
            "model_year": ["2030"],  # String instead of int
            "model_first_planning_year": [2025],
        }

        with pytest.raises(ValueError, match="must be integers"):
            assign_model_planning_years(settings, 2030)

    def test_with_invalid_year_values_floats(self):
        """Test that float year values raises ValueError."""
        settings = {
            "model_year": [2030.5],  # Float instead of int
            "model_first_planning_year": [2025],
        }

        with pytest.raises(ValueError, match="must be integers"):
            assign_model_planning_years(settings, 2030)

    def test_with_mixed_invalid_year_values(self):
        """Test that mixed valid and invalid year values raises ValueError."""
        settings = {
            "model_year": [2030, "2040"],  # Mixed types
            "model_first_planning_year": [2025, 2035],
        }

        with pytest.raises(ValueError, match="must be integers"):
            assign_model_planning_years(settings, 2030)

    def test_with_missing_required_keys(self):
        """Test that missing required keys raises KeyError."""
        settings = {}

        with pytest.raises(
            KeyError, match="should include either the key 'model_periods'"
        ):
            assign_model_planning_years(settings, 2030)

    def test_with_year_not_in_model_periods(self):
        """Test that year not in model_periods raises ValueError."""
        settings = {"model_periods": [(2025, 2030)]}

        with pytest.raises(ValueError, match="year 2040 is in your scenario"):
            assign_model_planning_years(settings, 2040)

    def test_with_year_not_in_model_year_list(self):
        """Test that year not in model_year list raises ValueError."""
        settings = {
            "model_year": [2030, 2040],
            "model_first_planning_year": [2025, 2035],
        }

        with pytest.raises(ValueError, match="year 2050 is in your scenario"):
            assign_model_planning_years(settings, 2050)

    def test_removes_original_keys(self):
        """Test that original keys are removed from settings."""
        settings = {
            "model_periods": [(2025, 2030)],  # This should be removed
            "model_year": [2040],
            "model_first_planning_year": [2035],
        }

        result = assign_model_planning_years(settings, 2030)

        assert "model_periods" not in result
        assert result["model_year"] == 2030
        assert result["model_first_planning_year"] == 2025

    def test_preserves_other_settings(self):
        """Test that other settings are preserved."""
        settings = {
            "model_periods": [(2025, 2030)],
            "other_setting": "other_value",
            "nested_setting": {"key": "value"},
        }

        result = assign_model_planning_years(settings, 2030)

        assert result["other_setting"] == "other_value"
        assert result["nested_setting"]["key"] == "value"

    def test_with_case_id_in_error_message(self):
        """Test that case_id is included in error message when available."""
        settings = {"case_id": "test_case", "model_periods": [(2025, 2030)]}

        with pytest.raises(ValueError, match="case test_case"):
            assign_model_planning_years(settings, 2040)


class TestAddModelTagsToGenColumns:
    """Test the add_model_tags_to_gen_columns function."""

    def test_returns_input_unmodified_if_not_list(self):
        """Test that non-list generator_columns are returned unchanged."""
        model_tags = {"tag1": {"tech1": 1}}
        regional_tags = {}
        generator_columns = "not_a_list"

        result = add_model_tags_to_gen_columns(
            model_tags, regional_tags, generator_columns
        )
        assert result == generator_columns

    def test_adds_model_tags_to_gen_columns(self):
        """Test that model tags are added to generator columns."""
        model_tags = {"cost": {"solar": 100, "wind": 150}}
        regional_tags = {"region1": {"other_tag": {"solar": 20, "wind": 25}}}
        generator_columns = ["capacity", "output"]

        result = add_model_tags_to_gen_columns(
            model_tags, regional_tags, generator_columns
        )

        assert "cost" in result
        assert "other_tag" in result
        assert "capacity" in result
        assert "output" in result

    def test_does_not_duplicate_existing_columns(self):
        """Test that existing columns are not duplicated."""
        model_tags = {"cost": {"solar": 100}}
        regional_tags = {}
        generator_columns = ["capacity", "cost", "output"]

        result = add_model_tags_to_gen_columns(
            model_tags, regional_tags, generator_columns
        )

        assert result.count("cost") == 1
        assert len(result) == 3

    def test_handles_empty_model_tags(self):
        """Test handling of empty model tags."""
        model_tags = {}
        regional_tags = {}
        generator_columns = ["capacity", "output"]

        result = add_model_tags_to_gen_columns(
            model_tags, regional_tags, generator_columns
        )

        assert result == generator_columns

    def test_handles_none_values(self):
        """Test handling of None values for tags."""
        model_tags = None
        regional_tags = None
        generator_columns = ["capacity", "output"]

        result = add_model_tags_to_gen_columns(
            model_tags, regional_tags, generator_columns
        )

        assert result == generator_columns


class TestFixParamNames:
    """Test the fix_param_names function."""

    def test_fixes_known_parameter_names(self):
        """Test that known parameter names are fixed."""
        settings = {
            "historical_load_region_maps": "old_value",
            "demand_response_resources": "old_value",
            "data_years": "old_value",
        }

        result = fix_param_names(settings)

        assert result["historical_load_region_map"] == "old_value"
        assert result["flexible_demand_resources"] == "old_value"
        assert result["eia_data_years"] == "old_value"

        # Original keys should still exist
        assert "historical_load_region_maps" in result
        assert "demand_response_resources" in result
        assert "data_years" in result

    def test_does_not_affect_other_parameters(self):
        """Test that other parameters are not affected."""
        settings = {"other_param": "value", "historical_load_region_maps": "old_value"}

        result = fix_param_names(settings)

        assert result["other_param"] == "value"
        assert result["historical_load_region_map"] == "old_value"


class TestBuildScenarioSettings:
    """Test the build_scenario_settings function."""

    def test_duplicate_case_year_raises(self):
        # Two identical rows for case 'X' in year 2030 should trigger the duplicate check
        df = pd.DataFrame(
            [
                {"case_id": "X", "year": 2030},
                {"case_id": "X", "year": 2030},
            ]
        )
        with pytest.raises(ValueError) as exc:
            build_scenario_settings({}, df)
        assert "are repeated" in str(exc.value)

    def test_all_years_all_cases_applied(self):
        # settings_management with an all_years->all_cases entry should apply to every scenario
        settings = {"settings_management": {"all_years": {"all_cases": {"foo": 100}}}}
        df = pd.DataFrame([{"case_id": 1, "year": 2040}])
        result = build_scenario_settings(settings, df)

        # Check that our "foo" parameter was injected
        assert 2040 in result
        assert 1 in result[2040]
        out = result[2040][1]
        assert out["foo"] == 100

        # Also verify the built-in fields
        assert out["case_id"] == 1
        assert out["case_period"] == 1

    def test_year_specific_category_level_applied(self):
        # settings_management for a specific year & category should override correctly
        settings = {
            "settings_management": {
                2035: {"category1": {"levelA": {"paramA": "valueA"}}}
            }
        }
        df = pd.DataFrame([{"case_id": "A", "year": 2035, "category1": "levelA"}])
        result = build_scenario_settings(settings, df)
        out = result[2035]["A"]

        # Expect our year-specific setting
        assert out["paramA"] == "valueA"
        assert out["case_id"] == "A"
        assert out["case_period"] == 1

    def test_case_period_increments_per_case(self):
        # Same case 'C' appears in two different years, period should increment
        settings = {"settings_management": {}}
        df = pd.DataFrame(
            [
                {"case_id": "C", "year": 2025},
                {"case_id": "C", "year": 2030},
            ]
        )
        result = build_scenario_settings(settings, df)

        # First appearance: period 1; second: period 2
        assert result[2025]["C"]["case_period"] == 1
        assert result[2030]["C"]["case_period"] == 2

    def test_duplicate_case_year_raises_error(self):
        """Test that duplicate case/year combinations raise ValueError."""
        settings = {}
        scenario_definitions = pd.DataFrame(
            {
                "case_id": ["case1", "case1"],
                "year": [2030, 2030],
                "tech_scenario": ["high", "low"],
            }
        )

        with pytest.raises(ValueError, match="repeated in your scenario definitions"):
            build_scenario_settings(settings, scenario_definitions)

    def test_settings_conflict_raises_error(self):
        """Test that conflicting settings raise ValueError."""
        settings = {
            "settings_management": {
                2030: {
                    "tech_scenario": {"high": {"conflict_param": "high_value"}},
                    "policy_scenario": {"strict": {"conflict_param": "strict_value"}},
                }
            }
        }

        scenario_definitions = pd.DataFrame(
            {
                "case_id": ["case1"],
                "year": [2030],
                "tech_scenario": ["high"],
                "policy_scenario": ["strict"],
            }
        )

        with pytest.raises(ValueError, match="is modified by both"):
            build_scenario_settings(settings, scenario_definitions)


class TestAutoFillSettings:
    """Test the auto_fill_settings decorator."""

    def setup_method(self):
        """Clear global settings before each test."""
        Settings.clear_global()

    def teardown_method(self):
        """Clear global settings after each test."""
        Settings.clear_global()

    def test_direct_mapping_fills_from_global_settings(self):
        """Test that parameters are auto-filled from global settings."""
        from powergenome.settings import auto_fill_settings

        @auto_fill_settings()
        def test_function(model_regions=None, model_year=None):
            return {"model_regions": model_regions, "model_year": model_year}

        settings = Settings(data={"model_regions": ["CA", "TX"], "model_year": 2030})
        Settings.set_global(settings)

        # Should work with global settings, no context needed
        result = test_function()
        assert result["model_regions"] == ["CA", "TX"]
        assert result["model_year"] == 2030

    def test_direct_mapping_fills_from_settings(self):
        """Test that parameters with names matching settings keys are auto-filled."""
        from powergenome.settings import auto_fill_settings

        @auto_fill_settings()
        def test_function(model_regions=None, model_year=None):
            return {"model_regions": model_regions, "model_year": model_year}

        settings = Settings(data={"model_regions": ["CA", "TX"], "model_year": 2030})

        with settings:
            result = test_function()
            assert result["model_regions"] == ["CA", "TX"]
            assert result["model_year"] == 2030

    def test_context_overrides_global_in_decorator(self):
        """Test that context settings override global settings in decorator."""
        from powergenome.settings import auto_fill_settings

        @auto_fill_settings()
        def test_function(model_year=None, source=None):
            return {"model_year": model_year, "source": source}

        global_settings = Settings(data={"model_year": 2030, "source": "global"})
        context_settings = Settings(data={"model_year": 2035, "source": "context"})

        Settings.set_global(global_settings)

        # Should use global settings
        result = test_function()
        assert result["model_year"] == 2030
        assert result["source"] == "global"

        # Should use context settings
        with context_settings:
            result = test_function()
            assert result["model_year"] == 2035
            assert result["source"] == "context"
