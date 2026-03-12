"""
Test the CLI functionality for run_powergenome.

This module tests the command line interface functionality including argument parsing,
main function execution flow, and integration with the PowerGenome components.
"""

import argparse
import logging
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, call, mock_open, patch

import pandas as pd
import pytest

import powergenome
from powergenome.run_powergenome import main, parse_command_line
from powergenome.settings import Settings

logger = logging.getLogger(powergenome.__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s [%(levelname)8s] %(name)s:%(lineno)s %(message)s",
    "%Y-%m-%d %H:%M:%S",
)
handler.setFormatter(formatter)
logger.addHandler(handler)


@pytest.fixture
def test_settings_path():
    """Get path to test system settings directory."""
    return Path(__file__).parent / "test_system" / "settings"


@pytest.fixture
def test_scenario_definitions_path():
    """Get path to test system scenario definitions file."""
    return (
        Path(__file__).parent / "test_system" / "extra_inputs" / "scenario_inputs.csv"
    )


@pytest.fixture()
def test_settings():
    settings = Settings(config_path="tests/test_system/settings")
    settings["RESOURCE_GROUPS"] = "tests/test_system/test_data/resource_groups"
    return settings


class TestParseCommandLine:
    """Test the command line argument parsing functionality."""

    def test_default_arguments(self):
        """Test parsing with default arguments."""
        argv = ["script_name"]
        args = parse_command_line(argv)

        assert args.settings_file == "example_settings.yml"
        assert args.current_gens is True
        assert args.gens is True
        assert args.load is True
        assert args.transmission is True
        assert args.sort_gens is False
        assert args.case_id is None
        assert (
            args.multi_period is True
        )  # Default is True, becomes False when flag is used

    def test_settings_file_argument(self):
        """Test specifying settings file."""
        argv = ["script_name", "-sf", "custom_settings.yml"]
        args = parse_command_line(argv)
        assert args.settings_file == "custom_settings.yml"

        argv = ["script_name", "--settings_file", "another_settings.yml"]
        args = parse_command_line(argv)
        assert args.settings_file == "another_settings.yml"

    def test_results_folder_argument(self):
        """Test specifying results folder."""
        argv = ["script_name", "-rf", "custom_results"]
        args = parse_command_line(argv)
        assert args.results_folder == "custom_results"

        argv = ["script_name", "--results_folder", "another_results"]
        args = parse_command_line(argv)
        assert args.results_folder == "another_results"

    def test_no_current_gens_flag(self):
        """Test the --no-current-gens flag."""
        argv = ["script_name", "--no-current-gens"]
        args = parse_command_line(argv)
        assert args.current_gens is False

    def test_no_gens_flag(self):
        """Test the --no-gens flag."""
        argv = ["script_name", "--no-gens"]
        args = parse_command_line(argv)
        assert args.gens is False

    def test_no_load_flag(self):
        """Test the --no-load flag."""
        argv = ["script_name", "--no-load"]
        args = parse_command_line(argv)
        assert args.load is False

    def test_no_transmission_flag(self):
        """Test the --no-transmission flag."""
        argv = ["script_name", "--no-transmission"]
        args = parse_command_line(argv)
        assert args.transmission is False

    def test_sort_gens_flag(self):
        """Test the --sort-gens flag."""
        argv = ["script_name", "-s"]
        args = parse_command_line(argv)
        assert args.sort_gens is True

        argv = ["script_name", "--sort-gens"]
        args = parse_command_line(argv)
        assert args.sort_gens is True

    def test_case_id_argument(self):
        """Test specifying case IDs."""
        argv = ["script_name", "-c", "case1", "case2", "case3"]
        args = parse_command_line(argv)
        assert args.case_id == ["case1", "case2", "case3"]

        argv = ["script_name", "--case-id", "single_case"]
        args = parse_command_line(argv)
        assert args.case_id == ["single_case"]

    def test_multi_period_flag(self):
        """Test the --multi-period flag."""
        argv = ["script_name", "-mp"]
        args = parse_command_line(argv)
        assert args.multi_period is False

        argv = ["script_name", "--multi-period"]
        args = parse_command_line(argv)
        assert args.multi_period is False

    def test_combined_arguments(self):
        """Test parsing multiple arguments together."""
        argv = [
            "script_name",
            "-sf",
            "test_settings.yml",
            "-rf",
            "test_results",
            "--no-current-gens",
            "--no-load",
            "-s",
            "-c",
            "case1",
            "case2",
            "-mp",
        ]
        args = parse_command_line(argv)

        assert args.settings_file == "test_settings.yml"
        assert args.results_folder == "test_results"
        assert args.current_gens is False
        assert args.gens is True
        assert args.load is False
        assert args.transmission is True
        assert args.sort_gens is True
        assert args.case_id == ["case1", "case2"]
        assert args.multi_period is False


class TestMainFunction:
    """Test the main function execution flow."""

    @patch("powergenome.run_powergenome.sys.argv", ["script_name"])
    @patch("powergenome.run_powergenome.Settings")
    @patch("powergenome.run_powergenome.initialize_data_manager")
    @patch("powergenome.run_powergenome.build_scenario_settings")
    def test_main_basic_setup(
        self,
        mock_build_scenario,
        mock_init_dm,
        mock_settings_class,
        test_settings_path,
        test_scenario_definitions_path,
        tmp_path,
    ):
        """Test basic main function setup and initialization."""
        mock_settings = MagicMock()
        mock_settings.__getitem__.side_effect = lambda key: {
            "data_location": "test_data.db",
            "input_folder": str(test_scenario_definitions_path.parent),
            "scenario_definitions_fn": test_scenario_definitions_path.name,
            "model_year": [2030],
            "model_first_planning_year": [2025],
        }.get(key)
        mock_settings.get.side_effect = lambda key, default=None: {
            "data_location": "test_data.db",
            "input_folder": str(test_scenario_definitions_path.parent),
            "scenario_definitions_fn": test_scenario_definitions_path.name,
            "model_year": [2030],
            "model_first_planning_year": [2025],
        }.get(key, default)
        mock_settings_class.return_value = mock_settings
        mock_build_scenario.return_value = {}

        with patch("pandas.read_csv") as mock_read_csv:
            real_df = pd.read_csv(test_scenario_definitions_path)
            mock_read_csv.return_value = real_df

            with patch("pathlib.Path.cwd", return_value=tmp_path):
                with patch("shutil.copy"):
                    main(
                        settings_file=str(test_settings_path),
                        results_folder="test_results",
                    )

        # Verify initialization calls
        mock_settings_class.assert_called_once_with(config_path=str(test_settings_path))
        mock_init_dm.assert_called_once_with(mock_settings, "test_data.db")

    @patch("powergenome.run_powergenome.sys.argv", ["script_name"])
    @patch("powergenome.run_powergenome.Settings")
    @patch("powergenome.run_powergenome.initialize_data_manager")
    def test_main_missing_case_ids_error(
        self,
        mock_init_dm,
        mock_settings_class,
        test_scenario_definitions_path,
        tmp_path,
    ):
        """Test error handling for missing case IDs."""
        mock_settings = MagicMock()
        mock_settings.__getitem__.side_effect = lambda key: {
            "data_location": "test_data.db",
            "input_folder": str(test_scenario_definitions_path.parent),
            "scenario_definitions_fn": test_scenario_definitions_path.name,
        }.get(key)
        mock_settings.get.side_effect = lambda key, default=None: {
            "data_location": "test_data.db",
            "input_folder": str(test_scenario_definitions_path.parent),
            "scenario_definitions_fn": test_scenario_definitions_path.name,
        }.get(key, default)
        mock_settings_class.return_value = mock_settings

        with patch("pandas.read_csv") as mock_read_csv:
            real_df = pd.read_csv(test_scenario_definitions_path)  # Only has "p1" case
            mock_read_csv.return_value = real_df

            with patch("pathlib.Path.cwd", return_value=tmp_path):
                with patch("shutil.copytree"):
                    with patch("shutil.copy"):
                        with pytest.raises(
                            ValueError,
                            match="The requested case IDs.*are not in your scenario",
                        ):
                            main(case_id=["missing_case"])

    @patch("powergenome.run_powergenome.sys.argv", ["script_name"])
    @patch("powergenome.run_powergenome.Settings")
    @patch("powergenome.run_powergenome.initialize_data_manager")
    def test_main_year_length_assertion(
        self, mock_init_dm, mock_settings_class, tmp_path
    ):
        """Test assertion for mismatched model_year and model_first_planning_year lengths."""
        mock_settings = MagicMock()
        mock_settings.__getitem__.side_effect = lambda key: {
            "data_location": "test_data.db",
            "input_folder": "inputs",
            "scenario_definitions_fn": "scenario_definitions.csv",
            "model_year": [2030, 2035],  # Length 2
            "model_first_planning_year": [2030],  # Length 1 - mismatch!
        }.get(key)
        mock_settings.get.side_effect = lambda key, default=None: {
            "data_location": "test_data.db",
            "input_folder": "inputs",
            "scenario_definitions_fn": "scenario_definitions.csv",
            "model_year": [2030, 2035],
            "model_first_planning_year": [2030],
        }.get(key, default)
        mock_settings_class.return_value = mock_settings

        with patch("pandas.read_csv") as mock_read_csv:
            mock_df = pd.DataFrame({"case_id": ["case1"], "year": [2030]})
            mock_read_csv.return_value = mock_df

            with patch("pathlib.Path.cwd", return_value=tmp_path):
                with patch("shutil.copytree"):
                    with patch("shutil.copy"):
                        with pytest.raises(AssertionError, match="must be the same"):
                            main()

    @patch("powergenome.run_powergenome.sys.argv", ["script_name"])
    @patch("powergenome.run_powergenome.Settings")
    @patch("powergenome.run_powergenome.initialize_data_manager")
    @patch("powergenome.run_powergenome.build_scenario_settings")
    def test_main_case_id_filtering(
        self, mock_build_scenario, mock_init_dm, mock_settings_class, tmp_path
    ):
        """Test case ID filtering functionality."""
        mock_settings = MagicMock()
        mock_settings.__getitem__.side_effect = lambda key: {
            "data_location": "test_data.db",
            "input_folder": "inputs",
            "scenario_definitions_fn": "scenario_definitions.csv",
            "model_year": [2030],
            "model_first_planning_year": [2025],
        }.get(key)
        mock_settings.get.side_effect = lambda key, default=None: {
            "data_location": "test_data.db",
            "input_folder": "inputs",
            "scenario_definitions_fn": "scenario_definitions.csv",
            "model_year": [2030],
            "model_first_planning_year": [2025],
        }.get(key, default)
        mock_settings_class.return_value = mock_settings
        mock_build_scenario.return_value = {}

        # Mock scenario definitions with multiple cases
        with patch("pandas.read_csv") as mock_read_csv:
            extended_df = pd.DataFrame(
                {
                    "case_id": ["p1", "case1", "case2", "case3", "case4"],
                    "year": [2030, 2030, 2030, 2030, 2030],
                    "time_series": ["full", "full", "full", "full", "full"],
                }
            )
            mock_read_csv.return_value = extended_df

            with patch("pathlib.Path.cwd", return_value=tmp_path):
                with patch("shutil.copytree"):
                    with patch("shutil.copy"):
                        main(case_id=["case1", "case3"])

        # Verify build_scenario_settings was called with filtered data
        call_args = mock_build_scenario.call_args[0]
        filtered_df = call_args[1]
        assert set(filtered_df["case_id"]) == {"case1", "case3"}

    @patch("powergenome.run_powergenome.sys.argv", ["script_name"])
    @patch("powergenome.run_powergenome.Settings")
    @patch("powergenome.run_powergenome.initialize_data_manager")
    @patch("powergenome.run_powergenome.build_scenario_settings")
    def test_main_multi_period_flag(
        self, mock_build_scenario, mock_init_dm, mock_settings_class, tmp_path
    ):
        """Test multi-period flag functionality."""
        mock_settings = MagicMock()
        mock_settings.__getitem__.side_effect = lambda key: {
            "data_location": "test_data.db",
            "input_folder": "inputs",
            "scenario_definitions_fn": "scenario_definitions.csv",
            "model_year": [2030],
            "model_first_planning_year": [2025],
        }.get(key)
        mock_settings.get.side_effect = lambda key, default=None: {
            "data_location": "test_data.db",
            "input_folder": "inputs",
            "scenario_definitions_fn": "scenario_definitions.csv",
            "model_year": [2030],
            "model_first_planning_year": [2025],
        }.get(key, default)
        mock_settings_class.return_value = mock_settings
        mock_build_scenario.return_value = {}

        with patch("pandas.read_csv") as mock_read_csv:
            mock_df = pd.DataFrame({"case_id": ["case1"], "year": [2030]})
            mock_read_csv.return_value = mock_df

            with patch("pathlib.Path.cwd", return_value=tmp_path):
                with patch("shutil.copytree"):
                    with patch("shutil.copy"):
                        with patch.object(logger, "info") as mock_info:
                            # Test with multi_period=False (should log message)
                            main(multi_period=False)

                            # Check that multi-period info message WAS logged
                            info_calls = [
                                call
                                for call in mock_info.call_args_list
                                if "version 0.6.2" in str(call)
                            ]
                            assert len(info_calls) == 1

    @patch("powergenome.run_powergenome.sys.argv", ["script_name"])
    @patch("powergenome.run_powergenome.Settings")
    @patch("powergenome.run_powergenome.initialize_data_manager")
    @patch("powergenome.run_powergenome.build_scenario_settings")
    def test_main_no_scenario_definitions_fn(
        self, mock_build_scenario, mock_init_dm, mock_settings_class, tmp_path
    ):
        """Test that when scenario_definitions_fn is absent, a synthetic
        scenario DataFrame is built from model_year and passed to build_scenario_settings.
        """
        mock_settings = MagicMock()
        settings_data = {
            "data_location": "test_data.db",
            "input_folder": "inputs",
            "model_year": [2030, 2040],
            "model_first_planning_year": [2025, 2035],
        }
        mock_settings.__getitem__.side_effect = lambda key: settings_data[key]
        mock_settings.get.side_effect = lambda key, default=None: settings_data.get(
            key, default
        )
        mock_settings_class.return_value = mock_settings
        mock_build_scenario.return_value = {}

        with patch("pathlib.Path.cwd", return_value=tmp_path):
            with patch("shutil.copytree"):
                with patch("shutil.copy"):
                    main(settings_file="some_settings_folder")

        # build_scenario_settings should be called with a synthetic DataFrame
        call_args = mock_build_scenario.call_args[0]
        synthetic_df = call_args[1]
        assert list(synthetic_df["case_id"]) == ["Inputs", "Inputs"]
        assert list(synthetic_df["year"]) == [2030, 2040]

    @patch("powergenome.run_powergenome.sys.argv", ["script_name"])
    @patch("powergenome.run_powergenome.Settings")
    @patch("powergenome.run_powergenome.initialize_data_manager")
    @patch("powergenome.run_powergenome.build_scenario_settings")
    def test_main_no_scenario_single_year(
        self, mock_build_scenario, mock_init_dm, mock_settings_class, tmp_path
    ):
        """Test that scalar model_year (not a list) also works without scenario file."""
        mock_settings = MagicMock()
        settings_data = {
            "data_location": "test_data.db",
            "input_folder": "inputs",
            "model_year": 2030,
            "model_first_planning_year": 2025,
        }
        mock_settings.__getitem__.side_effect = lambda key: settings_data[key]
        mock_settings.get.side_effect = lambda key, default=None: settings_data.get(
            key, default
        )
        mock_settings_class.return_value = mock_settings
        mock_build_scenario.return_value = {}

        with patch("pathlib.Path.cwd", return_value=tmp_path):
            with patch("shutil.copytree"):
                with patch("shutil.copy"):
                    main(settings_file="some_settings_folder")

        call_args = mock_build_scenario.call_args[0]
        synthetic_df = call_args[1]
        assert list(synthetic_df["case_id"]) == ["Inputs"]
        assert list(synthetic_df["year"]) == [2030]

    @patch("powergenome.run_powergenome.sys.argv", ["script_name"])
    @patch("powergenome.run_powergenome.Settings")
    @patch("powergenome.run_powergenome.initialize_data_manager")
    @patch("powergenome.run_powergenome.build_scenario_settings")
    def test_main_no_scenario_case_id_warning(
        self, mock_build_scenario, mock_init_dm, mock_settings_class, tmp_path, caplog
    ):
        """Test that using --case-id without scenario_definitions_fn emits a warning."""
        mock_settings = MagicMock()
        settings_data = {
            "data_location": "test_data.db",
            "input_folder": "inputs",
            "model_year": [2030],
            "model_first_planning_year": [2025],
        }
        mock_settings.__getitem__.side_effect = lambda key: settings_data[key]
        mock_settings.get.side_effect = lambda key, default=None: settings_data.get(
            key, default
        )
        mock_settings_class.return_value = mock_settings
        mock_build_scenario.return_value = {}

        with patch("pathlib.Path.cwd", return_value=tmp_path):
            with patch("shutil.copytree"):
                with patch("shutil.copy"):
                    with caplog.at_level(logging.WARNING, logger="powergenome"):
                        main(settings_file="some_settings_folder", case_id=["p1"])

        assert any(
            "--case-id flag is ignored" in record.message for record in caplog.records
        )

    def test_main_kwargs_override(self):
        """Test that kwargs properly override command line arguments."""
        with patch("powergenome.run_powergenome.parse_command_line") as mock_parse:
            mock_args = argparse.Namespace(
                settings_file="default.yml",
                results_folder="default_results",
                current_gens=True,
                gens=True,
                load=True,
                transmission=True,
                sort_gens=False,
                case_id=None,
                multi_period=True,
            )
            mock_parse.return_value = mock_args

            with patch.multiple(
                "powergenome.run_powergenome",
                Settings=Mock(),
                initialize_data_manager=Mock(),
                build_scenario_settings=Mock(return_value={}),
            ):
                with patch(
                    "pandas.read_csv",
                    return_value=pd.DataFrame({"case_id": [], "year": []}),
                ):
                    with patch("pathlib.Path.cwd"):
                        with patch("pathlib.Path.mkdir"):
                            try:
                                main(
                                    settings_file="override.yml",
                                    results_folder="override_results",
                                )
                            except Exception:
                                pass  # Expected to fail due to mocking

            # Verify that kwargs override the parsed arguments
            assert mock_args.settings_file == "override.yml"
            assert mock_args.results_folder == "override_results"
