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
from powergenome.run_powergenome import main, parse_command_line, resolve_output_formats
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


@pytest.fixture(autouse=True)
def _patch_validation(monkeypatch):
    """Patch validation functions so CLI tests are not affected by mock Settings objects.

    cli_test.py tests the pipeline control flow using MagicMock settings objects.
    The validation module is tested separately in validate_test.py.
    """
    monkeypatch.setattr(
        "powergenome.run_powergenome.validate_settings",
        lambda s: [],
    )
    monkeypatch.setattr(
        "powergenome.run_powergenome.validate_settings_with_data",
        lambda s, dm: [],
    )
    monkeypatch.setattr(
        "powergenome.run_powergenome.report_validation_results",
        lambda results, raise_on_error=True: None,
    )


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
        assert args.macro is False  # Macro output is opt-in
        assert args.genx is False  # GenX is the default in main(); flag is explicit

    def test_macro_flag(self):
        """Test the --macro flag enables Macro output."""
        argv = ["script_name", "--macro"]
        args = parse_command_line(argv)
        assert args.macro is True
        assert args.genx is False

    def test_genx_flag(self):
        """Test the --genx flag explicitly enables GenX output."""
        argv = ["script_name", "--genx"]
        args = parse_command_line(argv)
        assert args.genx is True
        assert args.macro is False

    def test_both_output_flags(self):
        """Test --macro and --genx together write both output formats."""
        argv = ["script_name", "--macro", "--genx"]
        args = parse_command_line(argv)
        assert args.macro is True
        assert args.genx is True

    def test_output_flags_case_insensitive(self):
        """Test output flags are matched case-insensitively."""
        argv = ["script_name", "--MACRO", "--Genx"]
        args = parse_command_line(argv)
        assert args.macro is True
        assert args.genx is True

        argv = ["script_name", "--Macro", "--GENX"]
        args = parse_command_line(argv)
        assert args.macro is True
        assert args.genx is True

    def test_output_flags_case_insensitive_unknown_preserved(self):
        """Test only flags (not their values) are lowercased."""
        argv = ["script_name", "--MACRO", "--settings_file", "MyCaps.Settings.yml"]
        args = parse_command_line(argv)
        assert args.macro is True
        assert args.settings_file == "MyCaps.Settings.yml"

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


class TestResolveOutputFormats:
    """Test the macro/genx output-format selection logic."""

    def _settings(self, **values):
        return MagicMock(get=lambda key, default=None: values.get(key, default))

    def test_default_genx_only(self):
        macro, genx = resolve_output_formats(
            Mock(macro=False, genx=False), self._settings()
        )
        assert macro is False
        assert genx is True

    def test_macro_flag_adds_macro_keeps_genx(self):
        macro, genx = resolve_output_formats(
            Mock(macro=True, genx=False), self._settings()
        )
        assert macro is True
        assert genx is True

    def test_macro_setting_adds_macro_keeps_genx(self):
        macro, genx = resolve_output_formats(
            Mock(macro=False, genx=False), self._settings(macro_output=True)
        )
        assert macro is True
        assert genx is True

    def test_macro_only_settings(self):
        macro, genx = resolve_output_formats(
            Mock(macro=False, genx=False),
            self._settings(macro_output=True, genx_output=False),
        )
        assert macro is True
        assert genx is False

    def test_genx_flag_overrides_false_setting(self):
        macro, genx = resolve_output_formats(
            Mock(macro=False, genx=True),
            self._settings(genx_output=False),
        )
        assert macro is False
        assert genx is True

    def test_both_flags_both_formats(self):
        macro, genx = resolve_output_formats(
            Mock(macro=True, genx=True), self._settings()
        )
        assert macro is True
        assert genx is True

    def test_boolish_string_values_case_insensitive(self):
        # A settings file loaded with a string (non-YAML) value.
        macro, genx = resolve_output_formats(
            Mock(macro=False, genx=False),
            self._settings(macro_output="TRUE", genx_output="False"),
        )
        assert macro is True
        assert genx is False

    def test_namespace_without_flag_attrs(self):
        # Some callers build a Namespace without macro/genx attributes.
        args = Mock()
        del args.macro
        del args.genx
        macro, genx = resolve_output_formats(args, self._settings())
        assert macro is False
        assert genx is True


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
    @patch("powergenome.run_powergenome.write_case_settings_file")
    @patch("powergenome.run_powergenome.process_genx_data", return_value=[])
    @patch("powergenome.run_powergenome.update_data_manager")
    @patch("powergenome.run_powergenome.resolve_settings_to_year")
    @patch("powergenome.run_powergenome.build_scenario_settings")
    @patch("powergenome.run_powergenome.initialize_data_manager")
    @patch("powergenome.run_powergenome.Settings")
    def test_main_no_scenario_definitions_fn(
        self,
        mock_settings_class,
        mock_init_dm,
        mock_build_scenario,
        mock_resolve,
        mock_update_dm,
        _mock_genx,
        _mock_write_settings,
        tmp_path,
    ):
        """Test that when scenario_definitions_fn is absent, resolve_settings_to_year
        is called once per planning year and build_scenario_settings is not used.
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
        mock_resolve.return_value = {}

        # Make scenario_settings_obj.get() return falsy defaults to skip
        # optional pipeline sections (reserves_fn, emission_policies_fn, etc.)
        scenario_obj = MagicMock()
        scenario_obj.get.side_effect = lambda k, d=None: d
        mock_settings_class.for_scenario.return_value.__enter__.return_value = (
            scenario_obj
        )

        with patch("pathlib.Path.cwd", return_value=tmp_path):
            with patch("shutil.copytree"):
                with patch("shutil.copy"):
                    main(
                        settings_file="some_settings_folder",
                        gens=False,
                        load=False,
                        transmission=False,
                    )

        # build_scenario_settings must NOT be called in the no-scenario path
        mock_build_scenario.assert_not_called()
        # resolve_settings_to_year must be called once per planning year
        assert mock_resolve.call_count == 2
        call_years = sorted([c[0][1] for c in mock_resolve.call_args_list])
        assert call_years == [2030, 2040]

    @patch("powergenome.run_powergenome.sys.argv", ["script_name"])
    @patch("powergenome.run_powergenome.write_case_settings_file")
    @patch("powergenome.run_powergenome.process_genx_data", return_value=[])
    @patch("powergenome.run_powergenome.update_data_manager")
    @patch("powergenome.run_powergenome.resolve_settings_to_year")
    @patch("powergenome.run_powergenome.build_scenario_settings")
    @patch("powergenome.run_powergenome.initialize_data_manager")
    @patch("powergenome.run_powergenome.Settings")
    def test_main_no_scenario_single_year(
        self,
        mock_settings_class,
        mock_init_dm,
        mock_build_scenario,
        mock_resolve,
        mock_update_dm,
        _mock_genx,
        _mock_write_settings,
        tmp_path,
    ):
        """Test that a scalar model_year (not a list) also works without a scenario file."""
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
        mock_resolve.return_value = {}

        scenario_obj = MagicMock()
        scenario_obj.get.side_effect = lambda k, d=None: d
        mock_settings_class.for_scenario.return_value.__enter__.return_value = (
            scenario_obj
        )

        with patch("pathlib.Path.cwd", return_value=tmp_path):
            with patch("shutil.copytree"):
                with patch("shutil.copy"):
                    main(
                        settings_file="some_settings_folder",
                        gens=False,
                        load=False,
                        transmission=False,
                    )

        mock_build_scenario.assert_not_called()
        assert mock_resolve.call_count == 1
        assert mock_resolve.call_args[0][1] == 2030

    @patch("powergenome.run_powergenome.sys.argv", ["script_name"])
    @patch("powergenome.run_powergenome.write_case_settings_file")
    @patch("powergenome.run_powergenome.process_genx_data", return_value=[])
    @patch("powergenome.run_powergenome.update_data_manager")
    @patch("powergenome.run_powergenome.resolve_settings_to_year")
    @patch("powergenome.run_powergenome.build_scenario_settings")
    @patch("powergenome.run_powergenome.initialize_data_manager")
    @patch("powergenome.run_powergenome.Settings")
    def test_main_no_scenario_case_id_warning(
        self,
        mock_settings_class,
        mock_init_dm,
        mock_build_scenario,
        mock_resolve,
        mock_update_dm,
        _mock_genx,
        _mock_write_settings,
        tmp_path,
        caplog,
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
        mock_resolve.return_value = {}

        scenario_obj = MagicMock()
        scenario_obj.get.side_effect = lambda k, d=None: d
        mock_settings_class.for_scenario.return_value.__enter__.return_value = (
            scenario_obj
        )

        with patch("pathlib.Path.cwd", return_value=tmp_path):
            with patch("shutil.copytree"):
                with patch("shutil.copy"):
                    with caplog.at_level(logging.WARNING, logger="powergenome"):
                        main(
                            settings_file="some_settings_folder",
                            case_id=["p1"],
                            gens=False,
                            load=False,
                            transmission=False,
                        )

        assert any(
            "--case-id flag is ignored" in record.message for record in caplog.records
        )

    @patch("powergenome.run_powergenome.sys.argv", ["script_name"])
    @patch("powergenome.run_powergenome.write_case_settings_file")
    @patch("powergenome.run_powergenome.process_genx_data", return_value=[])
    @patch("powergenome.run_powergenome.update_data_manager")
    @patch("powergenome.run_powergenome.resolve_settings_to_year")
    @patch("powergenome.run_powergenome.build_scenario_settings")
    @patch("powergenome.run_powergenome.initialize_data_manager")
    @patch("powergenome.run_powergenome.Settings")
    def test_main_no_scenario_case_folder_path(
        self,
        mock_settings_class,
        mock_init_dm,
        mock_build_scenario,
        mock_resolve,
        mock_update_dm,
        _mock_genx,
        _mock_write_settings,
        tmp_path,
    ):
        """Test that without scenario_definitions_fn the case_folder path starts with
        'Inputs/Inputs_p{N}' and does NOT include a case_id subdirectory."""
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
        mock_resolve.return_value = {}

        scenario_obj = MagicMock()
        scenario_obj.get.side_effect = lambda k, d=None: d
        scenario_obj.__getitem__.side_effect = lambda k: (
            1 if k == "case_period" else MagicMock()
        )
        mock_settings_class.for_scenario.return_value.__enter__.return_value = (
            scenario_obj
        )

        out_folder = tmp_path / "results"
        with patch("pathlib.Path.cwd", return_value=tmp_path):
            with patch("shutil.copytree"):
                with patch("shutil.copy"):
                    main(
                        settings_file="some_settings_folder",
                        results_folder="results",
                        gens=False,
                        load=False,
                        transmission=False,
                    )

        # The case_folder must be directly under out_folder/Inputs/Inputs_p1,
        # i.e. there must NOT be a case_id subdirectory between out_folder and Inputs.
        expected_case_folder = out_folder / "Inputs" / "Inputs_p1"
        assert expected_case_folder.exists(), (
            f"Expected case_folder {expected_case_folder} to be created, "
            f"but it was not. Contents of {out_folder}: {list(out_folder.iterdir()) if out_folder.exists() else 'folder missing'}"
        )
        # Also assert there is no case_id subfolder at the top level
        assert not (
            out_folder / "Inputs" / "Inputs" / "Inputs_p1"
        ).exists(), "case_folder should not include a case_id subdirectory in the no-scenario path"

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
