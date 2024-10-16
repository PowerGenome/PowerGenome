"""
Test util functions
"""

import csv
import logging
from collections.abc import Iterable

import pytest

import powergenome
from powergenome.util import (
    add_model_tags_to_gen_columns,
    add_row_to_csv,
    apply_all_tag_to_regions,
    assign_model_planning_years,
    check_case_settings,
    hash_string_sha256,
    make_iterable,
    sort_nested_dict,
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


def test_sort_nested_dict():
    test_dict1 = {"one": 1, "threeee": 3, "twoo": 2}
    sorted_dict1 = sort_nested_dict(test_dict1)
    # Here, the keys should be ordered as 'one', 'twoo' and 'threeee'
    assert list(sorted_dict1.keys()) == ["one", "twoo", "threeee"]

    test_dict2 = {"threeee": {"fourrrrr": 4, "twoo": 2}, "one": 1, "fiveeeee": 5}
    sorted_dict2 = sort_nested_dict(test_dict2)
    # Here, the keys at the top level should be ordered as 'one', 'threeee' and 'fiveeeee'
    assert list(sorted_dict2.keys()) == ["one", "threeee", "fiveeeee"]
    # And within the dictionary mapped to by 'threeee', the keys should be 'twoo' and 'fourrrrr'
    assert list(sorted_dict2["threeee"].keys()) == ["twoo", "fourrrrr"]

    test_dict3 = {"a": {"b": 4, "a": 2}, "c": 1, "b": 5}
    sorted_dict3 = sort_nested_dict(test_dict3)
    assert list(sorted_dict3.keys()) == ["a", "c", "b"]
    assert list(sorted_dict3["a"].keys()) == ["b", "a"]


def test_apply_all_tag_to_regions(caplog):
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
    caplog.set_level(logging.WARNING)
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


class TestHashStringSha256:
    # Returns a hash string for a given input string
    def test_returns_hash_string(self):
        # Arrange
        input_string = "Hello, World!"
        expected_hash = (
            "dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f"
        )

        # Act
        actual_hash = hash_string_sha256(input_string)

        # Assert
        assert actual_hash == expected_hash

    # Raises TypeError if input is not a string
    def test_raises_type_error(self):
        # Arrange
        input_string = 123

        # Act & Assert
        with pytest.raises(TypeError):
            hash_string_sha256(input_string)


class TestAddRowToCsv:
    # Adds a new row to an existing CSV file with headers, ensuring correct file permissions
    def test_add_row_with_headers_fixed_fixed(self, tmp_path):
        # Create a temporary CSV file with headers in the temporary directory
        file = tmp_path / "test.csv"
        headers = ["Name", "Age", "City"]
        with file.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

        # Call the function to add a new row
        new_row = ["John", "25", "New York"]
        try:
            add_row_to_csv(file, new_row)
        except PermissionError:
            pytest.fail("PermissionError: Unable to open file in append mode")

        # Check if the new row is added to the CSV file
        with file.open("r") as f:
            reader = csv.reader(f)
            data = list(reader)
            assert new_row in data

        # Clean up the temporary CSV file and directory
        file.unlink()

    # Raises ValueError if the file does not exist and no headers were provided
    def test_raises_value_error_if_file_does_not_exist_and_no_headers_provided(
        self, tmp_path
    ):
        with pytest.raises(ValueError):
            file = tmp_path / "test.csv"
            new_row = ["John", "25", "New York"]
            add_row_to_csv(file, new_row)

    # Creates a new CSV file with headers and adds a new row
    def test_add_row_with_headers_and_new_row_fixed_fixed(self, tmp_path):
        # Create a temporary CSV file without headers
        file = tmp_path / "test.csv"
        headers = ["Name", "Age", "City"]

        # Call the function to add a new row
        new_row = ["John", "25", "New York"]
        try:
            add_row_to_csv(file, new_row, headers)
        except ValueError as e:
            pytest.fail(str(e))

        # Check if the new row is added to the CSV file
        with file.open("r") as f:
            reader = csv.reader(f)
            data = list(reader)
            assert new_row in data

        # Clean up the temporary CSV file
        file.unlink()


class TestMakeIterable:

    # Returns an iterable version of a list
    def test_returns_iterable_list(self):
        # Arrange
        item = [1, 2, 3]

        # Act
        result = make_iterable(item)

        # Assert
        assert isinstance(result, Iterable)
        assert list(result) == item

    # Returns an iterable version of an integer
    def test_returns_iterable_integer(self):
        # Arrange
        item = 5

        # Act
        result = make_iterable(item)

        # Assert
        assert isinstance(result, Iterable)
        assert list(result) == [item]

    # Returns an iterable version of a string
    def test_returns_iterable_string(self):
        # Arrange
        item = "hello"

        # Act
        result = make_iterable(item)

        # Assert
        assert isinstance(result, Iterable)
        assert list(result) == [item]

    # Returns an iterable version of an empty list
    def test_returns_iterable_empty_list(self):
        # Arrange
        item = []

        # Act
        result = make_iterable(item)

        # Assert
        assert isinstance(result, Iterable)
        assert list(result) == item


class TestAssignModelPlanningYears:

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
    def test_with_invalid_model_periods_length(self):
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


class TestAddModelTagsToGenColumns:

    # Returns the input 'generator_columns' list unmodified if it is not a list.
    def test_returns_input_unmodified_if_not_list(self):
        generator_columns = "not a list"
        model_tag_values = {}
        regional_tag_values = {}
        result = add_model_tags_to_gen_columns(
            model_tag_values, regional_tag_values, generator_columns
        )
        assert result == generator_columns

    # Adds model resource tag keys to the 'generator_columns' list if they are not already present.
    def test_adds_model_tags_to_gen_columns(self):
        generator_columns = ["capacity", "output"]
        model_tag_values = {"cost": {"solar": 100, "wind": 150}}
        regional_tag_values = {"NA": {"efficiency": {"solar": 20, "wind": 25}}}
        expected_result = ["capacity", "output", "cost", "efficiency"]

        result = add_model_tags_to_gen_columns(
            model_tag_values, regional_tag_values, generator_columns
        )

        assert sorted(result) == sorted(expected_result)


# Generated by Qodo Gen

# Dependencies:
# pip install pytest-mock
import pytest


class TestCheckCaseSettings:

    # Processes scenario_settings with multiple years and cases correctly
    def test_multiple_years_and_cases(self):
        scenario_settings = {
            "2020": {
                "case1": {
                    "renewables_clusters": [
                        {"region": "A", "technology": "solar", "group": "G1"}
                    ]
                },
                "case2": {
                    "renewables_clusters": [
                        {"region": "B", "technology": "wind", "group": "G2"}
                    ]
                },
            },
            "2021": {
                "case1": {
                    "renewables_clusters": [
                        {"region": "A", "technology": "solar", "group": "G1"}
                    ]
                },
                "case3": {
                    "renewables_clusters": [
                        {"region": "C", "technology": "hydro", "group": "G3"}
                    ]
                },
            },
        }
        check_case_settings(scenario_settings)

    # Handles scenario_settings with missing renewables_clusters key
    def test_missing_renewables_clusters_key(self):
        scenario_settings = {
            "2020": {
                "case1": {
                    "other_key": [{"region": "A", "technology": "solar", "group": "G1"}]
                },
                "case2": {},
            },
            "2021": {
                "case1": {
                    "renewables_clusters": [
                        {"region": "A", "technology": "solar", "group": "G1"}
                    ]
                },
                "case3": {
                    "other_key": [{"region": "C", "technology": "hydro", "group": "G3"}]
                },
            },
        }
        check_case_settings(scenario_settings)

    def test_logs_warnings_when_mismatch_detected(self, caplog):
        scenario_settings = {
            "2020": {
                "case1": {
                    "renewables_clusters": [
                        {"region": "A", "technology": "solar", "group": "G1"}
                    ]
                },
                "case2": {
                    "renewables_clusters": [
                        {"region": "B", "technology": "wind", "group": "G2"}
                    ]
                },
            },
            "2021": {
                "case1": {
                    "renewables_clusters": [
                        {"region": "A", "technology": "solar", "group": "G1"}
                    ]
                },
                "case2": {
                    "renewables_clusters": [
                        {"region": "B", "technology": "wind", "group": "G3"}
                    ]
                },
            },
        }
        caplog.set_level(logging.WARNING)
        check_case_settings(scenario_settings)
        assert (
            "Mismatch found for renewables_clusters in case case2 region B"
            in caplog.text
        )
