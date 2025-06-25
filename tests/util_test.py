"""
Test util functions
"""

import csv
import logging
import os
import sqlite3
from collections.abc import Iterable
from pathlib import Path

import duckdb
import pandas as pd
import pytest

import powergenome
from powergenome.util import (
    add_model_tags_to_gen_columns,
    add_row_to_csv,
    apply_all_tag_to_regions,
    assign_model_planning_years,
    get_all_table_names,
    hash_string_sha256,
    load_data,
    load_data_file,
    load_table_from_db,
    make_iterable,
    prepend_db_to_tables,
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


@pytest.fixture
def tmp_sqlite_db(tmp_path):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE foo (a INTEGER, b TEXT)")
    conn.execute("INSERT INTO foo VALUES (1,'x'), (2,'y')")
    conn.commit()
    conn.close()
    return str(db_path)


@pytest.fixture
def tmp_duckdb(tmp_path):
    db_path = tmp_path / "test.duckdb"
    con = duckdb.connect(database=str(db_path))
    con.execute("CREATE TABLE bar(c FLOAT);")
    con.execute("INSERT INTO bar VALUES (3.14), (2.71);")
    con.close()
    return str(db_path)


def test_get_all_table_names_duckdb(tmp_duckdb):
    # should list "bar"
    tables = get_all_table_names(tmp_duckdb)
    assert "bar" in tables


def test_prepend_db_to_tables_simple():
    q = "SELECT * FROM foo JOIN baz ON foo.id = baz.fk"
    out = prepend_db_to_tables(q, table_names=["foo", "baz"], db_prefix="db.")
    # every standalone foo/baz should be prefixed, but not foo.id
    assert "db.foo JOIN db.baz" in out
    assert "db.foo.id" not in out


def test_load_data_file_csv(tmp_path):
    p = tmp_path / "data.csv"
    df_orig = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
    df_orig.to_csv(p, index=False)
    df = load_data_file(str(p))
    pd.testing.assert_frame_equal(df, df_orig)


def test_load_data_file_parquet(tmp_path):
    p = tmp_path / "data.parquet"
    df_orig = pd.DataFrame({"m": [3.5, 4.5], "n": [True, False]})
    df_orig.to_parquet(p)
    df = load_data_file(str(p))
    pd.testing.assert_frame_equal(df, df_orig)


@pytest.mark.parametrize(
    "db_fixture,table_name",
    [
        ("tmp_sqlite_db", "foo"),
        ("tmp_duckdb", "bar"),
    ],
)
def test_load_table_from_db(request, tmp_path, db_fixture, table_name):
    db = request.getfixturevalue(db_fixture)
    df = load_table_from_db(db, file_or_table_name=table_name)
    # check it has rows
    assert len(df) > 0
    # columns match known names
    assert table_name in df.columns or len(df.columns) > 0


def test_load_data_folder_mode(tmp_path):
    # create folder with a CSV
    d = tmp_path / "folder"
    d.mkdir()
    df_orig = pd.DataFrame({"u": [10, 20]})
    (d / "u.csv").write_text(df_orig.to_csv(index=False))
    df = load_data(str(d), file_or_table_name="u.csv")
    pd.testing.assert_frame_equal(df.reset_index(drop=True), df_orig)


def test_load_data_file_with_query(tmp_path):
    # Create a temporary CSV file
    file_path = tmp_path / "test.csv"
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    df.to_csv(file_path, index=False)

    # Query to select only rows where a > 1
    query = "SELECT * WHERE a > 1"
    result = load_data_file(file_path, query=query)

    # Check that the result is as expected
    assert isinstance(result, pd.DataFrame)
    assert list(result["a"]) == [2, 3]
    assert list(result["b"]) == ["y", "z"]


def test_load_data_db_mode_sqlite(tmp_path, tmp_sqlite_db):
    # no extension on file_or_table_name
    df = load_data(tmp_sqlite_db, file_or_table_name="foo")
    assert "a" in df.columns and "b" in df.columns


def test_load_data_db_mode_duckdb(tmp_path, tmp_duckdb):
    # no extension on file_or_table_name
    df = load_data(tmp_duckdb, file_or_table_name="bar")
    assert "c" in df.columns


def test_load_data_query(tmp_sqlite_db):
    # tmp_sqlite_db has table foo with rows (1,'x'), (2,'y')
    # Run a SQL query instead of passing a table name
    df = load_data(tmp_sqlite_db, query="SELECT a, b FROM foo WHERE a = 2")
    # Build expected
    expected = pd.DataFrame({"a": [2], "b": ["y"]})
    # Compare
    pd.testing.assert_frame_equal(df.reset_index(drop=True), expected)


def test_load_data_file_unsupported_extension(tmp_path):
    # .txt is not supported by load_data_file
    p = tmp_path / "data.txt"
    p.write_text("just some text")
    with pytest.raises(ValueError, match=r"Unsupported file type"):
        load_data_file(str(p))


def test_load_data_no_params(tmp_path):
    # Neither file_or_table_name nor query provided
    with pytest.raises(ValueError, match=r"Either file_or_table_name or query"):
        load_data(str(tmp_path))


def test_load_data_folder_no_filename(tmp_path):
    # Loading from a folder without specifying file_or_table_name
    d = tmp_path / "empty_folder"
    d.mkdir()
    with pytest.raises(
        ValueError, match=r"file_or_table_name or query must be provided"
    ):
        load_data(str(d))


def test_load_data_db_with_extension(tmp_sqlite_db):
    # Passing a file name with extension when data_location is a DB
    with pytest.raises(ValueError, match=r"should not have an extension"):
        load_data(tmp_sqlite_db, file_or_table_name="foo.csv")


def test_load_table_from_db_unsupported_type(tmp_path):
    # Using an unsupported database type
    fake_db = tmp_path / "not_a_db.txt"
    fake_db.write_text("dummy")
    with pytest.raises(ValueError, match=r"Unsupported database type"):
        load_table_from_db(str(fake_db), file_or_table_name="foo")
