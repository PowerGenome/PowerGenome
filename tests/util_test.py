"""
Test util functions
"""

import csv
import sqlite3
from collections.abc import Iterable

import duckdb
import pandas as pd
import pytest

from powergenome.util import (
    add_row_to_csv,
    build_where_clause_from_filters,
    get_all_table_names,
    hash_string_sha256,
    load_data,
    load_data_file,
    load_table_from_db,
    make_iterable,
    prepend_db_to_tables,
    sort_nested_dict,
    update_dictionary,
)


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


class TestBuildWhereClauseFromFilters:
    def test_empty_filters(self):
        assert build_where_clause_from_filters(None) is None
        assert build_where_clause_from_filters([]) is None

    def test_single_conjunction(self):
        filters = [[("col1", "=", "val1")]]
        expected = "WHERE (col1 = 'val1')"
        assert build_where_clause_from_filters(filters) == expected

    def test_multiple_conjunctions_and(self):
        filters = [[("col1", "=", "val1"), ("col2", ">", 5)]]
        expected = "WHERE (col1 = 'val1' AND col2 > 5)"
        assert build_where_clause_from_filters(filters) == expected

    def test_multiple_disjunctions_or(self):
        filters = [[("col1", "=", "val1")], [("col2", ">", 5)]]
        expected = "WHERE (col1 = 'val1') OR (col2 > 5)"
        assert build_where_clause_from_filters(filters) == expected

    def test_string_value_formatting(self):
        filters = [[("col1", "=", "string value")]]
        expected = "WHERE (col1 = 'string value')"
        assert build_where_clause_from_filters(filters) == expected

    def test_numeric_value_formatting(self):
        filters = [[("col1", "<", 10.5)]]
        expected = "WHERE (col1 < 10.5)"
        assert build_where_clause_from_filters(filters) == expected

    def test_in_clause_with_list(self):
        filters = [[("col1", "IN", [1, 2, 3])]]
        expected = "WHERE (col1 IN (1, 2, 3))"
        assert build_where_clause_from_filters(filters) == expected

    def test_in_clause_with_tuple_of_strings(self):
        filters = [[("col1", "IN", ("a", "b", "c"))]]
        expected = "WHERE (col1 IN ('a', 'b', 'c'))"
        assert build_where_clause_from_filters(filters) == expected

    def test_complex_dnf(self):
        filters = [
            [("col1", "=", "val1"), ("col2", ">", 5)],
            [("col3", "!=", "val3"), ("col4", "IN", [1, 2])],
        ]
        expected = (
            "WHERE (col1 = 'val1' AND col2 > 5) OR (col3 != 'val3' AND col4 IN (1, 2))"
        )
        assert build_where_clause_from_filters(filters) == expected


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


def test_load_data_file_with_filters(tmp_path):
    # Create a temporary CSV file
    file_path = tmp_path / "test.csv"
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    df.to_csv(file_path, index=False)

    # filters to select only rows where a > 1
    filters = [[("a", ">", 1)]]
    result = load_data_file(file_path, filters=filters)

    # Check that the result is as expected
    expected = pd.DataFrame({"a": [2, 3], "b": ["y", "z"]})
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_load_data_file_with_columns(tmp_path):
    # Create a temporary CSV file
    file_path = tmp_path / "test.csv"
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [True, False, True]})
    df.to_csv(file_path, index=False)

    # Load only columns 'a' and 'c'
    columns = ["a", "c"]
    result = load_data_file(file_path, columns=columns)

    # Check that the result is as expected
    expected = pd.DataFrame({"a": [1, 2, 3], "c": [True, False, True]})
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_load_data_db_mode_sqlite(tmp_path, tmp_sqlite_db):
    # no extension on file_or_table_name
    df = load_data(tmp_sqlite_db, file_or_table_name="foo")
    assert "a" in df.columns and "b" in df.columns


def test_load_data_db_mode_duckdb(tmp_path, tmp_duckdb):
    # no extension on file_or_table_name
    df = load_data(tmp_duckdb, file_or_table_name="bar")
    assert "c" in df.columns


def test_load_data_with_filters(tmp_sqlite_db):
    # tmp_sqlite_db has table foo with rows (1,'x'), (2,'y')
    # Run a SQL query instead of passing a table name
    filters = [[("a", "=", 2)]]
    df = load_data(tmp_sqlite_db, file_or_table_name="foo", filters=filters)
    # Build expected
    expected = pd.DataFrame({"a": [2], "b": ["y"]})
    # Compare
    pd.testing.assert_frame_equal(df, expected, check_dtype=False)


def test_load_data_with_filters_single_list(tmp_sqlite_db):
    # tmp_sqlite_db has table foo with rows (1,'x'), (2,'y')
    # Run a SQL query instead of passing a table name
    filters = [("a", "=", 2)]
    df = load_data(tmp_sqlite_db, file_or_table_name="foo", filters=filters)
    # Build expected
    expected = pd.DataFrame({"a": [2], "b": ["y"]})
    # Compare
    pd.testing.assert_frame_equal(df, expected, check_dtype=False)


def test_load_data_with_columns(tmp_sqlite_db):
    # tmp_sqlite_db has table foo with columns 'a' and 'b'
    # Load only column 'b'
    columns = ["b"]
    df = load_data(tmp_sqlite_db, file_or_table_name="foo", columns=columns)

    # Build expected
    expected = pd.DataFrame({"b": ["x", "y"]})
    # Compare
    pd.testing.assert_frame_equal(df, expected)


def test_load_data_file_unsupported_extension(tmp_path):
    # .txt is not supported by load_data_file
    p = tmp_path / "data.txt"
    p.write_text("just some text")
    with pytest.raises(ValueError, match=r"Unsupported file type"):
        load_data_file(str(p))


def test_load_data_no_params(tmp_path):
    # Neither file_or_table_name nor query provided
    with pytest.raises(ValueError, match=r"file_or_table_name must be provided"):
        load_data(str(tmp_path))


def test_load_data_folder_no_filename(tmp_path):
    # Loading from a folder without specifying file_or_table_name
    d = tmp_path / "empty_folder"
    d.mkdir()
    with pytest.raises(ValueError, match=r"file_or_table_name must be provided"):
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


def test_update_flat_dict():
    d = {"a": 1, "bb": 2}
    u = {"c": 3, "bb": 20}
    result = update_dictionary(d.copy(), u)
    # keys sorted by length: "a" (1), "c" (1), "bb" (2)
    assert list(result.keys()) == ["a", "c", "bb"]
    assert result["bb"] == 20
    assert result["c"] == 3


def test_update_nested_dict():
    d = {"x": {"y": 1}, "zz": 2}
    u = {"x": {"z": 3}}
    result = update_dictionary(d.copy(), u)
    # should merge into nested dict rather than overwrite completely
    assert isinstance(result["x"], dict)
    assert result["x"] == {"y": 1, "z": 3}
    assert "zz" in result


def test_update_none_initial():
    # Treat None as empty mapping
    result = update_dictionary(None, {"aa": 5})
    assert result == {"aa": 5}


def test_invalid_inputs_raise_type_error():
    with pytest.raises(TypeError):
        update_dictionary(123, {})
    with pytest.raises(TypeError):
        update_dictionary({}, 123)


def test_key_length_sorting_with_non_str_keys():
    d = {}
    u = {10: "ten", "x": "ex", (1, 2, 3): "tuple"}
    result = update_dictionary(d.copy(), u)
    # str(key) lengths: '10' -> 2, 'x' -> 1, '(1, 2, 3)' -> 9
    # sort by length: 'x'(1), '10'(2), '(1, 2, 3)'(9)
    keys = list(result.keys())
    assert keys[0] == "x"
    assert keys[1] == 10  # int key appears second
    assert keys[2] == (1, 2, 3)
    assert result[10] == "ten"
    assert result["x"] == "ex"
