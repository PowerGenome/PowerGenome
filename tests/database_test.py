import logging
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import duckdb
import pandas as pd
import pytest

from powergenome.database import (
    DataManager,
    _data_manager,
    execute_query,
    get_data,
    get_unique_values,
    initialize_data_manager,
    list_tables,
    table_info,
)


@pytest.fixture
def sample_settings_db():
    """Sample settings for testing."""
    return {
        "generation_table": "generators",
        "plant_region_table": "plant_regions",
        "fuel_price_table": "fuel_prices",
        "demand_table": "load_data",
    }


@pytest.fixture
def sample_settings_csv():
    """Sample settings for testing."""
    return {
        "generation_table": "generators.csv",
        "plant_region_table": "plant_regions.csv",
        "fuel_price_table": "fuel_prices.csv",
        "demand_table": "load_data.csv",
    }


@pytest.fixture
def sample_data():
    """Sample data for testing."""
    return {
        "generators": pd.DataFrame(
            {
                "plant_id": [1, 2, 3],
                "technology": ["solar", "wind", "gas"],
                "capacity": [100, 200, 300],
                "region": ["A", "B", "A"],
            }
        ),
        "plant_regions": pd.DataFrame(
            {
                "plant_id": [1, 2, 3],
                "region": ["A", "B", "A"],
                "state": ["CA", "TX", "CA"],
            }
        ),
        "fuel_prices": pd.DataFrame(
            {"fuel": ["gas", "coal"], "year": [2030, 2030], "price": [5.0, 3.0]}
        ),
        "load_data": pd.DataFrame(
            {
                "region": ["A", "B", "A", "B"],
                "year": [2030, 2030, 2031, 2031],
                "load": [1000, 1500, 1100, 1600],
            }
        ),
    }


@pytest.fixture
def temp_csv_folder(sample_data):
    """Create temporary folder with CSV files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create CSV files
        for table_name, df in sample_data.items():
            file_path = temp_path / f"{table_name}.csv"
            df.to_csv(file_path, index=False)

        yield temp_path


@pytest.fixture
def temp_sqlite_db(sample_data):
    """Create temporary SQLite database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_file:
        db_path = temp_file.name

    try:
        conn = sqlite3.connect(db_path)
        for table_name, df in sample_data.items():
            if table_name != "load_data":  # Skip CSV-specific data
                df.to_sql(table_name, conn, index=False, if_exists="replace")
        conn.close()

        yield Path(db_path)
    finally:
        Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def temp_duckdb_db(sample_data):
    """Create temporary DuckDB database."""
    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as temp_file:
        db_path = temp_file.name

    # Remove the empty file that was created
    Path(db_path).unlink()

    try:
        # Connect to create the database file properly
        conn = duckdb.connect(db_path)
        for table_name, df in sample_data.items():
            if table_name != "load_data":  # Skip CSV-specific data
                conn.register(f"{table_name}_temp", df)
                conn.execute(
                    f"CREATE TABLE {table_name} AS SELECT * FROM {table_name}_temp"
                )
                conn.unregister(f"{table_name}_temp")
        conn.close()

        yield Path(db_path)
    finally:
        Path(db_path).unlink(missing_ok=True)


class TestDataManager:
    """Tests for the DataManager singleton class."""

    def test_singleton_behavior(self):
        """Test that DataManager behaves as a singleton."""
        dm1 = DataManager()
        dm2 = DataManager()
        assert dm1 is dm2
        assert dm1 is _data_manager

    def test_initialization_csv_folder(self, sample_settings_csv, temp_csv_folder):
        """Test initialization with CSV folder."""
        dm = DataManager()
        dm.initialize(sample_settings_csv, temp_csv_folder)

        assert dm.data_location == temp_csv_folder
        assert dm.lazy_loading is True
        assert "generation" in dm.available_tables
        assert "plant_region" in dm.available_tables
        assert "demand" in dm.available_tables

    def test_initialization_sqlite_db(self, sample_settings_db, temp_sqlite_db):
        """Test initialization with SQLite database."""
        # Remove CSV-specific table from settings
        settings = sample_settings_db.copy()
        del settings["demand_table"]

        dm = DataManager()
        dm.initialize(settings, temp_sqlite_db)

        assert dm.data_location == temp_sqlite_db
        assert "generation" in dm.available_tables
        assert "plant_region" in dm.available_tables

    def test_initialization_duckdb_db(self, sample_settings_db, temp_duckdb_db):
        """Test initialization with DuckDB database."""
        # Remove CSV-specific table from settings
        settings = sample_settings_db.copy()
        del settings["demand_table"]

        dm = DataManager()
        dm.initialize(settings, temp_duckdb_db)

        assert dm.data_location == temp_duckdb_db
        assert "generation" in dm.available_tables
        assert "plant_region" in dm.available_tables

    def test_materialized_loading(self, sample_settings_csv, temp_csv_folder):
        """Test materialized table loading (lazy_loading=False)."""
        dm = DataManager()
        dm.initialize(sample_settings_csv, temp_csv_folder, lazy_loading=False)

        assert dm.lazy_loading is False
        assert "generation" in dm.available_tables

        # Should be able to query data
        data = dm.get_data("generation")
        assert len(data) == 3
        assert list(data.columns) == ["plant_id", "technology", "capacity", "region"]

    def test_get_data_basic(self, sample_settings_csv, temp_csv_folder):
        """Test basic data retrieval."""
        dm = DataManager()
        dm.initialize(sample_settings_csv, temp_csv_folder)

        data = dm.get_data("generation")
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 3
        assert "technology" in data.columns

    def test_get_data_with_columns(self, sample_settings_csv, temp_csv_folder):
        """Test data retrieval with specific columns."""
        dm = DataManager()
        dm.initialize(sample_settings_csv, temp_csv_folder)

        data = dm.get_data("generation", columns=["plant_id", "technology"])
        assert list(data.columns) == ["plant_id", "technology"]
        assert len(data) == 3

    def test_get_data_with_filters(self, sample_settings_csv, temp_csv_folder):
        """Test data retrieval with filters."""
        dm = DataManager()
        dm.initialize(sample_settings_csv, temp_csv_folder)

        # Single filter
        data = dm.get_data("generation", filters=[[("technology", "=", "solar")]])
        assert len(data) == 1
        assert data.iloc[0]["technology"] == "solar"

        # Multiple filters (OR condition)
        data = dm.get_data(
            "generation",
            filters=[[("technology", "=", "solar")], [("technology", "=", "wind")]],
        )
        assert len(data) == 2

    def test_get_data_with_custom_query(self, sample_settings_csv, temp_csv_folder):
        """Test data retrieval with custom SQL query."""
        dm = DataManager()
        dm.initialize(sample_settings_csv, temp_csv_folder)

        data = dm.get_data(
            "generation", query="SELECT * FROM generation WHERE capacity > 150"
        )
        assert len(data) == 2  # wind and gas plants
        assert all(data["capacity"] > 150)

    def test_get_unique_values(self, sample_settings_csv, temp_csv_folder):
        """Test getting unique values from a column."""
        dm = DataManager()
        dm.initialize(sample_settings_csv, temp_csv_folder)

        unique_techs = dm.get_unique_values("generation", "technology")
        assert set(unique_techs) == {"gas", "solar", "wind"}

        unique_regions = dm.get_unique_values("generation", "region")
        assert set(unique_regions) == {"A", "B"}

    def test_list_tables(self, sample_settings_csv, temp_csv_folder):
        """Test listing available tables."""
        dm = DataManager()
        dm.initialize(sample_settings_csv, temp_csv_folder)

        tables = dm.list_tables()
        assert isinstance(tables, list)
        assert "generation" in tables
        assert "plant_region" in tables
        assert "demand" in tables

    def test_table_info(self, sample_settings_csv, temp_csv_folder):
        """Test getting table schema information."""
        dm = DataManager()
        dm.initialize(sample_settings_csv, temp_csv_folder)

        info = dm.table_info("generation")
        assert isinstance(info, pd.DataFrame)
        assert "column_name" in info.columns
        assert "column_type" in info.columns

    def test_execute_query(self, sample_settings_csv, temp_csv_folder):
        """Test executing custom SQL queries."""
        dm = DataManager()
        dm.initialize(sample_settings_csv, temp_csv_folder)

        result = dm.execute_query("SELECT COUNT(*) as total FROM generation")
        assert result.iloc[0]["total"] == 3

        # Join query
        result = dm.execute_query(
            """
            SELECT g.plant_id, g.technology, pr.state
            FROM generation g
            JOIN plant_region pr ON g.plant_id = pr.plant_id
        """
        )
        assert len(result) == 3
        assert "state" in result.columns

    def test_error_handling_uninitialized(self):
        """Test error handling when DataManager is not initialized."""
        dm = DataManager()
        # Force uninitialized state
        dm.connection = None

        with pytest.raises(ValueError, match="DataManager not initialized"):
            dm.get_data("generation")

        with pytest.raises(ValueError, match="DataManager not initialized"):
            dm.get_unique_values("generation", "technology")

        with pytest.raises(ValueError, match="DataManager not initialized"):
            dm.table_info("generation")

        with pytest.raises(ValueError, match="DataManager not initialized"):
            dm.execute_query("SELECT * FROM generation")

    def test_error_handling_table_not_available(
        self, sample_settings_csv, temp_csv_folder
    ):
        """Test error handling when table is not available."""
        dm = DataManager()
        dm.initialize(sample_settings_csv, temp_csv_folder)

        with pytest.raises(ValueError, match="Table 'nonexistent' not available"):
            dm.get_data("nonexistent")

        with pytest.raises(ValueError, match="Table 'nonexistent' not available"):
            dm.get_unique_values("nonexistent", "column")

        with pytest.raises(ValueError, match="Table 'nonexistent' not available"):
            dm.table_info("nonexistent")

    def test_error_handling_invalid_query(self, sample_settings_csv, temp_csv_folder):
        """Test error handling for invalid SQL queries."""
        dm = DataManager()
        dm.initialize(sample_settings_csv, temp_csv_folder)

        with pytest.raises(RuntimeError, match="Query execution failed"):
            dm.get_data("generation", query="INVALID SQL SYNTAX")

    def test_dict_table_config(self, temp_csv_folder):
        """Test table configuration as dictionary."""
        settings = {
            "generation_table": {
                "table_name": "generators.csv",
                "filters": [[("technology", "=", "solar")]],
                "columns": ["plant_id", "technology"],
            }
        }

        dm = DataManager()
        dm.initialize(settings, temp_csv_folder)

        data = dm.get_data("generation")
        assert len(data) == 1
        assert list(data.columns) == ["plant_id", "technology"]
        assert data.iloc[0]["technology"] == "solar"

    def test_close_connection(self, sample_settings_csv, temp_csv_folder):
        """Test closing the database connection."""
        dm = DataManager()
        dm.initialize(sample_settings_csv, temp_csv_folder)

        assert dm.connection is not None
        dm.close()
        assert dm.connection is None
        assert len(dm.available_tables) == 0

    def test_reinitialize(self, sample_settings_csv, temp_csv_folder):
        """Test reinitializing DataManager with different settings."""
        dm = DataManager()
        dm.initialize(sample_settings_csv, temp_csv_folder)

        initial_tables = dm.list_tables()

        # Reinitialize with different settings
        new_settings = {"generation_table": "generators.csv"}
        dm.initialize(new_settings, temp_csv_folder)

        new_tables = dm.list_tables()
        assert len(new_tables) < len(initial_tables)
        assert "generation" in new_tables


class TestConvenienceFunctions:
    """Tests for the global convenience functions."""

    def test_initialize_data_manager(self, sample_settings_csv, temp_csv_folder):
        """Test the global initialize_data_manager function."""
        initialize_data_manager(sample_settings_csv, temp_csv_folder)

        assert _data_manager.data_location == temp_csv_folder
        assert "generation" in _data_manager.available_tables

    def test_get_data_global(self, sample_settings_csv, temp_csv_folder):
        """Test the global get_data function."""
        initialize_data_manager(sample_settings_csv, temp_csv_folder)

        data = get_data("generation")
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 3

    def test_get_unique_values_global(self, sample_settings_csv, temp_csv_folder):
        """Test the global get_unique_values function."""
        initialize_data_manager(sample_settings_csv, temp_csv_folder)

        unique_values = get_unique_values("generation", "technology")
        assert set(unique_values) == {"gas", "solar", "wind"}

    def test_list_tables_global(self, sample_settings_csv, temp_csv_folder):
        """Test the global list_tables function."""
        initialize_data_manager(sample_settings_csv, temp_csv_folder)

        tables = list_tables()
        assert isinstance(tables, list)
        assert "generation" in tables

    def test_table_info_global(self, sample_settings_csv, temp_csv_folder):
        """Test the global table_info function."""
        initialize_data_manager(sample_settings_csv, temp_csv_folder)

        info = table_info("generation")
        assert isinstance(info, pd.DataFrame)
        assert "column_name" in info.columns

    def test_execute_query_global(self, sample_settings_csv, temp_csv_folder):
        """Test the global execute_query function."""
        initialize_data_manager(sample_settings_csv, temp_csv_folder)

        result = execute_query("SELECT COUNT(*) as total FROM generation")
        assert result.iloc[0]["total"] == 3


class TestValidation:
    """Tests for the validation functionality in DataManager."""

    def test_csv_file_without_extension_warning(self, temp_csv_folder, caplog):
        """Test warning when CSV file names don't have extensions."""
        settings = {
            "generation_table": "generators",  # Missing .csv extension
            "plant_region_table": "plant_regions",  # Missing .csv extension
        }

        dm = DataManager()
        with caplog.at_level(logging.WARNING):
            dm.initialize(settings, temp_csv_folder)

        # Check that warnings were logged
        warning_messages = [
            record.message
            for record in caplog.records
            if record.levelno >= logging.WARNING
        ]
        assert any("does not have a file extension" in msg for msg in warning_messages)

        # Tables should still not be created since files don't exist
        assert len(dm.available_tables) == 0

    def test_csv_file_with_extension_no_warning(
        self, sample_settings_csv, temp_csv_folder, caplog
    ):
        """Test no warning when CSV file names have proper extensions."""
        dm = DataManager()
        with caplog.at_level(logging.WARNING):
            dm.initialize(sample_settings_csv, temp_csv_folder)

        # Check that no validation warnings were logged
        validation_warnings = [
            record.message
            for record in caplog.records
            if record.levelno >= logging.WARNING
            and "does not have a file extension" in record.message
        ]
        assert len(validation_warnings) == 0

        # Tables should be successfully created
        assert len(dm.available_tables) > 0

    def test_database_table_with_extension_warning(self, sample_data, caplog):
        """Test warning when database table names have file extensions."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_file:
            db_path = temp_file.name

        try:
            conn = sqlite3.connect(db_path)
            sample_data["generators"].to_sql(
                "generators", conn, index=False, if_exists="replace"
            )
            conn.close()

            settings = {
                "generation_table": "generators.csv"  # Table name with extension (wrong for DB)
            }

            dm = DataManager()
            with caplog.at_level(logging.WARNING):
                dm.initialize(settings, Path(db_path))

            # Check that warnings were logged
            warning_messages = [
                record.message
                for record in caplog.records
                if record.levelno >= logging.WARNING
            ]
            assert any(
                "appears to have a file extension" in msg for msg in warning_messages
            )

        finally:
            Path(db_path).unlink(missing_ok=True)

    def test_database_table_without_extension_no_warning(
        self, sample_settings_db, temp_sqlite_db, caplog
    ):
        """Test no warning when database table names don't have extensions."""
        # Remove CSV-specific table from settings
        settings = sample_settings_db.copy()
        del settings["demand_table"]

        dm = DataManager()
        with caplog.at_level(logging.WARNING):
            dm.initialize(settings, temp_sqlite_db)

        # Check that no validation warnings were logged
        validation_warnings = [
            record.message
            for record in caplog.records
            if record.levelno >= logging.WARNING
            and "appears to have a file extension" in record.message
        ]
        assert len(validation_warnings) == 0

        # Tables should be successfully created
        assert len(dm.available_tables) > 0

    def test_dict_config_validation_warning(self, temp_csv_folder, caplog):
        """Test validation with dictionary configuration."""
        settings = {
            "generation_table": {
                "table_name": "generators",  # Missing .csv extension
                "columns": ["plant_id", "technology"],
            }
        }

        dm = DataManager()
        with caplog.at_level(logging.WARNING):
            dm.initialize(settings, temp_csv_folder)

        # Check that warnings were logged
        warning_messages = [
            record.message
            for record in caplog.records
            if record.levelno >= logging.WARNING
        ]
        assert any("does not have a file extension" in msg for msg in warning_messages)

    def test_parquet_file_validation(self, sample_data, caplog):
        """Test validation works with .parquet files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create a parquet file
            parquet_file = temp_path / "generators.parquet"
            sample_data["generators"].to_parquet(parquet_file, index=False)

            # Test with proper .parquet extension
            settings_good = {"generation_table": "generators.parquet"}
            dm = DataManager()
            with caplog.at_level(logging.WARNING):
                dm.initialize(settings_good, temp_path)

            validation_warnings = [
                record.message
                for record in caplog.records
                if record.levelno >= logging.WARNING
                and "does not have a file extension" in record.message
            ]
            assert len(validation_warnings) == 0
            assert "generation" in dm.available_tables

            # Clear log records
            caplog.clear()

            # Test without extension (should warn)
            settings_bad = {"generation_table": "generators"}
            dm2 = DataManager()
            with caplog.at_level(logging.WARNING):
                dm2.initialize(settings_bad, temp_path)

            warning_messages = [
                record.message
                for record in caplog.records
                if record.levelno >= logging.WARNING
            ]
            assert any(
                "does not have a file extension" in msg for msg in warning_messages
            )

    def test_unsupported_file_extension_warning(self, temp_csv_folder, caplog):
        """Test that unsupported file extensions are handled gracefully."""
        # Create a .txt file
        txt_file = temp_csv_folder / "data.txt"
        txt_file.write_text("some,data\n1,test")

        settings = {"generation_table": "data.txt"}  # Unsupported extension

        dm = DataManager()
        with caplog.at_level(logging.WARNING):
            dm.initialize(settings, temp_csv_folder)

        # Should get a "Failed to create table" warning due to unsupported file type
        warning_messages = [
            record.message
            for record in caplog.records
            if record.levelno >= logging.WARNING
        ]
        assert any(
            "Failed to create table generation" in msg for msg in warning_messages
        )
        assert "generation" not in dm.available_tables

    def test_validation_with_mixed_file_types(self, sample_data, caplog):
        """Test validation with mixed file types (some correct, some incorrect)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create files
            csv_file = temp_path / "generators.csv"
            sample_data["generators"].to_csv(csv_file, index=False)

            settings = {
                "generation_table": "generators.csv",  # Correct
                "plant_region_table": "plant_regions",  # Missing extension
                "fuel_price_table": "fuel_prices.db",  # Wrong extension for folder
            }

            dm = DataManager()
            with caplog.at_level(logging.WARNING):
                dm.initialize(settings, temp_path)

            warning_messages = [
                record.message
                for record in caplog.records
                if record.levelno >= logging.WARNING
            ]

            # Should warn about missing extension and wrong extension
            extension_warnings = [
                msg
                for msg in warning_messages
                if "does not have a file extension" in msg
            ]
            assert len(extension_warnings) >= 1

            # Only the correctly named file should load
            assert "generation" in dm.available_tables
            assert len(dm.available_tables) == 1


@pytest.fixture(autouse=True)
def cleanup_data_manager():
    """Ensure DataManager is properly cleaned up after each test."""
    yield
    try:
        _data_manager.close()
    except:
        pass
