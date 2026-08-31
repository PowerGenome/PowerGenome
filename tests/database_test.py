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
    get_timeseries_data,
    get_unique_values,
    initialize_data_manager,
    list_tables,
    table_info,
    update_data_manager,
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
        "policies": pd.DataFrame(
            {"case_id": ["base"], "year": [2030], "region": ["A"]}
        ),
        "segments": pd.DataFrame({"Voll": [1000], "Demand_segment": [1]}),
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

    def test_initialization_policy_and_demand_segment_tables(self, temp_csv_folder):
        dm = DataManager()
        dm.initialize(
            {
                "emission_policies_table": {"table_name": "policies.csv"},
                "demand_segments_table": {"table_name": "segments.csv"},
            },
            temp_csv_folder,
        )

        assert {"emission_policies", "demand_segments"} <= dm.available_tables
        assert dm.get_data("emission_policies").iloc[0]["case_id"] == "base"
        assert dm.get_data("demand_segments").iloc[0]["Voll"] == 1000

    def test_initialization_policy_tables_legacy_fn_alias(self, temp_csv_folder):
        """Legacy ``*_fn`` settings keys still configure the same tables."""
        dm = DataManager()
        dm.initialize(
            {
                "emission_policies_fn": {"table_name": "policies.csv"},
                "demand_segments_fn": {"table_name": "segments.csv"},
            },
            temp_csv_folder,
        )

        assert {"emission_policies", "demand_segments"} <= dm.available_tables
        assert dm.get_data("emission_policies").iloc[0]["case_id"] == "base"
        assert dm.get_data("demand_segments").iloc[0]["Voll"] == 1000

    def test_initialization_policy_tables_table_key_takes_precedence(
        self, temp_csv_folder
    ):
        """When both ``*_table`` and legacy ``*_fn`` are set, ``*_table`` wins."""
        dm = DataManager()
        dm.initialize(
            {
                "emission_policies_table": "policies.csv",
                "emission_policies_fn": "does_not_exist.csv",
            },
            temp_csv_folder,
        )

        assert "emission_policies" in dm.available_tables
        assert dm.get_data("emission_policies").iloc[0]["case_id"] == "base"

    def test_initialization_multiple_locations(
        self, sample_settings_csv, temp_csv_folder, tmp_path
    ):
        """Test that tables can be loaded from multiple data locations."""
        second_folder = tmp_path / "second"
        second_folder.mkdir()
        (temp_csv_folder / "load_data.csv").rename(second_folder / "load_data.csv")

        dm = DataManager()
        dm.initialize(sample_settings_csv, [temp_csv_folder, second_folder])

        assert dm.get_data("generation").shape[0] == 3
        assert dm.get_data("demand").shape[0] == 4

    def test_duplicate_table_in_multiple_locations_raises(
        self, sample_settings_csv, temp_csv_folder, tmp_path
    ):
        """Test that duplicate table names are rejected."""
        second_folder = tmp_path / "second"
        second_folder.mkdir()
        (second_folder / "generators.csv").write_bytes(
            (temp_csv_folder / "generators.csv").read_bytes()
        )

        with pytest.raises(RuntimeError, match="multiple data locations"):
            DataManager().initialize(
                {"generation_table": sample_settings_csv["generation_table"]},
                [temp_csv_folder, second_folder],
            )

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

    def test_scenario_filter_applied_csv(self, temp_csv_folder):
        """Scenario in config should be converted into a filter and applied."""
        # Prepare a load_data.csv with a scenario column
        df = pd.DataFrame(
            {
                "region": ["A", "A", "B", "B"],
                "year": [2030, 2030, 2030, 2030],
                "scenario": ["HighEV", "Base", "HighEV", "Base"],
                "load": [1000, 900, 1500, 1400],
            }
        )
        (temp_csv_folder / "load_data.csv").write_text(df.to_csv(index=False))

        settings = {
            "demand_table": {"table_name": "load_data.csv", "scenario": "HighEV"}
        }

        dm = DataManager()
        dm.initialize(settings, temp_csv_folder)

        out = dm.get_data("demand")
        # Only HighEV rows should be present
        assert set(out["scenario"]) == {"HighEV"}
        assert len(out) == 2

    def test_scenario_augments_existing_filters(self, temp_csv_folder):
        """Scenario should be ANDed into each OR-clause when filters exist."""
        df = pd.DataFrame(
            {
                "region": ["A", "A", "B", "B", "C"],
                "year": [2030, 2030, 2030, 2030, 2030],
                "scenario": ["Base", "HighEV", "Base", "HighEV", "Base"],
                "load": [1, 2, 3, 4, 5],
            }
        )
        (temp_csv_folder / "load_data.csv").write_text(df.to_csv(index=False))

        settings = {
            "demand_table": {
                "table_name": "load_data.csv",
                "scenario": "Base",
                # region=A OR region=B
                "filters": [[["region", "=", "A"]], [["region", "=", "B"]]],
            }
        }

        dm = DataManager()
        dm.initialize(settings, temp_csv_folder)

        out = dm.get_data("demand")
        # Expect only Base rows for regions A and B -> rows with loads 1 and 3
        assert set(out["region"]) <= {"A", "B"}
        assert set(out["scenario"]) == {"Base"}
        assert sorted(out["load"].tolist()) == [1, 3]

    def test_scenario_not_duplicated_if_present(self, temp_csv_folder):
        """If scenario is already in a clause, it shouldn't be duplicated."""
        df = pd.DataFrame(
            {
                "region": ["A", "A"],
                "year": [2030, 2030],
                "scenario": ["HighEV", "Base"],
                "load": [10, 20],
            }
        )
        (temp_csv_folder / "load_data.csv").write_text(df.to_csv(index=False))

        settings = {
            "demand_table": {
                "table_name": "load_data.csv",
                "scenario": "HighEV",
                # Clause already has scenario
                "filters": [[("scenario", "=", "HighEV")]],
            }
        }

        dm = DataManager()
        dm.initialize(settings, temp_csv_folder)

        out = dm.get_data("demand")
        assert len(out) == 1
        assert out.iloc[0]["scenario"] == "HighEV"

    def test_get_timeseries_data_aggregation(self, temp_csv_folder):
        """Validate grouping/sum over time_index across regions/sectors."""
        df = pd.DataFrame(
            {
                "time_index": [0, 0, 1, 1],
                "region": ["A", "B", "A", "B"],
                "sector": ["res", "res", "res", "res"],
                "value": [1.0, 2.0, 3.0, 4.0],
            }
        )
        (temp_csv_folder / "demand.csv").write_text(df.to_csv(index=False))

        settings = {"demand_table": "demand.csv"}
        dm = DataManager()
        dm.initialize(settings, temp_csv_folder)

        # Sum across regions, grouped by time_index
        out = dm.get_timeseries_data(
            "demand", group_by=["time_index"], value_col="value", agg="sum"
        )
        # Expect time_index 0 -> 3.0, 1 -> 7.0
        assert list(out.columns) == ["time_index", "value"]
        assert out.loc[out["time_index"] == 0, "value"].iloc[0] == 3.0
        assert out.loc[out["time_index"] == 1, "value"].iloc[0] == 7.0

    def test_get_timeseries_data_with_filters_list_shapes(self, temp_csv_folder):
        """List-shaped filters should work with GROUP BY path too."""
        df = pd.DataFrame(
            {
                "time_index": [0, 0, 1, 1],
                "region": ["A", "B", "A", "B"],
                "value": [10, 20, 30, 40],
            }
        )
        (temp_csv_folder / "ts.csv").write_text(df.to_csv(index=False))

        dm = DataManager()
        dm.initialize({"demand_table": "ts.csv"}, temp_csv_folder)

        # Filter region = 'A' using list form and group by time_index
        filters = [
            ["region", "=", "A"],
        ]
        out = dm.get_timeseries_data(
            "demand", group_by=["time_index"], value_col="value", filters=filters
        )
        assert list(out["value"]) == [10, 30]

    def test_global_wrapper_get_timeseries_data(self, temp_csv_folder):
        df = pd.DataFrame(
            {
                "time_index": [0, 0, 1, 1],
                "region": ["A", "B", "A", "B"],
                "value": [5, 7, 11, 13],
            }
        )
        (temp_csv_folder / "ts2.csv").write_text(df.to_csv(index=False))
        initialize_data_manager({"demand_table": "ts2.csv"}, temp_csv_folder)
        out = get_timeseries_data("demand", group_by=["time_index"], value_col="value")
        assert list(out["value"]) == [12, 24]

    def test_get_timeseries_data_wide_pivot(self, temp_csv_folder):
        df = pd.DataFrame(
            {
                "time_index": [0, 0, 1, 1],
                "region": ["A", "B", "A", "B"],
                "value": [1.0, 2.0, 3.0, 4.0],
            }
        )
        (temp_csv_folder / "ts_wide.csv").write_text(df.to_csv(index=False))

        dm = DataManager()
        dm.initialize({"demand_table": "ts_wide.csv"}, temp_csv_folder)

        # Aggregate by time_index and region, then pivot wide on region
        tall = dm.get_timeseries_data(
            "demand",
            group_by=["time_index", "region"],
            value_col="value",
            agg="sum",
        )
        assert set(tall.columns) == {"time_index", "region", "value"}

        wide = dm.get_timeseries_data(
            "demand",
            group_by=["time_index", "region"],
            value_col="value",
            agg="sum",
            wide=True,
            pivot_columns="region",
        )
        # Expect columns time_index, A, B
        assert "time_index" in wide.columns
        assert "A" in wide.columns and "B" in wide.columns
        # Values per time_index
        assert wide.loc[wide["time_index"] == 0, "A"].iloc[0] == 1.0
        assert wide.loc[wide["time_index"] == 0, "B"].iloc[0] == 2.0
        assert wide.loc[wide["time_index"] == 1, "A"].iloc[0] == 3.0
        assert wide.loc[wide["time_index"] == 1, "B"].iloc[0] == 4.0

    def test_get_timeseries_data_wide_with_pivot_index_and_fill(self, temp_csv_folder):
        df = pd.DataFrame(
            {
                "time_index": [0, 0, 1],
                "region": ["A", "B", "A"],
                "sector": ["res", "res", "res"],
                "value": [10, 20, 30],
            }
        )
        (temp_csv_folder / "ts_wide2.csv").write_text(df.to_csv(index=False))

        dm = DataManager()
        dm.initialize({"demand_table": "ts_wide2.csv"}, temp_csv_folder)

        wide = dm.get_timeseries_data(
            "demand",
            group_by=["time_index", "sector", "region"],
            value_col="value",
            agg="sum",
            wide=True,
            pivot_columns=["region"],
            pivot_index=["time_index", "sector"],
            fill_value=0,
        )
        # Expect A column present and B column present, with fill applied for missing combos
        assert set(["A", "B"]).issubset(set(wide.columns))
        # time_index=1 has only region A -> B should be 0 due to fill_value
        row = wide.loc[wide["time_index"] == 1].iloc[0]
        assert row["A"] == 30
        assert row["B"] == 0

    def test_get_timeseries_data_wide_multi_pivot(self, temp_csv_folder):
        df = pd.DataFrame(
            {
                "time_index": [0, 0, 0, 0, 1, 1, 1, 1],
                "region": ["A", "A", "B", "B", "A", "A", "B", "B"],
                "sector": ["res", "com", "res", "com", "res", "com", "res", "com"],
                "value": [1, 2, 3, 4, 10, 20, 30, 40],
            }
        )
        (temp_csv_folder / "ts_wide_multi.csv").write_text(df.to_csv(index=False))

        dm = DataManager()
        dm.initialize({"demand_table": "ts_wide_multi.csv"}, temp_csv_folder)

        wide = dm.get_timeseries_data(
            "demand",
            group_by=["time_index", "region", "sector"],
            value_col="value",
            agg="sum",
            wide=True,
            pivot_columns=["region", "sector"],
            pivot_index=["time_index"],
        )
        # Expect flattened multi-pivot columns like A_res, A_com, B_res, B_com
        expected_cols = {"time_index", "A_res", "A_com", "B_res", "B_com"}
        assert expected_cols.issubset(set(wide.columns))
        # Check values for time_index 0
        row0 = wide.loc[wide["time_index"] == 0].iloc[0]
        assert row0["A_res"] == 1
        assert row0["A_com"] == 2
        assert row0["B_res"] == 3
        assert row0["B_com"] == 4
        # Check values for time_index 1
        row1 = wide.loc[wide["time_index"] == 1].iloc[0]
        assert row1["A_res"] == 10
        assert row1["A_com"] == 20
        assert row1["B_res"] == 30
        assert row1["B_com"] == 40


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
        with caplog.at_level(logging.INFO):  # Changed from WARNING to INFO
            # Should auto-detect and succeed
            dm.initialize(settings, temp_csv_folder)

        # Check that warnings were logged about missing extension
        warning_messages = [
            record.message
            for record in caplog.records
            if record.levelno >= logging.WARNING
        ]
        assert any("does not have a file extension" in msg for msg in warning_messages)

        # Check that info messages were logged about auto-detection
        info_messages = [
            record.message
            for record in caplog.records
            if record.levelno == logging.INFO
        ]
        assert any("Auto-detected" in msg for msg in info_messages)

        # Tables should be created successfully via auto-detection
        assert len(dm.available_tables) > 0

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

    def test_database_table_with_extension_not_found(self, sample_data):
        """Test error when a file with extension is not found next to the database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_file:
            db_path = temp_file.name

        try:
            conn = sqlite3.connect(db_path)
            sample_data["generators"].to_sql(
                "generators", conn, index=False, if_exists="replace"
            )
            conn.close()

            settings = {
                "generation_table": "generators.csv"  # file doesn't exist next to the DB
            }

            dm = DataManager()
            # Expect RuntimeError wrapping ValueError about file not found
            with pytest.raises(RuntimeError, match="Failed to create table generation"):
                dm.initialize(settings, Path(db_path))

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

    def test_csv_file_colocated_with_database(self, sample_data):
        """Test loading a CSV file co-located in the same directory as the database."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            db_path = tmp_path / "data.db"

            # Create a SQLite database
            conn = sqlite3.connect(str(db_path))
            sample_data["generators"].to_sql(
                "generators", conn, index=False, if_exists="replace"
            )
            conn.close()

            # Write a CSV file next to the database
            csv_path = tmp_path / "supplemental_demand.csv"
            supp_df = pd.DataFrame(
                {
                    "region": ["R1", "R1"],
                    "time_index": [1, 2],
                    "load_mw": [100.0, 200.0],
                }
            )
            supp_df.to_csv(csv_path, index=False)

            settings = {
                "generation_table": "generators",
                "supplemental_demand_table": "supplemental_demand.csv",
            }

            dm = DataManager()
            dm.initialize(settings, db_path)

            # Both tables should be available
            assert "generation" in dm.available_tables
            assert "supplemental_demand" in dm.available_tables

            # Verify the CSV data is accessible
            result = dm.get_data("supplemental_demand")
            assert len(result) == 2
            assert list(result["region"]) == ["R1", "R1"]

    def test_dict_config_validation_warning(self, temp_csv_folder, caplog):
        """Test validation with dictionary configuration."""
        settings = {
            "generation_table": {
                "table_name": "generators",  # Missing .csv extension
                "columns": ["plant_id", "technology"],
            }
        }

        dm = DataManager()
        with caplog.at_level(logging.INFO):  # Changed from WARNING to INFO
            # Should auto-detect and succeed
            dm.initialize(settings, temp_csv_folder)

        # Check that warnings were logged about missing extension
        warning_messages = [
            record.message
            for record in caplog.records
            if record.levelno >= logging.WARNING
        ]
        assert any("does not have a file extension" in msg for msg in warning_messages)

        # Tables should be created successfully via auto-detection
        assert len(dm.available_tables) > 0

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

            # Test without extension (should warn but auto-detect)
            settings_bad = {"generation_table": "generators"}
            dm2 = DataManager()
            with caplog.at_level(logging.INFO):  # Changed from WARNING to INFO
                dm2.initialize(settings_bad, temp_path)

            warning_messages = [
                record.message
                for record in caplog.records
                if record.levelno >= logging.WARNING
            ]
            assert any(
                "does not have a file extension" in msg for msg in warning_messages
            )

            # Should auto-detect parquet file
            info_messages = [
                record.message
                for record in caplog.records
                if record.levelno == logging.INFO
            ]
            assert any("Auto-detected Parquet" in msg for msg in info_messages)
            assert "generation" in dm2.available_tables

    def test_unsupported_file_extension_warning(self, temp_csv_folder, caplog):
        """Test that unsupported file extensions are handled gracefully."""
        # Create a .txt file
        txt_file = temp_csv_folder / "data.txt"
        txt_file.write_text("some,data\n1,test")

        settings = {"generation_table": "data.txt"}  # Unsupported extension

        dm = DataManager()
        with caplog.at_level(logging.WARNING):
            # Should fail because .txt is explicitly unsupported
            with pytest.raises(RuntimeError, match="Failed to create table generation"):
                dm.initialize(settings, temp_csv_folder)

        # Should NOT get auto-detection because .txt has an extension (just unsupported)
        assert "generation" not in dm.available_tables

    def test_validation_with_mixed_file_types(self, sample_data, caplog):
        """Test validation with mixed file types (some correct, some incorrect)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create files
            csv_file = temp_path / "generators.csv"
            sample_data["generators"].to_csv(csv_file, index=False)

            plant_regions_file = temp_path / "plant_regions.csv"
            sample_data["plant_regions"].to_csv(plant_regions_file, index=False)

            fuel_prices_file = temp_path / "fuel_prices.csv"
            sample_data["fuel_prices"].to_csv(fuel_prices_file, index=False)

            settings = {
                "generation_table": "generators.csv",  # Correct
                "plant_region_table": "plant_regions",  # Missing extension - should auto-detect
                "fuel_price_table": "fuel_prices.db",  # Wrong extension for folder
            }

            dm = DataManager()
            with caplog.at_level(logging.WARNING):
                # Should fail on fuel_prices.db (has extension but wrong type)
                with pytest.raises(
                    RuntimeError, match="Failed to create table fuel_price"
                ):
                    dm.initialize(settings, temp_path)

            warning_messages = [
                record.message
                for record in caplog.records
                if record.levelno >= logging.WARNING
            ]

            # Should warn about missing extension for plant_regions
            assert any(
                "plant_regions" in msg and "does not have a file extension" in msg
                for msg in warning_messages
            )

            # 2 tables should be created successfully (generation and plant_region)
            # using auto-detection
            assert len(dm.available_tables) == 2


class TestDataManagerUpdate:
    """Test suite for DataManager update functionality."""

    def test_update_table_configuration_string_to_string(
        self, sample_data, temp_csv_folder
    ):
        """Test updating a table configuration from one string to another."""
        # Create two different CSV files
        sample_data["generators"].to_csv(
            temp_csv_folder / "generators_v1.csv", index=False
        )
        sample_data["plant_regions"].to_csv(
            temp_csv_folder / "generators_v2.csv", index=False
        )

        # Initial setup
        initial_settings = {"generation_table": "generators_v1.csv"}
        dm = DataManager()
        dm.initialize(initial_settings, temp_csv_folder)

        # Verify initial state
        assert "generation" in dm.available_tables
        initial_data = dm.get_data("generation")
        assert "technology" in initial_data.columns

        # Update to different file
        updated_settings = {"generation_table": "generators_v2.csv"}
        dm.update(updated_settings)

        # Verify update
        assert "generation" in dm.available_tables
        updated_data = dm.get_data("generation")
        assert "state" in updated_data.columns  # plant_regions has state column
        assert "technology" not in updated_data.columns

    def test_update_table_configuration_string_to_dict(
        self, sample_data, temp_csv_folder
    ):
        """Test updating a table configuration from string to dictionary."""
        sample_data["generators"].to_csv(
            temp_csv_folder / "generators.csv", index=False
        )

        # Initial setup with string config
        initial_settings = {"generation_table": "generators.csv"}
        dm = DataManager()
        dm.initialize(initial_settings, temp_csv_folder)

        initial_data = dm.get_data("generation")
        assert len(initial_data) == 3

        # Update to dictionary config with filters
        updated_settings = {
            "generation_table": {
                "table_name": "generators.csv",
                "filters": [[("technology", "=", "solar")]],
                "columns": ["plant_id", "technology"],
            }
        }
        dm.update(updated_settings)

        # Verify filtered data
        updated_data = dm.get_data("generation")
        assert len(updated_data) == 1
        assert list(updated_data.columns) == ["plant_id", "technology"]
        assert updated_data.iloc[0]["technology"] == "solar"

    def test_update_add_new_table(self, sample_data, temp_csv_folder):
        """Test adding a new table via update."""
        sample_data["generators"].to_csv(
            temp_csv_folder / "generators.csv", index=False
        )
        sample_data["plant_regions"].to_csv(
            temp_csv_folder / "plant_regions.csv", index=False
        )

        # Initial setup with one table
        initial_settings = {"generation_table": "generators.csv"}
        dm = DataManager()
        dm.initialize(initial_settings, temp_csv_folder)

        assert len(dm.available_tables) == 1
        assert "generation" in dm.available_tables

        # Add new table via update
        updated_settings = {
            "generation_table": "generators.csv",
            "plant_region_table": "plant_regions.csv",
        }
        dm.update(updated_settings)

        # Verify both tables are available
        assert len(dm.available_tables) == 2
        assert "generation" in dm.available_tables
        assert "plant_region" in dm.available_tables

        # Verify data is accessible
        plant_data = dm.get_data("plant_region")
        assert len(plant_data) == 3
        assert "state" in plant_data.columns

    def test_update_no_changes(self, sample_data, temp_csv_folder):
        """Test update when no changes are made."""
        sample_data["generators"].to_csv(
            temp_csv_folder / "generators.csv", index=False
        )

        initial_settings = {"generation_table": "generators.csv"}
        dm = DataManager()
        dm.initialize(initial_settings, temp_csv_folder)

        initial_tables = dm.available_tables.copy()
        initial_configs = dm.table_configurations.copy()

        # Update with same settings
        dm.update(initial_settings)

        # Verify nothing changed
        assert dm.available_tables == initial_tables
        assert dm.table_configurations == initial_configs

    def test_update_without_settings_parameter(self, sample_data, temp_csv_folder):
        """Test update without providing updated_settings parameter."""
        sample_data["generators"].to_csv(
            temp_csv_folder / "generators.csv", index=False
        )
        sample_data["plant_regions"].to_csv(
            temp_csv_folder / "plant_regions.csv", index=False
        )

        initial_settings = {"generation_table": "generators.csv"}
        dm = DataManager()
        dm.initialize(initial_settings, temp_csv_folder)

        # Manually modify settings
        dm.settings["plant_region_table"] = "plant_regions.csv"

        # Update without parameters (should use current settings)
        dm.update()

        # Verify new table was added
        assert len(dm.available_tables) == 2
        assert "plant_region" in dm.available_tables

    def test_update_with_invalid_table_config(
        self, sample_data, temp_csv_folder, caplog
    ):
        """Test update with invalid table configuration."""
        sample_data["generators"].to_csv(
            temp_csv_folder / "generators.csv", index=False
        )

        initial_settings = {"generation_table": "generators.csv"}
        dm = DataManager()
        dm.initialize(initial_settings, temp_csv_folder)

        # Update with invalid config (nonexistent file)
        updated_settings = {"generation_table": "nonexistent.csv"}

        with caplog.at_level(logging.ERROR):
            dm.update(updated_settings)

        # Verify error was logged and table was removed
        error_messages = [
            record.message
            for record in caplog.records
            if record.levelno >= logging.ERROR
        ]
        assert any("Failed to update table generation" in msg for msg in error_messages)
        assert "generation" not in dm.available_tables

    def test_update_mixed_changes(self, sample_data, temp_csv_folder):
        """Test update with mixed changes (modify, add, remove)."""
        # Create test files
        sample_data["generators"].to_csv(
            temp_csv_folder / "generators_v1.csv", index=False
        )
        sample_data["plant_regions"].to_csv(
            temp_csv_folder / "generators_v2.csv", index=False
        )
        sample_data["plant_regions"].to_csv(
            temp_csv_folder / "plant_regions.csv", index=False
        )
        sample_data["fuel_prices"].to_csv(
            temp_csv_folder / "fuel_prices.csv", index=False
        )

        # Initial setup
        initial_settings = {
            "generation_table": "generators_v1.csv",
            "plant_region_table": "plant_regions.csv",
            "fuel_price_table": "fuel_prices.csv",
        }
        dm = DataManager()
        dm.initialize(initial_settings, temp_csv_folder)

        assert len(dm.available_tables) == 3

        # Mixed update: modify generation, remove fuel_price, add demand
        sample_data["load_data"].to_csv(temp_csv_folder / "load_data.csv", index=False)
        updated_settings = {
            "generation_table": "generators_v2.csv",  # Changed
            "plant_region_table": "plant_regions.csv",  # Unchanged
            "demand_table": "load_data.csv",  # New
            # fuel_price_table unchanged
        }
        dm.update(updated_settings)

        # Verify results
        assert len(dm.available_tables) == 4
        assert "generation" in dm.available_tables
        assert "plant_region" in dm.available_tables
        assert "demand" in dm.available_tables
        assert "fuel_price" in dm.available_tables

        # Verify modified table has new data structure
        gen_data = dm.get_data("generation")
        assert "state" in gen_data.columns  # Should now have plant_regions structure

    def test_update_lazy_vs_materialized(self, sample_data, temp_csv_folder):
        """Test update works with both lazy and materialized loading."""
        sample_data["generators"].to_csv(
            temp_csv_folder / "generators_v1.csv", index=False
        )
        sample_data["plant_regions"].to_csv(
            temp_csv_folder / "generators_v2.csv", index=False
        )

        # Test with lazy loading
        dm_lazy = DataManager()
        dm_lazy.initialize(
            {"generation_table": "generators_v1.csv"},
            temp_csv_folder,
            lazy_loading=True,
        )
        dm_lazy.update({"generation_table": "generators_v2.csv"})

        lazy_data = dm_lazy.get_data("generation")
        assert "state" in lazy_data.columns

        # Test with materialized loading
        dm_mat = DataManager()
        dm_mat.initialize(
            {"generation_table": "generators_v1.csv"},
            temp_csv_folder,
            lazy_loading=False,
        )
        dm_mat.update({"generation_table": "generators_v2.csv"})

        mat_data = dm_mat.get_data("generation")
        assert "state" in mat_data.columns

        # Data should be equivalent
        pd.testing.assert_frame_equal(lazy_data, mat_data)

    def test_update_with_database_file(self, sample_data, temp_sqlite_db):
        """Test update functionality with database files."""
        # Remove CSV-specific table and create different tables in DB
        settings_v1 = {"generation_table": "generators"}
        settings_v2 = {"generation_table": "plant_regions"}  # Different table

        dm = DataManager()
        dm.initialize(settings_v1, temp_sqlite_db)

        initial_data = dm.get_data("generation")
        assert "technology" in initial_data.columns

        # Update to different table
        dm.update(settings_v2)

        updated_data = dm.get_data("generation")
        assert "state" in updated_data.columns
        assert "technology" not in updated_data.columns

    def test_update_error_handling_uninitialized(self):
        """Test error handling when updating uninitialized DataManager."""
        dm = DataManager()
        dm.connection = None  # Force uninitialized state

        with pytest.raises(ValueError, match="DataManager not initialized"):
            dm.update({"generation_table": "test.csv"})

    def test_update_preserves_table_configurations(self, sample_data, temp_csv_folder):
        """Test that update properly maintains table_configurations."""
        sample_data["generators"].to_csv(
            temp_csv_folder / "generators.csv", index=False
        )
        sample_data["plant_regions"].to_csv(
            temp_csv_folder / "plant_regions.csv", index=False
        )

        initial_settings = {"generation_table": "generators.csv"}
        dm = DataManager()
        dm.initialize(initial_settings, temp_csv_folder)

        # Verify initial configuration is stored
        assert "generation" in dm.table_configurations
        assert dm.table_configurations["generation"]["config"] == "generators.csv"
        assert (
            dm.table_configurations["generation"]["setting_key"] == "generation_table"
        )

        # Update
        updated_settings = {
            "generation_table": "plant_regions.csv",
            "plant_region_table": "plant_regions.csv",
        }
        dm.update(updated_settings)

        # Verify configurations are properly updated
        assert dm.table_configurations["generation"]["config"] == "plant_regions.csv"
        assert "plant_region" in dm.table_configurations
        assert dm.table_configurations["plant_region"]["config"] == "plant_regions.csv"


class TestGlobalUpdateFunction:
    """Test suite for the global update_data_manager function."""

    def test_global_update_function(self, sample_data, temp_csv_folder):
        """Test the global update_data_manager function."""
        sample_data["generators"].to_csv(
            temp_csv_folder / "generators.csv", index=False
        )
        sample_data["plant_regions"].to_csv(
            temp_csv_folder / "plant_regions.csv", index=False
        )

        # Initialize using global function
        initial_settings = {"generation_table": "generators.csv"}
        initialize_data_manager(initial_settings, temp_csv_folder)

        assert len(list_tables()) == 1
        assert "generation" in list_tables()

        # Update using global function
        updated_settings = {
            "generation_table": "generators.csv",
            "plant_region_table": "plant_regions.csv",
        }
        update_data_manager(updated_settings)

        # Verify update worked
        assert len(list_tables()) == 2
        assert "plant_region" in list_tables()

        plant_data = get_data("plant_region")
        assert len(plant_data) == 3

    def test_global_update_function_no_settings(self, sample_data, temp_csv_folder):
        """Test global update function without providing settings."""
        sample_data["generators"].to_csv(
            temp_csv_folder / "generators.csv", index=False
        )
        sample_data["plant_regions"].to_csv(
            temp_csv_folder / "plant_regions.csv", index=False
        )

        initial_settings = {"generation_table": "generators.csv"}
        initialize_data_manager(initial_settings, temp_csv_folder)

        # Manually update global manager's settings
        _data_manager.settings["plant_region_table"] = "plant_regions.csv"

        # Update without parameters
        update_data_manager()

        # Verify new table was added
        assert "plant_region" in list_tables()


@pytest.fixture(autouse=True)
def cleanup_data_manager():
    """Ensure DataManager is properly cleaned up after each test."""
    yield
    try:
        _data_manager.close()
    except:
        pass
