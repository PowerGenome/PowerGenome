import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import duckdb
import pandas as pd

from powergenome.util import (
    build_where_clause_from_filters,
    get_all_table_names,
    load_data,
    prepend_db_to_tables,
)

logger = logging.getLogger(__name__)


class DataManager:
    """
    Singleton class for managing data access across the PowerGenome codebase.

    Creates an in-memory DuckDB database with standardized table names based on
    settings parameters. Supports loading data from both database files and
    CSV/Parquet files.
    """

    _instance = None
    _initialized = False

    # Mapping of settings parameters to standardized table names
    STANDARD_TABLE_MAPPING = {
        "generation_table": "generation",
        "plant_region_table": "plant_region",
        "resource_heat_rate_table": "resource_heat_rate",
        "resource_cost_table": "resource_cost",
        "operational_constraints_table": "operational_constraints",
        "transmission_constraints_table": "transmission_constraints",
        "fuel_price_table": "fuel_price",
        "dollar_year_table": "dollar_year",
        "regional_cost_factor_table": "regional_cost_factor",
        "transmission_cost_table": "transmission_cost",
        "demand_table": "demand",
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.connection = None
            self.data_location = None
            self.available_tables = set()
            self._initialized = True
            self.sqlite_attached = False
            self.duckdb_attached = False

    def initialize(
        self,
        settings: Dict[str, Any],
        data_location: Union[Path, str] = None,
        lazy_loading: bool = True,
    ):
        """
        Initialize the DataManager with settings and data location.

        Parameters
        ----------
        settings : Dict[str, Any]
            Settings dictionary containing table configuration parameters
        data_location : Union[Path, str], optional
            Path to database file or folder containing data files
        lazy_loading : bool, optional
            If True, create views instead of loading all data into memory, by default True
        """
        if self.connection is not None:
            logger.info("DataManager already initialized. Closing existing connection.")
            self.connection.close()

        # Create in-memory DuckDB connection
        self.connection = duckdb.connect(database=":memory:")
        self.data_location = Path(data_location) if data_location else None
        self.settings = settings
        self.lazy_loading = lazy_loading

        # Setup tables based on settings
        self._setup_tables()

        logger.info(f"DataManager initialized with {len(self.available_tables)} tables")

    def _setup_tables(self):
        """Setup standardized tables/views based on settings parameters."""
        self.available_tables.clear()

        for setting_key, standard_name in self.STANDARD_TABLE_MAPPING.items():
            table_config = self.settings.get(setting_key)

            if not table_config:
                continue

            try:
                self._create_table_view(table_config, standard_name)
                self.available_tables.add(standard_name)
                logger.debug(f"Created table/view: {standard_name}")
            except Exception as e:
                logger.warning(f"Failed to create table {standard_name}: {e}")

    def _validate_table_config(
        self, table_config: Union[str, Dict], standard_name: str
    ) -> Tuple[str, List, List[str]]:
        """
        Validate and normalize table configuration.

        Parameters
        ----------
        table_config : Union[str, Dict]
            Either a string table name or dict with table configuration
        standard_name : str
            Standardized name for the table in the in-memory database

        Returns
        -------
        Tuple[str, List, List[str]]
            Normalized source table name, filters, and columns

        Raises
        ------
        ValueError
            If table configuration is invalid or file/table names don't match data location type
        """
        if isinstance(table_config, str):
            source_table = table_config
            filters = None
            columns = None
        elif isinstance(table_config, dict):
            source_table = table_config.get("table_name") or table_config.get("name")
            filters = table_config.get("filters")
            columns = table_config.get("columns")
        else:
            raise ValueError(f"Invalid table configuration: {table_config}")

        if not source_table:
            raise ValueError(f"No table name found in configuration: {table_config}")

        # Validate file/table name format based on data location type
        if self.data_location and self.data_location.is_dir():
            # For folder-based data, require file extension
            if not any(
                source_table.lower().endswith(ext) for ext in [".csv", ".parquet"]
            ):
                logger.warning(
                    f"Table '{standard_name}' source '{source_table}' does not have a "
                    "file extension (.csv or .parquet). This may cause loading issues."
                )
        elif self.data_location and self.data_location.is_file():
            # For database files, table names should not have extensions
            if any(
                source_table.lower().endswith(ext)
                for ext in [".csv", ".parquet", ".db", ".sqlite", ".duckdb"]
            ):
                logger.warning(
                    f"Table '{standard_name}' source '{source_table}' appears to have a "
                    "file extension, but data_location is a database file. "
                    "Table names in databases should not have extensions."
                )

        return source_table, filters, columns

    def _create_table_view(self, table_config: Union[str, Dict], standard_name: str):
        """
        Create a table or view in the in-memory database.

        Parameters
        ----------
        table_config : Union[str, Dict]
            Either a string table name or dict with table configuration
        standard_name : str
            Standardized name for the table in the in-memory database
        """
        source_table, filters, columns = self._validate_table_config(
            table_config, standard_name
        )

        if self.lazy_loading and self.data_location:
            # Create views that reference external files/databases instead of loading data
            self._create_lazy_view(source_table, standard_name, filters, columns)
        else:
            # Original behavior - load all data into memory
            self._create_materialized_table(
                source_table, standard_name, filters, columns
            )

    def _create_lazy_view(
        self,
        source_table: str,
        standard_name: str,
        filters: List = None,
        columns: List[str] = None,
    ):
        """Create a view that references external data without loading it into memory."""
        if self.data_location.is_dir():
            # For file-based data
            file_path = self.data_location / source_table
            file_extension = file_path.suffix.lower()

            if file_extension == ".csv":
                source_query = f"read_csv_auto('{file_path}')"
            elif file_extension == ".parquet":
                source_query = f"read_parquet('{file_path}')"
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")

        else:
            # For database files
            if str(self.data_location).endswith((".db", ".sqlite")):
                # Attach SQLite database and reference the table
                if self.sqlite_attached is False:
                    self.connection.execute(
                        f"ATTACH '{self.data_location}' AS source_db"
                    )
                    self.sqlite_attached = True

                source_query = f"source_db.{source_table}"
            elif str(self.data_location).endswith(".duckdb"):
                # For DuckDB files, also use ATTACH method
                if self.duckdb_attached is False:
                    self.connection.execute(
                        f"ATTACH '{self.data_location}' AS source_db"
                    )
                    self.duckdb_attached = True

                source_query = f"source_db.{source_table}"
            else:
                raise ValueError("Unsupported database type")

        # Build the view query
        select_cols = ", ".join(columns) if columns else "*"
        where_clause = build_where_clause_from_filters(filters) if filters else ""

        view_query = f"CREATE VIEW {standard_name} AS SELECT {select_cols} FROM {source_query} {where_clause}"
        self.connection.execute(view_query)

    def _create_materialized_table(
        self,
        source_table: str,
        standard_name: str,
        filters: List = None,
        columns: List[str] = None,
    ):
        """Create a materialized table by loading all data into memory (original behavior)."""
        if self.data_location:
            df = load_data(
                data_location=self.data_location,
                file_or_table_name=source_table,
                filters=filters,
                columns=columns,
            )
        else:
            df = load_data(
                data_location=source_table,
                file_or_table_name=None,
                filters=filters,
                columns=columns,
            )

        # Create table in in-memory database
        self.connection.register(f"{standard_name}_temp", df)
        self.connection.execute(
            f"CREATE TABLE {standard_name} AS SELECT * FROM {standard_name}_temp"
        )
        self.connection.unregister(f"{standard_name}_temp")

    def get_data(
        self,
        table_name: str,
        filters: List[List[Tuple[str, str, Any]]] = None,
        columns: List[str] = None,
        query: str = None,
    ) -> pd.DataFrame:
        """
        Get data from a standardized table.

        Parameters
        ----------
        table_name : str
            Name of the standardized table
        filters : List[List[Tuple[str, str, Any]]], optional
            Filters to apply in DNF format
        columns : List[str], optional
            Columns to select
        query : str, optional
            Custom SQL query (overrides other parameters)

        Returns
        -------
        pd.DataFrame
            Requested data

        Raises
        ------
        ValueError
            If DataManager not initialized or table not available
        RuntimeError
            If custom query fails
        """
        if self.connection is None:
            raise ValueError("DataManager not initialized. Call initialize() first.")

        if query:
            try:
                return self.connection.execute(query).fetchdf()
            except Exception as e:
                raise RuntimeError(f"Query execution failed: {e}")

        if table_name not in self.available_tables:
            raise ValueError(
                f"Table '{table_name}' not available. Available tables: {self.available_tables}"
            )

        # Build query
        select_cols = ", ".join(columns) if columns else "*"
        where_clause = build_where_clause_from_filters(filters) if filters else ""

        query = f"SELECT {select_cols} FROM {table_name} {where_clause}"

        return self.connection.execute(query).fetchdf()

    def get_unique_values(self, table_name: str, column_name: str) -> List[Any]:
        """
        Get unique values from a specific column in a table.

        Parameters
        ----------
        table_name : str
            Name of the standardized table
        column_name : str
            Name of the column to get unique values from

        Returns
        -------
        List[Any]
            List of unique values in the column

        Raises
        ------
        ValueError
            If DataManager not initialized or table not available
        """
        if self.connection is None:
            raise ValueError("DataManager not initialized. Call initialize() first.")

        if table_name not in self.available_tables:
            raise ValueError(
                f"Table '{table_name}' not available. Available tables: {self.available_tables}"
            )

        query = (
            f"SELECT DISTINCT {column_name} FROM {table_name} ORDER BY {column_name}"
        )
        result = self.connection.execute(query).fetchall()
        return [row[0] for row in result]

    def list_tables(self) -> List[str]:
        """
        List all available standardized tables.

        Returns
        -------
        List[str]
            List of available table names
        """
        return sorted(list(self.available_tables))

    def table_info(self, table_name: str) -> pd.DataFrame:
        """
        Get information about a table's structure.

        Parameters
        ----------
        table_name : str
            Name of the table

        Returns
        -------
        pd.DataFrame
            Table schema information
        """
        if self.connection is None:
            raise ValueError("DataManager not initialized. Call initialize() first.")

        if table_name not in self.available_tables:
            raise ValueError(f"Table '{table_name}' not available.")

        return self.connection.execute(f"DESCRIBE {table_name}").fetchdf()

    def execute_query(self, query: str) -> pd.DataFrame:
        """
        Execute a custom SQL query on the in-memory database.

        Parameters
        ----------
        query : str
            SQL query to execute

        Returns
        -------
        pd.DataFrame
            Query results
        """
        if self.connection is None:
            raise ValueError("DataManager not initialized. Call initialize() first.")

        return self.connection.execute(query).fetchdf()

    def close(self):
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None
            self.available_tables.clear()
            self.sqlite_attached = False
            self.duckdb_attached = False
            logger.info("DataManager connection closed")

    def __del__(self):
        """Ensure connection is closed when object is destroyed."""
        self.close()


# Convenience functions for global access
_data_manager = DataManager()


def initialize_data_manager(
    settings: Dict[str, Any],
    data_location: Union[Path, str] = None,
    lazy_loading: bool = True,
):
    """
    Initialize the global DataManager instance.

    Parameters
    ----------
    settings : Dict[str, Any]
        Settings dictionary containing table configuration parameters
    data_location : Union[Path, str], optional
        Path to database file or folder containing data files
    lazy_loading : bool, optional
        If True, create views instead of loading all data into memory, by default True
    """
    _data_manager.initialize(settings, data_location, lazy_loading)


def get_data(
    table_name: str,
    filters: List[List[Tuple[str, str, Any]]] = None,
    columns: List[str] = None,
    query: str = None,
) -> pd.DataFrame:
    """
    Get data from a standardized table using the global DataManager.

    Parameters
    ----------
    table_name : str
        Name of the standardized table
    filters : List[List[Tuple[str, str, Any]]], optional
        Filters to apply in DNF format
    columns : List[str], optional
        Columns to select
    query : str, optional
        Custom SQL query (overrides other parameters)

    Returns
    -------
    pd.DataFrame
        Requested data
    """
    return _data_manager.get_data(table_name, filters, columns, query)


def get_unique_values(table_name: str, column_name: str) -> List[Any]:
    """
    Get unique values from a specific column in a table using the global DataManager.

    Parameters
    ----------
    table_name : str
        Name of the standardized table
    column_name : str
        Name of the column to get unique values from

    Returns
    -------
    List[Any]
        List of unique values in the column
    """
    return _data_manager.get_unique_values(table_name, column_name)


def list_tables() -> List[str]:
    """List all available standardized tables."""
    return _data_manager.list_tables()


def table_info(table_name: str) -> pd.DataFrame:
    """Get information about a table's structure."""
    return _data_manager.table_info(table_name)


def execute_query(query: str) -> pd.DataFrame:
    """Execute a custom SQL query on the in-memory database."""
    return _data_manager.execute_query(query)
