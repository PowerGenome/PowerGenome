import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import duckdb
import pandas as pd

from powergenome.settings import Settings
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

    Table configuration in settings
    - Each standardized table (e.g., generation_table, demand_table) can be provided as:
        - A string: the file name (for folder data) or table name (for DBs)
        - A dict with keys:
            - table_name (or name): source file/table
            - columns (optional): list[str] projection
            - filters (optional): DNF filters; accepts lists or tuples in shapes like
                [[("col", "=", val)], [("col2", ">", 0)]], [["col","=",val]], or ["col","=",val]
            - scenario (optional): convenience filter; ANDed into every OR-clause as (scenario = <value>)
    - If both filters and scenario are provided, scenario is added to each clause unless
        a scenario condition already exists in that clause.
    """

    _instance = None
    _initialized = False
    _lock = threading.Lock()

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
            with cls._lock:
                # Double-check pattern
                if cls._instance is None:
                    cls._instance = super(DataManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            with self._lock:
                # Double-check pattern
                if not self._initialized:
                    self.connection = None
                    self.data_location = None
                    self.available_tables = set()
                    self.table_configurations = {}  # Store original table configs
                    self._initialized = True
                    self.sqlite_attached = False
                    self.duckdb_attached = False

    def initialize(
        self,
        settings: Union[Dict[str, Any], Settings],
        data_location: Union[Path, str] = None,
        lazy_loading: bool = True,
    ):
        """
        Initialize the DataManager with settings and data location.

        Parameters
        ----------
        settings : Union[Dict[str, Any], Settings]
            Settings dictionary or Settings object containing table configuration parameters
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
        self.settings = self._convert_settings_to_dict(settings)
        self.lazy_loading = lazy_loading
        self.table_configurations = {}  # Reset configurations

        # Setup tables based on settings
        self._setup_tables()

        logger.info(f"DataManager initialized with {len(self.available_tables)} tables")

    def _convert_settings_to_dict(
        self, settings: Union[Dict[str, Any], Settings]
    ) -> Dict[str, Any]:
        """
        Convert Settings object to dictionary if needed.

        Parameters
        ----------
        settings : Union[Dict[str, Any], Settings]
            Settings dictionary or Settings object

        Returns
        -------
        Dict[str, Any]
            Settings as a dictionary
        """
        # Check if it's a Settings object (has to_dict method)
        if hasattr(settings, "to_dict") and callable(getattr(settings, "to_dict")):
            return settings.to_dict()
        # Check if it's a Settings object (has get_data method)
        elif hasattr(settings, "get_data") and callable(getattr(settings, "get_data")):
            return settings.get_data()
        # Check if it's dictionary-like (has .get method)
        elif hasattr(settings, "get"):
            return dict(settings)
        else:
            raise TypeError(
                f"Settings must be a dictionary or Settings object, got {type(settings)}"
            )

    def _setup_tables(self):
        """Setup standardized tables/views based on settings parameters."""
        self.available_tables.clear()

        for setting_key, standard_name in self.STANDARD_TABLE_MAPPING.items():
            table_config = self.settings.get(setting_key)

            if not table_config:
                continue

            try:
                # Store the configuration for potential updates
                self.table_configurations[standard_name] = {
                    "setting_key": setting_key,
                    "config": table_config,
                }

                self._create_table_view(table_config, standard_name)
                self.available_tables.add(standard_name)
                logger.debug(f"Created table/view: {standard_name}")
            except Exception as e:
                raise RuntimeError(f"Failed to create table {standard_name}: {e}")

    def _validate_table_config(
        self, table_config: Union[str, Dict], standard_name: str
    ) -> Tuple[str, List, List[str]]:
        """
        Validate and normalize table configuration.

        Parameters
        ----------
        table_config : Union[str, Dict]
            Either a string table name or dict with table configuration.
            Dict keys supported:
            - table_name/name: source file/table
            - columns (optional): list[str] projection
            - filters (optional): DNF filters; lists or tuples are accepted. Shapes:
                [col, op, val], [[col, op, val], ...], or [[(...), ...], [...]]
            - scenario (optional): value for a "scenario" column; this is ANDed into
                every OR-clause. If a clause already contains a scenario condition, it is
                not duplicated.
        standard_name : str
            Standardized name for the table in the in-memory database

        Returns
        -------
        Tuple[str, List, List[str]]
            Normalized source table name, filters (DNF, normalized), and columns

        Raises
        ------
        ValueError
            If table configuration is invalid or file/table names don't match data location type
        """
        if isinstance(table_config, str):
            source_table = table_config
            filters = None
            columns = None
            scenario = None
        elif isinstance(table_config, dict):
            source_table = table_config.get("table_name") or table_config.get("name")
            filters = table_config.get("filters")
            columns = table_config.get("columns")
            scenario = table_config.get("scenario")
        else:
            raise ValueError(f"Invalid table configuration: {table_config}")

        if not source_table:
            raise ValueError(f"No table name found in configuration: {table_config}")

        # If a scenario is provided, fold it into filters (DNF). We AND the scenario
        # condition into every OR-clause. If no filters exist, create a single clause.
        if scenario is not None:
            # Ensure DNF structure: List[List[cond]]
            if filters is None:
                filters = [[("scenario", "=", scenario)]]
            else:
                # Defensive: make tuples if lists are provided
                def as_tuple(cond):
                    return tuple(cond) if isinstance(cond, list) else cond

                # Normalize and append scenario condition if not already present
                normalized = []
                for clause in filters:
                    clause_conds = [as_tuple(c) for c in clause]
                    has_scenario = any(
                        (
                            isinstance(c, (list, tuple))
                            and len(c) >= 1
                            and c[0] == "scenario"
                        )
                        for c in clause_conds
                    )
                    if not has_scenario:
                        clause_conds.append(("scenario", "=", scenario))
                    normalized.append(clause_conds)
                filters = normalized

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

    def get_timeseries_data(
        self,
        table_name: str,
        group_by: Optional[List[str]] = None,
        value_col: str = "value",
        agg: str = "sum",
        filters: List[List[Tuple[str, str, Any]]] = None,
        order_by: Optional[List[str]] = None,
        limit: Optional[int] = None,
        wide: bool = False,
        pivot_columns: Optional[Union[str, List[str]]] = None,
        pivot_index: Optional[List[str]] = None,
        fill_value: Any = None,
    ) -> pd.DataFrame:
        """
        Aggregate timeseries data with SQL GROUP BY, returning one row per group.

        Parameters
        ----------
        table_name : str
            Name of the standardized table (e.g., "demand").
        group_by : List[str], optional
            Columns to group by. Defaults to ["time_index"].
        value_col : str
            Name of the numeric value column to aggregate. Defaults to "value".
        agg : str
            Aggregation function: one of {"sum", "avg", "min", "max", "count"}. Defaults to "sum".
        filters : List[List[Tuple[str, str, Any]]], optional
            Filters to apply in DNF format (same shapes as get_data).
        order_by : List[str], optional
            Columns to order by; defaults to group_by if not provided.
        limit : int, optional
            Optional LIMIT on the number of output rows.

        Returns
        -------
        pd.DataFrame
            - If wide is False: DataFrame containing group_by columns and a single aggregated column named "value".
            - If wide is True: Wide DataFrame pivoted by pivot_columns with values in "value". Index defaults to group_by \ pivot_columns.

        Raises
        ------
        ValueError
            If DataManager is not initialized, table is unavailable, or inputs are invalid.
        RuntimeError
            If the query fails to execute.
        """
        if self.connection is None:
            raise ValueError("DataManager not initialized. Call initialize() first.")

        if table_name not in self.available_tables:
            raise ValueError(
                f"Table '{table_name}' not available. Available tables: {self.available_tables}"
            )

        # Validate aggregation
        agg_lc = (agg or "").lower()
        allowed_aggs = {"sum", "avg", "min", "max", "count"}
        if agg_lc not in allowed_aggs:
            raise ValueError(
                f"Unsupported aggregation '{agg}'. Allowed: {sorted(allowed_aggs)}"
            )

        # Group-by defaults to time_index
        if group_by is None:
            group_by = ["time_index"]
        if not isinstance(group_by, (list, tuple)) or not all(
            isinstance(c, str) for c in group_by
        ):
            raise TypeError("group_by must be a list of column names")

        # Order by default mirrors group_by
        if order_by is None:
            order_by = list(group_by)

        select_group = ", ".join(group_by) if group_by else ""
        select_clause = (
            f"{select_group}, {agg_lc}({value_col}) AS value"
            if select_group
            else f"{agg_lc}({value_col}) AS value"
        )
        from_clause = table_name
        where_clause = build_where_clause_from_filters(filters) if filters else ""
        group_clause = f"GROUP BY {select_group}" if select_group else ""
        order_clause = f"ORDER BY {', '.join(order_by)}" if order_by else ""
        limit_clause = (
            f"LIMIT {int(limit)}" if isinstance(limit, int) and limit >= 0 else ""
        )

        query = f"SELECT {select_clause} FROM {from_clause} {where_clause} {group_clause} {order_clause} {limit_clause}"

        try:
            df = self.connection.execute(query).fetchdf()
        except Exception as e:
            raise RuntimeError(f"Timeseries aggregation query failed: {e}")

        # Optional wide pivot using pandas for simplicity and portability
        if wide:
            # Normalize pivot_columns to list
            if pivot_columns is None:
                raise ValueError("wide=True requires pivot_columns to be specified")
            if isinstance(pivot_columns, str):
                pivot_cols_list = [pivot_columns]
            elif isinstance(pivot_columns, (list, tuple)) and all(
                isinstance(c, str) for c in pivot_columns
            ):
                pivot_cols_list = list(pivot_columns)
            else:
                raise TypeError("pivot_columns must be a string or list of strings")

            # Determine index columns: default to group_by excluding pivot columns
            if pivot_index is None:
                pivot_index_cols = [
                    c for c in group_by if c not in set(pivot_cols_list)
                ]
                if not pivot_index_cols:
                    # If nothing left for index, use time_index if present or raise
                    pivot_index_cols = [c for c in ["time_index"] if c in df.columns]
                    if not pivot_index_cols:
                        raise ValueError(
                            "Cannot infer pivot index. Provide pivot_index or include a non-pivot column in group_by."
                        )
            else:
                if not isinstance(pivot_index, (list, tuple)) or not all(
                    isinstance(c, str) for c in pivot_index
                ):
                    raise TypeError("pivot_index must be a list of column names")
                pivot_index_cols = list(pivot_index)

            # Validate columns exist
            for c in pivot_index_cols + pivot_cols_list + ["value"]:
                if c not in df.columns:
                    raise ValueError(
                        f"Column '{c}' required for pivot is not in result"
                    )

            # Support single pivot column primarily; allow multi for advanced users
            if len(pivot_cols_list) == 1:
                wide_df = df.pivot_table(
                    index=pivot_index_cols,
                    columns=pivot_cols_list[0],
                    values="value",
                    fill_value=fill_value,
                    aggfunc="first",
                ).reset_index()
                # Flatten potential Index name on columns
                wide_df.columns.name = None
                return wide_df
            else:
                wide_df = df.pivot_table(
                    index=pivot_index_cols,
                    columns=pivot_cols_list,
                    values="value",
                    fill_value=fill_value,
                    aggfunc="first",
                ).reset_index()
                # Flatten MultiIndex columns to strings
                if isinstance(wide_df.columns, pd.MultiIndex):
                    wide_df.columns = [
                        "_".join(
                            [str(x) for x in tup if x is not None and x != ""]
                        ).strip("_")
                        for tup in wide_df.columns.values
                    ]
                return wide_df

        return df

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

    def update(self, updated_settings: Union[Dict[str, Any], Settings] = None):
        """
        Update source tables with new configurations.

        Parameters
        ----------
        updated_settings : Union[Dict[str, Any], Settings], optional
            New settings dictionary or Settings object with updated table configurations.
            If None, uses current settings to refresh all tables.
        """
        if self.connection is None:
            raise ValueError("DataManager not initialized. Call initialize() first.")

        # Use provided settings or current settings
        if updated_settings is not None:
            updated_settings_dict = self._convert_settings_to_dict(updated_settings)
            self.settings.update(updated_settings_dict)

        # Find tables where configuration has changed
        tables_to_update = set()
        for setting_key, standard_name in self.STANDARD_TABLE_MAPPING.items():
            new_config = self.settings.get(setting_key)
            old_config = self.table_configurations.get(standard_name, {}).get("config")

            if new_config != old_config:
                tables_to_update.add(standard_name)

        # Update each table
        for standard_name in tables_to_update:
            setting_key = None
            # Find the setting key for this table
            for key, name in self.STANDARD_TABLE_MAPPING.items():
                if name == standard_name:
                    setting_key = key
                    break

            if not setting_key:
                logger.warning(f"Could not find setting key for table {standard_name}")
                continue

            table_config = self.settings.get(setting_key)

            if not table_config:
                # Remove table if no longer configured
                self._remove_table(standard_name)
                continue

            try:
                # Remove existing table/view
                self._remove_table(standard_name)

                # Update stored configuration
                self.table_configurations[standard_name] = {
                    "setting_key": setting_key,
                    "config": table_config,
                }

                # Recreate table/view
                self._create_table_view(table_config, standard_name)
                self.available_tables.add(standard_name)

                logger.info(f"Updated table/view: {standard_name}")

            except Exception as e:
                logger.error(f"Failed to update table {standard_name}: {e}")
                # Remove from available tables if update failed
                self.available_tables.discard(standard_name)
                self.table_configurations.pop(standard_name, None)

    def _remove_table(self, table_name: str):
        """Remove a table or view from the in-memory database."""
        try:
            # Try to drop as view first, then as table
            try:
                self.connection.execute(f"DROP VIEW IF EXISTS {table_name}")
            except Exception as e:
                logger.debug(f"Failed to drop view {table_name}: {e}")
            try:
                self.connection.execute(f"DROP TABLE IF EXISTS {table_name}")
            except Exception as e:
                logger.debug(f"Failed to drop table {table_name}: {e}")

            self.available_tables.discard(table_name)
            logger.debug(f"Removed table/view: {table_name}")

        except Exception as e:
            logger.warning(f"Error removing table {table_name}: {e}")

    def close(self):
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None
            self.available_tables.clear()
            self.table_configurations.clear()
            self.sqlite_attached = False
            self.duckdb_attached = False
            logger.info("DataManager connection closed")

    def __del__(self):
        """Fallback cleanup - not guaranteed to be called."""
        try:
            if hasattr(self, "connection") and self.connection:
                self.connection.close()
        except Exception:
            # Suppress all exceptions in __del__ to avoid issues during shutdown
            pass


# Convenience functions for global access
_data_manager = DataManager()


def initialize_data_manager(
    settings: Union[Dict[str, Any], Settings],
    data_location: Union[Path, str] = None,
    lazy_loading: bool = True,
):
    """
    Initialize the global DataManager instance.

    Parameters
    ----------
    settings : Union[Dict[str, Any], Settings]
        Settings dictionary or Settings object containing table configuration parameters
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


def update_data_manager(updated_settings: Dict[str, Any] = None):
    """
    Update source tables in the global DataManager instance.

    Parameters
    ----------
    updated_settings : Dict[str, Any], optional
        New settings dictionary with updated table configurations.
        If None, uses current settings to refresh all tables.
    """
    _data_manager.update(updated_settings)


def get_timeseries_data(
    table_name: str,
    group_by: Optional[List[str]] = None,
    value_col: str = "value",
    agg: str = "sum",
    filters: List[List[Tuple[str, str, Any]]] = None,
    order_by: Optional[List[str]] = None,
    limit: Optional[int] = None,
    wide: bool = False,
    pivot_columns: Optional[Union[str, List[str]]] = None,
    pivot_index: Optional[List[str]] = None,
    fill_value: Any = None,
) -> pd.DataFrame:
    """
    Global convenience wrapper for DataManager.get_timeseries_data.

    See DataManager.get_timeseries_data for parameter details.
    """
    return _data_manager.get_timeseries_data(
        table_name=table_name,
        group_by=group_by,
        value_col=value_col,
        agg=agg,
        filters=filters,
        order_by=order_by,
        limit=limit,
        wide=wide,
        pivot_columns=pivot_columns,
        pivot_index=pivot_index,
        fill_value=fill_value,
    )
