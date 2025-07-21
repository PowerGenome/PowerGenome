"""
Transmission constraints between regions and line distance
"""

import itertools
import logging
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from typing import Dict, List

import pandas as pd

from powergenome.financials import inflation_price_adjustment
from powergenome.util import (
    find_centroid,
    load_data,
    map_agg_region_names,
    reverse_dict_of_lists,
)

logger = logging.getLogger(__name__)


def _validate_transmission_data(
    df: pd.DataFrame, data_table: str, tx_value_col: str
) -> None:
    """Validate transmission constraints data for required columns and duplicates."""
    if tx_value_col not in df.columns:
        raise KeyError(
            f"There is no column {tx_value_col} in the transmission capacity table '{data_table}'"
        )

    if df.duplicated(subset=["region_from", "region_to"]).any():
        dup_lines = df.loc[
            df.duplicated(subset=["region_from", "region_to"]),
            ["region_from", "region_to"],
        ]
        raise KeyError(
            "The transmission table has duplicate lines. This table should only have unique lines.\n",
            dup_lines,
        )


def _filter_and_map_regions(
    df: pd.DataFrame,
    model_regions: List[str],
    regional_aggregations: Dict[str, List[str]],
    tx_value_col: str,
) -> pd.DataFrame:
    """Filter regions and map aggregated region names."""
    region_agg_map = reverse_dict_of_lists(regional_aggregations or {})

    keep_regions = [
        x
        for x in model_regions + list(region_agg_map)
        if x not in region_agg_map.values()
    ]

    # Filter regions
    filtered_df = df.loc[
        (df.region_from.isin(keep_regions)) & (df.region_to.isin(keep_regions)),
        :,
    ].drop(columns="id", errors="ignore")

    # Map region names
    for col in ["region_from", "region_to"]:
        model_col = "model_" + col
        filtered_df = map_agg_region_names(
            df=filtered_df,
            region_agg_map=region_agg_map,
            original_col_name=col,
            new_col_name=model_col,
        )

    # Aggregate transmission capacity
    keep_cols = ["model_region_from", "model_region_to", tx_value_col]
    drop_cols = [c for c in filtered_df.columns if c not in keep_cols]
    filtered_df.drop(columns=drop_cols, inplace=True)

    return filtered_df.groupby(["model_region_from", "model_region_to"]).sum()


def _format_transmission_output(
    df: pd.DataFrame, zones: List[str], zone_num_map: Dict[str, int], tx_value_col: str
) -> pd.DataFrame:
    """Format the transmission constraints DataFrame for output."""
    combos = list(itertools.combinations(zones, 2))

    # Filter to valid combinations and reset index
    output_df = df.loc[[c for c in combos if c in df.index]].reset_index()

    # Add zone mappings and identifiers
    output_df["Start_Zone"] = output_df["model_region_from"].map(zone_num_map)
    output_df["End_Zone"] = output_df["model_region_to"].map(zone_num_map)

    network_lines = pd.Series(range(1, len(output_df) + 1), name="Network_Lines")
    network_zones = pd.Series(
        [f"z{zone_num_map[zone]}" for zone in zones], name="Network_zones"
    )

    output_df = pd.concat([output_df, network_lines, network_zones], axis=1)
    output_df["Line_Max_Flow_MW"] = output_df[tx_value_col]
    output_df["transmission_path_name"] = (
        output_df["model_region_from"] + "_to_" + output_df["model_region_to"]
    )

    # Apply dtype conversions
    output_df = output_df.astype(
        {
            "Network_zones": "string",
            "Network_Lines": "Int64",
            "Start_Zone": "Int64",
            "End_Zone": "Int64",
            "Line_Max_Flow_MW": "Float64",
            "model_region_from": "string",
            "model_region_to": "string",
            "transmission_path_name": "string",
        }
    )

    return output_df[
        [
            "Network_zones",
            "Network_Lines",
            "Start_Zone",
            "End_Zone",
            "Line_Max_Flow_MW",
            "model_region_from",
            "model_region_to",
            "transmission_path_name",
        ]
    ].rename(
        columns={"model_region_from": "start_region", "model_region_to": "dest_region"}
    )


def agg_transmission_constraints(
    data_location: Path | str,
    data_table: str = "reeds_ba_tx_NARIS_avg",
    model_regions: List[str] = None,
    regional_aggregations: Dict[str, List[str]] = None,
    zone_num_map: Dict[str, int] = None,
    tx_value_col: str = "firm_ttc_mw",
) -> pd.DataFrame:
    """Aggregate transmission constraints/capacity between model regions

    Model regions can consist of one or more individual regions. When two or more regions
    are in a model region, the transmission capacity between individual regions is
    combined.

    Values in a user transmission table will override the database transmission table.

    Parameters
    ----------
    data_location : Path | str
        Path to data location
    data_table : str, optional
        Name of the database table with transmission capacity, by default "reeds_ba_tx_NARIS_avg"
    model_regions : List[str], optional
        List of model region names. If not provided, it will be taken from `settings["model_regions"]`.
    regional_aggregations : Dict[str, List[str]], optional
        Dictionary mapping aggregated region names to lists of individual regions.
        If not provided, it will be taken from `settings["region_aggregations"]`.
    zone_num_map : Dict[str, int], optional
        Dictionary mapping region names to zone numbers (e.g., {"region1": 1, "region2": 2}).
        If not provided, it will be taken from `settings["zone_num_map"]`.
    tx_value_col : str, optional
        Name of the data column in `data_table` that contains transmission capacity values.
        Default is "firm_ttc_mw". If not specified, a warning will be logged and "firm_ttc_mw"
        will be used as the default value.

    Returns
    -------
    pd.DataFrame
        Network lines connecting regions and the min/max flow of each line (if different
        based on direction)

    Raises
    ------
    KeyError
        The specified data column is not in the database transmission table
    KeyError
        The database transmission table has duplicate lines in the same direction
    """
    # Determine transmission value column
    # tx_value_col = settings.get("tx_value_col")
    if not tx_value_col:
        logger.warning(
            "No transmission value column (e.g. firm vs non-firm) was specified in the "
            "settings. The column 'firm_ttc_mw' will be used as a default. This is a change "
            "from previous versions of PG, where 'nonfirm_ttc_mw' was used. Firm transmission "
            "capacity is lower or equal to non-firm capacity."
        )
        tx_value_col = "firm_ttc_mw"

    # Load and validate transmission data
    logger.debug("Loading transmission constraints from the database")
    transmission_constraints_table = load_data(data_location, data_table)
    _validate_transmission_data(
        transmission_constraints_table, data_table, tx_value_col
    )

    # Filter regions and aggregate transmission capacity
    logger.debug("Map and aggregate region names for transmission constraints")
    aggregated_data = _filter_and_map_regions(
        transmission_constraints_table,
        model_regions,
        regional_aggregations,
        tx_value_col,
    )

    # Format output
    return _format_transmission_output(
        aggregated_data, model_regions, zone_num_map, tx_value_col
    )


def load_tx_costs(
    data_location: Path | str,
    table_name: str,
    # model_regions: List[str],
    target_usd_year: int = None,
    zone_num_map: Dict[str, int] = None,
    dollar_year_table: str = None,
) -> pd.DataFrame:
    """Load transmission cost data and adjust for inflation.

    Load a data table with cost and line loss of each interregional transmission
    line. Map the region names to zones (z1 to zM) and adjust the total cost columns
    to the target dollar year if specified.

    Parameters
    ----------
    data_location : Path | str
        Path to the data location containing the transmission cost table.
    table_name : str
        Name of the table/file containing transmission costs. Should have columns
        "start_region", "dest_region", "total_interconnect_annuity_mw",
        "total_interconnect_cost_mw", and "dollar_year".
    model_regions : List[str]
        List of model region names. Should be sorted to match order in other functions.
    target_usd_year : int, optional
        Desired final dollar year for cost columns, by default None. If None, no
        inflation adjustment is made.
    zone_num_map : Dict[str, int], optional
        Dictionary mapping region names to zone numbers (e.g., {"region1": 1, "region2": 2}).
    dollar_year_table : str, optional
        Name of the table containing dollar year inflation data, by default "dollar_years".

    Returns
    -------
    pd.DataFrame
        Cost and line loss data for transmission lines between model regions.
        Contains columns "start_region", "dest_region", "zone_1", "zone_2",
        "total_interconnect_annuity_mw", "total_interconnect_cost_mw", "dollar_year",
        and "adjusted_dollar_year" (if target_usd_year is specified).

    Raises
    ------
    ValueError
        If target_usd_year is specified but settings is None or missing required keys.
    KeyError
        If required columns are missing from the loaded data.
    """
    df = load_data(data_location, table_name)

    # Validate required columns
    required_cols = [
        "start_region",
        "dest_region",
        "total_interconnect_annuity_mw",
        "total_interconnect_cost_mw",
        "dollar_year",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(
            f"Missing required columns in transmission cost data: {missing_cols}"
        )

    df["zone_1"] = df["start_region"].map(zone_num_map)
    df["zone_2"] = df["dest_region"].map(zone_num_map)
    df = df.dropna(subset=["zone_1", "zone_2"])

    if target_usd_year:
        if dollar_year_table is None:
            raise ValueError(
                "Dollar year table is required when target_usd_year is specified"
            )

        # Apply inflation adjustment
        adjusted_annuities = []
        adjusted_costs = []
        for row in df.itertuples():
            adj_annuity = inflation_price_adjustment(
                row.total_interconnect_annuity_mw,
                row.dollar_year,
                target_usd_year,
                data_location=data_location,
                table_name=dollar_year_table,
            )  # .round(0)
            adjusted_annuities.append(adj_annuity)

            adj_cost = inflation_price_adjustment(
                row.total_interconnect_cost_mw,
                row.dollar_year,
                target_usd_year,
                data_location=data_location,
                table_name=dollar_year_table,
            )  # .round(0)
            adjusted_costs.append(adj_cost)

        df["total_interconnect_annuity_mw"] = adjusted_annuities
        df["total_interconnect_cost_mw"] = adjusted_costs
        df["adjusted_dollar_year"] = target_usd_year

    return df


def insert_tx_costs(tx_df: pd.DataFrame, tx_costs: pd.DataFrame) -> pd.DataFrame:
    """Insert transmission costs and line loss data into transmission dataframe.

    Merge transmission cost data with the existing transmission constraints dataframe.
    The cost data can include lines not present in the original transmission dataframe,
    effectively creating new potential transmission lines with zero existing capacity.

    Parameters
    ----------
    tx_df : pd.DataFrame
        Dataframe of interregional transmission lines. Must have columns
        "start_region", "dest_region", and other transmission constraint columns.
    tx_costs : pd.DataFrame
        Dataframe of transmission costs and line loss. Should have columns
        "start_region", "dest_region", "zone_1", "zone_2", "total_interconnect_annuity_mw",
        "total_interconnect_cost_mw", and optionally "total_line_loss_frac".
        Cost values should already be adjusted to the desired dollar year.

    Returns
    -------
    pd.DataFrame
        Transmission dataframe with added cost and line loss columns:
        "Line_Reinforcement_Cost_per_MWyr", "Line_Reinforcement_Cost_per_MW",
        and "Line_Loss_Percentage".

    Notes
    -----
    This function creates bidirectional cost data by duplicating each cost entry
    with swapped start/destination regions to ensure costs are available for
    transmission in both directions.
    """
    if tx_df.empty:
        return tx_df

    # Clean and prepare cost data
    tx_costs_clean = tx_costs.dropna(subset=["zone_1", "zone_2"], how="any").copy()

    # Rename cost columns to match expected output format
    cost_column_mapping = {
        "total_interconnect_annuity_mw": "Line_Reinforcement_Cost_per_MWyr",
        "total_interconnect_cost_mw": "Line_Reinforcement_Cost_per_MW",
        "total_line_loss_frac": "Line_Loss_Percentage",
    }
    tx_costs_clean = tx_costs_clean.rename(columns=cost_column_mapping)

    # Create bidirectional cost data by including both directions
    tx_costs_bidirectional = pd.concat(
        [
            tx_costs_clean,
            tx_costs_clean.rename(
                columns={
                    "start_region": "dest_region",
                    "dest_region": "start_region",
                    "zone_1": "zone_2",
                    "zone_2": "zone_1",
                }
            ),
        ],
        ignore_index=True,
    )

    # Select only the columns needed for merging
    merge_columns = ["start_region", "dest_region"]
    cost_columns = [
        col
        for col in cost_column_mapping.values()
        if col in tx_costs_bidirectional.columns
    ]

    # Merge with transmission dataframe
    result_df = pd.merge(
        tx_df,
        tx_costs_bidirectional[merge_columns + cost_columns],
        on=merge_columns,
        how="left",
    )

    return result_df
