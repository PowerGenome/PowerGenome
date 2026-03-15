"""
Transmission constraints between regions and line distance
"""

import itertools
import logging
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd

from powergenome.database import get_data, list_tables, register_dataframe_as_table
from powergenome.financials import inflation_price_adjustment
from powergenome.settings import auto_fill_settings
from powergenome.util import map_agg_region_names, reverse_dict_of_lists

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
    region_aggregations: Dict[str, List[str]],
    tx_value_col: str,
) -> pd.DataFrame:
    """Filter regions and map aggregated region names."""
    region_agg_map = reverse_dict_of_lists(region_aggregations or {})

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

    # Add the reverse direction of each combination
    combos += [(to_zone, from_zone) for from_zone, to_zone in combos]

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


@auto_fill_settings()
def agg_transmission_constraints(
    model_regions: List[str] = None,
    region_aggregations: Dict[str, List[str]] = None,
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
    model_regions : List[str], optional
        List of model region names. If not provided, it will be taken from `settings["model_regions"]`.
    region_aggregations : Dict[str, List[str]], optional
        Dictionary mapping aggregated region names to lists of individual regions.
        If not provided, it will be taken from `settings["region_aggregations"]`.
    zone_num_map : Dict[str, int], optional
        Dictionary mapping region names to zone numbers (e.g., {"region1": 1, "region2": 2}).
        If not provided, it will be taken from `settings["zone_num_map"]`.
    tx_value_col : str, optional
        Name of the data column that contains transmission capacity values.
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
    transmission_constraints_table = get_data("transmission_constraints")
    _validate_transmission_data(
        transmission_constraints_table, "transmission_constraints", tx_value_col
    )

    # Filter regions and aggregate transmission capacity
    logger.debug("Map and aggregate region names for transmission constraints")
    aggregated_data = _filter_and_map_regions(
        transmission_constraints_table,
        model_regions,
        region_aggregations,
        tx_value_col,
    )

    # Format output
    return _format_transmission_output(
        aggregated_data, model_regions, zone_num_map, tx_value_col
    )


@auto_fill_settings()
def load_tx_costs(
    target_usd_year: int = None,
    zone_num_map: Dict[str, int] = None,
) -> pd.DataFrame:
    """Load transmission cost data and adjust for inflation.

    Load a data table with cost and line loss of each interregional transmission
    line. Map the region names to zones (z1 to zM) and adjust the total cost columns
    to the target dollar year if specified.

    Parameters
    ----------
    target_usd_year : int, optional
        Desired final dollar year for cost columns, by default None. If None, no
        inflation adjustment is made.
    zone_num_map : Dict[str, int], optional
        Dictionary mapping region names to zone numbers (e.g., {"region1": 1, "region2": 2}).

    Returns
    -------
    pd.DataFrame
        Cost and line loss data for transmission lines between model regions.
        Contains columns "start_region", "dest_region", "zone_1", "zone_2",
        "total_interconnect_annuity_mw", "total_interconnect_cost_mw", "dollar_year",
        and "adjusted_dollar_year" (if target_usd_year is specified).

    Raises
    ------
    KeyError
        If required columns are missing from the loaded data.
    """
    df = get_data("transmission_cost")

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
        # Apply inflation adjustment
        adjusted_annuities = []
        adjusted_costs = []
        for row in df.itertuples():
            adj_annuity = inflation_price_adjustment(
                row.total_interconnect_annuity_mw,
                row.dollar_year,
                target_usd_year,
            )  # .round(0)
            adjusted_annuities.append(adj_annuity)

            adj_cost = inflation_price_adjustment(
                row.total_interconnect_cost_mw,
                row.dollar_year,
                target_usd_year,
            )  # .round(0)
            adjusted_costs.append(adj_cost)

        df["total_interconnect_annuity_mw"] = adjusted_annuities
        df["total_interconnect_cost_mw"] = adjusted_costs
        df["adjusted_dollar_year"] = target_usd_year

    return df


def _get_demand_weights_for_regions(
    model_regions: List[str],
    region_aggregations: Dict[str, List[str]],
) -> Dict[str, float]:
    """Get demand-based weights for base regions, using the final data year.

    Loads the demand table, finds the maximum year available across all base
    regions, confirms common weather years, then returns the total demand for
    each base region in that year as a weight.  If no demand table is available,
    returns uniform weights of 1.0 for every base region.

    Parameters
    ----------
    model_regions : List[str]
        All model regions, including aggregated ones.
    region_aggregations : Dict[str, List[str]]
        Mapping from model region name to list of constituent base regions.

    Returns
    -------
    Dict[str, float]
        Mapping of base region name to total demand weight.
    """
    # Collect all base regions
    all_base_regions: List[str] = []
    for model_region in model_regions:
        if model_region in region_aggregations:
            all_base_regions.extend(region_aggregations[model_region])
        else:
            all_base_regions.append(model_region)

    if "demand" not in list_tables():
        logger.warning(
            "No demand table available. Using uniform weights of 1.0 for all base regions."
        )
        return {r: 1.0 for r in all_base_regions}

    demand_df = get_data("demand")
    demand_df = demand_df[demand_df["region"].isin(all_base_regions)].copy()

    if demand_df.empty:
        logger.warning(
            "No demand data found for the base regions. Using uniform weights of 1.0."
        )
        return {r: 1.0 for r in all_base_regions}

    # Find the maximum year that is available for ALL base regions
    region_max_years = demand_df.groupby("region")["year"].max()
    # Use the smallest of the per-region max years so every region has data
    final_year = int(region_max_years.min())
    logger.debug(f"Using final demand year {final_year} for region weighting.")

    year_demand = demand_df[demand_df["year"] == final_year].copy()

    # Confirm consistent weather years across all base regions
    if "weather_year" in year_demand.columns:
        region_weather_years = year_demand.groupby("region")["weather_year"].apply(set)
        if len(region_weather_years) > 0:
            common_weather_years = set.intersection(*region_weather_years)
        else:
            common_weather_years = set()

        all_same = all(wy_set == common_weather_years for wy_set in region_weather_years)
        if not all_same:
            logger.warning(
                "Not all base regions share the same weather years in the demand table "
                f"for year {final_year}. Using only the common weather years: "
                f"{sorted(common_weather_years)}"
            )
        year_demand = year_demand[year_demand["weather_year"].isin(common_weather_years)]

    # Sum demand per region
    demand_col = "load_mw" if "load_mw" in year_demand.columns else "value"
    demand_by_region = year_demand.groupby("region")[demand_col].sum()

    # For any base region missing from the demand table, use the mean weight
    mean_demand = float(demand_by_region.mean()) if not demand_by_region.empty else 1.0
    return {r: float(demand_by_region.get(r, mean_demand)) for r in all_base_regions}


def _get_connected_model_region_pairs(
    tx_constraints_df: pd.DataFrame,
    model_regions: List[str],
    region_aggregations: Dict[str, List[str]],
) -> List[Tuple[str, str]]:
    """Return unique model-region pairs that have at least one inter-base-region link.

    Parameters
    ----------
    tx_constraints_df : pd.DataFrame
        Transmission constraints table with ``region_from`` / ``region_to`` columns
        containing base region names.
    model_regions : List[str]
        All model region names.
    region_aggregations : Dict[str, List[str]]
        Mapping from model region name to constituent base regions.

    Returns
    -------
    List[Tuple[str, str]]
        Unique connected model-region pairs (alphabetically ordered tuples).
    """
    region_agg_map = reverse_dict_of_lists(region_aggregations or {})

    connected_pairs: set = set()
    for _, row in tx_constraints_df.iterrows():
        model_from = region_agg_map.get(row["region_from"], row["region_from"])
        model_to = region_agg_map.get(row["region_to"], row["region_to"])
        if (
            model_from in model_regions
            and model_to in model_regions
            and model_from != model_to
        ):
            pair = (model_from, model_to) if model_from < model_to else (model_to, model_from)
            connected_pairs.add(pair)
    return list(connected_pairs)


@auto_fill_settings()
def calc_network_upgrade_costs(
    model_regions: List[str] = None,
    region_aggregations: Dict[str, List[str]] = None,
    target_usd_year: int = None,
) -> pd.DataFrame:
    """Calculate network upgrade costs for aggregated model regions.

    Uses base-region transmission costs from the ``transmission_cost`` table to
    compute the per-MW capital and annual costs of inter-regional network
    upgrades.  For each pair of connected model regions the total cost is:

    1. **Minimum direct inter-regional connection cost** – the lowest
       ``total_interconnect_cost_mw`` entry in the cost table that connects a
       base region belonging to model region A with a base region belonging to
       model region B.

    2. **Intra-regional network expansion cost** for each aggregated model
       region – computed by:

       a. Building a minimum-spanning-tree (MST) over the base regions of the
          model region, using ``total_interconnect_cost_mw`` as the edge weight.
       b. Weighting each MST edge by the sum of demands of its two endpoint
          nodes, then normalizing so the weights sum to 1.
       c. Multiplying the normalized weights by the cost (and loss fraction) of
          each MST edge and summing to produce a single intra-regional cost.

    The results are saved as a new ``network_upgrade_costs`` table in the
    global :class:`~powergenome.database.DataManager` for use in subsequent
    planning periods.

    Parameters
    ----------
    model_regions : List[str], optional
        List of model region names.  A model region that is **not** listed in
        ``region_aggregations`` is treated as a single-base-region zone; its
        name must match the corresponding region name in the
        ``transmission_cost`` and ``transmission_constraints`` tables.
        Auto-filled from settings when *None*.
    region_aggregations : Dict[str, List[str]], optional
        Mapping from model region name to the list of base regions it
        aggregates.  Model regions absent from this mapping are treated as
        single-base-region zones.  Auto-filled from settings when *None*.
    target_usd_year : int, optional
        Target dollar year for inflation adjustment of cost columns.

    Returns
    -------
    pd.DataFrame
        One row per connected model-region pair with columns:

        * ``start_region`` – source model region name
        * ``dest_region`` – destination model region name
        * ``total_interconnect_cost_mw`` – combined capital cost ($/MW)
        * ``total_interconnect_annuity_mw`` – combined annual cost ($/MW-yr)
        * ``total_line_loss_frac`` – combined line-loss fraction
        * ``dollar_year`` – dollar year of the cost figures

    Raises
    ------
    KeyError
        If required columns are missing from the transmission cost table.
    """
    if region_aggregations is None:
        region_aggregations = {}

    # ------------------------------------------------------------------
    # 1. Load source tables
    # ------------------------------------------------------------------
    tx_cost_df = get_data("transmission_cost")
    tx_constraints_df = get_data("transmission_constraints")

    required_cols = [
        "start_region",
        "dest_region",
        "total_interconnect_cost_mw",
        "total_interconnect_annuity_mw",
        "dollar_year",
    ]
    missing_cols = [c for c in required_cols if c not in tx_cost_df.columns]
    if missing_cols:
        raise KeyError(
            f"Missing required columns in transmission cost data: {missing_cols}"
        )

    # ------------------------------------------------------------------
    # 2. Optional inflation adjustment
    # ------------------------------------------------------------------
    if target_usd_year:
        tx_cost_df = tx_cost_df.copy()
        adjusted_costs = []
        adjusted_annuities = []
        for row in tx_cost_df.itertuples():
            adjusted_costs.append(
                inflation_price_adjustment(
                    row.total_interconnect_cost_mw, row.dollar_year, target_usd_year
                )
            )
            adjusted_annuities.append(
                inflation_price_adjustment(
                    row.total_interconnect_annuity_mw, row.dollar_year, target_usd_year
                )
            )
        tx_cost_df["total_interconnect_cost_mw"] = adjusted_costs
        tx_cost_df["total_interconnect_annuity_mw"] = adjusted_annuities
        tx_cost_df["dollar_year"] = target_usd_year

    # ------------------------------------------------------------------
    # 3. Build a bidirectional cost lookup keyed by (region_a, region_b)
    # ------------------------------------------------------------------
    cost_lookup: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in tx_cost_df.itertuples():
        data: Dict[str, Any] = {
            "cost": float(row.total_interconnect_cost_mw),
            "annuity": float(row.total_interconnect_annuity_mw),
            "loss": float(getattr(row, "total_line_loss_frac", 0.0) or 0.0),
            "dollar_year": int(row.dollar_year),
        }
        for key in [(row.start_region, row.dest_region), (row.dest_region, row.start_region)]:
            if key not in cost_lookup:
                cost_lookup[key] = data

    # ------------------------------------------------------------------
    # 4. Demand weights for base regions
    # ------------------------------------------------------------------
    demand_weights = _get_demand_weights_for_regions(model_regions, region_aggregations)

    # ------------------------------------------------------------------
    # 5. Intra-regional expansion cost per (aggregated) model region
    # ------------------------------------------------------------------
    def get_base_regions(model_region: str) -> List[str]:
        return region_aggregations.get(model_region, [model_region])

    intra: Dict[str, Dict[str, float]] = {}
    zero_intra = {"cost": 0.0, "annuity": 0.0, "loss": 0.0}

    for model_region in model_regions:
        base_regs = get_base_regions(model_region)
        if len(base_regs) <= 1:
            intra[model_region] = zero_intra.copy()
            continue

        # Build graph over base regions with transmission cost as edge weight
        G = nx.Graph()
        G.add_nodes_from(base_regs)
        for r1, r2 in itertools.combinations(base_regs, 2):
            entry = cost_lookup.get((r1, r2))
            if entry is not None:
                G.add_edge(
                    r1, r2,
                    weight=entry["cost"],
                    cost=entry["cost"],
                    annuity=entry["annuity"],
                    loss=entry["loss"],
                )

        if not nx.is_connected(G):
            logger.warning(
                f"The base-region graph for model region '{model_region}' is not fully "
                "connected. Some base regions may lack transmission cost data between "
                "them; intra-regional cost will be computed from the connected subgraph."
            )
            # Use largest connected component
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()

        try:
            mst = nx.minimum_spanning_tree(G, weight="weight")
        except Exception as exc:
            logger.warning(
                f"Could not compute MST for model region '{model_region}': {exc}. "
                "Intra-regional cost set to zero."
            )
            intra[model_region] = zero_intra.copy()
            continue

        # Demand-weighted average cost over MST edges
        total_demand_weight = 0.0
        weighted_cost = 0.0
        weighted_annuity = 0.0
        weighted_loss = 0.0

        for u, v, edge_data in mst.edges(data=True):
            w = demand_weights.get(u, 1.0) + demand_weights.get(v, 1.0)
            total_demand_weight += w
            weighted_cost += w * edge_data["cost"]
            weighted_annuity += w * edge_data["annuity"]
            weighted_loss += w * edge_data["loss"]

        if total_demand_weight > 0:
            intra[model_region] = {
                "cost": weighted_cost / total_demand_weight,
                "annuity": weighted_annuity / total_demand_weight,
                "loss": weighted_loss / total_demand_weight,
            }
        else:
            intra[model_region] = zero_intra.copy()

    # ------------------------------------------------------------------
    # 6. Connected model-region pairs and inter-regional direct costs
    # ------------------------------------------------------------------
    connected_pairs = _get_connected_model_region_pairs(
        tx_constraints_df, model_regions, region_aggregations
    )

    results = []
    for region_a, region_b in connected_pairs:
        base_regs_a = get_base_regions(region_a)
        base_regs_b = get_base_regions(region_b)

        # Minimum cost direct connection between the two model regions
        best: Optional[Dict[str, Any]] = None
        for r_a in base_regs_a:
            for r_b in base_regs_b:
                entry = cost_lookup.get((r_a, r_b))
                if entry is not None and (best is None or entry["cost"] < best["cost"]):
                    best = entry

        if best is None:
            logger.warning(
                f"No transmission cost data found between model regions '{region_a}' "
                f"and '{region_b}'. Skipping this pair."
            )
            continue

        intra_a = intra.get(region_a, zero_intra)
        intra_b = intra.get(region_b, zero_intra)

        results.append(
            {
                "start_region": region_a,
                "dest_region": region_b,
                "total_interconnect_cost_mw": (
                    best["cost"] + intra_a["cost"] + intra_b["cost"]
                ),
                "total_interconnect_annuity_mw": (
                    best["annuity"] + intra_a["annuity"] + intra_b["annuity"]
                ),
                "total_line_loss_frac": (
                    best["loss"] + intra_a["loss"] + intra_b["loss"]
                ),
                "dollar_year": best["dollar_year"],
            }
        )

    result_df = pd.DataFrame(results)

    # ------------------------------------------------------------------
    # 7. Save to DataManager and return
    # ------------------------------------------------------------------
    if not result_df.empty:
        register_dataframe_as_table("network_upgrade_costs", result_df)
        logger.info(
            f"Saved network upgrade costs for {len(result_df)} model-region pairs "
            "as 'network_upgrade_costs' table in DataManager."
        )

    return result_df


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
    ).drop_duplicates(subset=["start_region", "dest_region"])

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
        validate="one_to_one",
    )

    return result_df
