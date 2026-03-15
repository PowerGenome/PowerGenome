"""
Transmission constraints between regions and line distance
"""

import itertools
import logging
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from typing import Dict, List

import networkx as nx
import pandas as pd
import sqlalchemy as sa

from powergenome.util import (
    find_centroid,
    find_region_col,
    map_agg_region_names,
    reverse_dict_of_lists,
)

logger = logging.getLogger(__name__)


def agg_transmission_constraints(
    pg_engine: sa.engine.base.Engine,
    settings: dict,
    pg_table: str = "transmission_single_epaipm",
    settings_agg_key: str = "region_aggregations",
) -> pd.DataFrame:
    """Aggregate transmission constraints/capacity between model regions

    Model regions can consist of one or more individual regions. When two or more regions
    are in a model region, the transmission capacity between individual regions is
    combined.

    Values in a user transmission table will override the database transmission table.

    Parameters
    ----------
    pg_engine : sa.engine.base.Engine
        Engine to conect with a database
    settings : dict
        Dictionary of settings parameters. Must include "model_regions". Optional parameters
        include "tx_value_col" (name of the data column in `pg_table`, default value is
        "firm_ttc_mw"), "user_transmission_constraints_fn" (name of user data file, must
        be combined with the parameter "input_folder"), and the value of `settings_agg_key`
        if any regions are aggregated.
    pg_table : str, optional
        Name of the database table with transmission capacity, by default "transmission_single_epaipm"
    settings_agg_key : str, optional
        Name of the settings parameter where regions are aggregated, by default "region_aggregations"

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
    KeyError
        The specified data column is not in the user supplied transmission table
    KeyError
        The user transmission table has duplicate lines in the same direction
    """
    tx_value_col = settings.get("tx_value_col")
    if not tx_value_col:
        logger.warning(
            "No transmission value column (e.g. firm vs non-firm) was specified in the "
            "settings. The column 'firm_ttc_mw' will be used as a default. This is a change "
            "from previous versions of PG, where 'nonfirm_ttc_mw' was used. Firm transmission "
            "capacity is lower or equal to non-firm capacity."
        )
        tx_value_col = "firm_ttc_mw"
    zones = settings["model_regions"]
    zone_num_map = {
        zone: f"z{number + 1}" for zone, number in zip(zones, range(len(zones)))
    }

    combos = list(itertools.combinations(zones, 2))
    reverse_combos = [(combo[-1], combo[0]) for combo in combos]

    logger.debug("Loading transmission constraints from the database")
    transmission_constraints_table = pd.read_sql_table(pg_table, con=pg_engine)

    if tx_value_col not in transmission_constraints_table.columns:
        raise KeyError(
            f"There is no column {tx_value_col} in the transmission capacity table '{pg_table}'"
        )
    if transmission_constraints_table.duplicated(
        subset=["region_from", "region_to"]
    ).any():
        dup_lines = transmission_constraints_table.loc[
            transmission_constraints_table.duplicated(
                subset=["region_from", "region_to"]
            ),
            ["region_from", "region_to"],
        ]

        raise KeyError(
            "The transmission table has duplicate lines. This table should only have unique lines.\n",
            dup_lines,
        )
    if settings.get("user_transmission_constraints_fn"):
        logger.debug("Adding user-supplied transmission constraint data")
        user_tx_constraints = pd.read_csv(
            Path(settings["input_folder"])
            / settings["user_transmission_constraints_fn"]
        )
        if tx_value_col not in user_tx_constraints.columns:
            raise KeyError(
                f"There is no column {tx_value_col} in the user supplied transmission capacity table"
            )
        if user_tx_constraints.duplicated(subset=["region_from", "region_to"]).any():
            dup_lines = user_tx_constraints.loc[
                user_tx_constraints.duplicated(subset=["region_from", "region_to"]),
                ["region_from", "region_to"],
            ]

            raise KeyError(
                "The user transmission table has duplicate lines. This table should only have unique lines.\n",
                dup_lines,
            )

        # user constraints are needed bidirectionaly
        transmission_constraints_table = pd.concat(
            [
                transmission_constraints_table,
                user_tx_constraints,
                user_tx_constraints.rename(
                    columns={"region_from": "region_to", "region_to": "region_from"}
                ),
            ]
        )

        if transmission_constraints_table.duplicated(
            subset=["region_from", "region_to"]
        ).any():
            logger.warning(
                "The user transmission capacity table duplicates some values from the "
                "database. Database values will be discarded in these cases."
            )

        transmission_constraints_table = transmission_constraints_table.drop_duplicates(
            keep="last"
        )

    # Settings has a dictionary of lists for regional aggregations. Need
    # to reverse this to use in a map method.
    region_agg_map = reverse_dict_of_lists(settings.get(settings_agg_key))

    # IPM regions to keep. Regions not in this list will be dropped from the
    # dataframe
    keep_regions = [
        x
        for x in settings["model_regions"] + list(region_agg_map)
        if x not in region_agg_map.values()
    ]

    # Create new column "model_region_from"  and "model_region_to" with labels that
    # we're using for aggregated regions
    transmission_constraints_table = transmission_constraints_table.loc[
        (transmission_constraints_table.region_from.isin(keep_regions))
        & (transmission_constraints_table.region_to.isin(keep_regions)),
        :,
    ].drop(columns="id", errors="ignore")

    logger.debug("Map and aggregate region names for transmission constraints")
    for col in ["region_from", "region_to"]:
        model_col = "model_" + col

        transmission_constraints_table = map_agg_region_names(
            df=transmission_constraints_table,
            region_agg_map=region_agg_map,
            original_col_name=col,
            new_col_name=model_col,
        )

    keep_cols = ["model_region_from", "model_region_to", tx_value_col]
    drop_cols = [
        c for c in transmission_constraints_table.columns if c not in keep_cols
    ]
    transmission_constraints_table.drop(columns=drop_cols, inplace=True)
    transmission_constraints_table = transmission_constraints_table.groupby(
        ["model_region_from", "model_region_to"]
    ).sum()

    # Build the final output dataframe
    logger.debug(
        "Build a new transmission constraints dataframe with a single line between "
        "regions"
    )
    tc_joined = pd.DataFrame(
        columns=["Network_Lines"] + zones + ["Line_Max_Flow_MW", "Line_Min_Flow_MW"],
        index=transmission_constraints_table.reindex(combos).dropna().index,
        data=0,
    )

    if tc_joined.empty:
        logger.warning(f"No transmission lines exist between model regions {combos}")
        tc_joined["transmission_path_name"] = None
        tc_joined.rename(columns=zone_num_map, inplace=True)
        return tc_joined.reset_index(drop=True)

    tc_joined["Network_Lines"] = range(1, len(tc_joined) + 1)
    tc_joined["Line_Max_Flow_MW"] = transmission_constraints_table.reindex(
        combos
    ).dropna()

    reverse_tc = transmission_constraints_table.reindex(reverse_combos).dropna() * -1
    reverse_tc.index = tc_joined.index
    tc_joined["Line_Min_Flow_MW"] = reverse_tc

    for idx, _ in tc_joined.iterrows():
        tc_joined.loc[idx, idx[0]] = 1
        tc_joined.loc[idx, idx[-1]] = -1

    tc_joined.rename(columns=zone_num_map, inplace=True)
    tc_joined = tc_joined.reset_index()
    tc_joined["transmission_path_name"] = (
        tc_joined["model_region_from"] + "_to_" + tc_joined["model_region_to"]
    )
    # tc_joined = tc_joined.set_index("Transmission Path Name")
    tc_joined.drop(columns=["model_region_from", "model_region_to"], inplace=True)

    return tc_joined


def calc_network_upgrade_costs(
    pg_engine: sa.engine.base.Engine,
    settings: dict,
    cost_table: str = "transmission_cost_nrel_reeds",
    demand_table: str = "load_curves_nrel_efs",
    tx_constraints_table: str = "transmission_single_epaipm",
    settings_agg_key: str = "region_aggregations",
) -> pd.DataFrame:
    """Calculate network upgrade costs for model regions from base region cost data.

    Rather than requiring users to provide transmission costs for their specific model
    region aggregations, this function automatically calculates network upgrade costs
    using base region cost data (which stays constant across different model
    configurations).

    The algorithm proceeds as follows:

    1. Connected model region pairs are identified from the transmission constraints
       table.
    2. For each connected pair, the lowest-cost direct connection between any pair of
       base regions (one from each model region) in the cost table is selected.
    3. For aggregated model regions (collections of base regions), the intra-regional
       network expansion cost is computed by:
          a. Building a graph of base regions within the model region and calculating
             its minimum spanning tree (MST), weighted by transmission cost.
          b. Each MST edge is assigned a normalized weight based on the sum of total
             annual demand of the two endpoint base regions, so that all edge weights
             sum to 1.
          c. The intra-regional cost and loss are the demand-weighted averages over
             all MST edges.
    4. The total upgrade cost for a model region pair is the direct inter-regional cost
       plus the intra-regional cost for each endpoint model region.

    Demand data from the final (maximum) year common to all base regions is used as
    weights. If regions have different ``weather_year`` values, only the set of weather
    years shared across all regions is used.

    Parameters
    ----------
    pg_engine : sa.engine.base.Engine
        Engine to connect to a PowerGenome database.
    settings : dict
        User parameter settings. Required key is ``"model_regions"``. Optional keys
        include ``"region_aggregations"`` (or the value of ``settings_agg_key``).
    cost_table : str, optional
        Name of the database table containing per-MW transmission costs between base
        regions, by default ``"transmission_cost_nrel_reeds"``. Required columns are
        ``"region_from"``, ``"region_to"``, ``"line_loss_frac"``, and ``"dollar_year"``.
        At least one of ``"capital_cost_mw"`` (capital $/MW) or ``"annum_cost_mw"``
        (annual $/MW-year) must also be present.
    demand_table : str, optional
        Name of the database table containing hourly demand profiles, by default
        ``"load_curves_nrel_efs"``. Used to derive demand-based weights for the
        intra-regional MST edge assignments.
    tx_constraints_table : str, optional
        Name of the database table with transmission constraints (capacity), by default
        ``"transmission_single_epaipm"``. Used only to determine which pairs of model
        regions are connected.
    settings_agg_key : str, optional
        Name of the settings parameter that maps aggregated model region names to their
        constituent base regions, by default ``"region_aggregations"``.

    Returns
    -------
    pd.DataFrame
        One row per connected model region pair with columns:

        * ``start_region``, ``dest_region`` – model region names.
        * ``total_interconnect_cost_mw`` – total capital cost per MW (present when
          ``capital_cost_mw`` exists in ``cost_table``).
        * ``total_interconnect_annuity_mw`` – total annual cost per MW-year (present
          when ``annum_cost_mw`` exists in ``cost_table``).
        * ``total_line_loss_frac`` – combined fractional line losses.
        * ``dollar_year`` – dollar year of the cost data.

        The result is compatible with :func:`~powergenome.external_data.insert_user_tx_costs`.

    Raises
    ------
    KeyError
        A required column is missing from the cost table.
    """
    model_regions = settings["model_regions"]
    region_aggregations = settings.get(settings_agg_key) or {}

    def get_base_regions(model_region: str) -> List[str]:
        """Return the list of base regions for a model region."""
        return region_aggregations.get(model_region, [model_region])

    # Map every base region back to its parent model region
    base_to_model: Dict[str, str] = {}
    for mr in model_regions:
        for br in get_base_regions(mr):
            base_to_model[br] = mr

    # ------------------------------------------------------------------ #
    # Load and validate the cost table                                      #
    # ------------------------------------------------------------------ #
    logger.debug("Loading transmission costs from '%s'", cost_table)
    tx_costs = pd.read_sql_table(cost_table, con=pg_engine)

    for col in ("region_from", "region_to", "line_loss_frac"):
        if col not in tx_costs.columns:
            raise KeyError(
                f"Required column '{col}' not found in cost table '{cost_table}'."
            )

    has_capital = "capital_cost_mw" in tx_costs.columns
    has_annuity = "annum_cost_mw" in tx_costs.columns
    if not has_capital and not has_annuity:
        raise KeyError(
            f"Cost table '{cost_table}' must contain at least one cost column: "
            "'capital_cost_mw' or 'annum_cost_mw'."
        )

    # Column used as weight when building the MST and selecting the minimum-cost
    # direct connection.  Capital cost is preferred over the annuity because it
    # represents the full up-front investment and is less sensitive to assumed
    # discount rate or project lifetime.
    mst_weight_col = "capital_cost_mw" if has_capital else "annum_cost_mw"

    # ------------------------------------------------------------------ #
    # Find connected model region pairs from the constraints table          #
    # ------------------------------------------------------------------ #
    logger.debug(
        "Loading transmission constraints from '%s' to find connected regions",
        tx_constraints_table,
    )
    tx_constraints = pd.read_sql_table(tx_constraints_table, con=pg_engine).drop(
        columns=["id"], errors="ignore"
    )

    connected_pairs: set = set()
    for _, row in tx_constraints.iterrows():
        mr_from = base_to_model.get(row["region_from"])
        mr_to = base_to_model.get(row["region_to"])
        if mr_from and mr_to and mr_from != mr_to:
            connected_pairs.add(tuple(sorted([mr_from, mr_to])))

    if not connected_pairs:
        logger.warning(
            "No connected model region pairs were found in '%s' for the given "
            "model_regions. Returning an empty DataFrame.",
            tx_constraints_table,
        )
        return pd.DataFrame()

    # ------------------------------------------------------------------ #
    # Load demand data to weight the intra-regional MST edges              #
    # ------------------------------------------------------------------ #
    # Collect base regions that belong to aggregated (multi-base) model regions
    agg_base_regions: List[str] = [
        br
        for mr, bases in region_aggregations.items()
        if len(bases) > 1
        for br in bases
    ]

    region_demand: pd.Series = pd.Series(dtype=float)

    if agg_base_regions:
        inst = sa.inspect(pg_engine)
        table_cols = [c["name"] for c in inst.get_columns(demand_table)]
        context = f"demand table '{demand_table}' in the PowerGenome database"
        region_col = find_region_col(table_cols, context)

        # Find which base regions actually have data
        # `?` placeholders are filled by the `params` list passed to
        # pd.read_sql_query; only the column *name* (`region_col`) is
        # interpolated directly into the query string (not user-supplied data).
        placeholders = ",".join(["?"] * len(agg_base_regions))
        s = (
            f"SELECT DISTINCT year, {region_col} AS region "
            f"FROM {demand_table} "
            f"WHERE {region_col} IN ({placeholders})"
        )
        region_years_df = pd.read_sql_query(
            sql=s, con=pg_engine, params=agg_base_regions
        )

        if region_years_df.empty:
            logger.warning(
                "No demand data found for aggregated base regions in '%s'. "
                "Intra-regional costs will be set to zero.",
                demand_table,
            )
        else:
            regions_with_data = region_years_df["region"].unique().tolist()
            missing = [r for r in agg_base_regions if r not in regions_with_data]
            if missing:
                logger.warning(
                    "Demand data not found for base region(s) %s in '%s'. "
                    "These regions will be excluded from intra-regional cost "
                    "weighting.",
                    missing,
                    demand_table,
                )

            # Determine the set of years common to every region with data
            years_per_region = {
                r: set(
                    region_years_df.loc[
                        region_years_df["region"] == r, "year"
                    ].tolist()
                )
                for r in regions_with_data
            }
            common_years = set.intersection(*years_per_region.values())

            if not common_years:
                logger.warning(
                    "No common data years found across all aggregated base regions. "
                    "Intra-regional costs will be set to zero."
                )
            else:
                # Optionally check weather_year consistency
                placeholders_wr = ",".join(["?"] * len(regions_with_data))
                if "weather_year" in table_cols:
                    s_wy = (
                        f"SELECT DISTINCT year, weather_year, {region_col} AS region "
                        f"FROM {demand_table} "
                        f"WHERE {region_col} IN ({placeholders_wr})"
                    )
                    wy_df = pd.read_sql_query(
                        sql=s_wy, con=pg_engine, params=regions_with_data
                    )
                    wy_per_region = {
                        r: set(
                            wy_df.loc[wy_df["region"] == r, "weather_year"].tolist()
                        )
                        for r in regions_with_data
                    }
                    common_wy = set.intersection(*wy_per_region.values())
                    all_wy = set.union(*wy_per_region.values())
                    if common_wy != all_wy:
                        logger.warning(
                            "Not all base regions share the same weather years. "
                            "Only weather years common to all regions will be used: %s",
                            sorted(common_wy),
                        )

                max_year = max(common_years)
                s_dem = (
                    f"SELECT {region_col} AS region, SUM(load_mw) AS total_demand "
                    f"FROM {demand_table} "
                    f"WHERE {region_col} IN ({placeholders_wr}) "
                    f"AND year = {max_year} "
                    f"GROUP BY region"
                )
                demand_df = pd.read_sql_query(
                    sql=s_dem, con=pg_engine, params=regions_with_data
                )
                region_demand = demand_df.set_index("region")["total_demand"]

    # ------------------------------------------------------------------ #
    # Calculate intra-regional MST costs for aggregated model regions      #
    # ------------------------------------------------------------------ #
    intra_capital: Dict[str, float] = {}
    intra_annuity: Dict[str, float] = {}
    intra_loss: Dict[str, float] = {}

    for mr in model_regions:
        bases = get_base_regions(mr)

        if len(bases) <= 1:
            # Single-base model region: no intra-regional expansion needed
            intra_capital[mr] = 0.0
            intra_annuity[mr] = 0.0
            intra_loss[mr] = 0.0
            continue

        # Build a graph of direct connections within this model region
        G = nx.Graph()
        G.add_nodes_from(bases)

        sub = tx_costs[
            tx_costs["region_from"].isin(bases) & tx_costs["region_to"].isin(bases)
        ]
        for _, row in sub.iterrows():
            # Use the minimum weight between duplicate (A,B) and (B,A) entries
            u, v = row["region_from"], row["region_to"]
            edge_weight = row[mst_weight_col]
            if G.has_edge(u, v):
                if edge_weight < G[u][v]["weight"]:
                    G[u][v].update(
                        weight=edge_weight,
                        capital_cost=row.get("capital_cost_mw", 0.0)
                        if has_capital
                        else 0.0,
                        annum_cost=row.get("annum_cost_mw", 0.0)
                        if has_annuity
                        else 0.0,
                        loss=row["line_loss_frac"],
                    )
            else:
                G.add_edge(
                    u,
                    v,
                    weight=edge_weight,
                    capital_cost=row.get("capital_cost_mw", 0.0)
                    if has_capital
                    else 0.0,
                    annum_cost=row.get("annum_cost_mw", 0.0) if has_annuity else 0.0,
                    loss=row["line_loss_frac"],
                )

        if not nx.is_connected(G):
            logger.warning(
                "Base regions in model region '%s' are not fully connected in the "
                "cost table '%s'. Intra-regional cost for this region will be zero.",
                mr,
                cost_table,
            )
            intra_capital[mr] = 0.0
            intra_annuity[mr] = 0.0
            intra_loss[mr] = 0.0
            continue

        mst = nx.minimum_spanning_tree(G, weight="weight")

        # Assign raw weights = sum of endpoint demands; then normalize
        raw_edge_weights: Dict[tuple, float] = {}
        for u, v in mst.edges():
            d_u = float(region_demand.get(u, 0.0))
            d_v = float(region_demand.get(v, 0.0))
            raw_edge_weights[(u, v)] = d_u + d_v

        total_raw = sum(raw_edge_weights.values())
        if total_raw == 0.0:
            logger.warning(
                "Total demand across base regions in model region '%s' is zero. "
                "Uniform edge weights will be used instead.",
                mr,
            )
            n_edges = len(raw_edge_weights)
            norm_weights = {k: 1.0 / n_edges for k in raw_edge_weights}
        else:
            norm_weights = {k: v / total_raw for k, v in raw_edge_weights.items()}

        mr_capital = 0.0
        mr_annuity = 0.0
        mr_loss = 0.0
        for (u, v), w in norm_weights.items():
            edge_data = mst[u][v]
            mr_capital += w * edge_data.get("capital_cost", 0.0)
            mr_annuity += w * edge_data.get("annum_cost", 0.0)
            mr_loss += w * edge_data["loss"]

        intra_capital[mr] = mr_capital
        intra_annuity[mr] = mr_annuity
        intra_loss[mr] = mr_loss

    # ------------------------------------------------------------------ #
    # Assemble results for each connected model region pair                #
    # ------------------------------------------------------------------ #
    results = []

    for pair in sorted(connected_pairs):
        mr_a, mr_b = pair
        bases_a = get_base_regions(mr_a)
        bases_b = get_base_regions(mr_b)

        # All direct connections between bases of mr_a and bases of mr_b
        direct = tx_costs[
            (
                tx_costs["region_from"].isin(bases_a)
                & tx_costs["region_to"].isin(bases_b)
            )
            | (
                tx_costs["region_from"].isin(bases_b)
                & tx_costs["region_to"].isin(bases_a)
            )
        ]

        if direct.empty:
            logger.warning(
                "No entry found in cost table '%s' for the direct connection between "
                "model regions '%s' and '%s'. This pair will be skipped.",
                cost_table,
                mr_a,
                mr_b,
            )
            continue

        # Select the lowest-cost direct connection
        min_idx = direct[mst_weight_col].idxmin()
        min_row = direct.loc[min_idx]

        direct_capital = float(min_row["capital_cost_mw"]) if has_capital else 0.0
        direct_annuity = float(min_row["annum_cost_mw"]) if has_annuity else 0.0
        direct_loss = float(min_row["line_loss_frac"])
        dollar_year = (
            int(min_row["dollar_year"])
            if "dollar_year" in min_row.index
            else None
        )

        total_capital = direct_capital + intra_capital[mr_a] + intra_capital[mr_b]
        total_annuity = direct_annuity + intra_annuity[mr_a] + intra_annuity[mr_b]
        total_loss = direct_loss + intra_loss[mr_a] + intra_loss[mr_b]

        result: dict = {
            "start_region": mr_a,
            "dest_region": mr_b,
            "total_line_loss_frac": total_loss,
        }
        if has_capital:
            result["total_interconnect_cost_mw"] = total_capital
        if has_annuity:
            result["total_interconnect_annuity_mw"] = total_annuity
        if dollar_year is not None:
            result["dollar_year"] = dollar_year

        results.append(result)

    return pd.DataFrame(results)


def haversine(lon1, lat1, lon2, lat2, units="mile"):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    https://gis.stackexchange.com/questions/166820/geopandas-return-lat-and-long-of-a-centroid-point
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))

    if units == "mile":
        r = 3956  # Radius of earth in miles. Use 6371 for kilometers, 3956 for miles
    elif units == "km":
        r = 6371
    else:
        raise ValueError(f"Units are {units}, but should be 'mile' or 'km'")

    return c * r


def getXY(pt):
    "Return the X and Y parts of a coordinate point"
    return (pt.x, pt.y)


def single_line_distance(line_name, region_centroids, units):
    """Calculate the transmission line distance between centroids of two regions.

    Parameters
    ----------
    line_name : str
        Two region names in the format <start>_to_<end>
    region_centroids : geoseries
        Centroid points of each region with region names as the index
    units : str
        Name of the distance units to use. Options are 'mile' or 'km'.

    Returns
    -------
    float
        The distance
    """

    start, end = line_name.split("_to_")
    start_lon, start_lat = getXY(region_centroids[start])
    end_lon, end_lat = getXY(region_centroids[end])
    distance = haversine(start_lon, start_lat, end_lon, end_lat, units=units)

    return distance


def transmission_line_distance(
    trans_constraints_df, ipm_shapefile, settings, units="mile"
):
    logger.debug("Calculating transmission line distance")
    ipm_shapefile["geometry"] = ipm_shapefile.buffer(0.01)
    model_polygons = ipm_shapefile.dissolve(by="model_region")
    model_polygons = model_polygons.to_crs(epsg=4326)
    region_centroids = find_centroid(model_polygons)

    distances = [
        single_line_distance(line_name, region_centroids, units=units)
        for line_name in trans_constraints_df["transmission_path_name"]
    ]
    trans_constraints_df[f"distance_{units}"] = distances
    trans_constraints_df[f"distance_{units}"] = trans_constraints_df[
        f"distance_{units}"
    ]

    return trans_constraints_df
