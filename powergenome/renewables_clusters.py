import glob
import itertools
import json
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy

WEIGHT = "gw"
MEANS = [
    "lcoe",
    "interconnect_annuity",
    "offshore_spur_miles",
    "spur_miles",
    "tx_miles",
    "site_substation_spur_miles",
    "substation_metro_tx_miles",
    "site_metro_spur_miles",
]
SUMS = ["area", "gw"]
PROFILE_KEYS = ["cbsa_id", "cluster_level", "cluster"]
HOURS_IN_YEAR = 8784


class ClusterBuilder:
    """
    Builds clusters of resources.

    Attributes
    ----------
    groups : list of dict
        Resource group metadata.
        - `metadata` (str): Relative path to resource metadata file.
        - `profiles` (str): Relative path to variable resource profiles file.
        - `technology` (str): Resource type.
        - ... and any additional (optional) keys.
    clusters : list of dict
        Resource clusters.
        - `group` (dict): Resource group from :attr:`groups`.
        - `kwargs` (dict): Arguments used to uniquely identify the group.
        - `region` (str): Model region in which the clustering was performed.
        - `clusters` (pd.DataFrame): Computed resource clusters.
        - `profiles` (np.ndarray): Computed profiles for the resource clusters.
    """

    def __init__(self, path: str = ".") -> None:
        """
        Initialize with resource group metadata.

        Arguments
        ---------
        path
            Path to the directory containing the metadata files ('*_group.json').

        Raises
        ------
        ValueError
            Group metadata missing required keys.
        """
        self.groups = load_groups(path)
        required = ("metadata", "profiles", "technology")
        for g in self.groups:
            missing = [k for k in required if k not in g]
            if missing:
                raise ValueError(f"Group metadata missing required keys {missing}: {g}")
            g["metadata"] = os.path.abspath(os.path.join(path, g["metadata"]))
            g["profiles"] = os.path.abspath(os.path.join(path, g["profiles"]))
        self.clusters: List[dict] = []

    def _test_clusters_exist(self) -> None:
        if not self.clusters:
            raise ValueError("No clusters have been built")

    def find_groups(self, **kwargs: Any) -> List[dict]:
        """
        Return the groups matching the specified arguments.

        Arguments
        ---------
        **kwargs
            Arguments to match against group metadata.
        """
        return [
            g
            for g in self.groups
            if all(k in g and g[k] == v for k, v in kwargs.items())
        ]

    def build_clusters(
        self,
        region: str,
        ipm_regions: Sequence[str],
        min_capacity: float = None,
        max_clusters: int = None,
        cap_multiplier: float = None,
        **kwargs: Any,
    ) -> None:
        """
        Build and append resource clusters to the collection.

        This method can be called as many times as desired before generating outputs.

        Arguments
        ---------
        region
            Model region (used only to label results).
        ipm_regions
            IPM regions in which to select resources.
        min_capacity
            Minimum total capacity (GW). Resources are selected,
            from lowest to highest levelized cost of energy (lcoe),
            until the minimum capacity is just exceeded.
            If `None`, all resources are selected for clustering.
        max_clusters
            Maximum number of resource clusters to compute.
            If `None`, no clustering is performed; resources are returned unchanged.
        cap_multiplier
            Capacity multiplier applied to resource metadata.
        **kwargs
            Arguments to :meth:`get_groups` for selecting the resource group.

        Raises
        ------
        ValueError
            Arguments match multiple resource groups.
        """
        groups = self.find_groups(**kwargs)
        if len(groups) > 1:
            raise ValueError(f"Arguments match multiple resource groups: {groups}")
        c: Dict[str, Any] = {}
        c["group"] = groups[0]
        c["kwargs"] = kwargs
        c["region"] = region
        metadata = load_metadata(c["group"]["metadata"], cap_multiplier=cap_multiplier)
        c["clusters"] = build_clusters(
            metadata,
            ipm_regions=ipm_regions,
            min_capacity=min_capacity,
            max_clusters=max_clusters,
        )
        c["profiles"] = build_cluster_profiles(
            c["group"]["profiles"], c["clusters"], metadata
        )
        self.clusters.append(c)

    def get_cluster_metadata(self) -> Optional[pd.DataFrame]:
        """
        Return computed cluster metadata.

        The following fields are added:
        - `region` (str): Region label passed to :meth:`build_clusters`.
        - `technology` (str): From resource metadata 'technology'.
        - `cluster` (int): Unique identifier for each cluster to link to other outputs.

        The following fields are renamed:
        - `max_capacity`: From 'gw'.
        - `area_km2`: From 'area'.

        Raises
        ------
        ValueError
            No clusters have yet been computed.
        """
        self._test_clusters_exist()
        dfs = []
        start = 0
        for c in self.clusters:
            df = c["clusters"].reset_index()
            columns = [x for x in np.unique([WEIGHT] + MEANS + SUMS) if x in df]
            n = len(df)
            df = (
                df[columns]
                .assign(
                    region=c["region"],
                    technology=c["group"]["technology"],
                    cluster=np.arange(start, start + n),
                )
                .rename(columns={"gw": "max_capacity", "area": "area_km2"})
            )
            dfs.append(df)
            start += n
        return pd.concat(dfs, axis=0, ignore_index=True, sort=False)

    def get_cluster_profiles(self) -> Optional[pd.DataFrame]:
        """
        Return computed cluster profiles.

        Uses multi-index columns ('region', 'Resource', 'cluster'), where
        - `region` (str): Region label passed to :meth:`build_clusters`.
        - `Resource` (str): From resource metadata 'technology'.
        - `cluster` (int): Unique identifier for each cluster to link to other outputs.

        The first column is the hour number.
        Subsequent columns are the cluster profiles.

        Raises
        ------
        ValueError
            No clusters have yet been computed.
        """
        self._test_clusters_exist()
        profiles = np.column_stack([c["profiles"] for c in self.clusters])
        columns = []
        start = 0
        for c in self.clusters:
            for _ in range(c["profiles"].shape[1]):
                columns.append((c["region"], c["group"]["technology"], start))
                start += 1
        df = pd.DataFrame(profiles, columns=columns)
        # Insert hour numbers into first column
        df.insert(
            loc=0,
            column=("region", "Resource", "cluster"),
            value=np.arange(HOURS_IN_YEAR),
        )
        return df


def load_groups(path: str = ".") -> List[dict]:
    """Load group metadata."""
    paths = glob.glob(os.path.join(path, "*_group.json"))
    groups = []
    for p in paths:
        with open(p, mode="r") as fp:
            groups.append(json.load(fp))
    return groups


def load_metadata(path: str, cap_multiplier: float = None) -> pd.DataFrame:
    """Load resource metadata."""
    df = pd.read_csv(path)
    if cap_multiplier:
        df["gw"] = df["gw"] * cap_multiplier
    df.set_index("id", drop=False, inplace=True)
    return df


def read_parquet(
    path: str, columns: Sequence[str] = None, filters: Sequence[dict] = None
) -> pd.DataFrame:
    """Read parquet file."""
    # NOTE: Due to pyarrow bug, cannot pass more than two dictionaries in `filters`.
    dnf = None
    if filters:
        # Convert to disjunctive normal form (dnf)
        dnf = [[(str(k), "=", v) for k, v in d.items()] for d in filters]
    # https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetDataset.html
    return pd.read_parquet(
        path,
        engine="pyarrow",
        columns=columns,
        filters=dnf,
        # Needed for unpartitioned datasets
        use_legacy_dataset=False,
        memory_map=True,
    )


def build_clusters(
    metadata: pd.DataFrame,
    ipm_regions: Sequence[str],
    min_capacity: float = None,
    max_clusters: int = None,
) -> pd.DataFrame:
    """Build resource clusters."""
    if max_clusters is None:
        max_clusters = np.inf
    if max_clusters < 1:
        raise ValueError("Max number of clusters must be greater than zero")
    df = metadata
    cdf = _get_base_clusters(df, ipm_regions).sort_values("lcoe")
    if cdf.empty:
        raise ValueError(f"No resources found in {ipm_regions}")
    if min_capacity:
        # Drop clusters with highest LCOE until min_capacity reached
        end = cdf["gw"].cumsum().searchsorted(min_capacity) + 1
        if end > len(cdf):
            raise ValueError(
                f"Capacity in {ipm_regions} ({cdf['gw'].sum()} GW) less than minimum ({min_capacity} GW)"
            )
        cdf = cdf[:end]
    # Track ids of base clusters through aggregation
    cdf["ids"] = [[x] for x in cdf["id"]]
    # Aggregate clusters within each metro area (cbsa_id)
    while len(cdf) > max_clusters:
        # Sort parents by lowest LCOE distance of children
        diff = lambda x: abs(x.max() - x.min())
        parents = (
            cdf.groupby("parent_id", sort=False)
            .agg(child_ids=("id", list), n=("id", "count"), lcoe=("lcoe", diff))
            .sort_values(["n", "lcoe"], ascending=[False, True])
        )
        if parents.empty:
            break
        if parents["n"].iloc[0] == 2:
            # Choose parent with lowest LCOE
            best = parents.iloc[0]
            # Compute parent
            parent = pd.Series(
                _merge_children(
                    cdf.loc[best["child_ids"]],
                    ids=_flat(*cdf.loc[best["child_ids"], "ids"]),
                    **df.loc[best.name],
                )
            )
            # Add parent
            cdf.loc[best.name] = parent
            # Drop children
            cdf.drop(best["child_ids"], inplace=True)
        else:
            # Promote child with deepest parent
            parent_id = df.loc[parents.index, "cluster_level"].idxmax()
            parent = df.loc[parent_id]
            child_id = parents.loc[parent_id, "child_ids"][0]
            # Update child
            columns = ["id", "parent_id", "cluster_level"]
            cdf.loc[child_id, columns] = parent[columns]
            # Update index
            cdf.rename(index={child_id: parent_id}, inplace=True)
    # Keep only computed columns
    columns = _flat(MEANS, SUMS, "ids")
    columns = [col for col in columns if col in cdf.columns]
    cdf = cdf[columns]
    cdf.reset_index(inplace=True, drop=True)
    if len(cdf) > max_clusters:
        # Aggregate singleton metro area clusters
        Z = scipy.cluster.hierarchy.linkage(cdf[["lcoe"]].values, method="ward")
        mask = [True] * len(cdf)
        for child_idx in Z[:, 0:2].astype(int):
            mask[child_idx] = False
            parent = _merge_children(
                cdf.loc[child_idx], ids=_flat(*cdf.loc[child_idx, "ids"])
            )
            cdf.loc[len(cdf)] = parent
            mask.append(True)
            if not sum(mask) > max_clusters:
                break
        cdf = cdf[mask]
    return cdf


def build_cluster_profiles(
    path: str, clusters: pd.DataFrame, metadata: pd.DataFrame
) -> np.ndarray:
    """Build cluster profiles."""
    results = np.zeros((HOURS_IN_YEAR, len(clusters)), dtype=float)
    for i, cids in enumerate(clusters["ids"]):
        weights = metadata.loc[cids, WEIGHT].values
        weights /= weights.sum()
        for j, cid in enumerate(cids):
            # Include ipm_region (partitioning column) to speed up filter
            filters = (
                metadata.loc[[cid], ["ipm_region"] + PROFILE_KEYS]
                .reset_index(drop=True)
                .to_dict("records")
            )
            # Assumes profile is already sorted by hour (ascending)
            df = read_parquet(path, filters=filters, columns=["capacity_factor"])
            results[:, i] += df["capacity_factor"].values * weights[j]
    return results


def _get_base_clusters(df: pd.DataFrame, ipm_regions: Sequence[str]) -> pd.DataFrame:
    return (
        df[df["ipm_region"].isin(ipm_regions)]
        .groupby("cbsa_id")
        .apply(lambda g: g[g["cluster_level"] == g["cluster_level"].max()])
        .reset_index(level=["cbsa_id"], drop=True)
    )


def _merge_children(df: pd.DataFrame, **kwargs: Any) -> dict:
    parent = kwargs
    for key in SUMS:
        parent[key] = df[key].sum()
    for key in [k for k in MEANS if k in df.columns]:
        parent[key] = (df[key] * df[WEIGHT]).sum() / df[WEIGHT].sum()
    return parent


def _flat(*args: Sequence) -> list:
    lists = [x if np.iterable(x) and not isinstance(x, str) else [x] for x in args]
    return list(itertools.chain(*lists))
