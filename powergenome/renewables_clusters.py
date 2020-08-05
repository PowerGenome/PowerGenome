import glob
import itertools
import json
import logging
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow.csv
import pyarrow.parquet as pq
import scipy.cluster.hierarchy

logger = logging.getLogger(__name__)

CAPACITY = "mw"
WEIGHT = CAPACITY
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
SUMS = ["area", CAPACITY]
PROFILE_KEYS = ["metro_id", "cluster_level", "cluster"]
HOURS_IN_YEAR = 8784
NREL_ATB_TECHNOLOGY_MAP = {
    ("utilitypv", None): {"technology": "utilitypv"},
    ("landbasedwind", None): {"technology": "landbasedwind"},
    ("offshorewind", None): {"technology": "offshorewind"},
    **{
        ("offshorewind", f"otrg{x}"): {
            "technology": "offshorewind",
            "turbine_type": "fixed",
        }
        for x in range(1, 6)
    },
    **{
        ("offshorewind", f"otrg{x}"): {
            "technology": "offshorewind",
            "turbine_type": "floating",
        }
        for x in range(6, 16)
    },
}


def _normalize(x: Optional[str]) -> Optional[str]:
    """
    Normalize string to lowercase and no whitespace.

    Examples
    --------
    >>> _normalize('Offshore Wind')
    'offshorewind'
    >>> _normalize('OffshoreWind')
    'offshorewind'
    >>> _normalize(None) is None
    True
    """
    if not x:
        return x
    return re.sub(r"\s+", "", x.lower())


def map_nrel_atb_technology(tech: str, detail: str = None) -> Dict[str, Any]:
    """
    Map NREL ATB technology to resource groups.

    Arguments
    ---------
    tech
        Technology.
    detail
        Technology detail.

    Returns
    -------
    dict
        Key, value pairs identifying one or more resource groups.

    Examples
    --------
    >>> map_nrel_atb_technology('UtilityPV', 'LosAngeles')
    {'technology': 'utilitypv'}
    >>> map_nrel_atb_technology('LandbasedWind', 'LTRG1')
    {'technology': 'landbasedwind'}
    >>> map_nrel_atb_technology('OffShoreWind')
    {'technology': 'offshorewind'}
    >>> map_nrel_atb_technology('OffShoreWind', 'OTRG3')
    {'technology': 'offshorewind', 'turbine_type': 'fixed'}
    >>> map_nrel_atb_technology('OffShoreWind', 'OTRG7')
    {'technology': 'offshorewind', 'turbine_type': 'floating'}
    >>> map_nrel_atb_technology('Unknown')
    {}
    """
    tech = _normalize(tech)
    detail = _normalize(detail)
    group = {}
    for k, v in NREL_ATB_TECHNOLOGY_MAP.items():
        if (tech == k[0] or not k[0]) and (detail == k[1] or not k[1]):
            group.update(v)
    return group


class Table:
    """
    Cached interface for tabular data.

    Supports parquet and csv formats.

    Parameters
    ----------
    path
        Path to dataset.
    df
        In-memory dataframe.

    Attributes
    ----------
    path : Union[str, os.PathLike]
        Path to the dataset.
    df : pd.DataFrame
        Cached dataframe.
    format : str
        Dataset format ('parquet' or 'csv'), or `None` if in-memory only.
    columns : Iterable[Union[str, int]]
        Dataset column names.

    Raises
    ------
    ValueError
        Missing either path or dataframe.

    Examples
    --------
    In-memory dataframe:

    >>> df = pd.DataFrame({'id': [1, 2], 'x': [10, 20]})
    >>> table = Table(df = df)
    >>> table.format is None
    True
    >>> table.columns
    ['id', 'x']
    >>> table.read()
       id   x
    0   1  10
    1   2  20
    >>> table.read(columns=['id'])
       id
    0   1
    1   2
    >>> table.clear()
    >>> table.df is not None
    True

    File dataset (csv):

    >>> import tempfile
    >>> fp = tempfile.NamedTemporaryFile()
    >>> df.to_csv(fp.name, index=False)
    >>> table = Table(path = fp.name)
    >>> table.format
    'csv'
    >>> table.columns
    ['id', 'x']
    >>> table.read(cache=False)
       id   x
    0   1  10
    1   2  20
    >>> table.df is None
    True
    >>> table.read(columns=['id'], cache=True)
       id
    0   1
    1   2
    >>> table.df is not None
    True
    >>> table.clear()
    >>> table.df is None
    True
    >>> fp.close()
    """

    def __init__(
        self, path: Union[str, os.PathLike] = None, df: pd.DataFrame = None
    ) -> None:
        self.path = path
        self.df = df
        self.format = None
        self._dataset = None
        self._columns = None
        if path is not None:
            try:
                self._dataset = pq.ParquetDataset(path)
                self._columns = self._dataset.schema.names
                self.format = "parquet"
            except pyarrow.lib.ArrowInvalid:
                # Assume CSV file
                self.format = "csv"
        if path is None and df is None:
            raise ValueError("Mising either path to tabular data or a pandas DataFrame")

    @property
    def columns(self) -> List[str]:
        if self.df is not None:
            return list(self.df.columns)
        if self._columns is None:
            if self.format == "csv":
                self._columns = pd.read_csv(self.path, nrows=0).columns
        return list(self._columns)

    def read(
        self, columns: Iterable[Union[str, int]] = None, cache: bool = None
    ) -> pd.DataFrame:
        """
        Read data from memory or from disk.

        Parameters
        ----------
        columns
            Names of column to read. If `None`, all columns are read. 
        cache
            Whether to cache the full dataset in memory. If `None`,
            the dataset is cached if `columns` is `None`, and not otherwise.

        Returns
        -------
        pd.DataFrame
            Data as a dataframe.
        """
        if self.df is not None:
            return self.df[columns] if columns is not None else self.df
        if cache is None:
            cache = columns is None
        read_columns = None if cache else columns
        if self.format == "csv":
            df = pd.read_csv(self.path, usecols=read_columns)
        elif self.format == "parquet":
            df = self._dataset.read(columns=read_columns).to_pandas()
        if cache:
            self.df = df
        return df[columns] if columns is not None else df

    def clear(self) -> None:
        """
        Clear the dataset cache.

        Only applies if :attr:`path` is set so that the dataset can be reread from file.
        """
        if self.path is not None:
            self.df = None


class ResourceGroup:
    """
    Group of resources sharing common attributes.

    Parameters
    ----------
    group
        Group metadata.

        - `technology` : str
          Resource type ('utilitypv', 'landbasedwind', or 'offshorewind').
        - `existing` : bool
          Whether resources are new (`False`, default) or existing (`True`).
        - `clustered` : str, optional
          The name of the resource metadata attribute by
          which to differentiate between multiple precomputed hierarchical trees.
          Defaults to `None` (resource group does not represent hierarchical trees).
        - `metadata` : str, optional
          Relative path to resource metadata dataset (optional if `metadata` is `None`).
        - `profiles` : str, optional
          Relative path to resource profiles dataset (optional if `profiles` is `None`).
        - ... and any additional (optional) keys.

    metadata
        Resource metadata, with one resource per row.

        - `id`: int
          Resource identifier, unique within the group.
        - `ipm_region` : str
          IPM region to which the resource delivers power.
        - `mw` : float
          Maximum resource capacity in MW.
        - `lcoe` : float, optional
          Levelized cost of energy, used to guide the selection
          (from lowest to highest) and clustering (by nearest) of resources.
          If missing, selection and clustering is by largest and nearest `mw`.

        Resources representing hierarchical trees (see `group.clustered`)
        require additional attributes.
    
        - `parent_id` : int
          Identifier of the resource formed by clustering this resource with the one
          other resource with the same `parent_id`.
          Only resources with `cluster_level` of 1 have no `parent_id`.
        - `cluster_level` : int
          Cluster level where the resource first appears, from `m`
          (the number of resources at the base of the tree), to 1.
        - `[group.clustered]` : Any
          Each unique value of this grouping attribute represents a precomputed
          hierarchical tree. When clustering resources, every tree is traversed to its
          crown before the singleton resources from the trees are clustered together.
        
        The following resource attributes (all float) are propagaged as:

        - weighted means (weighted by `mw`):

            - `lcoe`
            - `interconnect_annuity`
            - `tx_miles`
            - `spur_miles`
            - `offshore_spur_miles`
            - `site_substation_spur_miles`
            - `substation_metro_tx_miles`
            - `site_metro_spur_miles`
        
        - sums:

            - `mw`
            - `area`
    
    profiles
        Variable resource capacity profiles with normalized capacity factors
        (from 0 to 1) for every hour of the year (either 8760 or 8784 for a leap year).
        Each profile must be a column whose name matches the resource `metadata.id`.
    path
        Directory relative to which the file paths `group.metadata` and `group.profiles`
        should be read.

    Attributes
    ----------
    group : Dict[str, Any]
    metadata : Table
        Cached interface to resource metadata.
    profiles : Table
        Cached interface to resource profiles.

    Examples
    --------
    >>> group = {'technology': 'utilitypv'}
    >>> metadata = pd.DataFrame({'id': [0, 1], 'ipm_region': ['A', 'A'], 'mw': [1, 2]})
    >>> profiles = pd.DataFrame({0: np.random.rand(8784), 1: np.random.rand(8784)})
    >>> rg = ResourceGroup(group, metadata, profiles)
    >>> rg.test_metadata()
    >>> rg.test_profiles()
    """

    def __init__(
        self,
        group: Dict[str, Any],
        metadata: pd.DataFrame = None,
        profiles: pd.DataFrame = None,
        path: str = ".",
    ) -> None:
        self.group = {"existing": False, "clustered": None, **group.copy()}
        for key in ["metadata", "profiles"]:
            if self.group.get(key):
                # Convert relative paths (relative to group file) to absolute paths
                self.group[key] = os.path.abspath(os.path.join(path, self.group[key]))
        required = ["technology"]
        if metadata is None:
            required.append("metadata")
        if profiles is None:
            required.append("profiles")
        missing = [key for key in required if not self.group.get(key)]
        if missing:
            raise ValueError(
                f"Group metadata missing required keys {missing}: {self.group}"
            )
        self.metadata = Table(df=metadata, path=self.group.get("metadata"))
        self.profiles = Table(df=profiles, path=self.group.get("profiles"))

    @classmethod
    def from_json(cls, path: Union[str, os.PathLike]) -> "ResourceGroup":
        """
        Build from JSON file.

        Parameters
        ----------
        path
            Path to JSON file.
        """
        with open(path, mode="r") as fp:
            group = json.load(fp)
        return cls(group, path=os.path.dirname(path))

    def test_metadata(self) -> None:
        """
        Test that `:attr:metadata` is valid.

        Raises
        ------
        ValueError
            Resource metadata missing required keys.
        """
        columns = self.metadata.columns
        required = ["ipm_region", "id", "mw"]
        if self.group.get("clustered"):
            required.extend(["parent_id", "cluster_level", self.group["clustered"]])
        missing = [key for key in required if key not in columns]
        if missing:
            raise ValueError(f"Resource metadata missing required keys {missing}")

    def test_profiles(self) -> None:
        """
        Test that `:attr:profiles` is valid.

        Raises
        ------
        ValueError
            Resource profiles column names do not match resource identifiers.
        ValueError
            Resource profiles are not either 8760 or 8784 elements.
        """
        ids = self.metadata.read(columns=["id"])["id"]
        columns = self.profiles.columns
        if not set(columns) == set(ids):
            raise ValueError(
                f"Resource profiles column names do not match resource identifiers"
            )
        df = self.profiles.read(columns=columns[0])
        if len(df) not in [8760, 8784]:
            raise ValueError(f"Resource profiles are not either 8760 or 8784 elements")


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
        FileNotFoundError
            No group metadata files found.
        ValueError
            Group metadata missing required keys.
        """
        self.groups = load_groups(path)
        if not self.groups:
            raise FileNotFoundError(f"No group metadata files found in {path}")
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
        ipm_regions: Iterable[str] = None,
        min_capacity: float = None,
        max_clusters: int = None,
        max_lcoe: float = None,
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
            If `None`, all IPM regions are selected.
        min_capacity
            Minimum total capacity (MW). Resources are selected,
            from lowest to highest levelized cost of energy (lcoe),
            until the minimum capacity is just exceeded.
            If `None`, all resources are selected for clustering.
        max_clusters
            Maximum number of resource clusters to compute.
            If `None`, no clustering is performed; resources are returned unchanged.
        max_lcoe
            Select only the resources with a levelized cost of electricity (lcoe)
            below this maximum. Takes precedence over `min_capacity`.
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
            max_lcoe=max_lcoe,
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

        Raises
        ------
        ValueError
            No clusters have yet been computed.
        """
        self._test_clusters_exist()
        dfs = []
        for c in self.clusters:
            df = c["clusters"].reset_index()
            columns = [x for x in np.unique([WEIGHT] + MEANS + SUMS) if x in df]
            df = df[columns].assign(region=c["region"])
            dfs.append(df)
        return pd.concat(dfs, axis=0, ignore_index=True, sort=False)

    def get_cluster_profiles(self) -> np.ndarray:
        """
        Return computed cluster profiles.

        Returns
        -------
        np.ndarray
            Hourly normalized (0-1) generation profiles (n clusters, m hours).

        Raises
        ------
        ValueError
            No clusters have yet been computed.
        """
        self._test_clusters_exist()
        return np.row_stack([c["profiles"] for c in self.clusters])


def _tuple(x: Any) -> tuple:
    """
    Cast object to tuple.

    Examples
    --------
    >>> _tuple(1)
    (1,)
    >>> _tuple([1])
    (1,)
    >>> _tuple('string')
    ('string',)
    """
    if np.iterable(x) and not isinstance(x, str):
        return tuple(x)
    return (x,)


def merge_rows(
    df: pd.DataFrame, sums: Iterable = None, means: Iterable = None, weight=None
) -> dict:
    """
    Merge all rows in dataframe into one.

    Parameters
    ----------
    df
        Rows to merge.
    sums
        Names of columns to sum.
    means
        Names of columns to average.
    weight
        Name of column to use for weighted averages.
        If `None`, averages are not weighted.

    Returns
    -------
    dict
        Merged row as a dictionary.

    Examples
    --------
    >>> df = pd.DataFrame({'mw': [1, 2], 'area': [10, 20], 'lcoe': [0.1, 0.4]})
    >>> merge_rows(df, sums=['area', 'mw'], means=['lcoe'], weight='mw')
    {'area': 30, 'mw': 3, 'lcoe': 0.3}
    >>> merge_rows(df, sums=['area', 'mw'], means=['lcoe'])
    {'area': 30, 'mw': 3, 'lcoe': 0.25}
    """
    merge = {}
    if sums is not None:
        merge.update(df[sums].sum())
    if means is not None:
        if weight:
            merge.update(
                df[means].multiply(df[weight], axis=0).sum() / df[weight].sum()
            )
        else:
            merge.update(df[means].mean())
    return merge


def cluster_rows(
    df: pd.DataFrame, by: Iterable[Iterable], max_rows: int = 1, **kwargs: Any
) -> pd.DataFrame:
    """
    Merge rows in dataframe by hierarchical clustering.

    Uses the Ward variance minimization algorithm to incrementally merge rows.
    See :func:`scipy.cluster.hierarchy.linkage`.

    Parameters
    ----------
    df
        Rows to merge (m, ...).
    by
        2-dimensional array of observation vectors (m, ...) from which to compute
        distances between each row pair.
    max_rows
        Number of rows at which to stop merging rows.
    **kwargs
        Optional parameters to :func:`merge_rows`.

    Returns
    -------
    pd.DataFrame
        Merged rows as a dataframe.
        Their indices are tuples of the original row indices from which they were built.
        If original indices were already iterables, they are merged
        (e.g. (1, 2) and (3, ) becomes (1, 2, 3)).

    Raises
    ------
    ValueError
        Max number of rows must be greater than zero.

    Examples
    --------
    With the default (range) row index:

    >>> df = pd.DataFrame({'mw': [1, 2, 3], 'area': [4, 5, 6], 'lcoe': [0.1, 0.4, 0.2]})
    >>> kwargs = {'sums': ['area', 'mw'], 'means': ['lcoe'], 'weight': 'mw'}
    >>> cluster_rows(df, by=df[['lcoe']], max_rows=len(df), **kwargs)
          mw  area  lcoe
    (0,)   1     4   0.1
    (1,)   2     5   0.4
    (2,)   3     6   0.2
    >>> cluster_rows(df, by=df[['lcoe']], max_rows=2, **kwargs)
             mw  area   lcoe
    (1,)    2.0   5.0  0.400
    (0, 2)  4.0  10.0  0.175

    With a custom row index:

    >>> df.index = ['a', 'b', 'c']
    >>> cluster_rows(df, by=df[['lcoe']], max_rows=2, **kwargs)
             mw  area   lcoe
    (b,)    2.0   5.0  0.400
    (a, c)  4.0  10.0  0.175

    With an iterable row index:

    >>> df.index = [(1, 2), (4, ), (3, )]
    >>> cluster_rows(df, by=df[['lcoe']], max_rows=2, **kwargs)
                mw  area   lcoe
    (4,)       2.0   5.0  0.400
    (1, 2, 3)  4.0  10.0  0.175
    """
    if max_rows < 1:
        raise ValueError("Max number of rows must be greater than zero")
    index = [_tuple(x) for x in df.index]
    df = df.reset_index(drop=True)
    nrows = len(df)
    drows = nrows - max_rows
    if drows < 1:
        df.index = index
        return df
    # Preallocate new rows
    df = df.reindex(pd.Index(pd.RangeIndex(stop=nrows + drows)))
    Z = scipy.cluster.hierarchy.linkage(by, method="ward")
    mask = [True] * nrows
    for i, link in enumerate(Z[:drows, 0:2].astype(int)):
        mask[link[0]] = False
        mask[link[1]] = False
        df.loc[nrows + i] = pd.Series(merge_rows(df.loc[link], **kwargs))
        index.append(index[link[0]] + index[link[1]])
        mask.append(True)
    df = df[mask]
    df.index = [idx for m, idx in zip(mask, index) if m]
    return df


def cluster_row_trees(
    df: pd.DataFrame, by: str, max_rows: int = 1, tree: str = None, **kwargs: Any
) -> pd.DataFrame:
    """
    Merge rows in a dataframe following precomputed hierarchical trees.

    Parameters
    ----------
    df
        Rows to merge.
        Must have columns `parent_id` (matching values in index), `cluster_level`, and
        the columns named in **by** and **tree**.
    by
        Name of column to use for determining merge order.
        Children with the smallest pairwise distance on this column are merged first.
    max_rows
        Number of rows at which to stop merging rows.
        If smaller than the number of trees, :func:`cluster_rows` is used to merge
        tree heads.
    tree
        Name of column to use for differentiating between hierarchical trees.
        If `None`, assumes rows represent a single tree.
    **kwargs
        Optional parameters to :func:`merge_rows`.

    Returns
    -------
    pd.DataFrame
        Merged rows as a dataframe.
        Their indices are tuples of the original row indices from which they were built.
        If original indices were already iterables, they are merged
        (e.g. (1, 2) and (3, ) becomes (1, 2, 3)).

    Raises
    ------
    ValueError
        Max number of rows must be greater than zero.
    ValueError
        Missing required fields.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'cluster_level': [3, 3, 3, 2, 1],
    ...     'parent_id': [3, 3, 4, 4, float('nan')],
    ...     'mw': [0.1, 0.1, 0.1, 0.2, 0.3]
    ... }, index=[0, 1, 2, 3, 4])
    >>> cluster_row_trees(df, by='mw', sums=['mw'], max_rows=2)
            cluster_level  parent_id   mw
    (2,)                3        4.0  0.1
    (0, 1)              2        4.0  0.2
    >>> cluster_row_trees(df, by='mw', sums=['mw'])
               cluster_level  parent_id   mw
    (2, 0, 1)              1        NaN  0.3
    """
    if max_rows < 1:
        raise ValueError("Max number of rows must be greater than zero")
    required = ["parent_id", "cluster_level", by]
    if tree:
        required.append(tree)
    missing = [key for key in required if key not in df]
    if missing:
        raise ValueError(f"Missing required fields {missing}")
    if tree:
        mask = df["cluster_level"] == df[tree].map(
            df.groupby(tree)["cluster_level"].max()
        )
    else:
        mask = df["cluster_level"] == df["cluster_level"].max()
    drows = mask.sum() - max_rows
    if drows < 1:
        df = df.copy()
        df.index = [_tuple(x) for x in df.index]
        return df
    df["_id"] = df.index
    df["_ids"] = [_tuple(x) for x in df.index]
    df["_mask"] = mask
    diff = lambda x: abs(x.max() - x.min())
    while drows > 0:
        # Sort parents by ascending distance of children
        # NOTE: Inefficient to recompute for all parents every time
        parents = (
            df[df["_mask"]]
            .groupby("parent_id", sort=False)
            .agg(ids=("_id", list), n=("_id", "count"), distance=(by, diff))
            .sort_values(["n", "distance"], ascending=[False, True])
        )
        if parents.empty:
            break
        if parents["n"].iloc[0] == 2:
            # Choose complete parent with lowest distance of children
            best = parents.iloc[0]
            # Compute parent
            parent = {
                # Initial attributes
                **df.loc[best.name],
                # Merged children attributes
                # NOTE: Needed only if a child is incomplete
                **merge_rows(df.loc[best["ids"]], **kwargs),
                # Indices of all past children
                "_ids": df.loc[best["ids"][0], "_ids"] + df.loc[best["ids"][1], "_ids"],
                "_mask": True,
            }
            # Add parent
            df.loc[best.name] = pd.Series(parent)
            # Drop children
            df.loc[best["ids"], "_mask"] = False
            # Decrement rows
            drows -= 1
        else:
            # Promote child with deepest parent
            parent_id = df.loc[parents.index, "cluster_level"].idxmax()
            child_id = parents.loc[parent_id, "ids"][0]
            # Update child
            columns = ["_id", "parent_id", "cluster_level"]
            df.loc[child_id, columns] = df.loc[parent_id, columns]
            # Update index
            df.rename(index={child_id: parent_id}, inplace=True)
            # Drop parent
            df.loc[parent_id, "_mask"] = False
    # Apply mask
    df = df[df["_mask"]]
    # Drop temporary columns
    df.index = df["_ids"].values
    df = df.drop(columns=["_id", "_ids", "_mask"])
    if len(df) > max_rows:
        df = cluster_rows(df, by=df[[by]], max_rows=max_rows, **kwargs)
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
        df[CAPACITY] = df[CAPACITY] * cap_multiplier
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
    ipm_regions: Iterable[str] = None,
    min_capacity: float = None,
    max_clusters: int = None,
    max_lcoe: float = None,
) -> pd.DataFrame:
    """Build resource clusters."""
    if max_clusters is None:
        max_clusters = np.inf
    if max_clusters < 1:
        raise ValueError("Max number of clusters must be greater than zero")
    df = metadata
    cdf = _get_base_clusters(df, ipm_regions)
    if ipm_regions is None:
        ipm_regions = cdf["ipm_regions"].unique().tolist()
    cdf = cdf.sort_values("lcoe")
    if min_capacity:
        # Drop clusters with highest LCOE until min_capacity reached
        end = min(len(cdf), cdf[CAPACITY].cumsum().searchsorted(min_capacity) + 1)
        cdf = cdf[:end]
    if max_lcoe:
        # Drop clusters with LCOE above the cutoff
        cdf = cdf[cdf["lcoe"] <= max_lcoe]
    if cdf.empty:
        raise ValueError(f"No resources found or selected in {ipm_regions}")
    capacity = cdf[CAPACITY].sum()
    if min_capacity and capacity < min_capacity:
        logger.warning(
            f"Selected capacity in {ipm_regions} ({capacity} MW) less than minimum ({min_capacity} MW)"
        )
    # Track ids of base clusters through aggregation
    cdf["ids"] = [[x] for x in cdf["id"]]
    # Aggregate clusters within each metro area (metro_id)
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
            mask[child_idx[0]], mask[child_idx[1]] = False, False
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
    results = np.zeros((len(clusters), HOURS_IN_YEAR), dtype=float)
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
            results[i] += df["capacity_factor"].values * weights[j]
    return results


def _get_base_clusters(
    df: pd.DataFrame, ipm_regions: Iterable[str] = None
) -> pd.DataFrame:
    if ipm_regions is not None:
        df = df[df["ipm_region"].isin(ipm_regions)]
    return (
        df.groupby("metro_id")
        .apply(lambda g: g[g["cluster_level"] == g["cluster_level"].max()])
        .reset_index(level=["metro_id"], drop=True)
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
