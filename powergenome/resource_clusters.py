import copy
import glob
import json
import os
import re
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Union

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq
import scipy.cluster.hierarchy

CAPACITY = "mw"
MERGE = {
    "sums": [CAPACITY, "area"],
    "means": [
        "lcoe",
        "interconnect_annuity",
        "offshore_spur_miles",
        "spur_miles",
        "tx_miles",
        "site_substation_spur_miles",
        "substation_metro_tx_miles",
        "site_metro_spur_miles",
        "m_popden",
    ],
    "weight": CAPACITY,
    "uniques": ["ipm_region", "metro_id"],
}
NREL_ATB_TECHNOLOGY_MAP = {
    ("utilitypv", None): {"technology": "utilitypv"},
    ("landbasedwind", None): {"technology": "landbasedwind"},
    ("offshorewind", None): {"technology": "offshorewind"},
    ("hydropower", None): {"technology": "hydro"},
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
EIA_TECHNOLOGY_MAP = {
    "conventionalhydroelectric": {"technology": "hydro", "small": False},
    "smallhydroelectric": {"technology": "hydro", "small": True},
    "onshorewindturbine": {"technology": "landbasedwind"},
    "offshorewindturbine": {"technology": "offshorewind"},
    "solarphotovoltaic": {"technology": "utilitypv"},
}


def _normalize(x: Optional[str]) -> Optional[str]:
    """
    Normalize string to lowercase, no whitespace, and no underscores.

    Examples
    --------
    >>> _normalize('Offshore Wind')
    'offshorewind'
    >>> _normalize('OffshoreWind')
    'offshorewind'
    >>> _normalize('Offshore_Wind')
    'offshorewind'
    >>> _normalize(None) is None
    True
    """
    if not x:
        return x
    return re.sub(r"\s+|_", "", x.lower())


def map_nrel_atb_technology(tech: str, detail: str = None) -> Dict[str, Any]:
    """
    Map NREL ATB technology to resource groups.

    Parameters
    ----------
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
    >>> map_nrel_atb_technology('Hydropower')
    {'technology': 'hydro'}
    >>> map_nrel_atb_technology('Hydropower', 'NSD4')
    {'technology': 'hydro'}
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


def map_eia_technology(tech: str) -> Dict[str, Any]:
    """
    Map EIA technology to resource groups.

    Parameters
    ----------
    tech
        Technology.

    Returns
    -------
    dict
        Key, value pairs identifying one or more resource groups.

    Examples
    --------
    >>> map_eia_technology('Solar Photovoltaic')
    {'technology': 'utilitypv'}
    >>> map_eia_technology('solar_photovoltaic')
    {'technology': 'utilitypv'}
    >>> map_eia_technology('Onshore Wind Turbine')
    {'technology': 'landbasedwind'}
    >>> map_eia_technology('Offshore Wind Turbine')
    {'technology': 'offshorewind'}
    >>> map_eia_technology('Conventional Hydroelectric')
    {'technology': 'hydro', 'small': False}
    >>> map_eia_technology('Small Hydroelectric')
    {'technology': 'hydro', 'small': True}
    >>> map_eia_technology('Unknown')
    {}
    """
    tech = _normalize(tech)
    group = {}
    for k, v in EIA_TECHNOLOGY_MAP.items():
        if tech == k or not k:
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
    columns : list
        Dataset column names.

    Raises
    ------
    ValueError
        Missing either path or dataframe.
    ValueError
        Dataframe columns are not all strings.

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
        if df is not None:
            if any(not isinstance(x, str) for x in df.columns):
                raise ValueError("Dataframe columns are not all strings")
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
    def columns(self) -> list:
        if self.df is not None:
            return list(self.df.columns)
        if self._columns is None:
            if self.format == "csv":
                self._columns = pd.read_csv(self.path, nrows=0).columns
        return list(self._columns)

    def read(self, columns: Iterable = None, cache: bool = None) -> pd.DataFrame:
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
        - `tree` : str, optional
          The name of the resource metadata attribute by
          which to differentiate between multiple precomputed hierarchical trees.
          Defaults to `None` (resource group does not represent hierarchical trees).
        - `metadata` : str, optional
          Relative path to resource metadata dataset (optional if `metadata` is `None`).
        - `profiles` : str, optional
          Relative path to resource profiles dataset.
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

        Resources representing hierarchical trees (see `group.tree`)
        require additional attributes.

        - `parent_id` : int
          Identifier of the resource formed by clustering this resource with the one
          other resource with the same `parent_id`.
          Only resources with `level` of 1 have no `parent_id`.
        - `level` : int
          Level of tree where the resource first appears, from `m`
          (the number of resources at the base of the tree), to 1.
        - `[group.tree]` : Any
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

        - uniques:

            - `ipm_region`
            - `metro_id`

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
    profiles : Optional[Table]
        Cached interface to resource profiles.

    Examples
    --------
    >>> group = {'technology': 'utilitypv'}
    >>> metadata = pd.DataFrame({'id': [0, 1], 'ipm_region': ['A', 'A'], 'mw': [1, 2]})
    >>> profiles = pd.DataFrame({'0': np.full(8784, 0.1), '1': np.full(8784, 0.4)})
    >>> rg = ResourceGroup(group, metadata, profiles)
    >>> rg.test_metadata()
    >>> rg.test_profiles()
    >>> rg.get_clusters(max_clusters=1)
           ipm_region  mw                                            profile
    (1, 0)          A   3  [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, ...
    """

    def __init__(
        self,
        group: Dict[str, Any],
        metadata: pd.DataFrame = None,
        profiles: pd.DataFrame = None,
        path: str = ".",
    ) -> None:
        self.group = {"existing": False, "tree": None, **group.copy()}
        for key in ["metadata", "profiles"]:
            if self.group.get(key):
                # Convert relative paths (relative to group file) to absolute paths
                self.group[key] = os.path.abspath(os.path.join(path, self.group[key]))
        required = ["technology"]
        if metadata is None:
            required.append("metadata")
        missing = [key for key in required if not self.group.get(key)]
        if missing:
            raise ValueError(
                f"Group metadata missing required keys {missing}: {self.group}"
            )
        self.metadata = Table(df=metadata, path=self.group.get("metadata"))
        self.profiles = None
        if profiles is not None or self.group.get("profiles"):
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
        if self.group.get("tree"):
            required.extend(["parent_id", "level", self.group["tree"]])
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
        if self.profiles is None:
            return None
        # Cast identifiers to string to match profile columns
        ids = self.metadata.read(columns=["id"])["id"].astype(str)
        columns = self.profiles.columns
        if not set(columns) == set(ids):
            raise ValueError(
                f"Resource profiles column names do not match resource identifiers"
            )
        df = self.profiles.read(columns=columns[0])
        if len(df) not in [8760, 8784]:
            raise ValueError(f"Resource profiles are not either 8760 or 8784 elements")

    def get_clusters(
        self,
        ipm_regions: Iterable[str] = None,
        min_capacity: float = None,
        max_clusters: int = None,
        max_lcoe: float = None,
        cap_multiplier: float = None,
        profiles: bool = True,
        utc_offset: int = 0,
    ) -> pd.DataFrame:
        """
        Compute resource clusters.

        Parameters
        ----------
        ipm_regions
            IPM regions in which to select resources.
            If `None`, all IPM regions are selected.
        min_capacity
            Minimum total capacity (MW). Resources are selected,
            from lowest to highest levelized cost of energy (lcoe),
            or from highest to lowest capacity if lcoe not available,
            until the minimum capacity is just exceeded.
            If `None`, all resources are selected for clustering.
        max_clusters
            Maximum number of resource clusters to compute.
            If `None`, no clustering is performed; resources are returned unchanged.
        max_lcoe
            Select only the resources with a levelized cost of electricity (lcoe)
            below this maximum. Takes precedence over `min_capacity`.
        cap_multiplier
            Multiplier applied to resource capacity before selection by `min_capacity`.
        profiles
            Whether to include cluster profiles, if available, in column `profile`.

        Returns
        -------
        pd.DataFrame
            Clustered resources whose indices are tuples of the resource identifiers
            from which they were constructed.

        Raises
        ------
        ValueError
            No resources found or selected.
        """
        df = self.metadata.read().set_index("id")
        if ipm_regions is not None:
            # Filter by IPM region
            df = df[df["ipm_region"].isin(ipm_regions)]
        if cap_multiplier is not None:
            # Apply capacity multiplier
            df[CAPACITY] *= cap_multiplier
        # Sort resources by lcoe (ascending) or capacity (descending)
        by = "lcoe" if "lcoe" in df else CAPACITY
        df = df.sort_values(by, ascending=by == "lcoe")
        # Select base resources
        tree = self.group["tree"]
        if tree:
            max_level = df[tree].map(df.groupby(tree)["level"].max())
            base = (df["level"] == max_level).values
            mask = base.copy()
        else:
            mask = np.ones(len(df), dtype=bool)
        if min_capacity:
            # Select resources until min_capacity reached
            temp = (df.loc[mask, CAPACITY].cumsum() < min_capacity).values
            temp[temp.argmin()] = True
            mask[mask] = temp
        if max_lcoe and "lcoe" in df:
            # Select clusters with LCOE below the cutoff
            mask[mask] = df.loc[mask, "lcoe"] <= max_lcoe
        if not mask.any():
            raise ValueError(f"No resources found or selected")
        if tree:
            # Only keep trees with one ore more base resources
            selected = (
                pd.Series(mask, index=df.index)
                .groupby(df[tree])
                .transform(lambda x: x.sum() > 0)
            )
            # Add non-base resources to selected trees
            mask |= selected & ~base
        # Apply mask
        df = df[mask]
        # Prepare merge
        merge = copy.deepcopy(MERGE)
        # Prepare profiles
        if profiles and self.profiles is not None:
            df["profile"] = list(
                np.roll(
                    self.profiles.read(columns=df.index.astype(str)).values.T,
                    utc_offset,
                )
            )
            merge["means"].append("profile")
        # Compute clusters
        if tree:
            return cluster_trees(df, by=by, tree=tree, max_rows=max_clusters, **merge)
        return cluster_rows(df, by=df[[by]], max_rows=max_clusters, **merge)


class ClusterBuilder:
    """
    Builds clusters of resources.

    Parameters
    ----------
    groups
        Groups of resources. See :class:`ResourceGroup`.

    Attributes
    ----------
    groups : Iterable[ResourceGroup]

    Examples
    --------
    Prepare the resource groups.

    >>> groups = []
    >>> group = {'technology': 'utilitypv'}
    >>> metadata = pd.DataFrame({'id': [0, 1], 'ipm_region': ['A', 'A'], 'mw': [1, 2]})
    >>> profiles = pd.DataFrame({'0': np.full(8784, 0.1), '1': np.full(8784, 0.4)})
    >>> groups.append(ResourceGroup(group, metadata, profiles))
    >>> group = {'technology': 'utilitypv', 'existing': True}
    >>> metadata = pd.DataFrame({'id': [0, 1], 'ipm_region': ['B', 'B'], 'mw': [1, 2]})
    >>> profiles = pd.DataFrame({'0': np.full(8784, 0.1), '1': np.full(8784, 0.4)})
    >>> groups.append(ResourceGroup(group, metadata, profiles))
    >>> builder = ClusterBuilder(groups)

    Compute resource clusters.

    >>> builder.get_clusters(ipm_regions=['A'], max_clusters=1,
    ...     technology='utilitypv', existing=False)
          ids ipm_region  mw  ...         profile technology  existing
    0  (1, 0)          A   3  [0.3, 0.3, 0.3, ...  utilitypv     False
    >>> builder.get_clusters(ipm_regions=['B'], min_capacity=2,
    ...     technology='utilitypv', existing=True)
        ids ipm_region  mw  ...         profile technology  existing
    0  (1,)          B   2  [0.4, 0.4, 0.4, ...  utilitypv      True

    Errors arise if search criteria is either ambiguous or results in an empty result.

    >>> builder.get_clusters(ipm_regions=['A'], technology='utilitypv')
    Traceback (most recent call last):
      ...
    ValueError: Parameters match multiple resource groups: [{...}, {...}]
    >>> builder.get_clusters(ipm_regions=['B'], technology='utilitypv', existing=False)
    Traceback (most recent call last):
      ...
    ValueError: No resources found or selected
    """

    def __init__(self, groups: Iterable[ResourceGroup]) -> None:
        self.groups = groups

    @classmethod
    def from_json(cls, paths: Iterable[Union[str, os.PathLike]]) -> "ClusterBuilder":
        """
        Load resources from resource group JSON files.

        Parameters
        ----------
        paths
            Paths to resource group JSON files.

        Raises
        ------
        ValueError
            No resource groups specified.
        """
        paths = list(paths)
        if not paths:
            raise ValueError(f"No resource groups specified")
        return cls([ResourceGroup.from_json(path) for path in paths])

    def find_groups(self, **kwargs: Any) -> List[ResourceGroup]:
        """
        Return the resource groups matching the specified arguments.

        Parameters
        ----------
        **kwargs
            Parameters to match against resource group metadata.
        """
        return [
            rg
            for rg in self.groups
            if all(k in rg.group and rg.group[k] == v for k, v in kwargs.items())
        ]

    def get_clusters(
        self,
        ipm_regions: Iterable[str] = None,
        min_capacity: float = None,
        max_clusters: int = None,
        max_lcoe: float = None,
        cap_multiplier: float = None,
        utc_offset: int = 0,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Compute resource clusters.

        See :meth:`ResourceGroup.get_clusters` for parameter descriptions.

        The following fields are added:

        - `ids` (tuple): Original resource identifiers.
        - **kwargs: Parameters used to uniquely identify the group.

        Parameters
        ----------
        ipm_regions
        min_capacity
        max_clusters
        max_lcoe
        cap_multiplier
        **kwargs
            Parameters to :meth:`find_groups` for selecting the resource group.

        Raises
        ------
        ValueError
            Parameters do not match any resource groups.
        ValueError
            Parameters match multiple resource groups.
        """
        groups = self.find_groups(**kwargs)
        if not groups:
            raise ValueError(f"Parameters do not match any resource groups: {kwargs}")
        if len(groups) > 1:
            meta = [rg.group for rg in groups]
            raise ValueError(f"Parameters match multiple resource groups: {meta}")
        return (
            groups[0]
            .get_clusters(
                ipm_regions=ipm_regions,
                min_capacity=min_capacity,
                max_clusters=max_clusters,
                max_lcoe=max_lcoe,
                cap_multiplier=cap_multiplier,
                utc_offset=utc_offset,
            )
            .assign(**kwargs)
            .rename_axis("ids")
            .reset_index()
        )


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


def merge_row_pair(
    a: Mapping,
    b: Mapping,
    sums: Iterable = None,
    means: Iterable = None,
    weight: Any = None,
    uniques: Iterable = None,
) -> dict:
    """
    Merge two mappings into one.

    Parameters
    ----------
    a
        First mapping (e.g. :class:`dict`, :class:`pd.Series`).
    b
        Second mapping.
    means
        Keys of values to average.
    weight
        Key of values to use as weights for weighted averages.
        If `None`, averages are not weighted.
    uniques
        Keys of values for which to return the value if equal, and `None` if not.

    Returns
    -------
    dict
        Merged row as a dictionary.

    Examples
    --------
    >>> df = pd.DataFrame({'mw': [1, 2], 'area': [10, 20], 'lcoe': [0.1, 0.4]})
    >>> a, b = df.to_dict('records')
    >>> merge_row_pair(a, b, sums=['area', 'mw'], means=['lcoe'], weight='mw')
    {'area': 30, 'mw': 3, 'lcoe': 0.3}
    >>> merge_row_pair(a, b, sums=['area', 'mw'], means=['lcoe'])
    {'area': 30, 'mw': 3, 'lcoe': 0.25}
    >>> b['mw'] = 1
    >>> merge_row_pair(a, b, uniques=['mw', 'area'])
    {'mw': 1, 'area': None}
    """
    merge = {}
    if sums:
        for key in sums:
            merge[key] = a[key] + b[key]
    if means:
        if weight:
            total = a[weight] + b[weight]
            aw = a[weight] / total
            bw = b[weight] / total
        else:
            aw = 0.5
            bw = 0.5
        for key in means:
            merge[key] = a[key] * aw + b[key] * bw
    if uniques:
        for key in uniques:
            merge[key] = a[key] if a[key] == b[key] else None
    return merge


def cluster_rows(
    df: pd.DataFrame, by: Iterable[Iterable], max_rows: int = None, **kwargs: Any
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
        If `None`, no clustering is performed.
    **kwargs
        Optional parameters to :func:`merge_row_pair`.

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
    >>> kwargs = {'sums': ['mw', 'area'], 'means': ['lcoe'], 'weight': 'mw'}
    >>> cluster_rows(df, by=df[['lcoe']], **kwargs)
          mw  area  lcoe
    (0,)   1     4   0.1
    (1,)   2     5   0.4
    (2,)   3     6   0.2
    >>> cluster_rows(df, by=df[['lcoe']], max_rows=2, **kwargs)
            mw  area   lcoe
    (1,)     2     5  0.400
    (0, 2)   4    10  0.175

    With a custom row index:

    >>> df.index = ['a', 'b', 'c']
    >>> cluster_rows(df, by=df[['lcoe']], max_rows=2, **kwargs)
            mw  area   lcoe
    (b,)     2     5  0.400
    (a, c)   4    10  0.175

    With an iterable row index:

    >>> df.index = [(1, 2), (4, ), (3, )]
    >>> cluster_rows(df, by=df[['lcoe']], max_rows=2, **kwargs)
               mw  area   lcoe
    (4,)        2     5  0.400
    (1, 2, 3)   4    10  0.175
    """
    nrows = len(df)
    if max_rows is None:
        max_rows = len(df)
    elif max_rows < 1:
        raise ValueError("Max number of rows must be greater than zero")
    drows = nrows - max_rows
    index = [_tuple(x) for x in df.index] + [None] * drows
    merge = prepare_merge(kwargs, df)
    df = df[get_merge_columns(merge, df)].reset_index(drop=True)
    if drows < 1:
        df.index = index
        return df
    # Convert dataframe rows to dictionaries
    rows = df.to_dict("records")
    # Preallocate new rows
    rows += [None] * drows
    # Preallocate new rows
    Z = scipy.cluster.hierarchy.ward(by)
    n = nrows + drows
    mask = np.ones(n, dtype=bool)
    for i, link in enumerate(Z[:drows, 0:2].astype(int)):
        mask[link] = False
        pid = nrows + i
        rows[pid] = merge_row_pair(rows[link[0]], rows[link[1]], **merge)
        index[pid] = index[link[0]] + index[link[1]]
    clusters = pd.DataFrame([x for x, m in zip(rows, mask) if m])
    # Preserve original column order
    clusters = clusters[[x for x in df.columns if x in clusters]]
    clusters.index = [x for x, m in zip(index, mask) if m]
    return clusters


def build_tree(
    df: pd.DataFrame, by: Iterable[Iterable], max_level: int = None, **kwargs: Any
) -> pd.DataFrame:
    """
    Build a hierarchical tree of rows in a dataframe.

    Uses the Ward variance minimization algorithm to incrementally merge rows.
    See :func:`scipy.cluster.hierarchy.linkage`.

    Parameters
    ----------
    df
        Rows to merge (m, ...).
        Should not have columns `id`, `parent_id`, and `level`, as these are appended to
        the result dataframe.
    by
        2-dimensional array of observation vectors (m, ...) from which to compute
        distances between each row pair.
    max_level
        Maximum level of tree to return,
        from m (the number of rows in `df`, if `None`) to 1.
    **kwargs
        Optional parameters to :func:`merge_row_pair`.

    Returns
    -------
    pd.DataFrame
        Hierarchical tree as a dataframe.
        Row indices are tuples of the original row indices from which they were built.
        If original indices were already iterables, they are merged
        (e.g. (1, 2) and (3, ) becomes (1, 2, 3)).
        The following columns are added:

        - `id` (int): New row identifier (0, ..., 0 + n).
        - `parent_id` (Int64): New row identifer of parent row.
        - `level` (int): Tree level of row (max_level, ..., 1).

    Raises
    ------
    ValueError
        Max level of tree must be greater than zero.

    Examples
    --------
    >>> df = pd.DataFrame({'mw': [1, 2, 3], 'area': [4, 5, 6], 'lcoe': [0.1, 0.4, 0.2]})
    >>> kwargs = {'sums': ['area', 'mw'], 'means': ['lcoe'], 'weight': 'mw'}
    >>> build_tree(df, by=df[['lcoe']], **kwargs)
               mw  area   lcoe  id  parent_id  level
    (0,)        1     4  0.100   0          3      3
    (1,)        2     5  0.400   1          4      3
    (2,)        3     6  0.200   2          3      3
    (0, 2)      4    10  0.175   3          4      2
    (1, 0, 2)   6    15  0.250   4        NaN      1
    >>> build_tree(df, by=df[['lcoe']], max_level=2, **kwargs)
               mw  area   lcoe  id  parent_id  level
    (1,)        2     5  0.400   0          2      2
    (0, 2)      4    10  0.175   1          2      2
    (1, 0, 2)   6    15  0.250   2        NaN      1
    >>> build_tree(df, by=df[['lcoe']], max_level=1, **kwargs)
               mw  area  lcoe  id  parent_id  level
    (1, 0, 2)   6    15  0.25   0        NaN      1
    """
    nrows = len(df)
    if max_level is None:
        max_level = nrows
    else:
        max_level = min(max_level, nrows)
        if max_level < 1:
            raise ValueError("Max level of tree must be greater than zero")
    drows = nrows - 1
    index = [_tuple(x) for x in df.index] + [None] * drows
    df = df.reset_index(drop=True)
    merge = prepare_merge(kwargs, df)
    columns = get_merge_columns(merge, df)
    df = df[columns]
    if drows < 1:
        df.index = index
        return df
    # Convert dataframe rows to dictionaries
    rows = df.to_dict("records")
    # Preallocate new rows
    rows += [None] * drows
    Z = scipy.cluster.hierarchy.linkage(by, method="ward")
    n = nrows + drows
    mask = np.ones(n, dtype=bool)
    level = np.concatenate((np.full(nrows, nrows), np.arange(drows, 0, -1)))
    parent_id = np.zeros(n)
    drop = nrows - max_level
    for i, link in enumerate(Z[:, 0:2].astype(int)):
        if i < drop:
            mask[link] = False
        pid = nrows + i
        parent_id[link] = pid
        rows[pid] = merge_row_pair(rows[link[0]], rows[link[1]], **merge)
        index[pid] = index[link[0]] + index[link[1]]
    tree = pd.DataFrame([x for x, m in zip(rows, mask) if m])
    # Restore original column order
    tree = tree[columns]
    # Normalize ids to 0, ..., n
    old_ids = np.where(mask)[0]
    new_ids = np.arange(len(old_ids))
    new_parent_ids = pd.Series(np.searchsorted(old_ids, parent_id[mask]), dtype="Int64")
    new_parent_ids.iloc[-1] = np.nan
    # Bump lower levels to max_level
    level = level[mask]
    if max_level < nrows:
        stop = level.size - np.searchsorted(level[::-1], max_level, side="right")
        level[:stop] = max_level
    tree = tree.assign(id=new_ids, parent_id=new_parent_ids, level=level)
    tree.index = [x for x, m in zip(index, mask) if m]
    return tree


def cluster_trees(
    df: pd.DataFrame, by: str, tree: str = None, max_rows: int = None, **kwargs: Any
) -> pd.DataFrame:
    """
    Merge rows in a dataframe following precomputed hierarchical trees.

    Parameters
    ----------
    df
        Rows to merge.
        Must have columns `parent_id` (matching values in index), `level`, and
        the columns named in **by** and **tree**.
    by
        Name of column to use for determining merge order.
        Children with the smallest pairwise distance on this column are merged first.
    tree
        Name of column to use for differentiating between hierarchical trees.
        If `None`, assumes rows represent a single tree.
    max_rows
        Number of rows at which to stop merging rows.
        If smaller than the number of trees, :func:`cluster_rows` is used to merge
        tree heads.
        If `None`, no merging is performed and only the base rows are returned.
    **kwargs
        Optional parameters to :func:`merge_row_pair`.

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
    ValueError
        `by` column not included in row merge arguments (`kwargs`).

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'level': [3, 3, 3, 2, 1],
    ...     'parent_id': pd.Series([3, 3, 4, 4, float('nan')], dtype='Int64'),
    ...     'mw': [0.1, 0.1, 0.1, 0.2, 0.3],
    ...     'area': [1, 1, 1, 2, 3]
    ... }, index=[0, 1, 2, 3, 4])
    >>> cluster_trees(df, by='mw', sums=['mw', 'area'], max_rows=2)
             mw  area
    (2,)    0.1     1
    (0, 1)  0.2     2
    >>> cluster_trees(df, by='mw', sums=['mw'], max_rows=1)
                mw
    (2, 0, 1)  0.3
    >>> cluster_trees(df, by='mw', sums=['mw'])
           mw
    (0,)  0.1
    (1,)  0.1
    (2,)  0.1
    """
    required = ["parent_id", "level", by]
    if tree:
        required.append(tree)
    missing = [key for key in required if key not in df]
    if missing:
        raise ValueError(f"Missing required fields {missing}")
    if tree:
        mask = df["level"] == df[tree].map(df.groupby(tree)["level"].max())
    else:
        mask = df["level"] == df["level"].max()
    nrows = mask.sum()
    if max_rows is None:
        max_rows = nrows
    elif max_rows < 1:
        raise ValueError("Max number of rows must be greater than zero")
    merge = prepare_merge(kwargs, df)
    columns = get_merge_columns(merge, df)
    if by not in columns:
        raise ValueError(f"{by} not included in row merge arguments")
    drows = nrows - max_rows
    if drows < 1:
        df = df.loc[mask, columns].copy()
        df.index = [_tuple(x) for x in df.index]
        return df
    df = df[set(columns + required)].assign(
        _id=df.index, _ids=[_tuple(x) for x in df.index], _mask=mask
    )
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
            pid = parents.index[0]
            ids = parents["ids"].iloc[0]
            children = df.loc[ids].to_dict("records")
            # Compute parent
            parent = {
                # Initial attributes
                # Can access series because all columns integer
                **df.loc[pid, ["_id", "parent_id", "level"]],
                # Merged children attributes
                # NOTE: Needed only if a child is incomplete
                **merge_row_pair(children[0], children[1], **merge),
                # Indices of all past children
                "_ids": df.loc[ids[0], "_ids"] + df.loc[ids[1], "_ids"],
                "_mask": True,
            }
            # Add parent
            df.loc[pid] = pd.Series(parent, dtype=object)
            # Drop children
            df.loc[ids, "_mask"] = False
            # Decrement rows
            drows -= 1
        else:
            # Promote child with deepest parent
            parent_id = df.loc[parents.index, "level"].idxmax()
            child_id = parents.loc[parent_id, "ids"][0]
            # Update child
            tree_columns = ["_id", "parent_id", "level"]
            df.loc[child_id, tree_columns] = df.loc[parent_id, tree_columns]
            # Update index
            df.rename(index={child_id: parent_id, parent_id: np.nan}, inplace=True)
    # Apply mask
    df = df[df["_mask"]]
    # Drop temporary columns
    df.index = df["_ids"].values
    df = df.drop(columns=["_id", "_ids", "_mask"])
    if len(df) > max_rows:
        df = cluster_rows(df, by=df[[by]], max_rows=max_rows, **kwargs)
    return df[columns]


def group_rows(
    df: pd.DataFrame, ids: Iterable[Iterable]
) -> pd.core.groupby.DataFrameGroupBy:
    """
    Group dataframe rows by index.

    Parameters
    ----------
    df
        Dataframe to group.
    ids
        Groups of rows indices.

    Returns
    -------
    pd.core.groupby.DataFrameGroupBy
        Rows of `df` grouped by their membership in each index group.

    Examples
    --------
    >>> df = pd.DataFrame({'x': [2, 1, 3]}, index=[2, 1, 3])
    >>> group_rows(df, [(1, ), (2, 3), (1, 2, 3)]).sum()
       x
    0  1
    1  5
    2  6
    """
    groups = np.repeat(np.arange(len(ids)), [len(x) for x in ids])
    index = np.concatenate(ids)
    return df.loc[index].groupby(groups, sort=False)


def prune_tree(df: pd.DataFrame, level: int) -> pd.DataFrame:
    """
    Prune base levels of hierarchical tree.

    Parameters
    ----------
    df
        Dataframe representing a hierarchical tree.
        Must have columns `id`, `parent_id` and `level`.
    level
        Level at which to prune tree.

    Returns
    -------
    pd.DataFrame
        Pruned hierarchical tree.
        Column `id` (and `parent_id`) is reset to (0, ..., nrows - 1).

    Examples
    --------
    >>> parent_id = pd.Series([3, 3, 4, 4, None], dtype='Int64')
    >>> df = pd.DataFrame({
    ...     'id': [0, 1, 2, 3, 4],
    ...     'parent_id': parent_id,
    ...     'level': [3, 3, 3, 2, 1]
    ... })
    >>> prune_tree(df, level=2)
       id  parent_id  level
    2   0          2      2
    3   1          2      2
    4   2        NaN      1
    """
    levels = df["level"].max()
    if level > levels:
        return df
    # Drop direct children of all parents up to and including max_level
    pids = df["id"][(df["level"] >= level) & (df["level"] < levels)]
    df = df[~df["parent_id"].isin(pids)].copy()
    # Bump level of remaining children
    df.loc[df["level"] > level, "level"] = level
    # Normalize ids to 0, ..., n
    mask = ~df["parent_id"].isna()
    df.loc[mask, "parent_id"] = np.searchsorted(df["id"], df["parent_id"][mask])
    df["id"] = np.arange(len(df))
    return df


def prepare_merge(merge: dict, df: pd.DataFrame) -> dict:
    """
    Prepare merge for a target dataframe.

    Parameters
    ----------
    merge
        Parameters to :func:`merge_row_pair`.
    df
        Dataframe to prepare merge for.

    Raises
    ------
    ValueError
        Column names duplicated in merge.
    ValueError
        Weights not present in dataframe.
    ValueError
        Weights not included in merge.

    Examples
    --------
    >>> df = pd.DataFrame(columns=['mw', 'lcoe'])
    >>> merge = {'sums': ['mw', 'area'], 'means': ['lcoe'], 'weight': 'mw'}
    >>> prepare_merge(merge, df)
    {'sums': ['mw'], 'means': ['lcoe'], 'weight': 'mw'}
    """
    reduced = {}
    for key in "sums", "means", "uniques":
        if merge.get(key):
            reduced[key] = [x for x in merge[key] if x in df]
    columns = get_merge_columns(reduced)
    if reduced.get("means") and merge.get("weight"):
        weight = merge["weight"]
        if weight not in df:
            raise ValueError(f"Weights {weight} not present in dataframe")
        if weight not in columns:
            raise ValueError(f"Weights {weight} not included in merge")
        reduced["weight"] = weight
    return reduced


def get_merge_columns(merge: dict, df: pd.DataFrame = None) -> list:
    """
    Get columns included in merge.

    Parameters
    ----------
    merge
        Parameters to :func:`merge_row_pair`.
    df
        Dataframe.
        If provided, only matching column names are returned, in order of appearance.

    Raises
    ------
    ValueError
        Column names duplicated in merge.

    Examples
    --------
    >>> merge = {'sums': ['mw'], 'means': ['lcoe'], 'uniques': None, 'weight': 'lcoe'}
    >>> get_merge_columns(merge)
    ['mw', 'lcoe']
    >>> get_merge_columns(merge, pd.DataFrame(columns=['lcoe', 'mw']))
    ['lcoe', 'mw']
    >>> get_merge_columns({'sums': ['mw'], 'means': ['mw']})
    Traceback (most recent call last):
      ...
    ValueError: Column names duplicated in merge
    """
    columns = (
        (merge.get("sums") or [])
        + (merge.get("means") or [])
        + (merge.get("uniques") or [])
    )
    if len(columns) > len(set(columns)):
        raise ValueError("Column names duplicated in merge")
    if df is not None:
        return [x for x in df if x in columns]
    return columns
