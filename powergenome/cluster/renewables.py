"""
Flexible methods to cluster/aggregate renewable projects
"""

import logging
import operator
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.cluster import AgglomerativeClustering

from powergenome.resource_clusters import MERGE
from powergenome.util import deep_freeze_args, snake_case_str

logger = logging.getLogger(__name__)


def load_site_profiles(path: Path, site_ids: List[str]) -> pd.DataFrame:
    suffix = path.suffix
    site_ids = [s.replace(".0", "") for s in site_ids]
    if suffix == ".parquet":
        df = pq.read_table(path, columns=site_ids).to_pandas()
    elif suffix == ".csv":
        df = pd.read_csv(path, usecols=site_ids)
    return df


def value_bin(
    s: pd.Series,
    bins: Union[int, List[float]] = None,
    q: Union[int, List[float]] = None,
    weights: pd.Series = None,
) -> pd.Series:
    """Create a series of bin labels to split feature values

    Feature data can be split into bins based directly on values using pandas.cut, or
    quantiles of the data using pandas.qcut. If both "q" and "weights" are passed then
    a weighted quantile will be calculated.

    Parameters
    ----------
    s : pd.Series
        Data values for a feature
    bins : Union[int, List[float]], optional
        Integer number of equal-width bins or the bin edges to use, by default None
    q : Union[int, List[float]], optional
        Integer number of quantile bins or the quantile bin edges to use, by default None
    weights : pd.Series, optional
        Optional weight of each feature data point, only used in conjunction with "q",
        by default None

    Returns
    -------
    pd.Series
        Labels for each data point in the feature
    """
    try:
        from statsmodels.stats.weightstats import DescrStatsW
    except ModuleNotFoundError as e:
        logger.error(
            "\n\nThe package 'statsmodels' must be installed for weighted quantile calculations. "
            "Install it using mamba, pip, or conda, then try again.\n\n"
        )
        raise ModuleNotFoundError(e)
    if s.empty:
        return []
    # if all values are very close pandas will give a binning error
    if np.allclose(s.mean(), s.max()):
        return np.ones_like(s)
    if q:
        if q == 1:
            return np.ones_like(s)
        if weights is None:
            labels = pd.qcut(s, q=q, duplicates="drop", labels=False)
            return labels
        elif (weights == 0).all():
            weights = pd.Series(np.ones_like(weights))

        if isinstance(q, int):  # Calc feature values of bin edges from quantiles
            q = np.linspace(0, 1, q + 1)
        q.sort()
        wq = DescrStatsW(data=s, weights=weights)
        bins = wq.quantile(probs=q, return_pandas=False)
        labels = pd.cut(s, bins=bins, duplicates="drop", include_lowest=True)

    elif bins:
        if isinstance(bins, list):
            bins.sort()
            if bins[0] > s.min():
                logger.warning(
                    f"The minimum bin value in one of your renewables_clusters is "
                    f"{bins[0]}, which is greater than the minimum value of the data series. "
                    f"The min bin edge is being set to {s.min()}"
                )
                bins[0] = s.min()

            if bins[-1] < s.max():
                logger.warning(
                    f"The maximum bin value in one of your renewables_clusters is "
                    f"{bins[-1]}, which is less than the maximum value of the data series. "
                    f"The max bin edge is being set to {s.max()}"
                )
                bins[-1] = s.max()

            labels = pd.cut(s, bins=bins, duplicates="drop", include_lowest=True)
        else:
            if bins == 1:
                return np.ones_like(s)
            else:
                labels = pd.cut(s, bins=bins, duplicates="drop", include_lowest=True)
    else:
        logger.warning(
            "One of your renewables_clusters uses the 'bin' option but doesn't include "
            "either the 'bins' or 'q' argument. One of these is required to bin sites "
            "by a numeric feature. No binning will be performed on this group of sites."
        )
        labels = np.ones_like(s)

    return labels


@deep_freeze_args
@lru_cache()
def agg_cluster_profile(s: pd.Series, n_clusters: int) -> pd.DataFrame:
    if len(s) == 0:
        return []
    if len(s) == 1:
        return np.array([0])
    if n_clusters <= 0:
        logger.warning(
            f"You have entered a n_clusters parameter that is less than or equal to 0 "
            "in the settings parameter renewables_clusters. n_clusters must be >= 1. "
            "Manually setting n_clusters to 1 in this case."
        )
        n_clusters = 1
    if n_clusters > len(s):
        logger.warning(
            f"You have entered a n_clusters parameter that is greater than the number of "
            "renewable sites for one grouping in the settings parameter renewables_clusters."
            "Manually setting n_clusters to the number of renewable sites in this case."
        )
        n_clusters = len(s)

    clust = AgglomerativeClustering(n_clusters=n_clusters).fit(
        np.array([x for x in np.array(s)])
    )
    labels = clust.labels_
    return labels


@deep_freeze_args
@lru_cache()
def agg_cluster_other(s: pd.Series, n_clusters: int) -> pd.DataFrame:
    if len(s) == 0:
        return []
    if len(s) == 1:
        return np.array([0])
    if n_clusters <= 0:
        logger.warning(
            f"You have entered a n_clusters parameter that is less than or equal to 0 "
            "in the settings parameter renewables_clusters. n_clusters must be >= 1. "
            "Manually setting n_clusters to 1 in this case."
        )
        n_clusters = 1
    if n_clusters > len(s):
        logger.warning(
            f"You have entered a n_clusters parameter that is greater than the number of "
            "renewable sites for one grouping in the settings parameter renewables_clusters."
            "Manually setting n_clusters to the number of renewable sites in this case."
        )
        n_clusters = len(s)
    clust = AgglomerativeClustering(n_clusters=n_clusters).fit(
        np.array(s).reshape(-1, 1)
    )
    labels = clust.labels_
    return labels


def agglomerative_cluster_binned(
    data: pd.DataFrame,
    by: Union[str, List[str]],
    feature: str,
    n_clusters: Union[int, pd.Series, dict],
    **kwargs,
) -> pd.DataFrame:
    if data.empty:
        data["cluster"] = []
        return data

    if feature == "profile":
        func = agg_cluster_profile
    else:
        func = agg_cluster_other

    grouped = data.groupby(by)
    df_list = []
    first_label = 0
    if isinstance(n_clusters, int):
        _n_clusters = n_clusters
    for _, _df in grouped:
        if not _df.empty:
            if not isinstance(n_clusters, int):
                _n_clusters = n_clusters[_]
            labels = func(_df[feature], min(_n_clusters, len(_df)))
            labels += first_label
            first_label = max(labels) + 1
            _df["cluster"] = labels
            df_list.append(_df)
    df = pd.concat(df_list)

    return df


def agglomerative_cluster_no_bin(
    data: pd.DataFrame, feature: str, n_clusters: int, **kwargs
) -> pd.DataFrame:
    if data.empty:
        data["cluster"] = []
        return data
    _data = data.copy()
    if feature == "profile":
        func = agg_cluster_profile
    else:
        func = agg_cluster_other

    if len(_data) == 1:
        labels = 1
        _data["cluster"] = labels
    else:
        labels = func(_data[feature], min(n_clusters, len(_data)))
        _data["cluster"] = labels

    return _data


def agglomerative_cluster(binned: bool, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    kwargs.pop("method")
    if binned:
        return agglomerative_cluster_binned(data, **kwargs)
    else:
        return agglomerative_cluster_no_bin(data, **kwargs)


def value_filter(
    data: pd.DataFrame, feature: str, max_value: float = None, min_value: float = None
) -> pd.DataFrame:
    df = data.copy()
    if max_value:
        df = df.loc[df[feature] <= max_value, :]
    if min_value:
        df = df.loc[df[feature] >= min_value, :]

    return df


def min_capacity_mw(
    data: pd.DataFrame,
    min_cap: int = None,
    cap_col: str = "mw",
) -> pd.DataFrame:
    if "lcoe" not in data.columns:
        logger.warning(
            "You have selected a minimum capacity for one of the renewables_clusters "
            "groups. This feature is only available if the column 'lcoe' is included in "
            "your data file on renewable sites. No 'lcoe' column has been found, so all "
            "sites will be included in the final clusters."
        )
        return data
    if min_cap is None:
        return data

    _df = data.sort_values("lcoe")
    mask = np.ones(len(_df), dtype=bool)
    temp = (_df.loc[mask, cap_col].cumsum() < min_cap).values
    temp[temp.argmin()] = True
    mask[mask] = temp

    return _df.loc[mask, :]


def calc_cluster_values(
    df: pd.DataFrame,
    group: List[str] = None,
    sums: List[str] = MERGE["sums"],
    means: List[str] = MERGE["means"],
    weight: str = MERGE["weight"],
) -> pd.DataFrame:
    sums = [s for s in sums if s in df.columns]
    means = [m for m in means if m in df.columns]
    assert weight in df.columns
    df = df.reset_index(drop=True)
    df["weight"] = df[weight] / df[weight].sum()

    data = {}
    for s in sums:
        data[s] = df[s].sum()
    for m in means:
        data[m] = np.average(df[m], weights=df[weight])

    _df = pd.DataFrame(data, index=[0])
    profile = df.loc[0, "profile"] * df.loc[0, "weight"]
    for row in df.loc[1:, :].itertuples():
        profile += row.profile * row.weight

    profile /= df["weight"].sum()

    _df["profile"] = [profile]
    _df["cluster"] = df["cluster"].values[0]
    for g in group or []:
        _df["cluster"] = (
            str(_df["cluster"][0]) + f"_{g}:" + str(df[snake_case_str(g)].iloc[0])
        )

    return _df


CLUSTER_FUNCS = {
    "agglomerative": agglomerative_cluster,
    "agg": agglomerative_cluster,
    "hierarchical": agglomerative_cluster,
}


def assign_site_cluster(
    renew_data: pd.DataFrame,
    profile_path: Path,
    regions: List[str],
    site_map: pd.DataFrame = None,
    min_capacity: int = None,
    filter: List[dict] = None,
    bin: List[dict] = None,
    group: List[str] = None,
    cluster: List[dict] = None,
    utc_offset: int = 0,
    **kwargs: Any,
) -> pd.DataFrame:
    """Use settings options to group individual renewable sites.
    Sites are located within a model region that may be a collection of base regions.
    Site metadata such as the cpa_id and any numeric or categorial features (e.g.
    interconnection cost or the name of a geographic subregion that the site is located
    in) are specified in `renew_data`. Based on the user settings, the full list of
    sites in a region is filtered (numeric), binned (numeric), grouped(categorical), and
    clustered (numeric).
    Parameters
    ----------
    renew_data : pd.DataFrame
        Data on each renewable site. Must contain columns "cpa_id", "region", and either
        "interconnect_annuity" or "interconnect_capex_mw". Other columns are numeric or
        categorical features. Example columns might be a geographic location such as county
        (categorical), or the capacity factor (numeric).
    profile_path : Path
        Path to a parquet or CSV file with generation profiles. Column names should
        either correspond to the "cpa_id" from `renew_data` or mapped IDs from `site_map`.
    regions : List[str]
        The regions from "region" column of `renew_data` that will be used.
    site_map : pd.DataFrame, optional
        An optional mapping of `cpa_id` to the column names in the generation profile file,
        by default None
    min_capacity : int, optional
        A minimum amount of capacity to include, by default None. Can only be used if
        the column "lcoe" is included in `renew_data`. Sites are sorted based on lcoe
        and the least cost sites that satisfy the minimum capacity are retained.
    filter : List[dict], optional
        A list of filter parameters, by default None. Each dictionary should have the keys
        "feature" and at least one of "min" and "max". Only sites in `renew_data` with
        values that satisfy the min/max parameters for the specified features are retained.
    bin : List[dict], optional
        A list of binning parameters, by default None. Each dictionary should have the keys
        "feature" and either "bins" or "q". "bins" and "q" are input parameters to the
        Pandas functions `cut` and `qcut` -- they can be integers or a list of bin edge
        values.
    group : List[str], optional
        A list of categorical features in `renew_data` to group the sites, by default None.
    cluster : List[dict], optional
        A list of clustering parameters, by default None. Each dictionary should have the
        keys "feature", "method", and "n_clusters". Supported methods are "agglomerative".
        The feature should either be a column from `renew_data` or "profile" (use the
        generation profiles as a feature).
    utc_offset : int, optional
        Hours offset from UTC, by default 0. Generation data will be shifted from UTC
        by this value.
    Returns
    -------
    pd.DataFrame
        The renewable sites included in the study with a column "cluster". Other columns
        for binning and grouping may also be included.
    Raises
    ------
    KeyError
        The key "feature" is not included in a "bin" set of parameters
    KeyError
        The feature is not included in `renew_data`
    TypeError
        The feature specified by a "bin" set of parameters is not numeric.
    Examples
    --------
    This is an example of the YAML settings.
    renewables_clusters:
    - region: ISNE
        technology: landbasedwind
        min_capacity: 10000
        filter:
          - feature: lcoe
            max: 100
        bin:
          - feature: interconnect_annuity
            weights: mw
            bins: 3 # This can be an integer or list of bin edges
            # q: [0, .25, .5, .75, 1.] # Integer number of quantiles (e.g. quartiles) or list of quantile edges
        group:
          - state
        cluster:
          - feature: profile
            method: agglomerative
            n_clusters: 2
    """
    data = renew_data.loc[renew_data["region"].isin(regions), :]

    for filt in filter or []:
        data = value_filter(
            data=data,
            feature=snake_case_str(filt["feature"]),
            max_value=filt.get("max"),
            min_value=filt.get("min"),
        )
    if data.empty:
        data["cluster"] = []
        return data
    if min_capacity:
        data = min_capacity_mw(data, min_cap=min_capacity)
    if site_map is not None:
        site_ids = [site_map.loc[i] for i in data["cpa_id"]]
    else:
        site_ids = [str(int(i)) for i in data["cpa_id"]]
    if profile_path is not None:
        cpa_profiles = load_site_profiles(profile_path, site_ids=list(set(site_ids)))
        profiles = [np.roll(cpa_profiles[site].values, utc_offset) for site in site_ids]
        data["profile"] = profiles

    bin_features = []
    for b in bin or []:
        feature = snake_case_str(b.get("feature"))
        if not feature:
            raise KeyError(
                "One of your renewables_clusters uses the 'bin' option but doesn't include "
                "the 'feature' argument. You must specify a numeric feature (column) to "
                "split the renewable sites into bins."
            )
        if feature not in data.columns:
            raise KeyError(
                "One of your renewables_clusters uses the 'bin' option and includes the "
                f"feature argument '{feature}', which is not in the renewable site data. The "
                "feature must be one of the columns in your renewable site data file."
            )
        if not data[feature].dtype.kind in "iufc":
            raise TypeError(
                f"You specified the feature '{feature}' to bin one of your renewables_clusters. "
                f"'{feature}' is not a numeric column. Binning requires a numeric column."
            )

        weights_col = snake_case_str(b.get("weights"))
        if weights_col and weights_col not in data.columns:
            raise KeyError(
                "One of your renewables_clusters uses the 'bin' option and includes the "
                f"'weights' argument '{weights_col}', which is not in the renewable site data. The "
                "weights must be one of the columns in your renewable site data file.\n\n"
                "NOTE: Use the parameter 'mw' to weight by capacity."
            )
        elif weights_col:
            weights = data[weights_col]
        else:
            weights = None

        if not bin_features:
            b = num_bins_from_capacity(data, b)
            data[f"{feature}_bin"] = value_bin(
                data[feature], b.get("bins"), b.get("q"), weights=weights
            )
        else:
            df_list = []
            for _, _df in data.groupby(bin_features[-1]):
                _b = num_bins_from_capacity(_df, b.copy())
                _weights = None
                if weights is not None:
                    _weights = _df[weights_col]

                _df[f"{feature}_bin"] = value_bin(
                    _df[feature], _b.get("bins"), b.get("q"), weights=_weights
                )
                df_list.append(_df)
            data = pd.concat(df_list, ignore_index=True)
        bin_features.append(f"{feature}_bin")

    group_by = bin_features + ([snake_case_str(g) for g in group or []])
    prev_feature_cluster_col = None
    for clust in cluster or []:
        clust["feature"] = snake_case_str(clust["feature"])
        if "mw_per_cluster" in clust:
            if clust.get("n_clusters") is not None:
                logger.warning("Overwriting 'n_clusters' based on mw_cluster_size")
            if not group_by:
                clust["n_clusters"] = max(
                    int(data["mw"].sum() / clust["mw_per_cluster"]), 1
                )
            else:
                n_clusters = (
                    data.groupby(group_by)["mw"].sum() / clust["mw_per_cluster"]
                ).astype(int)
                clust["n_clusters"] = n_clusters.where(n_clusters > 0, 1)

        if "cluster" in data.columns and prev_feature_cluster_col:
            data = data.rename(columns={"cluster": prev_feature_cluster_col})
        if group_by:
            clust["by"] = group_by
            data = CLUSTER_FUNCS[clust["method"]](True, data, **clust)
        else:
            data = CLUSTER_FUNCS[clust["method"]](False, data, **clust)

        group_by.append(f"{clust['feature']}_clust")
        prev_feature_cluster_col = f"{clust['feature']}_clust"

    if "cluster" not in data.columns:
        if group_by:
            i = 1
            df_list = []
            for _, _df in data.groupby(group_by):
                _df["cluster"] = i
                df_list.append(_df)
                i += 1
            data = pd.concat(df_list)
        else:
            data["cluster"] = range(1, len(data) + 1)

    return data


def num_bins_from_capacity(data: pd.DataFrame, b: Dict[str, int]) -> Dict[str, int]:
    """Calculate the "bins" or "q" parameter based on available capacity.

    Either `mw_per_bin` or `mw_per_q` can be a key in the input dictionary. The
    number of bins/quantiles is calculated by dividing total site capacity by the
    mw_per_* value. If neither key is present, return the dictionary
    unaltered.

    Parameters
    ----------
    data : pd.DataFrame
        Must have column "mw"
    b : Dict[str, int]
        Must have either "mw_per_bin" or "mw_per_q" if the number of bins/quantiles
        should be decided based on available capacity.

    Returns
    -------
    Dict[str, int]
        Dictionary with either existing `bins`/`q` key or one of those keys calculated
        from available site capacity and `mw_per_bin`/`mw_per_q`
    """
    if "mw_per_bin" in b:
        if b.get("bins") is not None:
            logger.warning("Overwriting 'bins' based on mw_per_bin")
        b["bins"] = max(int(data["mw"].sum() / b["mw_per_bin"]), 1)
        del b["mw_per_bin"]
    elif "mw_per_q" in b:
        if b.get("q") is not None:
            logger.warning("Overwriting 'q' based on mw_per_q")
        b["q"] = max(int(data["mw"].sum() / b["mw_per_q"]), 1)
        del b["mw_per_q"]

    return b


def modify_renewable_group(
    df: pd.DataFrame, group_modifiers: List[Dict[str, Union[float, str, list]]] = None
) -> pd.DataFrame:
    """Modify values (e.g. cost) of a rewnewables cluster based on group membership.

    Parameters
    ----------
    df : pd.DataFrame
        Clustered renewable sites with averaged parameters like cost and profile. Must
        have column "cluster".
    group_modifiers : List[Dict[str, Union[float, str, list]]], optional
        List of dicts. Each must have keys "group" and "group_value". Any other keys should
         correspond to a column in `df` such as "capex_mw", "Inv_Cost_per_MW", etc. The
         values for these keys should either be a 2-item list with an operator and a value
         or a single numeric value.

         Rows of `df` are identified by string matching f"{group}:{group_value}" on the
         "cluster" column.

         By default None

    Returns
    -------
    pd.DataFrame
        Modified version of input df. No change in columns.

    Raises
    ------
    KeyError
        One dictionary in group_modifiers is missing either "group" or "group_value" keys
    ValueError
        The operator list is not a 2-item list. Must be 2 items (operator and value)
    ValueError
        The operator is not in the valid list (["add", "mul", "truediv", "sub"])
    """
    allowed_operators = ["add", "mul", "truediv", "sub"]
    for _group_mod in group_modifiers or []:
        group_mod = _group_mod.copy()
        missing_keys = [
            k for k in ["group", "group_value"] if k not in group_mod.keys()
        ]
        if missing_keys:
            raise KeyError(
                f"One of your 'renewables_clusters' has a 'group_modifiers' key but is "
                f"missing the key(s) {missing_keys}. These are required to modify values "
                "of a renewables cluster. Add these keys or remove the 'group_modifiers' "
                "section."
            )

        group = group_mod.pop("group")
        group_value = group_mod.pop("group_value")
        group_id = f"{group}:{group_value}"
        # for group, mod in (group_modifiers or {}).items():
        for key, op_list in group_mod.items():
            if isinstance(op_list, float) | isinstance(op_list, int):
                df.loc[df["cluster"].str.contains(group_id, case=False), key] = op_list
            else:
                if len(op_list) != 2:
                    raise ValueError(
                        "Either a single numeric value or a list of two values - an operator "
                        "and a numeric value - are needed in the parameter "
                        f"'{key}' whenever 'group_modifiers' are used in 'renewables_clusters'. "
                        f"One of your 'group_modifiers' has {op_list} instead."
                    )
                op, op_value = op_list
                if op not in allowed_operators:
                    raise ValueError(
                        f"One of the 'group_modifiers' in your 'renewables_clusters' with key "
                        f"{key} has {op} as the mathmatical operator. Only {allowed_operators} "
                        "in the format [<operator>, <value>] can be used to modify the "
                        "properties of a renewable cluster.\n"
                    )
                f = operator.attrgetter(op)
                df.loc[df["cluster"].str.contains(group_id, case=False), key] = f(
                    operator
                )(
                    df.loc[df["cluster"].str.contains(group_id, case=False), key],
                    op_value,
                )
    return df
