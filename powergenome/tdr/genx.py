"""GenX output assembly for time domain reduction.

This is the public entry point of the TDR backend. It mirrors the legacy
``powergenome.time_reduction.kmeans_time_clustering`` output contract while
following the numerical design of the MacroEnergy time domain reduction backend:
per-feature scaling, ``sqrt`` feature weights, forced extreme periods, occurrence
weights, and distinct representative periods.
"""

from __future__ import annotations

import datetime
import logging

import numpy as np
import pandas as pd

from powergenome.tdr.algorithms import CLUSTERING_METHODS, cluster_periods
from powergenome.tdr.matrix import build_candidate_matrix

logger = logging.getLogger(__name__)

_DAYS_IN_NON_LEAP_YEAR = 365


def reduce_time_domain(
    resource_profiles: pd.DataFrame,
    load_profiles: pd.DataFrame,
    demand_segments: pd.DataFrame,
    days_per_period: int,
    num_periods: int,
    include_peak_day: bool = True,
    demand_weight_factor: float = 1,
    variable_resources_only: bool = True,
    method: str = "kmeans",
    scaling: str = "normalize",
    n_restarts: int = 10,
    vre_profile_grouping: str = "none",
    resource_metadata: pd.DataFrame = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reduce hourly load and resource profiles to representative periods.

    Parameters
    ----------
    resource_profiles : pd.DataFrame
        Hourly generation profiles for all resources. Each column is a resource.
    load_profiles : pd.DataFrame
        Hourly demand profiles. Each column is a model region.
    demand_segments : pd.DataFrame
        Demand segments to repeat on every output row (``Voll`` and friends).
    days_per_period : int
        Number of 24-hour days in each representative period.
    num_periods : int
        Number of representative periods to model.
    include_peak_day : bool, optional
        Force the period containing the peak system load to be its own
        representative period, by default True.
    demand_weight_factor : float, optional
        Weight of the demand features, applied as ``sqrt(weight)``, by default 1.
    variable_resources_only : bool, optional
        Cluster only resource profiles with variance; the rest are written as 1.0,
        by default True.
    method : str, optional
        Clustering backend, ``"kmeans"`` or ``"kmedoids"``, by default "kmeans".
    scaling : str, optional
        ``"normalize"`` (min-max) or ``"standardize"`` (z-score), by default
        "normalize".
    n_restarts : int, optional
        Number of clustering initializations, by default 10.
    vre_profile_grouping : str, optional
        ``"none"`` or ``"region_technology"``. Grouping only changes the
        clustering matrix; every resource column is still written out, by default
        "none".
    resource_metadata : pd.DataFrame, optional
        Generator metadata indexed by resource name with ``region``,
        ``technology``, and ``VRE`` columns, used by VRE profile grouping.
    random_state : int, optional
        Seed for clustering initialization, by default 42.

    Returns
    -------
    tuple
        Reduced resource profiles, GenX ``Load_data`` output, the period map, and
        the representative-period slots. The reduced frames have
        ``num_periods * days_per_period * 24`` rows and ``Sub_Weights`` sums
        exactly to the input horizon.
    """
    if method not in CLUSTERING_METHODS:
        raise ValueError(
            f"Unknown time domain reduction method '{method}'. Expected one of "
            f"{CLUSTERING_METHODS}."
        )
    if len(load_profiles) != len(resource_profiles):
        raise ValueError(
            "Load and resource profiles must cover the same hours, got "
            f"{len(load_profiles)} and {len(resource_profiles)}."
        )

    n_hours = len(load_profiles)
    timesteps_per_period = days_per_period * 24
    n_periods, trailing_hours = divmod(n_hours, timesteps_per_period)
    if n_periods < 1:
        raise ValueError(
            f"A {n_hours}-hour horizon does not contain one complete "
            f"{timesteps_per_period}-hour period."
        )
    if num_periods < 1:
        raise ValueError(f"time_domain_periods must be at least 1, got {num_periods}.")
    if num_periods > n_periods:
        raise ValueError(
            f"time_domain_periods ({num_periods}) exceeds the {n_periods} complete "
            f"{timesteps_per_period}-hour periods in the {n_hours}-hour horizon."
        )

    candidates = build_candidate_matrix(
        load_profiles=load_profiles,
        resource_profiles=resource_profiles,
        timesteps_per_period=timesteps_per_period,
        scaling=scaling,
        demand_weight_factor=demand_weight_factor,
        variable_resources_only=variable_resources_only,
        vre_profile_grouping=vre_profile_grouping,
        resource_metadata=resource_metadata,
    )

    if num_periods == n_periods:
        # Every complete period is its own representative period.
        representative_periods = np.arange(1, n_periods + 1)
        period_to_representative = representative_periods.copy()
    else:
        representative_periods, period_to_representative = _select_representatives(
            matrix=candidates.matrix,
            n_periods=n_periods,
            num_periods=num_periods,
            timesteps_per_period=timesteps_per_period,
            load_profiles=load_profiles,
            include_peak_day=include_peak_day,
            method=method,
            n_restarts=n_restarts,
            random_state=random_state,
        )

    counts = np.bincount(
        _representative_index(representative_periods, period_to_representative),
        minlength=len(representative_periods),
    )
    sub_weights = counts * timesteps_per_period + _allocate_trailing_hours(
        trailing_hours, counts
    )

    if trailing_hours:
        logger.warning(
            f"Time domain reduction covers {n_hours} hours but only "
            f"{n_periods} complete {timesteps_per_period}-hour periods "
            f"({n_periods * timesteps_per_period} hours). The trailing "
            f"{trailing_hours} hours were excluded from clustering and their "
            f"weight was redistributed across the {len(representative_periods)} "
            f"representative periods in proportion to their occurrence counts, so "
            f"Sub_Weights still sum to {n_hours} hours."
        )

    offsets = [(period - 1) * timesteps_per_period for period in representative_periods]
    reduced_load_profile = _representative_rows(
        load_profiles, offsets, timesteps_per_period
    )
    reduced_resource_profile = _representative_rows(
        resource_profiles, offsets, timesteps_per_period
    )
    if candidates.excluded_resource_names:
        reduced_resource_profile[candidates.excluded_resource_names] = 1.0

    scale_factor = _annual_scale_factor(
        load_profiles,
        offsets,
        counts,
        timesteps_per_period,
        n_periods,
    )
    reduced_load_profile = reduced_load_profile * scale_factor

    reduced_resource_profile.index.name = "Resource"
    reduced_resource_profile.index = range(1, len(reduced_resource_profile) + 1)

    representative_index = _representative_index(
        representative_periods, period_to_representative
    )
    time_series_mapping = pd.DataFrame(
        {
            "Period_Index": np.arange(1, n_periods + 1),
            "Rep_Period_Index": representative_index + 1,
            "Month": [
                _month_for_day_of_year(days_per_period * period)
                for period in range(1, n_periods + 1)
            ],
            "Rep_Period": period_to_representative,
        }
    )

    reduced_load_output = pd.concat(
        [
            demand_segments,
            pd.Series([len(representative_periods)], name="Rep_Periods"),
            pd.Series([timesteps_per_period], name="Timesteps_per_Rep_Period"),
            pd.Series(sub_weights, name="Sub_Weights"),
            pd.Series(data=range(1, len(reduced_load_profile) + 1), name="Time_Index"),
            reduced_load_profile.round(0),
        ],
        axis=1,
    )

    representative_point = pd.DataFrame(
        {"slot": [f"p{period}" for period in representative_periods]}
    )

    return (
        reduced_resource_profile,
        reduced_load_output,
        time_series_mapping,
        representative_point,
    )


def _select_representatives(
    matrix: np.ndarray,
    n_periods: int,
    num_periods: int,
    timesteps_per_period: int,
    load_profiles: pd.DataFrame,
    include_peak_day: bool,
    method: str,
    n_restarts: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster the complete periods and return representatives and the period map."""
    forced_periods: list[int] = []
    if include_peak_day:
        if load_profiles.shape[1] == 0:
            raise ValueError("include_peak_day requires at least one demand column.")
        system_load = (
            load_profiles.iloc[: n_periods * timesteps_per_period]
            .sum(axis=1)
            .to_numpy(dtype=float)
        )
        forced_periods = [int(np.argmax(system_load)) // timesteps_per_period + 1]

    cluster_count = num_periods - len(forced_periods)
    if cluster_count < 1:
        raise ValueError(
            "include_peak_day leaves no periods to cluster; set "
            "time_domain_periods to at least 2 or include_peak_day to false."
        )

    candidate_periods = np.array(
        [period for period in range(1, n_periods + 1) if period not in forced_periods]
    )
    if cluster_count > len(candidate_periods):
        raise ValueError(
            f"Cannot select {num_periods} representative periods from {n_periods} "
            f"complete periods with {len(forced_periods)} forced extreme period(s)."
        )

    result = cluster_periods(
        matrix=matrix[:, candidate_periods - 1],
        n_clusters=cluster_count,
        method=method,
        n_restarts=n_restarts,
        random_state=random_state,
    )
    clustered_representatives = candidate_periods[result.representatives]

    period_to_representative = np.empty(n_periods, dtype=int)
    for period, assignment in zip(candidate_periods, result.assignments):
        period_to_representative[period - 1] = clustered_representatives[assignment]
    for period in forced_periods:
        period_to_representative[period - 1] = period

    representative_periods = np.array(
        sorted(set(forced_periods) | set(clustered_representatives.tolist()))
    )
    return representative_periods, period_to_representative


def _representative_index(
    representative_periods: np.ndarray, period_to_representative: np.ndarray
) -> np.ndarray:
    """Return the position of each period's representative in sorted order."""
    lookup = {int(period): index for index, period in enumerate(representative_periods)}
    return np.array([lookup[int(period)] for period in period_to_representative])


def _allocate_trailing_hours(trailing_hours: int, counts: np.ndarray) -> np.ndarray:
    """Split trailing hours across representatives by weight, largest remainder."""
    allocation = np.zeros(len(counts), dtype=int)
    if trailing_hours == 0:
        return allocation

    quota = trailing_hours * counts / counts.sum()
    allocation = np.floor(quota).astype(int)
    remainder = trailing_hours - int(allocation.sum())
    order = sorted(
        range(len(counts)),
        key=lambda index: (-(quota[index] - allocation[index]), index),
    )
    for index in order[:remainder]:
        allocation[index] += 1
    return allocation


def _representative_rows(
    frame: pd.DataFrame, offsets: list[int], timesteps_per_period: int
) -> pd.DataFrame:
    """Concatenate the raw hours of each representative period."""
    return pd.concat(
        [frame.iloc[offset : offset + timesteps_per_period] for offset in offsets],
        ignore_index=True,
    )


def _annual_scale_factor(
    load_profiles: pd.DataFrame,
    offsets: list[int],
    counts: np.ndarray,
    timesteps_per_period: int,
    n_periods: int,
) -> float:
    """Scale factor that matches the modeled annual demand to the raw demand."""
    load_values = load_profiles.iloc[: n_periods * timesteps_per_period].to_numpy(
        dtype=float
    )
    modeled = sum(
        load_values[offset : offset + timesteps_per_period].sum() * count
        for offset, count in zip(offsets, counts)
    )
    if modeled == 0:
        return 1.0
    return float(load_values.sum() / modeled)


def _month_for_day_of_year(day_of_year: int) -> int:
    """Calendar month of a 1-based day of year on the non-leap calendar."""
    day_of_year = min(max(day_of_year, 1), _DAYS_IN_NON_LEAP_YEAR)
    return (datetime.date(2011, 1, 1) + datetime.timedelta(days=day_of_year - 1)).month
