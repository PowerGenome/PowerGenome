"""Candidate-period matrix construction for time domain reduction.

Each complete period is one column of the resulting matrix and each clustering
feature contributes ``timesteps_per_period`` rows, which is the orientation the
MacroEnergy time domain reduction backend expects.

Features are scaled per feature (not globally, as the legacy implementation did)
and then multiplied by ``sqrt(weight)``. ``demand_weight_factor`` is the weight of
the demand features, so demand and resource features no longer need to compensate
for each other's scale.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SCALING_METHODS = ("normalize", "standardize")
VRE_PROFILE_GROUPINGS = ("none", "region_technology")


@dataclass(frozen=True)
class CandidateMatrix:
    """Scaled, weighted feature matrix for the complete candidate periods.

    Attributes
    ----------
    matrix : np.ndarray
        Shape ``(timesteps_per_period * n_features, n_periods)``. Columns are
        candidate periods in chronological order.
    feature_names : list[str]
        Name of every feature row block, demand features first.
    load_feature_names : list[str]
        Demand columns used as features.
    resource_feature_names : list[str]
        Resource features used for clustering. With VRE profile grouping these are
        group names rather than resource names.
    excluded_resource_names : list[str]
        Resource columns dropped by ``variable_resources_only``. They are filled
        with 1.0 in the reduced profiles, never clustered.
    timesteps_per_period : int
        Rows contributed by each feature.
    n_periods : int
        Number of complete periods in the horizon.
    trailing_hours : int
        Hours after the last complete period. They are not clustered.
    """

    matrix: np.ndarray
    feature_names: list[str]
    load_feature_names: list[str]
    resource_feature_names: list[str]
    excluded_resource_names: list[str]
    timesteps_per_period: int
    n_periods: int
    trailing_hours: int


def build_candidate_matrix(
    load_profiles: pd.DataFrame,
    resource_profiles: pd.DataFrame,
    timesteps_per_period: int,
    scaling: str = "normalize",
    demand_weight_factor: float = 1,
    variable_resources_only: bool = True,
    vre_profile_grouping: str = "none",
    resource_metadata: pd.DataFrame = None,
) -> CandidateMatrix:
    """Build the scaled, weighted candidate-period matrix.

    Parameters
    ----------
    load_profiles : pd.DataFrame
        Hourly demand. Each column is a model region.
    resource_profiles : pd.DataFrame
        Hourly resource profiles. Each column is a resource.
    timesteps_per_period : int
        Hours in each candidate period (days in the period times 24).
    scaling : str, optional
        ``"normalize"`` min-max scales each feature to [0, 1] over the candidate
        hours. ``"standardize"`` z-scores each feature over the candidate hours.
        By default "normalize".
    demand_weight_factor : float, optional
        Weight of every demand feature. Feature values are multiplied by
        ``sqrt(weight)``, by default 1.
    variable_resources_only : bool, optional
        Drop resource features without variance over the candidate hours, by
        default True.
    vre_profile_grouping : str, optional
        ``"region_technology"`` replaces each (region, technology) group of VRE
        resources with one feature holding the arithmetic mean of the group's raw
        profiles. ``"none"`` keeps one feature per resource, by default "none".
    resource_metadata : pd.DataFrame, optional
        Generator metadata indexed by resource name with ``region``,
        ``technology``, and ``VRE`` columns. Only used when
        ``vre_profile_grouping`` is enabled.

    Returns
    -------
    CandidateMatrix
        Candidate-period matrix and the feature bookkeeping needed to assemble
        the reduced outputs.

    Raises
    ------
    ValueError
        If the inputs have inconsistent lengths, no feature is usable, or a
        scaling/grouping mode is unknown.
    """
    if scaling not in SCALING_METHODS:
        raise ValueError(
            f"Unknown tdr_scaling '{scaling}'. Expected one of {SCALING_METHODS}."
        )
    if vre_profile_grouping not in VRE_PROFILE_GROUPINGS:
        raise ValueError(
            f"Unknown tdr_vre_profile_grouping '{vre_profile_grouping}'. "
            f"Expected one of {VRE_PROFILE_GROUPINGS}."
        )
    if demand_weight_factor < 0:
        raise ValueError(
            f"demand_weight_factor must not be negative, got {demand_weight_factor}."
        )
    if timesteps_per_period < 1:
        raise ValueError(
            f"timesteps_per_period must be positive, got {timesteps_per_period}."
        )
    if len(load_profiles) != len(resource_profiles):
        raise ValueError(
            "Load and resource profiles must cover the same hours, got "
            f"{len(load_profiles)} and {len(resource_profiles)}."
        )

    n_hours = len(load_profiles)
    n_periods, trailing_hours = divmod(n_hours, timesteps_per_period)
    if n_periods < 1:
        raise ValueError(
            f"A {n_hours}-hour horizon does not contain one complete "
            f"{timesteps_per_period}-hour period."
        )
    candidate_hours = n_periods * timesteps_per_period

    load_feature_names = list(load_profiles.columns)
    load_values = load_profiles.iloc[:candidate_hours].to_numpy(dtype=float)
    resource_values = resource_profiles.iloc[:candidate_hours]

    resource_features, excluded, resource_feature_names = _resource_features(
        resource_values,
        vre_profile_grouping,
        resource_metadata,
        variable_resources_only,
    )

    feature_values = [load_values[:, i] for i in range(len(load_feature_names))]
    feature_values.extend(resource_features)
    weights = [demand_weight_factor] * len(load_feature_names) + [1.0] * len(
        resource_feature_names
    )
    if not feature_values:
        raise ValueError(
            "Time domain reduction needs at least one demand or resource feature."
        )

    row_blocks = []
    for values, weight in zip(feature_values, weights):
        scaled = _scale_feature(values, scaling) * np.sqrt(weight)
        row_blocks.append(scaled.reshape(n_periods, timesteps_per_period).T)
    matrix = np.vstack(row_blocks)

    return CandidateMatrix(
        matrix=matrix,
        feature_names=load_feature_names + resource_feature_names,
        load_feature_names=load_feature_names,
        resource_feature_names=resource_feature_names,
        excluded_resource_names=excluded,
        timesteps_per_period=timesteps_per_period,
        n_periods=n_periods,
        trailing_hours=trailing_hours,
    )


def _scale_feature(values: np.ndarray, scaling: str) -> np.ndarray:
    """Scale one feature over its candidate hours.

    Constant features are all zeros, matching the upstream backend. Candidate
    periods occur once each, so the standardized form is the population z-score.
    """
    if scaling == "normalize":
        lower, upper = values.min(), values.max()
        if upper == lower:
            return np.zeros_like(values)
        return (values - lower) / (upper - lower)

    deviation = values.std()
    if deviation == 0:
        return np.zeros_like(values)
    return (values - values.mean()) / deviation


def _resource_features(
    resource_values: pd.DataFrame,
    grouping: str,
    metadata: pd.DataFrame,
    variable_resources_only: bool,
) -> tuple[list[np.ndarray], list[str], list[str]]:
    """Return resource feature values, excluded resource columns, and feature names.

    Grouping happens before the zero-variance filter, so the filter sees the
    feature that is actually clustered. A group is one feature: its members are
    averaged per timestep and it carries a single feature weight.
    """
    columns = list(resource_values.columns)
    features: dict[str, list[str]] = {}
    grouped_names: set[str] = set()
    fallback_count = 0

    for name in columns:
        key = None
        if grouping != "none":
            key, missing_metadata = _vre_group_key(name, metadata)
            fallback_count += int(missing_metadata)
        if key is None:
            features[name] = [name]
            continue
        feature_name = f"{key[0]}|{key[1]}"
        features.setdefault(feature_name, []).append(name)
        grouped_names.add(feature_name)

    if grouping == "none":
        logger.debug("TDR VRE profile grouping is disabled.")
    else:
        if metadata is None:
            logger.warning(
                "TDR VRE profile grouping is enabled but no resource metadata was "
                "supplied; every resource remains an individual clustering feature."
            )
        elif fallback_count:
            logger.warning(
                "TDR VRE profile grouping: %d resource(s) lack usable "
                "region/technology/VRE metadata and remain individual clustering "
                "features.",
                fallback_count,
            )
        n_vre_resources = sum(len(features[name]) for name in grouped_names)
        largest_group = max((len(features[name]) for name in grouped_names), default=0)
        logger.info(
            "TDR VRE profile grouping enabled: %d VRE resource(s) in %d group(s), "
            "largest group has %d resource(s).",
            n_vre_resources,
            len(grouped_names),
            largest_group,
        )

    values = []
    names = []
    excluded = []
    for name, members in features.items():
        feature = resource_values[members].to_numpy(dtype=float).mean(axis=1)
        # Keep the feature unless the zero-variance filter is on and it never varies.
        if not variable_resources_only or feature.std() > 0:
            values.append(feature)
            names.append(name)
        else:
            excluded.extend(members)

    return values, excluded, names


def _vre_group_key(
    resource: str, metadata: pd.DataFrame
) -> tuple[tuple[str, str] | None, bool]:
    """Return the grouping key for one resource and whether metadata was missing."""
    if metadata is None or resource not in metadata.index:
        return None, True
    if any(
        column not in metadata.columns for column in ("region", "technology", "VRE")
    ):
        return None, True

    row = metadata.loc[resource]
    if not _flag_is_true(row.get("VRE")):
        return None, False

    region, technology = row.get("region"), row.get("technology")
    if _is_missing(region) or _is_missing(technology):
        return None, True
    return (str(region), str(technology)), False


def _flag_is_true(value) -> bool:
    try:
        return float(value) == 1
    except (TypeError, ValueError):
        return False


def _is_missing(value) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False
