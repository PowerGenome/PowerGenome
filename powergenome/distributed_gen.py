import logging
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from powergenome.database import get_data
from powergenome.settings import auto_fill_settings
from powergenome.util import reverse_dict_of_lists

logger = logging.getLogger(__name__)

# Track which parameter combinations have been logged to avoid duplicate messages
_logged_capacity_calls: Set[Tuple[int, Tuple[str, ...], bool]] = set()


@auto_fill_settings(
    year="model_year",
    regions="model_regions",
    region_aggregations="region_aggregations",
)
def get_distributed_gen_capacity(
    year: int = None,
    regions: List[str] = None,
    region_aggregations: Optional[Dict[str, List[str]]] = None,
    aggregate_regions: bool = True,
) -> pd.DataFrame:
    """Get distributed generation capacity by region from DataManager.

    If capacity data is not available for the requested year, this function will
    attempt to interpolate values using linear interpolation between available years.
    If the requested year is outside the range of available data, the nearest year's
    value will be used (forward or backward extrapolation).

    Interpolation and extrapolation will only be performed if capacity data are not
    available for all regions in the requested year.

    Parameters
    ----------
    year : int
        Model planning year
    regions : List[str]
        Model regions (after aggregation)
    region_aggregations : Optional[Dict[str, List[str]]], optional
        Mapping of aggregated regions to their component regions, by default None
    aggregate_regions : bool, optional
        Whether to apply region aggregations to the capacity data, by default True

    Returns
    -------
    pd.DataFrame
        DataFrame with columns "region" and "capacity_mw"
    """
    if year is None:
        raise ValueError(
            "Model year must be provided to load distributed generation capacity (argument or settings)."
        )

    # Create a hashable key for this call to track if we've logged it before
    call_key = (
        year,
        tuple(sorted(regions)),
        tuple(region_aggregations or {}),
    )
    first_call = call_key not in _logged_capacity_calls

    if first_call:
        logger.info(f"Loading distributed generation capacity for year {year}")
        _logged_capacity_calls.add(call_key)

    # Get base regions (before aggregation)
    if region_aggregations and aggregate_regions:
        reverse_agg_map = reverse_dict_of_lists(region_aggregations)
        base_regions = [r for r in regions if r not in region_aggregations.keys()]
        base_regions.extend(list(reverse_agg_map.keys()))
        base_regions = list(set(base_regions))
    else:
        base_regions = regions

    # Load capacity data from DataManager
    filters = [[("year", "=", year), ("region", "in", base_regions)]]

    try:
        capacity_df = get_data(
            "distributed_capacity",
            filters=filters,
            columns=["region", "capacity_mw", "year"],
        )
    except Exception as e:
        logger.warning(
            f"Could not load distributed generation capacity data: {e}. "
            "Returning empty DataFrame."
        )
        return pd.DataFrame(columns=["region", "capacity_mw"])

    # If no data found for the exact year, try interpolation
    if capacity_df.empty:
        if first_call:
            logger.info(
                f"No distributed generation capacity data found for year {year}. "
                f"Attempting to interpolate from available years..."
            )

        # Load all available data for the required regions
        all_filters = [[("region", "in", base_regions)]]
        try:
            all_capacity_df = get_data(
                "distributed_capacity",
                filters=all_filters,
                columns=["region", "capacity_mw", "year"],
            )
        except Exception as e:
            logger.warning(
                f"Could not load distributed generation capacity data for interpolation: {e}. "
                "Returning empty DataFrame."
            )
            return pd.DataFrame(columns=["region", "capacity_mw"])

        if all_capacity_df.empty:
            logger.warning(
                f"No distributed generation capacity data found for regions {base_regions}. "
                "Cannot interpolate."
            )
            return pd.DataFrame(columns=["region", "capacity_mw"])

        # Interpolate or extrapolate for each region
        capacity_df_list = []
        interpolation_info = {
            "interpolated": [],
            "backward_extrapolated": [],
            "forward_extrapolated": [],
            "exact_match": [],
            "missing": [],
        }

        for region in base_regions:
            region_data = all_capacity_df[all_capacity_df["region"] == region].copy()

            if region_data.empty:
                interpolation_info["missing"].append(region)
                continue

            # Sort by year
            region_data = region_data.sort_values("year")
            available_years = region_data["year"].tolist()

            if year in available_years:
                # Exact match found (shouldn't happen since we're in the empty block, but being safe)
                capacity_value = region_data.loc[
                    region_data["year"] == year, "capacity_mw"
                ].values[0]
                interpolation_info["exact_match"].append(region)
            elif year < min(available_years):
                # Extrapolate backwards - use the earliest available year
                capacity_value = region_data.loc[
                    region_data["year"] == min(available_years), "capacity_mw"
                ].values[0]
                interpolation_info["backward_extrapolated"].append(
                    f"{region} (from {min(available_years)})"
                )
            elif year > max(available_years):
                # Extrapolate forwards - use the latest available year
                capacity_value = region_data.loc[
                    region_data["year"] == max(available_years), "capacity_mw"
                ].values[0]
                interpolation_info["forward_extrapolated"].append(
                    f"{region} (from {max(available_years)})"
                )
            else:
                # Interpolate between available years
                # Create a series with year as index
                capacity_series = region_data.set_index("year")["capacity_mw"]

                # Add the desired year with NaN value
                capacity_series.loc[year] = np.nan
                capacity_series = capacity_series.sort_index()

                # Interpolate linearly
                capacity_series = capacity_series.interpolate(method="index")
                capacity_value = capacity_series.loc[year]

                # Find the bounding years for logging
                lower_year = max(y for y in available_years if y < year)
                upper_year = min(y for y in available_years if y > year)
                interpolation_info["interpolated"].append(
                    f"{region} ({lower_year}-{upper_year})"
                )

            capacity_df_list.append(
                {
                    "region": region,
                    "capacity_mw": capacity_value,
                    "year": year,
                }
            )

        # Log summary of interpolation results
        summary_parts = []
        if interpolation_info["interpolated"]:
            summary_parts.append(
                f"Interpolated: {', '.join(interpolation_info['interpolated'])}"
            )
        if interpolation_info["backward_extrapolated"]:
            summary_parts.append(
                f"Backward extrapolated: {', '.join(interpolation_info['backward_extrapolated'])}"
            )
        if interpolation_info["forward_extrapolated"]:
            summary_parts.append(
                f"Forward extrapolated: {', '.join(interpolation_info['forward_extrapolated'])}"
            )
        if interpolation_info["exact_match"]:
            summary_parts.append(
                f"Exact match: {', '.join(interpolation_info['exact_match'])}"
            )
        if interpolation_info["missing"]:
            summary_parts.append(
                f"Missing data: {', '.join(interpolation_info['missing'])}"
            )

        if summary_parts and first_call:
            logger.info(
                f"Distributed generation capacity for year {year} - {'; '.join(summary_parts)}"
            )

        if not capacity_df_list:
            logger.warning(
                f"Could not interpolate capacity for any region in year {year}. "
                "Returning empty DataFrame."
            )
            return pd.DataFrame(columns=["region", "capacity_mw"])

        capacity_df = pd.DataFrame(capacity_df_list)

    # Apply region aggregations - sum capacity across aggregated regions
    if region_aggregations and aggregate_regions:
        for agg_region, component_regions in region_aggregations.items():
            mask = capacity_df["region"].isin(component_regions)
            if mask.any():
                agg_capacity = capacity_df.loc[mask, "capacity_mw"].sum()
                # Remove component regions and add aggregated region
                capacity_df = capacity_df.loc[~mask, :]
                capacity_df = pd.concat(
                    [
                        capacity_df,
                        pd.DataFrame(
                            [
                                {
                                    "region": agg_region,
                                    "capacity_mw": agg_capacity,
                                    "year": year,
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )

    return capacity_df[["region", "capacity_mw"]]


@auto_fill_settings(
    weather_year="weather_year",
    regions="model_regions",
    region_aggregations="region_aggregations",
    tz_offset="utc_offset",
    year="model_year",
)
def get_distributed_gen_profiles(
    weather_year: Optional[Union[int, List[int]]] = None,
    regions: List[str] = None,
    region_aggregations: Optional[Dict[str, List[str]]] = None,
    tz_offset: Optional[int] = None,
    year: Optional[int] = None,
) -> pd.DataFrame:
    """Get normalized distributed generation profiles by region from DataManager.

    Profiles are normalized (0-1 range) and represent the capacity factor at each hour.
    If regions are aggregated, profiles are weighted by capacity.

    Parameters
    ----------
    weather_year : Optional[Union[int, List[int]]], optional
        Weather year(s) for the profiles. Can be a single int or list of ints.
        If None, loads profiles for all available weather years.
        If multiple years provided, all matching time_index values are returned
        (the data structure depends on how time_index is structured in the source data).
    regions : List[str]
        Model regions (after aggregation)
    region_aggregations : Optional[Dict[str, List[str]]], optional
        Mapping of aggregated regions to their component regions, by default None
    tz_offset : Optional[int], optional
        Number of hours to shift profiles for timezone adjustment, by default None
    year : Optional[int], optional
        Model planning year, used for capacity-weighted aggregation, by default None

    Returns
    -------
    pd.DataFrame
        Wide DataFrame with columns for each region, indexed by time_index.
        The time_index structure depends on the source data.
    """
    # Normalize weather_year to a list
    if weather_year is None:
        weather_years = None
        logger.info(
            "weather_year not provided. Loading distributed generation profiles for ALL available weather years."
        )
    else:
        # Convert single int to list
        if isinstance(weather_year, int):
            weather_years = [weather_year]
        else:
            weather_years = weather_year
        logger.info(
            f"Loading distributed generation profiles for weather year(s) {weather_years}"
        )

    # Get base regions (before aggregation)
    if region_aggregations:
        reverse_agg_map = reverse_dict_of_lists(region_aggregations)
        base_regions = [r for r in regions if r not in region_aggregations.keys()]
        base_regions.extend(list(reverse_agg_map.keys()))
    else:
        base_regions = regions

    # Build filters (omit weather_year condition if None)
    if weather_years is None:
        filters = [[("region", "in", base_regions)]]
    else:
        filters = [
            [("weather_year", "in", weather_years), ("region", "in", base_regions)]
        ]

    try:
        profiles_df = get_data(
            "distributed_profiles",
            filters=filters,
            columns=["region", "time_index", "value", "weather_year"],
        ).sort_values(by=["weather_year", "region", "time_index"])

        # Set a time_index for each region that is continuous across multiple weather years
        # Go through each region and adjust time_index accordingly. Existing data might
        # have a single time_index across weather years or reset for each year.
        if profiles_df.weather_year.nunique() > 1:
            for region in base_regions:
                region_mask = profiles_df["region"] == region
                region_data = profiles_df.loc[region_mask].copy()

                if region_data.empty:
                    continue

                region_time_index = range(1, region_data.shape[0] + 1)
                profiles_df.loc[region_mask, "time_index"] = region_time_index

    except Exception as e:
        logger.warning(
            f"Could not load distributed generation profile data: {e}. "
            "Returning empty DataFrame."
        )
        return pd.DataFrame()

    if profiles_df.empty:
        msg_year = (
            f"weather year(s) {weather_years} "
            if weather_years is not None
            else "all weather years "
        )
        logger.warning(
            f"No distributed generation profile data found for {msg_year}and regions {base_regions}"
        )
        return pd.DataFrame()

    profiles_wide = profiles_df.pivot(
        index="time_index", columns="region", values="value"
    ).sort_index()

    # Reset index values to start at 1 and be continuous across all (possibly multi-year) data
    profiles_wide.index = range(1, len(profiles_wide) + 1)

    # Apply region aggregations - weighted average by capacity
    if region_aggregations:
        # Determine year to use for capacity weighting
        cap_year = year
        if cap_year is None:
            raise ValueError(
                "Cannot aggregate distributed generation profiles without a model_year "
                "(required for capacity-weighted averaging)."
            )

        # Load capacity data for weighting (required for aggregation)
        capacity_df = get_distributed_gen_capacity(
            year=cap_year,
            regions=base_regions,
            aggregate_regions=False,  # Don't aggregate yet
        )

        for agg_region, component_regions in region_aggregations.items():
            # Find which component regions exist in the profiles
            existing_components = [
                r for r in component_regions if r in profiles_wide.columns
            ]

            if not existing_components:
                continue

            # Get capacities for existing components - all must be present
            component_capacities = capacity_df.loc[
                capacity_df["region"].isin(existing_components)
            ].set_index("region")["capacity_mw"]

            # Check that we have capacity for all regions with profiles
            missing_capacity = set(existing_components) - set(
                component_capacities.index
            )
            if missing_capacity:
                raise ValueError(
                    f"Cannot aggregate distributed generation profiles for region '{agg_region}': "
                    f"missing capacity data for component region(s) {missing_capacity} in "
                    f"year {cap_year}. Capacity data is required for all regions with profiles."
                )

            # Calculate weighted average profile
            total_capacity = component_capacities.sum()
            agg_profile = pd.Series(0.0, index=profiles_wide.index)
            if total_capacity > 0:
                for comp_region in existing_components:
                    weight = component_capacities.loc[comp_region] / total_capacity
                    agg_profile += profiles_wide[comp_region] * weight

            profiles_wide[agg_region] = agg_profile
            profiles_wide = profiles_wide.drop(columns=existing_components)

    # Apply timezone offset
    if tz_offset is not None:
        for col in profiles_wide.columns:
            profiles_wide[col] = np.roll(profiles_wide[col], tz_offset)

    # Ensure index starts at 1 (not 0)
    if profiles_wide.index.min() == 0:
        profiles_wide.index = profiles_wide.index + 1

    profiles_wide.index.name = "time_index"

    return profiles_wide


@auto_fill_settings(
    year="model_year",
    weather_year="weather_year",
    regions="model_regions",
    region_aggregations="region_aggregations",
    tz_offset="utc_offset",
)
def get_distributed_gen_hourly_generation(
    year: int = None,
    weather_year: Optional[Union[int, List[int]]] = None,
    regions: List[str] = None,
    region_aggregations: Optional[Dict[str, List[str]]] = None,
    tz_offset: Optional[int] = None,
) -> pd.DataFrame:
    """Get hourly distributed generation in MW for each region.

    This function combines capacity and profiles to produce actual MW generation
    by hour for each region.

    Parameters
    ----------
    year : int
        Model planning year (for capacity data)
    weather_year : Optional[Union[int, List[int]]], optional
        Weather year(s) for profile data. Can be a single int or list of ints.
        If None, uses all available weather years.
        If multiple years provided, returns all matching time_index values
        (the length depends on the source data structure).
    regions : List[str]
        Model regions (after aggregation)
    region_aggregations : Optional[Dict[str, List[str]]], optional
        Mapping of aggregated regions to their component regions, by default None
    tz_offset : Optional[int], optional
        Number of hours to shift profiles for timezone adjustment, by default None

    Returns
    -------
    pd.DataFrame
        Wide DataFrame with hourly generation (MW) for each region.
        The number of rows depends on the source data's time_index structure.
    """
    if year is None:
        raise ValueError(
            "Model year must be provided (via argument or settings) for hourly generation."
        )
    logger.info(
        f"Calculating hourly distributed generation for year {year} (weather year {weather_year})"
    )

    # Get capacity by region
    capacity_df = get_distributed_gen_capacity(
        year=year,
        regions=regions,
        region_aggregations=region_aggregations,
    )

    if capacity_df.empty:
        logger.warning(
            "No capacity data available, returning empty generation DataFrame"
        )
        return pd.DataFrame()

    # Get normalized profiles by region
    profiles_df = get_distributed_gen_profiles(
        weather_year=weather_year,
        regions=regions,
        region_aggregations=region_aggregations,
        tz_offset=tz_offset,
        year=year,
    )

    if profiles_df.empty:
        logger.warning(
            "No profile data available, returning empty generation DataFrame"
        )
        return pd.DataFrame()

    # Multiply capacity by profile for each region
    hourly_gen = pd.DataFrame(index=profiles_df.index)

    for region in capacity_df["region"]:
        if region in profiles_df.columns:
            capacity = capacity_df.loc[
                capacity_df["region"] == region, "capacity_mw"
            ].values[0]
            hourly_gen[region] = profiles_df[region] * capacity
        else:
            logger.warning(
                f"Region {region} has capacity but no profile data. Skipping."
            )

    return hourly_gen
