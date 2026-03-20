"""
Settings and data validation for PowerGenome.

Provides two-phase validation:

  Phase 1 (``validate_settings``):
    Settings internal consistency — region keys, required keys, path existence,
    planning year definitions, fuel definitions, and model tag coverage.
    Runs without any data file access.

  Phase 2 (``validate_settings_with_data``):
    Settings vs. loaded data — fuel price coverage, new-resource cost year
    overlap, and transmission region consistency.
    Requires a fully initialized ``DataManager``.

Severity levels
---------------
ERROR
    Definite failures that will cause exceptions or produce completely wrong
    results (e.g. missing required settings keys, non-existent paths).
WARNING
    Likely mistakes that produce silently incorrect results (e.g. a $0 fuel
    price due to a missing table entry, or a transmission line that will be
    dropped without any logged message).

Example
-------
::

    from powergenome.settings import Settings
    from powergenome.database import initialize_data_manager
    from powergenome.validate import (
        validate_settings,
        validate_settings_with_data,
        report_validation_results,
    )

    settings = Settings(config_path="my_study/settings")

    # Phase 1 — no DataManager needed
    results = validate_settings(settings)
    report_validation_results(results)          # raises if any ERRORs

    initialize_data_manager(settings, settings["data_location"])

    # Phase 2 — requires initialised DataManager
    from powergenome.database import _data_manager
    results2 = validate_settings_with_data(settings, _data_manager)
    report_validation_results(results2)
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Core types
# ──────────────────────────────────────────────────────────────────────────────


class ValidationLevel(str, Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"


@dataclass
class ValidationResult:
    level: ValidationLevel
    category: str
    message: str
    detail: Optional[str] = None

    def __str__(self) -> str:
        detail_str = f"\n    Detail: {self.detail}" if self.detail else ""
        return f"[{self.level.value}] {self.category}: {self.message}{detail_str}"


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

# Tags that every resource must carry at least one of in the final GenX output.
_REQUIRED_TAGS = frozenset({"THERM", "VRE", "MUST_RUN", "STOR", "FLEX", "HYDRO"})


def _err(category: str, message: str, detail: str = None) -> ValidationResult:
    return ValidationResult(ValidationLevel.ERROR, category, message, detail)


def _warn(category: str, message: str, detail: str = None) -> ValidationResult:
    return ValidationResult(ValidationLevel.WARNING, category, message, detail)


def _make_iterable(x: Any) -> list:
    """Return *x* as a list; wraps scalars, passes lists/tuples through."""
    if isinstance(x, (list, tuple)):
        return list(x)
    if x is None:
        return []
    return [x]


def _settings_as_dict(settings: Any) -> dict:
    """Normalise a Settings object or plain dict to a plain dict."""
    if hasattr(settings, "to_dict"):
        return settings.to_dict()
    if hasattr(settings, "get_data"):
        return settings.get_data()
    if isinstance(settings, dict):
        return settings
    raise TypeError(f"settings must be a dict or Settings object, got {type(settings)}")


def _extract_planning_periods(settings: dict) -> List[Tuple[int, int]]:
    """Return ``(first_planning_year, model_year)`` pairs for all planning periods.

    Returns an empty list if the planning-year keys are absent, which the
    ``_check_required_keys`` validator will already flag as an ERROR.
    """
    if "model_periods" in settings and settings["model_periods"] is not None:
        raw = _make_iterable(settings["model_periods"])
        if not raw:
            return []
        # Handle single period stored as a flat [first, last] rather than [[first, last]]
        if not isinstance(raw[0], (list, tuple)):
            raw = [raw]
        result = []
        for p in raw:
            try:
                result.append((int(p[0]), int(p[1])))
            except (TypeError, IndexError, ValueError):
                pass
        return result

    if "model_year" in settings and "model_first_planning_year" in settings:
        model_years = _make_iterable(settings["model_year"])
        first_years = _make_iterable(settings["model_first_planning_year"])
        return [(int(fy), int(ey)) for fy, ey in zip(first_years, model_years)]

    return []


def _tech_matches_any_key(tech_name: str, tag_dict: dict) -> bool:
    """Return True if any key in *tag_dict* is a case-insensitive substring of *tech_name*.

    This mirrors the logic in ``generators.add_resource_tags()`` which uses
    ``DataFrame.str.contains(key, case=False)`` for each tag key.
    """
    tech_lower = tech_name.lower()
    for key in tag_dict:
        if str(key).lower() in tech_lower:
            return True
    return False


def _tech_has_required_tag(tech_name: str, model_tag_values: dict) -> bool:
    """Return True if *tech_name* will receive at least one required model tag."""
    for tag in _REQUIRED_TAGS:
        tag_dict = model_tag_values.get(tag) or {}
        if _tech_matches_any_key(tech_name, tag_dict):
            return True
    return False


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1 checks (settings-only, no data access)
# ──────────────────────────────────────────────────────────────────────────────


def _check_required_keys(settings: dict) -> List[ValidationResult]:
    """ERROR if ``model_regions`` or ``target_usd_year`` are absent.

    Also checks that planning year information is present as either
    ``model_periods`` or the pair ``model_year`` + ``model_first_planning_year``.
    """
    results = []

    for key in ("model_regions", "target_usd_year"):
        if not settings.get(key):
            results.append(
                _err(
                    "required_keys",
                    f"Required settings key '{key}' is missing or empty",
                )
            )

    has_periods = bool(settings.get("model_periods"))
    has_legacy = bool(settings.get("model_year")) and bool(
        settings.get("model_first_planning_year")
    )
    if not has_periods and not has_legacy:
        results.append(
            _err(
                "required_keys",
                "Settings must include either 'model_periods' or both "
                "'model_year' and 'model_first_planning_year'",
            )
        )

    return results


def _check_year_list_consistency(settings: dict) -> List[ValidationResult]:
    """Validate planning-year definitions for structural problems.

    For **model_periods**: each entry must be a 2-element list/tuple and the
    first year must not exceed the last year.

    For the legacy **model_year / model_first_planning_year** pair: both lists
    must have the same length and each first-year must not exceed its model-year.
    """
    results = []

    if settings.get("model_periods"):
        raw = _make_iterable(settings["model_periods"])
        # Detect single flat period [first, last] vs list of periods [[f,l], ...]
        if raw and not isinstance(raw[0], (list, tuple)):
            raw = [raw]

        bad = [p for p in raw if not isinstance(p, (list, tuple)) or len(p) != 2]
        if bad:
            results.append(
                _err(
                    "planning_years",
                    "Each entry in 'model_periods' must be a 2-element list "
                    "[first_year, end_year]",
                    detail=f"Invalid entries: {bad}",
                )
            )

        for p in raw:
            if isinstance(p, (list, tuple)) and len(p) == 2:
                try:
                    if int(p[0]) > int(p[1]):
                        results.append(
                            _err(
                                "planning_years",
                                f"model_periods entry has first_year ({p[0]}) > "
                                f"end_year ({p[1]})",
                            )
                        )
                except (TypeError, ValueError):
                    pass
        return results

    if settings.get("model_year") and settings.get("model_first_planning_year"):
        my_list = _make_iterable(settings["model_year"])
        fp_list = _make_iterable(settings["model_first_planning_year"])
        if len(my_list) != len(fp_list):
            results.append(
                _err(
                    "planning_years",
                    f"'model_year' ({len(my_list)} value(s)) and "
                    f"'model_first_planning_year' ({len(fp_list)} value(s)) "
                    "must have the same number of entries",
                    detail=f"model_year={my_list}, "
                    f"model_first_planning_year={fp_list}",
                )
            )
        for fy, ey in zip(fp_list, my_list):
            try:
                if int(fy) > int(ey):
                    results.append(
                        _err(
                            "planning_years",
                            f"model_first_planning_year ({fy}) > model_year ({ey})",
                        )
                    )
            except (TypeError, ValueError):
                pass

    return results


def _check_paths_exist(settings: dict) -> List[ValidationResult]:
    """ERROR if configured file-system paths do not exist."""
    results = []

    for key in ("data_location", "RESOURCE_GROUPS", "RESOURCE_GROUP_PROFILES"):
        val = settings.get(key)
        if val is None:
            continue
        if not Path(val).exists():
            results.append(
                _err("paths", f"Path specified in '{key}' does not exist: {val}")
            )

    input_folder = settings.get("input_folder")
    if input_folder:
        if not Path(input_folder).exists():
            results.append(
                _err("paths", f"'input_folder' path does not exist: {input_folder}")
            )
        else:
            scenario_fn = settings.get("scenario_definitions_fn")
            if scenario_fn:
                scenario_path = Path(input_folder) / scenario_fn
                if not scenario_path.exists():
                    results.append(
                        _err(
                            "paths",
                            f"'scenario_definitions_fn' file not found: {scenario_path}",
                        )
                    )

    return results


def _check_region_consistency(settings: dict) -> List[ValidationResult]:
    """WARNING if any region-keyed setting contains regions not in ``model_regions``.

    Checks the following settings keys whose top-level dict keys must be model
    region names:

    * ``regional_tag_values``
    * ``alt_num_clusters``
    * ``regional_no_grouping``
    * ``new_gen_not_available``
    * ``regional_hydro_factor``
    * ``cost_multiplier_region_map``
    * ``distributed_gen_method``
    * ``regional_capacity_reserves`` (nested: constraint → region → value)
    * ``small_hydro_regions`` (list)
    """
    results = []

    model_regions = set(settings.get("model_regions") or [])
    if not model_regions:
        return results  # already flagged by _check_required_keys

    simple_keys = [
        "regional_tag_values",
        "alt_num_clusters",
        "regional_no_grouping",
        "new_gen_not_available",
        "regional_hydro_factor",
        "cost_multiplier_region_map",
        "distributed_gen_method",
    ]
    for key in simple_keys:
        val = settings.get(key)
        if not val or not isinstance(val, dict):
            continue
        unknown = set(val.keys()) - model_regions
        if unknown:
            results.append(
                _warn(
                    "region_consistency",
                    f"'{key}' contains region keys not in 'model_regions': "
                    f"{sorted(unknown)}",
                    detail=f"model_regions={sorted(model_regions)}",
                )
            )

    # regional_capacity_reserves: {constraint: {region: value}}
    cap_reserves = settings.get("regional_capacity_reserves")
    if cap_reserves and isinstance(cap_reserves, dict):
        for constraint, region_map in cap_reserves.items():
            if not isinstance(region_map, dict):
                continue
            unknown = set(region_map.keys()) - model_regions
            if unknown:
                results.append(
                    _warn(
                        "region_consistency",
                        f"'regional_capacity_reserves[{constraint}]' contains region "
                        f"keys not in 'model_regions': {sorted(unknown)}",
                    )
                )

    # small_hydro_regions: list
    small_hydro_regions = settings.get("small_hydro_regions")
    if small_hydro_regions and isinstance(small_hydro_regions, list):
        unknown = set(small_hydro_regions) - model_regions
        if unknown:
            results.append(
                _warn(
                    "region_consistency",
                    f"'small_hydro_regions' contains regions not in 'model_regions': "
                    f"{sorted(unknown)}",
                )
            )

    return results


def _check_model_tag_coverage(settings: dict) -> List[ValidationResult]:
    """WARNING if a new resource does not match any required model tag.

    Uses the same case-insensitive substring matching as
    ``generators.add_resource_tags()``.  Resources without a required tag will
    cause ``GenX.check_resource_tags()`` to raise a ``ValueError`` later.

    Checks ``new_resources`` and ``modified_new_resources``.  Existing
    resources (``num_clusters``) cannot be checked without generator data.
    """
    results = []

    model_tag_values = settings.get("model_tag_values") or {}
    if not model_tag_values:
        return results

    untagged: List[str] = []

    # new_resources: [[technology, tech_detail, cost_case, size_mw], ...]
    for resource in settings.get("new_resources") or []:
        if not isinstance(resource, (list, tuple)) or len(resource) < 2:
            continue
        full_name = f"{resource[0]}_{resource[1]}"
        if not _tech_has_required_tag(full_name, model_tag_values):
            # Also try just the technology component alone
            if not _tech_has_required_tag(str(resource[0]), model_tag_values):
                untagged.append(full_name)

    # modified_new_resources: {user_name: {new_technology, new_tech_detail, ...}}
    for name, spec in (settings.get("modified_new_resources") or {}).items():
        if not isinstance(spec, dict):
            continue
        tech = spec.get("new_technology") or spec.get("technology") or name
        detail = spec.get("new_tech_detail") or spec.get("tech_detail") or ""
        full_name = f"{tech}_{detail}" if detail else str(tech)
        if not _tech_has_required_tag(full_name, model_tag_values):
            if not _tech_has_required_tag(str(tech), model_tag_values):
                untagged.append(f"{name} (technology={tech})")

    if untagged:
        results.append(
            _warn(
                "model_tag_coverage",
                f"{len(untagged)} new resource(s) do not match any required model tag "
                f"(THERM / VRE / MUST_RUN / STOR / FLEX / HYDRO) in 'model_tag_values'",
                detail=f"Untagged: {untagged}",
            )
        )

    return results


def _check_fuel_consistency(settings: dict) -> List[ValidationResult]:
    """Warn about missing or incomplete fuel definitions.

    Checks:

    * Every fuel referenced in ``tech_fuel_map`` must appear in
      ``fuel_scenarios`` or ``user_fuel_price``; otherwise the generator will
      receive a ``$0/MMBtu`` price via ``fillna(0)``.
    * Every CCS fuel in ``ccs_fuel_map`` must exist in ``ccs_capture_rate``;
      otherwise the CCS capture rate silently defaults to 0 % (no CO₂ removed).
    * Every CCS fuel's base fuel must be in ``fuel_emission_factors``; otherwise
      CO₂ content silently defaults to 0.
    """
    results = []

    tech_fuel_map = settings.get("tech_fuel_map") or {}
    fuel_scenarios = set((settings.get("fuel_scenarios") or {}).keys())
    user_fuel_price = set((settings.get("user_fuel_price") or {}).keys())
    all_defined_fuels = fuel_scenarios | user_fuel_price

    # Fuels in tech_fuel_map must be defined somewhere
    missing_fuels = {
        fuel
        for fuel in tech_fuel_map.values()
        if fuel and fuel not in all_defined_fuels
    }
    if missing_fuels:
        results.append(
            _warn(
                "fuel_consistency",
                f"Fuel(s) in 'tech_fuel_map' are not defined in 'fuel_scenarios' or "
                f"'user_fuel_price': {sorted(missing_fuels)}",
                detail=(
                    "Generators using these fuels will receive a $0/MMBtu price "
                    "via fillna(0)"
                ),
            )
        )

    # CCS-specific checks
    ccs_fuel_map = settings.get("ccs_fuel_map") or {}
    ccs_capture_rate = settings.get("ccs_capture_rate") or {}
    fuel_emission_factors = settings.get("fuel_emission_factors") or {}

    for tech, ccs_fuel_name in ccs_fuel_map.items():
        # Capture rate must be defined or CCS silently does nothing
        if ccs_fuel_name not in ccs_capture_rate:
            results.append(
                _warn(
                    "fuel_consistency",
                    f"CCS fuel '{ccs_fuel_name}' (mapped from tech '{tech}') is not in "
                    f"'ccs_capture_rate' — capture rate silently defaults to 0 %",
                    detail=(
                        "With 0 % capture, CCS carries full CO₂ emissions but still "
                        "incurs disposal costs"
                    ),
                )
            )

        # Base fuel must have an emission factor
        if "_ccs" in ccs_fuel_name.lower():
            idx = ccs_fuel_name.lower().index("_ccs")
            base_fuel = ccs_fuel_name[:idx]
        else:
            parts = ccs_fuel_name.split("_")
            base_fuel = "_".join(parts[:-1]) if len(parts) > 1 else ccs_fuel_name

        if base_fuel and base_fuel not in fuel_emission_factors:
            results.append(
                _warn(
                    "fuel_consistency",
                    f"CCS fuel '{ccs_fuel_name}' base fuel '{base_fuel}' is not in "
                    f"'fuel_emission_factors' — CO₂ content will default to 0",
                    detail=f"tech: {tech}, ccs_fuel_map entry: {ccs_fuel_name}",
                )
            )

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1 entry point
# ──────────────────────────────────────────────────────────────────────────────

_PHASE1_CHECKS = [
    _check_required_keys,
    _check_year_list_consistency,
    _check_paths_exist,
    _check_region_consistency,
    _check_model_tag_coverage,
    _check_fuel_consistency,
]


def validate_settings(settings: Any) -> List[ValidationResult]:
    """Phase 1: validate settings internal consistency without accessing any data.

    Parameters
    ----------
    settings : dict or Settings
        Loaded PowerGenome settings (before or after scenario resolution).

    Returns
    -------
    list[ValidationResult]
        All detected issues.  An empty list means no problems were found.
    """
    d = _settings_as_dict(settings)
    results: List[ValidationResult] = []
    for check_fn in _PHASE1_CHECKS:
        try:
            results.extend(check_fn(d))
        except Exception as exc:
            logger.debug(
                "Validation check %s raised an unexpected exception",
                check_fn.__name__,
                exc_info=True,
            )
            results.append(
                _err(
                    "internal_error",
                    f"Validation check '{check_fn.__name__}' raised an unexpected "
                    f"exception — this is a bug; some checks may have been skipped",
                    detail=f"{type(exc).__name__}: {exc}",
                )
            )
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2 checks (settings + DataManager)
# ──────────────────────────────────────────────────────────────────────────────


def _check_transmission_regions(settings: dict, dm: Any) -> List[ValidationResult]:
    """WARNING if transmission cost table contains regions absent from ``model_regions``.

    Such rows are silently dropped by ``transmission.load_tx_costs()`` via
    ``.dropna(subset=['zone_1', 'zone_2'])`` after the unmapped regions produce
    ``NaN`` in the zone-number columns.
    """
    results = []

    if "transmission_cost" not in (dm.available_tables or set()):
        return results

    try:
        tx_df = dm.get_data(
            "transmission_cost", columns=["start_region", "dest_region"]
        )
    except Exception as exc:
        results.append(
            _warn(
                "transmission_regions",
                f"Could not load transmission_cost table for validation: {exc}",
            )
        )
        return results

    model_regions = set(settings.get("model_regions") or [])
    # Base IPM regions (from region_aggregations values) are also valid here
    agg_regions = settings.get("region_aggregations") or {}
    base_regions: set = set()
    for base_list in agg_regions.values():
        if isinstance(base_list, (list, tuple)):
            base_regions.update(base_list)
    valid_regions = model_regions | base_regions

    all_in_table = set(tx_df["start_region"].dropna()) | set(
        tx_df["dest_region"].dropna()
    )
    unknown = all_in_table - valid_regions
    if unknown:
        results.append(
            _warn(
                "transmission_regions",
                f"'transmission_cost_table' contains {len(unknown)} region(s) not in "
                f"'model_regions' or their constituent IPM regions — these rows will "
                f"be silently dropped during transmission processing",
                detail=f"Unknown regions: {sorted(unknown)}",
            )
        )

    return results


def _check_fuel_price_coverage(settings: dict, dm: Any) -> List[ValidationResult]:
    """WARNING if any fuel/scenario/region/year combination is missing from the fuel-price table.

    Missing combinations produce ``$0/MMBtu`` in the GenX ``Fuels_data.csv``
    output via ``fillna(0)`` in ``fuels.fuel_cost_table()``.

    Checks every fuel defined in ``fuel_scenarios`` for each model year (the
    end year of each planning period) and each base region that makes up the
    model regions.
    """
    results = []

    if "fuel_price" not in (dm.available_tables or set()):
        return results

    planning_periods = _extract_planning_periods(settings)
    if not planning_periods:
        return results

    model_years = [p[1] for p in planning_periods]
    fuel_scenarios: Dict[str, str] = settings.get("fuel_scenarios") or {}
    model_regions: List[str] = list(settings.get("model_regions") or [])
    region_aggregations: Dict[str, list] = settings.get("region_aggregations") or {}

    if not fuel_scenarios or not model_regions:
        return results

    try:
        fuel_df = dm.get_data(
            "fuel_price", columns=["year", "fuel", "region", "scenario"]
        )
    except Exception as exc:
        results.append(
            _warn(
                "fuel_price_coverage",
                f"Could not load fuel_price table for validation: {exc}",
            )
        )
        return results

    if fuel_df.empty or "year" not in fuel_df.columns:
        return results

    # Coerce year column to int for comparison
    try:
        fuel_df["year"] = fuel_df["year"].astype(int)
    except Exception:
        pass

    has_region_col = "region" in fuel_df.columns
    has_scenario_col = "scenario" in fuel_df.columns

    missing: List[str] = []
    for fuel, scenario in fuel_scenarios.items():
        for model_year in model_years:
            # Resolve which base regions we need prices for
            base_regions_needed: List[str] = []
            for region in model_regions:
                if region in region_aggregations:
                    base_regions_needed.extend(region_aggregations[region])
                else:
                    base_regions_needed.append(region)

            for region in base_regions_needed:
                mask = fuel_df["year"] == model_year
                if has_region_col:
                    mask &= fuel_df["region"] == region
                if has_scenario_col and scenario:
                    mask &= fuel_df["scenario"] == scenario
                # Match on fuel name
                if "fuel" in fuel_df.columns:
                    mask &= fuel_df["fuel"] == fuel

                if not mask.any():
                    missing.append(
                        f"fuel={fuel}, scenario={scenario}, "
                        f"region={region}, year={model_year}"
                    )

    if missing:
        shown = missing[:10]
        truncated = f"  … and {len(missing) - 10} more" if len(missing) > 10 else ""
        results.append(
            _warn(
                "fuel_price_coverage",
                f"{len(missing)} fuel/scenario/region/year combination(s) are missing "
                f"from the fuel_price table — missing prices will become $0/MMBtu via "
                f"fillna(0)",
                detail="\n    ".join(shown) + truncated,
            )
        )

    return results


def _check_new_resource_cost_years(settings: dict, dm: Any) -> List[ValidationResult]:
    """WARNING if a new-resource/planning-period combination has no cost data.

    ``new_build.single_generator_row()`` averages costs over
    ``range(first_planning_year, model_year + 1)`` in the ``basis_year``
    column.  If no rows match, ``.mean()`` returns ``NaN`` which then becomes
    ``$0`` via ``fillna(0)`` — silently zeroing capex and fixed/variable O&M.
    """
    results = []

    if "resource_cost" not in (dm.available_tables or set()):
        return results

    new_resources = settings.get("new_resources") or []
    if not new_resources:
        return results

    planning_periods = _extract_planning_periods(settings)
    if not planning_periods:
        return results

    try:
        cost_df = dm.get_data(
            "resource_cost",
            columns=["technology", "tech_detail", "cost_case", "basis_year"],
        )
    except Exception as exc:
        results.append(
            _warn(
                "new_resource_cost_years",
                f"Could not load resource_cost table for validation: {exc}",
            )
        )
        return results

    if cost_df.empty:
        return results

    # Coerce basis_year to int for set operations
    try:
        cost_df["basis_year"] = cost_df["basis_year"].astype(int)
    except Exception:
        pass

    no_overlap: List[str] = []
    for resource in new_resources:
        if not isinstance(resource, (list, tuple)) or len(resource) < 3:
            continue
        technology, tech_detail, cost_case = (
            str(resource[0]),
            str(resource[1]),
            str(resource[2]),
        )

        available_years: set = set(
            cost_df.loc[
                (cost_df["technology"] == technology)
                & (cost_df["tech_detail"] == tech_detail)
                & (cost_df["cost_case"] == cost_case),
                "basis_year",
            ].dropna()
        )

        if not available_years:
            no_overlap.append(
                f"{technology}/{tech_detail}/{cost_case}: "
                f"no matching rows found in resource_cost table"
            )
            continue

        for first_year, end_year in planning_periods:
            period_range = set(range(first_year, end_year + 1))
            if period_range.isdisjoint(available_years):
                available_sample = sorted(available_years)[:5]
                ellipsis = "…" if len(available_years) > 5 else ""
                no_overlap.append(
                    f"{technology}/{tech_detail}/{cost_case}: "
                    f"no basis_year in {first_year}–{end_year} "
                    f"(available: {available_sample}{ellipsis})"
                )

    if no_overlap:
        results.append(
            _warn(
                "new_resource_cost_years",
                f"{len(no_overlap)} new resource/period combination(s) have no "
                f"matching basis_year in the resource_cost table — costs will be "
                f"$0 via fillna(0)",
                detail="\n    ".join(no_overlap),
            )
        )

    return results


def _check_data_tables_loaded(settings: dict, dm: Any) -> List[ValidationResult]:
    """ERROR if a settings-configured table was not loaded into DataManager.

    With lazy loading, ``DataManager`` creates a *view* pointing at the source
    file during ``initialize()``.  If the source file does not exist the view
    is still registered in ``available_tables`` but will fail at query time.
    This check catches the less common case where ``available_tables`` is
    actually missing a table that was requested.
    """
    results = []

    try:
        from powergenome.database import DataManager as _DM

        mapping = _DM.STANDARD_TABLE_MAPPING
    except Exception:
        return results

    available = dm.available_tables or set()
    for setting_key, standard_name in mapping.items():
        if settings.get(setting_key) and standard_name not in available:
            results.append(
                _err(
                    "data_tables",
                    f"Settings key '{setting_key}' is configured but the table "
                    f"'{standard_name}' was not loaded by DataManager — "
                    f"check that the file exists in 'data_location'",
                )
            )

    return results


def _check_aggregation_base_regions(settings: dict, dm: Any) -> List[ValidationResult]:
    """WARNING if a base region referenced in ``model_regions`` or ``region_aggregations`` is not found in any data table.

    Two sources of base regions are checked:

    1. **Aggregation values** — every region listed as a value in
       ``region_aggregations`` (e.g. ``AZ: [p1, q2]`` → checks ``p1`` and
       ``q2``).
    2. **Pass-through model regions** — entries in ``model_regions`` that have
       no entry in ``region_aggregations`` are treated by PowerGenome as a
       direct 1:1 base region (e.g. ``model_regions: [p1_2, q2]`` where
       ``q2`` is not a key in ``region_aggregations``).

    Both sets are validated against the region names present in the
    ``plant_region``, ``demand``, ``fuel_price``, and ``transmission_cost``
    tables (whichever are loaded).  A typo in either location will be caught
    because the misspelled name will not appear in any data table.
    """
    results = []

    model_regions: List[str] = list(settings.get("model_regions") or [])
    region_aggregations = settings.get("region_aggregations") or {}

    # All base regions listed across the aggregation values
    aggregation_base_regions: set = set()
    for base_list in region_aggregations.values():
        if isinstance(base_list, (list, tuple)):
            aggregation_base_regions.update(str(r) for r in base_list)

    # Pass-through model regions: in model_regions but NOT a key in region_aggregations
    passthrough_regions: set = {
        str(r) for r in model_regions if str(r) not in region_aggregations
    }

    all_base_regions = aggregation_base_regions | passthrough_regions
    if not all_base_regions:
        return results

    available = dm.available_tables or set()
    known_regions: set = set()

    # ── plant_region ──────────────────────────────────────────────────────────
    if "plant_region" in available:
        try:
            pr_df = dm.get_data("plant_region", columns=["region"])
            if "region" in pr_df.columns:
                known_regions.update(pr_df["region"].dropna().astype(str))
        except Exception:
            pass

    # ── demand ────────────────────────────────────────────────────────────────
    if "demand" in available:
        try:
            dem_df = dm.get_data("demand", columns=["region"])
            if "region" in dem_df.columns:
                known_regions.update(dem_df["region"].dropna().astype(str))
        except Exception:
            pass

    # ── fuel_price ────────────────────────────────────────────────────────────
    if "fuel_price" in available:
        try:
            fp_df = dm.get_data("fuel_price", columns=["region"])
            if "region" in fp_df.columns:
                known_regions.update(fp_df["region"].dropna().astype(str))
        except Exception:
            pass

    # ── transmission_cost ─────────────────────────────────────────────────────
    if "transmission_cost" in available:
        try:
            tx_df = dm.get_data(
                "transmission_cost", columns=["start_region", "dest_region"]
            )
            for col in ("start_region", "dest_region"):
                if col in tx_df.columns:
                    known_regions.update(tx_df[col].dropna().astype(str))
        except Exception:
            pass

    # If no tables were available or all failed, skip silently
    if not known_regions:
        return results

    unknown = all_base_regions - known_regions
    if unknown:
        # Build per-model-region detail lines for actionable error messages
        detail_lines: List[str] = []
        # Regions from region_aggregations values
        for model_region, base_list in region_aggregations.items():
            if not isinstance(base_list, (list, tuple)):
                continue
            bad = sorted(str(r) for r in base_list if str(r) in unknown)
            if bad:
                detail_lines.append(f"{model_region} (aggregation): {bad}")
        # Pass-through model regions
        bad_passthrough = sorted(r for r in passthrough_regions if r in unknown)
        for r in bad_passthrough:
            detail_lines.append(f"{r} (model_regions pass-through): not found in data")
        results.append(
            _warn(
                "aggregation_base_regions",
                f"{len(unknown)} base region(s) referenced in 'model_regions' or "
                f"'region_aggregations' do not appear in any data table "
                f"(plant_region, demand, fuel_price, transmission_cost) — check for typos",
                detail="\n    ".join(detail_lines),
            )
        )

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2 entry point
# ──────────────────────────────────────────────────────────────────────────────

_PHASE2_CHECKS = [
    _check_data_tables_loaded,
    _check_aggregation_base_regions,
    _check_transmission_regions,
    _check_fuel_price_coverage,
    _check_new_resource_cost_years,
]


def validate_settings_with_data(
    settings: Any, data_manager: Any
) -> List[ValidationResult]:
    """Phase 2: validate settings against loaded data tables.

    Requires ``DataManager`` to already be initialised (call
    ``initialize_data_manager()`` first).

    Parameters
    ----------
    settings : dict or Settings
        Loaded PowerGenome settings.
    data_manager : DataManager
        Initialised DataManager singleton.

    Returns
    -------
    list[ValidationResult]
        All detected issues.  An empty list means no problems were found.
    """
    d = _settings_as_dict(settings)
    results: List[ValidationResult] = []
    for check_fn in _PHASE2_CHECKS:
        try:
            results.extend(check_fn(d, data_manager))
        except Exception as exc:
            logger.debug(
                "Validation check %s raised an unexpected exception",
                check_fn.__name__,
                exc_info=True,
            )
            results.append(
                _err(
                    "internal_error",
                    f"Validation check '{check_fn.__name__}' raised an unexpected "
                    f"exception — this is a bug; some checks may have been skipped",
                    detail=f"{type(exc).__name__}: {exc}",
                )
            )
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────────────


def report_validation_results(
    results: List[ValidationResult],
    raise_on_error: bool = True,
) -> None:
    """Log all validation results and optionally raise on ERRORs.

    Parameters
    ----------
    results : list[ValidationResult]
        Results from :func:`validate_settings` or
        :func:`validate_settings_with_data`.
    raise_on_error : bool
        If ``True`` (default), raise ``ValueError`` if any ERROR-level results
        are present.

    Raises
    ------
    ValueError
        If any ERROR-level results are present and *raise_on_error* is ``True``.
    """
    warnings_ = [r for r in results if r.level == ValidationLevel.WARNING]
    errors_ = [r for r in results if r.level == ValidationLevel.ERROR]

    for result in warnings_:
        logger.warning(str(result))
    for result in errors_:
        logger.error(str(result))

    if raise_on_error and errors_:
        error_messages = "\n".join(str(r) for r in errors_)
        raise ValueError(
            f"PowerGenome settings validation found {len(errors_)} error(s):\n"
            f"{error_messages}"
        )


def _print_phase_results(
    results: List[ValidationResult], phase_name: str
) -> Tuple[int, int]:
    """Pretty-print validation results for one phase to stdout.

    Groups ERROR and WARNING results into clearly labelled sections with
    consistent indentation.  Returns ``(n_errors, n_warnings)``.
    """
    errors_ = [r for r in results if r.level == ValidationLevel.ERROR]
    warnings_ = [r for r in results if r.level == ValidationLevel.WARNING]
    n_errors, n_warnings = len(errors_), len(warnings_)

    sep = "─" * 64
    print(f"\n{sep}")
    print(f"  {phase_name}")
    print(f"{sep}\n")

    if not results:
        print("  No issues found.\n")
        return 0, 0

    if errors_:
        print("  ERRORS — must be corrected:\n")
        for r in errors_:
            print(f"    ✗  [{r.category}]  {r.message}")
            if r.detail:
                for line in r.detail.splitlines():
                    print(f"         {line.strip()}")
            print()

    if warnings_:
        print("  WARNINGS — may need to be addressed:\n")
        for r in warnings_:
            print(f"    ⚠  [{r.category}]  {r.message}")
            if r.detail:
                for line in r.detail.splitlines():
                    print(f"         {line.strip()}")
            print()

    summary_parts: List[str] = []
    if n_errors:
        summary_parts.append(f"{n_errors} error(s)")
    if n_warnings:
        summary_parts.append(f"{n_warnings} warning(s)")
    print(f"  Result: {', '.join(summary_parts) if summary_parts else 'no issues'}\n")

    return n_errors, n_warnings


# ──────────────────────────────────────────────────────────────────────────────
# Standalone CLI  (validate_powergenome)
# ──────────────────────────────────────────────────────────────────────────────


def _parse_validate_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="validate_powergenome",
        description=(
            "Validate PowerGenome settings for common configuration issues.\n\n"
            "Phase 1 (always run): checks settings internal consistency without "
            "reading any data files.\n"
            "Phase 2 (default): checks settings against loaded data tables via "
            "DataManager."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-sf",
        "--settings_file",
        dest="settings_file",
        type=str,
        required=True,
        help="Path to settings YAML file or directory.",
    )
    parser.add_argument(
        "--skip-data-checks",
        dest="skip_data_checks",
        action="store_true",
        default=False,
        help="Only run Phase 1 (settings-only) checks — skip DataManager-based checks.",
    )
    parser.add_argument(
        "--no-fail",
        dest="no_fail",
        action="store_true",
        default=False,
        help=(
            "Exit with code 0 even if errors are found.  "
            "Errors are still logged at ERROR level."
        ),
    )
    return parser.parse_args(argv)


def validate_powergenome() -> None:
    """Entry point for the ``validate_powergenome`` CLI command."""
    _logging = logging
    _log = _logging.getLogger("powergenome")
    _log.setLevel(_logging.DEBUG)
    handler = _logging.StreamHandler()
    handler.setFormatter(_logging.Formatter("%(levelname)-8s %(name)s: %(message)s"))
    handler.setLevel(_logging.INFO)
    _log.addHandler(handler)

    args = _parse_validate_args()
    has_errors = False

    # ── Phase 1 ────────────────────────────────────────────────────────────────
    from powergenome.settings import Settings

    logger.info("Loading settings from: %s", args.settings_file)
    settings = Settings(config_path=args.settings_file)

    logger.info("Running Phase 1 validation (settings-only checks) …")
    p1_results = validate_settings(settings)
    p1_errs, _ = _print_phase_results(p1_results, "Phase 1: Settings checks")
    has_errors = p1_errs > 0

    # ── Phase 2 ────────────────────────────────────────────────────────────────
    if not args.skip_data_checks:
        data_location = settings.get("data_location")
        if not data_location:
            logger.warning(
                "Skipping Phase 2 data checks: 'data_location' is not configured."
            )
        else:
            from powergenome.database import _data_manager, initialize_data_manager

            logger.info("Running Phase 2 validation (data checks) …")
            try:
                initialize_data_manager(settings, data_location)
                p2_results = validate_settings_with_data(settings, _data_manager)
                p2_errs, _ = _print_phase_results(p2_results, "Phase 2: Data checks")
                has_errors = has_errors or p2_errs > 0
            except Exception as exc:
                logger.error("Phase 2 validation failed unexpectedly: %s", exc)
                has_errors = True

    if has_errors and not args.no_fail:
        sys.exit(1)
