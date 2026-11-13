"""Shared matching logic for region-scoped new-build technology availability.

``new_resources`` is a flat, region-agnostic list: every entry is replicated to
all model regions when new-build generators are created. The
``new_gen_not_available`` setting is the supported way to keep a technology out
of a *subset* of regions, and this module holds the matching logic that both
:mod:`powergenome.new_build` (which applies the exclusions while building
resources) and :mod:`powergenome.validate` (which reports exclusion patterns
that can never match) share.

Keeping the logic in one place means validation and resource building cannot
disagree about what a pattern means. The module deliberately imports nothing
outside the standard library: Phase 1 validation runs without a ``DataManager``,
so importing pandas or duckdb here would make settings validation drag in the
whole data stack.

Matching rules
--------------
* A pattern is matched against the full resource name
  ``"<technology>_<tech_detail>_<cost_case>"`` — the same string that appears in
  the ``technology`` column of the generators dataframe.
* Patterns are case-insensitive ``fnmatch`` globs (``*``, ``?``, ``[seq]``). A
  pattern without a wildcard is therefore an exact-name match.
* The special region keys ``all`` and ``all_regions`` apply to every model
  region. They are additive with per-region keys rather than overriding them.

Example
-------
::

    from powergenome.tech_availability import is_technology_excluded

    settings = {
        "new_gen_not_available": {
            "ALL_REGIONS": ["*_CCS"],
            "AZ": ["OffshoreWind_*"],
        }
    }
    is_technology_excluded("OffshoreWind_Class1_Moderate", settings, "AZ")  # True
    is_technology_excluded("OffshoreWind_Class1_Moderate", settings, "CA_N")  # False
"""

from __future__ import annotations

import fnmatch
import logging
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

logger = logging.getLogger(__name__)

#: Region keys that apply to every model region. Compared case-insensitively.
ALL_REGION_KEYS = ("all", "all_regions")


def _setting(settings: Any, key: str) -> Any:
    """Return a raw settings value from a dict or ``Settings`` object.

    Parameters
    ----------
    settings : dict or Settings
        Loaded PowerGenome settings.
    key : str
        Settings key to look up.

    Returns
    -------
    Any
        The value stored under *key*, or None when *settings* is None or the key
        is missing.
    """
    if settings is None:
        return None
    if not hasattr(settings, "get"):
        raise TypeError(
            "settings must be a dict or Settings object, got "
            f"{type(settings).__name__}"
        )
    return settings.get(key)


def _as_text(value: Any) -> str:
    """Return *value* as text, mapping ``None`` and NaN-like values to ``""``.

    Parameters
    ----------
    value : Any
        A single part of a resource key (technology, tech_detail, cost_case).

    Returns
    -------
    str
        ``str(value)``, or an empty string for ``None``/NaN. The self-comparison
        test catches floating-point NaN without importing numpy or pandas;
        objects that raise on that comparison (e.g. ``pd.NA``) fall through to
        ``str()``.
    """
    if value is None:
        return ""
    try:
        if value != value:
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def technology_full_name(
    technology: Any,
    tech_detail: Any,
    cost_case: Any,
) -> str:
    """Concatenate the three resource key parts into a full resource name.

    Matches how :func:`powergenome.new_build.build_new_resources` builds the
    ``technology`` column with ``"_".join`` over the three key columns, so empty
    parts are still joined.

    Parameters
    ----------
    technology : Any
        Resource technology name, e.g. ``"UtilityPV"``.
    tech_detail : Any
        Resource technology detail, e.g. ``"Class1"``.
    cost_case : Any
        Resource cost case, e.g. ``"Moderate"``.

    Returns
    -------
    str
        The parts joined with ``"_"``. ``None``/NaN parts become empty strings,
        so ``("UtilityPV", "", "Moderate")`` gives ``"UtilityPV__Moderate"``.
    """
    parts = (technology, tech_detail, cost_case)
    return "_".join(_as_text(part) for part in parts)


def is_all_regions_key(region: Any) -> bool:
    """Return True if *region* is one of the all-regions wildcard keys.

    Parameters
    ----------
    region : Any
        A key from ``new_gen_not_available``. Non-string values are never
        wildcard keys.

    Returns
    -------
    bool
        True for ``all``/``all_regions`` in any casing and with surrounding
        whitespace, e.g. ``"ALL_REGIONS"`` or ``" All "``.
    """
    if not isinstance(region, str):
        return False
    return region.strip().lower() in ALL_REGION_KEYS


def _normalize_patterns(region: Any, value: Any) -> list[str]:
    """Normalize one region's exclusion value into a de-duplicated pattern list.

    Parameters
    ----------
    region : Any
        The region key the value belongs to, used only for log messages.
    value : Any
        The raw value from ``new_gen_not_available``.

    Returns
    -------
    list[str]
        Stripped, de-duplicated patterns in the order given. Unsupported values
        and non-string entries are logged at warning level and skipped; a single
        string is treated as a one-element list.
    """
    if value is None:
        return []
    if isinstance(value, str):
        values: list[Any] = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = list(value)
    else:
        logger.warning(
            "'new_gen_not_available' entry for region %r has unsupported value "
            "%r; expected a list of technology name patterns. Skipping.",
            region,
            value,
        )
        return []

    patterns: list[str] = []
    for pattern in values:
        if not isinstance(pattern, str):
            logger.warning(
                "'new_gen_not_available' entry for region %r contains a "
                "non-string pattern %r; skipping it.",
                region,
                pattern,
            )
            continue
        stripped = pattern.strip()
        if stripped and stripped not in patterns:
            patterns.append(stripped)
    return patterns


def normalize_exclusions(settings: Any) -> dict[str, list[str]]:
    """Normalize the ``new_gen_not_available`` setting into pattern lists.

    Parameters
    ----------
    settings : dict or Settings
        Loaded PowerGenome settings.

    Returns
    -------
    dict[str, list[str]]
        ``{region_key: [pattern, ...]}`` with region keys kept as given (only
        surrounding whitespace stripped) and patterns stripped, de-duplicated
        and in their original order. Keys with no usable patterns are omitted.
        A missing, empty or non-mapping setting gives an empty dict; a bare
        string value is accepted as a one-element list, while other scalar
        values are skipped with a warning.
    """
    raw = _setting(settings, "new_gen_not_available")
    if not raw:
        return {}
    if not isinstance(raw, Mapping):
        logger.warning(
            "'new_gen_not_available' should map region names to lists of "
            "technology name patterns; ignoring value of type %s.",
            type(raw).__name__,
        )
        return {}

    normalized: dict[str, list[str]] = {}
    for region, value in raw.items():
        region_key = region.strip() if isinstance(region, str) else region
        patterns = _normalize_patterns(region_key, value)
        if patterns:
            normalized[region_key] = patterns
    return normalized


def exclusion_patterns_for_region(settings: Any, region: Any) -> list[str]:
    """Return every exclusion pattern that applies to *region*.

    Parameters
    ----------
    settings : dict or Settings
        Loaded PowerGenome settings.
    region : str
        Model region name.

    Returns
    -------
    list[str]
        The region's own patterns followed by the patterns of every
        ``all``/``all_regions`` entry. Empty when nothing applies.
    """
    normalized = normalize_exclusions(settings)
    if not normalized:
        return []

    patterns: list[str] = []
    if isinstance(region, str):
        patterns.extend(normalized.get(region, []))
    for key, key_patterns in normalized.items():
        if is_all_regions_key(key):
            patterns.extend(key_patterns)
    return patterns


def matches_any_pattern(name: Any, patterns: Iterable[str] | None) -> bool:
    """Case-insensitively match *name* against any of *patterns*.

    Parameters
    ----------
    name : str
        Full resource name, e.g. ``"OffshoreWind_Class1_Moderate"``.
    patterns : iterable of str or None
        ``fnmatch``-style globs. A glob without wildcards is an exact match.

    Returns
    -------
    bool
        True if any pattern matches. False for an empty/None name or no
        patterns. Both sides are lowercased before ``fnmatch.fnmatchcase`` so
        behaviour is identical on every platform.
    """
    if not name or not patterns:
        return False

    target = str(name).lower()
    for pattern in patterns:
        if not pattern:
            continue
        if fnmatch.fnmatchcase(target, str(pattern).lower()):
            return True
    return False


def is_technology_excluded(name: Any, settings: Any, region: Any) -> bool:
    """Return True if *name* is excluded from *region* by the settings.

    Parameters
    ----------
    name : str
        Full resource name, e.g. ``"Coal_newAvgCF_Moderate"``.
    settings : dict or Settings
        Loaded PowerGenome settings.
    region : str
        Model region the resource would be built in.

    Returns
    -------
    bool
        True when any pattern applying to *region* matches *name*.
    """
    return matches_any_pattern(name, exclusion_patterns_for_region(settings, region))


def _dedupe(items: Iterable[str]) -> list[str]:
    """Return *items* without duplicates, preserving first-seen order."""
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def _resource_key_parts(entry: Any) -> tuple[Any, Any, Any] | None:
    """Return the first three parts of a ``new_resources`` entry, or None.

    Parameters
    ----------
    entry : Any
        One entry of the ``new_resources`` list, normally
        ``[technology, tech_detail, cost_case, size_mw]``.

    Returns
    -------
    tuple or None
        ``(technology, tech_detail, cost_case)``, or None (with a warning) when
        the entry is not a sequence of at least three items.
    """
    if isinstance(entry, (str, bytes, bytearray)) or not isinstance(entry, Sequence):
        logger.warning(
            "Skipping 'new_resources' entry %r: expected a sequence of "
            "[technology, tech_detail, cost_case, size_mw].",
            entry,
        )
        return None
    if len(entry) < 3:
        logger.warning(
            "Skipping 'new_resources' entry %r: expected at least three items "
            "(technology, tech_detail, cost_case).",
            entry,
        )
        return None
    return entry[0], entry[1], entry[2]


def new_resource_names(settings: Any) -> list[str]:
    """List every new-build resource name the settings could create.

    This is a *static* helper used for advisory validation. It covers
    ``new_resources`` and ``modified_new_resources`` but cannot resolve names
    contributed by ``additional_technologies_fn`` CSVs, so callers must treat
    unmatched-pattern warnings as advisory.

    Parameters
    ----------
    settings : dict or Settings
        Loaded PowerGenome settings.

    Returns
    -------
    list[str]
        Full resource names in original case, de-duplicated, in settings order.
        ``modified_new_resources`` names are built from the renamed
        ``new_technology``/``new_tech_detail``/``new_cost_case`` fields produced
        by :func:`powergenome.new_build.add_modified_generators`.
    """
    names: list[str] = []

    raw_new_resources = _setting(settings, "new_resources")
    if raw_new_resources:
        if isinstance(raw_new_resources, (str, bytes, Mapping)) or not isinstance(
            raw_new_resources, Sequence
        ):
            logger.warning(
                "'new_resources' should be a list of "
                "[technology, tech_detail, cost_case, size_mw] entries; "
                "ignoring value of type %s.",
                type(raw_new_resources).__name__,
            )
        else:
            for entry in raw_new_resources:
                parts = _resource_key_parts(entry)
                if parts is not None:
                    names.append(technology_full_name(*parts))

    modified = _setting(settings, "modified_new_resources")
    if modified:
        if not isinstance(modified, Mapping):
            logger.warning(
                "'modified_new_resources' should map names to resource "
                "definitions; ignoring value of type %s.",
                type(modified).__name__,
            )
        else:
            for name, spec in modified.items():
                if not isinstance(spec, Mapping):
                    logger.warning(
                        "Skipping 'modified_new_resources' entry %r: expected a "
                        "mapping, got %s.",
                        name,
                        type(spec).__name__,
                    )
                    continue
                names.append(
                    technology_full_name(
                        spec.get("new_technology", ""),
                        spec.get("new_tech_detail", ""),
                        spec.get("new_cost_case", ""),
                    )
                )

    return _dedupe(names)


def unmatched_patterns(settings: Any) -> list[tuple[str, str]]:
    """Find exclusion patterns that match no new-build resource name.

    Parameters
    ----------
    settings : dict or Settings
        Loaded PowerGenome settings.

    Returns
    -------
    list[tuple[str, str]]
        ``(region_key, pattern)`` pairs, sorted by region key then pattern, for
        patterns that match none of :func:`new_resource_names`. Empty when
        there are no known names to match against (nothing to judge) or when
        every pattern matches.
    """
    names = new_resource_names(settings)
    if not names:
        return []

    unmatched: list[tuple[str, str]] = []
    for region_key, patterns in normalize_exclusions(settings).items():
        for pattern in patterns:
            if not any(matches_any_pattern(name, [pattern]) for name in names):
                unmatched.append((str(region_key), pattern))
    return sorted(unmatched)
