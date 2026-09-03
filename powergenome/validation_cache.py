"""
Input fingerprinting and result caching for PowerGenome validation.

Phase 1 and Phase 2 validation (``powergenome.validate``) used to run on every
invocation even when neither the settings nor the data files changed.  This
module computes a lightweight fingerprint of everything the validation checks
depend on:

* the resolved settings dict (canonical JSON),
* the file-system paths the checks touch (existence and content fingerprint),
* the data files that ``DataManager`` resolves for the configured tables, and
* the validation code itself (source hash + package version + cache format
  version).

If the fingerprint is unchanged since a previous run, the stored
``ValidationResult`` list is replayed instead of recomputed — warnings are
still printed and cached ERRORs still fail the run; only the check computation
is skipped (and, in the standalone ``validate_powergenome`` CLI, the
``initialize_data_manager()`` call as well).

File fingerprints never hash whole files: large profile/parquet sources can be
multi-GB.  The fingerprint covers the file size plus the first and last 1 MB of
content.  This will miss a same-size edit confined to the middle of a file
larger than 2 MB — delete the ``validation_cache`` folder under
``input_folder`` to force re-validation after such an edit.

Cache entries are JSON files written to ``<input_folder>/validation_cache/``,
alongside the existing ``cluster_assignments`` cache.  Caching is disabled when
no ``input_folder`` is configured.  Set ``use_validation_cache: false`` in a
settings file (or pass ``--no-validation-cache``) to disable it entirely.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)

# Bump when the cache payload format or the fingerprinting scheme changes.
CACHE_FORMAT_VERSION = 1

# Number of bytes sampled from the head and tail of each data file.
SAMPLE_BYTES = 1024 * 1024

_DB_SUFFIXES = (".db", ".sqlite", ".duckdb")
_TABULAR_SUFFIXES = (".csv", ".parquet")

# Data-file existence markers used when a file cannot contribute content.
_MISSING = "missing"
_UNREADABLE = "unreadable"


def _as_bool(value: Any, default: bool = True) -> bool:
    """Coerce a settings value to bool, accepting common string forms."""
    if value is None:
        return default
    if isinstance(value, (bool, int)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


# ──────────────────────────────────────────────────────────────────────────────
# Fingerprints
# ──────────────────────────────────────────────────────────────────────────────


def fingerprint_file(file_path: Union[Path, str, None]) -> str:
    """Fingerprint a possibly very large file without hashing the whole thing.

    The digest covers the file size and the first and last 1 MB of content
    (the entire body for files of 2 MB or less).  Touching a file (mtime-only
    change) does not alter the fingerprint; content edits within the sampled
    regions or a size change do.

    Returns
    -------
    str
        ``"<size>_<sha256[:16]>"``, or ``"missing"``/``"unreadable"`` when the
        file cannot be read.
    """
    if not file_path:
        return _MISSING
    path = Path(file_path)
    try:
        size = path.stat().st_size
    except OSError:
        return _MISSING

    sha256_hash = hashlib.sha256()
    sha256_hash.update(str(size).encode())
    try:
        with open(path, "rb") as f:
            if size <= 2 * SAMPLE_BYTES:
                sha256_hash.update(f.read())
            else:
                sha256_hash.update(f.read(SAMPLE_BYTES))
                f.seek(-SAMPLE_BYTES, 2)
                sha256_hash.update(f.read(SAMPLE_BYTES))
    except OSError:
        return _UNREADABLE

    return f"{size}_{sha256_hash.hexdigest()[:16]}"


def _path_marker(path: Union[Path, str]) -> str:
    """Existence/content marker for a settings path of unknown type."""
    p = Path(path)
    if p.is_dir():
        return "dir"
    if p.is_file():
        return fingerprint_file(p)
    return _MISSING


def _canonicalize(value: Any) -> Any:
    """Return a JSON-serializable, order-canonical form of *value*.

    Settings can contain plain ``set`` values and lists derived from sets
    (e.g. ``generator_columns``); their iteration order varies between
    processes because of string hash randomization, which would produce a
    different fingerprint on every run.  Sets become sorted lists of their
    canonicalized members, and lists made entirely of strings are sorted
    (the validation checks only ever test membership for such lists, so
    permutation cannot change their results).  Lists that mix in numbers or
    other types keep their order — year-pairing checks are order-sensitive.
    """
    if isinstance(value, dict):
        return {
            str(k): _canonicalize(v)
            for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(value, (set, frozenset)):
        return sorted(
            json.dumps(_canonicalize(item), sort_keys=True, default=_json_default)
            for item in value
        )
    if isinstance(value, (list, tuple)):
        items = [_canonicalize(item) for item in value]
        if items and all(isinstance(item, str) for item in items):
            items = sorted(items)
        return items
    return value


def _json_default(obj: Any) -> Any:
    """Serialize set-like and arbitrary objects deterministically."""
    if isinstance(obj, (set, frozenset)):
        return sorted(
            json.dumps(_canonicalize(item), sort_keys=True, default=_json_default)
            for item in obj
        )
    return str(obj)


def fingerprint_settings(settings: Dict[str, Any]) -> str:
    """Hash the fully resolved settings dict in canonical JSON form."""
    canonical = json.dumps(
        _canonicalize(settings), sort_keys=True, default=_json_default
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def fingerprint_paths(settings: Dict[str, Any]) -> str:
    """Hash the paths that Phase 1's ``_check_paths_exist`` touches.

    Existence changes (a path appearing or disappearing) must invalidate the
    Phase 1 cache even though the settings text is unchanged.
    """
    parts: List[str] = []
    for key in ("data_location", "RESOURCE_GROUPS", "RESOURCE_GROUP_PROFILES"):
        val = settings.get(key)
        if val is None:
            continue
        values = val if isinstance(val, (list, tuple)) else [val]
        for value in values:
            parts.append(f"{key}:{value}:{_path_marker(value)}")

    input_folder = settings.get("input_folder")
    if input_folder:
        parts.append(f"input_folder:{input_folder}:{_path_marker(input_folder)}")
        scenario_fn = settings.get("scenario_definitions_fn")
        if scenario_fn:
            scenario_path = Path(input_folder) / scenario_fn
            parts.append(
                f"scenario_definitions_fn:{scenario_path}:"
                f"{fingerprint_file(scenario_path)}"
            )

    joined = "|".join(parts)
    return hashlib.sha256(joined.encode()).hexdigest()[:16]


def code_fingerprint() -> str:
    """Hash of the validation module source + package + cache format versions.

    Any change to the validation checks (or this cache format) invalidates
    previously stored results.
    """
    from powergenome import __version__

    try:
        import powergenome.validate as validate_module

        source = inspect.getsource(validate_module)
    except (TypeError, OSError):  # pragma: no cover - frozen/optimized installs
        source = ""

    payload = f"{CACHE_FORMAT_VERSION}|{__version__}|{source}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _table_source_locations(settings: Dict[str, Any]) -> List[Path]:
    """Data locations ``DataManager.initialize`` would consider, from settings."""
    locations: List[Path] = []
    for key in ("data_location", "input_folder"):
        val = settings.get(key)
        if val is None:
            continue
        values = val if isinstance(val, (list, tuple)) else [val]
        for value in values:
            path = Path(value)
            if path not in locations:
                locations.append(path)
    return locations


def _resolve_table_sources(locations: Sequence[Path], source: str) -> List[str]:
    """Resolve a configured table source to the backing file path.

    Mirrors ``DataManager._validate_table_config``/``_resolve_table_source``
    existence semantics without loading any data: a directory location
    contributes ``<location>/<source>`` (auto-detecting a .csv/.parquet
    extension), while a database-file location contributes the database file
    itself (or a tabular file located next to it).
    """
    matches: List[str] = []
    for location in locations:
        if location.is_dir():
            candidate = location / source
            if candidate.is_file():
                matches.append(str(candidate))
            elif Path(source).suffix == "":
                for extension in _TABULAR_SUFFIXES:
                    with_ext = location / f"{source}{extension}"
                    if with_ext.is_file():
                        matches.append(str(with_ext))
                        break
        elif location.is_file() and location.suffix.lower() in _DB_SUFFIXES:
            if source.lower().endswith(_TABULAR_SUFFIXES):
                neighbor = location.parent / source
                if neighbor.is_file():
                    matches.append(str(neighbor))
            elif Path(source).suffix == "":
                # Table inside the database file — the DB file is the input.
                matches.append(str(location))
    return matches


def resolve_data_files(settings: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Map each configured DataManager table to the file backing it.

    Returns a sorted list of ``(standard_table_name, file_path_or_marker)``
    pairs.  Resolution uses file-existence checks only; no data is read and no
    ``DataManager`` instance is required.
    """
    # Imported lazily so that importing this module stays light (database.py
    # pulls in transmission and other heavy pipeline modules).
    from powergenome.database import DataManager, get_table_setting_value

    locations = _table_source_locations(settings)
    entries: List[Tuple[str, str]] = []
    for setting_key, standard_name in DataManager.STANDARD_TABLE_MAPPING.items():
        table_config = get_table_setting_value(settings, setting_key)
        if not table_config:
            continue
        if isinstance(table_config, dict):
            source = table_config.get("table_name") or table_config.get("name")
        else:
            source = table_config
        if not source:
            continue
        matches = _resolve_table_sources(locations, str(source))
        if not matches:
            entries.append((standard_name, "unresolved"))
        else:
            entries.extend((standard_name, match) for match in matches)
    return sorted(entries)


def fingerprint_data_files(
    settings: Dict[str, Any],
) -> Tuple[str, List[str]]:
    """Fingerprint every data file the Phase 2 checks can observe.

    Returns ``(hash, audit_parts)`` where ``audit_parts`` lists
    ``table=path:size+content`` entries so a cache file records exactly which
    inputs produced its key.
    """
    audit_parts: List[str] = []
    for standard_name, resolved in resolve_data_files(settings):
        marker = _MISSING if resolved == "unresolved" else fingerprint_file(resolved)
        audit_parts.append(f"{standard_name}={resolved}:{marker}")
    joined = "|".join(audit_parts)
    digest = hashlib.sha256(joined.encode()).hexdigest()[:16]
    return digest, audit_parts


# ──────────────────────────────────────────────────────────────────────────────
# Cache key and storage
# ──────────────────────────────────────────────────────────────────────────────


def validation_input_fingerprints(
    phase: str, settings: Dict[str, Any]
) -> Dict[str, str]:
    """Compute every fingerprint for ``phase`` ("phase1" or "phase2").

    Phase 1 depends on code + settings + path existence; Phase 2 additionally
    depends on the contents of the data files behind the configured tables.
    """
    if phase not in ("phase1", "phase2"):
        raise ValueError(f"phase must be 'phase1' or 'phase2', got {phase!r}")

    fingerprints = {
        "code": code_fingerprint(),
        "settings": fingerprint_settings(settings),
        "paths": fingerprint_paths(settings),
    }
    if phase == "phase2":
        data_hash, data_parts = fingerprint_data_files(settings)
        fingerprints["data_files"] = data_hash
        fingerprints["_data_parts"] = "\n".join(data_parts)
    fingerprints["key"] = hashlib.sha256(
        "|".join(v for k, v in fingerprints.items() if not k.startswith("_")).encode()
    ).hexdigest()[:32]
    return fingerprints


def cache_folder(settings: Dict[str, Any]) -> Optional[Path]:
    """Return the validation cache folder, or None when caching is unavailable."""
    input_folder = settings.get("input_folder")
    if not input_folder:
        return None
    return Path(input_folder) / "validation_cache"


def _serialize_results(results: List[Any]) -> List[Dict[str, Optional[str]]]:
    return [
        {
            "level": str(r.level.value if hasattr(r.level, "value") else r.level),
            "category": r.category,
            "message": r.message,
            "detail": r.detail,
        }
        for r in results
    ]


def _deserialize_results(payload: List[Dict[str, Any]]) -> List[Any]:
    from powergenome.validate import ValidationLevel, ValidationResult

    return [
        ValidationResult(
            level=ValidationLevel(item["level"]),
            category=item["category"],
            message=item["message"],
            detail=item.get("detail"),
        )
        for item in payload
    ]


def _cache_path(settings: Dict[str, Any], phase: str, key: str) -> Optional[Path]:
    folder = cache_folder(settings)
    if folder is None:
        return None
    return folder / f"{phase}_{key}.json"


def load_cached_results(
    phase: str,
    settings: Dict[str, Any],
    fingerprints: Optional[Dict[str, str]] = None,
) -> Optional[List[Any]]:
    """Return stored ``ValidationResult``s for the current inputs, else None."""
    if fingerprints is None:
        fingerprints = validation_input_fingerprints(phase, settings)
    path = _cache_path(settings, phase, fingerprints["key"])
    if path is None or not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("key") != fingerprints["key"]:
            return None
        return _deserialize_results(payload["results"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.debug("Ignoring unreadable validation cache file %s: %s", path, exc)
        return None


def save_cached_results(
    phase: str,
    settings: Dict[str, Any],
    results: List[Any],
    fingerprints: Optional[Dict[str, str]] = None,
) -> None:
    """Store validation results keyed by the current input fingerprints."""
    if fingerprints is None:
        fingerprints = validation_input_fingerprints(phase, settings)
    path = _cache_path(settings, phase, fingerprints["key"])
    if path is None:
        return
    payload = {
        "phase": phase,
        "key": fingerprints["key"],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "fingerprints": {
            k: v for k, v in fingerprints.items() if not k.startswith("_")
        },
        "data_files": fingerprints.get("_data_parts", ""),
        "results": _serialize_results(results),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.debug("Wrote validation cache entry %s", path)
    except (OSError, TypeError, ValueError) as exc:
        logger.warning("Could not write validation cache %s: %s", path, exc)


def run_cached_validation(
    phase: str,
    settings: Dict[str, Any],
    compute_fn: Callable[[], List[Any]],
    use_cache: bool = True,
) -> Tuple[List[Any], bool]:
    """Run ``compute_fn`` unless identical inputs were validated before.

    Parameters
    ----------
    phase : str
        ``"phase1"`` or ``"phase2"``.
    settings : dict
        Plain resolved settings dictionary (already converted via
        ``validate._settings_as_dict``).
    compute_fn : callable
        Returns the list of ``ValidationResult`` for this phase.  On a Phase 2
        cache miss in the standalone CLI this callable may initialize the
        ``DataManager`` itself.
    use_cache : bool
        When False, always computes and never reads or writes the cache.

    Returns
    -------
    tuple[list[ValidationResult], bool]
        The results and whether they were replayed from the cache.
    """
    if not use_cache:
        return compute_fn(), False

    results_: Optional[List[Any]] = None
    try:
        fingerprints = validation_input_fingerprints(phase, settings)
        # A None cache folder means caching is unavailable (no input_folder).
        if _cache_path(settings, phase, fingerprints["key"]) is not None:
            results_ = load_cached_results(phase, settings, fingerprints)
    except Exception as exc:  # fingerprinting must never break validation
        logger.debug("Validation cache lookup failed, running checks: %s", exc)
        return compute_fn(), False

    if results_ is not None:
        logger.debug("Validation cache hit for %s (key=%s)", phase, fingerprints["key"])
        return results_, True

    results = compute_fn()

    try:
        save_cached_results(phase, settings, results, fingerprints)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Could not save validation cache: %s", exc)

    return results, False
