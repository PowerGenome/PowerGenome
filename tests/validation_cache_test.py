"""Tests for powergenome.validation_cache — input fingerprinting and result cache."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from powergenome.validate import (
    ValidationLevel,
    ValidationResult,
    _parse_validate_args,
    report_validation_results,
)
from powergenome.validation_cache import (
    cache_folder,
    fingerprint_data_files,
    fingerprint_file,
    fingerprint_paths,
    fingerprint_settings,
    load_cached_results,
    resolve_data_files,
    run_cached_validation,
    save_cached_results,
    validation_input_fingerprints,
)


def _mk_results():
    return [
        ValidationResult(ValidationLevel.WARNING, "cat_a", "warning one", "detail"),
        ValidationResult(ValidationLevel.ERROR, "cat_b", "error two"),
    ]


def _base_settings(tmp_path, **overrides):
    """Settings dict that enables the cache: input_folder + data_location exist."""
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    (tmp_path / "input").mkdir(exist_ok=True)
    settings = {
        "input_folder": str(tmp_path / "input"),
        "data_location": str(data),
    }
    settings.update(overrides)
    return settings


# ─────────────────────────────────────────────────────────────────────────────
# fingerprint_file
# ─────────────────────────────────────────────────────────────────────────────


class TestFingerprintFile:
    def test_stable_across_touch(self, tmp_path):
        f = tmp_path / "x.csv"
        f.write_text("a,b\n1,2\n")
        before = fingerprint_file(f)
        os.utime(f, (0, 0))
        assert fingerprint_file(f) == before

    def test_changes_on_content_edit(self, tmp_path):
        f = tmp_path / "x.csv"
        f.write_text("a,b\n1,2\n")
        before = fingerprint_file(f)
        f.write_text("a,b\n3,4\n")
        assert fingerprint_file(f) != before

    def test_missing_file(self, tmp_path):
        assert fingerprint_file(tmp_path / "nope.csv") == "missing"
        assert fingerprint_file(None) == "missing"

    def test_large_file_head_change(self, tmp_path):
        f = tmp_path / "big.bin"
        data = bytearray(b"0" * (3 * 1024 * 1024))
        f.write_bytes(bytes(data))
        before = fingerprint_file(f)
        data[10] = 49  # change within the first 1 MB
        f.write_bytes(bytes(data))
        assert fingerprint_file(f) != before

    def test_large_file_tail_change(self, tmp_path):
        f = tmp_path / "big.bin"
        data = bytearray(b"0" * (3 * 1024 * 1024))
        f.write_bytes(bytes(data))
        before = fingerprint_file(f)
        data[-11] = 49  # change within the last 1 MB
        f.write_bytes(bytes(data))
        assert fingerprint_file(f) != before

    def test_middle_edit_of_large_file_detected(self, tmp_path):
        """Same-size edits anywhere in a file invalidate its fingerprint."""
        f = tmp_path / "big.bin"
        data = bytearray(b"0" * (4 * 1024 * 1024))
        f.write_bytes(bytes(data))
        before = fingerprint_file(f)
        mid = 2 * 1024 * 1024
        data[mid] = 49
        f.write_bytes(bytes(data))
        assert fingerprint_file(f) != before


# ─────────────────────────────────────────────────────────────────────────────
# Other fingerprints
# ─────────────────────────────────────────────────────────────────────────────


class TestFingerprints:
    def test_settings_fingerprint_sensitive_to_value(self):
        s1 = {"model_regions": ["A", "B"], "model_year": 2030}
        s2 = {"model_regions": ["A", "C"], "model_year": 2030}
        assert fingerprint_settings(s1) == fingerprint_settings(dict(s1))
        assert fingerprint_settings(s1) != fingerprint_settings(s2)

    def test_settings_fingerprint_key_order_insensitive(self):
        assert fingerprint_settings({"a": 1, "b": 2}) == fingerprint_settings(
            {"b": 2, "a": 1}
        )

    def test_set_values_are_hash_seed_independent(self):
        """Set/list-derived values vary across processes (hash randomization);
        the settings fingerprint must stay stable anyway."""
        import subprocess
        import sys

        script = (
            "from powergenome.validation_cache import fingerprint_settings; "
            "print(fingerprint_settings({'cols': ['a', 'b', 'c', 'd', 'e', 'x'], "
            "'s': {'p', 'q', 'r'}}))"
        )
        digests = set()
        for seed in ("1", "2", "12345"):
            out = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                check=True,
                env={**os.environ, "PYTHONHASHSEED": seed},
            )
            digests.add(out.stdout.strip())
        assert (
            len(digests) == 1
        ), f"settings fingerprint varies with PYTHONHASHSEED: {digests}"

    def test_string_list_permutation_is_stable(self):
        """Membership-only string lists must not fingerprint by order."""
        assert fingerprint_settings({"x": ["a", "b"]}) == fingerprint_settings(
            {"x": ["b", "a"]}
        )

    def test_numeric_list_order_still_matters(self):
        """Year-pairing checks are order-sensitive; keep numeric lists ordered."""
        assert fingerprint_settings({"model_year": [2030, 2040]}) != (
            fingerprint_settings({"model_year": [2040, 2030]})
        )

    def test_path_fingerprint_detects_disappearing_path(self, tmp_path):
        d = tmp_path / "data"
        d.mkdir()
        settings = {"data_location": str(d)}
        before = fingerprint_paths(settings)
        os.rename(d, tmp_path / "data_renamed")
        assert fingerprint_paths(settings) != before

    def test_resolve_data_files(self, tmp_path):
        data = tmp_path / "data"
        data.mkdir()
        (data / "generation.csv").write_text("a\n1\n")
        (data / "demand_timeseries.parquet").write_bytes(b"PAR1")
        settings = {
            "data_location": str(data),
            "generation_table": "generation.csv",
            "demand_table": "demand_timeseries",  # extension auto-detected
            "fuel_price_table": "fuel_prices.csv",  # does not exist
        }
        resolved = dict(resolve_data_files(settings))
        assert resolved["generation"] == str(data / "generation.csv")
        assert resolved["demand"] == str(data / "demand_timeseries.parquet")
        assert resolved["fuel_price"] == "unresolved"

    def test_resolve_data_files_includes_duplicate_sources(self, tmp_path):
        first = tmp_path / "first"
        second = tmp_path / "second"
        first.mkdir()
        second.mkdir()
        (first / "generation.csv").write_text("a\n1\n")
        (second / "generation.csv").write_text("a\n2\n")
        settings = {
            "data_location": [str(first), str(second)],
            "generation_table": "generation.csv",
        }

        assert resolve_data_files(settings) == [
            ("generation", str(first / "generation.csv")),
            ("generation", str(second / "generation.csv")),
        ]

    def test_data_fingerprint_changes_on_file_edit(self, tmp_path):
        data = tmp_path / "data"
        data.mkdir()
        target = data / "generation.csv"
        target.write_text("a\n1\n")
        settings = {"data_location": str(data), "generation_table": "generation.csv"}
        before, _ = fingerprint_data_files(settings)
        target.write_text("a\n2\n")
        after, _ = fingerprint_data_files(settings)
        assert before != after

    def test_phase2_key_includes_data_phase1_does_not(self, tmp_path):
        settings = _base_settings(tmp_path, generation_table="generation.csv")
        target = Path(settings["data_location"]) / "generation.csv"
        target.write_text("a\n1\n")
        p1_before = validation_input_fingerprints("phase1", settings)["key"]
        p2_before = validation_input_fingerprints("phase2", settings)["key"]
        target.write_text("a\n2\n")
        assert validation_input_fingerprints("phase1", settings)["key"] == p1_before
        assert validation_input_fingerprints("phase2", settings)["key"] != p2_before


# ─────────────────────────────────────────────────────────────────────────────
# Cache round-trip
# ─────────────────────────────────────────────────────────────────────────────


class TestRunCachedValidation:
    def test_hit_skips_compute_and_replays_results(self, tmp_path):
        settings = _base_settings(tmp_path)
        calls = {"n": 0}

        def compute():
            calls["n"] += 1
            return _mk_results()

        r1, cached1 = run_cached_validation("phase1", settings, compute)
        r2, cached2 = run_cached_validation("phase1", settings, compute)
        assert cached1 is False and cached2 is True
        assert calls["n"] == 1
        assert [str(r) for r in r1] == [str(r) for r in r2]
        assert r2[0].level == ValidationLevel.WARNING
        assert r2[0].detail == "detail"
        assert r2[1].level == ValidationLevel.ERROR

    def test_cache_file_written_under_input_folder(self, tmp_path):
        settings = _base_settings(tmp_path)
        run_cached_validation("phase1", settings, lambda: [])
        folder = cache_folder(settings)
        entries = list(folder.glob("phase1_*.json"))
        assert len(entries) == 1
        payload = json.loads(entries[0].read_text())
        assert payload["phase"] == "phase1"
        assert "fingerprints" in payload and "created_at" in payload

    def test_use_cache_false_never_reads_or_writes(self, tmp_path):
        settings = _base_settings(tmp_path)
        calls = {"n": 0}

        def compute():
            calls["n"] += 1
            return _mk_results()

        run_cached_validation("phase1", settings, compute, use_cache=False)
        run_cached_validation("phase1", settings, compute, use_cache=False)
        assert calls["n"] == 2
        assert not (tmp_path / "input" / "validation_cache").exists()

    def test_no_input_folder_disables_caching(self, tmp_path):
        settings = {"data_location": str(tmp_path)}
        calls = {"n": 0}

        def compute():
            calls["n"] += 1
            return _mk_results()

        _, c1 = run_cached_validation("phase1", settings, compute)
        _, c2 = run_cached_validation("phase1", settings, compute)
        assert c1 is False and c2 is False
        assert calls["n"] == 2

    def test_settings_change_invalidates(self, tmp_path):
        settings = _base_settings(tmp_path)
        calls = {"n": 0}

        def compute():
            calls["n"] += 1
            return _mk_results()

        run_cached_validation("phase1", settings, compute)
        settings2 = dict(settings, model_year=2035)
        run_cached_validation("phase1", settings2, compute)
        assert calls["n"] == 2

    def test_data_edit_invalidates_phase2_only(self, tmp_path):
        settings = _base_settings(tmp_path, generation_table="generation.csv")
        target = Path(settings["data_location"]) / "generation.csv"
        target.write_text("a\n1\n")
        calls1 = {"n": 0}
        calls2 = {"n": 0}

        def compute1():
            calls1["n"] += 1
            return []

        def compute2():
            calls2["n"] += 1
            return []

        run_cached_validation("phase1", settings, compute1)
        run_cached_validation("phase2", settings, compute2)
        assert calls1["n"] == 1 and calls2["n"] == 1

        target.write_text("a\n2\n")
        run_cached_validation("phase1", settings, compute1)  # still a hit
        _, cached = run_cached_validation("phase2", settings, compute2)  # miss
        assert calls1["n"] == 1
        assert calls2["n"] == 2 and cached is False

    def test_phase_keys_are_independent(self, tmp_path):
        settings = _base_settings(tmp_path)
        run_cached_validation("phase1", settings, lambda: _mk_results())
        # Phase 2 has its own key; a phase-1 hit must not satisfy phase 2.
        calls = {"n": 0}

        def compute():
            calls["n"] += 1
            return []

        run_cached_validation("phase2", settings, compute)
        assert calls["n"] == 1

    def test_corrupt_cache_file_is_a_miss(self, tmp_path):
        settings = _base_settings(tmp_path)
        run_cached_validation("phase1", settings, lambda: _mk_results())
        folder = cache_folder(settings)
        for f in folder.glob("phase1_*.json"):
            f.write_text("{ not json")
        calls = {"n": 0}

        def compute():
            calls["n"] += 1
            return _mk_results()

        results, cached = run_cached_validation("phase1", settings, compute)
        assert cached is False and calls["n"] == 1
        assert len(results) == 2

    def test_empty_result_list_is_cacheable(self, tmp_path):
        settings = _base_settings(tmp_path)
        calls = {"n": 0}

        def compute():
            calls["n"] += 1
            return []

        run_cached_validation("phase1", settings, compute)
        results, cached = run_cached_validation("phase1", settings, compute)
        assert cached is True and results == [] and calls["n"] == 1

    def test_invalid_phase_raises(self, tmp_path):
        with pytest.raises(ValueError):
            validation_input_fingerprints("phase3", _base_settings(tmp_path))


# ─────────────────────────────────────────────────────────────────────────────
# save/load direct API + replay behavior
# ─────────────────────────────────────────────────────────────────────────────


class TestSaveLoadAndReplay:
    def test_save_load_roundtrip(self, tmp_path):
        settings = _base_settings(tmp_path)
        save_cached_results("phase1", settings, _mk_results())
        loaded = load_cached_results("phase1", settings)
        assert loaded is not None
        assert loaded[1].level == ValidationLevel.ERROR
        assert loaded[1].detail is None

    def test_load_returns_none_for_different_inputs(self, tmp_path):
        settings = _base_settings(tmp_path)
        save_cached_results("phase1", settings, _mk_results())
        assert load_cached_results("phase1", dict(settings, model_year=2040)) is None

    def test_cached_error_replay_still_raises(self, tmp_path):
        """Replayed ERRORs must keep failing the run exactly like fresh ones."""
        settings = _base_settings(tmp_path)
        run_cached_validation("phase1", settings, _mk_results)
        results, cached = run_cached_validation("phase1", settings, lambda: [])
        assert cached is True
        with pytest.raises(ValueError, match="found 1 error"):
            report_validation_results(results)


# ─────────────────────────────────────────────────────────────────────────────
# CLI flag parsing
# ─────────────────────────────────────────────────────────────────────────────


class TestCliFlags:
    def test_defaults_to_none_meaning_follow_settings(self):
        args = _parse_validate_args(["-sf", "settings"])
        assert args.use_validation_cache is None

    @pytest.mark.parametrize("flag", ["--no-validation-cache", "--force-validation"])
    def test_flags_disable_cache(self, flag):
        args = _parse_validate_args(["-sf", "settings", flag])
        assert args.use_validation_cache is False
