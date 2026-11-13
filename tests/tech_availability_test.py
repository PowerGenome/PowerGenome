"""Tests for powergenome.tech_availability — region-scoped new-build exclusions.

The helper module holds the matching rules shared by resource building and
settings validation. Tests here pin the pattern semantics (case-insensitive
fnmatch against ``<technology>_<tech_detail>_<cost_case>``), the additive
``all``/``all_regions`` keys, and the lenient normalization of malformed
settings.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from powergenome.tech_availability import (
    ALL_REGION_KEYS,
    exclusion_patterns_for_region,
    is_all_regions_key,
    is_technology_excluded,
    matches_any_pattern,
    new_resource_names,
    normalize_exclusions,
    technology_full_name,
    unmatched_patterns,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# ─────────────────────────────────────────────────────────────────────────────
# technology_full_name
# ─────────────────────────────────────────────────────────────────────────────


def test_technology_full_name_joins_parts():
    assert technology_full_name("UtilityPV", "Class1", "Moderate") == (
        "UtilityPV_Class1_Moderate"
    )


def test_technology_full_name_keeps_empty_parts():
    # build_new_resources joins every column, so a missing detail leaves a
    # double underscore rather than collapsing the separators.
    assert technology_full_name("UtilityPV", None, "Moderate") == "UtilityPV__Moderate"
    assert technology_full_name("UtilityPV", "", "") == "UtilityPV__"
    assert technology_full_name(float("nan"), "CT", "Moderate") == "_CT_Moderate"


def test_technology_full_name_matches_build_new_resources_expression():
    df = pd.DataFrame(
        {
            "technology": ["NaturalGas", "UtilityPV"],
            "tech_detail": ["CT", ""],
            "cost_case": ["Moderate", "Moderate"],
        }
    )
    expected = df.astype(str).agg("_".join, axis=1).tolist()
    actual = [
        technology_full_name(row.technology, row.tech_detail, row.cost_case)
        for row in df.itertuples()
    ]
    assert actual == expected


# ─────────────────────────────────────────────────────────────────────────────
# matches_any_pattern
# ─────────────────────────────────────────────────────────────────────────────


def test_matches_any_pattern_case_insensitive_wildcard():
    assert matches_any_pattern("NaturalGas_CT_Moderate", ["naturalgas_*"])
    assert matches_any_pattern("offshorewind_class1_moderate", ["OffshoreWind_*"])


def test_matches_any_pattern_without_wildcard_is_exact_only():
    name = "Coal_newAvgCF_Moderate"
    assert not matches_any_pattern(name, ["Coal_new"])
    assert matches_any_pattern(name, ["Coal_new*"])


def test_matches_any_pattern_empty_inputs():
    assert not matches_any_pattern("Coal_newAvgCF_Moderate", [])
    assert not matches_any_pattern("Coal_newAvgCF_Moderate", None)
    assert not matches_any_pattern("", ["*"])
    assert not matches_any_pattern(None, ["*"])


# ─────────────────────────────────────────────────────────────────────────────
# Region keys
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "key",
    ["all", "ALL", "All", "all_regions", "ALL_REGIONS", " all ", "All_Regions "],
)
def test_is_all_regions_key_accepts_variants(key):
    assert is_all_regions_key(key)


@pytest.mark.parametrize("key", ["R1", "all_region", "allregions", None, 5])
def test_is_all_regions_key_rejects_other_values(key):
    assert not is_all_regions_key(key)


def test_all_region_keys_apply_to_every_region():
    settings = {"new_gen_not_available": {"ALL_REGIONS": ["Coal_*"]}}
    assert is_technology_excluded("Coal_newAvgCF_Moderate", settings, "R1")
    assert is_technology_excluded("Coal_newAvgCF_Moderate", settings, "R2")
    assert exclusion_patterns_for_region(settings, "R2") == ["Coal_*"]
    assert set(ALL_REGION_KEYS) == {"all", "all_regions"}


def test_region_patterns_apply_only_to_their_region():
    settings = {"new_gen_not_available": {"R1": ["OffshoreWind_*"]}}
    assert is_technology_excluded("OffshoreWind_Class1_Moderate", settings, "R1")
    assert not is_technology_excluded("OffshoreWind_Class1_Moderate", settings, "R2")


def test_region_patterns_come_before_all_region_patterns():
    settings = {
        "new_gen_not_available": {
            "all": ["Coal_*"],
            "R1": ["OffshoreWind_*"],
        }
    }
    assert exclusion_patterns_for_region(settings, "R1") == [
        "OffshoreWind_*",
        "Coal_*",
    ]
    assert exclusion_patterns_for_region(settings, "R2") == ["Coal_*"]


# ─────────────────────────────────────────────────────────────────────────────
# normalize_exclusions
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "settings",
    [
        pytest.param({}, id="missing"),
        pytest.param(None, id="none-settings"),
        pytest.param({"new_gen_not_available": None}, id="none-value"),
        pytest.param({"new_gen_not_available": {}}, id="empty-dict"),
        pytest.param({"new_gen_not_available": []}, id="empty-list"),
    ],
)
def test_normalize_exclusions_empty_inputs(settings):
    assert normalize_exclusions(settings) == {}


def test_normalize_exclusions_accepts_bare_string():
    settings = {"new_gen_not_available": {"R1": "Coal_*"}}
    assert normalize_exclusions(settings) == {"R1": ["Coal_*"]}


def test_normalize_exclusions_omits_none_values():
    settings = {"new_gen_not_available": {"R1": None, "R2": ["Coal_*"]}}
    assert normalize_exclusions(settings) == {"R2": ["Coal_*"]}


def test_normalize_exclusions_strips_and_dedupes():
    settings = {
        "new_gen_not_available": {
            " R1 ": [" Coal_* ", "Coal_*", "  ", "NaturalGas*"],
        }
    }
    assert normalize_exclusions(settings) == {
        "R1": ["Coal_*", "NaturalGas*"],
    }


def test_normalize_exclusions_skips_wrong_type_with_warning(caplog):
    settings = {"new_gen_not_available": {"R1": 5, "R2": ["Coal_*"]}}

    with caplog.at_level(logging.WARNING, logger="powergenome.tech_availability"):
        result = normalize_exclusions(settings)

    # The malformed entry is dropped rather than raising, so one bad region does
    # not discard the valid exclusions elsewhere.
    assert result == {"R2": ["Coal_*"]}
    assert any("'R1'" in record.getMessage() for record in caplog.records)


# ─────────────────────────────────────────────────────────────────────────────
# new_resource_names / unmatched_patterns
# ─────────────────────────────────────────────────────────────────────────────


def test_new_resource_names_includes_modified_resources():
    settings = {
        "new_resources": [["NaturalGas", "CT", "Moderate", 100]],
        "modified_new_resources": {
            "gas_advanced": {
                "technology": "NaturalGas",
                "tech_detail": "CT",
                "cost_case": "Moderate",
                "size_mw": 100,
                "new_technology": "NaturalGas",
                "new_tech_detail": "CC",
                "new_cost_case": "Advanced",
            }
        },
    }
    assert new_resource_names(settings) == [
        "NaturalGas_CT_Moderate",
        "NaturalGas_CC_Advanced",
    ]


def test_unmatched_patterns_returns_sorted_pairs():
    settings = {
        "new_resources": [
            ["Coal_newAvgCF", "Moderate", "Moderate", 100],
            ["NaturalGas", "CT", "Moderate", 100],
        ],
        "new_gen_not_available": {
            "R2": ["Solar_*"],
            "R1": ["Wind_*", "Coal_new*"],
        },
    }
    assert unmatched_patterns(settings) == [
        ("R1", "Wind_*"),
        ("R2", "Solar_*"),
    ]


def test_unmatched_patterns_empty_without_candidate_names():
    settings = {"new_gen_not_available": {"R1": ["Coal_*"]}}
    assert unmatched_patterns(settings) == []


# ─────────────────────────────────────────────────────────────────────────────
# Import isolation
# ─────────────────────────────────────────────────────────────────────────────


def test_module_imports_without_data_stack():
    # Must run in a subprocess: pytest has already imported pandas by the time
    # this test executes, so checking sys.modules in-process would be vacuous.
    # Phase 1 validation imports this module without a DataManager, so pulling
    # in pandas/duckdb here would make settings validation drag in the data
    # stack.
    code = (
        "import sys\n"
        "import powergenome.tech_availability\n"
        "bad = [m for m in ('pandas', 'numpy', 'duckdb') if m in sys.modules]\n"
        "sys.exit(','.join(bad) or 0)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
