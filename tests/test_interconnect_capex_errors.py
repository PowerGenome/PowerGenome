import pandas as pd
import pytest

from powergenome.generators import _parse_interconnect_capex


def _make_resource_df():
    return pd.DataFrame(
        {
            "region": ["R1", "R2", "R1", "R2"],
            "technology": ["solar_pv", "wind_onshore", "battery_storage", "hydro"],
        }
    )


# A. Non-numeric, non-dict specification type
def test_spec_type_error():
    df = _make_resource_df()
    with pytest.raises(TypeError):
        _parse_interconnect_capex([1, 2, 3], df)


# B. Unknown keys are rejected (legacy dict keys are no longer supported)
def test_unknown_key_error():
    df = _make_resource_df()
    with pytest.raises(ValueError, match="explicit schema only allows keys"):
        _parse_interconnect_capex({"default": 5}, df)


# C. Fallback value must be numeric
def test_fallback_non_numeric():
    df = _make_resource_df()
    with pytest.raises(TypeError):
        _parse_interconnect_capex({"fallback_capex_mw": "x"}, df)


# D. by_technology must be a dict
def test_by_technology_must_be_dict():
    df = _make_resource_df()
    with pytest.raises(TypeError):
        _parse_interconnect_capex({"by_technology": 1}, df)


# E. by_region must be a dict
def test_by_region_must_be_dict():
    df = _make_resource_df()
    with pytest.raises(TypeError):
        _parse_interconnect_capex({"by_region": 1}, df)


# F. by_region_technology must be a dict
def test_by_region_technology_must_be_dict():
    df = _make_resource_df()
    with pytest.raises(TypeError):
        _parse_interconnect_capex({"by_region_technology": 1}, df)


# G. by_technology values must be numeric
def test_by_technology_value_non_numeric():
    df = _make_resource_df()
    with pytest.raises(TypeError):
        _parse_interconnect_capex({"by_technology": {"solar": "high"}}, df)


# H. by_region values must be numeric
def test_by_region_value_non_numeric():
    df = _make_resource_df()
    with pytest.raises(TypeError):
        _parse_interconnect_capex({"by_region": {"R1": "five"}}, df)


# I. by_region rejects missing regions
def test_by_region_missing_region():
    df = _make_resource_df()
    with pytest.raises(KeyError):
        _parse_interconnect_capex({"by_region": {"BAD": 5}}, df)


# J. by_region_technology entries must be dicts
def test_by_region_technology_inner_must_be_dict():
    df = _make_resource_df()
    with pytest.raises(TypeError):
        _parse_interconnect_capex({"by_region_technology": {"R1": 5}}, df)


# K. by_region_technology rejects missing regions
def test_by_region_technology_missing_region():
    df = _make_resource_df()
    with pytest.raises(KeyError):
        _parse_interconnect_capex(
            {"by_region_technology": {"BAD": {"solar": 5}}},
            df,
        )


# L. by_region_technology values must be numeric
def test_by_region_technology_value_non_numeric():
    df = _make_resource_df()
    with pytest.raises(TypeError):
        _parse_interconnect_capex(
            {"by_region_technology": {"R1": {"solar": "five"}}},
            df,
        )
