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


# B. Default value not numeric
def test_default_non_numeric():
    df = _make_resource_df()
    with pytest.raises(TypeError):
        _parse_interconnect_capex({"default": "x"}, df)


# C. Technology-only pattern with non-numeric value
def test_technology_only_value_non_numeric():
    df = _make_resource_df()
    with pytest.raises(TypeError):
        _parse_interconnect_capex({"solar": "high"}, df)


# D. Mixing region and technology keys at top level (numeric values)
def test_mixed_region_tech_numeric_values():
    df = _make_resource_df()
    # R1 is a region; 'solar' is a technology substring
    with pytest.raises(ValueError):
        _parse_interconnect_capex({"R1": 10, "solar": 20}, df)


# E. Region->tech pattern invalid region value type (list)
def test_region_value_invalid_type():
    df = _make_resource_df()
    with pytest.raises(TypeError):
        _parse_interconnect_capex({"R1": [10], "R2": 5}, df)


# F. Technology->region pattern with top-level tech substring mapping to non-dict value (list)
def test_tech_region_non_dict_mapping():
    df = _make_resource_df()
    # Force nested pattern by including at least one dict value so all_numbers is False
    with pytest.raises(TypeError):
        _parse_interconnect_capex({"solar": {"R1": 5}, "wind": [10]}, df)


# G. Technology->region mapping referencing missing region
def test_tech_region_missing_region():
    df = _make_resource_df()
    with pytest.raises(KeyError):
        _parse_interconnect_capex({"solar": {"BAD": 5}}, df)


# H. Technology->region mapping with non-numeric region value
def test_tech_region_value_non_numeric():
    df = _make_resource_df()
    with pytest.raises(TypeError):
        _parse_interconnect_capex({"solar": {"R1": "five"}}, df)


# I. Mixed region & technology dict patterns triggering final invalid specification ValueError
def test_final_invalid_mixed_nested_patterns():
    df = _make_resource_df()
    # Region key 'R1' has dict (region->tech), technology key 'wind' has dict (tech->region)
    with pytest.raises(ValueError):
        _parse_interconnect_capex({"R1": {"solar": 10}, "wind": {"R1": 5}}, df)
