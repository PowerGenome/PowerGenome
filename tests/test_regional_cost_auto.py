"""
Test auto-generation of regional cost multiplier mappings
"""

import pandas as pd
import pytest

from powergenome.database import get_data, initialize_data_manager
from powergenome.nrelatb import (
    auto_create_region_map,
    auto_create_technology_map,
    validate_cost_coverage,
)
from powergenome.settings import Settings


@pytest.fixture()
def regional_cost_settings():
    settings = Settings(config_path="tests/test_system/settings")
    settings["RESOURCE_GROUPS"] = "tests/test_system/test_data/resource_groups"
    settings["data_location"] = "tests/test_system/test_data"
    initialize_data_manager(settings, settings["data_location"])
    return settings


def test_auto_create_region_map_direct(regional_cost_settings):
    """Test auto-creation of region map when regions are directly in cost data"""
    regional_cost_df = get_data("regional_cost_factor")

    # Test with regions that exist directly in cost data
    model_regions = ["NWPP", "TRE", "BASN"]
    region_map = auto_create_region_map(
        model_regions,
        regional_cost_df,
        region_aggregations=None,
    )

    # Should map directly to themselves
    assert region_map["NWPP"] == "NWPP"
    assert region_map["TRE"] == "TRE"
    assert region_map["BASN"] == "BASN"


def test_auto_create_region_map_aggregated(regional_cost_settings):
    """Test auto-creation of region map with aggregated regions"""
    regional_cost_df = get_data("regional_cost_factor")

    # Test with aggregated region (p1_2 aggregates p1 and p2)
    model_regions = ["p1_2", "p3"]
    region_aggregations = {"p1_2": ["p1", "p2"]}

    region_map = auto_create_region_map(
        model_regions,
        regional_cost_df,
        region_aggregations=region_aggregations,
    )

    # p1_2 should map to NWPP (which contains both p1 and p2 in test settings)
    # or map to one of its base regions if they exist
    assert "p1_2" in region_map
    # p3 should map directly
    assert "p3" in region_map


def test_auto_create_technology_map(regional_cost_settings):
    """Test auto-creation of technology mapping"""
    regional_cost_df = get_data("regional_cost_factor")

    # Create a test dataframe mimicking new_gen_df before concatenation
    new_gen_df = pd.DataFrame(
        {
            "technology": ["NaturalGas", "LandbasedWind", "Nuclear"],
            "tech_detail": [
                "1-on-1 Combined Cycle (H-Frame)",
                "Class3",
                "Nuclear - Large",
            ],
            "cost_case": ["Moderate", "Moderate", "Moderate"],
        }
    )

    tech_map = auto_create_technology_map(
        new_gen_df,
        regional_cost_df,
        modified_resources=None,
    )

    # Should create mappings for technologies
    assert "NaturalGas_1-on-1 Combined Cycle (H-Frame)_Moderate" in tech_map
    assert "LandbasedWind_Class3_Moderate" in tech_map
    assert "Nuclear_Nuclear - Large_Moderate" in tech_map

    # Check that mappings point to valid cost technologies
    available_cost_techs = set(regional_cost_df["technology"].unique())
    for tech_name, cost_tech in tech_map.items():
        assert cost_tech in available_cost_techs, f"{cost_tech} not in cost data"


def test_validate_cost_coverage_success(regional_cost_settings):
    """Test validation when all technologies are covered"""
    regional_cost_df = get_data("regional_cost_factor")

    new_resources = [
        ["NaturalGas", "1-on-1 Combined Cycle (H-Frame)", "Moderate", 500],
        ["LandbasedWind", "Class3", "Moderate", 1],
    ]

    # Create tech map with actual cost table technology names
    tech_map = {
        "NaturalGas_1-on-1 Combined Cycle (H-Frame)_Moderate": "NaturalGas_1-on-1 Combined Cycle (H-Frame)",
        "LandbasedWind_Class3_Moderate": "LandbasedWind",
    }

    # Should not raise any exceptions
    validate_cost_coverage(
        new_resources,
        modified_resources=None,
        tech_map=tech_map,
        regional_cost_df=regional_cost_df,
    )


def test_validate_cost_coverage_with_warnings(regional_cost_settings, caplog):
    """Test validation warns about missing technologies"""
    regional_cost_df = get_data("regional_cost_factor")

    new_resources = [
        ["UnknownTech", "UnknownDetail", "Moderate", 500],
    ]

    # Empty tech map (technology not matched)
    tech_map = {}

    # Should log warnings
    validate_cost_coverage(
        new_resources,
        modified_resources=None,
        tech_map=tech_map,
        regional_cost_df=regional_cost_df,
    )

    # Check that a warning was logged
    assert "not covered by regional cost corrections" in caplog.text


def test_auto_mapping_with_substring_match(regional_cost_settings):
    """Test that substring matching works for technology names"""
    regional_cost_df = get_data("regional_cost_factor")

    # Create test data with partial name matches
    new_gen_df = pd.DataFrame(
        {
            "technology": ["NaturalGas", "Biomass", "OffShoreWind"],
            "tech_detail": ["CCAvgCF", "Dedicated", "Class3"],
            "cost_case": ["Moderate", "Moderate", "Moderate"],
        }
    )

    tech_map = auto_create_technology_map(
        new_gen_df,
        regional_cost_df,
        modified_resources=None,
    )

    # Should find matches based on substring
    # "NaturalGas" should match to something like "CC - multi shaft" or similar
    assert len(tech_map) > 0

    # At least some technologies should be matched
    for full_tech, cost_tech in tech_map.items():
        assert cost_tech in regional_cost_df["technology"].values


def test_modified_resources_validation(regional_cost_settings, caplog):
    """Test that modified resources trigger informational message"""
    import logging

    caplog.set_level(logging.INFO)

    regional_cost_df = get_data("regional_cost_factor")

    modified_resources = {
        "CustomTech": {
            "new_technology": "CustomNG",
            "new_tech_detail": "Modified",
            "new_cost_case": "Advanced",
        }
    }

    tech_map = {}  # Empty map, modified resource not matched

    validate_cost_coverage(
        new_resources=[],
        modified_resources=modified_resources,
        tech_map=tech_map,
        regional_cost_df=regional_cost_df,
    )

    # Should log info message about modified resource
    assert (
        "Modified resource 'CustomTech'" in caplog.text
        or "CustomNG_Modified_Advanced" in caplog.text
    )
    assert "cost_multiplier_technology_map" in caplog.text


def test_auto_create_technology_map_with_modified_resources(
    regional_cost_settings, caplog
):
    """Test that modified resources are mapped to same multiplier as original"""
    import logging

    caplog.set_level(logging.INFO)

    regional_cost_df = get_data("regional_cost_factor")

    # Create test data with a base technology
    new_gen_df = pd.DataFrame(
        {
            "technology": ["NaturalGas"],
            "tech_detail": ["Combustion Turbine (F-Frame)"],
            "cost_case": ["Moderate"],
        }
    )

    # Define a modified resource based on the original
    modified_resources = {
        "hydrogen_turbine": {
            "technology": "NaturalGas",
            "tech_detail": "Combustion Turbine (F-Frame)",
            "cost_case": "Moderate",
            "new_technology": "hydrogen",
            "new_tech_detail": "combustion turbine",
            "new_cost_case": "Advanced",
        }
    }

    tech_map = auto_create_technology_map(
        new_gen_df,
        regional_cost_df,
        modified_resources=modified_resources,
    )

    # Both the original and modified should map to the same cost table technology
    assert "NaturalGas_Combustion Turbine (F-Frame)_Moderate" in tech_map
    assert "hydrogen_combustion turbine_Advanced" in tech_map

    # They should map to the same cost multiplier
    assert (
        tech_map["NaturalGas_Combustion Turbine (F-Frame)_Moderate"]
        == tech_map["hydrogen_combustion turbine_Advanced"]
    )

    # Should be the cost table technology
    assert (
        tech_map["hydrogen_combustion turbine_Advanced"]
        == "NaturalGas_Combustion Turbine (F-Frame)"
    )

    # Should log that the modified resource was mapped
    assert "hydrogen_combustion turbine_Advanced" in caplog.text
    assert "NaturalGas_Combustion Turbine (F-Frame)" in caplog.text
