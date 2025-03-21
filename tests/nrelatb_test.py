from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from powergenome.nrelatb import (
    load_renew_data,
    load_resource_group_data,
    plot_supply_curve,
    regional_capex_multiplier,
)


def test_load_renew_data_no_matching_groups(mocker):
    cluster_builder = mocker.Mock()
    cluster_builder.find_groups.return_value = []
    _scenario = {"technology": "solar", "region": "region1"}

    with pytest.raises(ValueError, match="Parameters do not match any resource groups"):
        load_renew_data(cluster_builder, _scenario)


def test_load_renew_data_multiple_matching_groups(mocker):
    cluster_builder = mocker.Mock()
    resource_group1 = mocker.Mock()
    resource_group2 = mocker.Mock()
    cluster_builder.find_groups.return_value = [resource_group1, resource_group2]
    _scenario = {"technology": "solar", "region": "region1"}

    with pytest.raises(ValueError, match="Parameters match multiple resource groups"):
        load_renew_data(cluster_builder, _scenario)


def test_load_renew_data_single_matching_group(mocker):
    cluster_builder = mocker.Mock()
    resource_group = mocker.Mock()
    cluster_builder.find_groups.return_value = [resource_group]
    _scenario = {"technology": "solar", "region": "region1"}

    mock_renew_data = pd.DataFrame({"data": [1, 2, 3]})
    mock_site_map = {"site1": "map1"}
    mock_load_resource_group_data = mocker.patch(
        "powergenome.nrelatb.load_resource_group_data"
    )
    mock_load_resource_group_data.return_value = (mock_renew_data, mock_site_map)

    resource_groups, renew_data, site_map = load_renew_data(cluster_builder, _scenario)

    assert resource_groups == [resource_group]
    assert renew_data.equals(mock_renew_data)
    assert site_map == mock_site_map
    mock_load_resource_group_data.assert_called_once_with(resource_group, cache=True)


def test_load_renew_data_no_matching_groups(mocker):
    cluster_builder = mocker.Mock()
    cluster_builder.find_groups.return_value = []
    _scenario = {"technology": "solar", "region": "region1"}

    with pytest.raises(ValueError, match="Parameters do not match any resource groups"):
        load_renew_data(cluster_builder, _scenario)


def test_load_renew_data_multiple_matching_groups(mocker):
    cluster_builder = mocker.Mock()
    resource_group1 = mocker.Mock()
    resource_group2 = mocker.Mock()
    cluster_builder.find_groups.return_value = [resource_group1, resource_group2]
    _scenario = {"technology": "solar", "region": "region1"}

    with pytest.raises(ValueError, match="Parameters match multiple resource groups"):
        load_renew_data(cluster_builder, _scenario)


def test_load_renew_data_single_matching_group(mocker):
    cluster_builder = mocker.Mock()
    resource_group = mocker.Mock()
    cluster_builder.find_groups.return_value = [resource_group]
    _scenario = {"technology": "solar", "region": "region1"}

    mock_renew_data = pd.DataFrame({"data": [1, 2, 3]})
    mock_site_map = {"site1": "map1"}
    mock_load_resource_group_data = mocker.patch(
        "powergenome.nrelatb.load_resource_group_data"
    )
    mock_load_resource_group_data.return_value = (mock_renew_data, mock_site_map)

    resource_groups, renew_data, site_map = load_renew_data(cluster_builder, _scenario)

    assert resource_groups == [resource_group]
    assert renew_data.equals(mock_renew_data)
    assert site_map == mock_site_map
    mock_load_resource_group_data.assert_called_once_with(resource_group, cache=True)


def test_plot_supply_curve(mocker, tmp_path: Path):
    data = pd.DataFrame({"cpa_id": [1, 2, 3], "cluster": [1, 2, 1]})
    renew_data = pd.DataFrame(
        {"cpa_id": [1, 2, 3], "cpa_mw": [100, 200, 300], "lcoe": [50, 60, 70]}
    )
    region = "region1"
    technology = "solar"
    new_tech_suffix = "_test"
    folder = tmp_path

    mocker.patch("matplotlib.pyplot.savefig")
    plot_supply_curve(data, renew_data, region, technology, new_tech_suffix, folder)

    plt.savefig.assert_called_once()
    args, kwargs = plt.savefig.call_args
    assert args[0] == folder / "region1_solar_testsite_supply_curve.png"
    assert kwargs["bbox_inches"] == "tight"


def test_plot_supply_curve_key_error(mocker, tmp_path: Path):
    data = pd.DataFrame({"cpa_id": [1, 2, 3], "cluster": [1, 2, 1]})
    renew_data = pd.DataFrame(
        {"cpa_id": [1, 2, 3], "cpa_mw": [100, 200, 300]}  # Missing "lcoe" column
    )
    region = "region1"
    technology = "solar"
    new_tech_suffix = "_test"
    folder = tmp_path

    with pytest.raises(KeyError, match="'lcoe'"):
        plot_supply_curve(data, renew_data, region, technology, new_tech_suffix, folder)


class TestRegionalCapexMultiplier:
    """Test class for regional_capex_multiplier function."""

    @pytest.fixture(scope="class")
    def sample_data(self):
        """Create a sample DataFrame with different technology names, including special regex characters."""
        data = {
            "technology": ["Solar PV", "Wind*", "Battery+", "CCGT (new)."],
            "Inv_Cost_per_MWyr": [1000, 2000, 3000, 4000],
            "Inv_Cost_per_MWhyr": [500, 1000, 1500, 2000],
        }
        return pd.DataFrame(data)

    @pytest.fixture(scope="class")
    def region_map(self):
        """Create a mapping of regions to cost adjustment regions."""
        return {"Region1": "CostRegionA", "Region2": "CostRegionB"}

    @pytest.fixture(scope="class")
    def tech_map(self):
        """Create a mapping of technology names to their EIA equivalents."""
        return {
            "Solar PV": "Solar PV",
            "Wind*": "Wind",
            "Battery+": "Battery Storage",
            "CCGT (new).": "Combined Cycle",
        }

    @pytest.fixture(scope="class")
    def regional_multipliers(self):
        """Create a DataFrame of regional multipliers."""
        data = {
            "CostRegionA": {
                "Solar PV": 1.1,
                "Wind": 1.2,
                "Battery Storage": 0.9,
                "Combined Cycle": 1.05,
            },
            "CostRegionB": {
                "Solar PV": 1.0,
                "Wind": 1.1,
                "Battery Storage": 0.95,
                "Combined Cycle": 1.02,
            },
        }
        return pd.DataFrame(data).T

    def test_regional_capex_multiplier(
        self, sample_data, region_map, tech_map, regional_multipliers
    ):
        """Test that regional multipliers are applied correctly, including for technologies with special regex characters."""
        region = "Region1"
        modified_df = regional_capex_multiplier(
            sample_data, region, region_map, tech_map, regional_multipliers
        )

        expected_inv_cost = [1000 * 1.1, 2000 * 1.2, 3000 * 0.9, 4000 * 1.05]
        expected_inv_cost_mwh = [500 * 1.1, 1000 * 1.2, 1500 * 0.9, 2000 * 1.05]

        assert modified_df["Inv_Cost_per_MWyr"].tolist() == pytest.approx(
            expected_inv_cost, rel=1e-3
        )
        assert modified_df["Inv_Cost_per_MWhyr"].tolist() == pytest.approx(
            expected_inv_cost_mwh, rel=1e-3
        )

    def test_missing_technology_in_multipliers(self, sample_data, region_map, tech_map):
        """Test that missing technologies in the multipliers default to the average multiplier."""
        region = "Region1"
        # Modify multipliers so "CCGT (new)." is missing
        regional_multipliers = pd.DataFrame(
            {
                "CostRegionA": {
                    "Solar PV": 1.1,
                    "Wind": 1.2,
                    "Battery Storage": 0.9,
                    "Combined Cycle": np.nan,
                }
            }
        ).T

        modified_df = regional_capex_multiplier(
            sample_data, region, region_map, tech_map, regional_multipliers
        )

        # The "CCGT (new)." should use the average multiplier (1.067)
        avg_multiplier = sum([1.1, 1.2, 0.9]) / 3
        expected_inv_cost = [1000 * 1.1, 2000 * 1.2, 3000 * 0.9, 4000 * avg_multiplier]

        assert modified_df["Inv_Cost_per_MWyr"].tolist() == pytest.approx(
            expected_inv_cost, rel=1e-3
        )

    def test_empty_dataframe(self, region_map, tech_map, regional_multipliers):
        """Test handling of an empty input DataFrame."""
        empty_df = pd.DataFrame(
            columns=["technology", "Inv_Cost_per_MWyr", "Inv_Cost_per_MWhyr"]
        )
        region = "Region1"

        modified_df = regional_capex_multiplier(
            empty_df, region, region_map, tech_map, regional_multipliers
        )
        assert modified_df.empty

    def test_invalid_region(
        self, sample_data, region_map, tech_map, regional_multipliers
    ):
        """Test behavior when an invalid region is provided."""
        invalid_region = "UnknownRegion"

        with pytest.raises(KeyError):
            regional_capex_multiplier(
                sample_data, invalid_region, region_map, tech_map, regional_multipliers
            )
