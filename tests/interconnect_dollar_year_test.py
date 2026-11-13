"""Tests for converting resource group interconnection costs to ``target_usd_year``."""

import logging

import pandas as pd
import pytest

from powergenome.database import initialize_data_manager
from powergenome.financials import (
    adjust_interconnect_dollar_year,
    inflation_price_adjustment,
)
from powergenome.new_build import (
    _resource_cluster_cache_names,
    load_resource_group_data,
)
from powergenome.resource_clusters import ResourceGroup
from powergenome.settings import Settings

TARGET_YEAR = 2020


@pytest.fixture
def settings():
    """Settings with a DataManager holding the CPI table used for conversion."""
    settings = Settings(config_path="tests/test_system/settings")
    settings["data_location"] = "tests/test_system/test_data"
    initialize_data_manager(settings, settings["data_location"])
    return settings


def _factor(base_year, target_year=TARGET_YEAR):
    return inflation_price_adjustment(1.0, base_year, target_year)


def _cost_df(dollar_years=None, **kwargs):
    data = {
        "capacity_mw": [100.0, 200.0],
        "lcoe": [10.0, 20.0],
        "interconnect_annuity": [100.0, 200.0],
        "interconnect_capex_mw": [1000.0, 2000.0],
    }
    if dollar_years is not None:
        data["dollar_year"] = dollar_years
    data.update(kwargs)
    return pd.DataFrame(data)


class TestAdjustInterconnectDollarYear:
    def test_converts_each_row_from_its_own_dollar_year(self, settings):
        df = _cost_df(dollar_years=[2017, 2019])

        result = adjust_interconnect_dollar_year(df, target_usd_year=TARGET_YEAR)

        assert result.loc[0, "interconnect_annuity"] == pytest.approx(
            100 * _factor(2017)
        )
        assert result.loc[1, "interconnect_annuity"] == pytest.approx(
            200 * _factor(2019)
        )
        assert result.loc[0, "interconnect_capex_mw"] == pytest.approx(
            1000 * _factor(2017)
        )
        assert result.loc[1, "interconnect_capex_mw"] == pytest.approx(
            2000 * _factor(2019)
        )
        # Non-cost columns are untouched and the input frame is not modified.
        assert result["lcoe"].tolist() == [10.0, 20.0]
        assert df["interconnect_annuity"].tolist() == [100.0, 200.0]

    def test_uses_fallback_year_without_a_dollar_year_column(self, settings):
        df = _cost_df()

        result = adjust_interconnect_dollar_year(
            df, target_usd_year=TARGET_YEAR, fallback_dollar_year=2017
        )

        assert result["interconnect_annuity"].tolist() == pytest.approx(
            [100 * _factor(2017), 200 * _factor(2017)]
        )

    def test_row_year_overrides_fallback_and_missing_rows_use_fallback(self, settings):
        df = _cost_df(dollar_years=[2019, None])

        result = adjust_interconnect_dollar_year(
            df, target_usd_year=TARGET_YEAR, fallback_dollar_year=2017
        )

        assert result.loc[0, "interconnect_annuity"] == pytest.approx(
            100 * _factor(2019)
        )
        assert result.loc[1, "interconnect_annuity"] == pytest.approx(
            200 * _factor(2017)
        )

    def test_missing_years_are_left_unchanged_with_a_warning(self, settings, caplog):
        df = _cost_df(dollar_years=[2017, None])

        with caplog.at_level(logging.WARNING, logger="powergenome.financials"):
            result = adjust_interconnect_dollar_year(df, target_usd_year=TARGET_YEAR)

        assert result.loc[0, "interconnect_annuity"] == pytest.approx(
            100 * _factor(2017)
        )
        assert result.loc[1, "interconnect_annuity"] == 200.0
        assert "no dollar year and no fallback" in caplog.text

    def test_no_dollar_year_leaves_costs_unchanged(self, settings, caplog):
        df = _cost_df()

        with caplog.at_level(logging.WARNING, logger="powergenome.financials"):
            result = adjust_interconnect_dollar_year(df, target_usd_year=TARGET_YEAR)

        pd.testing.assert_frame_equal(result, df)
        assert "resource_group_dollar_year" in caplog.text

    def test_target_year_of_none_is_a_noop(self, settings):
        df = _cost_df(dollar_years=[2017, 2017])

        result = adjust_interconnect_dollar_year(df, target_usd_year=None)

        pd.testing.assert_frame_equal(result, df)

    def test_missing_cost_columns_are_ignored(self, settings):
        df = pd.DataFrame({"capacity_mw": [1.0], "dollar_year": [2017]})

        result = adjust_interconnect_dollar_year(df, target_usd_year=TARGET_YEAR)

        pd.testing.assert_frame_equal(result, df)


class TestLoadResourceGroupData:
    def _resource_group(self):
        metadata = pd.DataFrame(
            {
                "CPA_ID": [1, 2],
                "region": ["A", "A"],
                "capacity_mw": [100.0, 200.0],
                "interconnect_annuity": [10.0, 20.0],
                "interconnect_capex_mw": [100.0, 200.0],
                "dollar_year": [2017, 2017],
            }
        )
        return ResourceGroup({"technology": "landbasedwind"}, metadata=metadata)

    def test_converts_costs_without_poisoning_the_metadata_cache(self, settings):
        rg = self._resource_group()
        settings.update({"target_usd_year": TARGET_YEAR})

        data, site_map = load_resource_group_data(rg, cache=True, settings=settings)
        again, _ = load_resource_group_data(rg, cache=True, settings=settings)

        assert site_map is None
        expected = [10 * _factor(2017), 20 * _factor(2017)]
        assert data["interconnect_annuity"].tolist() == pytest.approx(expected)
        assert again["interconnect_annuity"].tolist() == pytest.approx(expected)
        assert rg.metadata.df["interconnect_annuity"].tolist() == [10.0, 20.0]
        assert rg.metadata.df["interconnect_capex_mw"].tolist() == [100.0, 200.0]

    def test_uses_resource_group_dollar_year_setting(self, settings):
        metadata = self._resource_group().metadata.df.drop(columns="dollar_year")
        rg = ResourceGroup({"technology": "landbasedwind"}, metadata=metadata)
        settings.update(
            {"target_usd_year": TARGET_YEAR, "resource_group_dollar_year": 2017}
        )

        data, _ = load_resource_group_data(rg, cache=False, settings=settings)

        assert data["interconnect_annuity"].tolist() == pytest.approx(
            [10 * _factor(2017), 20 * _factor(2017)]
        )


class TestLegacyClusterPath:
    def _resource_group(self):
        metadata = pd.DataFrame(
            {
                "id": [0, 1],
                "region": ["A", "A"],
                "capacity_mw": [50.0, 50.0],
                "interconnect_annuity": [100.0, 200.0],
                "dollar_year": [2017, 2017],
            }
        )
        return ResourceGroup({"technology": "landbasedwind"}, metadata=metadata)

    def test_get_clusters_converts_costs_before_aggregating(self, settings):
        rg = self._resource_group()

        clusters = rg.get_clusters(
            regions=["A"],
            max_clusters=1,
            profiles=False,
            target_usd_year=TARGET_YEAR,
        )

        assert clusters["interconnect_annuity"].iloc[0] == pytest.approx(
            150 * _factor(2017)
        )

    def test_get_clusters_without_target_year_keeps_costs(self, settings):
        rg = self._resource_group()

        clusters = rg.get_clusters(regions=["A"], max_clusters=1, profiles=False)

        assert clusters["interconnect_annuity"].iloc[0] == pytest.approx(150.0)


class TestResourceClusterCacheNames:
    def test_dollar_years_change_the_cache_key(self):
        args = ("R1", "landbasedwind", "_detail", 0, 2012, "abc123")

        name, unique_hash = _resource_cluster_cache_names(
            *args, target_usd_year=2020, resource_group_dollar_year=None
        )
        other_target, other_target_hash = _resource_cluster_cache_names(
            *args, target_usd_year=2030, resource_group_dollar_year=None
        )
        other_fallback, other_fallback_hash = _resource_cluster_cache_names(
            *args, target_usd_year=2020, resource_group_dollar_year=2017
        )

        assert "usd2020" in name
        assert len({unique_hash, other_target_hash, other_fallback_hash}) == 3
        assert other_target != name
        assert other_fallback != name

    def test_same_settings_produce_the_same_cache_key(self):
        args = ("R1", "landbasedwind", "_detail", 0, 2012, "abc123")

        first = _resource_cluster_cache_names(
            *args, target_usd_year=2020, resource_group_dollar_year=None
        )
        second = _resource_cluster_cache_names(
            *args, target_usd_year=2020, resource_group_dollar_year=None
        )

        assert first == second
