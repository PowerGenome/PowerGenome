"""
Test functions related to load profiles
"""

import numpy as np
import pandas as pd
import pytest

import powergenome.load_profiles as lp_mod
from powergenome.load_profiles import add_supplemental_demand
from powergenome.load_profiles import grow_historical_load
from powergenome.load_profiles import make_load_curves as _make_load_curves
from powergenome.load_profiles import subtract_distributed_generation


def test_grow_historical_load():
    base_load = {"region": ["A", "A", "B", "B"], "load_mw": [1, 1, 1, 1]}
    base_load_df = pd.DataFrame(base_load)
    hist_region_map = {
        "A": "MWRCE",
        "B": "TRE",
    }
    future_region_map = {
        "A": "MCW",
        "B": "TRE",
    }
    keep_regions = ["A", "B"]
    start_year = 2012
    load_aeo_year = 2022
    end_year, load_2019 = grow_historical_load(
        base_load_df.copy(),
        start_year,
        load_aeo_year,
        keep_regions,
        hist_region_map,
        future_region_map,
    )

    assert end_year > start_year
    assert not base_load_df.equals(load_2019)

    end_year, load_2021 = grow_historical_load(
        load_2019.copy(),
        end_year,
        load_aeo_year,
        keep_regions,
        hist_region_map,
        future_region_map,
    )

    assert end_year == 2021
    assert not load_2021.equals(load_2019)

    aeo_sector_map = {
        "commercial": "COMM",
        "industrial": "IDAL",
        "residential": "RESD",
        "transportation": "TRN",
    }
    base_load = {
        "region": ["A", "A", "B", "B"],
        "load_mw": [1, 1, 1, 1],
        "sector": ["commercial"] * 4,
    }
    base_load_df = pd.DataFrame(base_load)
    hist_region_map = {
        "A": "MWRCE",
        "B": "TRE",
    }
    future_region_map = {
        "A": "MCW",
        "B": "TRE",
    }
    keep_regions = ["A", "B"]
    start_year = 2013
    end_year, load_2019_sector = grow_historical_load(
        base_load_df.copy(),
        start_year,
        load_aeo_year,
        keep_regions,
        hist_region_map,
        future_region_map,
        aeo_sector_map,
    )

    assert not load_2019_sector.equals(load_2019)

    assert end_year > start_year
    assert not base_load_df.equals(load_2019_sector)

    end_year, load_2021_sector = grow_historical_load(
        load_2019_sector.copy(),
        end_year,
        load_aeo_year,
        keep_regions,
        hist_region_map,
        future_region_map,
        aeo_sector_map,
    )

    assert end_year == 2021
    assert not load_2021_sector.equals(load_2019_sector)


def test_subtract_distributed_generation_no_data(monkeypatch):
    # Setup baseline load
    load_curves = pd.DataFrame({"A": [10.0, 10.0, 10.0]})

    # Patch DG hourly gen to be empty
    monkeypatch.setattr(
        lp_mod,
        "get_distributed_gen_hourly_generation",
        lambda **kwargs: pd.DataFrame(),
    )

    out = subtract_distributed_generation(
        load_curves.copy(),
        model_year=2020,
        model_regions=["A"],
        region_aggregations=None,
        utc_offset=0,
        avg_distribution_loss=0.0,
    )

    pd.testing.assert_frame_equal(out, load_curves)


def test_subtract_distributed_generation_with_data(monkeypatch):
    # Baseline load
    load_curves = pd.DataFrame({"A": [100.0, 100.0, 100.0]})

    # DG hourly generation for region A
    dg = pd.DataFrame({"A": [10.0, 0.0, 5.0]})

    monkeypatch.setattr(
        lp_mod,
        "get_distributed_gen_hourly_generation",
        lambda **kwargs: dg,
    )

    out = subtract_distributed_generation(
        load_curves.copy(),
        model_year=2020,
        model_regions=["A"],
        region_aggregations=None,
        utc_offset=0,
        avg_distribution_loss=0.1,  # subtract DG * (1 + loss)
    )

    expected = pd.DataFrame({"A": [89.0, 100.0, 94.5]})
    pd.testing.assert_frame_equal(out, expected)


def test_make_load_curves_timezone_shift(monkeypatch):
    # Provide synthetic demand data via get_data
    def fake_get_data(table_name, columns=None, filters=None, query=None):
        # Query is "PRAGMA table_info('demand')"
        # Return a schema-like DataFrame with name column
        if query is not None:
            return pd.DataFrame({"name": ["year", "region", "time_index", "load_mw"]})

        # Simulate a single region with 4 hours
        return pd.DataFrame(
            {
                "year": [2020, 2020, 2020, 2020],
                "region": ["R1", "R1", "R1", "R1"],
                "time_index": [1, 2, 3, 4],
                "load_mw": [0.0, 1.0, 2.0, 3.0],
            }
        )

    monkeypatch.setattr(lp_mod, "get_data", fake_get_data)

    settings = {
        "model_regions": ["R1"],
        "model_year": 2020,
        "region_aggregations": {},
        "utc_offset": 1,  # shift forward by one hour (np.roll +1)
    }

    out = _make_load_curves(settings)
    # Expect np.roll([0,1,2,3], 1) == [3,0,1,2]
    assert list(out["R1"]) == [3.0, 0.0, 1.0, 2.0]


def test_make_load_curves_with_weather_year(monkeypatch):
    """Test make_load_curves with weather_year filter."""

    def fake_get_data(table_name, columns=None, filters=None, query=None):
        # Query is "PRAGMA table_info('demand')"
        if query is not None:
            return pd.DataFrame(
                {"name": ["year", "region", "time_index", "load_mw", "weather_year"]}
            )

        # Simulate filtered data - only return 2012 data based on filters
        # The actual get_data would filter based on filters parameter
        return pd.DataFrame(
            {
                "year": [2020] * 4,
                "region": ["R1"] * 4,
                "time_index": [1, 2, 3, 4],
                "load_mw": [10.0, 20.0, 30.0, 40.0],
                "weather_year": [2012] * 4,
            }
        )

    monkeypatch.setattr(lp_mod, "get_data", fake_get_data)

    settings = {
        "model_regions": ["R1"],
        "model_year": 2020,
        "region_aggregations": {},
        "utc_offset": 0,
        "weather_year": 2012,  # Filter to only 2012
    }

    out = _make_load_curves(settings)
    # Should have 4 rows renumbered to 1-4
    assert len(out) == 4
    assert list(out["R1"]) == [10.0, 20.0, 30.0, 40.0]


def test_make_load_curves_with_multiple_weather_years(monkeypatch):
    """Test make_load_curves with multiple weather years concatenated."""

    def fake_get_data(table_name, columns=None, filters=None, query=None):
        # Query is "PRAGMA table_info('demand')"
        if query is not None:
            return pd.DataFrame(
                {"name": ["year", "region", "time_index", "load_mw", "weather_year"]}
            )

        # Return data for both requested years (simulating filtering)
        return pd.DataFrame(
            {
                "year": [2020] * 8,
                "region": ["R1"] * 8,
                "time_index": [1, 2, 3, 4, 1, 2, 3, 4],
                "load_mw": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
                "weather_year": [2012] * 4 + [2013] * 4,
            }
        )

    monkeypatch.setattr(lp_mod, "get_data", fake_get_data)

    settings = {
        "model_regions": ["R1"],
        "model_year": 2020,
        "region_aggregations": {},
        "utc_offset": 0,
        "weather_year": [2012, 2013],  # Request both years
    }

    out = _make_load_curves(settings)
    # Should concatenate both years with renumbered time_index (1-8)
    assert len(out) == 8
    # Values should be renumbered
    assert list(out["R1"]) == [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]


def test_make_load_curves_missing_weather_year(monkeypatch):
    """Test make_load_curves raises error when requested weather_year is missing."""

    def fake_get_data(table_name, columns=None, filters=None, query=None):
        if query is not None:
            return pd.DataFrame(
                {"name": ["year", "region", "time_index", "load_mw", "weather_year"]}
            )

        # Only return 2012 data (simulating that 2015 is not available)
        return pd.DataFrame(
            {
                "year": [2020] * 4,
                "region": ["R1"] * 4,
                "time_index": [1, 2, 3, 4],
                "load_mw": [10.0, 20.0, 30.0, 40.0],
                "weather_year": [2012] * 4,
            }
        )

    monkeypatch.setattr(lp_mod, "get_data", fake_get_data)

    settings = {
        "model_regions": ["R1"],
        "model_year": 2020,
        "region_aggregations": {},
        "utc_offset": 0,
        "weather_year": [2012, 2015],  # 2015 is not available
    }

    # Should raise ValueError about missing weather years
    with pytest.raises(ValueError, match="weather_years were requested"):
        _make_load_curves(settings)


def test_make_load_curves_with_single_element_list_weather_year(monkeypatch):
    """Test make_load_curves when weather_year is a single-element list."""

    def fake_get_data(table_name, columns=None, filters=None, query=None):
        if query is not None:
            return pd.DataFrame(
                {"name": ["year", "region", "time_index", "load_mw", "weather_year"]}
            )
        return pd.DataFrame(
            {
                "year": [2020] * 4,
                "region": ["R1"] * 4,
                "time_index": [1, 2, 3, 4],
                "load_mw": [10.0, 20.0, 30.0, 40.0],
                "weather_year": [2012] * 4,
            }
        )

    monkeypatch.setattr(lp_mod, "get_data", fake_get_data)

    settings = {
        "model_regions": ["R1"],
        "model_year": 2020,
        "region_aggregations": {},
        "utc_offset": 0,
        "weather_year": [2012],  # Single-element list
    }

    out = _make_load_curves(settings)
    assert len(out) == 4
    assert list(out["R1"]) == [10.0, 20.0, 30.0, 40.0]


def test_make_load_curves_all_weather_years(monkeypatch):
    """Test make_load_curves when weather_year key is absent (load all years)."""

    def fake_get_data(table_name, columns=None, filters=None, query=None):
        if query is not None:
            # Distinct weather_year query path
            if "SELECT DISTINCT weather_year" in query:
                return pd.DataFrame({"weather_year": [2012, 2013]})
            # Schema introspection path
            if "PRAGMA" in query or "table_info" in query:
                return pd.DataFrame(
                    {
                        "name": [
                            "year",
                            "region",
                            "time_index",
                            "load_mw",
                            "weather_year",
                        ]
                    }
                )
        # Return data containing both weather years
        return pd.DataFrame(
            {
                "year": [2020] * 8,
                "region": ["R1"] * 8,
                "time_index": [1, 2, 3, 4, 1, 2, 3, 4],
                "load_mw": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
                "weather_year": [2012] * 4 + [2013] * 4,
            }
        )

    monkeypatch.setattr(lp_mod, "get_data", fake_get_data)

    settings = {
        "model_regions": ["R1"],
        "model_year": 2020,
        "region_aggregations": {},
        "utc_offset": 0,
        # No weather_year key
    }

    out = _make_load_curves(settings)
    # Expect concatenated all years (renumbered 1-8)
    assert len(out) == 8
    assert list(out["R1"]) == [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]


# ---------------------------------------------------------------------------
# Tests for add_supplemental_demand
# ---------------------------------------------------------------------------


def _base_load_curves(n_hours=4, regions=("R1", "R2"), base_load=100.0):
    """Helper: build a simple wide load_curves DataFrame."""
    idx = pd.RangeIndex(1, n_hours + 1, name="time_index")
    return pd.DataFrame(
        {r: [base_load] * n_hours for r in regions},
        index=idx,
        dtype=float,
    )


def test_add_supplemental_demand_no_table(monkeypatch):
    """When supplemental_demand table is absent, load_curves is returned unchanged."""
    monkeypatch.setattr(lp_mod, "list_tables", lambda: [])

    lc = _base_load_curves()
    out = add_supplemental_demand(lc, model_year=2030, model_regions=["R1", "R2"])
    pd.testing.assert_frame_equal(out, lc)


def test_add_supplemental_demand_all_hours(monkeypatch):
    """Rows with time_index='all_hours' are expanded to every hour."""

    supp = pd.DataFrame(
        {
            "region": ["R1"],
            "time_index": ["all_hours"],
            "load_mw": [50.0],
        }
    )

    monkeypatch.setattr(lp_mod, "list_tables", lambda: ["supplemental_demand"])
    monkeypatch.setattr(
        lp_mod,
        "get_data",
        lambda table_name, filters=None, columns=None, query=None: (
            pd.DataFrame({"name": ["region", "time_index", "load_mw"]})
            if query is not None
            else supp
        ),
    )

    lc = _base_load_curves(n_hours=4)
    out = add_supplemental_demand(lc, model_year=2030, model_regions=["R1", "R2"])

    assert list(out["R1"]) == [150.0, 150.0, 150.0, 150.0]
    assert list(out["R2"]) == [100.0, 100.0, 100.0, 100.0]  # unchanged


def test_add_supplemental_demand_specific_time_index(monkeypatch):
    """Rows with a specific integer time_index modify only that hour."""

    supp = pd.DataFrame(
        {
            "region": ["R1"],
            "time_index": [2],
            "load_mw": [30.0],
        }
    )

    monkeypatch.setattr(lp_mod, "list_tables", lambda: ["supplemental_demand"])
    monkeypatch.setattr(
        lp_mod,
        "get_data",
        lambda table_name, filters=None, columns=None, query=None: (
            pd.DataFrame({"name": ["region", "time_index", "load_mw"]})
            if query is not None
            else supp
        ),
    )

    lc = _base_load_curves(n_hours=4)
    out = add_supplemental_demand(lc, model_year=2030, model_regions=["R1", "R2"])

    # Without weather_year column, no tiling; only hour 2 gets the supplement.
    assert list(out["R1"]) == [100.0, 130.0, 100.0, 100.0]
    assert list(out["R2"]) == [100.0, 100.0, 100.0, 100.0]


def test_add_supplemental_demand_year_filter(monkeypatch):
    """When a 'year' column exists, rows are filtered to model_year."""

    # Two rows: one for 2030, one for 2035
    supp_all = pd.DataFrame(
        {
            "year": [2030, 2035],
            "region": ["R1", "R1"],
            "time_index": ["all_hours", "all_hours"],
            "load_mw": [10.0, 999.0],
        }
    )
    # Simulate DataManager filtering: only return the row matching model_year=2030
    supp_filtered = supp_all[supp_all["year"] == 2030].copy()

    monkeypatch.setattr(lp_mod, "list_tables", lambda: ["supplemental_demand"])
    monkeypatch.setattr(
        lp_mod,
        "get_data",
        lambda table_name, filters=None, columns=None, query=None: (
            pd.DataFrame({"name": ["year", "region", "time_index", "load_mw"]})
            if query is not None
            else supp_filtered
        ),
    )

    lc = _base_load_curves(n_hours=4)
    out = add_supplemental_demand(lc, model_year=2030, model_regions=["R1", "R2"])

    assert list(out["R1"]) == [110.0, 110.0, 110.0, 110.0]


def test_add_supplemental_demand_tiled_across_weather_years(monkeypatch):
    """With weather_year col + NULL wy, specific time indices tile across weather-year blocks."""

    # load_curves has 8 hours = 2 weather years × 4 hours each
    supp = pd.DataFrame(
        {
            "region": ["R1"],
            "time_index": [1],  # should apply to hour 1 of each 4-hour block
            "load_mw": [20.0],
            "weather_year": [None],  # NULL → tile across all blocks
        }
    )

    monkeypatch.setattr(lp_mod, "list_tables", lambda: ["supplemental_demand"])
    monkeypatch.setattr(
        lp_mod,
        "get_data",
        lambda table_name, filters=None, columns=None, query=None: (
            pd.DataFrame({"name": ["region", "time_index", "load_mw", "weather_year"]})
            if query is not None
            else supp
        ),
    )

    # Use hours_per_year=4 so that 8 hours = 2 blocks of 4.
    lc = _base_load_curves(n_hours=8)
    out = add_supplemental_demand(
        lc, model_year=2030, model_regions=["R1"], hours_per_year=4
    )

    # Hours 1 and 5 should get +20
    assert out.loc[1, "R1"] == 120.0
    assert out.loc[5, "R1"] == 120.0
    # All other hours unchanged
    for idx in [2, 3, 4, 6, 7, 8]:
        assert out.loc[idx, "R1"] == 100.0


def test_add_supplemental_demand_unknown_region(monkeypatch):
    """Supplemental demand for a region not in load_curves is silently skipped."""

    supp = pd.DataFrame(
        {
            "region": ["UNKNOWN"],
            "time_index": ["all_hours"],
            "load_mw": [500.0],
        }
    )

    monkeypatch.setattr(lp_mod, "list_tables", lambda: ["supplemental_demand"])
    monkeypatch.setattr(
        lp_mod,
        "get_data",
        lambda table_name, filters=None, columns=None, query=None: (
            pd.DataFrame({"name": ["region", "time_index", "load_mw"]})
            if query is not None
            else supp
        ),
    )

    lc = _base_load_curves(n_hours=4)
    out = add_supplemental_demand(lc, model_year=2030, model_regions=["R1", "R2"])

    # Both regions should be unchanged
    pd.testing.assert_frame_equal(out, lc)


def test_add_supplemental_demand_empty_table(monkeypatch):
    """An empty supplemental demand table leaves load_curves unchanged."""

    monkeypatch.setattr(lp_mod, "list_tables", lambda: ["supplemental_demand"])
    monkeypatch.setattr(
        lp_mod,
        "get_data",
        lambda table_name, filters=None, columns=None, query=None: (
            pd.DataFrame({"name": ["region", "time_index", "load_mw"]})
            if query is not None
            else pd.DataFrame(columns=["region", "time_index", "load_mw"])
        ),
    )

    lc = _base_load_curves(n_hours=4)
    out = add_supplemental_demand(lc, model_year=2030, model_regions=["R1", "R2"])

    pd.testing.assert_frame_equal(out, lc)


def test_add_supplemental_demand_with_weather_year_null(monkeypatch):
    """Rows with NULL weather_year tile across all blocks when weather_year col exists."""

    # load_curves: 8 hours (2 blocks of 4)
    supp = pd.DataFrame(
        {
            "region": ["R1"],
            "time_index": [2],
            "load_mw": [15.0],
            "weather_year": [None],  # NULL → apply to all blocks
        }
    )

    monkeypatch.setattr(lp_mod, "list_tables", lambda: ["supplemental_demand"])
    monkeypatch.setattr(
        lp_mod,
        "get_data",
        lambda table_name, filters=None, columns=None, query=None: (
            pd.DataFrame(
                {"name": ["region", "time_index", "load_mw", "weather_year"]}
            )
            if query is not None
            else supp
        ),
    )

    lc = _base_load_curves(n_hours=8)
    out = add_supplemental_demand(
        lc, model_year=2030, model_regions=["R1"], hours_per_year=4
    )

    # time_index 2 and 6 (= 2 + 4) should get +15
    assert out.loc[2, "R1"] == 115.0
    assert out.loc[6, "R1"] == 115.0
    for idx in [1, 3, 4, 5, 7, 8]:
        assert out.loc[idx, "R1"] == 100.0
