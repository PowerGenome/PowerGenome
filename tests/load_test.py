"""
Test functions related to load profiles
"""

import numpy as np
import pandas as pd
import pytest

import powergenome.load_profiles as lp_mod
from powergenome.load_profiles import add_supplemental_demand, grow_historical_load
from powergenome.load_profiles import make_load_curves as _make_load_curves
from powergenome.load_profiles import subtract_distributed_generation


@pytest.fixture(autouse=True)
def _isolate_data_manager(monkeypatch):
    """Isolate ``load_profiles.list_tables`` from any DataManager state leaked by
    other test modules.

    Some test modules (e.g. ``fuel_test``) initialize the global DataManager
    singleton against the ``tests/test_system`` data folder, which registers a
    ``supplemental_demand`` table.  Tests in this module that call
    ``make_load_curves`` directly only patch ``get_data``, so without this the
    real ``list_tables()`` would report ``supplemental_demand`` and the fake
    ``get_data`` would serve demand rows as supplemental demand (double-counting
    every hour).  Default to *no* supplemental table; tests that exercise
    supplemental demand re-patch ``list_tables`` to register the table.
    """
    monkeypatch.setattr(lp_mod, "list_tables", lambda: [])


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
# Tests for supplemental demand (integrated into make_load_curves)
#
# Supplemental demand is now applied INSIDE make_load_curves in long format
# (before hours are renumbered and base regions aggregated), so these tests
# exercise the full pipeline via make_load_curves with a fake get_data that
# serves both the `demand` and `supplemental_demand` tables.
# ---------------------------------------------------------------------------


def _make_supp_get_data(demand, supp, demand_cols, supp_cols):
    """Build a fake get_data serving the demand and supplemental_demand tables.

    The fake mirrors the DataManager's behavior well enough for these tests:
    a ``query`` returns the table schema (PRAGMA) and filters are applied in DNF
    form. The caller is responsible for pre-filtering ``supp`` when modeling
    DataManager-level (dict-format) filters like ``scenario``.
    """

    def fake_get_data(table_name, columns=None, filters=None, query=None):
        if query is not None:
            if table_name == "demand" and "DISTINCT weather_year" in query:
                return pd.DataFrame(
                    {"weather_year": sorted(demand["weather_year"].dropna().unique())}
                )
            cols = demand_cols if table_name == "demand" else supp_cols
            return pd.DataFrame({"name": cols}).astype(object)
        data = demand if table_name == "demand" else supp
        df = data.copy()
        if filters:
            # DataManager filters are DNF: OR-of-AND conjunctions.
            keep = pd.Series(False, index=df.index)
            for conjunction in filters:
                mask = pd.Series(True, index=df.index)
                for col, op, val in conjunction:
                    if op == "=":
                        mask &= df[col] == val
                    elif op == "in":
                        mask &= df[col].isin(val)
                    elif op == "!=":
                        mask &= df[col] != val
                keep |= mask
            df = df.loc[keep]
        return df

    return fake_get_data


def _base_load_curves(n_hours=4, regions=("R1", "R2"), base_load=100.0):
    """Helper: build a simple wide load_curves DataFrame (WIDE-path tests)."""
    idx = pd.RangeIndex(1, n_hours + 1, name="time_index")
    return pd.DataFrame(
        {r: [base_load] * n_hours for r in regions},
        index=idx,
        dtype=float,
    )


def _long_load(wy_hours, base_load, regions=("R1",), model_year=2030):
    """Build a long demand frame: one row per (region, wy, hour).

    ``wy_hours`` maps weather year -> number of hours in that year, so years can
    have different lengths (e.g. leap vs non-leap).
    """
    rows = []
    for wy, n in wy_hours.items():
        for r in regions:
            for h in range(1, n + 1):
                rows.append((model_year, r, wy, h, base_load))
    return pd.DataFrame(
        rows,
        columns=["year", "region", "weather_year", "time_index", "load_mw"],
    )


DEMAND_COLS = ["year", "region", "time_index", "load_mw", "weather_year"]
SUPP_COLS = ["region", "time_index", "load_mw"]


def _run_make_load_curves(
    monkeypatch,
    demand,
    supp=None,
    supp_cols=None,
    model_regions=("R1",),
    region_aggregations=None,
    weather_year=None,
    model_year=2030,
    **extra,
):
    """Run make_load_curves with fake demand + (optional) supplemental tables."""
    monkeypatch.setattr(
        lp_mod,
        "list_tables",
        lambda: ["supplemental_demand"] if supp is not None else [],
    )
    monkeypatch.setattr(
        lp_mod,
        "get_data",
        _make_supp_get_data(
            demand,
            supp if supp is not None else pd.DataFrame(),
            DEMAND_COLS,
            supp_cols if supp_cols is not None else SUPP_COLS,
        ),
    )
    settings = {
        "model_regions": list(model_regions),
        "model_year": model_year,
        "region_aggregations": region_aggregations or {},
        "utc_offset": 0,
    }
    if weather_year is not None:
        settings["weather_year"] = weather_year
    settings.update(extra)
    return _make_load_curves(settings)


def test_make_load_curves_supp_no_table(monkeypatch):
    """No supplemental_demand table -> load_curves unchanged."""
    demand = _long_load({2012: 4}, 100.0)
    out = _run_make_load_curves(monkeypatch, demand, weather_year=[2012])
    assert list(out["R1"]) == [100.0] * 4


def test_make_load_curves_supp_missing_required_columns(monkeypatch):
    """Missing required columns (region/time_index/load_mw) raise a descriptive error."""
    supp = pd.DataFrame({"region": ["R1"], "load_mw": [50.0]})
    demand = _long_load({2012: 4}, 100.0)
    with pytest.raises(ValueError, match="time_index"):
        _run_make_load_curves(
            monkeypatch,
            demand,
            supp=supp,
            supp_cols=["region", "load_mw"],
            weather_year=[2012],
        )


def test_make_load_curves_supp_empty_table(monkeypatch):
    """An empty supplemental_demand table leaves load_curves unchanged."""
    supp = pd.DataFrame(columns=["region", "time_index", "load_mw"])
    demand = _long_load({2012: 4}, 100.0)
    out = _run_make_load_curves(
        monkeypatch,
        demand,
        supp=supp,
        weather_year=[2012],
    )
    assert list(out["R1"]) == [100.0] * 4


def test_make_load_curves_supp_no_wy_col_all_hours(monkeypatch):
    """all_hours expansion when the supplemental table has no weather_year column."""
    supp = pd.DataFrame(
        {"region": ["R1"], "time_index": ["all_hours"], "load_mw": [50.0]}
    )
    demand = _long_load({2012: 4}, 100.0)
    out = _run_make_load_curves(
        monkeypatch,
        demand,
        supp=supp,
        weather_year=[2012],
    )
    assert list(out["R1"]) == [150.0] * 4


def test_make_load_curves_supp_no_wy_col_specific_ti(monkeypatch):
    """A specific time_index without a weather_year column hits only that hour."""
    supp = pd.DataFrame({"region": ["R1"], "time_index": [2], "load_mw": [30.0]})
    demand = _long_load({2012: 4}, 100.0)
    out = _run_make_load_curves(
        monkeypatch,
        demand,
        supp=supp,
        weather_year=[2012],
    )
    assert list(out["R1"]) == [100.0, 130.0, 100.0, 100.0]


def test_make_load_curves_supp_year_filter(monkeypatch):
    """Rows with a year column are filtered to model_year."""
    supp = pd.DataFrame(
        {
            "year": [2030, 2035],
            "region": ["R1", "R1"],
            "time_index": ["all_hours", "all_hours"],
            "load_mw": [10.0, 999.0],
        }
    )
    demand = _long_load({2012: 4}, 100.0)
    # The fake get_data applies the year='=' filter for us.
    out = _run_make_load_curves(
        monkeypatch,
        demand,
        supp=supp,
        supp_cols=["year", "region", "time_index", "load_mw"],
        weather_year=[2012],
        model_year=2030,
    )
    assert list(out["R1"]) == [110.0] * 4


def test_make_load_curves_supp_base_region(monkeypatch):
    """Supplemental demand can name a BASE region (mapped to the aggregated model region)."""
    demand = _long_load({2012: 4}, 10.0, regions=("b1", "b2"))
    # Base region b1, supplemental 5 per hour. M1 = b1 + b2 when aggregated.
    supp = pd.DataFrame(
        {
            "region": ["b1"],
            "time_index": ["all_hours"],
            "load_mw": [5.0],
            "weather_year": [2012],
        }
    )
    out = _run_make_load_curves(
        monkeypatch,
        demand,
        supp=supp,
        supp_cols=["region", "time_index", "load_mw", "weather_year"],
        model_regions=("M1",),
        region_aggregations={"M1": ["b1", "b2"]},
        weather_year=[2012],
    )
    # Each hour: (b1 10 + supp 5) + b2 10 = 25
    assert list(out["M1"]) == [25.0] * 4


def test_make_load_curves_supp_model_region(monkeypatch):
    """Supplemental demand can name a MODEL region (mapped to its first base region)."""
    demand = _long_load({2012: 4}, 10.0, regions=("b1", "b2"))
    # Model region M1 -> added to first base region b1 (adds exactly the
    # supplemental amount to the summed aggregate profile).
    supp = pd.DataFrame(
        {
            "region": ["M1"],
            "time_index": ["all_hours"],
            "load_mw": [7.0],
            "weather_year": [2012],
        }
    )
    out = _run_make_load_curves(
        monkeypatch,
        demand,
        supp=supp,
        supp_cols=["region", "time_index", "load_mw", "weather_year"],
        model_regions=("M1",),
        region_aggregations={"M1": ["b1", "b2"]},
        weather_year=[2012],
    )
    # Each hour: (b1 10 + supp 7) + b2 10 = 27
    assert list(out["M1"]) == [27.0] * 4


def test_make_load_curves_supp_unknown_region(monkeypatch):
    """Supplemental demand for an unmapped region is skipped with a warning."""
    demand = _long_load({2012: 4}, 100.0)
    supp = pd.DataFrame(
        {"region": ["Z9"], "time_index": ["all_hours"], "load_mw": [500.0]}
    )
    out = _run_make_load_curves(
        monkeypatch,
        demand,
        supp=supp,
        weather_year=[2012],
    )
    assert list(out["R1"]) == [100.0] * 4


def test_make_load_curves_supp_unequal_wy_lengths(monkeypatch):
    """Supplemental demand lands in the correct per-weather-year hours with NO
    fixed weather-year-length assumption (2012=4h, 2013=5h)."""
    demand = _long_load({2012: 4, 2013: 5}, 100.0)
    supp = pd.DataFrame(
        {
            "region": ["R1", "R1"],
            "time_index": ["all_hours", "all_hours"],
            "load_mw": [20.0, 300.0],
            "weather_year": [2012, 2013],
        }
    )
    out = _run_make_load_curves(
        monkeypatch,
        demand,
        supp=supp,
        supp_cols=["region", "time_index", "load_mw", "weather_year"],
        weather_year=[2012, 2013],
    )
    # Renumbered hours: 2012 block = 1..4 (100+20), 2013 block = 5..9 (100+300).
    assert list(out["R1"]) == [120.0] * 4 + [400.0] * 5


def test_make_load_curves_supp_blank_wy_skipped(monkeypatch):
    """Rows with a blank weather_year are skipped, not tiled or applied."""
    supp = pd.DataFrame(
        {
            "region": ["R1"],
            "time_index": ["all_hours"],
            "load_mw": [50.0],
            "weather_year": [None],
        }
    )
    demand = _long_load({2012: 4, 2013: 4}, 100.0)
    out = _run_make_load_curves(
        monkeypatch,
        demand,
        supp=supp,
        supp_cols=["region", "time_index", "load_mw", "weather_year"],
        weather_year=[2012, 2013],
    )
    assert list(out["R1"]) == [100.0] * 8


def test_make_load_curves_supp_coverage_raises(monkeypatch):
    """A weather year in load data not covered by the table raises ValueError."""
    supp = pd.DataFrame(
        {
            "region": ["R1"],
            "time_index": ["all_hours"],
            "load_mw": [50.0],
            "weather_year": [2012],
        }
    )
    demand = _long_load({2012: 4, 2013: 4}, 100.0)
    with pytest.raises(ValueError, match="does not cover weather year"):
        _run_make_load_curves(
            monkeypatch,
            demand,
            supp=supp,
            supp_cols=["region", "time_index", "load_mw", "weather_year"],
            weather_year=[2012, 2013],
        )


def test_make_load_curves_supp_no_wy_col_no_error(monkeypatch):
    """No weather_year column in the supplemental table -> no coverage check."""
    supp = pd.DataFrame(
        {"region": ["R1"], "time_index": ["all_hours"], "load_mw": [50.0]}
    )
    demand = _long_load({2012: 4, 2013: 4}, 100.0)
    out = _run_make_load_curves(
        monkeypatch,
        demand,
        supp=supp,
        weather_year=[2012, 2013],
    )
    assert list(out["R1"]) == [150.0] * 8


def test_make_load_curves_supp_specific_wy_unknown_raises(monkeypatch):
    """A specific weather_year that cannot be resolved to a load weather year
    raises a descriptive ValueError telling the user to set `weather_year`."""
    # An "all" row covers both years so the coverage check passes, then the
    # specific 2015 row hits the unresolvable-year error.
    supp = pd.DataFrame(
        {
            "region": ["R1", "R1"],
            "time_index": ["all_hours", "all_hours"],
            "load_mw": [10.0, 50.0],
            "weather_year": ["all", 2015],
        }
    )
    demand = _long_load({2012: 4, 2013: 4}, 100.0)
    with pytest.raises(ValueError) as exc_info:
        _run_make_load_curves(
            monkeypatch,
            demand,
            supp=supp,
            supp_cols=["region", "time_index", "load_mw", "weather_year"],
            weather_year=[2012, 2013],
        )
    err = str(exc_info.value)
    assert "cites weather_year 2015" in err
    assert "Set `weather_year`" in err


def test_make_load_curves_supp_all_alias(monkeypatch):
    """time_index='all' behaves like 'all_hours'."""
    supp = pd.DataFrame({"region": ["R1"], "time_index": ["all"], "load_mw": [25.0]})
    demand = _long_load({2012: 4}, 100.0)
    out = _run_make_load_curves(
        monkeypatch,
        demand,
        supp=supp,
        weather_year=[2012],
    )
    assert list(out["R1"]) == [125.0] * 4


def test_make_load_curves_supp_all_hours_alias_upper(monkeypatch):
    """time_index matching is case-insensitive."""
    supp = pd.DataFrame(
        {"region": ["R1"], "time_index": ["ALL_HOURS"], "load_mw": [25.0]}
    )
    demand = _long_load({2012: 4}, 100.0)
    out = _run_make_load_curves(
        monkeypatch,
        demand,
        supp=supp,
        weather_year=[2012],
    )
    assert list(out["R1"]) == [125.0] * 4


def test_make_load_curves_supp_specific_ti_with_wy(monkeypatch):
    """A specific time_index + specific weather_year hits only that block's hour."""
    supp = pd.DataFrame(
        {
            "region": ["R1", "R1"],
            "time_index": [2, "all_hours"],
            "load_mw": [15.0, 0.0],
            "weather_year": [2012, 2013],
        }
    )
    demand = _long_load({2012: 4, 2013: 4}, 100.0)
    out = _run_make_load_curves(
        monkeypatch,
        demand,
        supp=supp,
        supp_cols=["region", "time_index", "load_mw", "weather_year"],
        weather_year=[2012, 2013],
    )
    # Hour 2 of 2012 block (renumbered 2) gets +15; the 2013 block is +0.
    assert list(out["R1"]) == [100.0, 115.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]


def test_make_load_curves_supp_all_wy_expansion(monkeypatch):
    """weather_year='all' expands to EVERY weather year present in the load data."""
    supp = pd.DataFrame(
        {
            "region": ["R1"],
            "time_index": ["all_hours"],
            "load_mw": [10.0],
            "weather_year": ["all"],
        }
    )
    demand = _long_load({2012: 4, 2013: 5}, 100.0)
    out = _run_make_load_curves(
        monkeypatch,
        demand,
        supp=supp,
        supp_cols=["region", "time_index", "load_mw", "weather_year"],
        weather_year=[2012, 2013],
    )
    assert list(out["R1"]) == [110.0] * 9


def test_make_load_curves_supp_all_wy_expansion_no_setting(monkeypatch):
    """weather_year='all' expands to every weather year when the weather_year
    setting is absent (years discovered from the demand table)."""
    supp = pd.DataFrame(
        {
            "region": ["R1"],
            "time_index": ["all_hours"],
            "load_mw": [10.0],
            "weather_year": ["all"],
        }
    )
    demand = _long_load({2012: 4, 2013: 4}, 100.0)
    out = _run_make_load_curves(
        monkeypatch,
        demand,
        supp=supp,
        supp_cols=["region", "time_index", "load_mw", "weather_year"],
        # No weather_year setting: make_load_curves discovers years via a
        # DISTINCT query on the demand table, which the fake handles.
    )
    assert list(out["R1"]) == [110.0] * 8


def test_make_load_curves_supp_multiple_scenarios_raises(monkeypatch):
    """Multiple scenarios without selection raise a descriptive ValueError."""
    supp = pd.DataFrame(
        {
            "region": ["R1", "R1"],
            "time_index": ["all_hours", "all_hours"],
            "load_mw": [100.0, 200.0],
            "scenario": ["low_demand", "high_demand"],
        }
    )
    demand = _long_load({2012: 4}, 100.0)
    with pytest.raises(ValueError, match="multiple scenarios"):
        _run_make_load_curves(
            monkeypatch,
            demand,
            supp=supp,
            supp_cols=["region", "time_index", "load_mw", "scenario"],
            weather_year=[2012],
        )


def test_make_load_curves_supp_error_shows_scenario_options(monkeypatch):
    """Error message lists the available scenario names and the settings key."""
    supp = pd.DataFrame(
        {
            "region": ["R1", "R1"],
            "time_index": ["all_hours", "all_hours"],
            "load_mw": [100.0, 200.0],
            "scenario": ["baseline", "high_data_center"],
        }
    )
    demand = _long_load({2012: 4}, 100.0)
    with pytest.raises(ValueError) as exc_info:
        _run_make_load_curves(
            monkeypatch,
            demand,
            supp=supp,
            supp_cols=["region", "time_index", "load_mw", "scenario"],
            weather_year=[2012],
        )
    error_msg = str(exc_info.value)
    assert "baseline" in error_msg
    assert "high_data_center" in error_msg
    assert "supplemental_demand_table" in error_msg
    assert "scenario:" in error_msg


def test_make_load_curves_supp_single_scenario_no_error(monkeypatch):
    """A single scenario in the table (after filtering) proceeds without error."""
    supp = pd.DataFrame(
        {
            "region": ["R1", "R1"],
            "time_index": ["all_hours", "all_hours"],
            "load_mw": [50.0, 50.0],
            "scenario": ["high_data_center", "high_data_center"],
        }
    )
    demand = _long_load({2012: 4}, 100.0)
    out = _run_make_load_curves(
        monkeypatch,
        demand,
        supp=supp,
        supp_cols=["region", "time_index", "load_mw", "scenario"],
        weather_year=[2012],
    )
    # 50 + 50 = 100 MW added to every hour.
    assert list(out["R1"]) == [200.0] * 4


def test_supp_wide_path_no_wy_col_all_hours(monkeypatch):
    """The slim WIDE variant (user-supplied load path) still applies all_hours."""
    supp = pd.DataFrame(
        {"region": ["R1"], "time_index": ["all_hours"], "load_mw": [50.0]}
    )
    monkeypatch.setattr(lp_mod, "list_tables", lambda: ["supplemental_demand"])
    monkeypatch.setattr(
        lp_mod,
        "get_data",
        lambda table_name, columns=None, filters=None, query=None: (
            pd.DataFrame({"name": ["region", "time_index", "load_mw"]})
            if query is not None
            else supp
        ),
    )
    lc = _base_load_curves(n_hours=4)
    out = add_supplemental_demand(lc, model_year=2030, model_regions=["R1", "R2"])
    assert list(out["R1"]) == [150.0] * 4
    assert list(out["R2"]) == [100.0] * 4


def test_supp_wide_path_specific_wy_raises(monkeypatch):
    """The wide (user-supplied) path rejects weather-year-specific rows."""
    supp = pd.DataFrame(
        {
            "region": ["R1"],
            "time_index": ["all_hours"],
            "load_mw": [50.0],
            "weather_year": [2012],
        }
    )
    monkeypatch.setattr(lp_mod, "list_tables", lambda: ["supplemental_demand"])
    monkeypatch.setattr(
        lp_mod,
        "get_data",
        lambda table_name, columns=None, filters=None, query=None: (
            pd.DataFrame({"name": ["region", "time_index", "load_mw", "weather_year"]})
            if query is not None
            else supp
        ),
    )
    lc = _base_load_curves(n_hours=4)
    with pytest.raises(ValueError, match="does not support weather-year"):
        add_supplemental_demand(lc, model_year=2030, model_regions=["R1"])
