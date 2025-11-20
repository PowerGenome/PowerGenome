"""
Test functions related to load profiles
"""

import pandas as pd

import powergenome.load_profiles as lp_mod
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
    def fake_get_data(table, columns=None, filters=None):
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
