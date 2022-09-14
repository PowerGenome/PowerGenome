"""
Test functions related to load profiles
"""

import pandas as pd
from powergenome.load_profiles import grow_historical_load


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
    start_year = 2019
    end_year, load_2020 = grow_historical_load(
        base_load_df.copy(),
        start_year,
        keep_regions,
        hist_region_map,
        future_region_map,
    )

    assert end_year > start_year
    assert not base_load_df.equals(load_2020)

    end_year, load_2021 = grow_historical_load(
        load_2020.copy(), end_year, keep_regions, hist_region_map, future_region_map
    )

    assert end_year == 2021
    assert not load_2021.equals(load_2020)

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
    start_year = 2019
    end_year, load_2020_sector = grow_historical_load(
        base_load_df.copy(),
        start_year,
        keep_regions,
        hist_region_map,
        future_region_map,
        aeo_sector_map,
    )

    assert not load_2020_sector.equals(load_2020)

    assert end_year > start_year
    assert not base_load_df.equals(load_2020_sector)

    end_year, load_2021_sector = grow_historical_load(
        load_2020_sector.copy(),
        end_year,
        keep_regions,
        hist_region_map,
        future_region_map,
        aeo_sector_map,
    )

    assert end_year == 2021
    assert not load_2021_sector.equals(load_2020_sector)
