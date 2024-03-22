"Test functions for loading distributed generation scenario data"

from pathlib import Path

import pandas as pd
import pytest

from powergenome.distributed_gen import (
    distributed_gen_profiles,
    interp_dg,
    load_region_pop_frac,
)

CWD = Path.cwd()
DATA_FOLDER = CWD / "tests" / "data" / "dist_gen"


def test_load_pop_frac():
    fn = "ipm_state_pop_weight_20220329.csv"
    pop_frac = load_region_pop_frac(path_in=DATA_FOLDER, fn=fn)

    fn = "ipm_state_pop_weight_20220329.parquet"
    pop_frac = load_region_pop_frac(path_in=DATA_FOLDER, fn=fn)


def test_interp():
    year1 = 2020
    year2 = 2030
    data = {
        "time_index": [0, 1, 2, 3] * 2,
        "year": [year1] * 4 + [year2] * 4,
        "region_distpv_mwh": [0] * 4 + [10] * 4,
    }
    df = pd.DataFrame(data)

    target_year = 2025
    interp_results = interp_dg(df, year1, year2, target_year)
    assert interp_results.mean() == 5

    target_year = 2027
    interp_results = interp_dg(df, year2, year1, target_year)
    assert interp_results.mean() == 7

    target_year = 2030
    interp_results = interp_dg(df, year1, year2, target_year)
    assert interp_results.mean() == 10

    target_year = 2020
    interp_results = interp_dg(df, year1, year2, target_year)
    assert interp_results.mean() == 0


def test_distributed_gen_profiles():
    profile_fn = "nrel_cambium_distr_pv_2022_slim.parquet"
    scenario = "MidCase"

    dg = distributed_gen_profiles(
        profile_fn=profile_fn,
        year=2025,
        scenario=scenario,
        regions=["WECC_PNW", "WECC_MT"],
        path_in=DATA_FOLDER,
    )
    assert not dg.empty
    assert len(dg.columns) == 2
    assert dg.mean().mean() > 0

    dg = distributed_gen_profiles(
        profile_fn=profile_fn,
        year=2024,
        scenario=scenario,
        regions=["WECC_PNW", "WECC_SNV", "WECC_NNV"],
        path_in=DATA_FOLDER,
        region_aggregations={"NV": ["WECC_SNV", "WECC_NNV"]},
    )
    assert not dg.empty
    assert len(dg.columns) == 2
    assert dg.mean().mean() > 0

    with pytest.raises(ValueError):
        dg = distributed_gen_profiles(
            profile_fn=profile_fn,
            year=2025,
            scenario="invalid",
            regions=["WECC_PNW", "WECC_MT"],
            path_in=DATA_FOLDER,
        )

    with pytest.raises(KeyError):
        dg = distributed_gen_profiles(
            profile_fn=profile_fn,
            year=2025,
            scenario=scenario,
            regions=["invalid"],
            path_in=DATA_FOLDER,
        )

    with pytest.raises(KeyError):
        dg = distributed_gen_profiles(
            path_in=None,
            profile_fn=profile_fn,
            year=2025,
            scenario=scenario,
            regions=["invalid"],
        )
