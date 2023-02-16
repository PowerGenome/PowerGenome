from pathlib import Path
from typing import List
import os
import logging
from functools import lru_cache

import pandas as pd

from powergenome.util import (
    deep_freeze_args,
    find_region_col,
    snake_case_col,
)
from powergenome.params import DATA_PATHS, SETTINGS

logger = logging.getLogger(__name__)


def load_region_pop_frac(
    path_in: Path,
    fn: str = "ipm_reeds_pop_weight_20230210.csv",
) -> pd.DataFrame:
    if (path_in / fn).suffix == ".csv":
        pop = pd.read_csv(path_in / fn)
    elif (path_in / fn).suffix == ".parquet":
        pop = pd.read_parquet(path_in / fn)
    context = "Loading region population fraction file for distributed generation construction."
    # region_col = find_region_col(pop.columns, context)
    # pop = pop.rename(columns={region_col: "region"})
    pop_cols = ["region", "dg_region", "frac_dg_in_region"]
    return pop[pop_cols]


@deep_freeze_args
@lru_cache()
def dg_profiles(
    profile_fn: str, year: int, scenario: str, regions: List[str], path_in: Path
) -> pd.DataFrame:
    if not path_in:
        try:
            path_in = Path(SETTINGS["DG_DATA"])
        except TypeError:
            raise KeyError("The variable 'DG_DATA' is not included in your .env file.")

    pop_files = path_in.glob("*pop_weight*")
    newest_pop_file = max(pop_files, key=os.path.getctime)
    pop = load_region_pop_frac(path_in=path_in, fn=newest_pop_file.name)
    pop = pop.loc[pop["region"].isin(regions), :]
    dg_regions = pop["dg_region"].unique()
    scenario_profile = pd.read_parquet(
        path_in / profile_fn, filters=[("region", "in", dg_regions)]
    )

    # scenario_profile = pd.read_parquet(
    #     path_in / profile_fn,
    #     filters=[("scenario", "=", scenario), ("region", "in", dg_regions), ("year", "in", "years")]
    # )
    # scenario_profile.rename(columns={"region": "dg_region"}, inplace=True)
    valid_scenarios = list(scenario_profile.scenario.unique())
    if scenario not in valid_scenarios:
        raise ValueError(
            f"Your 'electrification_scenario' parameter value '{scenario}' was not "
            f"found in the 'electrification_stock_fn' file '{profile_fn}'. Valid scenarios "
            f"are: {valid_scenarios}."
        )
    # NREL data are available every other year.
    dg_years = scenario_profile.year.unique()
    if year not in dg_years:
        years = [y for y in [year - 1, year + 1] if y in dg_years]
    else:
        years = [year]
    scenario_profile = scenario_profile.loc[
        (scenario_profile["year"].isin(years))
        & (scenario_profile["scenario"] == scenario),
        # & (scenario_profile["region"].isin(dg_regions)),
        :,
    ].rename(columns={"region": "dg_region"})
    scenario_profile["region"] = scenario_profile["dg_region"].map(
        pop.set_index("dg_region")["region"]
    )
    scenario_profile = scenario_profile.groupby(
        ["time_index", "region"], as_index=True
    )["distpv_MWh"].sum() / len(years)
    # scenario_profile = scenario_profile.reset_index()

    dg_profiles = scenario_profile.unstack()

    return dg_profiles
