from typing import Dict, List, Union
import numpy as np
import pandas as pd
import os
import logging
from functools import lru_cache

from pathlib import Path


from powergenome.util import (
    deep_freeze_args,
    snake_case_col,
)
from powergenome.params import DATA_PATHS, SETTINGS

logger = logging.getLogger(__name__)


us_state_abbrev = {
    "Alabama": "AL",
    "Alaska": "AK",
    "American Samoa": "AS",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "District Of Columbia": "DC",
    "District of Columbia": "DC",
    "Florida": "FL",
    "Georgia": "GA",
    "Guam": "GU",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Northern Mariana Islands": "MP",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Puerto Rico": "PR",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virgin Islands": "VI",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
}


def load_region_pop_frac(
    path_in: Path,
    fn: str = "ipm_state_pop_weight.parquet",
) -> pd.DataFrame:
    # TODO #178 finalize state pop weight file and filename
    # read in state proportions
    # how much state load should be distributed to GenXRegion

    if (path_in / fn).suffix == ".csv":
        pop = pd.read_csv(path_in / fn)
    elif (path_in / fn).suffix == ".parquet":
        pop = pd.read_parquet(path_in / fn)
    pop["state"] = pop["state"].map(us_state_abbrev)
    region_col = [c for c in pop.columns if "region" in c][0]
    pop = pop.rename(columns={region_col: "region"})
    pop_cols = ["region", "state", "state_prop"]
    return pop[pop_cols]


running_sectors = {
    "res_space_heat_cool": ("Residential", "space heating and cooling"),
    "res_water_heat": ("Residential", "water heating"),
    "comm_space_heat_cool": ("Commercial", "space heating and cooling"),
    "comm_water_heat": ("Commercial", "water heating"),
    "trans_light_duty": ("Transportation", "light-duty vehicles"),
    "trans_medium_duty": ("Transportation", "medium-duty trucks"),
    "trans_heavy_duty": ("Transportation", "heavy-duty trucks"),
    "trans_bus": ("Transportation", "transit buses"),
}


def CreateOutputFolder(case_folder):
    path = case_folder / "extra_outputs"
    if not os.path.exists(path):
        os.makedirs(path)


def create_subsector_ts(
    sector: str,
    subsector: str,
    year: int,
    scenario_stock: pd.DataFrame,
    path_in: Path = None,
) -> pd.DataFrame:
    if not path_in:
        try:
            path_in = Path(SETTINGS["EFS_DATA"])
        except TypeError:
            logger.warning("The variable 'EFS_DATA' is not included in your .env file.")

    timeseries = pd.read_parquet(
        path_in / f"{sector}_{subsector}_Incremental_Factor.parquet"
    )
    timeseries.columns = snake_case_col(timeseries.columns)
    timeseries = timeseries.rename(columns={"localhourid": "time_index"})
    ts_cols = ["state", "year", "time_index", "unit", "factor_type1", "factor_type2"]
    timeseries = timeseries.loc[:, ts_cols]
    stock_temp = scenario_stock.loc[
        (scenario_stock["sector"] == sector)
        & (scenario_stock["subsector"] == subsector),
        ["state", "year", "agg_stock_type1", "agg_stock_type2"],
    ]

    factor_years = timeseries["year"].unique()
    if not year in factor_years:
        diff = np.array(factor_years - year)
        index = diff[np.where(diff <= 0)].argmax()
        year_approx = factor_years[index]
        timeseries = timeseries.loc[timeseries["year"] == year_approx, :]
        timeseries["year"] = year
        logger.warning(
            f"No incremental factor available for {sector}/{subsector} "
            f"in year {year}: using factors from year {year_approx} instead."
        )

    timeseries = pd.merge(timeseries, stock_temp, on=["state", "year"], validate="m:1")
    timeseries = timeseries.assign(
        load_mw=timeseries["agg_stock_type1"] * timeseries["factor_type1"]
        + timeseries["agg_stock_type2"] * timeseries["factor_type2"]
    )
    return timeseries[["state", "time_index", "load_mw"]].dropna()


def state_demand_to_region(
    df: pd.DataFrame, pop: pd.DataFrame, col_name: str
) -> pd.DataFrame:
    temp = pd.merge(df, pop, on=["state"], how="left")
    temp["load_mw"] *= temp["state_prop"]
    temp = temp.groupby(["time_index", "region"], as_index=False)["load_mw"].sum()
    return temp


@deep_freeze_args
@lru_cache()
def electrification_profiles(
    stock_fn: str,
    year: int,
    elec_scenario: str,
    regions: List[str],
    path_in: Path = None,
) -> pd.DataFrame:
    """Create demand profiles for potentially flexible resources that will be electrified.

    Parameters
    ----------
    stock_fn : str
        Name of the data file with stock values for each year.
    year : int
        Planning period or model year. Used to select stock values for flexible resources
        and their demand profiles.
    elec_scenario : str
        Name of a scenario from the stock data file.
    regions : List[str]
        All of the base regions that will have profiles created.
    path_in : Path, optional
        Folder where stock and incremental factor (profile) data are located, by default
        None.

    Returns
    -------
    pd.DataFrame
        Columns include "time_index", "region"

    Raises
    ------
    ValueError
        _description_
    """
    if not path_in:
        try:
            path_in = Path(SETTINGS["EFS_DATA"])
        except TypeError:
            logger.warning("The variable 'EFS_DATA' is not included in your .env file.")

    pop_files = path_in.glob("*pop_weight*")
    newest_pop_file = max(pop_files, key=os.path.getctime)
    pop = load_region_pop_frac(path_in=path_in, fn=newest_pop_file.name)
    pop = pop.loc[pop["region"].isin(regions), :]
    states = pop["state"].unique()
    scenario_stock = pd.read_parquet(path_in / stock_fn)
    scenario_stock.columns = snake_case_col(scenario_stock.columns)
    valid_scenarios = list(scenario_stock.scenario.unique())
    if elec_scenario not in valid_scenarios:
        raise ValueError(
            f"Your 'electrification_scenario' parameter value '{elec_scenario}' was not "
            f"found in the 'electrification_stock_fn' file '{stock_fn}'. Valid scenarios "
            f"are: {valid_scenarios}."
        )
    scenario_stock = scenario_stock.loc[
        (scenario_stock["year"] == year)
        & (scenario_stock["scenario"] == elec_scenario)
        & (scenario_stock["state"].isin(states)),
        :,
    ]

    subsector_ts_dfs = []
    for name, (sector, subsector) in running_sectors.items():
        df = create_subsector_ts(sector, subsector, year, scenario_stock, path_in).pipe(
            state_demand_to_region, pop, name
        )

        df["resource"] = name
        subsector_ts_dfs.append(df)

    elec_profiles = pd.concat(subsector_ts_dfs)

    return elec_profiles
