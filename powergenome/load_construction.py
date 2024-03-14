import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd

from powergenome.params import DATA_PATHS, SETTINGS
from powergenome.util import deep_freeze_args, find_region_col, snake_case_col

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
    context = "Loading region population fraction file for EFS load construction."
    region_col = find_region_col(pop.columns, context)
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
    utc_offset: int = 0,
    path_in: Path = None,
) -> pd.DataFrame:
    """Build hourly demand for a single sector/subsector in a given year and scenario.

    This uses files with stock projections and "incremental factors" of hourly demand
    for each stock type. Two stock types and hourly factors are supported (based on how
    data were originally derived from NREL EFS). Data can be shifted from the original
    timezone (e.g. UTC) to a different timezone.

    Sector and subsector names must correspond to the stock and incremental factor files
    used.

    Stock values must be for the model year specified.

    Parameters
    ----------
    sector : str
        Name of the stock sector (e.g. Residential)
    subsector : str
        Name of the stock subsector (e.g. water heating)
    year : int
        Modeling year, for selecting incremental factors
    scenario_stock : pd.DataFrame
        Stock of different sector/subsectors in the model year in applicable states. Must
        have the columns "sector", "subsector", "state", "year", "agg_stock_type1", and
        "agg_stock_type2".
    utc_offset : int, optional
        Number of hours to shift the data away from UTC, by default 0
    path_in : Path, optional
        Folder with incremental factor timeseries data, by default None. There should be
        a file in this folder named with the convention "{sector}_{subsector}_Incremental_Factor.parquet"

    Returns
    -------
    pd.DataFrame
        Hourly demand of a single sector/subsector across one or more states, adjusted
        to the specified timezone (relative to data storage)
    """
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
    timeseries = (
        timeseries[["state", "time_index", "load_mw"]]
        .dropna()
        .groupby("state")
        .apply(utc_offset_state_load, utc_offset)
    )
    return timeseries


def utc_offset_state_load(df: pd.DataFrame, utc_offset: int = 0) -> pd.DataFrame:
    """Shift hourly load within a single dataframe from UTC to an offset timezone

    Parameters
    ----------
    df : pd.DataFrame
        Hourly load of some type. Must have the column "load_mw"
    utc_offset : int, optional
        Number of hours to shift away from the data timezone (e.g. UTC), by default 0

    Returns
    -------
    pd.DataFrame
        Time-shifted demand
    """
    df["load_mw"] = np.roll(df["load_mw"], utc_offset)
    return df


def state_demand_to_region(
    df: pd.DataFrame,
    pop: pd.DataFrame,
    on: List[str] = ["state"],
    by: List[str] = ["time_index", "region"],
) -> pd.DataFrame:
    """Allocate hourly demand from states to regions based on population proportion

    Parameters
    ----------
    df : pd.DataFrame
        Hourly demand by one or more resource types in different states. Must have columns
        "state" and "time_index"
    pop : pd.DataFrame
        Proportion of state population in each region. Must have columns "state", "state_prop",
        and "region".


    Returns
    -------
    pd.DataFrame
        Hourly demand data ("load_mw") grouped by "time_index" and "region"
    """
    temp = pd.merge(df, pop, on=on, how="left")
    temp["load_mw"] *= temp["state_prop"]
    temp = temp.groupby(by, as_index=False)["load_mw"].sum()
    return temp


@deep_freeze_args
@lru_cache()
def electrification_profiles(
    stock_fn: str,
    year: int,
    elec_scenario: str,
    regions: List[str],
    utc_offset: int = 0,
    path_in: Path = None,
) -> pd.DataFrame:
    """Create demand profiles for potentially flexible resources that will be electrified.

    Profiles are shifted from stored timezone (assume UTC) to the model timezone.

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
    utc_offset : int, optional
        Hours to shift data from UTC to the desired timezone for the model, by default 0.
    path_in : Path, optional
        Folder where stock and incremental factor (profile) data are located, by default
        None.

    Returns
    -------
    pd.DataFrame
        Columns include "time_index", "region", "load_mw", and "resource".

    Raises
    ------
    KeyError
        No "EFS_DATA" key is included in the .env file.
    FileNotFoundError
        The "EFS_DATA" or "path_in" folder does not exist.
    ValueError
        The "electrificication_scenario" parameter does not match scenarios in the stock
        file.
    """
    if not path_in:
        try:
            path_in = Path(SETTINGS["EFS_DATA"])
        except TypeError:
            raise KeyError("The variable 'EFS_DATA' is not included in your .env file.")
    if not path_in.is_dir():
        raise FileNotFoundError(
            f"The folder with EFS/flexible demand data ({str(path_in)}) was not found."
        )

    pop_files = path_in.glob("*pop_weight*")
    newest_pop_file = max(pop_files, key=os.path.getmtime)
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
        df = create_subsector_ts(
            sector, subsector, year, scenario_stock, utc_offset, path_in
        ).pipe(state_demand_to_region, pop)

        df["resource"] = name
        subsector_ts_dfs.append(df)

    elec_profiles = pd.concat(subsector_ts_dfs)

    return elec_profiles
