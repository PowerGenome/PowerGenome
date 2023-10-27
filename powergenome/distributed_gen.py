import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import pandas as pd
import pyarrow.parquet as pq

from powergenome.load_construction import us_state_abbrev
from powergenome.params import SETTINGS
from powergenome.util import deep_freeze_args, snake_case_col

logger = logging.getLogger(__name__)


def load_region_pop_frac(
    path_in: Path,
    fn: str = "ipm_state_pop_weight_20220329.csv",
) -> pd.DataFrame:
    if (path_in / fn).suffix == ".csv":
        pop = pd.read_csv(path_in / fn)
    elif (path_in / fn).suffix == ".parquet":
        pop = pd.read_parquet(path_in / fn)
    pop_cols = ["region", "dg_region", "frac_dg_in_region"]
    return pop[pop_cols]


def interp_dg(df: pd.DataFrame, year1: int, year2: int, target_year: int) -> pd.Series:
    """Linear interpolation of distributed generation between two data years

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with columns "time_index", "year", and "region_distpv_mwh"
    year1 : int
        First year of data in the df
    year2 : int
        Second year of data in the df
    target_year : int
        The interpolation target year

    Returns
    -------
    pd.Series
        Interpolated values.
    """
    if year1 > year2:
        year1, year2 = year2, year1

    s1 = df.loc[df["year"] == year1, :].set_index("time_index")["region_distpv_mwh"]
    s2 = df.loc[df["year"] == year2, :].set_index("time_index")["region_distpv_mwh"]

    if year1 < target_year < year2:
        w1 = 1 / abs(target_year - year1)
        w2 = 1 / abs(year2 - target_year)
        return ((s1 * w1) + (s2 * w2)) / (w1 + w2)
    elif year1 == target_year:
        return s1
    elif year2 == target_year:
        return s2


@deep_freeze_args
@lru_cache()
def distributed_gen_profiles(
    profile_fn: str,
    year: int,
    scenario: str,
    regions: List[str],
    path_in: Path,
    region_aggregations: Dict[str, str] = None,
) -> pd.DataFrame:
    """Build distributed generation profiles for each region

    Parameters
    ----------
    profile_fn : str
        Name of the data file with distributed generation profiles
    year : int
        Model planning year
    scenario : str
        Scenario name that matches a value from the data file
    regions : List[str]
        Base model regions (not aggregated). These should match the population weight file,
        not the regions in the data file.
    path_in : Path
        Folder where the distributed generation data file and population weight file are
        found.
    region_aggregations : Dict[str, str]
        Mapping of region aggregations from base regions to model regions (where used),
        by default None

    Returns
    -------
    pd.DataFrame
        Wide dataframe with hourly distributed generation from each model region

    Raises
    ------
    KeyError
        No path_in was provided with the DISTRIBUTED_GEN_DATA parameter in the settings
        file or the .env file
    KeyError
        None of the base regions were found in the population weight file
    KeyError
        The distributed generation data file is missing required columns
    ValueError
        The electrification_scenario parameter does not match the values in the data file
    """
    if not path_in:
        try:
            path_in = Path(SETTINGS["DISTRIBUTED_GEN_DATA"])
        except TypeError:
            raise KeyError(
                "The variable 'DISTRIBUTED_GEN_DATA' is not included in your .env file."
            )
    else:
        path_in = Path(path_in)

    pop_files = path_in.glob("*pop_weight*")
    newest_pop_file = max(pop_files, key=os.path.getmtime)
    pop = load_region_pop_frac(path_in=path_in, fn=newest_pop_file.name)
    pop = pop.loc[pop["region"].isin(regions), :]

    # NREL stores their state names as abbreviations. Change to full state name if found.
    if any([p in us_state_abbrev.keys() for p in pop["dg_region"]]):
        for dg_reg in pop["dg_region"].unique():
            pop.loc[pop["dg_region"] == dg_reg, "dg_region"] = us_state_abbrev[dg_reg]

    dg_regions = pop["dg_region"].unique()
    if dg_regions.size == 0:
        raise KeyError(
            "None of your regions were found in the population weighting file "
            f"({newest_pop_file})."
        )

    # Check file to identify possible errors in advance
    pf = pq.ParquetFile(path_in / profile_fn)
    col_names = pf.schema.names
    missing_names = []
    for col in ["region", "time_index", "year", "scenario", "distpv_MWh"]:
        if col.lower() not in [c.lower() for c in col_names]:
            missing_names.append(col)
    if missing_names:
        raise KeyError(
            f"The required columns {missing_names} are not in your distributed generation file"
        )

    scenario_profile = pd.read_parquet(
        path_in / profile_fn,
        filters=[("region", "in", dg_regions), ("scenario", "==", scenario)],
    )
    scenario_profile.columns = snake_case_col(scenario_profile.columns)
    if scenario_profile.empty:
        full_df = pd.read_parquet(
            path_in / profile_fn,
        )
        valid_scenarios = list(full_df.scenario.unique())
        if scenario not in valid_scenarios:
            raise ValueError(
                f"Your 'distributed_gen_scenario' parameter value '{scenario}' was not "
                f"found in the 'distributed_gen_fn' file '{profile_fn}'. Valid "
                f"scenarios are: {valid_scenarios}."
            )
    if not set(dg_regions) == set(scenario_profile["region"].unique()):
        missing_regions = set(dg_regions) - set(scenario_profile["region"].unique())
        logger.warning(
            f"No distributed generation for the regions {missing_regions} was found in "
            f"your distributed generation data for the scenario {scenario}. This is an "
            "informational warning -- check your data file if you think there should be "
            "distributed generation data in this/these region(s)."
        )
    # NREL data are available every other year.
    dg_years = scenario_profile.year.unique()
    if year in dg_years:
        years = [year]
    elif year not in dg_years and (min(dg_years) < year < max(dg_years)):
        year1 = max([y for y in dg_years if y < year])
        year2 = min([y for y in dg_years if y > year])
        years = [year1, year2]
        logger.info(
            f"Distributed generation profiles are not available for the year {year}. "
            f"Using a linear interpolation between {year1} and {year2}."
        )
    elif year not in dg_years and all(year < dg_years):
        years = [min(dg_years)]
        logger.warning(
            f"The first data year for distributed generation profiles is {years[0]}, "
            f"which is after your planning period year ({year}). Using the first data "
            "year since no earlier data have been provided."
        )
    else:
        years = [max(dg_years)]
        logger.warning(
            f"The last data year for distributed generation profiles is {years[0]}, "
            f"which is before your planning period year ({year}). Using the last data "
            "year since no later data have been provided."
        )
    scenario_profile = scenario_profile.loc[
        scenario_profile["year"].isin(years), :
    ].rename(columns={"region": "dg_region"})
    scenario_profile = pd.merge(scenario_profile, pop, on=["dg_region"])

    if not set(regions) == set(scenario_profile["region"].unique()):
        missing_regions = set(regions) - set(scenario_profile["region"].unique())
        logger.warning(
            f"No distributed generation for the regions {missing_regions} was found in "
            f"your distributed generation data for the scenario {scenario}. This is an "
            "informational warning -- check your data file if you think there should be "
            "distributed generation data in this/these region(s)."
        )

    for k, v in (region_aggregations or {}).items():
        scenario_profile.loc[scenario_profile["region"].isin(v), "region"] = k

    scenario_profile["region_distpv_mwh"] = (
        scenario_profile.distpv_mwh * scenario_profile.frac_dg_in_region
    )
    if len(years) == 2:
        # First combine by year/region. A region might be represented more than once.
        region_scenario_year_profile = scenario_profile.groupby(
            ["year", "region", "time_index"], as_index=False
        )["region_distpv_mwh"].sum()

        # Then interpolate between data years
        region_scenario_profile = (
            region_scenario_year_profile.groupby("region")
            .apply(interp_dg, years[0], years[1], year)
            .T
        )
    else:
        region_scenario_profile = scenario_profile.groupby(
            ["time_index", "region"], as_index=True
        )["region_distpv_mwh"].sum()
        region_scenario_profile = region_scenario_profile.unstack()

    return region_scenario_profile.reset_index(drop=True)
