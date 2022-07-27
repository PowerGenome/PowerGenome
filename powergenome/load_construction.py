from typing import Dict, List, Union
import numpy as np
import pandas as pd
import os
import logging
from functools import lru_cache, reduce

from pathlib import Path
from joblib import Memory

import powergenome.load_profiles as load_profiles

from powergenome.util import (
    deep_freeze_args,
    init_pudl_connection,
    regions_to_keep,
    snake_case_col,
)
from powergenome.params import DATA_PATHS, SETTINGS

logger = logging.getLogger(__name__)

memory = Memory(location=DATA_PATHS["cache"], verbose=0)

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

historical_load_region_map = {
    "TRE": ["ERC_PHDL", "ERC_REST", "ERC_WEST"],
    "FLRC": ["FRCC"],
    "MWRCE": ["MIS_WUMS"],
    "MWRCW": ["MIS_MAPP", "SPP_WAUE", "SPP_NEBR", "MIS_MIDA", "MIS_IA", "MIS_MNWI"],
    "NPCCNE": ["NENG_ME", "NENG_CT", "NENGREST"],
    "NPCCNYWE": ["NY_Z_J"],
    "NPCCLI": ["NY_Z_K"],
    "NPCCUPNY": ["NY_Z_A", "NY_Z_B", "NY_Z_C&E", "NY_Z_D", "NY_Z_F", "NY_Z_G-I"],
    "RFCET": ["PJM_WMAC", "PJM_EMAC", "PJM_SMAC", "PJM_PENE", "PJM_Dom"],
    "RFCMI": ["MIS_LMI"],
    "RFCWT": ["PJM_West", "PJM_AP", "PJM_ATSI", "PJM_COMD"],
    "SERCDLT": ["MIS_WOTA", "MIS_LA", "MIS_AMSO", "MIS_AR", "MIS_D_MS"],
    "SERCGW": ["MIS_MO", "S_D_AECI", "MIS_IL", "MIS_INKY"],
    "SERCSOES": ["S_SOU"],
    "SERCCNT": ["S_C_TVA", "S_C_KY"],
    "SERCVC": ["S_VACA"],
    "SWPPNO": ["SPP_N"],
    "SWPPSO": ["SPP_SPS", "SPP_WEST"],
    "WECCSW": ["WECC_AZ", "WECC_NM", "WECC_SNV"],
    "WECCCA": ["WEC_CALN", "WEC_BANC", "WECC_IID", "WECC_SCE", "WEC_LADW", "WEC_SDGE"],
    "WENWPP": ["WECC_PNW", "WECC_MT", "WECC_ID", "WECC_WY", "WECC_UT", "WECC_NNV"],
    "WECCRKS": ["WECC_CO"],
}

future_load_region_map = {
    "TRE": ["ERC_PHDL", "ERC_REST", "ERC_WEST"],
    "FLRC": ["FRCC"],
    "MCW": ["MIS_WUMS", "MIS_MNWI", "MIS_IA"],
    "MCE": ["MIS_LMI"],
    "PJMCE": ["PJM_COMD"],
    "MCC": ["MIS_IL", "MIS_MO", "S_D_AECI", "MIS_INKY"],
    "SWPPNO": ["MIS_MAPP", "SPP_WAUE", "SPP_NEBR", "MIS_MIDA"],
    "SWPPC": ["SPP_N"],
    "SWPPSO": ["SPP_WEST", "SPP_SPS"],
    "MCS": ["MIS_AMSO", "MIS_WOTA", "MIS_LA", "MIS_AR", "MIS_D_MS"],
    "SERCSOES": ["S_SOU"],
    "SERCE": ["S_VACA"],
    "PJMD": ["PJM_Dom"],
    "PJMW": ["PJM_West", "PJM_AP", "PJM_ATSI"],
    "PJME": ["PJM_WMAC", "PJM_EMAC", "PJM_SMAC", "PJM_PENE"],
    "SERCCNT": ["S_C_TVA", "S_C_KY"],
    "NPCCUPNY": ["NY_Z_A", "NY_Z_B", "NY_Z_C&E", "NY_Z_D", "NY_Z_F", "NY_Z_G-I"],
    "NENYCLI": ["NY_Z_J", "NY_Z_K"],
    "NPCCNE": ["NENG_ME", "NENGREST", "NENG_CT"],
    "WECCRKS": ["WECC_CO"],
    "WECCB": ["WECC_ID", "WECC_WY", "WECC_UT", "WECC_NNV"],
    "WENWPP": ["WECC_PNW", "WECC_MT"],
    "WECCCAN": ["WEC_CALN", "WEC_BANC"],
    "WECCCAS": ["WECC_IID", "WECC_SCE", "WEC_LADW", "WEC_SDGE"],
    "WECCSW": ["WECC_AZ", "WECC_NM", "WECC_SNV"],
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


@memory.cache
def CreateBaseLoad(
    years: List[int],
    regions: List[str],
    future_load_region_map: Dict[str, dict],
    growth_scenario: str,
    eia_aeo_year: int,
    regular_load_growth_start_year: int = 2019,
    alt_growth_rate: Dict[str, float] = {},
    pop_fn: Union[str, Path] = None,
    path_in: Path = None,
) -> pd.DataFrame:
    if not path_in:
        try:
            path_in = Path(SETTINGS["EFS_DATA"])
        except TypeError:
            logger.warning("The variable 'EFS_DATA' is not included in your .env file.")
    load_dtypes = {
        "Year": "category",
        "LocalHourID": "category",
        "Sector": "category",
        "Subsector": "category",
        "LoadMW": np.float32,
    }
    if pop_fn:
        pop = load_region_pop_frac(path_in=path_in, fn=pop_fn)
    else:
        pop = load_region_pop_frac()
    model_states = pop.loc[pop["ipm_region"].isin(regions), "state"]
    efs_2020_load_prof = pd.read_parquet(path_in / "EFS_REF_load_2020.parquet")
    efs_2020_load_prof = efs_2020_load_prof.astype(load_dtypes)
    efs_2020_load_prof.columns = snake_case_col(efs_2020_load_prof.columns)
    total_efs_2020_load = efs_2020_load_prof["loadmw"].sum()

    efs_2020_load_prof = efs_2020_load_prof.loc[
        efs_2020_load_prof["state"].isin(model_states), :
    ]
    efs_2020_load_prof = pd.merge(efs_2020_load_prof, pop, on=["state"])
    efs_2020_load_prof = efs_2020_load_prof.assign(
        weighted=efs_2020_load_prof["loadmw"] * efs_2020_load_prof["state_prop"]
    )
    efs_2020_load_prof = efs_2020_load_prof.groupby(
        ["year", "ipm_region", "localhourid", "sector", "subsector"],
        as_index=False,
        observed=True,
    ).agg({"weighted": "sum"})

    # Read in 2019 Demand
    demand_settings = {
        "historical_load_region_map": historical_load_region_map,
        "future_load_region_map": future_load_region_map,
        "model_year": 2019,
        "model_regions": list(pop["ipm_region"].unique()),
        "utc_offset": -5,
    }
    pudl_engine, pudl_out, pg_engine = init_pudl_connection()
    original_load_2019 = load_profiles.make_load_curves(pg_engine, demand_settings)
    original_load_2019 = original_load_2019.reset_index().rename(
        columns={"time_index": "localhourid"}
    )
    original_load_2019 = original_load_2019.melt(id_vars="localhourid").rename(
        columns={"variable": "ipm_region", "value": "loadmw_original"}
    )

    ratio_A = original_load_2019["loadmw_original"].sum() / total_efs_2020_load
    efs_2020_load_prof["weighted"] *= ratio_A

    base_load_2019 = efs_2020_load_prof.rename(columns={"weighted": "loadmw"})

    # Create Base loads
    base_load_2019 = base_load_2019.loc[base_load_2019["ipm_region"].isin(regions), :]
    base_load_2019.loc[
        (base_load_2019["sector"] == "Industrial")
        & (base_load_2019["subsector"].isin(["process heat", "machine drives"])),
        "subsector",
    ] = "other"
    base_load_2019 = base_load_2019.loc[base_load_2019["subsector"] == "other", :]
    base_load_2019 = base_load_2019.groupby(
        ["year", "localhourid", "ipm_region", "sector"], as_index=False
    ).agg({"loadmw": "sum"})
    base_load = base_load_2019
    for y in years:
        growth_factor = load_profiles.calc_growth_factors(
            regions,
            load_region_map=future_load_region_map,
            growth_scenario=growth_scenario,
            eia_aeo_year=eia_aeo_year,
            start_year=regular_load_growth_start_year,
            end_year=y,
            alt_growth_rate=alt_growth_rate,
        )
        base_load_temp = base_load_2019.copy()
        base_load_temp["year"] = y
        base_load_temp["loadmw"] *= base_load_temp["ipm_region"].map(growth_factor)
        base_load = base_load.append(base_load_temp, ignore_index=True)
    return base_load


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


# @hash_dict
# @freezeargs
@deep_freeze_args
@lru_cache()
def AddElectrification(
    stock_fn: str,
    year: int,
    elec_scenario: str,
    regions: List[str],
    output_folder: Path,
    future_load_region_map: Dict[str, dict],
    growth_scenario: str,
    eia_aeo_year: int,
    regular_load_growth_start_year: int = 2019,
    alt_growth_rate: Dict[str, float] = {},
    path_in: Path = None,
) -> pd.DataFrame:
    if not path_in:
        try:
            path_in = Path(SETTINGS["EFS_DATA"])
        except TypeError:
            logger.warning("The variable 'EFS_DATA' is not included in your .env file.")
    try:
        if (output_folder / "load_by_region_sector.parquet").exists():
            return pd.read_parquet(output_folder / "load_by_region_sector.parquet")
    except:
        pass
    # Creating Time-series
    pop_files = path_in.glob("*pop_weight*")
    newest_pop_file = max(pop_files, key=os.path.getctime)
    pop = load_region_pop_frac(path_in=path_in, fn=newest_pop_file.name)
    states = pop.loc[pop["ipm_region"].isin(regions), "state"].unique()
    scenario_stock = pd.read_parquet(path_in / stock_fn)
    scenario_stock.columns = snake_case_col(scenario_stock.columns)
    scenario_stock = scenario_stock[
        (scenario_stock["year"] == year)
        & (scenario_stock["scenario"] == elec_scenario)
        & (scenario_stock["state"].isin(states))
    ]

    subsector_ts_dfs = {}
    for name, (sector, subsector) in running_sectors.items():
        subsector_ts_dfs[name] = create_subsector_ts(
            sector, subsector, year, scenario_stock, path_in
        )

    column_names = [
        "Res_SPH",
        "Res_WH",
        "Com_SPH",
        "Com_WH",
        "LDV",
        "MDV",
        "HDV",
        "BUS",
    ]
    for col_name, (name, df) in zip(column_names, subsector_ts_dfs.items()):
        subsector_ts_dfs[name] = state_demand_to_region(df, pop, col_name)

    ######
    # Construct Total Load
    df_list = []
    for r in regions:
        _df = CreateBaseLoad(
            [year],
            [r],
            future_load_region_map,
            growth_scenario,
            eia_aeo_year,
            regular_load_growth_start_year,
            alt_growth_rate,
            newest_pop_file,
            path_in,
        )
        df_list.append(_df)
    base_load = pd.concat(df_list, ignore_index=True)

    base_load = base_load.rename(columns={"loadmw": "base_MW"})

    base_load.loc[(base_load["sector"] == "Commercial"), "Subsector"] = "Base_Com_other"
    base_load.loc[
        (base_load["sector"] == "Residential"), "Subsector"
    ] = "Base_Res_other"
    base_load.loc[
        (base_load["sector"] == "Transportation"), "Subsector"
    ] = "Base_Trans_other"
    base_load.loc[(base_load["sector"] == "Industrial"), "Subsector"] = "Base_Ind"
    base_load = base_load.drop(columns=["sector"])
    base_load = base_load.pivot_table(
        index=["localhourid", "ipm_region", "year"],
        columns="Subsector",
        values="base_MW",
    ).reset_index()
    base_load = base_load.astype({"year": "category", "ipm_region": "category"})
    print("grouping base load")
    base_load = base_load.groupby(
        ["localhourid", "ipm_region", "year"], as_index=False
    ).agg(
        {
            "Base_Com_other": "sum",
            "Base_Res_other": "sum",
            "Base_Trans_other": "sum",
            "Base_Ind": "sum",
        }
    )

    total_load = pd.DataFrame()
    for name, df in subsector_ts_dfs.items():
        if total_load.empty:
            total_load = df
        else:
            total_load = pd.merge(
                total_load, df, on=["scenario", "year", "localhourid", "ipm_region"]
            )
    total_load = pd.merge(
        total_load, base_load, on=["year", "localhourid", "ipm_region"]
    )

    total_load = total_load.astype(
        {
            "scenario": "category",
            "year": "category",
            "Res_WH_MW": np.float32,
            "Com_WH_MW": np.float32,
            "Res_SPH_MW": np.float32,
            "Com_SPH_MW": np.float32,
            "LDV_MW": np.float32,
            "MDV_MW": np.float32,
            "HDV_MW": np.float32,
            "Base_Com_other": np.float32,
            "Base_Res_other": np.float32,
            "Base_Trans_other": np.float32,
            "Base_Ind": np.float32,
        }
    )
    if output_folder:
        total_load.to_parquet(
            output_folder / "load_by_region_sector.parquet", index=False
        )

    return total_load


def build_total_load(
    stock_fn: str,
    year: int,
    elec_scenario: str,
    regions: List[str],
    output_folder: Path,
    future_load_region_map: Dict[str, dict],
    growth_scenario: str,
    eia_aeo_year: int,
    regular_load_growth_start_year: int = 2019,
    alt_growth_rate: Dict[str, float] = {},
    path_in: Path = None,
) -> pd.DataFrame:

    total_load = AddElectrification(
        stock_fn,
        year,
        elec_scenario,
        regions,
        output_folder,
        future_load_region_map,
        growth_scenario,
        eia_aeo_year,
        regular_load_growth_start_year,
        alt_growth_rate,
        path_in,
    )

    if output_folder and not (output_folder / "load_by_region_sector.parquet").exists():
        total_load.to_parquet(
            output_folder / "load_by_region_sector.parquet", index=False
        )
    return total_load


def FilterTotalProfile(settings: dict, total_load: pd.DataFrame) -> pd.DataFrame:
    total_load = total_load.assign(
        TotalMW=total_load["Res_WH_MW"]
        + total_load["Com_WH_MW"]
        + total_load["Res_SPH_MW"]
        + total_load["Com_SPH_MW"]
        + total_load["LDV_MW"]
        + total_load["MDV_MW"]
        + total_load["HDV_MW"]
        + total_load["Base_Com_other"]
        + total_load["Base_Res_other"]
        + total_load["Base_Trans_other"]
        + total_load["Base_Ind"]
    ).drop(
        columns=[
            "Res_WH_MW",
            "Com_WH_MW",
            "Res_SPH_MW",
            "Com_SPH_MW",
            "LDV_MW",
            "MDV_MW",
            "HDV_MW",
            "Base_Com_other",
            "Base_Res_other",
            "Base_Trans_other",
            "Base_Ind",
            "BUS_MW",
            "scenario",
            "year",
        ]
    )

    total_load = total_load.rename(
        columns={"localhourid": "time_index", "ipm_region": "region"}
    )
    total_load["region"] = total_load["region"].astype("object")

    keep_regions, region_agg_map = regions_to_keep(
        settings["model_regions"], settings.get("region_aggregations")
    )

    total_load.loc[:, "region_id_epaipm"] = total_load.loc[:, "region"]
    total_load.loc[
        total_load.region_id_epaipm.isin(region_agg_map.keys()), "region"
    ] = total_load.loc[
        total_load.region_id_epaipm.isin(region_agg_map.keys()), "region_id_epaipm"
    ].map(
        region_agg_map
    )

    logger.info("Aggregating load curves in grouped regions")
    total_load = total_load.groupby(["region", "time_index"]).sum()

    total_load = total_load.unstack(level=0)
    total_load.columns = total_load.columns.droplevel()
    return total_load
