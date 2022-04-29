from typing import Dict, List
import numpy as np
import pandas as pd
import os
import logging
from functools import lru_cache

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

try:
    path_in = Path(SETTINGS["EFS_DATA"])
except TypeError:
    logger.warning("The variable 'EFS_DATA' is not included in your .env file.")

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
    fn: str = "ipm_state_pop_weight_20210517.parquet",
) -> pd.DataFrame:
    # TODO #178 finalize state pop weight file and filename
    # read in state proportions
    # how much state load should be distributed to GenXRegion
    pop_cols = ["ipm_region", "state", "state_prop"]
    pop = pd.read_parquet(path_in / fn, columns=pop_cols)
    pop["state"] = pop["state"].map(us_state_abbrev)
    return pop


running_sectors = {
    "res_sph": ("Residential", "space heating and cooling"),
    "res_wh": ("Residential", "water heating"),
    "com_sph": ("Commercial", "space heating and cooling"),
    "com_wh": ("Commercial", "water heating"),
    "trans_ldv": ("Transportation", "light-duty vehicles"),
    "trans_mdv": ("Transportation", "medium-duty trucks"),
    "trans_hdv": ("Transportation", "heavy-duty trucks"),
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
) -> pd.DataFrame:
    load_dtypes = {
        "Year": "category",
        "LocalHourID": "category",
        "Sector": "category",
        "Subsector": "category",
        "LoadMW": np.float32,
    }
    pop = load_region_pop_frac()
    model_states = pop.loc[pop["ipm_region"].isin(regions), "state"]
    EFS_2020_LoadProf = pd.read_parquet(path_in / "EFS_REF_load_2020.parquet")
    EFS_2020_LoadProf = EFS_2020_LoadProf.astype(load_dtypes)
    EFS_2020_LoadProf.columns = snake_case_col(EFS_2020_LoadProf.columns)
    total_efs_2020_load = EFS_2020_LoadProf["loadmw"].sum()

    EFS_2020_LoadProf = EFS_2020_LoadProf.loc[
        EFS_2020_LoadProf["state"].isin(model_states), :
    ]
    EFS_2020_LoadProf = pd.merge(EFS_2020_LoadProf, pop, on=["state"])
    EFS_2020_LoadProf = EFS_2020_LoadProf.assign(
        weighted=EFS_2020_LoadProf["loadmw"] * EFS_2020_LoadProf["state_prop"]
    )
    EFS_2020_LoadProf = EFS_2020_LoadProf.groupby(
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
    EFS_2020_LoadProf["weighted"] *= ratio_A

    Base_Load_2019 = EFS_2020_LoadProf.rename(columns={"weighted": "loadmw"})

    # Create Base loads
    Base_Load_2019 = Base_Load_2019.loc[Base_Load_2019["ipm_region"].isin(regions), :]
    Base_Load_2019.loc[
        (Base_Load_2019["sector"] == "Industrial")
        & (Base_Load_2019["subsector"].isin(["process heat", "machine drives"])),
        "subsector",
    ] = "other"
    Base_Load_2019 = Base_Load_2019.loc[Base_Load_2019["subsector"] == "other", :]
    Base_Load_2019 = Base_Load_2019.groupby(
        ["year", "localhourid", "ipm_region", "sector"], as_index=False
    ).agg({"loadmw": "sum"})
    Base_Load = Base_Load_2019
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
        base_load_temp = Base_Load_2019.copy()
        base_load_temp["year"] = y
        base_load_temp["loadmw"] *= base_load_temp["ipm_region"].map(growth_factor)
        Base_Load = Base_Load.append(base_load_temp, ignore_index=True)
    return Base_Load


def create_subsector_ts(
    sector: str, subsector: str, year: int, scenario_stock: pd.DataFrame
) -> pd.DataFrame:
    ts_cols = ["State", "Year", "LocalHourID", "Unit", "Factor_Type1", "Factor_Type2"]
    timeseries = pd.read_parquet(
        path_in / f"{sector}_{subsector}_Incremental_Factor.parquet", columns=ts_cols
    )
    timeseries.columns = snake_case_col(timeseries.columns)
    stock_temp = scenario_stock[
        (scenario_stock["sector"] == sector)
        & (scenario_stock["subsector"] == subsector)
    ]
    stock_temp = stock_temp[
        ["scenario", "state", "year", "agg_stock_type1", "agg_stock_type2"]
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

    timeseries = pd.merge(timeseries, stock_temp, on=["state", "year"])
    timeseries = timeseries.assign(
        loadmw=timeseries["agg_stock_type1"] * timeseries["factor_type1"]
        + timeseries["agg_stock_type2"] * timeseries["factor_type2"]
    )
    return timeseries[["scenario", "state", "year", "localhourid", "loadmw"]].dropna()


def state_demand_to_region(
    df: pd.DataFrame, pop: pd.DataFrame, col_name: str
) -> pd.DataFrame:
    temp = pd.merge(df, pop, on=["state"], how="left")
    temp = (
        temp.assign(weighted=temp["loadmw"] * temp["state_prop"])
        .groupby(["scenario", "year", "localhourid", "ipm_region"], as_index=False)[
            "weighted"
        ]
        .sum()
        .rename(columns={"weighted": col_name + "_MW"})
    )
    return temp


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
) -> pd.DataFrame:
    try:
        if (output_folder / "load_by_region_sector.parquet").exists():
            return pd.read_parquet(output_folder / "load_by_region_sector.parquet")
    except:
        pass
    # Creating Time-series
    pop = load_region_pop_frac()
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
            sector, subsector, year, scenario_stock
        )

    # Res_SPH = pd.read_parquet(
    #     path_result
    #     / f"Residential_space heating and cooling_Scenario_Timeseries.parquet"
    # )
    # Res_SPH = Res_SPH.rename(columns={"loadmw": "Res_SPH_LoadMW"})
    # Res_SPH_sum = Res_SPH
    # Res_SPH_sum = Res_SPH.groupby(["SCENARIO", "state", "year"], as_index=False)[
    #     "Res_SPH_LoadMW"
    # ].agg({"Total_Res_SPH_TWh": "sum"})
    # Res_SPH_sum["Total_Res_SPH_TWh"] = 10 ** -6 * Res_SPH_sum["Total_Res_SPH_TWh"]

    # Res_WH = pd.read_parquet(
    #     path_result / f"Residential_water heating_Scenario_Timeseries.parquet"
    # )
    # Res_WH = Res_WH.rename(columns={"loadmw": "Res_WH_LoadMW"})
    # Res_WH_sum = Res_WH
    # Res_WH_sum = Res_WH.groupby(["SCENARIO", "state", "year"], as_index=False)[
    #     "Res_WH_LoadMW"
    # ].agg({"Total_Res_WH_TWh": "sum"})
    # Res_WH_sum["Total_Res_WH_TWh"] = 10 ** -6 * Res_WH_sum["Total_Res_WH_TWh"]

    # Com_SPH = pd.read_parquet(
    #     path_result
    #     / f"Commercial_space heating and cooling_Scenario_Timeseries.parquet"
    # )
    # Com_SPH = Com_SPH.rename(columns={"loadmw": "Com_SPH_LoadMW"})
    # Com_SPH_sum = Com_SPH
    # Com_SPH_sum = Com_SPH.groupby(["SCENARIO", "state", "year"], as_index=False)[
    #     "Com_SPH_LoadMW"
    # ].agg({"Total_Com_SPH_TWh": "sum"})
    # Com_SPH_sum["Total_Com_SPH_TWh"] = 10 ** -6 * Com_SPH_sum["Total_Com_SPH_TWh"]

    # Com_WH = pd.read_parquet(
    #     path_result / f"Commercial_water heating_Scenario_Timeseries.parquet"
    # )
    # Com_WH = Com_WH.rename(columns={"loadmw": "Com_WH_LoadMW"})
    # Com_WH_sum = Com_WH
    # Com_WH_sum = Com_WH.groupby(["SCENARIO", "state", "year"], as_index=False)[
    #     "Com_WH_LoadMW"
    # ].agg({"Total_Com_WH_TWh": "sum"})
    # Com_WH_sum["Total_Com_WH_TWh"] = 10 ** -6 * Com_WH_sum["Total_Com_WH_TWh"]

    # Trans_LDV = pd.read_parquet(
    #     path_result / f"Transportation_light-duty vehicles_Scenario_Timeseries.parquet"
    # )
    # Trans_LDV = Trans_LDV.rename(columns={"loadmw": "LDV_LoadMW"})
    # Trans_LDV_sum = Trans_LDV
    # Trans_LDV_sum = Trans_LDV.groupby(["SCENARIO", "state", "year"], as_index=False)[
    #     "LDV_LoadMW"
    # ].agg({"Total_Trans_LDV_TWh": "sum"})
    # Trans_LDV_sum["Total_Trans_LDV_TWh"] = (
    #     10 ** -6 * Trans_LDV_sum["Total_Trans_LDV_TWh"]
    # )

    # Trans_MDV = pd.read_parquet(
    #     path_result / f"Transportation_medium-duty trucks_Scenario_Timeseries.parquet"
    # )
    # Trans_MDV = Trans_MDV.rename(columns={"loadmw": "MDV_LoadMW"})
    # Trans_MDV_sum = Trans_MDV
    # Trans_MDV_sum = Trans_MDV.groupby(["SCENARIO", "state", "year"], as_index=False)[
    #     "MDV_LoadMW"
    # ].agg({"Total_Trans_MDV_TWh": "sum"})
    # Trans_MDV_sum["Total_Trans_MDV_TWh"] = (
    #     10 ** -6 * Trans_MDV_sum["Total_Trans_MDV_TWh"]
    # )

    # Trans_HDV = pd.read_parquet(
    #     path_result / f"Transportation_heavy-duty trucks_Scenario_Timeseries.parquet"
    # )
    # Trans_HDV = Trans_HDV.rename(columns={"loadmw": "HDV_LoadMW"})
    # Trans_HDV_sum = Trans_HDV
    # Trans_HDV_sum = Trans_HDV.groupby(["SCENARIO", "state", "year"], as_index=False)[
    #     "HDV_LoadMW"
    # ].agg({"Total_Trans_HDV_TWh": "sum"})
    # Trans_HDV_sum["Total_Trans_HDV_TWh"] = (
    #     10 ** -6 * Trans_HDV_sum["Total_Trans_HDV_TWh"]
    # )

    # Trans_BUS = pd.read_parquet(
    #     path_result / f"Transportation_transit buses_Scenario_Timeseries.parquet"
    # )
    # Trans_BUS = Trans_BUS.rename(columns={"loadmw": "BUS_LoadMW"})
    # Trans_BUS_sum = Trans_BUS
    # Trans_BUS_sum = Trans_BUS.groupby(["SCENARIO", "state", "year"], as_index=False)[
    #     "BUS_LoadMW"
    # ].agg({"Total_Trans_BUS_TWh": "sum"})
    # Trans_BUS_sum["Total_Trans_BUS_TWh"] = (
    #     10 ** -6 * Trans_BUS_sum["Total_Trans_BUS_TWh"]
    # )

    # del (
    #     Res_SPH_sum,
    #     Res_WH_sum,
    #     Com_SPH_sum,
    #     Trans_LDV_sum,
    #     Trans_MDV_sum,
    #     Trans_HDV_sum,
    #     Trans_BUS_sum,
    # )

    # ################
    # # Distribute Load to GenX.Region
    # print("Distribute Load to GenX.Region")
    # subsectors = [
    #     Res_SPH,
    #     Res_WH,
    #     Com_SPH,
    #     Com_WH,
    #     Trans_LDV,
    #     Trans_MDV,
    #     Trans_HDV,
    #     Trans_BUS,
    # ]
    # subsector_names = [
    #     "Res_SPH",
    #     "Res_WH",
    #     "Com_SPH",
    #     "Com_WH",
    #     "Trans_LDV",
    #     "Trans_MDV",
    #     "Trans_HDV",
    #     "Trans_BUS",
    # ]
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
    )

    if output_folder and not (output_folder / "load_by_region_sector.parquet").exists():
        total_load.to_parquet(
            output_folder / "load_by_region_sector.parquet", index=False
        )
    return total_load


def MakeLoadProfiles(settings: dict, case_folder: Path) -> pd.DataFrame:
    output_folder = case_folder / "extra_outputs"
    output_folder.mkdir(exist_ok=True)

    years = []
    regions = []
    electrification = []
    for year in settings:
        for cases, _settings in settings[year].items():
            years.append(_settings["model_year"])
            regions = (
                regions
                + regions_to_keep(
                    _settings["model_regions"], settings.get("region_aggregations")
                )[0]
            )
            electrification.append(_settings["NZA_electrification"])

            if _settings.get("custom_stock"):
                path_stock = _settings["input_folder"] / _settings["custom_stock"]
            else:
                path_stock = None
            if _settings.get("custom_growthrate"):
                path_growthrate = (
                    _settings["input_folder"] / _settings["custom_growthrate"]
                )
            else:
                path_growthrate = None
            # try:
            #     path_stock = (
            #         str(_settings["input_folder"]) + "\\" + _settings["custom_stock"]
            #     )
            #     path_growthrate = (
            #         str(_settings["input_folder"])
            #         + "\\"
            #         + _settings["custom_growthrate"]
            #     )
            # except:
            #     path_stock = ""
            #     path_growthrate = ""
            # scenarios
    # years = list(set(years))
    regions = list(set(regions))
    # electrification = list(set(electrification))
    # CreateBaseLoad(years, regions, output_folder, path_growthrate)
    return AddElectrification(
        settings, years, regions, electrification, output_folder, path_stock
    )


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
