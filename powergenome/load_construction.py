#######################
# Header.R

from datetime import time
from operator import index
from os import path, times
from typing import Dict, List
import numpy as np
import pandas as pd
import os
import logging

from pathlib import Path
from joblib import Memory

from pandas.core.reshape.merge import merge
from powergenome.load_profiles import calc_growth_factors, make_final_load_curves
from powergenome.util import init_pudl_connection, regions_to_keep, snake_case_col
from powergenome.us_state_abbrev import state2abbr, abbr2state
from powergenome.params import DATA_PATHS, SETTINGS

path_in = Path(SETTINGS["EFS_DATA"])

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
    # read in state proportions
    # how much state load should be distributed to GenXRegion
    # pop = pd.read_parquet(path_in + "\GenX_State_Pop_Weight.parquet")
    pop_cols = ["ipm_region", "state", "state_prop"]
    # pop_dtypes = {"State Prop": np.float32}
    pop = pd.read_parquet(path_in / fn, columns=pop_cols)
    pop["state"] = pop["state"].map(us_state_abbrev)
    return pop


# pop = pop.astype(pop_dtypes)
# states = pop.drop_duplicates(subset=["state"])["state"]
# states_abb = list(map(state2abbr, states))
# pop["state"] = list(map(state2abbr, pop["state"]))
# states_eastern_abbr = [
#     "ME",
#     "VT",
#     "NH",
#     "MA",
#     "RI",
#     "CT",
#     "NY",
#     "PA",
#     "NJ",
#     "DE",
#     "MD",
#     "DC",
#     "MI",
#     "IN",
#     "OH",
#     "KY",
#     "WV",
#     "VA",
#     "NC",
#     "SC",
#     "GA",
#     "FL",
# ]
# states_central_abbr = [
#     "IL",
#     "MO",
#     "TN",
#     "AL",
#     "MS",
#     "WI",
#     "AR",
#     "LA",
#     "TX",
#     "OK",
#     "KS",
#     "NE",
#     "SD",
#     "ND",
#     "IA",
#     "MN",
# ]
# states_mountain_abbr = ["MT", "WY", "CO", "NM", "AZ", "UT", "ID"]
# states_pacific_abbr = ["CA", "NV", "OR", "WA"]
# states_eastern = list(map(abbr2state, states_eastern_abbr))
# states_central = list(map(abbr2state, states_central_abbr))
# states_mountain = list(map(abbr2state, states_mountain_abbr))
# states_pacific = list(map(abbr2state, states_pacific_abbr))

# # some parameters
# stated_states = ["New Jersey", "New York", "Virginia"]
# # Date Jan 29, 2021
# # (2) PA, NJ, VA, NY, MI all have EV and heat pump stocks from NZA DD case
# # consistent with their economywide decarbonization goals.
# # https://www.c2es.org/content/state-climate-policy/
# # Date Feb 10, 2021
# # Remove high electrification growth in PA and MI in stated policies;
# # they dont have clean energy goals so kind of confusing/inconsistent to require high electrification in these states.
# # So our new "Stated Policies" definition for electrification is states
# # with BOTH economywide emissions goals + 100% carbon-free electricity standards
# # = NY, NJ, VA.
# stated_states_abbr = list(map(state2abbr, stated_states))
# # years = ["2022", "2025", "2030", "2040", "2050"]
# cases = ["current_policy", "stated_policy", "deep_decarbonization"]

running_sector = [
    "Residential",
    "Residential",
    "Commercial",
    "Commercial",
    "Transportation",
    "Transportation",
    "Transportation",
    "Transportation",
]
running_subsector = [
    "space heating and cooling",
    "water heating",
    "space heating and cooling",
    "water heating",
    "light-duty vehicles",
    "medium-duty trucks",
    "heavy-duty trucks",
    "transit buses",
]
Nsubsector = len(running_subsector)

logger = logging.getLogger(__name__)


def CreateOutputFolder(case_folder):
    path = case_folder / "extra_outputs"
    if not os.path.exists(path):
        os.makedirs(path)


######################################
# CreatingBaseLoad.R
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
    ## Method 3: annually
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
    # EFS_2020_LoadProf = EFS_2020_LoadProf.astype(load_dtypes)
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
        "historical_load_region_maps": historical_load_region_map,
        "future_load_region_map": future_load_region_map,
        "model_year": 2019,
        "model_regions": list(pop["ipm_region"].unique()),
        "utc_offset": -5,
    }
    pudl_engine, pudl_out = init_pudl_connection()
    original_load_2019 = make_final_load_curves(pudl_engine, demand_settings)
    # original_load_2019 = pd.read_parquet(path_in / "ipm_load_curves_2019_EST.parquet")
    # original_load_2019.columns = snake_case_col(original_load_2019.columns)
    original_load_2019 = original_load_2019.reset_index().rename(
        columns={"time_index": "localhourid"}
    )
    # original_load_2019 = original_load_2019.rename(columns={"ipm_region": "ipm_region"})
    # Reorganize Demand
    original_load_2019 = original_load_2019.melt(id_vars="localhourid").rename(
        columns={"variable": "ipm_region", "value": "loadmw_original"}
    )
    # Original_Load_2019 = Original_Load_2019.groupby(
    #     ["LocalHourID"], as_index=False
    # ).agg({"LoadMW_original": "sum"})

    ratio_A = (
        original_load_2019["loadmw_original"].sum()
        # / EFS_2020_LoadProf["weighted"].sum()
        / total_efs_2020_load
    )
    EFS_2020_LoadProf["weighted"] *= ratio_A

    Base_Load_2019 = EFS_2020_LoadProf.rename(columns={"weighted": "loadmw"})
    # breakpoint()
    # Read in the Growth Rate
    # if path_growthrate:
    #     GrowthRate = pd.read_parquet(path_growthrate)
    # else:
    #     GrowthRate = pd.read_parquet(path_in / "ipm_growthrate_2019.parquet")

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
        growth_factor = calc_growth_factors(
            regions,
            load_region_map=future_load_region_map,
            growth_scenario=growth_scenario,
            eia_aeo_year=eia_aeo_year,
            start_year=regular_load_growth_start_year,
            end_year=y,
            alt_growth_rate=alt_growth_rate,
        )
        # ScaleFactor = growth_factor.assign(
        #     ScaleFactor=(1 + GrowthRate["growth_rate"]) ** (int(y) - 2019)
        # ).drop(columns="growth_rate")
        # Base_Load_temp = pd.merge(Base_Load_2019, ScaleFactor, on=["ipm_region"])
        base_load_temp = Base_Load_2019.copy()
        base_load_temp["year"] = y
        base_load_temp["loadmw"] *= base_load_temp["ipm_region"].map(growth_factor)
        # base_load_temp = base_load_temp.assign(
        #     Year=y, loadmw=base_load_temp["loadmw"] * base_load_temp["ScaleFactor"]
        # ).drop(columns="ScaleFactor")
        Base_Load = Base_Load.append(base_load_temp, ignore_index=True)
    # Base_Load.to_parquet(path_result / "Base_Load.parquet", index=False)
    return Base_Load
    # del (
    #     Base_Load,
    #     Base_Load_2019,
    #     Base_Load_temp,
    #     ScaleFactor,
    #     GrowthRate,
    #     Original_Load_2019,
    # )


#####################################
# Add_Electrification.R
def AddElectrification(
    settings: dict,
    years: List[int],
    regions: List[str],
    electrification: List[str],
    output_folder: Path,
    path_stock: Path = None,
) -> pd.DataFrame:
    path_processed = path_in
    path_result = output_folder
    # Creating Time-series
    pop = load_region_pop_frac()
    states = pop.loc[pop["ipm_region"].isin(regions), "state"].unique()
    SCENARIO_STOCK = pd.read_parquet(path_processed / "SCENARIO_STOCK.parquet")
    SCENARIO_STOCK = SCENARIO_STOCK[
        (SCENARIO_STOCK["YEAR"].isin(years))
        & (SCENARIO_STOCK["SCENARIO"].isin(electrification))
        & (SCENARIO_STOCK["STATE"].isin(states))
    ]
    SCENARIO_STOCK_temp = pd.DataFrame()
    for year, case in zip(years, electrification):
        SCENARIO_STOCK_temp = SCENARIO_STOCK_temp.append(
            SCENARIO_STOCK[
                (SCENARIO_STOCK["YEAR"] == year) & (SCENARIO_STOCK["SCENARIO"] == case)
            ]
        )
    SCENARIO_STOCK = SCENARIO_STOCK_temp
    del SCENARIO_STOCK_temp

    if path_stock:
        CUSTOM_STOCK = pd.read_parquet(path_stock)
        CUSTOM_STOCK = CUSTOM_STOCK[
            (CUSTOM_STOCK["YEAR"].isin(years))
            & (CUSTOM_STOCK["SCENARIO"].isin(electrification))
        ]
        SCENARIO_STOCK = SCENARIO_STOCK.append(CUSTOM_STOCK)

    # Method 1 Calculate from Type1 and Type 2
    for i in range(0, Nsubsector):
        ts_cols = [
            "State",
            "Year",
            "LocalHourID",
            "Unit",
            "Factor_Type1",
            "Factor_Type2",
        ]
        timeseries = pd.read_parquet(
            path_processed
            / f"{running_sector[i]}_{running_subsector[i]}_Incremental_Factor.parquet",
            columns=ts_cols,
        )
        timeseries = timeseries.rename(
            columns={
                "State": "state",
                "Year": "year",
                "LocalHourID": "localhourid",
                "Factor_Type1": "factor_type1",
                "Factor_Type2": "factor_type2",
            }
        )
        # timeseries = timeseries[
        #     ["state", "year", "LocalHourID", "Unit", "factor_type1", "factor_type2"]
        # ]
        stock_temp = SCENARIO_STOCK[
            (SCENARIO_STOCK["SECTOR"] == running_sector[i])
            & (SCENARIO_STOCK["SUBSECTOR"] == running_subsector[i])
        ]
        stock_temp = stock_temp[
            ["SCENARIO", "STATE", "YEAR", "AGG_STOCK_TYPE1", "AGG_STOCK_TYPE2"]
        ].rename(columns={"STATE": "state", "YEAR": "year"})
        years_pd = pd.Series(years)
        IF_years = pd.Series(timeseries["year"].unique())
        for year in years_pd:
            exists = year in IF_years.values
            if not exists:
                diff = np.array(IF_years - year)
                index = diff[np.where(diff <= 0)].argmax()
                year_approx = IF_years[index]
                timeseries_temp = timeseries[timeseries["year"] == year_approx]
                timeseries_temp["year"] = year
                logger.warning(
                    "No incremental factor available for year "
                    + str(year)
                    + ": using factors from year "
                    + str(year_approx)
                    + "."
                )
                timeseries = timeseries.append(timeseries_temp)

        timeseries = pd.merge(timeseries, stock_temp, on=["state", "year"])
        timeseries = timeseries.assign(
            loadmw=timeseries["AGG_STOCK_TYPE1"] * timeseries["factor_type1"]
            + timeseries["AGG_STOCK_TYPE2"] * timeseries["factor_type2"]
        )
        timeseries = timeseries[
            ["SCENARIO", "state", "year", "localhourid", "loadmw"]
        ].dropna()
        timeseries.to_parquet(
            path_result
            / f"{running_sector[i]}_{running_subsector[i]}_Scenario_Timeseries_Method1.parquet",
            index=False,
        )
    # del timeseries, stock_temp

    ##########################
    # Read in time series and combine them
    print("Read in time series and combine them")
    Method = "Method1"
    Res_SPH = pd.read_parquet(
        path_result
        / f"Residential_space heating and cooling_Scenario_Timeseries_{Method}.parquet"
    )
    Res_SPH = Res_SPH.rename(columns={"loadmw": "Res_SPH_LoadMW"})
    Res_SPH_sum = Res_SPH
    Res_SPH_sum = Res_SPH.groupby(["SCENARIO", "state", "year"], as_index=False)[
        "Res_SPH_LoadMW"
    ].agg({"Total_Res_SPH_TWh": "sum"})
    Res_SPH_sum["Total_Res_SPH_TWh"] = 10 ** -6 * Res_SPH_sum["Total_Res_SPH_TWh"]

    Res_WH = pd.read_parquet(
        path_result / f"Residential_water heating_Scenario_Timeseries_{Method}.parquet"
    )
    Res_WH = Res_WH.rename(columns={"loadmw": "Res_WH_LoadMW"})
    Res_WH_sum = Res_WH
    Res_WH_sum = Res_WH.groupby(["SCENARIO", "state", "year"], as_index=False)[
        "Res_WH_LoadMW"
    ].agg({"Total_Res_WH_TWh": "sum"})
    Res_WH_sum["Total_Res_WH_TWh"] = 10 ** -6 * Res_WH_sum["Total_Res_WH_TWh"]

    Com_SPH = pd.read_parquet(
        path_result
        / f"Commercial_space heating and cooling_Scenario_Timeseries_{Method}.parquet"
    )
    Com_SPH = Com_SPH.rename(columns={"loadmw": "Com_SPH_LoadMW"})
    Com_SPH_sum = Com_SPH
    Com_SPH_sum = Com_SPH.groupby(["SCENARIO", "state", "year"], as_index=False)[
        "Com_SPH_LoadMW"
    ].agg({"Total_Com_SPH_TWh": "sum"})
    Com_SPH_sum["Total_Com_SPH_TWh"] = 10 ** -6 * Com_SPH_sum["Total_Com_SPH_TWh"]

    Com_WH = pd.read_parquet(
        path_result / f"Commercial_water heating_Scenario_Timeseries_{Method}.parquet"
    )
    Com_WH = Com_WH.rename(columns={"loadmw": "Com_WH_LoadMW"})
    Com_WH_sum = Com_WH
    Com_WH_sum = Com_WH.groupby(["SCENARIO", "state", "year"], as_index=False)[
        "Com_WH_LoadMW"
    ].agg({"Total_Com_WH_TWh": "sum"})
    Com_WH_sum["Total_Com_WH_TWh"] = 10 ** -6 * Com_WH_sum["Total_Com_WH_TWh"]

    Trans_LDV = pd.read_parquet(
        path_result
        / f"Transportation_light-duty vehicles_Scenario_Timeseries_{Method}.parquet"
    )
    Trans_LDV = Trans_LDV.rename(columns={"loadmw": "LDV_LoadMW"})
    Trans_LDV_sum = Trans_LDV
    Trans_LDV_sum = Trans_LDV.groupby(["SCENARIO", "state", "year"], as_index=False)[
        "LDV_LoadMW"
    ].agg({"Total_Trans_LDV_TWh": "sum"})
    Trans_LDV_sum["Total_Trans_LDV_TWh"] = (
        10 ** -6 * Trans_LDV_sum["Total_Trans_LDV_TWh"]
    )

    Trans_MDV = pd.read_parquet(
        path_result
        / f"Transportation_medium-duty trucks_Scenario_Timeseries_{Method}.parquet"
    )
    Trans_MDV = Trans_MDV.rename(columns={"loadmw": "MDV_LoadMW"})
    Trans_MDV_sum = Trans_MDV
    Trans_MDV_sum = Trans_MDV.groupby(["SCENARIO", "state", "year"], as_index=False)[
        "MDV_LoadMW"
    ].agg({"Total_Trans_MDV_TWh": "sum"})
    Trans_MDV_sum["Total_Trans_MDV_TWh"] = (
        10 ** -6 * Trans_MDV_sum["Total_Trans_MDV_TWh"]
    )

    Trans_HDV = pd.read_parquet(
        path_result
        / f"Transportation_heavy-duty trucks_Scenario_Timeseries_{Method}.parquet"
    )
    Trans_HDV = Trans_HDV.rename(columns={"loadmw": "HDV_LoadMW"})
    Trans_HDV_sum = Trans_HDV
    Trans_HDV_sum = Trans_HDV.groupby(["SCENARIO", "state", "year"], as_index=False)[
        "HDV_LoadMW"
    ].agg({"Total_Trans_HDV_TWh": "sum"})
    Trans_HDV_sum["Total_Trans_HDV_TWh"] = (
        10 ** -6 * Trans_HDV_sum["Total_Trans_HDV_TWh"]
    )

    Trans_BUS = pd.read_parquet(
        path_result
        / f"Transportation_transit buses_Scenario_Timeseries_{Method}.parquet"
    )
    Trans_BUS = Trans_BUS.rename(columns={"loadmw": "BUS_LoadMW"})
    Trans_BUS_sum = Trans_BUS
    Trans_BUS_sum = Trans_BUS.groupby(["SCENARIO", "state", "year"], as_index=False)[
        "BUS_LoadMW"
    ].agg({"Total_Trans_BUS_TWh": "sum"})
    Trans_BUS_sum["Total_Trans_BUS_TWh"] = (
        10 ** -6 * Trans_BUS_sum["Total_Trans_BUS_TWh"]
    )

    del (
        Res_SPH_sum,
        Res_WH_sum,
        Com_SPH_sum,
        Trans_LDV_sum,
        Trans_MDV_sum,
        Trans_HDV_sum,
        Trans_BUS_sum,
    )

    ################
    # Distribute Load to GenX.Region
    print("Distribute Load to GenX.Region")
    Method = "Method1"
    subsectors = [
        Res_SPH,
        Res_WH,
        Com_SPH,
        Com_WH,
        Trans_LDV,
        Trans_MDV,
        Trans_HDV,
        Trans_BUS,
    ]
    subsector_names = [
        "Res_SPH",
        "Res_WH",
        "Com_SPH",
        "Com_WH",
        "Trans_LDV",
        "Trans_MDV",
        "Trans_HDV",
        "Trans_BUS",
    ]
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

    for i, sc in enumerate(subsectors):
        temp = pd.merge(sc, pop, on=["state"], how="left")
        temp = (
            temp.assign(
                weighted=temp[column_names[i] + "_" + "LoadMW"] * temp["state_prop"]
            )
            .groupby(["SCENARIO", "year", "localhourid", "ipm_region"], as_index=False)[
                "weighted"
            ]
            .sum()
            .rename(columns={"weighted": column_names[i] + "_MW"})
        )
        temp.to_parquet(
            path_result / f"{subsector_names[i]}_By_region.parquet", index=False
        )
    del (
        temp,
        subsectors,
        Res_SPH,
        Res_WH,
        Com_SPH,
        Com_WH,
        Trans_LDV,
        Trans_MDV,
        Trans_HDV,
        Trans_BUS,
    )

    ######
    # Construct Total Load
    print("Construct Total Load")

    # Base_Load = pd.read_parquet(path_result / "Base_Load.parquet")
    df_list = []
    for year, scenario_settings in settings.items():
        for scenario, _settings in scenario_settings.items():
            keep_regions, region_agg_map = regions_to_keep(
                _settings["model_regions"],
                _settings.get("region_aggregations", {}) or {},
            )
            for r in keep_regions:

                _df = CreateBaseLoad(
                    [year],
                    [r],
                    _settings["future_load_region_map"],
                    _settings["growth_scenario"],
                    _settings["eia_aeo_year"],
                    _settings.get("regular_load_growth_start_year", 2019),
                    _settings.get("alt_growth_rate", {}) or {},
                )
                df_list.append(_df)
    print("individual base loads constructed")
    Base_Load = pd.concat(df_list, ignore_index=True)
    print("base load concat complete")
    Base_Load = Base_Load.rename(columns={"loadmw": "base_MW"})

    Base_Load.loc[(Base_Load["sector"] == "Commercial"), "Subsector"] = "Base_Com_other"
    Base_Load.loc[
        (Base_Load["sector"] == "Residential"), "Subsector"
    ] = "Base_Res_other"
    Base_Load.loc[
        (Base_Load["sector"] == "Transportation"), "Subsector"
    ] = "Base_Trans_other"
    Base_Load.loc[(Base_Load["sector"] == "Industrial"), "Subsector"] = "Base_Ind"
    # return Base_Load
    Base_Load = Base_Load.drop(columns=["sector"])
    # Base_Load = Base_Load.astype(
    #     {"year": "category", "ipm_region": "category", "Subsector": "category"}
    # )
    # The df index used to be part of the index parameter below. Removed because I think it was uncessary.
    Base_Load = (
        Base_Load.pivot_table(
            index=["localhourid", "ipm_region", "year"],
            columns="Subsector",
            values="base_MW",
        )
        # .reset_index(["localhourid", "ipm_region", "year"])
        .reset_index()  # .fillna(0)
    )
    Base_Load = Base_Load.astype({"year": "category", "ipm_region": "category"})
    print("grouping base load")
    Base_Load = Base_Load.groupby(
        ["localhourid", "ipm_region", "year"], as_index=False
    ).agg(
        {
            "Base_Com_other": "sum",
            "Base_Res_other": "sum",
            "Base_Trans_other": "sum",
            "Base_Ind": "sum",
        }
    )
    print("reading by region files")
    Res_WH = pd.read_parquet(path_result / "Res_WH_By_region.parquet")
    Com_WH = pd.read_parquet(path_result / "Com_WH_By_region.parquet")
    Res_SPH = pd.read_parquet(path_result / "Res_SPH_By_region.parquet")
    Com_SPH = pd.read_parquet(path_result / "Com_SPH_By_region.parquet")
    Trans_LDV = pd.read_parquet(path_result / "Trans_LDV_By_region.parquet")
    Trans_MDV = pd.read_parquet(path_result / "Trans_MDV_By_region.parquet")
    Trans_HDV = pd.read_parquet(path_result / "Trans_HDV_By_region.parquet")
    # Trans_BUS = pd.read_parquet(path_result + "\Trans_BUS_By_region.parquet")
    print("merging by region dataframes")
    Total_Load = pd.merge(
        Res_WH, Com_WH, on=["SCENARIO", "year", "localhourid", "ipm_region"]
    )
    Total_Load = pd.merge(
        Total_Load, Res_SPH, on=["SCENARIO", "year", "localhourid", "ipm_region"]
    )
    Total_Load = pd.merge(
        Total_Load, Com_SPH, on=["SCENARIO", "year", "localhourid", "ipm_region"]
    )
    Total_Load = pd.merge(
        Total_Load, Trans_LDV, on=["SCENARIO", "year", "localhourid", "ipm_region"]
    )
    Total_Load = pd.merge(
        Total_Load, Trans_MDV, on=["SCENARIO", "year", "localhourid", "ipm_region"]
    )
    Total_Load = pd.merge(
        Total_Load, Trans_HDV, on=["SCENARIO", "year", "localhourid", "ipm_region"]
    )
    # Total_Load = pd.merge(Total_Load, Trans_BUS, on = ["SCENARIO","year","LocalHourID","ipm_region"])
    Total_Load = pd.merge(
        Total_Load, Base_Load, on=["year", "localhourid", "ipm_region"]
    )
    Total_Load = Total_Load[Total_Load["year"].isin(years)]
    del Base_Load, Res_WH, Com_WH, Res_SPH, Com_SPH, Trans_LDV, Trans_MDV, Trans_HDV

    Total_Load = Total_Load[
        (Total_Load["year"].isin(years)) & (Total_Load["ipm_region"].isin(regions))
    ]
    Total_Load = Total_Load.astype(
        {
            "SCENARIO": "category",
            "year": "category",
            "localhourid": "category",
            "ipm_region": "category",
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
    Total_Load.to_parquet(
        path_result / "Total_load_by_region_full.parquet", index=False
    )

    return Total_Load


def MakeLoadProfiles(settings: dict, case_folder: Path) -> pd.DataFrame:
    # path_processed = r"C:\Users\ritib\Dropbox\Project_LoadConstruction\data\processed"
    # path_result = r"C:\Users\ritib\Dropbox\Project_LoadConstruction\data\result"
    # CreateOutputFolder(case_folder)

    output_folder = case_folder / "extra_outputs"
    output_folder.mkdir(exist_ok=True)

    years = []
    regions = []
    electrification = []
    # scenarios = pd.DataFrame()
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


def FilterTotalProfile(settings, total_load):
    TotalLoad = total_load
    settings = settings
    TotalLoad = TotalLoad[TotalLoad["year"] == settings["model_year"]]
    TotalLoad = TotalLoad.assign(
        TotalMW=TotalLoad["Res_WH_MW"]
        + TotalLoad["Com_WH_MW"]
        + TotalLoad["Res_SPH_MW"]
        + TotalLoad["Com_SPH_MW"]
        + TotalLoad["LDV_MW"]
        + TotalLoad["MDV_MW"]
        + TotalLoad["HDV_MW"]
        + TotalLoad["Base_Com_other"]
        + TotalLoad["Base_Res_other"]
        + TotalLoad["Base_Trans_other"]
        + TotalLoad["Base_Ind"]
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
        ]
    )
    TotalLoad = TotalLoad[TotalLoad["SCENARIO"] == settings["NZA_electrification"]]
    TotalLoad = TotalLoad.drop(columns=["SCENARIO", "year"]).rename(
        columns={"localhourid": "time_index", "ipm_region": "region"}
    )

    keep_regions, region_agg_map = regions_to_keep(settings)

    TotalLoad.loc[:, "region_id_epaipm"] = TotalLoad.loc[:, "region"]
    TotalLoad.loc[
        TotalLoad.region_id_epaipm.isin(region_agg_map.keys()), "region"
    ] = TotalLoad.loc[
        TotalLoad.region_id_epaipm.isin(region_agg_map.keys()), "region_id_epaipm"
    ].map(
        region_agg_map
    )

    logger.info("Aggregating load curves in grouped regions")
    TotalLoad = TotalLoad.groupby(["region", "time_index"]).sum()

    TotalLoad = TotalLoad.unstack(level=0)
    TotalLoad.columns = TotalLoad.columns.droplevel()
    # TotalLoad = TotalLoad.pivot_table(index = 'time_index', columns = 'region', values = 'TotalMW')
    return TotalLoad
