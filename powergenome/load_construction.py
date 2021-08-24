#######################
# Header.R

from datetime import time
from operator import index
from os import path, times
import numpy as np
import pandas as pd
import os
import logging

from pathlib import Path

from pandas.core.reshape.merge import merge
from powergenome.util import regions_to_keep
from powergenome.us_state_abbrev import state2abbr, abbr2state

path_in = Path(
    "/Volumes/Extreme SSD/load_profiles_data/input"
)  # r"..\data\load_profiles_data\input"  # fix

# read in state proportions
# how much state load should be distributed to GenXRegion
# pop = pd.read_parquet(path_in + "\GenX_State_Pop_Weight.parquet")
pop = pd.read_parquet(path_in + "\ipm_state_pop_weight_20210517.parquet")
states = pop.drop_duplicates(subset=["State"])["State"]
states_abb = list(map(state2abbr, states))
pop["State"] = list(map(state2abbr, pop["State"]))
states_eastern_abbr = [
    "ME",
    "VT",
    "NH",
    "MA",
    "RI",
    "CT",
    "NY",
    "PA",
    "NJ",
    "DE",
    "MD",
    "DC",
    "MI",
    "IN",
    "OH",
    "KY",
    "WV",
    "VA",
    "NC",
    "SC",
    "GA",
    "FL",
]
states_central_abbr = [
    "IL",
    "MO",
    "TN",
    "AL",
    "MS",
    "WI",
    "AR",
    "LA",
    "TX",
    "OK",
    "KS",
    "NE",
    "SD",
    "ND",
    "IA",
    "MN",
]
states_mountain_abbr = ["MT", "WY", "CO", "NM", "AZ", "UT", "ID"]
states_pacific_abbr = ["CA", "NV", "OR", "WA"]
states_eastern = list(map(abbr2state, states_eastern_abbr))
states_central = list(map(abbr2state, states_central_abbr))
states_mountain = list(map(abbr2state, states_mountain_abbr))
states_pacific = list(map(abbr2state, states_pacific_abbr))

# some parameters
stated_states = ["New Jersey", "New York", "Virginia"]
# Date Jan 29, 2021
# (2) PA, NJ, VA, NY, MI all have EV and heat pump stocks from NZA DD case
# consistent with their economywide decarbonization goals.
# https://www.c2es.org/content/state-climate-policy/
# Date Feb 10, 2021
# Remove high electrification growth in PA and MI in stated policies;
# they dont have clean energy goals so kind of confusing/inconsistent to require high electrification in these states.
# So our new "Stated Policies" definition for electrification is states
# with BOTH economywide emissions goals + 100% carbon-free electricity standards
# = NY, NJ, VA.
stated_states_abbr = list(map(state2abbr, stated_states))
# years = ["2022", "2025", "2030", "2040", "2050"]
cases = ["current_policy", "stated_policy", "deep_decarbonization"]

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

# Define function for adjusting time-difference
def addhour(x):
    x += 1
    x = x.replace(8761, 1)
    return x


def SolveThreeUnknowns(a1, b1, c1, d1, a2, b2, c2, d2, a3, b3, c3, d3):
    D = (
        a1 * b2 * c3
        + b1 * c2 * a3
        + c1 * a2 * b3
        - a1 * c2 * b3
        - b1 * a2 * c3
        - c1 * b2 * a3
    )
    Dx = (
        d1 * b2 * c3
        + b1 * c2 * d3
        + c1 * d2 * b3
        - d1 * c2 * b3
        - b1 * d2 * c3
        - c1 * b2 * d3
    )
    Dy = (
        a1 * d2 * c3
        + d1 * c2 * a3
        + c1 * a2 * d3
        - a1 * c2 * d3
        - d1 * a2 * c3
        - c1 * d2 * a3
    )
    Dz = (
        a1 * b2 * d3
        + b1 * d2 * a3
        + d1 * a2 * b3
        - a1 * d2 * b3
        - b1 * a2 * d3
        - d1 * b2 * a3
    )
    Sx = Dx / D
    Sy = Dy / D
    Sz = Dz / D
    d = {"Sx": Sx, "Sy": Sy, "Sz": Sz}
    return pd.DataFrame(d)


def SolveTwoUnknowns(a1, b1, c1, a2, b2, c2):
    D = a1 * b2 - a2 * b1
    Dx = c1 * b2 - c2 * b1
    Dy = a1 * c2 - a2 * c1
    Sx = Dx / D
    Sy = Dy / D
    d = {"Sx": Sx, "Sy": Sy}
    return pd.DataFrame(d)


def CreateOutputFolder(case_folder):
    path = case_folder / "extra_outputs"
    if not os.path.exists(path):
        os.makedirs(path)


######################################
# CreatingBaseLoad.R
def CreateBaseLoad(
    years: List[int],
    regions: List[str],
    output_folder: Path,
    path_growthrate: Path = None,
) -> None:
    path_processed = path_in
    path_result = output_folder

    ## Method 3: annually
    EFS_2020_LoadProf = pd.read_parquet(path_in / "EFS_REF_load_2020.parquet")
    EFS_2020_LoadProf = pd.merge(EFS_2020_LoadProf, pop, on=["State"])
    EFS_2020_LoadProf = EFS_2020_LoadProf.assign(
        weighted=EFS_2020_LoadProf["LoadMW"] * EFS_2020_LoadProf["State Prop"]
    )
    EFS_2020_LoadProf = EFS_2020_LoadProf.groupby(
        ["Year", "GenX.Region", "LocalHourID", "Sector", "Subsector"], as_index=False
    ).agg({"weighted": "sum"})

    # Read in 2019 Demand
    Original_Load_2019 = pd.read_parquet(path_in / "ipm_load_curves_2019_EST.parquet")
    # Reorganize Demand
    Original_Load_2019 = Original_Load_2019.melt(id_vars="LocalHourID").rename(
        columns={"variable": "GenX.Region", "value": "LoadMW_original"}
    )
    Original_Load_2019 = Original_Load_2019.groupby(
        ["LocalHourID"], as_index=False
    ).agg({"LoadMW_original": "sum"})

    ratio_A = (
        Original_Load_2019["LoadMW_original"].sum()
        / EFS_2020_LoadProf["weighted"].sum()
    )
    EFS_2020_LoadProf = EFS_2020_LoadProf.assign(
        weighted=EFS_2020_LoadProf["weighted"] * ratio_A
    )

    Base_Load_2019 = EFS_2020_LoadProf.rename(columns={"weighted": "LoadMW"})
    # breakpoint()
    # Read in the Growth Rate
    if path_growthrate:
        GrowthRate = pd.read_parquet(path_growthrate)
    else:
        GrowthRate = pd.read_parquet(path_in / "ipm_growthrate_2019.parquet")

    # Create Base loads
    Base_Load_2019 = Base_Load_2019[Base_Load_2019["GenX.Region"].isin(regions)]
    Base_Load_2019.loc[
        (Base_Load_2019["Sector"] == "Industrial")
        & (Base_Load_2019["Subsector"].isin(["process heat", "machine drives"])),
        "Subsector",
    ] = "other"
    Base_Load_2019 = Base_Load_2019[Base_Load_2019["Subsector"] == "other"]
    Base_Load_2019 = Base_Load_2019.groupby(
        ["Year", "LocalHourID", "GenX.Region", "Sector"], as_index=False
    ).agg({"LoadMW": "sum"})
    Base_Load = Base_Load_2019
    for y in years:
        ScaleFactor = GrowthRate.assign(
            ScaleFactor=(1 + GrowthRate["growth_rate"]) ** (int(y) - 2019)
        ).drop(columns="growth_rate")
        Base_Load_temp = pd.merge(Base_Load_2019, ScaleFactor, on=["GenX.Region"])
        Base_Load_temp = Base_Load_temp.assign(
            Year=y, LoadMW=Base_Load_temp["LoadMW"] * Base_Load_temp["ScaleFactor"]
        ).drop(columns="ScaleFactor")
        Base_Load = Base_Load.append(Base_Load_temp, ignore_index=True)
    Base_Load.to_parquet(path_result + "\Base_Load.parquet", index=False)
    del (
        Base_Load,
        Base_Load_2019,
        Base_Load_temp,
        ScaleFactor,
        GrowthRate,
        Original_Load_2019,
    )


#####################################
# Add_Electrification.R
def AddElectrification(
    years: List[int],
    regions: List[str],
    electrification: List[str],
    output_folder: Path,
    path_stock: Path = None,
) -> pd.DataFrame:
    path_processed = path_in
    path_result = output_folder
    # Creating Time-series

    SCENARIO_STOCK = pd.read_parquet(path_processed / "SCENARIO_STOCK.parquet")
    SCENARIO_STOCK = SCENARIO_STOCK[
        (SCENARIO_STOCK["YEAR"].isin(years))
        & (SCENARIO_STOCK["SCENARIO"].isin(electrification))
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
        timeseries = pd.read_parquet(
            path_processed
            / f"{running_sector[i]}_{running_subsector[i]}_Incremental_Factor.parquet"
        )
        timeseries = timeseries[
            ["State", "Year", "LocalHourID", "Unit", "Factor_Type1", "Factor_Type2"]
        ]
        stock_temp = SCENARIO_STOCK[
            (SCENARIO_STOCK["SECTOR"] == running_sector[i])
            & (SCENARIO_STOCK["SUBSECTOR"] == running_subsector[i])
        ]
        stock_temp = stock_temp[
            ["SCENARIO", "STATE", "YEAR", "AGG_STOCK_TYPE1", "AGG_STOCK_TYPE2"]
        ].rename(columns={"STATE": "State", "YEAR": "Year"})
        years_pd = pd.Series(years)
        IF_years = pd.Series(timeseries["Year"].unique())
        for year in years_pd:
            exists = year in IF_years.values
            if not exists:
                diff = np.array(IF_years - year)
                index = diff[np.where(diff <= 0)].argmax()
                year_approx = IF_years[index]
                timeseries_temp = timeseries[timeseries["Year"] == year_approx]
                timeseries_temp["Year"] = year
                logger.warning(
                    "No incremental factor available for year "
                    + str(year)
                    + ": using factors from year "
                    + str(year_approx)
                    + "."
                )
                timeseries = timeseries.append(timeseries_temp)

        timeseries = pd.merge(timeseries, stock_temp, on=["State", "Year"])
        timeseries = timeseries.assign(
            LoadMW=timeseries["AGG_STOCK_TYPE1"] * timeseries["Factor_Type1"]
            + timeseries["AGG_STOCK_TYPE2"] * timeseries["Factor_Type2"]
        )
        timeseries = timeseries[
            ["SCENARIO", "State", "Year", "LocalHourID", "LoadMW"]
        ].dropna()
        timeseries.to_parquet(
            path_result
            / f"{running_sector[i]}_{running_subsector[i]}_Scenario_Timeseries_Method1.parquet",
            index=False,
        )
    # del timeseries, stock_temp

    ##########################
    # Read in time series and combine them
    Method = "Method1"
    Res_SPH = pd.read_parquet(
        path_result
        / f"Residential_space heating and cooling_Scenario_Timeseries_{Method}.parquet"
    )
    Res_SPH = Res_SPH.rename(columns={"LoadMW": "Res_SPH_LoadMW"})
    Res_SPH_sum = Res_SPH
    Res_SPH_sum = Res_SPH.groupby(["SCENARIO", "State", "Year"], as_index=False)[
        "Res_SPH_LoadMW"
    ].agg({"Total_Res_SPH_TWh": "sum"})
    Res_SPH_sum["Total_Res_SPH_TWh"] = 10 ** -6 * Res_SPH_sum["Total_Res_SPH_TWh"]

    Res_WH = pd.read_parquet(
        path_result / f"Residential_water heating_Scenario_Timeseries_{Method}.parquet"
    )
    Res_WH = Res_WH.rename(columns={"LoadMW": "Res_WH_LoadMW"})
    Res_WH_sum = Res_WH
    Res_WH_sum = Res_WH.groupby(["SCENARIO", "State", "Year"], as_index=False)[
        "Res_WH_LoadMW"
    ].agg({"Total_Res_WH_TWh": "sum"})
    Res_WH_sum["Total_Res_WH_TWh"] = 10 ** -6 * Res_WH_sum["Total_Res_WH_TWh"]

    Com_SPH = pd.read_parquet(
        path_result
        / f"Commercial_space heating and cooling_Scenario_Timeseries_{Method}.parquet"
    )
    Com_SPH = Com_SPH.rename(columns={"LoadMW": "Com_SPH_LoadMW"})
    Com_SPH_sum = Com_SPH
    Com_SPH_sum = Com_SPH.groupby(["SCENARIO", "State", "Year"], as_index=False)[
        "Com_SPH_LoadMW"
    ].agg({"Total_Com_SPH_TWh": "sum"})
    Com_SPH_sum["Total_Com_SPH_TWh"] = 10 ** -6 * Com_SPH_sum["Total_Com_SPH_TWh"]

    Com_WH = pd.read_parquet(
        path_result / f"Commercial_water heating_Scenario_Timeseries_{Method}.parquet"
    )
    Com_WH = Com_WH.rename(columns={"LoadMW": "Com_WH_LoadMW"})
    Com_WH_sum = Com_WH
    Com_WH_sum = Com_WH.groupby(["SCENARIO", "State", "Year"], as_index=False)[
        "Com_WH_LoadMW"
    ].agg({"Total_Com_WH_TWh": "sum"})
    Com_WH_sum["Total_Com_WH_TWh"] = 10 ** -6 * Com_WH_sum["Total_Com_WH_TWh"]

    Trans_LDV = pd.read_parquet(
        path_result
        / f"Transportation_light-duty vehicles_Scenario_Timeseries_{Method}.parquet"
    )
    Trans_LDV = Trans_LDV.rename(columns={"LoadMW": "LDV_LoadMW"})
    Trans_LDV_sum = Trans_LDV
    Trans_LDV_sum = Trans_LDV.groupby(["SCENARIO", "State", "Year"], as_index=False)[
        "LDV_LoadMW"
    ].agg({"Total_Trans_LDV_TWh": "sum"})
    Trans_LDV_sum["Total_Trans_LDV_TWh"] = (
        10 ** -6 * Trans_LDV_sum["Total_Trans_LDV_TWh"]
    )

    Trans_MDV = pd.read_parquet(
        path_result
        / f"Transportation_medium-duty trucks_Scenario_Timeseries_{Method}.parquet"
    )
    Trans_MDV = Trans_MDV.rename(columns={"LoadMW": "MDV_LoadMW"})
    Trans_MDV_sum = Trans_MDV
    Trans_MDV_sum = Trans_MDV.groupby(["SCENARIO", "State", "Year"], as_index=False)[
        "MDV_LoadMW"
    ].agg({"Total_Trans_MDV_TWh": "sum"})
    Trans_MDV_sum["Total_Trans_MDV_TWh"] = (
        10 ** -6 * Trans_MDV_sum["Total_Trans_MDV_TWh"]
    )

    Trans_HDV = pd.read_parquet(
        path_result
        / f"Transportation_heavy-duty trucks_Scenario_Timeseries_{Method}.parquet"
    )
    Trans_HDV = Trans_HDV.rename(columns={"LoadMW": "HDV_LoadMW"})
    Trans_HDV_sum = Trans_HDV
    Trans_HDV_sum = Trans_HDV.groupby(["SCENARIO", "State", "Year"], as_index=False)[
        "HDV_LoadMW"
    ].agg({"Total_Trans_HDV_TWh": "sum"})
    Trans_HDV_sum["Total_Trans_HDV_TWh"] = (
        10 ** -6 * Trans_HDV_sum["Total_Trans_HDV_TWh"]
    )

    Trans_BUS = pd.read_parquet(
        path_result
        / f"Transportation_transit buses_Scenario_Timeseries_{Method}.parquet"
    )
    Trans_BUS = Trans_BUS.rename(columns={"LoadMW": "BUS_LoadMW"})
    Trans_BUS_sum = Trans_BUS
    Trans_BUS_sum = Trans_BUS.groupby(["SCENARIO", "State", "Year"], as_index=False)[
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
        temp = pd.merge(sc, pop, on=["State"], how="left")
        temp = (
            temp.assign(
                weighted=temp[column_names[i] + "_" + "LoadMW"] * temp["State Prop"]
            )
            .groupby(
                ["SCENARIO", "Year", "LocalHourID", "GenX.Region"], as_index=False
            )["weighted"]
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

    Base_Load = pd.read_parquet(path_result / "Base_Load.parquet")
    Base_Load = Base_Load.rename(columns={"LoadMW": "Base_MW"})

    Base_Load.loc[(Base_Load["Sector"] == "Commercial"), "Subsector"] = "Base_Com_other"
    Base_Load.loc[
        (Base_Load["Sector"] == "Residential"), "Subsector"
    ] = "Base_Res_other"
    Base_Load.loc[
        (Base_Load["Sector"] == "Transportation"), "Subsector"
    ] = "Base_Trans_other"
    Base_Load.loc[(Base_Load["Sector"] == "Industrial"), "Subsector"] = "Base_Ind"

    Base_Load = Base_Load.drop(columns=["Sector"])
    Base_Load = (
        Base_Load.pivot_table(
            index=[Base_Load.index.values, "LocalHourID", "GenX.Region", "Year"],
            columns="Subsector",
            values="Base_MW",
        )
        .reset_index(["LocalHourID", "GenX.Region", "Year"])
        .fillna(0)
    )
    Base_Load = Base_Load.groupby(
        ["LocalHourID", "GenX.Region", "Year"], as_index=False
    ).agg(
        {
            "Base_Com_other": "sum",
            "Base_Res_other": "sum",
            "Base_Trans_other": "sum",
            "Base_Ind": "sum",
        }
    )

    Res_WH = pd.read_parquet(path_result / "Res_WH_By_region.parquet")
    Com_WH = pd.read_parquet(path_result / "Com_WH_By_region.parquet")
    Res_SPH = pd.read_parquet(path_result / "Res_SPH_By_region.parquet")
    Com_SPH = pd.read_parquet(path_result / "Com_SPH_By_region.parquet")
    Trans_LDV = pd.read_parquet(path_result / "Trans_LDV_By_region.parquet")
    Trans_MDV = pd.read_parquet(path_result / "Trans_MDV_By_region.parquet")
    Trans_HDV = pd.read_parquet(path_result / "Trans_HDV_By_region.parquet")
    # Trans_BUS = pd.read_parquet(path_result + "\Trans_BUS_By_region.parquet")

    Total_Load = pd.merge(
        Res_WH, Com_WH, on=["SCENARIO", "Year", "LocalHourID", "GenX.Region"]
    )
    Total_Load = pd.merge(
        Total_Load, Res_SPH, on=["SCENARIO", "Year", "LocalHourID", "GenX.Region"]
    )
    Total_Load = pd.merge(
        Total_Load, Com_SPH, on=["SCENARIO", "Year", "LocalHourID", "GenX.Region"]
    )
    Total_Load = pd.merge(
        Total_Load, Trans_LDV, on=["SCENARIO", "Year", "LocalHourID", "GenX.Region"]
    )
    Total_Load = pd.merge(
        Total_Load, Trans_MDV, on=["SCENARIO", "Year", "LocalHourID", "GenX.Region"]
    )
    Total_Load = pd.merge(
        Total_Load, Trans_HDV, on=["SCENARIO", "Year", "LocalHourID", "GenX.Region"]
    )
    # Total_Load = pd.merge(Total_Load, Trans_BUS, on = ["SCENARIO","Year","LocalHourID","GenX.Region"])
    Total_Load = pd.merge(
        Total_Load, Base_Load, on=["Year", "LocalHourID", "GenX.Region"]
    )
    Total_Load = Total_Load[Total_Load["Year"].isin(years)]
    del Base_Load, Res_WH, Com_WH, Res_SPH, Com_SPH, Trans_LDV, Trans_MDV, Trans_HDV

    Total_Load = Total_Load[
        (Total_Load["Year"].isin(years)) & (Total_Load["GenX.Region"].isin(regions))
    ]
    Total_Load.to_parquet(
        path_result + "\Total_load_by_region_full.parquet", index=False
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
            regions = regions + regions_to_keep(_settings)[0]
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
    CreateBaseLoad(years, regions, output_folder, path_growthrate)
    return AddElectrification(
        years, regions, electrification, output_folder, path_stock
    )


def FilterTotalProfile(settings, total_load):
    TotalLoad = total_load
    settings = settings
    TotalLoad = TotalLoad[TotalLoad["Year"] == settings["model_year"]]
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
    TotalLoad = TotalLoad.drop(columns=["SCENARIO", "Year"]).rename(
        columns={"LocalHourID": "time_index", "GenX.Region": "region"}
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
