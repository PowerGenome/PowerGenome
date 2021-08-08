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
from powergenome.us_state_abbrev import (state2abbr, abbr2state)
path_in = r"..\data\load_profiles_data\input" # fix

#read in state proportions 
#how much state load should be distributed to GenXRegion
# pop = pd.read_parquet(path_in + "\GenX_State_Pop_Weight.parquet")
pop = pd.read_parquet(path_in + "\ipm_state_pop_weight_20210517.parquet")
states = pop.drop_duplicates(subset=["State"])["State"]
states_abb = list(map(state2abbr, states))
pop["State"] = list(map(state2abbr, pop["State"]))
states_eastern_abbr = ["ME","VT","NH","MA","RI","CT","NY","PA","NJ","DE","MD","DC","MI","IN","OH","KY","WV","VA","NC","SC","GA","FL"]
states_central_abbr = ["IL","MO","TN","AL","MS","WI","AR","LA","TX","OK","KS","NE","SD","ND","IA","MN"]
states_mountain_abbr = ["MT","WY","CO","NM","AZ","UT","ID"]
states_pacific_abbr = ["CA","NV","OR","WA"]
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
#years = ["2022", "2025", "2030", "2040", "2050"]
cases = ["current_policy", "stated_policy", "deep_decarbonization"]

running_sector = ['Residential','Residential', 'Commercial', 'Commercial','Transportation','Transportation','Transportation', 'Transportation']
running_subsector = ['space heating and cooling','water heating', 'space heating and cooling', 'water heating','light-duty vehicles','medium-duty trucks','heavy-duty trucks','transit buses']
Nsubsector = len(running_subsector)

logger = logging.getLogger(__name__)

#Define function for adjusting time-difference
def addhour(x):
    x += 1
    x = x.replace(8761, 1)
    return x

def SolveThreeUnknowns(a1, b1, c1, d1, a2, b2, c2, d2, a3, b3, c3, d3):
    D  = a1*b2*c3 + b1*c2*a3 + c1*a2*b3 - a1*c2*b3 - b1*a2*c3 - c1*b2*a3
    Dx = d1*b2*c3 + b1*c2*d3 + c1*d2*b3 - d1*c2*b3 - b1*d2*c3 - c1*b2*d3
    Dy = a1*d2*c3 + d1*c2*a3 + c1*a2*d3 - a1*c2*d3 - d1*a2*c3 - c1*d2*a3
    Dz = a1*b2*d3 + b1*d2*a3 + d1*a2*b3 - a1*d2*b3 - b1*a2*d3 - d1*b2*a3
    Sx = Dx/D
    Sy = Dy/D
    Sz = Dz/D
    d = {'Sx':Sx, 'Sy':Sy, 'Sz':Sz}
    return pd.DataFrame(d)
    
def SolveTwoUnknowns(a1, b1, c1, a2, b2, c2):
    D  = a1*b2 - a2*b1
    Dx = c1*b2 - c2*b1
    Dy = a1*c2 - a2*c1
    Sx = Dx/D
    Sy = Dy/D
    d = {'Sx':Sx, 'Sy':Sy}
    return pd.DataFrame(d)

def CreateOutputFolder(case_folder):
    path = case_folder / "extra_outputs"
    if not os.path.exists(path):
        os.makedirs(path)



######################################
# CreatingBaseLoad.R
def CreateBaseLoad(years, regions, output_folder):
    path_processed = path_in
    path_result = output_folder.__str__()
    years = years
    regions = regions

    #Calculate how much the ratio is DC load of MD load for each sector, each year
    EFS_Stock = pd.read_parquet(path_processed + "\EFS_STOCK_AGG.parquet")
    EFS_Stock = EFS_Stock[EFS_Stock["SCENARIO"] == "REFERENCE ELECTRIFICATION - MODERATE TECHNOLOGY ADVANCEMENT"].drop(columns="SCENARIO") ##Note that in the stock of heating include the air conditioners, chillers, etc.
    EFS_Stock_MD = EFS_Stock[EFS_Stock["STATE"] == 'MD'].rename(columns = {"AGG_STOCK" : "MD_AGG_STOCK"}).drop(columns = ["AGG_STOCK_TYPE1", "AGG_STOCK_TYPE2", "STATE", "TYPE1-C", "NONELEC", "NONELEC-C"])
    EFS_Stock_DC = EFS_Stock[EFS_Stock["STATE"] == 'DC'].rename(columns = {"AGG_STOCK" : "DC_AGG_STOCK"}).drop(columns = ["AGG_STOCK_TYPE1", "AGG_STOCK_TYPE2", "STATE", "TYPE1-C", "NONELEC", "NONELEC-C"])
    EFS_Stock_MD_DC = pd.merge(EFS_Stock_MD, EFS_Stock_DC, on = ["SECTOR", "SUBSECTOR", "YEAR", "UNIT"])
    EFS_Stock_MD_DC = EFS_Stock_MD_DC.assign(Ratio = EFS_Stock_MD_DC["DC_AGG_STOCK"]/EFS_Stock_MD_DC["MD_AGG_STOCK"])
    EFS_Stock_MD_DC.columns = EFS_Stock_MD_DC.columns.str.title()
    EFS_Stock_MD_DC["Ratio"].fillna(1)

    #----Read in data
    EFS_Reference_Load = pd.read_parquet(path_processed + "\EFS_REF_MOD_LOADPROF.parquet")
    EFS_Reference_Load_Eastern = EFS_Reference_Load[EFS_Reference_Load["State"].isin(states_eastern_abbr)]
    EFS_Reference_Load_Central = EFS_Reference_Load[EFS_Reference_Load["State"].isin(states_central_abbr)]
    EFS_Reference_Load_Central = EFS_Reference_Load_Central.assign(LocalHourID = addhour(EFS_Reference_Load_Central["LocalHourID"]))
    EFS_Reference_Load_Mountain = EFS_Reference_Load[EFS_Reference_Load["State"].isin(states_mountain_abbr)]
    EFS_Reference_Load_Mountain = EFS_Reference_Load_Mountain.assign(LocalHourID = addhour(addhour(EFS_Reference_Load_Mountain["LocalHourID"])))
    EFS_Reference_Load_Pacific = EFS_Reference_Load[EFS_Reference_Load["State"].isin(states_pacific_abbr)]
    EFS_Reference_Load_Pacific = EFS_Reference_Load_Pacific.assign(LocalHourID = addhour(addhour(addhour(EFS_Reference_Load_Pacific["LocalHourID"]))))
    EFS_Reference_Load_DC = pd.merge(EFS_Reference_Load[EFS_Reference_Load["State"] == "MD"], EFS_Stock_MD_DC, on = ["Sector", "Subsector", "Year"])
    EFS_Reference_Load_DC = EFS_Reference_Load_DC.assign(LoadMW = EFS_Reference_Load_DC["LoadMW"] * EFS_Reference_Load_DC["Ratio"], State = "DC").drop(columns= ["Unit", "Dc_Agg_Stock", "Md_Agg_Stock", "Ratio"])
    EFS_Reference_Load_Synced = EFS_Reference_Load_Eastern.append([EFS_Reference_Load_Central,EFS_Reference_Load_DC], ignore_index=True)
    try:
        EFS_Reference_Load_Synced = EFS_Reference_Load_Synced.append(EFS_Reference_Load_Mountain, ignore_index=True)
    except AttributeError:
        pass
    except NameError:
        pass
    try:
        EFS_Reference_Load_Synced = EFS_Reference_Load_Synced.append(EFS_Reference_Load_Pacific, ignore_index=True)
    except AttributeError:
        pass
    except NameError:
        pass
    del EFS_Stock, EFS_Stock_MD, EFS_Stock_DC, EFS_Stock_MD_DC, EFS_Reference_Load, EFS_Reference_Load_Eastern, EFS_Reference_Load_DC, \
        EFS_Reference_Load_Central, EFS_Reference_Load_Mountain, EFS_Reference_Load_Pacific

    # Create Load to be removed from 2020 load profiles
    EFS_Reference_2020 = EFS_Reference_Load_Synced[EFS_Reference_Load_Synced["Year"] == 2020].drop(columns = ["Electrification", "TechnologyAdvancement"]).drop_duplicates()
    #EFS_Reference_2020 = EFS_Reference_2020.groupby(["Year", "LocalHourID", "State"], as_index=False).agg({"LoadMW" : "sum"})
    EFS_Reference_2020 = pd.merge(EFS_Reference_2020, pop, on = ["State"])
    EFS_Reference_2020 = EFS_Reference_2020.assign(weighted = EFS_Reference_2020["LoadMW"]*EFS_Reference_2020["State Prop"])
    EFS_Reference_2020 = EFS_Reference_2020.groupby(["Year", "LocalHourID", "GenX.Region", "Sector", "Subsector"], as_index = False).agg({"weighted" : "sum"})

    del EFS_Reference_Load_Synced

    ## Method 3: annually

    EFS_2020_LoadProf = pd.read_parquet(path_in + "\EFS_REF_load_2020.parquet")
    EFS_2020_LoadProf = pd.merge(EFS_2020_LoadProf, pop, on = ["State"])
    EFS_2020_LoadProf = EFS_2020_LoadProf.assign(weighted = EFS_2020_LoadProf["LoadMW"]*EFS_2020_LoadProf["State Prop"])
    EFS_2020_LoadProf = EFS_2020_LoadProf.groupby(["Year", "GenX.Region", "LocalHourID", "Sector", "Subsector"], as_index = False).agg({"weighted" : "sum"})

    Original_Load_2019 = pd.read_parquet(path_in + "\ipm_load_curves_2019_EST.parquet")
    Original_Load_2019 = Original_Load_2019.melt(id_vars="LocalHourID").rename(columns={"variable" : "GenX.Region", "value": "LoadMW_original"})
    Original_Load_2019 = Original_Load_2019.groupby(["LocalHourID"], as_index = False).agg({"LoadMW_original" : "sum"})

    ratio_A = Original_Load_2019["LoadMW_original"].sum() / EFS_2020_LoadProf["weighted"].sum() 
    EFS_2020_LoadProf = EFS_2020_LoadProf.assign(weighted = EFS_2020_LoadProf["weighted"]*ratio_A)

    Base_Load_2019 = EFS_2020_LoadProf.rename(columns ={"weighted" : "LoadMW"})

    # Read in 2019 Demand
    Original_Load_2019 = pd.read_parquet(path_in + "\ipm_load_curves_2019_EST.parquet")
    # Reorganize Demand
    Original_Load_2019 = Original_Load_2019.melt(id_vars="LocalHourID").rename(columns={"variable" : "GenX.Region", "value": "LoadMW_original"})
    # Merge 2020 Demand with EFS 2020 Load Demand, and take the difference
    # We assume EFS's 2020 is almost the same as 2019
    Base_Load_2019 = pd.merge(Original_Load_2019, EFS_Reference_2020, on = ["GenX.Region", "LocalHourID"])
    Base_Load_2019 = Base_Load_2019.assign(LoadMW = Base_Load_2019["LoadMW_original"] - Base_Load_2019["weighted"], Year = 2019)\
        .drop(columns = ["LoadMW_original", "weighted"])
    Base_Load_2019 = Base_Load_2019[Base_Load_2019["GenX.Region"].isin(regions)]
    # Read in the Growth Rate
    GrowthRate = pd.read_parquet(path_in + "\ipm_growthrate_2019.parquet")
    # Create Base loads
    Base_Load = Base_Load_2019
    for y in years:
        ScaleFactor = GrowthRate.assign(ScaleFactor = (1+GrowthRate["growth_rate"])**(int(y) - 2019)) \
            .drop(columns = "growth_rate")
        Base_Load_temp = pd.merge(Base_Load_2019, ScaleFactor, on = ["GenX.Region"])
        Base_Load_temp = Base_Load_temp.assign(Year = y, LoadMW = Base_Load_temp["LoadMW"]*Base_Load_temp["ScaleFactor"])\
                .drop(columns = "ScaleFactor")
        Base_Load = Base_Load.append(Base_Load_temp, ignore_index=True)
    Base_Load.to_parquet(path_result + "\Base_Load.parquet", index = False)

    del Base_Load, Base_Load_2019, Base_Load_temp, ScaleFactor,GrowthRate, Original_Load_2019

#####################################
# Add_Electrification.R
def AddElectrification(years, regions, electrification, output_folder, path_stock):
    path_processed = path_in
    path_result = output_folder.__str__()
    path_stock = path_stock
    years = years
    electrification = electrification    
    regions = regions
    #Creating Time-series

    SCENARIO_STOCK = pd.read_parquet(path_processed + "\SCENARIO_STOCK.parquet")
    SCENARIO_STOCK = SCENARIO_STOCK[(SCENARIO_STOCK["YEAR"].isin(years)) & (SCENARIO_STOCK["SCENARIO"].isin(electrification))]
    
    try:
        CUSTOM_STOCK = pd.read_parquet(path_stock)
        CUSTOM_STOCK = CUSTOM_STOCK[(CUSTOM_STOCK["YEAR"].isin(years)) & (CUSTOM_STOCK["SCENARIO"].isin(electrification))]
        SCENARIO_STOCK = SCENARIO_STOCK.append(CUSTOM_STOCK)
    except:
        pass
    
    #Method 1 Calculate from Type1 and Type 2
    for i in range(0, Nsubsector):
        timeseries = pd.read_parquet(path_processed + "\\" + running_sector[i] + "_" + running_subsector[i] + "_Incremental_Factor.parquet")
        timeseries = timeseries[["State", "Year", "LocalHourID", "Unit", "Factor_Type1", "Factor_Type2" ]]
        stock_temp = SCENARIO_STOCK[(SCENARIO_STOCK["SECTOR"] == running_sector[i]) & (SCENARIO_STOCK["SUBSECTOR"] == running_subsector[i])]
        stock_temp = stock_temp[["SCENARIO", "STATE", "YEAR", "AGG_STOCK_TYPE1", "AGG_STOCK_TYPE2"]].rename(columns={"STATE" : "State", "YEAR" : "Year"})
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
                logger.warning("No incremental factor available for year " + str(year) + ": using factors from year " + str(year_approx) + ".")
            timeseries = timeseries.append(timeseries_temp)
 
        timeseries = pd.merge(timeseries, stock_temp, on = ["State", "Year"])
        timeseries = timeseries.assign(LoadMW = timeseries["AGG_STOCK_TYPE1"]*timeseries["Factor_Type1"] + timeseries["AGG_STOCK_TYPE2"]*timeseries["Factor_Type2"])
        timeseries = timeseries[["SCENARIO", "State", "Year", "LocalHourID", "LoadMW"]].dropna()
        timeseries.to_parquet(path_result + "\\" + running_sector[i] + "_" + running_subsector[i] + "_Scenario_Timeseries_Method1.parquet", index = False)
    del timeseries, stock_temp

    ##########################
    # Read in time series and combine them
    Method = "Method1"
    Res_SPH = pd.read_parquet(path_result + "\Residential_space heating and cooling_Scenario_Timeseries_" + Method + ".parquet")
    Res_SPH = Res_SPH.rename(columns={"LoadMW" : "Res_SPH_LoadMW"})
    Res_SPH_sum = Res_SPH
    Res_SPH_sum = Res_SPH.groupby(["SCENARIO", "State", "Year"], as_index = False)["Res_SPH_LoadMW"].agg({"Total_Res_SPH_TWh" : "sum"})
    Res_SPH_sum["Total_Res_SPH_TWh"] = 10**-6*Res_SPH_sum["Total_Res_SPH_TWh"]

    Res_WH = pd.read_parquet(path_result + "\Residential_water heating_Scenario_Timeseries_" + Method +".parquet")
    Res_WH = Res_WH.rename(columns ={"LoadMW" : "Res_WH_LoadMW"})
    Res_WH_sum = Res_WH
    Res_WH_sum = Res_WH.groupby(["SCENARIO", "State", "Year"], as_index = False)["Res_WH_LoadMW"].agg({"Total_Res_WH_TWh" : "sum"})
    Res_WH_sum["Total_Res_WH_TWh"] = 10**-6*Res_WH_sum["Total_Res_WH_TWh"]

    Com_SPH = pd.read_parquet(path_result + "\Commercial_space heating and cooling_Scenario_Timeseries_" + Method +".parquet")
    Com_SPH = Com_SPH.rename(columns={"LoadMW" : "Com_SPH_LoadMW"})
    Com_SPH_sum = Com_SPH
    Com_SPH_sum = Com_SPH.groupby(["SCENARIO", "State", "Year"], as_index = False)["Com_SPH_LoadMW"].agg({"Total_Com_SPH_TWh" : "sum"})
    Com_SPH_sum["Total_Com_SPH_TWh"] = 10**-6*Com_SPH_sum["Total_Com_SPH_TWh"]

    Com_WH = pd.read_parquet(path_result + "\Commercial_water heating_Scenario_Timeseries_" + Method +".parquet")
    Com_WH = Com_WH.rename(columns ={"LoadMW" : "Com_WH_LoadMW"})
    Com_WH_sum = Com_WH
    Com_WH_sum = Com_WH.groupby(["SCENARIO", "State", "Year"], as_index = False)["Com_WH_LoadMW"].agg({"Total_Com_WH_TWh" : "sum"})
    Com_WH_sum["Total_Com_WH_TWh"] = 10**-6*Com_WH_sum["Total_Com_WH_TWh"]

    Trans_LDV = pd.read_parquet(path_result + "\Transportation_light-duty vehicles_Scenario_Timeseries_" + Method +".parquet")
    Trans_LDV = Trans_LDV.rename(columns ={"LoadMW" : "LDV_LoadMW"})
    Trans_LDV_sum = Trans_LDV
    Trans_LDV_sum = Trans_LDV.groupby(["SCENARIO", "State", "Year"], as_index = False)["LDV_LoadMW"].agg({"Total_Trans_LDV_TWh" : "sum"})
    Trans_LDV_sum["Total_Trans_LDV_TWh"] = 10**-6*Trans_LDV_sum["Total_Trans_LDV_TWh"]

    Trans_MDV = pd.read_parquet(path_result + "\Transportation_medium-duty trucks_Scenario_Timeseries_" + Method +".parquet")
    Trans_MDV = Trans_MDV.rename(columns ={"LoadMW" : "MDV_LoadMW"})
    Trans_MDV_sum = Trans_MDV
    Trans_MDV_sum = Trans_MDV.groupby(["SCENARIO", "State", "Year"], as_index = False)["MDV_LoadMW"].agg({"Total_Trans_MDV_TWh" : "sum"})
    Trans_MDV_sum["Total_Trans_MDV_TWh"] = 10**-6*Trans_MDV_sum["Total_Trans_MDV_TWh"]

    Trans_HDV = pd.read_parquet(path_result + "\Transportation_heavy-duty trucks_Scenario_Timeseries_" + Method +".parquet")
    Trans_HDV = Trans_HDV.rename(columns ={"LoadMW" : "HDV_LoadMW"})
    Trans_HDV_sum = Trans_HDV
    Trans_HDV_sum = Trans_HDV.groupby(["SCENARIO", "State", "Year"], as_index = False)["HDV_LoadMW"].agg({"Total_Trans_HDV_TWh" : "sum"})
    Trans_HDV_sum["Total_Trans_HDV_TWh"] = 10**-6*Trans_HDV_sum["Total_Trans_HDV_TWh"]

    Trans_BUS = pd.read_parquet(path_result + "\Transportation_transit buses_Scenario_Timeseries_" + Method +".parquet")
    Trans_BUS = Trans_BUS.rename(columns ={"LoadMW" : "BUS_LoadMW"})
    Trans_BUS_sum = Trans_BUS
    Trans_BUS_sum = Trans_BUS.groupby(["SCENARIO", "State", "Year"], as_index = False)["BUS_LoadMW"].agg({"Total_Trans_BUS_TWh" : "sum"})
    Trans_BUS_sum["Total_Trans_BUS_TWh"] = 10**-6*Trans_BUS_sum["Total_Trans_BUS_TWh"]

    # Total_SPH = pd.merge(Res_SPH, Com_SPH, on = ["SCENARIO","State","Year","LocalHourID"])
    # Total_SPH = Total_SPH.assign(LoadMW = Total_SPH['Res_SPH_LoadMW'] + Total_SPH["Com_SPH_LoadMW"]).drop(columns = ["Res_SPH_LoadMW", "Com_SPH_LoadMW"])
    # Total_WH = pd.merge(Res_WH, Com_WH, on = ["SCENARIO","State","Year","LocalHourID"])
    # Total_WH = Total_WH.assign(LoadMW = Total_WH['Res_WH_LoadMW'] + Total_WH["Com_WH_LoadMW"]).drop(columns = ["Res_WH_LoadMW", "Com_WH_LoadMW"])
    # Total_MHDV = pd.merge(Trans_MDV, Trans_HDV, on = ["SCENARIO", "State", "Year", "LocalHourID"])
    # Total_MHDV = pd.merge(Total_MHDV, Trans_BUS, on = ["SCENARIO", "State", "Year", "LocalHourID"])
    # Total_MHDV = Total_MHDV.assign(LoadMW = Total_MHDV["MDV_LoadMW"] + Total_MHDV["HDV_LoadMW"] + Total_MHDV["BUS_LoadMW"]).drop(columns=["MDV_LoadMW", "HDV_LoadMW", "BUS_LoadMW"])

    #Total_SPH.to_parquet(path_or_buf = path_result + "\Total_SPH_" + Method + ".parquet", index=False)
    #Total_WH.to_parquet(path_or_buf = path_result + "\Total_WH_" + Method + ".parquet", index = False)
    #Trans_LDV.to_parquet(path_or_buf = path_result + "\Trans_LDV_" + Method + ".parquet", index = False)
    #Total_MHDV.to_parquet(path_or_buf = path_result + "\Total_MHDV_" + Method + ".parquet", index = False)

    # Total_SPH_Sum = Total_SPH.groupby(["SCENARIO", "State", "Year"], as_index=False)["LoadMW"].sum()
    # Total_SPH_Sum = Total_SPH_Sum.rename(columns = {"LoadMW" : "TotalMW"})
    # Total_WH_Sum = Total_WH.groupby(["SCENARIO", "State", "Year"], as_index=False)["LoadMW"].sum()
    # Total_WH_Sum = Total_WH_Sum.rename(columns = {"LoadMW" : "TotalMW"})
    # Trans_LDV_Sum = Trans_LDV.groupby(["SCENARIO", "State", "Year"], as_index=False)["LDV_LoadMW"].sum()
    # Trans_LDV_Sum = Trans_LDV_Sum.rename(columns = {"LDV_LoadMW" : "TotalMW"})
    # Total_MHDV_Sum = Total_MHDV.groupby(["SCENARIO", "State", "Year"], as_index=False)["LoadMW"].sum()
    # Total_MHDV_Sum = Total_MHDV_Sum.rename(columns = {"LoadMW" : "TotalMW"})

    #Total_SPH_Sum.to_parquet(path_or_buf= path_result + "\Total_SPH_sum_" + Method + ".parquet", index = False)
    #Total_WH_Sum.to_parquet(path_or_buf= path_result + "\Total_WH_sum_" + Method + ".parquet", index = False)
    #Trans_LDV_Sum.to_parquet(path_or_buf= path_result + "\Trans_LDV_sum_" + Method + ".parquet", index = False)
    #Total_MHDV_Sum.to_parquet(path_or_buf= path_result + "\Total_MHDV_sum_" + Method + ".parquet", index = False)

    del  \
        Res_SPH_sum, Res_WH_sum, Com_SPH_sum, Trans_LDV_sum, \
             Trans_MDV_sum, Trans_HDV_sum, Trans_BUS_sum

    ################
    #Distribute Load to GenX.Region
    Method = 'Method1'
    subsectors = [Res_SPH, Res_WH, Com_SPH, Com_WH, Trans_LDV, Trans_MDV, Trans_HDV, Trans_BUS]
    subsector_names = ['Res_SPH', 'Res_WH', 'Com_SPH', 'Com_WH', 'Trans_LDV', 'Trans_MDV', 'Trans_HDV', 'Trans_BUS']
    column_names = ['Res_SPH', 'Res_WH', 'Com_SPH', 'Com_WH', 'LDV', 'MDV', 'HDV', 'BUS']
    j = 0

    for i in subsectors:
        temp = pd.merge(i, pop, on = ["State"], how = 'left')
        temp = temp.assign(weighted = temp[column_names[j] + "_" + "LoadMW"] * temp["State Prop"]).groupby(["SCENARIO", "Year", "LocalHourID", "GenX.Region"], as_index=False)["weighted"].sum()\
            .rename(columns={"weighted" : column_names[j] + "_MW"})
        temp.to_parquet(path_result + "\\" + subsector_names[j] + "_By_region.parquet", index = False)
        j = j + 1
    del temp, subsectors, Res_SPH, Res_WH, Com_SPH, Com_WH, Trans_LDV, Trans_MDV, Trans_HDV, Trans_BUS
    
    #Total_WH = pd.read_parquet(path_result + "\Total_WH_" + Method + ".parquet")
    # Total_WH = pd.merge(Total_WH, pop, on = ["State"], how = 'left')
    # Total_WH = Total_WH.assign(weighted = Total_WH["LoadMW"] * Total_WH["State Prop"]).groupby(["SCENARIO", "Year", "LocalHourID", "GenX.Region"], as_index=False)["weighted"].sum()\
    #     .rename(columns={"weighted" : "water_heat_MW"})
    # Total_WH = Total_WH[Total_WH["GenX.Region"].isin(regions)]
    # Total_WH.to_parquet(path_result + "\Total_WH_By_region.parquet", index = False)
    # del Total_WH

    # #Total_SPH = pd.read_parquet(path_result + "\Total_SPH_" + Method + ".parquet")
    # Total_SPH = pd.merge(Total_SPH, pop, on = ["State"], how = 'left')
    # Total_SPH = Total_SPH.assign(weighted = Total_SPH["LoadMW"] * Total_SPH["State Prop"]).groupby(["SCENARIO", "Year", "LocalHourID", "GenX.Region"], as_index=False)["weighted"].sum()\
    #     .rename(columns={"weighted" : "space_heat_MW"})
    
    # Total_SPH.to_parquet(path_result + "\Total_SPH_By_region.parquet", index = False)
    # del Total_SPH


    # #Total_LDEV = pd.read_parquet(path_result + "\Trans_LDV_" + Method + ".parquet")
    # Total_LDEV = Trans_LDV
    # Total_LDEV = pd.merge(Total_LDEV, pop, on = ["State"], how = 'left')
    # Total_LDEV = Total_LDEV.assign(weighted = Total_LDEV["LDV_LoadMW"] * Total_LDEV["State Prop"]).groupby(["SCENARIO", "Year", "LocalHourID", "GenX.Region"], as_index=False)["weighted"].sum()\
    #     .rename(columns={"weighted" : "LDEV_MW"})
    # Total_LDEV.to_parquet(path_result + "\Total_LDEV_By_region.parquet", index = False)
    # del Total_LDEV


    # #Total_MHDV = pd.read_parquet(path_result + "\Total_MHDV_" + Method + ".parquet")
    # Total_MHDV = pd.merge(Total_MHDV, pop, on = ["State"], how = 'left')
    # Total_MHDV = Total_MHDV.assign(weighted = Total_MHDV["LoadMW"] * Total_MHDV["State Prop"]).groupby(["SCENARIO", "Year", "LocalHourID", "GenX.Region"], as_index=False)["weighted"].sum()\
    #     .rename(columns={"weighted" : "MHBEV_MW"})
    # Total_MHDV.to_parquet(path_result + "\Total_MHDV_By_region.parquet", index = False)
    # del Total_MHDV

    ######
    #Construct Total Load

    Base_Load = pd.read_parquet(path_result + "\Base_Load.parquet")
    Base_Load = Base_Load.rename(columns= {"LoadMW" : "Base_MW"})
    Base_Load.loc[(Base_Load["Sector"] == "Commercial") & (Base_Load["Subsector"] == "space heating and cooling"), "Subsector"] = "Base_Com_SPH"
    Base_Load.loc[(Base_Load["Sector"] == "Commercial") & (Base_Load["Subsector"] == "water heating"), "Subsector"] = "Base_Com_WH"
    Base_Load.loc[(Base_Load["Sector"] == "Residential") & (Base_Load["Subsector"] == "space heating and cooling"), "Subsector"] = "Base_Res_SPH"
    Base_Load.loc[(Base_Load["Sector"] == "Residential") & (Base_Load["Subsector"] == "water heating"), "Subsector"] = "Base_Res_WH"
    Base_Load.loc[(Base_Load["Sector"] == "Transportation") & (Base_Load["Subsector"] == "light-duty vehicles"), "Subsector"] = "Base_Trans_LDV"
    Base_Load.loc[(Base_Load["Sector"] == "Transportation") & (Base_Load["Subsector"] == "medium-duty trucks"), "Subsector"] = "Base_Trans_MDV"
    Base_Load.loc[(Base_Load["Sector"] == "Transportation") & (Base_Load["Subsector"] == "heavy-duty trucks"), "Subsector"] = "Base_Trans_HDV"
    Base_Load.loc[(Base_Load["Sector"] == "Transportation") & (Base_Load["Subsector"] == "transit buses"), "Subsector"] = "Base_Trans_BUS"

    Base_Load = Base_Load.drop(columns = ["Sector"])
    Base_Load = Base_Load.pivot_table(index = [Base_Load.index.values, 'LocalHourID', "GenX.Region", "Year"], columns = 'Subsector', values = 'Base_MW')\
        .reset_index(['LocalHourID', "GenX.Region", "Year"]).fillna(0)
    Base_Load = Base_Load.groupby(['LocalHourID', 'GenX.Region', 'Year'], as_index=False).agg({'Base_Com_SPH' : 'sum', 'Base_Com_WH' : 'sum', 'Base_Res_SPH' : 'sum', 'Base_Res_WH' : 'sum',\
    'Base_Trans_HDV' : 'sum','Base_Trans_LDV' : 'sum', 'Base_Trans_MDV' : 'sum'})
        
    Res_WH = pd.read_parquet(path_result + "\Res_WH_By_region.parquet")
    Com_WH = pd.read_parquet(path_result + "\Com_WH_By_region.parquet")
    Res_SPH = pd.read_parquet(path_result + "\Res_SPH_By_region.parquet")
    Com_SPH = pd.read_parquet(path_result + "\Com_SPH_By_region.parquet")
    Trans_LDV = pd.read_parquet(path_result + "\Trans_LDV_By_region.parquet")
    Trans_MDV = pd.read_parquet(path_result + "\Trans_MDV_By_region.parquet")
    Trans_HDV = pd.read_parquet(path_result + "\Trans_HDV_By_region.parquet")
    #Trans_BUS = pd.read_parquet(path_result + "\Trans_BUS_By_region.parquet")

    Total_Load = pd.merge(Res_WH, Com_WH, on = ["SCENARIO","Year","LocalHourID","GenX.Region"])
    Total_Load = pd.merge(Total_Load, Res_SPH, on = ["SCENARIO","Year","LocalHourID","GenX.Region"])
    Total_Load = pd.merge(Total_Load, Com_SPH, on = ["SCENARIO","Year","LocalHourID","GenX.Region"])
    Total_Load = pd.merge(Total_Load, Trans_LDV, on = ["SCENARIO","Year","LocalHourID","GenX.Region"])
    Total_Load = pd.merge(Total_Load, Trans_MDV, on = ["SCENARIO","Year","LocalHourID","GenX.Region"])
    Total_Load = pd.merge(Total_Load, Trans_HDV, on = ["SCENARIO","Year","LocalHourID","GenX.Region"])
    #Total_Load = pd.merge(Total_Load, Trans_BUS, on = ["SCENARIO","Year","LocalHourID","GenX.Region"])
    Total_Load = pd.merge(Total_Load, Base_Load, on = ["Year","LocalHourID","GenX.Region"])
    Total_Load = Total_Load[Total_Load["Year"].isin(years)]
    del Base_Load, Res_WH, Com_WH, Res_SPH, Com_SPH, Trans_LDV, Trans_MDV, Trans_HDV

    Total_Load = Total_Load[(Total_Load["Year"].isin(years)) & (Total_Load["GenX.Region"].isin(regions))]
    Total_Load.to_parquet(path_result + "\Total_load_by_region_full.parquet", index = False)

    Total_Load_plot = Total_Load.assign(Total = Total_Load["Base_Com_SPH"] + Total_Load["Base_Res_SPH"] + Total_Load["Base_Res_WH"] + Total_Load["Base_Com_WH"] \
        + Total_Load["Base_Trans_LDV"] + Total_Load["Base_Trans_MDV"] + Total_Load["Base_Trans_HDV"] + Total_Load["Res_SPH_MW"] + Total_Load['Res_WH_MW']\
        + Total_Load['LDV_MW'] + Total_Load["MDV_MW"] + Total_Load["HDV_MW"] + Total_Load["Com_WH_MW"] + Total_Load["Com_SPH_MW"])\
            .melt(id_vars=["SCENARIO", "Year","LocalHourID","GenX.Region"])

    Total_Load_sum = Total_Load_plot.groupby(["SCENARIO", "Year", "GenX.Region", "variable"], as_index = False)["value"].sum()
    Total_Load_sum = Total_Load_sum.rename(columns={"value" : "AnnualTWh"})
    Total_Load_sum["AnnualTWh"] = round(10**-6*Total_Load_sum["AnnualTWh"], 2)
    del Total_Load_plot
    #Total_Load_sum_by_state = pd.merge(Total_Load_sum, pop, on = ["GenX.Region"], how = 'left')
    #Total_Load_sum_by_state = Total_Load_sum_by_state.assign(weighted = Total_Load_sum_by_state["AnnualTWh"] * Total_Load_sum_by_state["Zone Prop"])\
    #    .groupby(["SCENARIO", "State", "Year", "variable"])["weighted"].sum().rename(columns={"weighted" : "AnnualTWh"})
    #Total_Load_sum_by_state.to_parquet(path_result + "\Total_MWh_by_state.parquet", index = False)
    del Total_Load_sum

    
    #Total_Load = Total_Load.assign(Total_MW = Total_Load["water_heat_MW"] + Total_Load["space_heat_MW"] + Total_Load["LDEV_MW"] + Total_Load["MHBEV_MW"] + Total_Load["Base_MW"])\
    #        .drop(columns = ["water_heat_MW", "space_heat_MW", "LDEV_MW", "MHBEV_MW", "Base_MW", "SCENARIO", "Year"]) \
    #        .pivot_table(index = 'time_index', columns = 'region', values = 'Total_MW')

    return Total_Load


def MakeLoadProfiles(settings, case_folder):
    #path_processed = r"C:\Users\ritib\Dropbox\Project_LoadConstruction\data\processed"
    #path_result = r"C:\Users\ritib\Dropbox\Project_LoadConstruction\data\result"
    CreateOutputFolder(case_folder)
  
    output_folder = case_folder / "extra_outputs"

    years = []
    regions = []
    electrification = []
    # scenarios = pd.DataFrame()
    for year in settings:
        for cases, _settings in settings[year].items():
            years.append(_settings["model_year"])
            regions = regions +  regions_to_keep(_settings)[0]
            electrification.append(_settings["NZA_electrification"])
            try:
                path_stock = str(_settings["input_folder"]) + "\\" + _settings["custom_stock"]
            except TypeError:
                path_stock = ""
            # scenarios
    years = list(set(years))
    regions = list(set(regions))
    electrification = list(set(electrification))
    CreateBaseLoad(years, regions, output_folder)
    return AddElectrification(years, regions, electrification, output_folder, path_stock)

def FilterTotalProfile(settings, total_load):
    TotalLoad = total_load
    settings = settings
    TotalLoad = TotalLoad[TotalLoad["Year"] == settings["model_year"]]
    TotalLoad = TotalLoad.assign(TotalMW = TotalLoad['Res_WH_MW'] + TotalLoad['Com_WH_MW'] + TotalLoad['Res_SPH_MW'] + TotalLoad['Com_SPH_MW']\
        + TotalLoad['LDV_MW'] + TotalLoad['MDV_MW'] + TotalLoad['HDV_MW'] + TotalLoad['Base_Com_SPH'] + TotalLoad['Base_Com_WH']\
             + TotalLoad['Base_Res_SPH'] + TotalLoad['Base_Res_WH'] + TotalLoad['Base_Trans_HDV'] + TotalLoad['Base_Trans_LDV']\
                 + TotalLoad['Base_Trans_MDV']).drop(columns = ['Res_WH_MW', 'Com_WH_MW', 'Res_SPH_MW', 'Com_SPH_MW', 'LDV_MW', 'MDV_MW', 'HDV_MW',\
                    'Base_Com_SPH', 'Base_Com_WH', 'Base_Res_SPH', 'Base_Res_WH', 'Base_Trans_HDV', 'Base_Trans_LDV', 'Base_Trans_MDV'])
    TotalLoad = TotalLoad[TotalLoad["SCENARIO"] == settings["NZA_electrification"]]
    TotalLoad = TotalLoad.drop(columns = ["SCENARIO", "Year"]).rename(columns = {"LocalHourID" : "time_index", "GenX.Region" : "region"})

    
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
    #TotalLoad = TotalLoad.pivot_table(index = 'time_index', columns = 'region', values = 'TotalMW')
    return TotalLoad
