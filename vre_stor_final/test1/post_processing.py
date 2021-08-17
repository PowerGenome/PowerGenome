#!/usr/bin/env python
# coding: utf-8

# In[32]:


import pandas as pd
import numpy as np
import os
import yaml
from IPython import embed as IP
path = os.getcwd()
# In[33]:


year = [2030]

for y in year:
    path1 = path+'/'+str(y)+'/p1_2030_old_no_policy_mid/Inputs'


    # MID COST RESOURCES
    # List of PV/wind resources
    vre_mid = ['landbasedwind_ltrg1_mid_110','utilitypv_losangeles_mid_90_0_2','landbasedwind_ltrg1_mid_90','landbasedwind_ltrg1_mid_70', 'utilitypv_losangeles_mid_70_0_2','battery_mid']
    wind_mid_list = ['landbasedwind_ltrg1_mid_110','landbasedwind_ltrg1_mid_90','landbasedwind_ltrg1_mid_70']
    pv_mid_list = ['utilitypv_losangeles_mid_90_0_2','utilitypv_losangeles_mid_70_0_2']
    

    # GENERATORS DATA

    generator = pd.read_csv(os.path.join(path1, "Generators_data.csv"), header='infer', sep=',')
    
    # Get these PV/wind/battery generators
    indices_mid = np.where(generator.technology.isin(vre_mid)) 
    indices_pv = np.where(generator.technology.isin(pv_mid_list)) 
    mid_vre = generator.loc[generator.technology.isin(vre_mid)]

    # Get rid of these resources from generator_data.csv
    generator = generator.loc[~generator.technology.isin(vre_mid)]

    # Change New Build

    # Create dataframe
    vre_mid_list = pd.DataFrame()

    vre_mid_list["region"] = mid_vre["region"]
    vre_mid_list.loc[mid_vre.technology.isin(wind_mid_list), "technology"] = "hybrid_wind"
    vre_mid_list.loc[mid_vre.technology.isin(pv_mid_list), "technology"] = "hybrid_pv"
    vre_mid_list.loc[mid_vre.technology == "battery_mid", "technology"] = "standalone_storage"
    vre_mid_list["Resource_VRE"] = mid_vre["technology"]
    vre_mid_list["Resource_STOR"] = "hybrid_storage"
    vre_mid_list["Resource_GRID"] = "hybrid_grid"
    vre_mid_list["Zone"] = mid_vre["Zone"]
    vre_mid_list["cluster"] = mid_vre["cluster"] 
    vre_mid_list["ESR_1"] = mid_vre["ESR_1"] 
    vre_mid_list["ESR_2"] = mid_vre["ESR_2"]
    vre_mid_list["ESR_3"] = mid_vre["ESR_3"]
    vre_mid_list["ESR_4"] = mid_vre["ESR_4"]
    vre_mid_list["ESR_5"] = mid_vre["ESR_5"] 
    vre_mid_list["ESR_6"] = mid_vre["ESR_6"] 
    vre_mid_list["CapRes_1"] = mid_vre["CapRes_1"]
    vre_mid_list["CapRes_2"] = mid_vre["CapRes_2"] 
    vre_mid_list["CapRes_3"] = mid_vre["CapRes_3"]
    vre_mid_list["CapRes_4"] = mid_vre["CapRes_4"]  
    vre_mid_list["Existing_Cap_MW"] = 0
    vre_mid_list["Existing_Cap_MWh"] = 0
    vre_mid_list["Existing_Cap_Grid_MW"] = 0
    vre_mid_list["Min_Cap_VRE_MW"] = 0
    vre_mid_list.loc[mid_vre.technology != "battery_mid", "Max_Cap_VRE_MW"] = mid_vre["Max_Cap_MW"] * 1.3
    vre_mid_list["Min_Cap_Stor_MWh"] = 0
    vre_mid_list["Max_Cap_Stor_MWh"] = -1 # changes based upon scenario (standalone vs. hybrid)
    vre_mid_list["Min_Cap_Grid_MW"] = 0
    vre_mid_list["capex_VRE"] = mid_vre["capex_mw"]   # will change output post-process
    vre_mid_list["Inv_Cost_VRE_per_MWyr"] = mid_vre["Inv_Cost_per_MWyr"] # will change output post-process
    vre_mid_list["Fixed_OM_VRE_Cost_per_MWyr"] = mid_vre["Fixed_OM_Cost_per_MWyr"] # will change output post-process
    vre_mid_list["capex_mwh"] = mid_vre["capex_mwh"] # will change output post-process
    vre_mid_list["Inv_Cost_per_MWhyr"] = mid_vre["Inv_Cost_per_MWhyr"] # will change output post-process
    vre_mid_list["Fixed_OM_Cost_per_MWhyr"] = mid_vre["Fixed_OM_Cost_per_MWhyr"] # will change output post-process
    vre_mid_list["capex_GRID"] = 0 # will change output post-process
    vre_mid_list["Inv_Cost_GRID_per_MWyr"] = 0 # will change output post-process
    vre_mid_list["Fixed_OM_GRID_Cost_per_MWyr"] = 0 # will change output post-process
    vre_mid_list["Var_OM_Cost_per_MWh"] = 0
    vre_mid_list.loc[mid_vre.technology.isin(wind_mid_list), "Var_OM_Cost_per_MWh"] = -7.2
    vre_mid_list["Var_OM_Cost_per_MWh_In"] = 0 
    vre_mid_list["Fuel"] = "None"
    vre_mid_list["Heat_Rate_MMBTU_per_MWh"] = 0
    vre_mid_list["Self_Disch"] = 0.05
    vre_mid_list["Eff_Up"] = 0.95
    vre_mid_list["Eff_Down"] = 0.95
    vre_mid_list["EtaInverter"] = 0.9
    vre_mid_list["Inverter_Ratio"] = -1 # changes based upon scenario
    vre_mid_list["Power_To_Energy_Ratio"] = 0.25
    vre_mid_list["spur_line_costs"] = mid_vre["interconnect_annuity"]

    # NEED UNIQUE IDENTIFIERS
    nuclear_list = ['nuclear', 'nuclear_mid']
    generator.loc[(generator.region == 'CA_N')&(generator.Resource.isin(nuclear_list)), 'New_Build'] = 0
    generator.loc[(generator.region == 'CA_S')&(generator.Resource.isin(nuclear_list)), 'New_Build'] = 0
    generator.loc[generator.Resource == 'nuclear', 'New_Build'] = 0
    generator.to_csv(os.path.join(path1, "new_generators_data.csv"),encoding='utf-8',index=False)
    vre_mid_list.to_csv(os.path.join(path1, "vre_and_storage_data.csv"),encoding='utf-8',index=False)

    # GENERATORS VARIABILITY
    gen_var = pd.read_csv(os.path.join(path1, "Generators_variability.csv"), header='infer', sep=',')
    indices_mid = [x+1 for x in indices_mid]
    indices_pv = [x+1 for x in indices_pv]
    gen_var.iloc[:,indices_pv[0]] = gen_var.iloc[:,indices_pv[0]] / 0.9 / 1.34
    vre_variability = gen_var.iloc[:,indices_mid[0]]
    vre_variability.insert(loc=0, column='Time_Index', value=gen_var.iloc[:,0])
    vre_variability.to_csv(os.path.join(path1, "vre_and_storage_variability.csv"),encoding='utf-8',index=False)

    gen_var = gen_var.drop(gen_var.iloc[:,indices_mid[0]], axis = 1)
    gen_var.to_csv(os.path.join(path1, "new_generators_variability.csv"),encoding='utf-8',index=False)



    # LOW COSTS
    path2 = path+'/'+str(y)+'/p2_2030_old_no_policy_low/Inputs'


    # LOW COST RESOURCES
    # List of PV/wind resources
    vre_low = ['landbasedwind_ltrg1_low_110','utilitypv_losangeles_low_90_0_2','landbasedwind_ltrg1_low_90','landbasedwind_ltrg1_low_70','utilitypv_losangeles_low_70_0_2','battery_low']
    wind_low_list = ['landbasedwind_ltrg1_low_110','landbasedwind_ltrg1_low_90','landbasedwind_ltrg1_low_70']
    pv_low_list = ['utilitypv_losangeles_low_90_0_2','utilitypv_losangeles_low_70_0_2']
    

    generator = pd.read_csv(os.path.join(path2, "Generators_data.csv"), header='infer', sep=',')
    
    # Get these PV/wind/battery generators
    indices_low= np.where(generator.technology.isin(vre_low)) 
    indices_pv = np.where(generator.technology.isin(pv_low_list)) 
    low_vre = generator.loc[generator.technology.isin(vre_low)]

    # Get rid of these resources from generator_data.csv
    generator = generator.loc[~generator.technology.isin(vre_low)]

    # Create dataframe
    vre_low_list = pd.DataFrame()

    vre_low_list["region"] = low_vre["region"]
    vre_low_list.loc[low_vre.technology.isin(wind_low_list), "technology"] = "hybrid_wind"
    vre_low_list.loc[low_vre.technology.isin(pv_low_list), "technology"] = "hybrid_pv"
    vre_low_list.loc[low_vre.technology == "battery_low", "technology"] = "standalone_storage"
    vre_low_list["Resource_VRE"] = low_vre["technology"]
    vre_low_list["Resource_STOR"] = "hybrid_storage"
    vre_low_list["Resource_GRID"] = "hybrid_grid"
    vre_low_list["Zone"] = low_vre["Zone"]
    vre_low_list["cluster"] = low_vre["cluster"] 
    vre_low_list["ESR_1"] = low_vre["ESR_1"] 
    vre_low_list["ESR_2"] = low_vre["ESR_2"]
    vre_low_list["ESR_3"] = low_vre["ESR_3"]
    vre_low_list["ESR_4"] = low_vre["ESR_4"]
    vre_low_list["ESR_5"] = low_vre["ESR_5"] 
    vre_low_list["ESR_6"] = low_vre["ESR_6"] 
    vre_low_list["CapRes_1"] = low_vre["CapRes_1"]
    vre_low_list["CapRes_2"] = low_vre["CapRes_2"] 
    vre_low_list["CapRes_3"] = low_vre["CapRes_3"]
    vre_low_list["CapRes_4"] = low_vre["CapRes_4"]  
    vre_low_list["Existing_Cap_MW"] = 0
    vre_low_list["Existing_Cap_MWh"] = 0
    vre_low_list["Existing_Cap_Grid_MW"] = 0
    vre_low_list["Min_Cap_VRE_MW"] = 0
    vre_low_list.loc[low_vre.technology != "battery_low", "Max_Cap_VRE_MW"] = low_vre["Max_Cap_MW"] * 1.3
    vre_low_list["Min_Cap_Stor_MWh"] = 0
    vre_low_list["Max_Cap_Stor_MWh"] = -1 # changes based upon scenario (standalone vs. hybrid)
    vre_low_list["Min_Cap_Grid_MW"] = 0
    vre_low_list["capex_VRE"] = low_vre["capex_mw"] # will change output post-process
    vre_low_list["Inv_Cost_VRE_per_MWyr"] = low_vre["Inv_Cost_per_MWyr"] # will change output post-process
    vre_low_list["Fixed_OM_VRE_Cost_per_MWyr"] = low_vre["Fixed_OM_Cost_per_MWyr"] # will change output post-process
    vre_low_list["capex_mwh"] = low_vre["capex_mwh"] # will change output post-process
    vre_low_list["Inv_Cost_per_MWhyr"] = low_vre["Inv_Cost_per_MWhyr"] # will change output post-process
    vre_low_list["Fixed_OM_Cost_per_MWhyr"] = low_vre["Fixed_OM_Cost_per_MWhyr"] # will change output post-process
    vre_low_list["capex_GRID"] = 0 # will change output post-process
    vre_low_list["Inv_Cost_GRID_per_MWyr"] = 0 # will change output post-process
    vre_low_list["Fixed_OM_GRID_Cost_per_MWyr"] = 0 # will change output post-process
    vre_low_list["Var_OM_Cost_per_MWh"] = 0 
    vre_low_list.loc[low_vre.technology.isin(wind_low_list), "Var_OM_Cost_per_MWh"] = -7.2
    vre_low_list["Var_OM_Cost_per_MWh_In"] = 0 
    vre_low_list["Fuel"] = "None"
    vre_low_list["Heat_Rate_MMBTU_per_MWh"] = 0
    vre_low_list["Self_Disch"] = 0
    vre_low_list["Eff_Up"] = 0.95
    vre_low_list["Eff_Down"] = 0.95
    vre_low_list["EtaInverter"] = 0.9
    vre_low_list["Inverter_Ratio"] = -1 # changes based upon scenario
    vre_low_list["Power_To_Energy_Ratio"] = 0.25
    vre_low_list["spur_line_costs"] = low_vre["interconnect_annuity"]

    # NEED UNIQUE IDENTIFIERS
    nuclear_list = ['nuclear', 'nuclear_mid']
    generator.loc[(generator.region == 'CA_N')&(generator.Resource.isin(nuclear_list)), 'New_Build'] = 0
    generator.loc[(generator.region == 'CA_S')&(generator.Resource.isin(nuclear_list)), 'New_Build'] = 0
    generator.loc[generator.Resource == 'nuclear', 'New_Build'] = 0

    
    generator.to_csv(os.path.join(path2, "new_generators_data.csv"),encoding='utf-8',index=False)
    vre_low_list.to_csv(os.path.join(path2, "vre_and_storage_data.csv"),encoding='utf-8',index=False)

    # GENERATORS VARIABILITY
    gen_var = pd.read_csv(os.path.join(path2, "Generators_variability.csv"), header='infer', sep=',')
    indices_low = [x+1 for x in indices_low]
    indices_pv = [x+1 for x in indices_pv]
    gen_var.iloc[:,indices_pv[0]] = gen_var.iloc[:,indices_pv[0]] / 0.9 / 1.34
    vre_variability = gen_var.iloc[:,indices_low[0]]
    vre_variability.insert(loc=0, column='Time_Index', value=gen_var.iloc[:,0])
    vre_variability.to_csv(os.path.join(path2, "vre_and_storage_variability.csv"),encoding='utf-8',index=False)

    gen_var = gen_var.drop(gen_var.iloc[:,indices_low[0]], axis = 1)
    gen_var.to_csv(os.path.join(path2, "new_generators_variability.csv"),encoding='utf-8',index=False)
    