"""
Functions to fetch and modify NREL ATB data from PUDL
"""

import copy
import collections
import logging
import operator
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import sqlalchemy

from powergenome.params import CLUSTER_BUILDER, DATA_PATHS
from powergenome.price_adjustment import inflation_price_adjustment
from powergenome.resource_clusters import map_nrel_atb_technology
from powergenome.util import reverse_dict_of_lists

idx = pd.IndexSlice
logger = logging.getLogger(__name__)


def fetch_atb_costs(
    pudl_engine: sqlalchemy.engine.base.Engine,
    settings: dict,
    offshore_spur_costs: pd.DataFrame = None,
) -> pd.DataFrame:
    """Get NREL ATB power plant cost data from database, filter where applicable.

    This function can also remove NREL ATB offshore spur costs if more accurate costs
    will be included elsewhere (e.g. as part of total interconnection costs).

    Parameters
    ----------
    pudl_engine : sqlalchemy.Engine
        A sqlalchemy connection for use by pandas
    settings : dict
        User-defined parameters from a settings file. Needs to have keys
        `atb_data_year`, `atb_new_gen`, and `target_usd_year`. If the key
        `atb_financial_case` is not included, the default value will be "Market".
    offshore_spur_costs : pd.DataFrame
        An optional dataframe with spur costs for offshore wind resources. These costs
        are included in ATB for a fixed distance (same for all sites). PowerGenome
        interconnection costs for offshore sites include a spur cost calculated
        using actual distance from shore.

    Returns
    -------
    pd.DataFrame
        Power plant cost data with columns:
        ['technology', 'cap_recovery_years', 'cost_case', 'financial_case',
       'basis_year', 'tech_detail', 'fixed_o_m_mw', 'variable_o_m_mwh', 'capex', 'cf',
       'fuel', 'lcoe', 'wacc_real']
    """
    logger.info("Loading NREL ATB data")

    col_names = [
        "technology",
        "tech_detail",
        "cost_case",
        "parameter",
        "basis_year",
        "parameter_value",
        "dollar_year",
    ]
    atb_year = settings["atb_data_year"]
    fin_case = settings.get("atb_financial_case", "Market")

    # Fetch cost data from sqlite and create dataframe. Only get values for techs/cases
    # listed in the settings file.
    all_rows = []
    wacc_rows = []
    tech_list = []
    techs = settings["atb_new_gen"]
    mod_techs = []
    if settings.get("modified_atb_new_gen"):
        for _, m in settings.get("modified_atb_new_gen").items():
            mod_techs.append(
                [m["atb_technology"], m["atb_tech_detail"], m["atb_cost_case"], None]
            )

    cost_params = (
        "capex_mw",
        "fixed_o_m_mw",
        "variable_o_m_mwh",
        "capex_mwh",
        "fixed_o_m_mwh",
    )
    # add_pv_wacc = True
    for tech in techs + mod_techs:
        tech, tech_detail, cost_case, _ = tech
        # if tech == "UtilityPV":
        #     add_pv_wacc = False

        s = f"""
        SELECT technology, tech_detail, cost_case, parameter, basis_year, parameter_value, dollar_year
        from technology_costs_nrelatb
        where
            technology == "{tech}"
            AND tech_detail == "{tech_detail}"
            AND financial_case == "{fin_case}"
            AND cost_case == "{cost_case}"
            AND atb_year == {atb_year}
            AND parameter IN ({','.join('?'*len(cost_params))})
        """
        all_rows.extend(pudl_engine.execute(s, cost_params).fetchall())

        if tech not in tech_list:
            # ATB2020 summary file provides a single WACC for each technology and a single
            # tech detail of "*", so need to fetch this separately from other cost params.
            # Only need to fetch once per technology.
            wacc_s = f"""
            select technology, cost_case, basis_year, parameter_value
            from technology_costs_nrelatb
            where
                technology == "{tech}"
                AND financial_case == "{fin_case}"
                AND cost_case == "{cost_case}"
                AND atb_year == {atb_year}
                AND parameter == "wacc_real"
            """
            wacc_rows.extend(pudl_engine.execute(wacc_s).fetchall())

        tech_list.append(tech)

    if "Battery" not in tech_list:
        df = pd.DataFrame(all_rows, columns=col_names)
        wacc_df = pd.DataFrame(
            wacc_rows, columns=["technology", "cost_case", "basis_year", "wacc_real"]
        )
    else:
        # ATB doesn't have a WACC for battery storage. We use UtilityPV WACC as a default
        # stand-in -- make sure we have it in case.
        s = 'SELECT DISTINCT("technology") from technology_costs_nrelatb WHERE parameter == "wacc_real"'
        atb_techs = [x[0] for x in pudl_engine.execute(s).fetchall()]
        battery_wacc_standin = settings.get("atb_battery_wacc")
        battery_tech = [x for x in techs if x[0] == "Battery"][0]
        if isinstance(battery_wacc_standin, float):
            if battery_wacc_standin > 0.1:
                logger.warning(
                    f"You defined a battery WACC of {battery_wacc_standin}, which seems"
                    " very high. Check settings parameter `atb_battery_wacc`."
                )
            battery_wacc_rows = [
                (battery_tech[0], battery_tech[2], year, battery_wacc_standin)
                for year in range(2017, 2051)
            ]
            wacc_rows.extend(battery_wacc_rows)
        elif battery_wacc_standin in atb_techs:
            # if battery_wacc_standin in tech_list:
            #     pass
            # else:
            logger.info(
                f"Using {battery_wacc_standin} {fin_case} WACC for Battery storage."
            )
            wacc_s = f"""
            select technology, cost_case, basis_year, parameter_value
            from technology_costs_nrelatb
            where
                technology == "{battery_wacc_standin}"
                AND financial_case == "{fin_case}"
                AND cost_case == "Mid"
                AND atb_year == {atb_year}
                AND parameter == "wacc_real"

            """
            b_rows = pudl_engine.execute(wacc_s).fetchall()
            battery_wacc_rows = [
                (battery_tech[0], battery_tech[2], b_row[2], b_row[3])
                for b_row in b_rows
            ]
            wacc_rows.extend(battery_wacc_rows)
        else:
            raise ValueError(
                f"The settings key `atb_battery_wacc` value is {battery_wacc_standin}. It "
                f"should either be a float or a string from the list {atb_techs}."
            )

        df = pd.DataFrame(all_rows, columns=col_names)
        wacc_df = pd.DataFrame(
            wacc_rows, columns=["technology", "cost_case", "basis_year", "wacc_real"]
        )

    # Transform from tidy to wide dataframe, which makes it easier to fill generator
    # rows with the correct values.
    atb_costs = (
        df.drop_duplicates()
        .set_index(
            [
                "technology",
                "tech_detail",
                "cost_case",
                "dollar_year",
                "basis_year",
                "parameter",
            ]
        )
        .unstack(level=-1)
    )
    atb_costs.columns = atb_costs.columns.droplevel(0)
    atb_costs = (
        atb_costs.reset_index()
        .merge(wacc_df, on=["technology", "cost_case", "basis_year"], how="left")
        .drop_duplicates()
    )
    atb_costs = atb_costs.fillna(0)

    usd_columns = [
        "fixed_o_m_mw",
        "fixed_o_m_mwh",
        "variable_o_m_mwh",
        "capex_mw",
        "capex_mwh",
    ]
    for col in usd_columns:
        if col not in atb_costs.columns:
            atb_costs[col] = 0

    atb_target_year = settings["target_usd_year"]
    atb_costs[usd_columns] = atb_costs.apply(
        lambda row: inflation_price_adjustment(
            row[usd_columns], base_year=row["dollar_year"], target_year=atb_target_year
        ),
        axis=1,
    )

    if any("PV" in tech for tech in tech_list) and atb_year == 2019:
        print("Inflating ATB 2019 PV costs from DC to AC")
        atb_costs.loc[
            atb_costs["technology"].str.contains("PV"),
            ["capex_mw", "fixed_o_m_mw", "variable_o_m_mwh"],
        ] *= settings.get("pv_ac_dc_ratio", 1.3)
    elif atb_year > 2019:
        logger.info("PV costs are already in AC units, not inflating the cost.")

    if offshore_spur_costs is not None and "OffShoreWind" in atb_costs["technology"]:
        idx_cols = ["technology", "tech_detail", "cost_case", "basis_year"]
        offshore_spur_costs = offshore_spur_costs.set_index(idx_cols)
        atb_costs = atb_costs.set_index(idx_cols)

        atb_costs.loc[idx["OffShoreWind", :, :, :], "capex_mw"] = (
            atb_costs.loc[idx["OffShoreWind", :, :, :], "capex_mw"]  # .values
            - offshore_spur_costs.loc[
                idx["OffShoreWind", :, :, :], "capex_mw"
            ]  # .values
        )
        atb_costs = atb_costs.reset_index()

    return atb_costs


def fetch_atb_offshore_spur_costs(
    pudl_engine: sqlalchemy.engine.base.Engine, settings: dict
) -> pd.DataFrame:
    """Load offshore spur-line costs and convert to desired dollar-year.

    Parameters
    ----------
    pudl_engine : sqlalchemy.Engine
        A sqlalchemy connection for use by pandas
    settings : dict
        User-defined parameters from a settings file. Needs to have keys `atb_data_year`
        and `target_usd_year`.

    Returns
    -------
    pd.DataFrame
        Total offshore spur line capex from ATB for each technology/tech_detail/
        basis_year/cost_case combination.
    """
    spur_costs = pd.read_sql_table("offshore_spur_costs_nrelatb", pudl_engine)
    spur_costs = spur_costs.loc[spur_costs["atb_year"] == settings["atb_data_year"], :]

    atb_target_year = settings["target_usd_year"]

    spur_costs["capex_mw"] = spur_costs.apply(
        lambda row: inflation_price_adjustment(
            row["capex_mw"], base_year=row["dollar_year"], target_year=atb_target_year
        ),
        axis=1,
    )

    # ATB assumes a 30km distance for offshore spur. Normalize to per mile
    spur_costs["capex_mw_mile"] = spur_costs["capex_mw"] / 30 / 1.60934

    return spur_costs


def fetch_atb_heat_rates(
    pudl_engine: sqlalchemy.engine.base.Engine, settings: dict
) -> pd.DataFrame:
    """Get heat rate projections for power plants

    Data is originally from AEO, NREL does a linear interpolation between current and
    final years.

    Parameters
    ----------
    pudl_engine : sqlalchemy.Engine
        A sqlalchemy connection for use by pandas
    settings : dict
        User-defined parameters from a settings file. Needs to have key `atb_data_year`.

    Returns
    -------
    pd.DataFrame
        Power plant heat rate data by year with columns:
        ['technology', 'tech_detail', 'basis_year', 'heat_rate']
    """

    heat_rates = pd.read_sql_table("technology_heat_rates_nrelatb", pudl_engine)
    heat_rates = heat_rates.loc[heat_rates["atb_year"] == settings["atb_data_year"], :]

    return heat_rates


def atb_fixed_var_om_existing(
    results: pd.DataFrame,
    atb_hr_df: pd.DataFrame,
    settings: dict,
    pudl_engine: sqlalchemy.engine.base.Engine,
    coal_fgd_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add fixed and variable O&M for existing power plants

    ATB O&M data for new power plants are used as reference values. Fixed and variable
    O&M for each technology and heat rate are calculated. Assume that O&M scales with
    heat rate from new plants to existing generators. A separate multiplier for fixed
    O&M is specified in the settings file.

    Parameters
    ----------
    results : DataFrame
        Compiled results of clustered power plants with weighted average heat rates.
        Note that column names should include "technology", "Heat_rate_MMBTU_per_MWh",
        and "region". Technology names should not yet be converted to snake case.
    atb_hr_df : DataFrame
        Heat rate data from NREL ATB
    settings : dict
        User-defined parameters from a settings file

    Returns
    -------
    DataFrame
        Same as incoming "results" dataframe but with new columns
        "Fixed_OM_cost_per_MWyr" and "Var_OM_cost_per_MWh"
    """
    logger.info("Adding fixed and variable O&M for existing plants")

    existing_year = settings["atb_existing_year"]

    # ATB string is <technology>_<tech_detail>
    # techs = {eia: atb.split("_") for eia, atb in techs.items()}
    techs = {}
    for eia, atb in settings["eia_atb_tech_map"].items():
        if not isinstance(atb, list):
            atb = [atb]
        techs[eia] = atb[0].split("_")

    df_list = []
    grouped_results = results.reset_index().groupby(
        ["plant_id_eia", "technology"], as_index=False
    )
    for group, _df in grouped_results:

        plant_id, eia_tech = group
        existing_hr = _df["heat_rate_mmbtu_mwh"].mean()
        try:
            atb_tech, tech_detail = techs[eia_tech]
        except KeyError:
            if eia_tech in settings["tech_groups"]:
                raise KeyError(
                    f"{eia_tech} is defined in 'tech_groups' but doesn't have a "
                    "corresponding ATB technology in 'eia_atb_tech_map'"
                )

            else:
                raise KeyError(
                    f"{eia_tech} doesn't have a corresponding ATB technology in "
                    "'eia_atb_tech_map'"
                )

        try:
            new_build_hr = (
                atb_hr_df.query(
                    "technology==@atb_tech & tech_detail==@tech_detail"
                    "& basis_year==@existing_year"
                )
                .squeeze()
                .at["heat_rate"]
            )
        except ValueError:
            # Not all technologies have a heat rate. If they don't, just set both values
            # to 1
            existing_hr = 1
            new_build_hr = 1
        try:
            s = f"""
                select parameter_value
                from technology_costs_nrelatb
                where
                    technology == "{atb_tech}"
                    AND tech_detail == "{tech_detail}"
                    AND basis_year == "{existing_year}"
                    AND financial_case == "Market"
                    AND cost_case == "Mid"
                    AND atb_year == "{settings['atb_data_year']}"
                    AND parameter == "variable_o_m_mwh"

                """
            atb_var_om_mwh = pudl_engine.execute(s).fetchall()[0][0]
        except IndexError:
            # logger.warning(f"No variable O&M for {atb_tech}")
            atb_var_om_mwh = 0

        try:
            s = f"""
                select parameter_value
                from technology_costs_nrelatb
                where
                    technology == "{atb_tech}"
                    AND tech_detail == "{tech_detail}"
                    AND basis_year == "{existing_year}"
                    AND financial_case == "Market"
                    AND cost_case == "Mid"
                    AND atb_year == "{settings['atb_data_year']}"
                    AND parameter == "fixed_o_m_mw"

                """
            atb_fixed_om_mw_yr = pudl_engine.execute(s).fetchall()[0][0]
        except IndexError:
            # logger.warning(f"No fixed O&M for {atb_tech}")
            atb_fixed_om_mw_yr = 0

        nems_o_m_techs = [
            "Combined Cycle",
            "Combustion Turbine",
            "Coal",
            "Steam Turbine",
            "Hydroelectric",
            "Geothermal",
            "Nuclear",
        ]
        if any(t in eia_tech for t in nems_o_m_techs):
            # Change CC and CT O&M to EIA NEMS values, which are much higher for CCs and
            # lower for CTs than a heat rate & linear mulitpler correction to the ATB
            # values.
            # Add natural gas steam turbine O&M.
            # Also using the new values for coal plants, assuming 40-50 yr age and half
            # FGD
            # https://www.eia.gov/analysis/studies/powerplants/generationcost/pdf/full_report.pdf
            # logger.info(f"Using NEMS values for {eia_tech} fixed/variable O&M")
            target_usd_year = settings["target_usd_year"]
            simple_o_m = {
                "Combined Cycle": {
                    # "fixed_o_m_mw": inflation_price_adjustment(
                    #     13.08 * 1000, 2017, target_usd_year
                    # ),
                    # "variable_o_m_mwh": inflation_price_adjustment(
                    #     3.91, 2017, target_usd_year
                    # )
                },
                # "Combustion Turbine": {
                #     # This includes both the Fixed O&M and Capex. Capex includes
                #     # variable O&M, which is split out in the calculations below.
                #     "fixed_o_m_mw": inflation_price_adjustment(
                #         (5.33 + 6.90) * 1000, 2017, target_usd_year
                #     ),
                #     "variable_o_m_mwh": 0,
                # },
                "Natural Gas Steam Turbine": {
                    # NEMS documenation splits capex and fixed O&M across 2 tables
                    # "fixed_o_m_mw": inflation_price_adjustment(
                    #     (15.96 + 24.68) * 1000, 2017, target_usd_year
                    # ),
                    "variable_o_m_mwh": inflation_price_adjustment(
                        1.0, 2017, target_usd_year
                    )
                },
                "Coal": {
                    # "fixed_o_m_mw": inflation_price_adjustment(
                    #     ((22.2 + 27.88) / 2 + 46.01) * 1000, 2017, target_usd_year
                    # ),
                    "variable_o_m_mwh": inflation_price_adjustment(
                        1.78, 2017, target_usd_year
                    )
                },
                "Conventional Hydroelectric": {
                    "fixed_o_m_mw": inflation_price_adjustment(
                        44.56 * 1000, 2017, target_usd_year
                    ),
                    "variable_o_m_mwh": 0,
                },
                "Geothermal": {
                    "fixed_o_m_mw": inflation_price_adjustment(
                        198.04 * 1000, 2017, target_usd_year
                    ),
                    "variable_o_m_mwh": 0,
                },
                "Pumped Hydro": {
                    "fixed_o_m_mw": inflation_price_adjustment(
                        (23.63 + 14.83) * 1000, 2017, target_usd_year
                    ),
                    "variable_o_m_mwh": 0,
                },
            }

            if "Combined Cycle" in eia_tech:
                # https://www.eia.gov/analysis/studies/powerplants/generationcost/pdf/full_report.pdf
                plant_capacity = _df[settings["capacity_col"]].sum()
                assert plant_capacity > 0
                if plant_capacity < 500:
                    fixed = 15.62 * 1000
                    variable = 4.31
                elif 500 <= plant_capacity < 1000:
                    fixed = 9.27 * 1000
                    variable = 3.42
                else:
                    fixed = 11.68 * 1000
                    variable = 3.37

                _df["Fixed_OM_cost_per_MWyr"] = inflation_price_adjustment(
                    fixed, 2017, target_usd_year
                )
                _df["Var_OM_cost_per_MWh"] = inflation_price_adjustment(
                    variable, 2017, target_usd_year
                )

            if "Combustion Turbine" in eia_tech:
                # need to adjust the EIA fixed/variable costs because they have no
                # variable cost per MWh for existing CTs but they do have per MWh for
                # new build. Assume $11/MWh from new-build and 4% CF:
                # (11*8760*0.04/1000)=$3.85/kW-yr. Scale the new-build variable
                # (~$11/MWh) by relative heat rate and subtract a /kW-yr value as
                # calculated above from the FOM.
                # Based on conversation with Jesse J. on Dec 20, 2019.
                plant_capacity = _df[settings["capacity_col"]].sum()
                op, op_value = (
                    settings.get("atb_modifiers", {})
                    .get("ngct", {})
                    .get("Var_OM_cost_per_MWh", (None, None))
                )

                if op:
                    f = operator.attrgetter(op)
                    atb_var_om_mwh = f(operator)(atb_var_om_mwh, op_value)

                variable = atb_var_om_mwh  # * (existing_hr / new_build_hr)

                if plant_capacity < 100:
                    annual_capex = 9.0 * 1000
                    fixed = annual_capex + 5.96 * 1000
                elif 100 <= plant_capacity <= 300:
                    annual_capex = 6.18 * 1000
                    fixed = annual_capex + 6.43 * 1000
                else:
                    annual_capex = 6.95 * 1000
                    fixed = annual_capex + 3.99 * 1000

                fixed = fixed - (variable * 8760 * 0.04)

                _df["Fixed_OM_cost_per_MWyr"] = inflation_price_adjustment(
                    fixed, 2017, target_usd_year
                )
                _df["Var_OM_cost_per_MWh"] = inflation_price_adjustment(
                    variable, 2017, target_usd_year
                )

            if "Natural Gas Steam Turbine" in eia_tech:
                # https://www.eia.gov/analysis/studies/powerplants/generationcost/pdf/full_report.pdf
                plant_capacity = _df[settings["capacity_col"]].sum()
                assert plant_capacity > 0
                if plant_capacity < 500:
                    annual_capex = 18.86 * 1000
                    fixed = annual_capex + 29.73 * 1000
                elif 500 <= plant_capacity < 1000:
                    annual_capex = 11.57 * 1000
                    fixed = annual_capex + 17.98 * 1000
                else:
                    annual_capex = 10.82 * 1000
                    fixed = annual_capex + 14.51 * 1000

                _df["Fixed_OM_cost_per_MWyr"] = inflation_price_adjustment(
                    fixed, 2017, target_usd_year
                )
                _df["Var_OM_cost_per_MWh"] = simple_o_m["Natural Gas Steam Turbine"][
                    "variable_o_m_mwh"
                ]

            if "Coal" in eia_tech:

                plant_capacity = _df[settings["capacity_col"]].sum()
                assert plant_capacity > 0

                age = settings["model_year"] - _df.operating_date.dt.year
                age = age.fillna(age.mean())
                age = age.fillna(40)
                gen_ids = _df["generator_id"].to_list()
                fgd = coal_fgd_df.query(
                    "plant_id_eia == @plant_id & generator_id in @gen_ids"
                )["fgd"].values
                if not np.any(fgd):
                    gen_ids = [g.lstrip("0") for g in gen_ids]
                    fgd = coal_fgd_df.query(
                        "plant_id_eia == @plant_id & generator_id in @gen_ids"
                    )["fgd"].values

                # https://www.eia.gov/analysis/studies/powerplants/generationcost/pdf/full_report.pdf
                annual_capex = (16.53 + (0.126 * age) + (5.68 * fgd)) * 1000

                if plant_capacity < 500:
                    fixed = 44.21 * 1000
                elif 500 <= plant_capacity < 1000:
                    fixed = 34.02 * 1000
                elif 1000 <= plant_capacity < 2000:
                    fixed = 28.52 * 1000
                else:
                    fixed = 33.27 * 1000

                _df["Fixed_OM_cost_per_MWyr"] = inflation_price_adjustment(
                    fixed + annual_capex, 2017, target_usd_year
                )
                _df["Var_OM_cost_per_MWh"] = simple_o_m["Coal"]["variable_o_m_mwh"]
            if "Hydroelectric" in eia_tech:
                _df["Fixed_OM_cost_per_MWyr"] = simple_o_m[
                    "Conventional Hydroelectric"
                ]["fixed_o_m_mw"]
                _df["Var_OM_cost_per_MWh"] = simple_o_m["Conventional Hydroelectric"][
                    "variable_o_m_mwh"
                ]
            if "Geothermal" in eia_tech:
                _df["Fixed_OM_cost_per_MWyr"] = simple_o_m["Geothermal"]["fixed_o_m_mw"]
                _df["Var_OM_cost_per_MWh"] = simple_o_m["Geothermal"][
                    "variable_o_m_mwh"
                ]
            if "Pumped" in eia_tech:
                _df["Fixed_OM_cost_per_MWyr"] = simple_o_m["Pumped Hydro"][
                    "fixed_o_m_mw"
                ]
                _df["Var_OM_cost_per_MWh"] = simple_o_m["Pumped Hydro"][
                    "variable_o_m_mwh"
                ]
            if "Nuclear" in eia_tech:
                num_units = len(_df)
                plant_capacity = _df[settings["capacity_col"]].sum()

                # Operating costs for different size/num units in 2016 INL report
                # "Economic and Market Challenges Facing the U.S. Nuclear Fleet"
                # https://gain.inl.gov/Shared%20Documents/Economics-Nuclear-Fleet.pdf,
                # table 1. Average of the two costs are used in each case.
                # The costs in that report include fuel and VOM. Assume $0.66/mmbtu
                # and $2.32/MWh plus 90% CF (ATB 2020) to get the costs below.
                # The INL report doesn't give a dollar year for costs, assume 2015.
                if num_units == 1 and plant_capacity < 900:
                    fixed = 315000
                elif num_units == 1 and plant_capacity >= 900:
                    fixed = 252000
                else:
                    fixed = 177000
                # age = (settings["model_year"] - _df.operating_date.dt.year).values
                # age = age.fillna(age.mean())
                # age = age.fillna(40)
                # EIA, 2020, "Assumptions to Annual Energy Outlook, Electricity Market Module,"
                # Available: https://www.eia.gov/outlooks/aeo/assumptions/pdf/electricity.pdf
                # fixed = np.ones_like(age)
                # fixed[age < 30] *= 27 * 1000
                # fixed[age >= 30] *= (27+37) * 1000

                _df["Fixed_OM_cost_per_MWyr"] = inflation_price_adjustment(
                    fixed, 2015, target_usd_year
                )
                _df["Var_OM_cost_per_MWh"] = atb_var_om_mwh * (
                    existing_hr / new_build_hr
                )

        else:
            _df["Fixed_OM_cost_per_MWyr"] = atb_fixed_om_mw_yr
            _df["Var_OM_cost_per_MWh"] = atb_var_om_mwh * (existing_hr / new_build_hr)

        df_list.append(_df)

    mod_results = pd.concat(df_list, ignore_index=True)
    # mod_results = mod_results.sort_values(["model_region", "technology", "cluster"])
    mod_results.loc[:, "Fixed_OM_cost_per_MWyr"] = mod_results.loc[
        :, "Fixed_OM_cost_per_MWyr"
    ].astype(int)
    mod_results.loc[:, "Var_OM_cost_per_MWh"] = mod_results.loc[
        :, "Var_OM_cost_per_MWh"
    ]

    return mod_results


def single_generator_row(
    atb_costs_hr: pd.DataFrame,
    new_gen_type: str,
    model_year_range: Union[Tuple[int], List[int]],
) -> pd.DataFrame:
    """Create a data row with NREL ATB costs and performace for a single technology

    Parameters
    ----------
    atb_costs : pd.DataFrame
        Data from the sqlite tables of both resources costs and heat rates
    new_gen_type : str
        type of generating resource
    model_year_range : Union[Tuple[int], List[int]]
        All of the years that should be averaged over

    Returns
    -------
    pd.DataFrame
        A single row dataframe with average cost and performence values over the study
        period.
    """

    technology, tech_detail, cost_case, size_mw = new_gen_type
    numeric_cols = [
        "basis_year",
        "fixed_o_m_mw",
        "fixed_o_m_mwh",
        "variable_o_m_mwh",
        "capex_mw",
        "capex_mwh",
        # "cf",
        # "fuel",
        # "lcoe",
        # "o_m",
        "wacc_real",
        "heat_rate",
    ]
    s = atb_costs_hr.loc[
        (atb_costs_hr["technology"] == technology)
        & (atb_costs_hr["tech_detail"] == tech_detail)
        & (atb_costs_hr["basis_year"].isin(model_year_range)),
        numeric_cols,
    ].mean()
    cols = ["technology", "cost_case", "tech_detail"] + numeric_cols
    row = pd.DataFrame([technology, cost_case, tech_detail] + s.to_list(), index=cols).T

    row["Cap_size"] = size_mw

    return row


def investment_cost_calculator(capex, wacc, cap_rec_years):
    capex = np.asarray(capex, dtype=float)
    wacc = np.asarray(wacc, dtype=float)

    for variable in capex, wacc, cap_rec_years:
        if np.isnan(variable).any():
            raise ValueError(f"Investment costs contains nan values")

    inv_cost = capex * (
        np.exp(wacc * cap_rec_years)
        * (np.exp(wacc) - 1)
        / (np.exp(wacc * cap_rec_years) - 1)
    )

    return inv_cost


def regional_capex_multiplier(df, region, region_map, tech_map, regional_multipliers):

    cost_region = region_map[region]
    tech_multiplier = regional_multipliers.loc[cost_region, :].squeeze()
    avg_multiplier = tech_multiplier.mean()

    tech_multiplier = tech_multiplier.fillna(avg_multiplier)

    tech_multiplier_map = {}
    for atb_tech, eia_tech in tech_map.items():
        if df["technology"].str.contains(atb_tech).sum() > 0:
            full_atb_tech = df.loc[
                df["technology"].str.contains(atb_tech).idxmax(), "technology"
            ]
            tech_multiplier_map[full_atb_tech] = tech_multiplier.at[eia_tech]

    df["Inv_cost_per_MWyr"] *= df["technology"].map(tech_multiplier_map)
    df["Inv_cost_per_MWhyr"] *= df["technology"].map(tech_multiplier_map)
    df["regional_cost_multiplier"] = df["technology"].map(tech_multiplier_map)

    return df


def add_modified_atb_generators(
    settings: dict,
    atb_costs_hr: pd.DataFrame,
    model_year_range: Union[Tuple[int], List[int]],
) -> pd.DataFrame:
    """Create a modified version of an ATB generator.

    For each parameter (capex, heat_rate, etc) that users want modified they should
    specify a list of [<operator>, <value>]. The operator can be add, mul, truediv, or
    sub (substract). This is used to modify individual parameters of the ATB resource.

    Parameters
    ----------
    settings : dict
        User-defined parameters from a settings file
    atb_costs_hr : pd.DataFrame
        Cost and heat rate data for ATB resources
    model_year_range : Union[Tuple[int], List[int]]
        A list or range of years to average ATB values from.

    Returns
    -------
    pd.DataFrame
        Row or rows of modified ATB resources. Each row includes the colums:
        ['technology', 'cost_case', 'tech_detail', 'basis_year', 'fixed_o_m_mw',
       'fixed_o_m_mwh', 'variable_o_m_mwh', 'capex', 'capex_mwh', 'cf', 'fuel',
       'lcoe', 'o_m', 'wacc_real', 'heat_rate', 'Cap_size'].
    """

    # copy settings so popped keys aren't removed permenantly
    _settings = copy.deepcopy(settings)

    allowed_operators = ["add", "mul", "truediv", "sub"]

    mod_tech_list = []
    for name, mod_tech in _settings["modified_atb_new_gen"].items():
        atb_technology = mod_tech.pop("atb_technology")
        atb_tech_detail = mod_tech.pop("atb_tech_detail")
        atb_cost_case = mod_tech.pop("atb_cost_case")
        size_mw = mod_tech.pop("size_mw")

        new_gen_type = (atb_technology, atb_tech_detail, atb_cost_case, size_mw)

        gen = single_generator_row(atb_costs_hr, new_gen_type, model_year_range)
        gen["technology"] = mod_tech.pop("new_technology")
        gen["tech_detail"] = mod_tech.pop("new_tech_detail", "")
        gen["cost_case"] = mod_tech.pop("new_cost_case")

        for parameter, op_list in mod_tech.items():
            assert len(op_list) == 2, (
                "Two values, an operator and a numeric value, are needed in the parameter\n"
                f"'{parameter}' for technology '{name}' in 'modified_atb_new_gen'."
            )
            op, op_value = op_list

            assert parameter in gen.columns, (
                f"'{parameter}' is not a valid parameter for new resources. Check '{name}'\n"
                "in 'modified_atb_new_gen' of the settings file."
            )
            assert op in allowed_operators, (
                f"The key {parameter} for technology {name} needs a valid operator from the list\n"
                f"{allowed_operators}\n"
                "in the format [<operator>, <value>] to modify the properties of an existing generator.\n"
            )

            f = operator.attrgetter(op)
            gen[parameter] = f(operator)(gen[parameter], op_value)

        mod_tech_list.append(gen)

    mod_gens = pd.concat(mod_tech_list, ignore_index=True)

    return mod_gens


def atb_new_generators(atb_costs, atb_hr, settings):
    """Add rows for new generators in each region

    Parameters
    ----------
    atb_costs : DataFrame
        All cost parameters from the SQL table for new generators. Should include:
        ['technology', 'cost_case', 'financial_case', 'basis_year', 'tech_detail',
        'capex', 'capex_mwh', 'fixed_o_m_mw', 'fixed_o_m_mwh', 'variable_o_m_mwh',
        'wacc_real']
    atb_hr : DataFrame
        The technology, tech_detail, and heat_rate of new generators from ATB.
    settings : dict
        User-defined parameters from a settings file

    Returns
    -------
    DataFrame
        New generating resources in every region. Contains the columns:
        ['technology', 'basis_year', 'Fixed_OM_cost_per_MWyr',
       'Fixed_OM_cost_per_MWhyr', 'Var_OM_cost_per_MWh', 'capex', 'capex_mwh',
       'Inv_cost_per_MWyr', 'Inv_cost_per_MWhyr', 'Heat_rate_MMBTU_per_MWh',
       'Cap_size', 'region']
    """
    logger.info("Creating new resources for each region.")
    new_gen_types = settings["atb_new_gen"]
    model_year = settings["model_year"]
    try:
        first_planning_year = settings["model_first_planning_year"]
        model_year_range = range(first_planning_year, model_year + 1)
    except KeyError:
        model_year_range = list(range(model_year + 1))

    regions = settings["model_regions"]

    atb_costs_hr = atb_costs.merge(
        atb_hr, on=["technology", "tech_detail", "basis_year"], how="left"
    )

    new_gen_df = pd.concat(
        [
            single_generator_row(atb_costs_hr, new_gen, model_year_range)
            for new_gen in new_gen_types
        ],
        ignore_index=True,
    )
    # Add user-defined technologies
    # This should probably be separate from ATB techs, and the regional cost multipliers
    # should be its own function.
    if settings.get("additional_technologies_fn"):
        if isinstance(settings.get("additional_new_gen"), list):
            # user_costs, user_hr = load_user_defined_techs(settings)
            user_tech = load_user_defined_techs(settings)
            # new_gen_df = pd.concat([new_gen_df, user_costs], ignore_index=True, sort=False)
            new_gen_df = pd.concat(
                [new_gen_df, user_tech], ignore_index=True, sort=False
            )
            # atb_hr = pd.concat([atb_hr, user_hr], ignore_index=True, sort=False)
        else:
            logger.warning(
                "A filename for additional technologies was included but no technologies"
                " were specified in the settings file."
            )

    if settings.get("modified_atb_new_gen"):
        modified_gens = add_modified_atb_generators(
            settings, atb_costs_hr, model_year_range
        )
        new_gen_df = pd.concat(
            [new_gen_df, modified_gens], ignore_index=True, sort=False
        )

    new_gen_df = new_gen_df.rename(
        columns={
            "heat_rate": "Heat_rate_MMBTU_per_MWh",
            "fixed_o_m_mw": "Fixed_OM_cost_per_MWyr",
            "fixed_o_m_mwh": "Fixed_OM_cost_per_MWhyr",
            "variable_o_m_mwh": "Var_OM_cost_per_MWh",
        }
    )

    # Adjust values for CT/CC generators to match advanced techs in NEMS rather than
    # ATB average of advanced and conventional.
    # This is now generalized for changes to ATB values for any technology type.
    for tech, _tech_modifiers in (settings.get("atb_modifiers") or {}).items():
        tech_modifiers = copy.deepcopy(_tech_modifiers)
        assert isinstance(tech_modifiers, dict), (
            "The settings parameter 'atb_modifiers' must be a nested list.\n"
            "Each top-level key is a short name of the technology, with a nested"
            " dictionary of items below it."
        )
        assert (
            "technology" in tech_modifiers
        ), "Each nested dictionary in atb_modifiers must have a 'technology' key."
        assert (
            "tech_detail" in tech_modifiers
        ), "Each nested dictionary in atb_modifiers must have a 'tech_detail' key."

        technology = tech_modifiers.pop("technology")
        tech_detail = tech_modifiers.pop("tech_detail")

        allowed_operators = ["add", "mul", "truediv", "sub"]

        for key, op_list in tech_modifiers.items():

            assert len(op_list) == 2, (
                "Two values, an operator and a numeric value, are needed in the parameter\n"
                f"'{key}' for technology '{tech}' in 'atb_modifiers'."
            )
            op, op_value = op_list

            assert op in allowed_operators, (
                f"The key {key} for technology {tech} needs a valid operator from the list\n"
                f"{allowed_operators}\n"
                "in the format [<operator>, <value>] to modify the properties of an existing generator.\n"
            )

            f = operator.attrgetter(op)
            new_gen_df.loc[
                (new_gen_df.technology == technology)
                & (new_gen_df.tech_detail == tech_detail),
                key,
            ] = f(operator)(
                new_gen_df.loc[
                    (new_gen_df.technology == technology)
                    & (new_gen_df.tech_detail == tech_detail),
                    key,
                ],
                op_value,
            )

    new_gen_df["technology"] = (
        new_gen_df[["technology", "tech_detail", "cost_case"]]
        .astype(str)
        .agg("_".join, axis=1)
    )

    new_gen_df["cap_recovery_years"] = settings["atb_cap_recovery_years"]

    for tech, years in (settings.get("alt_atb_cap_recovery_years") or {}).items():
        new_gen_df.loc[
            new_gen_df["technology"].str.lower().str.contains(tech.lower()),
            "cap_recovery_years",
        ] = years

    new_gen_df["Inv_cost_per_MWyr"] = investment_cost_calculator(
        capex=new_gen_df["capex_mw"],
        wacc=new_gen_df["wacc_real"],
        cap_rec_years=new_gen_df["cap_recovery_years"],
    )

    new_gen_df["Inv_cost_per_MWhyr"] = investment_cost_calculator(
        capex=new_gen_df["capex_mwh"],
        wacc=new_gen_df["wacc_real"],
        cap_rec_years=new_gen_df["cap_recovery_years"],
    )

    keep_cols = [
        "technology",
        "basis_year",
        "Fixed_OM_cost_per_MWyr",
        "Fixed_OM_cost_per_MWhyr",
        "Var_OM_cost_per_MWh",
        "capex_mw",
        "capex_mwh",
        "Inv_cost_per_MWyr",
        "Inv_cost_per_MWhyr",
        "Heat_rate_MMBTU_per_MWh",
        "Cap_size",
        "cap_recovery_years",
        "wacc_real",
    ]
    new_gen_df = new_gen_df[keep_cols]
    # Set no capacity limit on new resources that aren't renewables.
    new_gen_df["Max_Cap_MW"] = -1

    regional_cost_multipliers = pd.read_csv(
        DATA_PATHS["cost_multipliers"] / "AEO_2020_regional_cost_corrections.csv",
        index_col=0,
    )
    if settings.get("user_regional_cost_multiplier_fn"):
        user_cost_multipliers = pd.read_csv(
            Path(settings["input_folder"])
            / settings["user_regional_cost_multiplier_fn"],
            index_col=0,
        )
        regional_cost_multipliers = pd.concat(
            [regional_cost_multipliers, user_cost_multipliers], axis=1
        )
    rev_mult_region_map = reverse_dict_of_lists(settings["cost_multiplier_region_map"])
    rev_mult_tech_map = reverse_dict_of_lists(
        settings["cost_multiplier_technology_map"]
    )
    df_list = []
    for region in regions:
        _df = new_gen_df.copy()
        _df["region"] = region
        _df = regional_capex_multiplier(
            _df,
            region,
            rev_mult_region_map,
            rev_mult_tech_map,
            regional_cost_multipliers,
        )
        _df = add_renewables_clusters(_df, region, settings)

        if region in (settings.get("new_gen_not_available") or {}):
            techs = settings["new_gen_not_available"][region]
            for tech in techs:
                _df = _df.loc[~_df["technology"].str.contains(tech), :]

        df_list.append(_df)

    results = pd.concat(df_list, ignore_index=True, sort=False)

    int_cols = [
        "Fixed_OM_cost_per_MWyr",
        "Fixed_OM_cost_per_MWhyr",
        "Inv_cost_per_MWyr",
        "Inv_cost_per_MWhyr",
        "cluster",
    ]
    results = results.fillna(0)
    results[int_cols] = results[int_cols].astype(int)
    results["Var_OM_cost_per_MWh"] = results["Var_OM_cost_per_MWh"].astype(float)

    return results


def add_renewables_clusters(
    df: pd.DataFrame, region: str, settings: dict
) -> pd.DataFrame:
    """
    Add renewables clusters

    Parameters
    ----------
    df
        New generation technologies.
            - `technology`: NREL ATB technology in the format
                <technology>_<tech_detail>_<cost_case>. Must be unique.
            - `region`: Model region.
    region
        Model region.
    settings
        Dictionary with the following keys:
            - `renewables_clusters`: Determines the clusters built for the region.
            - `region_aggregations`: Maps the model region to IPM regions.

    Returns
    -------
    pd.DataFrame
        Copy of the input dataframe joined to rows for renewables clusters
        on matching NREL ATB technology and model region.

    Raises
    ------
    ValueError
        NREL ATB technologies are not unique.
    ValueError
        Renewables clusters do not match NREL ATB technologies.
    ValueError
        Renewables clusters match multiple NREL ATB technologies.
    """
    if not df["technology"].is_unique:
        raise ValueError(
            f"NREL ATB technologies are not unique: {df['technology'].to_list()}"
        )
    atb_map = {
        x: map_nrel_atb_technology(x.split("_")[0], x.split("_")[1])
        for x in df["technology"]
    }
    mask = df["technology"].isin([tech for tech, match in atb_map.items() if match]) & (
        df["region"] == region
    )
    cdfs = []
    if region in settings.get("region_aggregations", {}):
        ipm_regions = settings.get("region_aggregations", {})[region]
    else:
        ipm_regions = [region]
    for scenario in settings.get("renewables_clusters", []):
        if scenario["region"] != region:
            continue
        # Match cluster technology to NREL ATB technologies
        technologies = [
            k
            for k, v in atb_map.items()
            if v and all([scenario.get(ki) == vi for ki, vi in v.items()])
        ]
        if not technologies:
            raise ValueError(
                f"Renewables clusters do not match NREL ATB technologies: {scenario}"
            )
        if len(technologies) > 1:
            raise ValueError(
                f"Renewables clusters match multiple NREL ATB technologies: {scenario}"
            )
        technology = technologies[0]
        # region not an argument to ClusterBuilder.get_clusters()
        scenario = scenario.copy()
        scenario.pop("region")
        clusters = (
            CLUSTER_BUILDER.get_clusters(
                **scenario,
                ipm_regions=ipm_regions,
                existing=False,
                utc_offset=settings.get("utc_offset", 0),
            )
            .rename(columns={"mw": "Max_Cap_MW"})
            .assign(technology=technology, region=region)
        )
        clusters["cluster"] = range(1, 1 + len(clusters))
        if scenario.get("min_capacity"):
            # Warn if total capacity less than expected
            capacity = clusters["Max_Cap_MW"].sum()
            if capacity < scenario["min_capacity"]:
                logger.warning(
                    f"Selected technology {scenario['technology']} capacity"
                    + f" in region {region}"
                    + f" less than minimum ({capacity} < {scenario['min_capacity']} MW)"
                )
        row = df[df["technology"] == technology].to_dict("records")[0]
        new_tech_name = "_".join(
            [
                str(v)
                for k, v in scenario.items()
                if k not in ["region", "technology", "max_clusters", "min_capacity"]
            ]
        )
        clusters["technology"] = clusters["technology"] + "_" + new_tech_name
        kwargs = {k: v for k, v in row.items() if k not in clusters}
        cdfs.append(clusters.assign(**kwargs))
    return pd.concat([df[~mask]] + cdfs, sort=False)


def load_user_defined_techs(settings: dict) -> pd.DataFrame:
    """Load user-defined technologies from a CSV file. Returns cost columns and heat
    rate.

    Parameters
    ----------
    settings : dict
        User-defined parameters from a settings file. It must have the key
        'additional_technologies_fn'. The value can either be a string (name of a single
        file) or a dictionary. If the value is a dictionary it should have integer keys
        corresponding to model years and corresponding string values (file name).

        settings['additional_technologies_fn'] = 'user_techs.csv'
        OR
        settings['additional_technologies_fn'] = {
            2030: 'user_techs_2030.csv',
            2045: 'user_techs_2045.csv'
        }

    Returns
    -------
    pd.DataFrame
        A dataframe of user-defined resources with cost and heat rate columns.
    """
    if isinstance(settings["additional_technologies_fn"], collections.abc.Mapping):
        fn = settings["additional_technologies_fn"][settings["model_year"]]
    else:
        fn = settings["additional_technologies_fn"]

    # Search the extra inputs folder first, then the legacy additional_techs folder
    # in repo
    if (Path(settings["input_folder"]) / fn).exists():
        user_techs = pd.read_csv(Path(settings["input_folder"]) / fn)
    else:
        logger.warning(
            "The file with your user defined technologies is not in the user input "
            "folder. Reading the file from PowerGenome/data/additional_technolgies "
            "instead. This may be depreciated in a future version, please move "
            f"{fn} to the folder {settings['input_folder']}."
        )
        user_techs = pd.read_csv(DATA_PATHS["additional_techs"] / fn)

    user_techs = user_techs.loc[
        (user_techs["technology"].isin(settings["additional_new_gen"]))
        & (user_techs["planning_year"] == settings["model_year"]),
        :,
    ]

    user_techs = user_techs.fillna(0)

    if "tech_detail" not in user_techs.columns:
        user_techs["tech_detail"] = ""
    if "cost_case" not in user_techs.columns:
        user_techs["cost_case"] = ""
    if "Cap_size" not in user_techs.columns:
        user_techs["Cap_size"] = 1

    if "dollar_year" in user_techs.columns:
        for idx, row in user_techs.iterrows():
            for col in [
                "capex_mw",
                "capex_mwh",
                "fixed_o_m_mw",
                "fixed_o_m_mwh",
                "variable_o_m_mwh",
            ]:
                user_techs.loc[idx, col] = inflation_price_adjustment(
                    row[col], row["dollar_year"], settings["target_usd_year"]
                )

    cols = [
        "technology",
        "tech_detail",
        "cost_case",
        "capex_mw",
        "capex_mwh",
        "fixed_o_m_mw",
        "fixed_o_m_mwh",
        "variable_o_m_mwh",
        "wacc_real",
        "heat_rate",
        "Cap_size",
        "dollar_year",
    ]

    return user_techs[cols]
