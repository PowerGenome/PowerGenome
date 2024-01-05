"""
Functions to fetch and modify NREL ATB data from PUDL
"""

import collections
import copy
import logging
import operator
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import sqlalchemy
from joblib import Parallel, delayed

from powergenome.cluster.renewables import assign_site_cluster, calc_cluster_values
from powergenome.financials import investment_cost_calculator
from powergenome.params import DATA_PATHS, SETTINGS, build_resource_clusters
from powergenome.price_adjustment import inflation_price_adjustment
from powergenome.resource_clusters import (
    ClusterBuilder,
    ResourceGroup,
    Table,
    map_nrel_atb_technology,
)
from powergenome.util import (
    apply_all_tag_to_regions,
    remove_leading_zero,
    reverse_dict_of_lists,
    snake_case_col,
    snake_case_str,
)

idx = pd.IndexSlice
logger = logging.getLogger(__name__)


def fetch_atb_costs(
    pg_engine: sqlalchemy.engine.base.Engine,
    settings: dict,
    offshore_spur_costs: pd.DataFrame = None,
) -> pd.DataFrame:
    """Get NREL ATB power plant cost data from database, filter where applicable.

    This function can also remove NREL ATB offshore spur costs if more accurate costs
    will be included elsewhere (e.g. as part of total interconnection costs).

    Parameters
    ----------
    pg_engine : sqlalchemy.Engine
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
    cols = ["technology", "tech_detail", "financial_case", "cost_case", "atb_year"]
    valid_inputs = db_col_values(pg_engine, "technology_costs_nrelatb", cols)
    for tech in techs + mod_techs:
        tech, tech_detail, cost_case, _ = tech
        # if tech == "UtilityPV":
        #     add_pv_wacc = False
        tech_params = [tech, tech_detail, fin_case, cost_case, atb_year]
        for param in tech_params:
            if param not in valid_inputs:
                raise ValueError(
                    f"When getting technology costs, the parameter {param} does not have "
                    "a valid matching value in the database table."
                )
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
        all_rows.extend(pg_engine.execute(s, cost_params).fetchall())

        if (tech, cost_case) not in tech_list:
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
            wacc_rows.extend(pg_engine.execute(wacc_s).fetchall())

        tech_list.append((tech, cost_case))
    tech_names = [t[0] for t in tech_list]
    if "Battery" not in tech_names:
        df = pd.DataFrame(all_rows, columns=col_names)
        wacc_df = pd.DataFrame(
            wacc_rows, columns=["technology", "cost_case", "basis_year", "wacc_real"]
        )
    else:
        # ATB doesn't have a WACC for battery storage. We use UtilityPV WACC as a default
        # stand-in -- make sure we have it in case.
        s = 'SELECT DISTINCT("technology") from technology_costs_nrelatb WHERE parameter == "wacc_real"'
        atb_techs = [x[0] for x in pg_engine.execute(s).fetchall()]
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
            for cost_case in ["Mid", "Moderate"]:
                wacc_s = f"""
                select technology, cost_case, basis_year, parameter_value
                from technology_costs_nrelatb
                where
                    technology == "{battery_wacc_standin}"
                    AND financial_case == "{fin_case}"
                    AND cost_case == "{cost_case}"
                    AND atb_year == {atb_year}
                    AND parameter == "wacc_real"

                """
                b_rows = pg_engine.execute(wacc_s).fetchall()
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
    if not atb_costs.empty:
        atb_costs[usd_columns] = atb_costs.apply(
            lambda row: inflation_price_adjustment(
                row[usd_columns],
                base_year=row["dollar_year"],
                target_year=atb_target_year,
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


def db_col_values(
    engine: sqlalchemy.engine, table: str, cols: List[str]
) -> List[Union[str, int]]:
    """Find all distinct values in one or more columns of a database table.

    This function can be used to check user inputs that are applied as filters against
    existing table values to reduce the risk of SQL injection attacks. All distinct
    values are returned in a single list for simplicity.

    Parameters
    ----------
    engine : sqlalchemy.engine
        Connection engine to the database.
    table : str
        Name of the database table.
    cols : List[str]
        Name of the column(s) to check.

    Returns
    -------
    List[Union[str, int]]
        All distinct values from the database table column(s).
    """
    valid_inputs = []
    for col in cols:
        s = f"SELECT DISTINCT {col} from {table}"
        valid_inputs.extend(pd.read_sql_query(s, engine)[col].to_list())

    return valid_inputs


def fetch_atb_offshore_spur_costs(
    pg_engine: sqlalchemy.engine.base.Engine, settings: dict
) -> pd.DataFrame:
    """Load offshore spur-line costs and convert to desired dollar-year.

    Parameters
    ----------
    pg_engine : sqlalchemy.Engine
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
    spur_costs = pd.read_sql_table("offshore_spur_costs_nrelatb", pg_engine)
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
    pg_engine: sqlalchemy.engine.base.Engine, settings: dict
) -> pd.DataFrame:
    """Get heat rate projections for power plants

    Data is originally from AEO, NREL does a linear interpolation between current and
    final years.

    Parameters
    ----------
    pg_engine : sqlalchemy.Engine
        A sqlalchemy connection for use by pandas
    settings : dict
        User-defined parameters from a settings file. Needs to have key `atb_data_year`.

    Returns
    -------
    pd.DataFrame
        Power plant heat rate data by year with columns:
        ['technology', 'tech_detail', 'cost_case', 'basis_year', 'heat_rate']
    """

    heat_rates = pd.read_sql_table("technology_heat_rates_nrelatb", pg_engine)
    if settings["atb_data_year"] not in heat_rates["atb_year"].unique():
        max_atb_year = heat_rates["atb_year"].max()
        logger.warning(
            f"Your settings file has parameter `atb_year` of {settings['atb_data_year']}"
            ", which isn't in the table `technology_heat_rates_nrelatb`. Using "
            f"{max_atb_year} instead."
        )
        settings["atb_data_year"] = max_atb_year
    heat_rates = heat_rates.loc[heat_rates["atb_year"] == settings["atb_data_year"], :]

    return heat_rates


def atb_fixed_var_om_existing(
    results: pd.DataFrame,
    atb_hr_df: pd.DataFrame,
    settings: dict,
    pg_engine: sqlalchemy.engine.base.Engine,
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
        Note that column names should include "technology", "Heat_Rate_MMBTU_per_MWh",
        and "region". Technology names should not yet be converted to snake case.
    atb_hr_df : DataFrame
        Heat rate data from NREL ATB
    settings : dict
        User-defined parameters from a settings file

    Returns
    -------
    DataFrame
        Same as incoming "results" dataframe but with new columns
        "Fixed_OM_Cost_per_MWyr" and "Var_OM_Cost_per_MWh"
    """
    logger.info("Adding fixed and variable O&M for existing plants")

    existing_year = settings["atb_existing_year"]

    techs = {}
    missing_techs = []
    for eia, atb in settings["eia_atb_tech_map"].items():
        if not isinstance(atb, list):
            atb = [atb]
        missing = True
        for tech_detail in atb:
            tech, detail = tech_detail.split("_")
            if not atb_hr_df.query(
                "technology == @tech and tech_detail == @detail"
            ).empty:
                techs[eia] = [tech, detail]
                missing = False
                break
        if missing is True and eia in results["technology"].unique():
            missing_techs.append(eia)

        # techs[eia] = atb[0].split("_")
    if missing_techs:
        s = (
            f"The EIA technologies {missing_techs} do not have an ATB counterpart with a "
            "valid heat rate. Not all ATB technologies *should* have a valid heat rate "
            "(e.g. wind, solar, and hydro). Check the 'eia_atb_tech_map' parameter in your "
            "settings file(s) if you think one of these technologies should be mapped to "
            "an ATB technology with a valid heat rate."
        )
        logger.warning(s)

    # Find valid ATB tech/tech_details with O&M costs where heat rate was missing.
    s = """
            SELECT
            technology,
            tech_detail
        FROM
            technology_costs_nrelatb
        WHERE
            basis_year == ?
            AND financial_case == "Market"
            AND cost_case in("Mid", "Moderate")
            AND atb_year == ?
            AND parameter in("variable_o_m_mwh", "fixed_o_m_mw")
    """
    params = [existing_year, settings["atb_data_year"]]
    atb_om_names = pd.read_sql_query(
        s,
        pg_engine,
        params=params,
    ).drop_duplicates()
    _missing_techs = missing_techs.copy()
    for eia_tech in missing_techs:
        atb = settings["eia_atb_tech_map"][eia_tech]
        if not isinstance(atb, list):
            atb = [atb]
        missing = True
        for tech_detail in atb:
            tech, detail = tech_detail.split("_")
            if (
                not atb_om_names.query(
                    "technology == @tech and tech_detail == @detail"
                ).empty
                and missing is True
            ):
                techs[eia_tech] = [tech, detail]
                missing = False
                # break
        if missing is False:
            _missing_techs.remove(eia_tech)
        elif missing is True and eia_tech in results["technology"].unique():
            techs[eia_tech] = atb[0].split("_")
        # else:
        #     techs[eia_tech] = atb[0].split("_")
    if _missing_techs:
        s = (
            f"The EIA technologies {_missing_techs} do not have an ATB counterpart with "
            "valid fixed or variable O&M costs. All ATB technologies *should* have valid "
            "fixed/variable O&M costs. Check the 'eia_atb_tech_map' parameter in your "
            "settings file(s)."
        )
        logger.warning(s)

    target_usd_year = settings["target_usd_year"]
    simple_o_m = {
        "Natural Gas Steam Turbine": {
            "variable_o_m_mwh": inflation_price_adjustment(1.0, 2017, target_usd_year)
        },
        "Coal": {
            "variable_o_m_mwh": inflation_price_adjustment(1.78, 2017, target_usd_year)
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

    # Load all atb O&M values at once rather than querying the db thousands of times
    nems_o_m_techs = [
        "Combined Cycle",
        "Combustion Turbine",
        "Coal",
        "Steam Turbine",
        "Hydroelectric",
        "Geothermal",
    ]
    atb_techs = [
        (tech, tech_detail)
        for k, (tech, tech_detail) in techs.items()
        if tech not in nems_o_m_techs
    ]
    s = f"""
        select technology, tech_detail, parameter, AVG(parameter_value) as parameter_value
        from technology_costs_nrelatb
        where
            basis_year == ?
            AND financial_case == "Market"
            AND cost_case in ("Mid", "Moderate")
            AND atb_year == ?
            AND parameter in ("variable_o_m_mwh", "fixed_o_m_mw", "fixed_o_m_mwh")
            AND
                ({' OR '.join(["(technology==? and tech_detail==?)"]*len(atb_techs))})
            GROUP BY technology, tech_detail, parameter
        """
    params = [existing_year, settings["atb_data_year"]] + [
        item for sublist in atb_techs for item in sublist
    ]
    atb_om = pd.read_sql_query(
        s,
        pg_engine,
        params=params,
        index_col=["technology", "tech_detail", "parameter"],
    )
    atb_hr_df = atb_hr_df.set_index(
        ["technology", "tech_detail", "basis_year"]
    ).sort_index()

    df_list = []
    grouped_results = results.reset_index().groupby(["technology"], as_index=False)
    for group, _df in grouped_results:
        _df = calc_om(
            _df,
            atb_hr_df,
            atb_om,
            settings,
            pg_engine,
            coal_fgd_df,
            existing_year,
            techs,
            simple_o_m,
            group,
        )

        df_list.append(_df)

    mod_results = pd.concat(df_list, ignore_index=True)

    # Fill na FOM values, first by technology and then across all techs
    if not mod_results.loc[mod_results["Fixed_OM_Cost_per_MWyr"].isna()].empty:
        df_list = []
        for tech, _df in mod_results.groupby("technology"):
            _df.loc[
                _df["Fixed_OM_Cost_per_MWyr"].isna(), "Fixed_OM_Cost_per_MWyr"
            ] = _df["Fixed_OM_Cost_per_MWyr"].mean()
            df_list.append(_df)
        mod_results = pd.concat(df_list, ignore_index=True)
        mod_results.loc[
            mod_results["Fixed_OM_Cost_per_MWyr"].isna(), "Fixed_OM_Cost_per_MWyr"
        ] = mod_results["Fixed_OM_Cost_per_MWyr"].mean()
    mod_results.loc[:, "Fixed_OM_Cost_per_MWyr"] = mod_results.loc[
        :, "Fixed_OM_Cost_per_MWyr"
    ].astype(int)
    mod_results.loc[:, "Fixed_OM_Cost_per_MWhyr"] = mod_results.loc[
        :, "Fixed_OM_Cost_per_MWhyr"
    ].astype(int)
    mod_results.loc[:, "Var_OM_Cost_per_MWh"] = mod_results.loc[
        :, "Var_OM_Cost_per_MWh"
    ]

    return mod_results


def calc_om(
    df: pd.DataFrame,
    atb_hr_df: pd.DataFrame,
    atb_om_df: pd.DataFrame,
    settings: dict,
    pg_engine: sqlalchemy.engine.base.Engine,
    coal_fgd_df: pd.DataFrame,
    existing_year: int,
    techs: Dict[str, List[str]],
    simple_o_m: Dict[str, float],
    group: str,
) -> pd.DataFrame:
    """Calculate fixed and variable O&M for a single technology.

    Parameters
    ----------
    df : pd.DataFrame
        Units of generators at a single plant
    atb_hr_df : pd.DataFrame
        Heat rate data from NREL ATB
    atb_om_df : pd.DataFrame
        O&M costs from NREL ATB
    settings : dict
        User-defined parameters from a settings file
    pg_engine : sqlalchemy.engine.base.Engine
        Connection to the PowerGenome database
    coal_fgd_df : pd.DataFrame
        Table showing if a coal generating unit has FGD control technology
    existing_year : int
        Year of ATB to use as a proxy for existing plants
    techs : Dict[str, List[str]]
        Mapping of EIA technology name (key) to the ATB technology, tech_detail (value)
    simple_o_m : Dict[str, float]
        Mapping of simple O&M costs used by some generators (not dependent on size/age)
    group : str
        The EIA technology name

    Returns
    -------
    pd.DataFrame
        Modified copy of input with fixed and variable O&M costs

    Raises
    ------
    KeyError
        _description_
    KeyError
        _description_
    """
    df["Fixed_OM_Cost_per_MWhyr"] = 0
    target_usd_year = settings["target_usd_year"]
    eia_tech = group

    try:
        atb_tech, tech_detail = techs[eia_tech]
    except KeyError:
        if eia_tech in settings.get("tech_groups", {}) or {}:
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
        new_build_hr = atb_hr_df.loc[
            (atb_tech, tech_detail, existing_year),
            "heat_rate",
        ].mean()
        if not isinstance(new_build_hr, float):
            logger.warning(
                "\n\n****************\nCAUTION!!!\n\n"
                f"The calculated new build heat rate for {atb_tech}, {tech_detail} "
                f"should be a single value but is {new_build_hr}. This could cause "
                f"issues with your variable O&M costs for {eia_tech}. Please report "
                "this as an issue on the PowerGenome repository.\n"
            )
    except (ValueError, TypeError, KeyError):
        # Not all technologies have a heat rate. If they don't, just set both values
        # to 10.34 (33% efficiency)
        df.loc[df["heat_rate_mmbtu_mwh"].isna(), "heat_rate_mmbtu_mwh"] = 10.34
        new_build_hr = 10.34

    try:
        atb_var_om_mwh = atb_om_df.loc[
            (atb_tech, tech_detail, "variable_o_m_mwh"), "parameter_value"
        ]
    except KeyError:
        atb_var_om_mwh = 0

    nems_o_m_techs = [
        "Combined Cycle",
        "Combustion Turbine",
        "Coal",
        "Steam Turbine",
        # "Hydroelectric",
        # "Geothermal",
        "Nuclear",
    ]

    if any(t in eia_tech for t in nems_o_m_techs):
        df_list = []
        for plant_id, _df in df.groupby("plant_id_eia", as_index=False):
            # Change CC and CT O&M to EIA NEMS values, which are much higher for CCs and
            # lower for CTs than a heat rate & linear mulitpler correction to the ATB
            # values.
            # Add natural gas steam turbine O&M.
            # Also using the new values for coal plants, assuming 40-50 yr age and half
            # FGD
            # https://www.eia.gov/analysis/studies/powerplants/generationcost/pdf/full_report.pdf
            # logger.info(f"Using NEMS values for {eia_tech} fixed/variable O&M")

            if "Combined Cycle" in eia_tech:
                # https://www.eia.gov/analysis/studies/powerplants/generationcost/pdf/full_report.pdf
                assert not _df[settings["capacity_col"]].isnull().all()
                plant_capacity = _df[settings["capacity_col"]].sum()
                if plant_capacity < 500:
                    fixed = 15.62 * 1000
                    variable = 4.31
                elif 500 <= plant_capacity < 1000:
                    fixed = 9.27 * 1000
                    variable = 3.42
                else:
                    fixed = 11.68 * 1000
                    variable = 3.37

                _df["Fixed_OM_Cost_per_MWyr"] = inflation_price_adjustment(
                    fixed, 2017, target_usd_year
                )
                _df["Var_OM_Cost_per_MWh"] = inflation_price_adjustment(
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
                    (settings.get("atb_modifiers", {}) or {})
                    .get("ngct", {})
                    .get("Var_OM_Cost_per_MWh", (None, None))
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

                _df["Fixed_OM_Cost_per_MWyr"] = inflation_price_adjustment(
                    fixed, 2017, target_usd_year
                )
                _df["Var_OM_Cost_per_MWh"] = inflation_price_adjustment(
                    variable, 2017, target_usd_year
                )

            if "Natural Gas Steam Turbine" in eia_tech:
                # https://www.eia.gov/analysis/studies/powerplants/generationcost/pdf/full_report.pdf
                assert not _df[settings["capacity_col"]].isnull().all()
                plant_capacity = _df[settings["capacity_col"]].sum()
                if plant_capacity < 500:
                    annual_capex = 18.86 * 1000
                    fixed = annual_capex + 29.73 * 1000
                elif 500 <= plant_capacity < 1000:
                    annual_capex = 11.57 * 1000
                    fixed = annual_capex + 17.98 * 1000
                else:
                    annual_capex = 10.82 * 1000
                    fixed = annual_capex + 14.51 * 1000

                _df["Fixed_OM_Cost_per_MWyr"] = inflation_price_adjustment(
                    fixed, 2017, target_usd_year
                )
                _df["Var_OM_Cost_per_MWh"] = simple_o_m["Natural Gas Steam Turbine"][
                    "variable_o_m_mwh"
                ]

            if "Coal" in eia_tech:
                assert not _df[settings["capacity_col"]].isnull().all()
                plant_capacity = _df[settings["capacity_col"]].sum()

                age = settings["model_year"] - _df.operating_date.dt.year
                try:
                    age = age.fillna(age.mean())
                except:
                    age = age.fillna(40)
                gen_ids = _df["generator_id"].to_list()
                fgd = coal_fgd_df.query(
                    "plant_id_eia == @plant_id & generator_id in @gen_ids"
                )["fgd"].values
                if not np.any(fgd):
                    gen_ids = [remove_leading_zero(g) for g in gen_ids]
                    fgd = coal_fgd_df.query(
                        "plant_id_eia == @plant_id & generator_id in @gen_ids"
                    )["fgd"].values
                if not np.any(fgd):
                    # If FGD isn't found, use average of with/without FGD
                    fgd = np.ones_like(age) * 0.5

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

                _df["Fixed_OM_Cost_per_MWyr"] = inflation_price_adjustment(
                    fixed + annual_capex, 2017, target_usd_year
                )
                _df["Var_OM_Cost_per_MWh"] = simple_o_m["Coal"]["variable_o_m_mwh"]

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

                _df["Fixed_OM_Cost_per_MWyr"] = inflation_price_adjustment(
                    fixed, 2015, target_usd_year
                )

                # If nuclear heat rates are NaN, set them to new build value
                _df.loc[
                    _df["heat_rate_mmbtu_mwh"].isna(), "heat_rate_mmbtu_mwh"
                ] = new_build_hr
                _df["Var_OM_Cost_per_MWh"] = atb_var_om_mwh * (
                    _df["heat_rate_mmbtu_mwh"].mean() / new_build_hr
                )

            df_list.append(_df)

        return pd.concat(df_list, ignore_index=True)

    elif "Hydroelectric" in eia_tech:
        df["Fixed_OM_Cost_per_MWyr"] = simple_o_m["Conventional Hydroelectric"][
            "fixed_o_m_mw"
        ]
        df["Var_OM_Cost_per_MWh"] = simple_o_m["Conventional Hydroelectric"][
            "variable_o_m_mwh"
        ]
    elif "Geothermal" in eia_tech:
        df["Fixed_OM_Cost_per_MWyr"] = simple_o_m["Geothermal"]["fixed_o_m_mw"]
        df["Var_OM_Cost_per_MWh"] = simple_o_m["Geothermal"]["variable_o_m_mwh"]
    elif "Pumped" in eia_tech:
        df["Fixed_OM_Cost_per_MWyr"] = simple_o_m["Pumped Hydro"]["fixed_o_m_mw"]
        df["Var_OM_Cost_per_MWh"] = simple_o_m["Pumped Hydro"]["variable_o_m_mwh"]
    else:
        try:
            atb_fixed_om_mw_yr = atb_om_df.loc[
                (atb_tech, tech_detail, "fixed_o_m_mw"), "parameter_value"
            ]
        except KeyError:
            atb_fixed_om_mw_yr = 0
        df["Fixed_OM_Cost_per_MWyr"] = atb_fixed_om_mw_yr
        df["Var_OM_Cost_per_MWh"] = atb_var_om_mwh * (
            df["heat_rate_mmbtu_mwh"] / new_build_hr
        )
    if atb_tech == "Battery":
        atb_fixed_om_mwh = atb_om_df.loc[
            (atb_tech, tech_detail, "fixed_o_m_mwh"), "parameter_value"
        ]
        df["Fixed_OM_Cost_per_MWhyr"] = atb_fixed_om_mwh

    return df


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
        "wacc_real",
        "heat_rate",
    ]
    s = atb_costs_hr.loc[
        (atb_costs_hr["technology"] == technology)
        & (atb_costs_hr["tech_detail"] == tech_detail)
        & (atb_costs_hr["cost_case"] == cost_case)
        & (atb_costs_hr["basis_year"].isin(model_year_range)),
        numeric_cols,
    ].mean()
    cols = ["technology", "cost_case", "tech_detail"] + numeric_cols
    row = pd.DataFrame([technology, cost_case, tech_detail] + s.to_list(), index=cols).T

    row["Cap_Size"] = size_mw

    return row


def regional_capex_multiplier(
    df: pd.DataFrame,
    region: str,
    region_map: Dict[str, str],
    tech_map: Dict[str, str],
    regional_multipliers: pd.DataFrame,
) -> pd.DataFrame:
    cost_region = region_map[region]
    tech_multiplier = regional_multipliers.loc[cost_region, :].squeeze()
    avg_multiplier = tech_multiplier.mean()

    tech_multiplier = tech_multiplier.fillna(avg_multiplier)

    tech_multiplier_map = {}
    for atb_tech, eia_tech in tech_map.items():
        if df["technology"].str.contains(atb_tech, case=False).sum() > 0:
            full_atb_tech = df.loc[
                df["technology"].str.contains(atb_tech, case=False).idxmax(),
                "technology",
            ]
            tech_multiplier_map[full_atb_tech] = tech_multiplier.at[eia_tech]
        if df["technology"].str.contains(atb_tech).sum() > 1:
            s = f"""
    ***************************
    There is an issue with assigning regional cost multipliers. In your settings file
    under the parameter 'cost_multiplier_technology_map`, the EIA technology '{eia_tech}'
    has an ATB technology '{atb_tech}'. This ATB name matches more than one new ATB tech
    listed in the settings parameter 'atb_new_gen'. Only the first matching tech in
    'atb_new_gen' will get a valid regional cost multiplier; the rest will have values of
    0, which will lead to annual investment costs of $0.
        """
            logger.warning(s)
    df["Inv_Cost_per_MWyr"] *= df["technology"].map(tech_multiplier_map)
    df["Inv_Cost_per_MWhyr"] *= df["technology"].map(tech_multiplier_map)
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
       'lcoe', 'o_m', 'wacc_real', 'heat_rate', 'Cap_Size'].
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
            if isinstance(op_list, float) | isinstance(op_list, int):
                gen[parameter] = op_list
            else:
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


def atb_new_generators(atb_costs, atb_hr, settings, cluster_builder=None):
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
    cluster_builder : ClusterBuilder
        ClusterBuilder object. Reuse to save time. None by default.

    Returns
    -------
    DataFrame
        New generating resources in every region. Contains the columns:
        ['technology', 'basis_year', 'Fixed_OM_Cost_per_MWyr',
       'Fixed_OM_Cost_per_MWhyr', 'Var_OM_Cost_per_MWh', 'capex', 'capex_mwh',
       'Inv_Cost_per_MWyr', 'Inv_Cost_per_MWhyr', 'Heat_Rate_MMBTU_per_MWh',
       'Cap_Size', 'region']
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
        atb_hr, on=["technology", "tech_detail", "cost_case", "basis_year"], how="left"
    )

    if new_gen_types:
        new_gen_df = pd.concat(
            [
                single_generator_row(atb_costs_hr, new_gen, model_year_range)
                for new_gen in new_gen_types
            ],
            ignore_index=True,
        )
    else:
        new_gen_df = pd.DataFrame(
            columns=["region", "technology", "tech_detail", "cost_case"]
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
            "heat_rate": "Heat_Rate_MMBTU_per_MWh",
            "fixed_o_m_mw": "Fixed_OM_Cost_per_MWyr",
            "fixed_o_m_mwh": "Fixed_OM_Cost_per_MWhyr",
            "variable_o_m_mwh": "Var_OM_Cost_per_MWh",
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
            if isinstance(op_list, float) | isinstance(op_list, int):
                new_gen_df.loc[
                    (new_gen_df.technology == technology)
                    & (new_gen_df.tech_detail == tech_detail),
                    key,
                ] = op_list
            else:
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

    if new_gen_df.empty:
        results = new_gen_df.copy()
    else:
        for tech, years in (settings.get("alt_atb_cap_recovery_years") or {}).items():
            new_gen_df.loc[
                new_gen_df["technology"].str.lower().str.contains(tech.lower()),
                "cap_recovery_years",
            ] = years

        new_gen_df["Inv_Cost_per_MWyr"] = investment_cost_calculator(
            capex=new_gen_df["capex_mw"],
            wacc=new_gen_df["wacc_real"],
            cap_rec_years=new_gen_df["cap_recovery_years"],
            compound_method=settings.get("interest_compound_method", "discrete"),
        )

        new_gen_df["Inv_Cost_per_MWhyr"] = investment_cost_calculator(
            capex=new_gen_df["capex_mwh"],
            wacc=new_gen_df["wacc_real"],
            cap_rec_years=new_gen_df["cap_recovery_years"],
            compound_method=settings.get("interest_compound_method", "discrete"),
        )

        # Set no capacity limit on new resources that aren't renewables.
        new_gen_df["Max_Cap_MW"] = -1
        new_gen_df["Max_Cap_MWh"] = -1
        regional_cost_multipliers = pd.read_csv(
            DATA_PATHS["cost_multipliers"]
            / settings.get(
                "cost_multiplier_fn", "AEO_2020_regional_cost_corrections.csv"
            ),
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
        rev_mult_region_map = reverse_dict_of_lists(
            settings["cost_multiplier_region_map"]
        )
        rev_mult_tech_map = reverse_dict_of_lists(
            settings["cost_multiplier_technology_map"]
        )

        df_list = []
        settings = apply_all_tag_to_regions(settings)

        df_list = Parallel(n_jobs=settings.get("clustering_n_jobs", 1))(
            delayed(parallel_region_renewables)(
                settings,
                new_gen_df,
                regional_cost_multipliers,
                rev_mult_region_map,
                rev_mult_tech_map,
                region,
                cluster_builder,
            )
            for region in regions
        )

        results = pd.concat(df_list, ignore_index=True, sort=False)

        int_cols = [
            "Fixed_OM_Cost_per_MWyr",
            "Fixed_OM_Cost_per_MWhyr",
            "Inv_Cost_per_MWyr",
            "Inv_Cost_per_MWhyr",
            "cluster",
        ]
        int_cols = [c for c in int_cols if c in results.columns]
        results = results.fillna(0)
        results[int_cols] = results[int_cols].astype(int)
        results["Var_OM_Cost_per_MWh"] = results["Var_OM_Cost_per_MWh"].astype(float)

    return results


def parallel_region_renewables(
    settings: dict,
    new_gen_df: pd.DataFrame,
    regional_cost_multipliers: pd.DataFrame,
    rev_mult_region_map: Dict[str, List[str]],
    rev_mult_tech_map: Dict[str, List[str]],
    region: str,
    cluster_builder: ClusterBuilder = None,
) -> pd.DataFrame:
    """Wrapper function to run regional capex and add renewable clusters in parallel

    Parameters
    ----------
    settings : dict
        Can have keys "renewables_clusters" and "region_aggregations"
    new_gen_df : pd.DataFrame
        Rows are new-build resources specified by the user
    regional_cost_multipliers : pd.DataFrame
        Cost multiplier for each technology type in different regions
    rev_mult_region_map : Dict[str, List[str]]
        Mapping of cost regions to model regions
    rev_mult_tech_map : Dict[str, List[str]]
        Mapping of technologies from cost map to technologies in new_gen_df
    region : str
        Name of the model region
    cluster_builder
        ClusterBuilder object. Reuse to save time. None by default.

    Returns
    -------
    pd.DataFrame
        New-build resources in a single region. Includes the regionally corrected cost
        and renewable resource clusters as specified by the user.
    """
    _df = new_gen_df.copy()
    _df["region"] = region
    _df = regional_capex_multiplier(
        _df,
        region,
        rev_mult_region_map,
        rev_mult_tech_map,
        regional_cost_multipliers,
    )
    _df = add_renewables_clusters(
        _df,
        region,
        settings,
        cluster_builder,
    )

    return _df


def load_resource_group_data(
    rg: ResourceGroup, cache=True
) -> Tuple[pd.DataFrame, Union[pd.Series, None]]:
    """Load metadata for the specified resource group.

    Metadata is information on individual renewable sites such as the site ID, capacity,
    interconnection cost, etc. If the resource group has the attribute "site_map", then
    a mapping of site IDs to generation profile IDs is also returned.

    Parameters
    ----------
    rg : ResourceGroup
        A resource group object
    cache : bool, optional
        A flag indicating whether to cache the data, by default True

    Returns
    -------
    Tuple[pd.DataFrame, Union[pd.Series, None]]
        A tuple of the metadata dataframe and the site map as a Series with site ID as
        the index
    """
    data = rg.metadata.read(cache=cache)
    data.columns = snake_case_col(data.columns)
    if "metro_region" in data.columns and "region" not in data.columns:
        data["region"] = data.loc[:, "metro_region"]
    if "cpa_mw" in data.columns and "mw" not in data.columns:
        data["mw"] = data.loc[:, "cpa_mw"]
    data = data.loc[data["mw"] > 0, :]
    profile_path = Path(rg.group["profiles"])
    if rg.group.get("site_map"):
        table = Table(profile_path.parent / rg.group["site_map"])
        cols = table.columns
        df = table.read().set_index(cols[0])
        site_map = df[df.columns[0]]
    else:
        site_map = None

    return data, site_map


def flatten_cluster_def(
    scenario: Union[dict, list, str, int, float], detail_suffix: str
) -> str:
    """Turn a nested dictionary of clustering instructions into a string.

    Parameters
    ----------
    scenario : Union[dict, list, str, int, float]
        Either a dictionary, list, str, or numeric object -- base level must be a string
        or numeric.
    detail_suffix : str
        A string used to separate individual objects

    Returns
    -------
    str
        Flattened string representation
    """
    "Return cluster definition as a string for unique filenames"
    if isinstance(scenario, dict):
        for k, v in scenario.items():
            detail_suffix += flatten_cluster_def(k, "")
            detail_suffix += flatten_cluster_def(v, "")
    elif isinstance(scenario, list):
        for l in scenario:
            detail_suffix += flatten_cluster_def(l, "")
    else:
        detail_suffix += f"{scenario}_"

    return detail_suffix


def add_renewables_clusters(
    df: pd.DataFrame,
    region: str,
    settings: dict,
    cluster_builder: ClusterBuilder = None,
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
    if not cluster_builder:
        cluster_builder = build_resource_clusters(
            settings.get("RESOURCE_GROUPS"), settings.get("RESOURCE_GROUP_PROFILES")
        )
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
    if region in (settings.get("region_aggregations", {}) or {}):
        regions = settings.get("region_aggregations", {})[region]
        regions.append(region)  # Add model region, sometimes listed in RG file
    else:
        regions = [region]
    for scenario in settings.get("renewables_clusters", []) or []:
        if scenario["region"] != region:
            continue
        # Match cluster technology to NREL ATB technologies
        technologies = [
            k
            for k, v in atb_map.items()
            if v and all([scenario.get(ki) == vi for ki, vi in v.items()])
        ]
        if not technologies:
            s = (
                f"You have a renewables_cluster for technology '{scenario.get('technology')} "
                f"in region '{scenario.get('region')}', but no comparable new-build technology "
                "was specified in your settings file."
            )

            logger.warning(s)
            continue
        if len(technologies) > 1:
            raise ValueError(
                f"Renewables clusters match multiple NREL ATB technologies: {scenario}"
            )
        technology = technologies[0]
        _scenario = scenario.copy()
        # ClusterBuilder.get_clusters() does not take region as an argument
        _scenario.pop("region")

        # Assume not preclustering renewables unless set to True in settings or the
        # old parameters are used.
        precluster = False
        precluster_keys = ["max_clusters", "max_lcoe"]
        if settings.get("precluster_renewables") is True:
            precluster = True
        if any([k in precluster_keys for k in _scenario.keys()]):
            precluster = True

            # Create name suffex with unique id info like turbine_type and pref_site
        new_tech_suffix = "_" + "_".join(
            [
                str(v)
                for k, v in _scenario.items()
                if k
                not in [
                    "region",
                    "technology",
                    "max_clusters",
                    "min_capacity",
                    "filter",
                    "bin",
                    "group",
                    "cluster",
                ]
            ]
        )
        detail_suffix = flatten_cluster_def(_scenario, "_")
        cache_cluster_fn = f"{region}_{technology}_{detail_suffix}_cluster_data.parquet"
        cache_site_assn_fn = f"{region}_{technology}_{detail_suffix}_site_assn.parquet"
        sub_folder = settings.get("RESOURCE_GROUPS") or SETTINGS.get("RESOURCE_GROUPS")
        sub_folder = str(sub_folder).replace("/", "_").replace("\\", "_")
        cache_folder = Path(
            settings["input_folder"] / "cluster_assignments" / sub_folder
        )
        cache_cluster_fpath = cache_folder / cache_cluster_fn
        cache_site_assn_fpath = cache_folder / cache_site_assn_fn
        if precluster is False:
            if cache_cluster_fpath.exists() and cache_site_assn_fpath.exists():
                clusters = pd.read_parquet(cache_cluster_fpath)
                data = pd.read_parquet(cache_site_assn_fpath)
            else:
                drop_keys = ["min_capacity", "filter", "bin", "group", "cluster"]
                group_kwargs = dict(
                    [(k, v) for k, v in _scenario.items() if k not in drop_keys]
                )
                resource_groups = cluster_builder.find_groups(
                    existing=False,
                    **group_kwargs,
                )
                if not resource_groups:
                    raise ValueError(
                        f"Parameters do not match any resource groups: {group_kwargs}"
                    )
                if len(resource_groups) > 1:
                    meta = [rg.group for rg in resource_groups]
                    raise ValueError(
                        f"Parameters match multiple resource groups: {meta}"
                    )
                renew_data, site_map = load_resource_group_data(
                    resource_groups[0], cache=False
                )
                data = assign_site_cluster(
                    renew_data=renew_data,
                    profile_path=resource_groups[0].group.get("profiles"),
                    regions=regions,
                    site_map=site_map,
                    utc_offset=settings.get("utc_offset", 0),
                    **_scenario,
                )
                if data.empty:
                    continue
                clusters = (
                    data.groupby("cluster", as_index=False)
                    .apply(calc_cluster_values)
                    .rename(columns={"mw": "Max_Cap_MW"})
                    .assign(technology=technology, region=region)
                )

                cache_folder.mkdir(parents=True, exist_ok=True)
                if not cache_cluster_fpath.exists():
                    clusters.to_parquet(cache_cluster_fpath)
                if not cache_site_assn_fpath.exists():
                    cols = ["cpa_id", "cluster"]
                    data[cols].to_parquet(cache_site_assn_fpath)
            if settings.get("extra_outputs"):
                # fn = f"{region}_{technology}{new_tech_suffix}_site_cluster_assignments.csv"
                Path(settings["extra_outputs"]).mkdir(parents=True, exist_ok=True)
                cols = ["cpa_id", "cluster"]
                fn = f"{region}_{technology}{new_tech_suffix}_site_cluster_assignments.csv"
                data.loc[:, cols].to_csv(
                    Path(settings["extra_outputs"]) / fn, index=False
                )
        else:
            if cache_cluster_fpath.exists():
                clusters = pd.read_parquet(cache_cluster_fpath)
                data = None
            else:
                clusters = (
                    cluster_builder.get_clusters(
                        **_scenario,
                        ipm_regions=regions,
                        existing=False,
                        utc_offset=settings.get("utc_offset", 0),
                    )
                    .rename(columns={"mw": "Max_Cap_MW"})
                    .assign(technology=technology, region=region)
                )
                clusters["cluster"] = range(1, 1 + len(clusters))
                data = None
        cache_folder.mkdir(parents=True, exist_ok=True)
        if not cache_cluster_fpath.exists():
            clusters.to_parquet(cache_cluster_fpath)
        if not cache_site_assn_fpath.exists() and not data is None:
            cols = ["cpa_id", "cluster"]
            data[cols].to_parquet(cache_site_assn_fpath)
        if _scenario.get("min_capacity"):
            # Warn if total capacity less than expected
            capacity = clusters["Max_Cap_MW"].sum()
            if capacity < scenario["min_capacity"]:
                logger.warning(
                    f"Selected technology {_scenario['technology']} capacity"
                    + f" in region {region}"
                    + f" less than minimum ({capacity} < {_scenario['min_capacity']} MW)"
                )
        row = df[df["technology"] == technology].to_dict("records")[0]
        clusters["technology"] = clusters["technology"] + new_tech_suffix
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
    if "Cap_Size" not in user_techs.columns:
        user_techs["Cap_Size"] = 1

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
        "Cap_Size",
        "dollar_year",
    ]

    return user_techs[cols]
