"""
Functions to fetch and modify NREL ATB data from PUDL
"""

import logging

import numpy as np
import pandas as pd
from powergenome.params import DATA_PATHS
from powergenome.price_adjustment import inflation_price_adjustment
from powergenome.util import reverse_dict_of_lists

idx = pd.IndexSlice
logger = logging.getLogger(__name__)


def fetch_atb_costs(pudl_engine, settings):
    """Get NREL ATB power plant cost data from database, filter where applicable

    Parameters
    ----------
    pudl_engine : sqlalchemy.Engine
        A sqlalchemy connection for use by pandas
    settings : dict
        User-defined parameters from a settings file

    Returns
    -------
    DataFrame
        Power plant cost data with columns:
        ['technology', 'cap_recovery_years', 'cost_case', 'financial_case',
       'basis_year', 'tech_detail', 'o_m_fixed_mw', 'o_m_variable_mwh', 'capex', 'cf',
       'fuel', 'lcoe', 'o_m', 'waccnomtech']
    """
    logger.info("Loading NREL ATB data")
    atb_costs = pd.read_sql_table("technology_costs_nrelatb", pudl_engine)

    index_cols = [
        "technology",
        "cap_recovery_years",
        "cost_case",
        "financial_case",
        "basis_year",
        "tech_detail",
    ]
    atb_costs.set_index(index_cols, inplace=True)
    atb_costs.drop(columns=["key", "id"], inplace=True)

    cap_recovery = str(settings["atb_cap_recovery_years"])
    financial = settings["atb_financial_case"]

    atb_costs = atb_costs.loc[idx[:, cap_recovery, :, financial, :, :], :]
    atb_costs = atb_costs.reset_index().fillna(0)

    atb_base_year = settings["atb_usd_year"]
    atb_target_year = settings["target_usd_year"]
    usd_columns = [
        "o_m_fixed_mw",
        "o_m_fixed_mwh",
        "o_m_variable_mwh",
        "capex",
        "capex_mwh",
    ]
    logger.info(
        f"Changing NREL ATB costs from {atb_base_year} to {atb_target_year} USD"
    )
    atb_costs.loc[:, usd_columns] = inflation_price_adjustment(
        price=atb_costs.loc[:, usd_columns],
        base_year=atb_base_year,
        target_year=atb_target_year,
    )

    logger.info("Inflating PV costs for DC to AC")

    atb_costs.loc[
        atb_costs["technology"].str.contains("PV"), ["o_m_fixed_mw", "o_m_variable_mwh"]
    ] *= settings["pv_ac_dc_ratio"]

    return atb_costs


def fetch_atb_heat_rates(pudl_engine):
    """Get heat rate projections for power plants

    Data is originally from AEO, NREL does a linear interpolation between current and
    final years.

    Parameters
    ----------
    pudl_engine : sqlalchemy.Engine
        A sqlalchemy connection for use by pandas

    Returns
    -------
    DataFrame
        Power plant heat rate data by year with columns:
        ['technology', 'tech_detail', 'basis_year', 'heat_rate']
    """

    heat_rates = pd.read_sql_table("technology_heat_rates_nrelatb", pudl_engine)

    return heat_rates


def atb_fixed_var_om_existing(results, atb_costs_df, atb_hr_df, settings):
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
    atb_costs_df : DataFrame
        Cost data from NREL ATB
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
    techs = settings["eia_atb_tech_map"]
    existing_year = settings["atb_existing_year"]

    # ATB string is <technology>_<tech_detail>
    techs = {eia: atb_costs_df.split("_") for eia, atb_costs_df in techs.items()}

    df_list = []
    grouped_results = results.reset_index().groupby(
        ["technology", "Heat_rate_MMBTU_per_MWh"], as_index=False
    )
    for group, _df in grouped_results:

        eia_tech, existing_hr = group
        atb_tech, tech_detail = techs[eia_tech]
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

        if ("Natural Gas Fired" in eia_tech or "Coal" in eia_tech) and settings[
            "use_nems_coal_ng_om"
        ]:
            # Change CC and CT O&M to EIA NEMS values, which are much higher for CCs and
            # lower for CTs than a heat rate & linear mulitpler correction to the ATB
            # values.
            # Also using the new values for coal plants, assuming 40-50 yr age and half
            # FGD
            # https://www.eia.gov/analysis/studies/powerplants/generationcost/pdf/full_report.pdf
            logger.info(f"Using NEMS values for {eia_tech} fixed/variable O&M")
            target_usd_year = settings["target_usd_year"]
            ng_o_m = {
                "Combined Cycle": {
                    "o_m_fixed_mw": inflation_price_adjustment(
                        28.84 * 1000, 2017, target_usd_year
                    ),
                    "o_m_variable_mwh": inflation_price_adjustment(
                        3.91, 2017, target_usd_year
                    ),
                },
                "Combustion Turbine": {
                    "o_m_fixed_mw": inflation_price_adjustment(
                        12.23 * 1000, 2017, target_usd_year
                    ),
                    "o_m_variable_mwh": 0,
                },
                "Coal": {
                    "o_m_fixed_mw": inflation_price_adjustment(
                        ((22.2 + 27.88) / 2 + 46.01) * 1000, 2017, target_usd_year
                    ),
                    "o_m_variable_mwh": inflation_price_adjustment(
                        1.78, 2017, target_usd_year
                    ),
                },
            }

            if "Combined Cycle" in eia_tech:
                fixed = ng_o_m["Combined Cycle"]["o_m_fixed_mw"]
                variable = ng_o_m["Combined Cycle"]["o_m_variable_mwh"]
                _df["Fixed_OM_cost_per_MWyr"] = fixed
                _df["Var_OM_cost_per_MWh"] = variable

            if "Combustion Turbine" in eia_tech:
                # need to adjust the EIA fixed/variable costs because they have no
                # variable cost per MWh for existing CTs but they do have per MWh for
                # new build. Assume $11/MWh from new-build and 4% CF:
                # (11*8760*0.04/1000)=$3.85/kW-yr. Scale the new-build variable
                # (~$11/MWh) by relative heat rate and subtract a /kW-yr value as
                # calculated above from the FOM.
                # Based on conversation with Jesse J. on Dec 20, 2019.

                atb_var_om_mwh = (
                    atb_costs_df.query(
                        "technology==@atb_tech & cost_case=='Mid' "
                        "& tech_detail==@tech_detail & basis_year==@existing_year"
                    )
                    .squeeze()
                    .at["o_m_variable_mwh"]
                    * settings["atb_multipliers"]["ngct"]["Var_OM_cost_per_MWh"]
                )
                variable = atb_var_om_mwh * (existing_hr / new_build_hr)

                fixed = ng_o_m["Combustion Turbine"]["o_m_fixed_mw"]
                fixed = fixed - (variable * 8760 * 0.04)

                _df["Fixed_OM_cost_per_MWyr"] = fixed
                _df["Var_OM_cost_per_MWh"] = variable

            if "Coal" in eia_tech:
                fixed = ng_o_m["Coal"]["o_m_fixed_mw"]
                variable = ng_o_m["Coal"]["o_m_variable_mwh"]
                _df["Fixed_OM_cost_per_MWyr"] = fixed
                _df["Var_OM_cost_per_MWh"] = variable
        else:

            atb_fixed_om_mw_yr = (
                atb_costs_df.query(
                    "technology==@atb_tech & cost_case=='Mid' "
                    "& tech_detail==@tech_detail & basis_year==@existing_year"
                )
                .squeeze()
                .at["o_m_fixed_mw"]
            )
            atb_var_om_mwh = (
                atb_costs_df.query(
                    "technology==@atb_tech & cost_case=='Mid' "
                    "& tech_detail==@tech_detail & basis_year==@existing_year"
                )
                .squeeze()
                .at["o_m_variable_mwh"]
            )
            _df["Fixed_OM_cost_per_MWyr"] = (
                atb_fixed_om_mw_yr
                * settings["existing_om_multiplier"]
                * (existing_hr / new_build_hr)
            )
            _df["Var_OM_cost_per_MWh"] = atb_var_om_mwh * (existing_hr / new_build_hr)

        df_list.append(_df)

    mod_results = pd.concat(df_list, ignore_index=True)
    mod_results = mod_results.sort_values(["region", "technology", "cluster"])
    mod_results.loc[:, "Fixed_OM_cost_per_MWyr"] = mod_results.loc[
        :, "Fixed_OM_cost_per_MWyr"
    ].astype(int)
    mod_results.loc[:, "Var_OM_cost_per_MWh"] = mod_results.loc[
        :, "Var_OM_cost_per_MWh"
    ].round(1)

    return mod_results


def single_generator_row(atb_costs, new_gen_type, model_year_range):
    """Create a data row with NREL ATB costs and performace for a single technology

    Parameters
    ----------
    atb_costs : dataframe
        Data from the sqlite table
    new_gen_type : str
        type of generating resource
    model_year_range : list
        All of the years that should be averaged over

    Returns
    -------
    dataframe
        A single row dataframe with average cost and performence values over the study
        period.
    """

    technology, tech_detail, cost_case, size_mw = new_gen_type
    numeric_cols = [
        "basis_year",
        "o_m_fixed_mw",
        "o_m_fixed_mwh",
        "o_m_variable_mwh",
        "capex",
        "capex_mwh",
        "cf",
        "fuel",
        "lcoe",
        "o_m",
        "waccnomtech",
    ]
    s = atb_costs.query(
        "technology==@technology & tech_detail==@tech_detail "
        "& cost_case==@cost_case & basis_year.isin(@model_year_range)"
    )[numeric_cols].mean()
    cols = [
        "technology",
        "cost_case",
        "tech_detail",
        "basis_year",
        "o_m_fixed_mw",
        "o_m_fixed_mwh",
        "o_m_variable_mwh",
        "capex",
        "capex_mwh",
        "cf",
        "fuel",
        "lcoe",
        "o_m",
        "waccnomtech",
    ]

    row = pd.DataFrame([technology, cost_case, tech_detail] + s.to_list(), index=cols).T

    row["Cap_size"] = size_mw

    return row


def investment_cost_calculator(capex, wacc, cap_rec_years):

    # wacc comes through as an object type series now that we're averaging across years
    if not isinstance(wacc, float):
        wacc = wacc.astype(float)
    inv_cost = capex * (
        np.exp(wacc * cap_rec_years)
        * (np.exp(wacc) - 1)
        / (np.exp(wacc * cap_rec_years) - 1)
    )

    return inv_cost


def regional_capex_multiplier(df, region, region_map, tech_map, regional_multipliers):

    cost_region = region_map[region]
    tech_multiplier = regional_multipliers.loc[cost_region, :].squeeze()

    tech_multiplier_map = {}
    for atb_tech, eia_tech in tech_map.items():
        if df["technology"].str.contains(atb_tech).sum() > 0:
            full_atb_tech = df.loc[
                df["technology"].str.contains(atb_tech).idxmax(), "technology"
            ]
            tech_multiplier_map[full_atb_tech] = tech_multiplier.at[eia_tech]

    df["Inv_cost_per_MWyr"] *= df["technology"].map(tech_multiplier_map)
    df["Inv_cost_per_MWhyr"] *= df["technology"].map(tech_multiplier_map)

    return df


def atb_new_generators(results, atb_costs, atb_hr, settings):
    """Add rows for new generators in each region

    Parameters
    ----------
    results : DataFrame
        Compiled results of clustered power plants with weighted average heat
    atb_costs : [type]
        [description]
    atb_hr : [type]
        [description]
    settings : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """

    new_gen_types = settings["atb_new_gen"]
    model_year = settings["model_year"]
    try:
        first_planning_year = settings["model_first_planning_year"]
        model_year_range = range(first_planning_year, model_year + 1)
    except KeyError:
        model_year_range = list(range(model_year + 1))

    regions = settings["model_regions"]

    new_gen_df = pd.concat(
        [
            single_generator_row(atb_costs, new_gen, model_year_range)
            for new_gen in new_gen_types
        ],
        ignore_index=True,
    )

    if isinstance(settings["atb_battery_wacc"], float):
        new_gen_df.loc[new_gen_df["technology"] == "Battery", "waccnomtech"] = settings[
            "atb_battery_wacc"
        ]
    elif isinstance(settings["atb_battery_wacc"], str):
        solar_wacc = new_gen_df.loc[
            new_gen_df["technology"].str.contains("UtilityPV"), "waccnomtech"
        ].values[0]

        new_gen_df.loc[
            new_gen_df["technology"] == "Battery", "waccnomtech"
        ] = solar_wacc

    # Add user-defined technologies
    # This should probably be separate from ATB techs, and the regional cost multipliers
    # should be its own function.
    if settings["additional_technologies_fn"] is not None:
        user_costs, user_hr = load_user_defined_techs(settings)
        new_gen_df = pd.concat([new_gen_df, user_costs], ignore_index=True, sort=False)
        atb_hr = pd.concat([atb_hr, user_hr], ignore_index=True, sort=False)

    new_gen_df = new_gen_df.merge(
        atb_hr, on=["technology", "tech_detail", "basis_year"], how="left"
    )

    new_gen_df = new_gen_df.rename(
        columns={
            "heat_rate": "Heat_rate_MMBTU_per_MWh",
            "o_m_fixed_mw": "Fixed_OM_cost_per_MWyr",
            "o_m_fixed_mwh": "Fixed_OM_cost_per_MWhyr",
            "o_m_variable_mwh": "Var_OM_cost_per_MWh",
        }
    )

    # Adjust values for CT/CC generators to match advanced techs in NEMS rather than
    # ATB average of advanced and conventional.
    # This is now generalized for changes to ATB values for any technology type.
    for tech, tech_multipliers in settings["atb_multipliers"].items():
        assert isinstance(tech_multipliers, dict), (
            "The settings parameter 'atb_multipliers' must be a nested list.\n"
            "Each top-level key is a short name of the technology, with a nested"
            " dictionary of items below it."
        )
        assert (
            "technology" in tech_multipliers.keys()
        ), "Each nested dictionary in atb_multipliers must have a 'technology' key."
        assert (
            "tech_detail" in tech_multipliers.keys()
        ), "Each nested dictionary in atb_multipliers must have a 'tech_detail' key."

        technology = tech_multipliers.pop("technology")
        tech_detail = tech_multipliers.pop("tech_detail")

        for key, multiplier in tech_multipliers.items():

            new_gen_df.loc[
                (new_gen_df.technology == technology)
                & (new_gen_df.tech_detail == tech_detail),
                key,
            ] *= multiplier

    new_gen_df["technology"] = (
        new_gen_df["technology"]
        + "_"
        + new_gen_df["tech_detail"].astype(str)
        + "_"
        + new_gen_df["cost_case"]
    )

    new_gen_df["Inv_cost_per_MWyr"] = investment_cost_calculator(
        capex=new_gen_df["capex"],
        wacc=new_gen_df["waccnomtech"],
        cap_rec_years=settings["atb_cap_recovery_years"],
    )

    new_gen_df["Inv_cost_per_MWhyr"] = investment_cost_calculator(
        capex=new_gen_df["capex_mwh"],
        wacc=new_gen_df["waccnomtech"],
        cap_rec_years=settings["atb_cap_recovery_years"],
    )

    # Some technologies might have a different capital recovery period
    if settings["alt_atb_cap_recovery_years"] is not None:
        for tech, years in settings["alt_atb_cap_recovery_years"].items():
            tech_mask = new_gen_df["technology"].str.contains(tech)

            new_gen_df.loc[tech_mask, "Inv_cost_per_MWyr"] = investment_cost_calculator(
                capex=new_gen_df.loc[tech_mask, "capex"],
                wacc=new_gen_df.loc[tech_mask, "waccnomtech"],
                cap_rec_years=years,
            )

            new_gen_df.loc[
                tech_mask, "Inv_cost_per_MWhyr"
            ] = investment_cost_calculator(
                capex=new_gen_df.loc[tech_mask, "capex_mwh"],
                wacc=new_gen_df.loc[tech_mask, "waccnomtech"],
                cap_rec_years=years,
            )

    keep_cols = [
        "technology",
        "basis_year",
        "Fixed_OM_cost_per_MWyr",
        "Fixed_OM_cost_per_MWhyr",
        "Var_OM_cost_per_MWh",
        "capex",
        "capex_mwh",
        "Inv_cost_per_MWyr",
        "Inv_cost_per_MWhyr",
        "Heat_rate_MMBTU_per_MWh",
        "Cap_size",
    ]

    regional_cost_multipliers = pd.read_csv(
        DATA_PATHS["cost_multipliers"] / "EIA regional cost multipliers.csv",
        index_col=0,
    )
    rev_mult_region_map = reverse_dict_of_lists(settings["cost_multiplier_region_map"])
    rev_mult_tech_map = reverse_dict_of_lists(
        settings["cost_multiplier_technology_map"]
    )
    df_list = []
    for region in regions:
        _df = new_gen_df.loc[:, keep_cols].copy()
        _df["region"] = region
        _df = regional_capex_multiplier(
            _df,
            region,
            rev_mult_region_map,
            rev_mult_tech_map,
            regional_cost_multipliers,
        )
        _df = add_extra_wind_solar_rows(_df, region, settings)

        if region in settings["new_gen_not_available"].keys():
            techs = settings["new_gen_not_available"][region]
            for tech in techs:
                _df = _df.loc[~_df["technology"].str.contains(tech), :]

        df_list.append(_df)

    results = pd.concat(
        [results, pd.concat(df_list, ignore_index=True)], ignore_index=True
    )

    int_cols = [
        "Fixed_OM_cost_per_MWyr",
        "Fixed_OM_cost_per_MWhyr",
        "Inv_cost_per_MWyr",
        "Inv_cost_per_MWhyr",
    ]
    results = results.fillna(0)
    results.loc[:, int_cols] = results.loc[:, int_cols].astype(int)
    results.loc[:, "Var_OM_cost_per_MWh"] = (
        results.loc[:, "Var_OM_cost_per_MWh"].astype(float).round(1)
    )

    return results


def add_extra_wind_solar_rows(df, region, settings):
    wind_bins = settings["new_wind_solar_regional_bins"]["LandbasedWind"][region]
    extra_wind_bins = wind_bins - 1
    solar_bins = settings["new_wind_solar_regional_bins"]["UtilityPV"][region]
    extra_solar_bins = solar_bins - 1

    for i in range(extra_wind_bins):
        row_iloc = np.argwhere(df.technology.str.contains("LandbasedWind")).flatten()[
            -1
        ]
        df = pd.concat([df.iloc[:row_iloc], df.iloc[[row_iloc]], df.iloc[row_iloc:]])

    # if len(df.query("technology.str.contains('Wind')")) != wind_bins:
    #     wind_rows = len(df.query("technology.str.contains('LandbasedWind')"))
    #     print(df)
    #     raise ValueError(f"Number of wind rows in {region} is {wind_rows}, need {wind_bins}")

    for i in range(extra_solar_bins):
        row_iloc = np.argwhere(df.technology.str.contains("UtilityPV")).flatten()[-1]
        df = pd.concat([df.iloc[:row_iloc], df.iloc[[row_iloc]], df.iloc[row_iloc:]])

    df = df.reset_index(drop=True)

    return df


def load_user_defined_techs(settings):
    """Load user-defined technologies from a CSV file. Returns cost columns and heat
    rate as separate dataframes.

    Parameters
    ----------
    settings : dict
        User-defined parameters from a settings file

    Returns
    -------
    DataFrames
        A tuple of 2 dataframes. The first contains cost columns, the second contains
        heat rate. Both have technology, tech_detail, and cost_case.
    """

    fn = settings["additional_technologies_fn"]
    user_techs = pd.read_csv(DATA_PATHS["additional_techs"] / fn)

    user_techs = user_techs.loc[
        user_techs["technology"].isin(settings["additional_new_gen"]), :
    ]

    if "tech_detail" not in user_techs.columns:
        user_techs["tech_detail"] = ""
    if "cost_case" not in user_techs.columns:
        user_techs["cost_case"] = ""

    cost_cols = [
        "technology",
        "tech_detail",
        "cost_case",
        "basis_year",
        "capex",
        "capex_mwh",
        "o_m_fixed_mw",
        "o_m_fixed_mwh",
        "o_m_variable_mwh",
        "waccnomtech",
        "dollar_year",
    ]

    hr_cols = ["technology", "tech_detail", "basis_year", "heat_rate"]
    user_costs = user_techs.loc[:, cost_cols]
    user_hr = user_techs.loc[:, hr_cols]

    return user_costs, user_hr
