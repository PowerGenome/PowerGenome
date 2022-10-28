"""
Load fuel prices needed for the model
"""

from asyncio.log import logger
import pandas as pd

from powergenome.eia_opendata import add_user_fuel_prices


def fuel_cost_table(
    fuel_costs: pd.DataFrame, generators: pd.DataFrame, settings: dict
) -> pd.DataFrame:
    """Use fuel costs and generators to create an output table for GenX

    Parameters
    ----------
    fuel_costs : pd.DataFrame
        The cost each fuel in different regions/years. Columns are "year", "price", "fuel",
        "region", "scenario", "full_fuel_name"
    generators : pd.DataFrame
        A single row for each generator/resource. Required columns are "Fuel".
    settings : dict
        Required parameters are "model_year". Optional parameters are "fuel_emission_factors",
        "user_fuel_price", "aeo_fuel_scenarios", "ccs_fuel_map", "ccs_disposal_cost",
        "ccs_capture_rate", "carbon_tax", "reduce_time_domain", "time_domain_days_per_period",
        and "time_domain_periods".

    Returns
    -------
    pd.DataFrame
        A dataframe with each unique regional fuel. The first row is the fuel emission factor
        and subsequent rows are hourly prices.
    """
    unique_fuels = generators["Fuel"].drop_duplicates()
    model_year_costs = fuel_costs.loc[fuel_costs["year"] == settings["model_year"], :]
    fuel_df = pd.DataFrame(unique_fuels)

    fuel_price_map = {
        row.full_fuel_name: row.price
        for row in model_year_costs.itertuples(index=False, name="row")
    }

    emission_dict = settings.get("fuel_emission_factors", {}) or {}
    user_fuels = settings.get("user_fuel_price", []) or []
    for u_f in user_fuels:
        if u_f not in emission_dict.keys():
            logger.warning(
                "\n\n**********************\n"
                f"The user fuel {u_f} does not have an emissions factor specified in "
                "the settings parameter 'fuel_emission_factors'. This is fine if the "
                "emission factor should be 0, otherwise be sure to add a value.\n"
            )
    fuel_emission_map = {}
    for full_fuel_name in fuel_price_map:
        if (
            full_fuel_name.split("_")[-1]
            in (settings.get("aeo_fuel_scenarios", {}) or {}).keys()
        ):
            base_fuel_name = full_fuel_name.split("_")[-1]
        elif (
            full_fuel_name.split("_")[-1]
            in (settings.get("user_fuel_price", {}) or {}).keys()
        ):
            base_fuel_name = full_fuel_name.split("_")[-1]
        else:
            base_fuel_name = full_fuel_name
        if base_fuel_name in emission_dict:
            fuel_emission_map[full_fuel_name] = emission_dict[base_fuel_name]
        else:
            fuel_emission_map[full_fuel_name] = 0

    ccs_fuels = (settings.get("ccs_fuel_map", {}) or {}).values()
    for ccs_fuel in ccs_fuels:
        fuels = generators.loc[
            generators["Fuel"].str.contains(ccs_fuel), "Fuel"
        ].unique()
        for f in fuels:
            # keep the non-ccs price
            base_name = ("_").join(f.split("_")[:-1])
            fuel_price_map[f] = fuel_price_map[base_name]
            fuel_emission_map[f] = fuel_emission_map[base_name]

    fuel_df["Cost_per_MMBtu"] = fuel_df["Fuel"].map(fuel_price_map)
    fuel_df["CO2_content_tons_per_MMBtu"] = fuel_df["Fuel"].map(fuel_emission_map)

    # Slow to loop through all of the rows this way but the df shouldn't be too long
    fuel_df = fuel_df.apply(adjust_ccs_fuels, axis=1, settings=settings)
    fuel_df = add_carbon_tax(fuel_df, settings)
    fuel_df["Cost_per_MMBtu"] = fuel_df["Cost_per_MMBtu"]
    fuel_df["CO2_content_tons_per_MMBtu"] = fuel_df["CO2_content_tons_per_MMBtu"]
    fuel_df.fillna(0, inplace=True)

    if settings.get("reduce_time_domain"):
        days = settings["time_domain_days_per_period"]
        time_periods = settings["time_domain_periods"]
        num_hours = days * time_periods * 24
    else:
        num_hours = 8760

    fuel_df_prices = pd.DataFrame(
        [fuel_df["Cost_per_MMBtu"]], index=range(1, num_hours + 1)
    )
    fuel_df_prices = fuel_df_prices.round(2)
    fuel_df_prices.columns = unique_fuels

    fuel_df_top = pd.DataFrame([fuel_df["CO2_content_tons_per_MMBtu"]])
    fuel_df_top = fuel_df_top.round(5)
    fuel_df_top.columns = unique_fuels
    fuel_df_top.index = [0]

    fuel_frames = [fuel_df_top, fuel_df_prices]
    fuel_df_new = pd.concat(fuel_frames)
    fuel_df_new.index.name = "Time_Index"
    return fuel_df_new


# def modify_fuel_new_genx():


def adjust_ccs_fuels(ccs_fuel_row: pd.Series, settings: dict) -> pd.Series:
    """If a fuel is used in CCS, adjust the carbon content and cost to reflect capture.

    Parameters
    ----------
    ccs_fuel_row : pd.Series
        A dataframe row with columns "Fuel", "CO2_content_tons_per_MMBtu", and "Cost_per_MMBtu"
    settings : dict
        If the parameter "ccs_fuel_map" is included (mapping technologies to ccs fuel names),
        then the parameter "ccs_capture_rate" with keys equal to the "ccs_fuel_map" values
        must also be included. The parameter "ccs_disposal_cost" is optional.

    Returns
    -------
    pd.Series
        Modified version of the input row/series.
    """
    base_fuel_name = None
    for ccs_fuel in (settings.get("ccs_fuel_map", {}) or {}).values():
        if ccs_fuel in ccs_fuel_row["Fuel"]:
            base_fuel_name = ccs_fuel
    if base_fuel_name:

        # USD/tonne disposal
        disposal_cost = settings.get("ccs_disposal_cost")
        if not disposal_cost:
            logger.warning(
                "You did not specify a CCS disposal cost, so it will be set to $0. "
                "Set a non-zero value with the settings parameter 'ccs_disposal_cost'."
            )
            disposal_cost = 0

        capture_rate = settings["ccs_capture_rate"][base_fuel_name]

        co2_captured = ccs_fuel_row["CO2_content_tons_per_MMBtu"] * capture_rate

        ccs_fuel_row["CO2_content_tons_per_MMBtu"] -= co2_captured
        ccs_fuel_row["Cost_per_MMBtu"] += co2_captured * disposal_cost

    else:
        pass

    return ccs_fuel_row


def add_carbon_tax(fuel_df, settings):

    ctax = settings.get("carbon_tax") or 0

    fuel_df.loc[:, "Cost_per_MMBtu"] = fuel_df.loc[:, "Cost_per_MMBtu"] + (
        fuel_df.loc[:, "CO2_content_tons_per_MMBtu"] * ctax
    )

    return fuel_df
