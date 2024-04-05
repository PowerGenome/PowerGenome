"""
Load fuel prices needed for the model
"""

from asyncio.log import logger
from typing import Dict, List

import pandas as pd

from powergenome.eia_opendata import add_user_fuel_prices


def fuel_cost_table(
    fuel_costs: pd.DataFrame, generators: pd.DataFrame, settings: dict
) -> pd.DataFrame:
    """Create a table of fuel costs formatted for the GenX model.

    Costs are based on the `fuel_costs` dataframe and any values listed in the settings
    dictionary under the key "user_fuel_price". If CCS fuels or a carbon tax are defined
    in the settings then the

    Parameters
    ----------
    fuel_costs : pd.DataFrame
        A table of fuel prices. Must have columns "year", "price", "fuel", "region", and
        "full_fuel_name". "fuel" should be a base fuel name such as coal or distillate and
        cannot include an underscore. If the full fuel name is a combination of the region,
        a scenario, and the base fuel name, the base fuel name should be the last element
        so that it is selected when the string is split on underscores.

        >>> <full_fuel_name>.split("_")[-1] = <fuel>
    generators : pd.DataFrame
        A table of generators with the column "Fuel". The values in this column should
        correspond to either the "full_fuel_name" column or one of the fuels in the
        settings key "user_fuel_price". If regional prices are provided in the settings
        then the fuel name should be <region>_<fuel>.
    settings : dict
        Should include the key "fuel_emission_factors" with CO2 emissions in tonnes
        per MMBTU for each fuel type used.

        If adding user prices, should have the key "user_fuel_price" with value of a
        dictionary matching user fuel names and prices. Prices can either be a single
        price for all regions or a price per region. For example this shows biomass with
        different prices in two regions and ZCF with the same price in all regions:

        settings["user_fuel_price"] = {
            "biomass": {"SC_VACA": 10, "PJM_DOM": 5},
            "ZCF": 15
        }

        If the keys "target_usd_year" and "user_fuel_usd_year" are also included, fuel
        prices will be corrected to the correct USD year. "user_fuel_usd_year" should
        be a dictionary with fuel name: USD year pairings. Only fuels included in this
        dictionary will have their prices changed to the target USD year.

    Returns
    -------
    pd.DataFrame
        The cost of fuels used by generators in the final year of a modeling period.
        Formatted for GenX, where headers are the fuel names, the first row is the
        fuel CO2 content (tonnes per MMBTU), and subsequent rows are hourly prices.
        Prices are identical in all hours. The first (index) column has the header
        "Time_Index" and values from 0-N, where N is the number of hours used in the model.
    """
    all_fuel_costs = add_user_fuel_prices(settings, fuel_costs)
    unique_fuels = generators["Fuel"].drop_duplicates()
    model_year_costs = all_fuel_costs.loc[
        all_fuel_costs["year"] == settings["model_year"], :
    ]
    fuel_df = pd.DataFrame(unique_fuels)

    fuel_price_map = {
        row.full_fuel_name: row.price
        for row in model_year_costs.itertuples(index=False, name="row")
    }

    emission_dict = settings.get("fuel_emission_factors", {}) or {}
    user_fuels = set(all_fuel_costs["fuel"]) - set(fuel_costs["fuel"])
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
    if settings.get("co2_pipeline_filters") and settings.get("co2_pipeline_cost_fn"):
        ccs_disposal_cost = 0
    else:
        ccs_disposal_cost = settings.get("ccs_disposal_cost", 0)
    fuel_df = fuel_df.apply(
        adjust_ccs_fuels,
        axis=1,
        ccs_fuels=(settings.get("ccs_fuel_map", {}) or {}).values(),
        ccs_capture_rate=(settings.get("ccs_capture_rate", {}) or {}),
        ccs_disposal_cost=ccs_disposal_cost,
    )
    fuel_df = add_carbon_tax(fuel_df, settings.get("carbon_tax"))
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


def adjust_ccs_fuels(
    ccs_fuel_row: pd.Series,
    ccs_fuels: List[str] = None,
    ccs_capture_rate: Dict[str, float] = {},
    ccs_disposal_cost: float = None,
) -> pd.Series:
    """Adjust the "CO2_content_tons_per_MMBtu" and "Cost_per_MMBtu" values to account for
    the value from settings parameter "ccs_capture_rate".

    If using this function to adjust the CO2 content and cost for CCS-specific fuels,
    the settings dict should map the names of technologies to a base CCS fuel name in the
    parameter "ccs_fuel_map". The base CCS fuel names do not include a region or scenario,
    they are something like "naturalgas_ccs90".


    Parameters
    ----------
    ccs_fuel_row : pd.Series
        A single row from the larger fuel dataframe with columns "Fuel", "Cost_per_MMBtu",
        and "CO2_content_tons_per_MMBtu".
    ccs_fuels : List[str], optional
        A list of CCS fuels mapped to generator types, by default None
    ccs_capture_rate : Dict[str, float], optional
        The capture rate (0-1) for each CCS fuel type in `ccs_fuels`, by default {}
    ccs_disposal_cost : float, optional
        The cost in USD per tonne of CO2 disposal that should be added to a fuel price,
        by default None

    Returns
    -------
    pd.Series
        If the fuel is mapped to a CCS technology, the "CO2_content_tons_per_MMBtu" and
        "Cost_per_MMBtu" values will be modified.

    Raises
    ------
    KeyError
        One of the CCS fuels mapped to a technology is not included in the "ccs_capture_rate"
        dict.
    """

    base_fuel_name = None
    for ccs_fuel in ccs_fuels or []:
        if ccs_fuel not in ccs_capture_rate.keys():
            raise KeyError(
                f"The CCS fuel name {ccs_fuel} from settings parameter 'ccs_fuel_map' "
                "does not have capture rate in the settings parameter 'ccs_capture_rate'."
                "Adjust your settings to include the capture rate or remove the fuel."
            )
        if ccs_fuel in ccs_fuel_row["Fuel"]:
            base_fuel_name = ccs_fuel
    if base_fuel_name:
        # USD/tonne disposal
        if not ccs_disposal_cost:
            logger.warning(
                "You did not specify a CCS disposal cost, so it will be set to $0. "
                "Set a non-zero value with the settings parameter 'ccs_disposal_cost'."
            )
            ccs_disposal_cost = 0

        capture_rate = ccs_capture_rate.get(base_fuel_name, 0)

        co2_captured = ccs_fuel_row["CO2_content_tons_per_MMBtu"] * capture_rate

        ccs_fuel_row["CO2_content_tons_per_MMBtu"] -= co2_captured
        ccs_fuel_row["Cost_per_MMBtu"] += co2_captured * ccs_disposal_cost

    else:
        pass

    return ccs_fuel_row


def add_carbon_tax(
    fuel_df: pd.DataFrame, carbon_tax_value: float = None
) -> pd.DataFrame:
    """Increases fuel prices to account for a carbon tax

    Parameters
    ----------
    fuel_df : pd.DataFrame
        Table with columns "Cost_per_MMBtu" and "CO2_content_tons_per_MMBtu"
    carbon_tax_value : float, optional
        The carbon tax cost in USD per tonne CO2, by default None.

    Returns
    -------
    pd.DataFrame
        Modified version of input df with fuel prices increased to reflect the carbon tax.
        The df is returned unaltered if no carbon tax is provided.
    """
    if not carbon_tax_value:
        return fuel_df

    for col in ["Cost_per_MMBtu", "CO2_content_tons_per_MMBtu"]:
        if col not in fuel_df.columns:
            raise KeyError(
                f"The required column {col} is missing from your fuel dataframe. Cannot "
                "apply a carbon tax to fuel prices without this column."
            )

    fuel_df.loc[:, "Cost_per_MMBtu"] = fuel_df.loc[:, "Cost_per_MMBtu"] + (
        fuel_df.loc[:, "CO2_content_tons_per_MMBtu"] * carbon_tax_value
    )

    return fuel_df
