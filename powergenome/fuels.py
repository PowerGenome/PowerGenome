"""
Load fuel prices needed for the model
"""

import operator
from asyncio.log import logger
from typing import Dict, List

import pandas as pd

from powergenome.database import get_data
from powergenome.financials import inflation_price_adjustment
from powergenome.settings import auto_fill_settings
from powergenome.util import reverse_dict_of_lists


@auto_fill_settings()
def fuel_cost_table(
    fuel_costs: pd.DataFrame,
    generators: pd.DataFrame,
    settings: dict = None,
    num_hours: int = None,
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
            in (settings.get("fuel_scenarios", {}) or {}).keys()
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
    elif num_hours is None:
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
            logger.debug(
                "You did not specify a fuel-modifying CCS disposal cost, so it will be set to $0. "
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


def fetch_fuel_prices(settings: dict, inflate_price: bool = True) -> pd.DataFrame:
    """
    Get fuel prices for all regions, fuel types, and scenarios (series IDs)
    included in the settings from the DataManager's standardized "fuel_price" table.
    Combine the region, scenario, and fuel name into a full
    fuel name column. Optionally, adjust the prices to the target dollar year.

    Parameters
    ----------
    settings : dict
        Should include the following keys:
            fuel_data_year (int)
            fuel_series_region_names (dict)
            fuel_series_names (dict)
            fuel_series_scenario_names (dict)
    inflate_price : bool, optional
        If True, adjust the fuel prices to the year "target_usd_year" from the settings.
        If False, do not adjust the prices. Requires the column "dollar_year" in the
        fuel price data table.

    Returns
    -------
    pd.DataFrame
        All fuel price data for the specified data year, with columns:
        ['year', 'price', 'fuel', 'region', 'scenario', 'full_fuel_name'].

    Raises
    ------
    KeyError
        If 'fuel_data_year' is missing from settings and the data file contains multiple data years.
        If the specified 'fuel_data_year' is not found in the 'data_year' column of the data file.
    FileNotFoundError
        If the specified data file is empty.
    """

    data_year = settings.get("fuel_data_year")

    all_fuel_data = get_data("fuel_price")
    if all_fuel_data.empty:
        raise FileNotFoundError(
            f"The fuel price table is empty. "
            "Please check the data location and table name."
        )
    if data_year is not None and "data_year" in all_fuel_data.columns:
        if data_year not in all_fuel_data["data_year"].unique():
            raise KeyError(
                f"The data year specified in parameter 'fuel_data_year' ({data_year}) is "
                "not in the 'data_year' column of your fuel price data table. "
                "Please check the data file and the settings parameter."
            )
        fuel_data = all_fuel_data.query("data_year == @data_year")
    elif (
        data_year is None
        and "year" in all_fuel_data.columns
        and all_fuel_data["data_year"].nunique() > 1
    ):
        raise KeyError(
            "The parameter 'fuel_data_year' is not in your settings files and the fuel price "
            "data table contains multiple data years."
        )
    else:
        fuel_data = all_fuel_data.copy()

    if settings.get("fuel_series_names"):
        for k, v in settings["fuel_series_names"].items():
            fuel_data.loc[fuel_data["fuel"].str.lower() == v.lower(), "fuel"] = k
    if settings.get("fuel_series_scenario_names"):
        for k, v in settings["fuel_series_scenario_names"].items():
            fuel_data.loc[
                fuel_data["scenario"].str.lower() == v.lower(), "scenario"
            ] = k
    if settings.get("fuel_series_region_names"):
        for k, v in settings["fuel_series_region_names"].items():
            fuel_data.loc[fuel_data["region"].str.lower() == v.lower(), "region"] = k

    fuel_data["full_fuel_name"] = (
        fuel_data["region"] + "_" + fuel_data["scenario"] + "_" + fuel_data["fuel"]
    )
    fuel_data = fuel_data.dropna(subset=["full_fuel_name"])

    if inflate_price:
        try:
            df_list = []
            for year, _df in fuel_data.groupby("dollar_year"):
                _df.loc[:, "price"] = inflation_price_adjustment(
                    price=_df.loc[:, "price"],
                    base_year=year,
                    target_year=settings["target_usd_year"],
                    data_location=settings["data_location"],
                    table_name=settings["dollar_year_table"],
                )
                df_list.append(_df)
            fuel_data = pd.concat(df_list, ignore_index=True, sort=False)
        except (KeyError, TypeError):
            logger.warning(
                """
    ************
    Unable to inflate fuel prices. Check your settings file to ensure the keys
    "target_usd_year" and "aeo_fuel_usd_year" are valid integers.
    ************
                """
            )

    return fuel_data


def modify_fuel_prices(
    prices: pd.DataFrame,
    fuel_region_map: dict,
    regional_fuel_adjustments: dict = None,
) -> pd.DataFrame:
    """Modify the AEO fuel prices by model region or fuel within a model region.

    Parameters
    ----------
    prices : pd.DataFrame
        Fuel prices from AEO, with columns ['year', 'price', 'fuel', 'region', 'scenario',
        'full_fuel_name']
    fuel_region_map : dict
        Mapping of AEO census division fuel names to lists of model regions
    regional_fuel_adjustments : dict, optional
        Modifications of fuel prices by region or fuel within region, by default None

    Returns
    -------
    pd.DataFrame
        Full input dataframe with modified copies for model regions and fuels specified
        in `regional_fuel_adjustments`.

    Raises
    ------
    KeyError
        The required parameter 'fuel_region_map' is missing
    KeyError
        One or more model regions having fuel prices modified is not in `fuel_region_map`
    KeyError
        Invalid operator type
    KeyError
        Invalid fuel name
    KeyError
        Invalid operator type
    TypeError
        Fuel price modifiers are not a list or a dictionary of lists
    """

    if not regional_fuel_adjustments:
        return prices

    if not fuel_region_map:
        raise KeyError("The required parameter 'fuel_region_map' is missing.")

    allowed_operators = ["add", "mul", "truediv", "sub"]
    model_regions = list(regional_fuel_adjustments)
    model_fuel_region_map = reverse_dict_of_lists(fuel_region_map)
    if not all(r in model_fuel_region_map for r in model_regions):
        raise KeyError(
            "All model regions listed in the settings parameter 'regional_fuel_adjustments' "
            "should also be included in `fuel_region_map`. One or more regions was "
            "not found."
        )

    df_list = []
    for region, adj in regional_fuel_adjustments.items():
        aeo_region = model_fuel_region_map[region]
        if isinstance(adj, list):
            op, op_value = adj
            if op not in allowed_operators:
                raise KeyError(
                    f"The regional fuel price adjustment for {region} needs a valid "
                    f"operator from the list\n{allowed_operators}\n"
                    "in the format [<operator>, <value>].\n"
                )
            f = operator.attrgetter(op)
            df = prices.loc[prices["region"] == aeo_region, :]
            df.loc[:, "region"] = region
            df.loc[:, "price"] = f(operator)(df["price"], op_value)
            df.loc[:, "full_fuel_name"] = df["full_fuel_name"].str.replace(
                aeo_region, region
            )
            df_list.append(df)
        elif isinstance(adj, dict):
            for fuel, op_list in adj.items():
                if fuel not in prices["fuel"].unique():
                    raise KeyError(
                        f"The fuel '{fuel}' is listed under the region {region} in your settings "
                        "parameter 'regional_fuel_adjustments'. There was no AEO fuel "
                        "price fetched for this fuel so it cannot be modified."
                    )
                op, op_value = op_list
                if op not in allowed_operators:
                    raise KeyError(
                        f"The regional fuel price adjustment for '{fuel}' in {region} "
                        f"needs to be an operator from the list {allowed_operators}. "
                        f"You supplied '{op}', which is not a valid operator."
                    )
                f = operator.attrgetter(op)
                df = prices.loc[
                    (prices["region"] == aeo_region) & (prices["fuel"] == fuel.lower()),
                    :,
                ]
                df.loc[:, "region"] = region
                df.loc[:, "price"] = f(operator)(df["price"], op_value)
                df.loc[:, "full_fuel_name"] = df["full_fuel_name"].str.replace(
                    aeo_region, region
                )
                df_list.append(df)
        else:
            raise TypeError(
                "Fuel price modifiers in the settings parameter 'regional_fuel_adjustments' "
                "must be a list of the form '[<op>, <value>]', or a similar list for a "
                "specific fuel. "
                f"Your value look like '{adj}' for region '{region}'."
            )

    mod_prices = pd.concat([prices] + df_list, ignore_index=True, sort=False)

    return mod_prices


def add_user_fuel_prices(settings: dict, df: pd.DataFrame = None) -> pd.DataFrame:
    """Add user fuel prices to a dataframe of user prices from AEO (or elsewhere)

    Parameters
    ----------
    settings : dict
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
    df : pd.DataFrame, optional
        A dataframe with fuel prices from AEO (or elsewhere), by default None. Should
        have columns ["year", "price", "fuel", "region", "scenario", "full_fuel_name"]

    Returns
    -------
    pd.DataFrame
        The combined dataframes of user prices and the other price dataframe provided
        as input. Columns are ["year", "price", "fuel", "region", "scenario", "full_fuel_name"].
    """

    if not settings.get("user_fuel_price"):
        if df is not None:
            return df
    cols = ["year", "price", "fuel", "region", "scenario", "full_fuel_name"]
    if df is not None and not df.empty:
        years = df["year"].unique()
    else:
        years = range(2020, 2051)
    fuel_data = {c: [] for c in cols}

    for fuel, val in settings["user_fuel_price"].items():
        if isinstance(val, dict):
            for region, price in val.items():
                fuel_name = f"{region}_{fuel}"
                fuel_data["year"].extend(years)
                fuel_data["price"].extend([price] * len(years))
                fuel_data["fuel"].extend([fuel] * len(years))
                fuel_data["region"].extend([region] * len(years))
                fuel_data["scenario"].extend(["user"] * len(years))
                fuel_data["full_fuel_name"].extend([fuel_name] * len(years))
        else:
            fuel_data["year"].extend(years)
            fuel_data["price"].extend([val] * len(years))
            fuel_data["fuel"].extend([fuel] * len(years))
            fuel_data["region"].extend([""] * len(years))
            fuel_data["scenario"].extend(["user"] * len(years))
            fuel_data["full_fuel_name"].extend([fuel] * len(years))

    user_fuel_price = pd.DataFrame(fuel_data)
    if settings.get("target_usd_year"):
        for fuel, year in (settings.get("user_fuel_usd_year", {}) or {}).items():
            user_fuel_price.loc[user_fuel_price["fuel"] == fuel, "price"] = (
                inflation_price_adjustment(
                    user_fuel_price.loc[user_fuel_price["fuel"] == fuel, "price"],
                    year,
                    settings["target_usd_year"],
                    data_location=settings["data_location"],
                    table_name=settings["dollar_year_table"],
                )
            )
    if df is not None:
        user_fuel_price = pd.concat([df, user_fuel_price])
    return user_fuel_price
