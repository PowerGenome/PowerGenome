"""
Load fuel prices needed for the model
"""

import pandas as pd


def fuel_cost_table(fuel_costs, generators, settings):

    unique_fuels = generators["Fuel"].drop_duplicates()
    model_year_costs = fuel_costs.loc[fuel_costs["year"] == settings["model_year"], :]
    fuel_df = pd.DataFrame(unique_fuels)

    fuel_price_map = {
        row.full_fuel_name: row.price
        for row in model_year_costs.itertuples(index=False, name="row")
    }

    emission_dict = settings["fuel_emission_factors"]
    fuel_emission_map = {}
    for full_fuel_name in fuel_price_map.keys():
        base_fuel_name = full_fuel_name.split("_")[-1]
        if base_fuel_name in emission_dict.keys():
            fuel_emission_map[full_fuel_name] = emission_dict[base_fuel_name]
        else:
            fuel_emission_map[full_fuel_name] = 0

    ccs_fuels = generators.loc[generators["Fuel"].str.contains("ccs"), "Fuel"].unique()
    for ccs_fuel in ccs_fuels:
        # keep the non-ccs price
        base_name = ("_").join(ccs_fuel.split("_")[:-1])
        fuel_price_map[ccs_fuel] = fuel_price_map[base_name]
        fuel_emission_map[ccs_fuel] = fuel_emission_map[base_name]

    fuel_df["Cost_per_MMBtu"] = fuel_df["Fuel"].map(fuel_price_map)
    fuel_df["CO2_content_tons_per_MMBtu"] = fuel_df["Fuel"].map(fuel_emission_map)

    # Slow to loop through all of the rows this way but the df shouldn't be too long
    fuel_df = fuel_df.apply(adjust_ccs_fuels, axis=1, settings=settings)
    fuel_df = add_carbon_tax(fuel_df, settings)
    fuel_df["Cost_per_MMBtu"] = fuel_df["Cost_per_MMBtu"].round(2)
    fuel_df["CO2_content_tons_per_MMBtu"] = fuel_df["CO2_content_tons_per_MMBtu"].round(
        5
    )
    fuel_df.fillna(0, inplace=True)

    return fuel_df


def adjust_ccs_fuels(ccs_fuel_row, settings):

    if "ccs" in ccs_fuel_row["Fuel"]:

        # USD/tonne disposal
        disposal_cost = settings["ccs_disposal_cost"]

        base_fuel_name = ("_").join(ccs_fuel_row["Fuel"].split("_")[-2:])
        capture_rate = settings["ccs_capture_rate"][base_fuel_name]

        co2_captured = ccs_fuel_row["CO2_content_tons_per_MMBtu"] * capture_rate

        ccs_fuel_row["CO2_content_tons_per_MMBtu"] -= co2_captured
        ccs_fuel_row["Cost_per_MMBtu"] += co2_captured * disposal_cost

    else:
        pass

    return ccs_fuel_row


def add_carbon_tax(fuel_df, settings):

    if "carbon_tax" not in settings.keys():
        ctax = 0
    else:
        ctax = settings["carbon_tax"]

    fuel_df.loc[:, "Cost_per_MMBtu"] = fuel_df.loc[:, "Cost_per_MMBtu"] + (
        fuel_df.loc[:, "CO2_content_tons_per_MMBtu"] * ctax
    )

    return fuel_df
