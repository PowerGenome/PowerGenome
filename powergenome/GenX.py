"Functions specific to GenX outputs"

import pandas as pd

from powergenome.external_data import load_policy_scenarios


def add_emission_policies(transmission_df, settings, DistrZones=None):
    """Add emission policies to the transmission dataframe

    Parameters
    ----------
    transmission_df : DataFrame
        Zone to zone transmission constraints
    settings : dict
        User-defined parameters from a settings file. Should have keys of `input_folder`
        (a Path object of where to find user-supplied data) and
        `emission_policies_fn` (the file to load).
    DistrZones : [type], optional
        Placeholder setting, by default None

    Returns
    -------
    DataFrame
        The emission policies provided by user next to the transmission constraints.
    """

    model_year = settings["model_year"]
    case_id = settings["case_id"]

    policies = load_policy_scenarios(settings)
    year_case_policy = policies.loc[(case_id, model_year), :]

    zones = settings["model_regions"]
    zone_num_map = {
        zone: f"z{number + 1}" for zone, number in zip(zones, range(len(zones)))
    }

    zone_cols = ["Region description", "Network_zones", "DistrZones"] + list(
        policies.columns
    )
    zone_df = pd.DataFrame(columns=zone_cols)
    zone_df["Region description"] = zones
    zone_df["Network_zones"] = zone_df["Region description"].map(zone_num_map)

    if DistrZones is None:
        zone_df["DistrZones"] = 0

        # Add code here to make DistrZones something else!

    if year_case_policy["region"].lower() == "all":
        for col, value in year_case_policy.iteritems():
            if col == "CO_2_Max_Mtons":
                zone_df.loc[:, col] = 0
                zone_df.loc[0, col] = value
            else:
                zone_df.loc[:, col] = value
    else:
        print("REGIONAL POLICIES HAVE NOT BEEN IMPLEMENETED YET")

    zone_df = zone_df.drop(columns="region")

    network_df = pd.concat([zone_df, transmission_df.reset_index()], axis=1)

    return network_df
