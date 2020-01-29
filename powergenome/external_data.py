# Read in and add external inputs (user-supplied files) to PowerGenome outputs

import pandas as pd


def make_demand_response_profiles(path, resource_name, settings):
    """Read files with DR profiles across years and scenarios. Return the hourly
    load profiles for a single resource in the model year.

    Parameters
    ----------
    path : path-like
        Where to load the file from
    resource_name : str
        Name of of the demand response resource
    settings : dict
        User-defined parameters from a settings file

    Returns
    -------
    DataFrame
        8760 hourly profiles of DR load for each region where the resource is available.
        Column names are the regions plus 'scenario'.
    """
    year = settings["model_year"]
    scenario = settings["demand_response"]

    df = pd.read_csv(path, header=[0, 1, 2, 3])

    # Use the MultiIndex columns to just get columns with the correct resource listed
    # in the top row of the csv. The resource name is dropped from the columns.
    resource_df = df.loc[:, resource_name]

    assert year in set(
        resource_df.columns.get_level_values(0).astype(int)
    ), f"The model year is not in the years of data for DR resource {resource_name}"

    resource_df = resource_df.loc[:, str(year)]

    assert scenario in set(
        resource_df.columns.get_level_values(0)
    ), f"The scenario {scenario} is not included for DR resource {resource_name}"

    resource_df = resource_df.loc[:, scenario]
    resource_df = resource_df.reset_index(drop=True)

    return resource_df


def demand_response_resource_capacity(df, resource_name, settings):
    """Calculate the maximum capacity value to assign a demand response/DSM resource

    Parameters
    ----------
    df : DataFrame
        Hourly demand profile in each region of a single DR resource
    resource_name : str
        Name of the resource, which should match the settings file
    settings : dict
        User-defined parameters from a settings file

    Returns
    -------
    DataFrame
        Index of scenarios and columns of regions, values represent shiftable capacity.
    """

    year = settings["model_year"]
    fraction_shiftable = settings["demand_response_resources"][year][resource_name][
        "fraction_shiftable"
    ]

    # peak_load = df.groupby(["scenario"]).max()
    peak_load = df.max()
    shiftable_capacity = peak_load * fraction_shiftable

    return shiftable_capacity
