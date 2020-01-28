# Read in and add external inputs (user-supplied files) to PowerGenome outputs

import pandas as pd
from powergenome.load_profiles import load_curves


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
    # scenarios = settings["demand_response_resources"][year][resource_name]["scenarios"]
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


def make_distributed_gen_profiles(dg_profiles_path, pudl_engine, settings):
    """Create 8760 annual generation profiles for distributed generation in regions.
    Uses a distribution loss parameter in the settings file when DG generation is
    defined a fraction of delivered load.

    Parameters
    ----------
    dg_profiles_path : path-like
        Where to load the file from
    pudl_engine : sqlalchemy.Engine
        A sqlalchemy connection for use by pandas. Needed to create base load profiles.
    settings : dict
        User-defined parameters from a settings file

    Returns
    -------
    DataFrame
        Hourly generation profiles for DG resources in each region. Not all regions
        need to be accounted for.

    Raises
    ------
    KeyError
        If the calculation method specified in settings is not 'capacity' or 'fraction_load'
    """

    year = settings["model_year"]

    hourly_norm_profiles = pd.read_csv(dg_profiles_path)
    profile_regions = hourly_norm_profiles.columns

    dg_calc_methods = settings["distributed_gen_method"]
    dg_calc_values = settings["distributed_gen_values"]

    assert (
        year in dg_calc_values.keys()
    ), f"The years in settings parameter 'distributed_gen_values' do not match the model years."

    for region, values in dg_calc_values[year].items():
        assert region in set(profile_regions), (
            "The profile regions in settings parameter 'distributed_gen_values' do not\n"
            f"match the regions in {settings['distributed_gen_profiles_fn']} for year {year}"
        )

    if "fraction_load" in dg_calc_methods.values():
        regional_load = load_curves(pudl_engine, settings)

    dg_hourly_gen = pd.DataFrame(columns=dg_calc_methods.keys())

    for region, method in dg_calc_methods.items():
        region_norm_profile = hourly_norm_profiles[region]
        region_calc_value = dg_calc_values[year][region]

        if method == "capacity":
            dg_hourly_gen[region] = calc_dg_capacity_method(
                region_norm_profile, region_calc_value
            )
        elif method == "fraction_load":
            region_load = regional_load[region]
            dg_hourly_gen[region] = calc_dg_frac_load_method(
                region_norm_profile, region_calc_value, region_load, settings
            )
        else:
            raise KeyError(
                "The settings parameter 'distributed_gen_method' can only have key "
                "values of 'capapacity' or 'fraction_load' for each region.\n"
                f"The value in your settings file is {method}"
            )

    return dg_hourly_gen


def calc_dg_capacity_method(dg_profile, dg_capacity):
    """Calculate the hourly distributed generation in a single region when given
    installed capacity.

    Parameters
    ----------
    dg_profile : Series
        Hourly normalized generation profile
    dg_capacity : float
        Total installed DG capacity

    Returns
    -------
    Series
        8760 hourly generation
    """

    hourly_gen = dg_profile * dg_capacity

    return hourly_gen.values


def calc_dg_frac_load_method(dg_profile, dg_requirement, regional_load, settings):
    """Calculate the hourly distributed generation in a single region where generation
    required to be a fraction of total sales.

    Parameters
    ----------
    dg_profile : Series
        Hourly normalized generation profile
    dg_requirement : float
        The fraction of total sales that DG must constitute
    regional_load : Series
        Hourly load for a given region
    settings : dict
        User-defined parameters from a settings file

    Returns
    -------
    Series
        8760 hourly generation
    """

    annual_load = regional_load.sum()
    dg_capacity_factor = dg_profile.mean()
    distribution_loss = settings["avg_distribution_loss"]

    required_dg_gen = annual_load * dg_requirement * (1 - distribution_loss)
    dg_capacity = required_dg_gen / 8760 / dg_capacity_factor

    hourly_gen = dg_profile * dg_capacity

    return hourly_gen
