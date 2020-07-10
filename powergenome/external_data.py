# Read in and add external inputs (user-supplied files) to PowerGenome outputs

import logging
import numpy as np
import pandas as pd
from pathlib import Path

from powergenome.price_adjustment import inflation_price_adjustment
from powergenome.util import snake_case_col

logger = logging.getLogger(__name__)


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


def add_resource_max_cap_spur(new_resource_df, settings, capacity_col="Max_Cap_MW"):
    """Load user supplied maximum capacity and spur line data for new resources. Add
    those values to the resources dataframe.

    Parameters
    ----------
    new_resource_df : DataFrame
        New resources that can be built. Each row should be a single resource, with
        columns 'region' and 'technology'. The number of copies for a region/resource
        (e.g. more than one UtilityPV resource in a region) should match what is
        given in the user-supplied file.
    settings : dict
        User-defined parameters from a settings file. Should have keys of `input_folder`
        (a Path object of where to find user-supplied data) and
        `capacity_limit_spur_fn` (the file to load).
    capacity_col : str, optional
        The column that indicates maximum capacity constraints, by default "Max_Cap_MW"

    Returns
    -------
    DataFrame
        A modified version of new_resource_df with spur_miles and the maximum
        capacity for a resource. Copies of a resource within a region should have a
        cluster name to uniquely identify them.
    """

    defaults = {
        "cluster": 1,
        "spur_miles": 0,
        capacity_col: -1,
        "interconnect_annuity": 0,
    }
    # Prepare file
    path = Path(settings["input_folder"]) / settings["capacity_limit_spur_fn"]
    df = pd.read_csv(path)
    for col in "region", "technology":
        if col not in df:
            raise KeyError(f"The max capacity/spur file must have column {col}")
    for key, value in defaults.items():
        df[key] = df[key].fillna(value) if key in df else value
        new_resource_df[key] = (
            new_resource_df[key].fillna(value) if key in new_resource_df else value
        )
    # Update resources
    grouped_df = df.groupby(["region", "technology"])
    for (region, tech), _df in grouped_df:
        mask = (new_resource_df["region"] == region) & (
            new_resource_df["technology"].str.lower().str.contains(tech.lower())
        )
        if mask.sum() > 1:
            raise ValueError(
                f"Resource {tech} in region {region} from file "
                f"{settings['capacity_limit_spur_fn']} matches multiple resources"
            )
        for key, value in defaults.items():
            _key = "max_capacity" if key == capacity_col else key
            new_resource_df.loc[mask & (new_resource_df[key] == value), key] = _df[
                _key
            ].values
    logger.info(
        f"Inflating external interconnect annuity costs from 2017 to "
        f"{settings['target_usd_year']}"
    )
    new_resource_df["interconnect_annuity"] = inflation_price_adjustment(
        new_resource_df["interconnect_annuity"], 2017, settings["target_usd_year"],
    )
    return new_resource_df


def make_generator_variability(resource_df, settings):
    """Make a generator variability dataframe with normalized (0-1) hourly profiles
    for each resource in resource_df.
    
    Any resources that are not supplied in the file or have a profile in column
    `variability` are assumed to have constant hourly profiles with a value of 1.
    February 29 is removed from any profiles of length 8784 (leap year).

    Matching between the file and resources in `resource_df` are done by:
        - region (exact, case-sensitive)
        - Resource (exact after conversion to snake case)

    Parameters
    ----------
    resource_df : DataFrame
        All resources (new and existing), with a single row for each resource.
        Must have columns `region`, `Resource`, `cluster`, and
        (optional) `variability` which takes precedence over the file.
    settings : dict
        User-defined parameters from a settings file. Should have keys of `input_folder`
        (a Path object of where to find user-supplied data) and
        `resource_variability_fn` (the file to load).

    Returns
    -------
    DataFrame
        A new DataFrame with one column for each row of `resource_df` and 8760 rows
        of generation profiles.

    Raises
    ------
    MergeError
        More than one match from the user supplied data was found for a resource in
        `resource_df`.
    """

    # Load resource variability (8760 values) and make it into a df with columns
    # 'region', 'Resource', 'cluster', and then the hourly values.
    path = Path(settings["input_folder"]) / settings["resource_variability_fn"]
    df = pd.read_csv(path, index_col=0)
    df = pd.DataFrame(
        {
            "region": df.columns.str.replace(r"\.[0-9]+$", ""),
            "Resource": snake_case_col(df.iloc[0]).values,
            "cluster": df.iloc[1].fillna(1).values,
            "variability": list(df.iloc[2:].values.astype(float).T),
        }
    )

    merge = pd.merge(
        resource_df.fillna({"cluster": 1}),
        df,
        how="left",
        on=["region", "Resource", "cluster"],
        suffixes=["__x", "__y"],
        validate="many_to_one",
    ).sort_values("R_ID")
    if "variability__x" in merge:
        # Fill missing with variability from file
        # TODO: Ensure missing always NaN, not also 0, ...
        missing = ~merge["variability__x"].apply(isinstance, args=[np.ndarray])
        merge.loc[missing, "variability__x"] = merge.loc[missing, "variability__y"]
        merge = merge.rename(columns={"variability__x": "variability"}).drop(
            columns=["variability__y"]
        )
    # Fill missing with default (1)
    merge["variability"] = merge["variability"].map(
        lambda x: x if isinstance(x, np.ndarray) else np.ones(8760, dtype=float)
    )
    # Remove February 29 from leap year
    merge["variability"] = merge["variability"].map(
        lambda x: np.delete(x, slice(1416, 1440)) if len(x) == 8784 else x
    )
    return pd.DataFrame(
        np.column_stack(merge["variability"]),
        columns=merge[["R_ID", "Resource"]].astype(str).agg("_".join, axis=1),
        index=range(1, 8761),
    )


def load_policy_scenarios(settings):
    """Load the policy scenarios and copy cases where indicated. The policy file should
    start with columns `case_id` and `year`, and can contain an optional `copy_case_id`.
    Other columns should match the desired output format. The value `None` is included
    in na_values.

    Parameters
    ----------
    settings : dict
        User-defined parameters from a settings file. Should have keys of `input_folder`
        (a Path object of where to find user-supplied data) and
        `emission_policies_fn` (the file to load).

    Returns
    -------
    DataFrame
        Emission policies for each case_id/year.
    """

    path = Path(settings["input_folder"]) / settings["emission_policies_fn"]
    policies = pd.read_csv(path, na_values=["None", "none"])

    # Update the policies. The column `copy_case_id` can be used to copy values from
    # another policy to reduce human copy/paste errors.
    if "copy_case_id" in policies.columns:
        policies = copy_case_values(policies, match_cols=["case_id", "year", "region"])

    policies = policies.drop(columns="copy_case_id")
    policies = policies.set_index(["case_id", "year"])

    return policies


def copy_case_values(df, match_cols):
    """Copy values for one case to others in an external data file. Must have column
    "copy_case_id"

    Parameters
    ----------
    df : DataFrame
        Policies, settings, or some other type of data for each case/year.
    match_cols : list
        The columns used to match cases. Must include "case_id", can also include things
        like "year" and "region".

    Returns
    -------
    DataFrame
        Modified copy of the original with values copied to rows with valid
        "copy_case_id" values.
    """
    match_cols_len = len(match_cols)
    settings_cols = [
        col for col in df.columns if col not in match_cols + ["copy_case_id"]
    ]
    for row in df.itertuples(index=False):
        if not pd.isna(row.copy_case_id):
            # Create a dictionary with keys of each col from match_cols and values
            # from the row. Then replace case_id key with copy_case_id value. This way
            # we get the new case id but same region, year, etc.
            filter_dict = {
                col: value
                for col, value in zip(df.columns[:match_cols_len], row[:match_cols_len])
            }
            filter_dict["case_id"] = row.copy_case_id  # Set to the case we want to copy

            # Use the filter dictionary to filter the original df and assign the values
            # from that row to the row in question.
            # https://stackoverflow.com/a/34162576
            new_values = df.loc[
                (df[list(filter_dict)] == pd.Series(filter_dict)).all(axis=1),
                settings_cols,
            ].values

            filter_dict["case_id"] = row.case_id  # Set back to case we want to modify

            df.loc[
                (df[list(filter_dict)] == pd.Series(filter_dict)).all(axis=1),
                settings_cols,
            ] = new_values

    return df


def load_demand_segments(settings):
    """Load a file of demand segments as defined by the user in CSV.

    Parameters
    ----------
    settings : dict
        User defined PowerGenome settings. Must have the keys "input_folder" and
        "demand_segments_fn".

    Returns
    -------
    DataFrame
        Demand segments with columns such as "Voll", "Demand_segment", etc.
    """

    path = Path(settings["input_folder"]) / settings["demand_segments_fn"]
    demand_segments = pd.read_csv(path)

    return demand_segments


def load_user_genx_settings(settings):
    """Load a file with user-supplied GenX settings and copy values to cases where
    needed.

    Parameters
    ----------
    settings : dict
        User defined PowerGenome settings. Must have the keys "input_folder" and
        "case_genx_settings_fn".

    Returns
    -------
    DataFrame
        Values for GenX settings across cases and years as defined by the user. The
        index is ["case_id", "year"] and columns are GenX parameters.
    """

    path = Path(settings["input_folder"]) / settings["case_genx_settings_fn"]
    genx_case_settings = pd.read_csv(path)

    if "copy_case_id" in genx_case_settings.columns:
        genx_case_settings = copy_case_values(
            genx_case_settings, match_cols=["case_id", "year"]
        )

    genx_case_settings = genx_case_settings.drop(columns=["copy_case_id"])
    genx_case_settings = genx_case_settings.set_index(["case_id", "year"])

    return genx_case_settings


def overwrite_wind_pv_capacity(df, settings):
    """Use external data to overwrite the wind and solarpv capacity extracted from
    EIA860.

    Parameters
    ----------
    df : DataFrame
        Existing generators dataframe, with columns "region", "technology", and
        "Existing_Cap_MW". The technologies should include "Solar Photovoltaic"
        and "Onshore Wind Turbine".
    settings : dict
        User defined PowerGenome settings. Must have the keys "input_folder" and
        "region_wind_pv_cap_fn".

    Returns
    -------
    DataFrame
        Same as input dataframe but with new capacity values for technologies defined
        in the "region_wind_pv_cap_fn" file.
    """
    from powergenome.util import reverse_dict_of_lists

    idx = pd.IndexSlice

    path = settings["input_folder"] / settings["region_wind_pv_cap_fn"]

    wind_pv_ipm_region_capacity = pd.read_csv(path)

    region_agg_map = reverse_dict_of_lists(settings.get("region_aggregations"))

    # Set model_region as IPM_region to start
    wind_pv_ipm_region_capacity["model_region"] = wind_pv_ipm_region_capacity[
        "IPM_Region"
    ]
    # Change any aggregated regions to the user-defined model_region
    wind_pv_ipm_region_capacity.loc[
        wind_pv_ipm_region_capacity["IPM_Region"].isin(region_agg_map), "model_region"
    ] = wind_pv_ipm_region_capacity.loc[
        wind_pv_ipm_region_capacity["IPM_Region"].isin(region_agg_map), "IPM_Region"
    ].map(
        region_agg_map
    )
    wind_pv_model_region_capacity = wind_pv_ipm_region_capacity.groupby(
        ["model_region", "technology"]
    ).sum()

    df = df.reset_index()

    for region in df["region"].unique():
        for tech in ["Solar Photovoltaic", "Onshore Wind Turbine"]:
            if tech in df.query("region == @region")["technology"].to_list():
                df.loc[
                    (df["region"] == region) & (df["technology"] == tech),
                    "Existing_Cap_MW",
                ] = wind_pv_model_region_capacity.loc[
                    idx[region, tech], "nameplate_capacity_mw"
                ]

    df = df.set_index(["region", "technology", "cluster"])

    return df


def make_usr_demand_profiles(path, settings):
    idx = pd.IndexSlice
    year = settings["model_year"]
    scenario = settings["electrification"]

    df = pd.read_csv(path, header=[0, 1, 2])
    scenario_df = df.loc[:, idx[str(year), scenario]]

    return scenario_df
