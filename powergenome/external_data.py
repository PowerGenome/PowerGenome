# Read in and add external inputs (user-supplied files) to PowerGenome outputs

import pandas as pd

from powergenome.util import snake_case_col


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


def add_resource_max_cap_spur_line(
    new_resource_df, settings, capacity_col="Max_Cap_MW"
):
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
        `capacity_limit_spur_line_fn` (the file to load).
    capacity_col : str, optional
        The column that indicates maximum capacity constraints, by default "Max_Cap_MW"

    Returns
    -------
    DataFrame
        A modified version of new_resource_df with spur_line_miles and the maximum
        capacity for a resource. Copies of a resource within a region should have a
        cluster name to uniquely identify them.
    """

    path = settings["input_folder"] / settings["capacity_limit_spur_line_fn"]
    df = pd.read_csv(path)

    required_cols = [
        "region",
        "technology",
        "cluster",
        "spur_line_miles",
        "max_capacity",
    ]

    for col in required_cols:
        assert (
            col in df.columns
        ), f"The capacity limit/spur line distance file must have column {col}"

    grouped_df = df.groupby(["region", "technology"])

    new_resource_df[capacity_col] = -1
    new_resource_df["spur_line_miles"] = 0
    for (region, tech), _df in grouped_df:
        mask = (new_resource_df["region"] == region) & (
            new_resource_df["technology"].str.lower().str.contains(tech.lower())
        )

        assert len(new_resource_df.loc[mask, :]) == len(_df), (
            f"There is a mismatch between the number of {tech} resources in {region} "
            f"and the number of spur line distances provided in"
            f" {settings['capacity_limit_spur_line_fn']}"
        )

        new_resource_df.loc[mask, capacity_col] = _df["max_capacity"].values
        new_resource_df.loc[mask, "spur_line_miles"] = _df["spur_line_miles"].values
        new_resource_df.loc[mask, "cluster"] = _df["cluster"].values

        if mask.sum() > 1:
            assert new_resource_df.loc[mask, "cluster"].isna().sum() == 0, (
                f"Error with cluster names in user-supplied file"
                f" {settings['capacity_limit_spur_line_fn']}.\n"
                f"The resource {tech} in region {region} has multiple rows and each row"
                " must have a cluster value."
            )

    return new_resource_df


def make_generator_variability(resource_df, settings, resource_col="Resource"):
    """Make a generator variability dataframe with normalized (0-1) hourly profiles
    for each resource in resource_df. Any resources that are not supplied in the user
    file are assumed to have constant hourly profiles with a value of 1.

    Parameters
    ----------
    resource_df : DataFrame
        All resources (new and existing), with a single row for each resource. Must have
        columns of `region` and `cluster` in addition to a resource/technology column
        that can be changed with the parameter `resource_col`.
    settings : dict
        User-defined parameters from a settings file. Should have keys of `input_folder`
        (a Path object of where to find user-supplied data) and
        `resource_variability_fn` (the file to load).
    resource_col : str, optional
        Name of the column with resource/technology names, by default "Resource"

    Returns
    -------
    DataFrame
        A new DataFrame with one column for each row of `resource_df` and 8760 rows
        of generation profiles.

    Raises
    ------
    KeyError
        More than one match from the user supplied data was found for a resource in
        `resource_df`. These matches are done using str.contains() with names that have
        been transformed to snake case. e.g. the user file might have `PV` as a resource
        and `resource_df` might have both `UtilityPV` and `IndustrialPV`.
    """

    # Load resource variability (8760 values) and make it into a df with columns
    # 'region', 'Resource', 'cluster', and then the hourly values.
    path = settings["input_folder"] / settings["resource_variability_fn"]
    df = pd.read_csv(path, index_col=0)
    df.index = [str(x) for x in df.index]
    df = df.T

    region_list = [x.split(".")[0] for x in df.index]
    df = df.reset_index()
    df.rename(columns={"index": "region"}, inplace=True)
    df["region"] = region_list
    df["Resource"] = snake_case_col(df["Resource"])
    df["cluster"] = df["cluster"].fillna(1)

    # Make a modified version of resource_df, where a new _resource column is populated
    # with the names from generator variability. Resource has already been snake_case
    # transformed.
    df_list = []
    for resource, _df in resource_df.groupby("Resource"):
        _df["cluster"] = _df["cluster"].fillna(1)
        count = 0
        for r in df.Resource.unique():
            if r in resource:
                _df["_resource"] = r
                count += 1
        if count > 1:
            raise KeyError(
                f"More than one name match was found for {resource} in the generator"
                " variability file."
            )
        df_list.append(_df)
    mod_resource_df = pd.concat(df_list)
    mod_resource_df = mod_resource_df.sort_values("R_ID")

    keep_cols = ["Resource_x"] + [str(x) for x in range(1, 8761)]

    gen_variability = (
        mod_resource_df.merge(
            df.reset_index(drop=True),
            left_on=["region", "_resource", "cluster"],
            right_on=["region", "Resource", "cluster"],
            how="left",
        )
        .loc[:, keep_cols]
        .fillna(1)
        .T
    )

    gen_variability = gen_variability.rename(index={"Resource_x": "Resource"})
    gen_variability.columns = gen_variability.loc["Resource", :]
    gen_variability = gen_variability.drop(index="Resource")
    gen_variability.index = [int(x) for x in gen_variability.index]

    assert (
        any(gen_variability.isna().any()) is False
    ), "Something went wrong creating the gen variability file. The data has some NaN values"

    return gen_variability


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

    path = settings["input_folder"] / settings["emission_policies_fn"]
    policies = pd.read_csv(path, na_values=["None", "none"])

    # Update the policies. The column `copy_case_id` can be used to copy values from
    # another policy to reduce human copy/paste errors.
    if "copy_case_id" in policies.columns:
        grouped_policies = policies.groupby(["year", "region", "copy_case_id"])
        df_list = []
        for (year, region, copy_case_id), _df in grouped_policies:

            if copy_case_id in policies.case_id.tolist():
                for col in _df.columns[2:]:
                    _df.loc[:, col] = policies.loc[
                        (policies.year == year)
                        & (policies.case_id == copy_case_id)
                        & (policies.region == region),
                        :,
                    ].squeeze()[col]
            df_list.append(_df)

        updated_policies = pd.concat(df_list)
        policies.loc[updated_policies.index, :] = updated_policies

    policies = policies.drop(columns="copy_case_id")
    policies = policies.set_index(["case_id", "year"])

    return policies
