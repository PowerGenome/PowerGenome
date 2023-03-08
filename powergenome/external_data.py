# Read in and add external inputs (user-supplied files) to PowerGenome outputs

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, List

from powergenome.util import snake_case_col, remove_feb_29
from powergenome.price_adjustment import inflation_price_adjustment

logger = logging.getLogger(__name__)


def make_demand_response_profiles(
    path: Path, resource_name: str, year: int, scenario: str
) -> pd.DataFrame:
    """Read files with DR profiles across years and scenarios. Return the hourly
    load profiles for a single resource in the model year.

    Parameters
    ----------
    path : path-like
        Where to load the file from
    resource_name : str
        Name of of the demand response resource
    year : int
        Year of data to use from the user demand response file
    scenario : str
        Name of scenario to use from the user demand response file

    Returns
    -------
    DataFrame
        Hourly profiles of DR load for each region where the resource is available.
        Column names are the regions plus 'scenario'.
    """

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

    if len(resource_df) == 8784:
        remove_feb_29(resource_df)

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
    fraction_shiftable = settings["flexible_demand_resources"][year][resource_name][
        "fraction_shiftable"
    ]

    # peak_load = df.groupby(["scenario"]).max()
    peak_load = df.max()
    shiftable_capacity = peak_load * fraction_shiftable

    return shiftable_capacity


def add_resource_max_cap_spur(
    new_resource_df: pd.DataFrame, settings: dict, capacity_col: str = "Max_Cap_MW"
) -> pd.DataFrame:
    """Load user supplied maximum capacity and spur line data for new resources. Add
    those values to the resources dataframe.

    Parameters
    ----------
    new_resource_df : pd.DataFrame
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
    pd.DataFrame
        A modified version of new_resource_df with spur_miles and the maximum
        capacity for a resource. Copies of a resource within a region should have a
        cluster name to uniquely identify them.
    """

    defaults = {
        "cluster": 1,
        "spur_miles": 0,
        capacity_col: -1,
        "interconnect_annuity": 0,
        "interconnect_capex_mw": 0,
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
            new_resource_df["technology"]
            .str.replace("_\*", "_all")
            .str.contains(tech.replace("_\*", "_all"), case=False)
        )
        if mask.sum() > 1:
            resources = new_resource_df.loc[mask, "technology"].to_list()
            raise ValueError(
                f"Resource {tech} in region {region} from file "
                f"{settings['capacity_limit_spur_fn']} matches multiple resources:"
                f"\n{resources}"
            )
        else:
            pass
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
        new_resource_df["interconnect_annuity"], 2017, settings["target_usd_year"]
    )
    return new_resource_df


def make_generator_variability(
    df: pd.DataFrame, remove_feb_29: bool = True
) -> pd.DataFrame:
    """Make a generator variability dataframe with normalized (0-1) hourly profiles
    for each resource in resource_df.

    Any resources that do not have a profile in column
    `profile` are assumed to have constant hourly profiles with a value of 1.
    February 29 is removed from any profiles of length 8784 (leap year).

    Parameters
    ----------
    df
        Dataframe with a single row for each resource. May have column `profile`.

    Returns
    -------
    pd.DataFrame
        A new dataframe with one column for each row of `df` and 8760 rows
        of generation profiles.

    Examples
    --------
    >>> df = pd.DataFrame({'profile': [[0] * 8760, np.ones(8784) / 2, None]})
    >>> make_generator_variability(df)
            0    1    2
    0     0.0  0.5  1.0
    1     0.0  0.5  1.0
    2     0.0  0.5  1.0
    3     0.0  0.5  1.0
    4     0.0  0.5  1.0
    ...   ...  ...  ...
    8755  0.0  0.5  1.0
    8756  0.0  0.5  1.0
    8757  0.0  0.5  1.0
    8758  0.0  0.5  1.0
    8759  0.0  0.5  1.0
    <BLANKLINE>
    [8760 rows x 3 columns]
    """

    def profile_len(x: Any) -> int:
        if isinstance(x, (list, np.ndarray)):
            return len(x)
        return 1

    def format_profile(
        x: Any, remove_feb_29: bool = True, hours: int = 8760
    ) -> np.ndarray:
        # from IPython import embed
        # embed()
        if isinstance(x, np.ndarray):
            if len(x) == 8784:
                # Remove February 29 from leap year
                if remove_feb_29:
                    return np.delete(x, slice(1416, 1440))
            return x
        if isinstance(x, list):
            return format_profile(np.array(x), remove_feb_29, hours)
        # Fill missing with default [1, ...]
        return np.ones(hours, dtype=float)

    if "profile" in df:
        hours = df["profile"].apply(profile_len).max()
        if remove_feb_29 and hours == 8784:
            hours = 8760
        kwargs = {"remove_feb_29": remove_feb_29, "hours": hours}
        profiles = np.column_stack(df["profile"].apply(format_profile, **kwargs).values)
        # Make sure values are not less than 0
        profiles = np.where(profiles >= 0, profiles, 0)
    # elif not remove_feb_29:
    #     profiles = np.ones((8760, len(df)), dtype=float)
    else:
        profiles = np.ones((8760, len(df)), dtype=float)
    return pd.DataFrame(profiles, columns=np.arange(len(df)).astype(str))


def load_policy_scenarios(settings: dict) -> pd.DataFrame:
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

    region_agg_map = reverse_dict_of_lists(settings.get("region_aggregations", {}))

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
                try:
                    df.loc[
                        (df["region"] == region) & (df["technology"] == tech),
                        "Existing_Cap_MW",
                    ] = wind_pv_model_region_capacity.loc[
                        idx[region, tech], "nameplate_capacity_mw"
                    ]
                except:
                    pass

    df = df.set_index(["region", "technology", "cluster"])

    return df


def make_usr_demand_profiles(path, settings):
    idx = pd.IndexSlice
    year = settings["model_year"]
    scenario = settings.get("electrification")
    if not scenario:
        scenario_file = settings.get("scenario_definitions_fn")
        load_file = settings.get("regional_load_fn")
        raise KeyError(
            f"The scenario definitions file {scenario_file} must have a column "
            f"'electrification' with values that correspond to the second row of "
            f"{load_file}"
        )

    df = pd.read_csv(path, header=[0, 1, 2])
    scenario_df = df.loc[:, idx[str(year), scenario]]

    return scenario_df


def load_user_tx_costs(
    path: Path, model_regions: List[str], target_usd_year: int = None
) -> pd.DataFrame:
    """Load a user data file with cost and line loss of each interregional transmission
    line. Map the region names to zones (z1 to zM) and adjust the total cost columns
    to the dollar year specified in settings.

    Parameters
    ----------
    path : Path
        Path to the CSV file, which should have columns "start_region", "dest_region",
        "total_interconnect_annuity_mw", "total_interconnect_cost_mw", and "dollar_year".
    model_regions : List[str]
        List of model region names. Should be sorted to match order in other functions.
    target_usd_year : int, optional
        Desired final dollar year of cost columns, by default None. If None, no adjustment
        is made.

    Returns
    -------
    pd.DataFrame
        Cost and line loss of each potential transmission line between model regions.
        Contains columns "start_region", "dest_region", "zone_1", "zone_2",
        "total_interconnect_annuity_mw", "total_interconnect_cost_mw", "dollar_year",
        and "adjusted_dollar_year".
    """
    df = pd.read_csv(path)

    zones = model_regions
    zone_num_map = {
        zone: f"z{number + 1}" for zone, number in zip(zones, range(len(zones)))
    }
    df["zone_1"] = df["start_region"].map(zone_num_map)
    df["zone_2"] = df["dest_region"].map(zone_num_map)
    df = df.dropna(subset=["zone_1", "zone_2"])

    if target_usd_year:
        adjusted_annuities = []
        adjusted_costs = []
        for row in df.itertuples():
            adj_annuity = inflation_price_adjustment(
                row.total_interconnect_annuity_mw, row.dollar_year, target_usd_year
            )
            adjusted_annuities.append(adj_annuity)

            adj_cost = inflation_price_adjustment(
                row.total_interconnect_cost_mw, row.dollar_year, target_usd_year
            )
            adjusted_costs.append(adj_cost)
        df["total_interconnect_annuity_mw"] = adjusted_annuities
        df["total_interconnect_cost_mw"] = adjusted_costs
        df["adjusted_dollar_year"] = target_usd_year

    return df


def insert_user_tx_costs(tx_df: pd.DataFrame, user_costs: pd.DataFrame) -> pd.DataFrame:
    """Insert costs and line loss from the user. Can include more lines than were in the
    original transmission dataframe to create lines with zero existing capacity.

    Parameters
    ----------
    tx_df : pd.DataFrame
        Dataframe of interregional transmission lines created by PG. Must have the column
        "Network_Lines" (integer from 1 to N), a column for each network zone (z1 to zM),
        "Line_Max_Flow_MW", "Line_Min_Flow_MW", and "transmission_path_name".
    user_costs : pd.DataFrame
        Dataframe of costs and line loss created by the user. Should have columns
        "zone_1" and "zone_2" (possible values are z1 through zM), "total_interconnect_annuity_mw",
        "total_interconnect_cost_mw", and "total_line_loss_frac". Cost values should
        already be adjusted to the desired dollar year.

    Returns
    -------
    pd.DataFrame
        Supplemented interregional transmission lines
    """

    unused_lines = []
    for row in user_costs.itertuples():
        line_row = tx_df.loc[(tx_df[row.zone_1] != 0) & (tx_df[row.zone_2] != 0), :]
        assert not len(line_row) > 1

        if line_row.empty:
            unused_lines.append(row)
        tx_df.loc[
            line_row.index, "Line_Reinforcement_Cost_per_MWyr"
        ] = row.total_interconnect_annuity_mw
        tx_df.loc[
            line_row.index, "Line_Reinforcement_Cost_per_MW"
        ] = row.total_interconnect_cost_mw
        tx_df.loc[line_row.index, "Line_Loss_Percentage"] = row.total_line_loss_frac

    unused_line_df = pd.DataFrame(unused_lines)
    unused_line_df = unused_line_df.rename(
        columns={
            "total_interconnect_annuity_mw": "Line_Reinforcement_Cost_per_MWyr",
            "total_interconnect_cost_mw": "Line_Reinforcement_Cost_per_MW",
            "total_line_loss_frac": "Line_Loss_Percentage",
        }
    )

    zone_cols = [c for c in tx_df.columns if c[0] == "z"]
    unused_line_df[zone_cols] = 0
    for idx, row in unused_line_df.iterrows():
        unused_line_df.loc[idx, row["zone_1"]] = 1
        unused_line_df.loc[idx, row["zone_2"]] = -1
        unused_line_df.loc[
            idx, "transmission_path_name"
        ] = f"{row.start_region}_to_{row.dest_region}"

    cols = [c for c in tx_df.columns if c in unused_line_df.columns]
    tx_df = pd.concat([tx_df, unused_line_df[cols]], ignore_index=True)
    tx_df["Network_Lines"] = range(1, len(tx_df) + 1)
    tx_df["Line_Max_Flow_MW"] = tx_df["Line_Max_Flow_MW"].fillna(0)
    tx_df["Line_Min_Flow_MW"] = tx_df["Line_Max_Flow_MW"].fillna(0)

    return tx_df
