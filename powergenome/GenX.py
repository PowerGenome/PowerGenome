"Functions specific to GenX outputs"

import logging
import re
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from powergenome.database import get_data
from powergenome.external_data import (
    load_demand_segments,
    load_policy_scenarios,
    make_generator_variability,
)
from powergenome.financials import investment_cost_calculator
from powergenome.settings import auto_fill_settings
from powergenome.time_reduction import kmeans_time_clustering
from powergenome.util import find_region_col, snake_case_col, snake_case_str

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FolderStructure:
    resource: Path
    policy_assignments: Path
    system: Path
    policy: Path


@dataclass(frozen=True, slots=True)
class GenXInputData:
    """A class to abstract the dataframes that will be written out in the GenX case folder."""

    tag: str
    folder: Path
    file_name: str
    dataframe: pd.DataFrame

    def __post_init__(self):
        """Validate the data after initialization."""
        if not isinstance(self.folder, Path):
            object.__setattr__(self, "folder", Path(self.folder))
        if not self.file_name.endswith(".csv"):
            raise ValueError(f"file_name must end with .csv, got {self.file_name}")
        if not isinstance(self.dataframe, (pd.DataFrame, type(None))):
            raise TypeError(
                f"dataframe must be a pandas DataFrame or None, got {type(self.dataframe)}"
            )


INT_COLS = [
    "Inv_Cost_per_MWyr",
    "Fixed_OM_Cost_per_MWyr",
    "Inv_Cost_per_MWhyr",
    "Fixed_OM_Cost_per_MWhyr",
    "Line_Reinforcement_Cost_per_MW_yr",
    "Up_Time",
    "Down_Time",
]

COL_ROUND_VALUES = {
    "Var_OM_Cost_per_MWh": 2,
    "Var_OM_Cost_per_MWh_in": 2,
    "Start_Cost_per_MW": 0,
    "Cost_per_MMBtu": 2,
    "CO2_content_tons_per_MMBtu": 5,
    "Cap_Size": 2,
    "Heat_Rate_MMBTU_per_MWh": 2,
    "distance_mile": 4,
    "Line_Max_Reinforcement_MW": 0,
    "distance_miles": 1,
    "distance_km": 1,
    "Existing_Cap_MW": 1,
    "Existing_Cap_MWh": 1,
    "Max_Cap_MW": 1,
    "Max_Cap_MWh": 1,
    "Min_Cap_MW": 1,
    "Min_Cap_MWh": 1,
    "Line_Loss_Percentage": 4,
}

# RESOURCE_TAGS = ["THERM", "VRE", "MUST_RUN", "STOR", "FLEX", "HYDRO", "LDS"]
RESOURCE_TAGS = ["THERM", "VRE", "MUST_RUN", "STOR", "FLEX", "HYDRO"]

DEFAULT_COLS = ["Resource", "Zone", "New_Build", "Can_Retire", "Existing_Cap_MW"]

# Specific columns for each resource type
THERM_COLUMNS = [
    "Down_Time",
    "Min_Power",
    "Ramp_Up_Percentage",
    "Ramp_Dn_Percentage",
    "Up_Time",
]

VRE_COLUMNS = ["Num_VRE_Bins"]

MUST_RUN_COLUMNS = []

STOR_COLUMNS = [
    "Eff_Down",
    "Eff_Up",
    "Existing_Cap_MWh",
    "Existing_Charge_Cap_MW",
    "Fixed_OM_Cost_Charge_per_MWyr",
    "Fixed_OM_Cost_per_MWhyr",
    "Inv_Cost_Charge_per_MWyr",
    "Inv_Cost_per_MWhyr",
    "LDS",
    "Max_Cap_MWh",
    "Max_Charge_Cap_MW",
    "Max_Duration",
    "Min_Cap_MWh",
    "Min_Charge_Cap_MW",
    "Min_Duration",
    "Self_Disch",
    "Var_OM_Cost_per_MWh_In",
]

FLEX_COLUMNS = [
    "Flexible_Demand_Energy_Eff",
    "Max_Flexible_Demand_Delay",
    "Max_Flexible_Demand_Advance",
    "Var_OM_Cost_per_MWh_In",
]

HYDRO_COLUMNS = [
    "Eff_Up",
    "Eff_Down",
    "Hydro_Energy_to_Power_Ratio",
    "LDS",
    "Min_Power",
    "Ramp_Up_Percentage",
    "Ramp_Dn_Percentage",
]

# For future use
# HYDROGEN_COLUMNS = [
#     "Hydrogen_MWh_Per_Tonne",
#     "Hydrogen_Price_Per_Tonne",
#     "Min_Power",
#     "Ramp_Up_Percentage",
#     "Ramp_Dn_Percentage",
# ]

# Create a mapping of resource tags -> columns
RESOURCE_COLUMNS = {
    "THERM": THERM_COLUMNS,
    "VRE": VRE_COLUMNS,
    "MUST_RUN": MUST_RUN_COLUMNS,
    "STOR": STOR_COLUMNS,
    "FLEX": FLEX_COLUMNS,
    "HYDRO": HYDRO_COLUMNS,
    # 'ELECTROLYZER': HYDROGEN_COLUMNS,
}

# Create a mapping of resource tags -> filenames
RESOURCE_FILENAMES = {
    "THERM": "Thermal.csv",
    "VRE": "Vre.csv",
    "MUST_RUN": "Must_run.csv",
    "STOR": "Storage.csv",
    "FLEX": "Flex_demand.csv",
    "HYDRO": "Hydro.csv",
    # "ELECTROLYZER": "Electrolyzer.csv",
}

MULTISTAGE_COLS = [
    "WACC",
    "Capital_Recovery_Period",
    "Lifetime",
    "Min_Retired_Cap_MW",
    "Min_Retired_Energy_Cap_MW",
    "Min_Retired_Charge_Cap_MW",
]

# Create a mapping to update the policy tag columns
POLICY_TAGS = [
    ("ESR", {"oldtag": "ESR_", "newtag": "ESR_"}),
    ("CAP_RES", {"oldtag": "CapRes_", "newtag": "Derating_factor_"}),
    ("MIN_CAP", {"oldtag": "MinCapTag_", "newtag": "Min_Cap_"}),
    ("MAX_CAP", {"oldtag": "MaxCapTag_", "newtag": "Max_Cap_"}),
]

POLICY_TAGS_FILENAMES = {
    "ESR": "Resource_energy_share_requirement.csv",
    "CAP_RES": "Resource_capacity_reserve_margin.csv",
    "MIN_CAP": "Resource_minimum_capacity_requirement.csv",
    "MAX_CAP": "Resource_maximum_capacity_requirement.csv",
}

INCENTIVE_FILENAMES = {
    "investment_policy": "Investment_incentive.csv",
    "production_policy": "Production_incentive.csv",
    "resource_investment": "Resource_investment_incentive.csv",
    "resource_production": "Resource_production_incentive.csv",
}

PRODUCTION_TYPE_MAP = {
    "mwh": "MWh",
    "tonne_co2": "Tonne_CO2",
    "ton_co2": "Tonne_CO2",
}


def _clean_tech_name(value: str) -> str:
    """
    Normalize technology strings for matching incentives.

    Parameters
    ----------
    value : str or None
        The technology name to normalize. If None or an empty string is provided,
        it will be treated as an empty string.

    Returns
    -------
    str
        The normalized technology name: all whitespace and underscores are removed,
        and the result is converted to lowercase. If the input is None or empty,
        returns an empty string.
    """
    return re.sub(r"[\s_]", "", (value or "")).lower()


def _parse_incentive_suffix(name: str, prefix: str) -> int:
    pattern = re.compile(rf"^{prefix}_Incentive_(\d+)$")
    match = pattern.match(name)
    if not match:
        raise ValueError(
            f"Incentive name '{name}' must follow the pattern {prefix}_Incentive_<number>."
        )
    return int(match.group(1))


def _normalize_production_type(raw_value: Any, policy_name: str) -> str:
    if raw_value is None:
        raise ValueError(
            f"Production incentive '{policy_name}' is missing a production type."
        )
    cleaned = re.sub(r"\s+", "_", str(raw_value)).lower()
    canonical = PRODUCTION_TYPE_MAP.get(cleaned)
    if not canonical:
        raise ValueError(
            f"Production incentive '{policy_name}' has unsupported type '{raw_value}'. "
            f"Allowed types are {sorted(PRODUCTION_TYPE_MAP.values())}."
        )
    return canonical


def _validate_and_sort_incentives(
    incentives: Dict[str, Dict[str, Any]],
    prefix: str,
    require_type: bool = False,
) -> List[Dict[str, Any]]:
    if not incentives:
        return []

    parsed = []
    for name, config in incentives.items():
        suffix = _parse_incentive_suffix(name, prefix)
        if "value" not in (config or {}):
            raise ValueError(f"Incentive '{name}' is missing a 'value'.")
        production_type = None
        if require_type:
            production_type = _normalize_production_type(
                (config or {}).get("type"), name
            )
        technologies = (config or {}).get("technologies", []) or []
        if not isinstance(technologies, list):
            raise TypeError(
                f"Incentive '{name}' technologies must be a list, got {type(technologies)}"
            )
        parsed.append(
            {
                "name": name,
                "config": config or {},
                "suffix": suffix,
                "production_type": production_type,
                "technologies": technologies,
            }
        )

    suffixes = [p["suffix"] for p in parsed]
    if len(suffixes) != len(set(suffixes)):
        raise ValueError(
            f"Duplicate incentive numbers detected in {prefix}_Incentive definitions."
        )
    expected = list(range(1, len(suffixes) + 1))
    if sorted(suffixes) != expected:
        raise ValueError(
            f"Incentive numbering for {prefix}_Incentive must start at 1 and be consecutive. "
            f"Found {sorted(suffixes)}."
        )

    return sorted(parsed, key=lambda item: item["suffix"])


def _build_incentive_policy_df(
    incentives: List[Dict[str, Any]], include_production_type: bool = False
) -> pd.DataFrame:
    if not incentives:
        return pd.DataFrame()

    records = []
    for item in incentives:
        description = item["config"].get("description") or item["name"]
        record = {
            "Policy_ID": item["suffix"],
            "PolicyDescription": description,
            "Value": item["config"].get("value"),
        }
        if include_production_type:
            record["Production_Type"] = item["production_type"]
        records.append(record)

    return pd.DataFrame.from_records(records)


def _build_resource_incentive_df(
    gen_data: pd.DataFrame, incentives: List[Dict[str, Any]]
) -> pd.DataFrame:
    if not incentives or gen_data is None or gen_data.empty:
        return pd.DataFrame()

    if "Resource" not in gen_data.columns or "technology" not in gen_data.columns:
        raise KeyError(
            "Gen data must include 'Resource' and 'technology' columns to assign incentives."
        )

    tech_series = gen_data["technology"].fillna("")
    cleaned_tech = tech_series.map(_clean_tech_name)

    resource_df = pd.DataFrame({"Resource": gen_data["Resource"].values})

    for item in incentives:
        technologies = item.get("technologies", [])
        eligible = pd.Series(False, index=gen_data.index)
        for tech in technologies:
            tech_clean = _clean_tech_name(str(tech))
            if not tech_clean:
                continue
            eligible |= cleaned_tech.str.contains(tech_clean, regex=False)

        resource_df[item["name"]] = eligible.astype(int)

    # Keep only resources that qualify for at least one incentive
    incentive_cols = [item["name"] for item in incentives]
    if incentive_cols:
        qualifying_mask = resource_df[incentive_cols].sum(axis=1) > 0
        resource_df = resource_df.loc[qualifying_mask].reset_index(drop=True)

    return resource_df


def create_incentive_inputs(
    gen_data: pd.DataFrame, settings: Dict[str, Any]
) -> Dict[str, pd.DataFrame]:
    """
    Create incentive policy and assignment tables from settings and generator data.

    This function processes the generator data and settings to generate four DataFrames:
    - Investment incentive policy table
    - Production incentive policy table
    - Resource-to-investment incentive assignment table
    - Resource-to-production incentive assignment table

    Parameters
    ----------
    gen_data : pd.DataFrame
        DataFrame containing generator data. Must include at least 'Resource' and 'technology' columns.
    settings : dict
        Dictionary of model settings, which may include 'investment_incentives' and 'production_incentives'
        as lists of incentive definitions.

    Returns
    -------
    dict of str to pd.DataFrame
        Dictionary with the following keys:
            - 'investment_incentive': DataFrame of investment incentive policies.
            - 'production_incentive': DataFrame of production incentive policies.
            - 'resource_investment_incentive': DataFrame mapping resources to investment incentives.
            - 'resource_production_incentive': DataFrame mapping resources to production incentives.
        If no incentives are defined, returns an empty dictionary.

    Notes
    -----
    The function does not raise exceptions directly, but may propagate errors from internal validation
    if required columns are missing or incentive definitions are invalid.
    """
    inv_incentives = _validate_and_sort_incentives(
        settings.get("investment_incentives", {}), prefix="Inv"
    )
    prod_incentives = _validate_and_sort_incentives(
        settings.get("production_incentives", {}), prefix="Prod", require_type=True
    )

    if not inv_incentives and not prod_incentives:
        return {}

    investment_policy_df = _build_incentive_policy_df(inv_incentives)
    production_policy_df = _build_incentive_policy_df(
        prod_incentives, include_production_type=True
    )

    investment_resource_df = _build_resource_incentive_df(gen_data, inv_incentives)
    production_resource_df = _build_resource_incentive_df(gen_data, prod_incentives)

    return {
        "investment_incentive": investment_policy_df,
        "production_incentive": production_policy_df,
        "resource_investment_incentive": investment_resource_df,
        "resource_production_incentive": production_resource_df,
    }


@auto_fill_settings()
def create_policy_req(col_str_match: str, settings: dict = None) -> pd.DataFrame:
    model_year = settings["model_year"]
    case_id = settings["case_id"]

    policies = load_policy_scenarios(settings)
    policy_cols = [c for c in policies.columns if col_str_match in c]
    if len(policy_cols) == 0:
        return None

    year_case_policy = policies.loc[(case_id, model_year), ["region"] + policy_cols]
    if isinstance(year_case_policy, pd.DataFrame):
        year_case_policy = year_case_policy.dropna(subset=policy_cols)
    elif isinstance(year_case_policy, pd.Series):
        year_case_policy = year_case_policy.dropna()
    else:
        raise TypeError(
            "Somehow the object 'year_case_policy' is not a dataframe of a series."
            "Please submit this as an issue in the PowerGenome repository."
            f"{year_case_policy}"
        )
    # Bug where multiple regions for a case will return this as a df, even if the policy
    # for this case applies to all regions (code below expects a Series)
    ycp_shape = year_case_policy.shape
    if ycp_shape[0] == 1 and len(ycp_shape) > 1:
        year_case_policy = year_case_policy.squeeze()  # convert to series

    zones = settings["model_regions"]
    zone_num_map = settings["zone_num_map"]

    zone_cols = ["Region_description", "Network_zones"] + policy_cols
    zone_df = pd.DataFrame(columns=zone_cols, dtype=float)
    zone_df["Region_description"] = zones
    zone_df["Network_zones"] = zone_df["Region_description"].map(zone_num_map)
    # If there is only one region, assume that the policy is applied across all regions.
    if isinstance(year_case_policy, pd.Series):
        logger.info(
            "Only one zone was found in the emissions policy file."
            " The same emission policies are being applied to all zones."
        )
        for col, value in year_case_policy[policy_cols].iteritems():
            if "CO_2_Max_Mtons" in col:
                zone_df.loc[:, col] = 0
                if value > 0:
                    zone_df.loc[0, col] = value
            else:
                zone_df.loc[:, col] = value
    else:
        for region, col in product(
            year_case_policy["region"].unique(), year_case_policy[policy_cols].columns
        ):
            zone_df.loc[zone_df["Region_description"] == region, col] = (
                year_case_policy.loc[year_case_policy.region == region, col].values[0]
            )

    # zone_df = zone_df.drop(columns="region")

    return zone_df


@auto_fill_settings()
def create_regional_cap_res(settings: dict = None) -> pd.DataFrame:
    """Create a dataframe of regional capacity reserve constraints from settings params

    Parameters
    ----------
    settings : dict
        PowerGenome settings dictionary with parameters 'model_regions' and
        'regional_capacity_reserves'

    Returns
    -------
    pd.DataFrame
        A dataframe with a 'Network_zones' column and one 'CapRes_*' for each capacity
        reserve constraint

    Raises
    ------
    KeyError
        A region listed in a capacity reserve constraint is not a valid model region
    """

    if not settings.get("regional_capacity_reserves"):
        return None
    else:
        zones = settings["model_regions"]
        zone_num_map = settings["zone_num_map"]
        cap_res_cols = list(settings["regional_capacity_reserves"])
        cap_res_df = pd.DataFrame(index=zones, columns=["Network_zones"] + cap_res_cols)
        cap_res_df["Network_zones"] = cap_res_df.index.map(zone_num_map)

        for col, cap_res_dict in settings["regional_capacity_reserves"].items():
            for region, val in cap_res_dict.items():
                if region not in zones:
                    raise KeyError(
                        f"The region {region} in 'regional_capacity_reserves', {col} "
                        "is not a valid model region."
                    )
                cap_res_df.loc[region, col] = val

        cap_res_df = cap_res_df.fillna(0)

        return cap_res_df


def label_cap_res_lines(path_names: List[str], dest_regions: List[str]) -> List[int]:
    """Label if each transmission line is part of a capacity reserve constraint and
    if line flow is into or out of the constraint region.

    Parameters
    ----------
    path_names : List[str]
        String names of transmission lines, with format <region>_to_<region>
    dest_regions : List[str]
        Names of model regions, corresponding to region names used in 'path_names'

    Returns
    -------
    List[int]
        Same length as 'path_names'. Values of 1 mean the line connects a region within
        a capacity reserve constraint to a region outside the constraint. Values of -1
        mean it connects a region outside a capacity reserve constraint to a region
        within the constraint. Values of 0 mean it connects two regions that are both
        within or outside the constraint.
    """
    cap_res_list = []
    for name in path_names:
        s_r = name.split("_to_")[0]
        e_r = name.split("_to_")[-1]
        if (s_r in dest_regions) and e_r not in dest_regions:
            cap_res_list.append(1)
        elif (e_r in dest_regions) and s_r not in dest_regions:
            cap_res_list.append(-1)
        else:
            cap_res_list.append(0)

    assert len(cap_res_list) == len(path_names)

    return cap_res_list


@auto_fill_settings()
def add_cap_res_network(tx_df: pd.DataFrame, settings: dict = None) -> pd.DataFrame:
    """Add capacity reserve colums to the transmission dataframe (Network.csv)

    Parameters
    ----------
    tx_df : pd.DataFrame
        Transmission dataframe with rows for each transmission line, a column
        'transmission_path_name', and columns 'z1', 'z2', etc for each model region. The
        'z*' columns have values of -1 (line is outbound from region), 0 (not connected),
        or 1 (inbound to region).
    settings : dict
        PowerGenome settings, with parameters 'model_regions', 'regional_capacity_reserves',
        and 'cap_res_network_derate_default'.


    Returns
    -------
    pd.DataFrame
        A copy of the input dataframe with additional columns 'CapRes_*', 'DerateCapRes_*',
        and 'CapRes_Excl_*' for each capacity reserve constraint
    """
    if (
        "Transmission Path Name" in tx_df.columns
        and "transmission_path_name" not in tx_df.columns
    ):
        tx_df["transmission_path_name"] = tx_df["Transmission Path Name"]
    original_cols = tx_df.columns.to_list()

    path_names = tx_df["transmission_path_name"].dropna().to_list()
    policy_nums = []

    # Loop through capacity reserve constraints (CapRes_*) and determine network
    # parameters for each
    for cap_res in settings.get("regional_capacity_reserves", {}) or {}:
        cap_res_num = int(cap_res.split("_")[-1])  # the number of the capres constraint
        policy_nums.append(cap_res_num)
        # TODO #179 fix reference to regional_capacity_reserves key of settings dict
        dest_regions = list(
            settings["regional_capacity_reserves"][cap_res].keys()
        )  # list of regions in the CapRes
        # May add ability to have different values by CapRes and line in the future
        tx_df[f"DerateCapRes_{cap_res_num}"] = settings.get(
            "cap_res_network_derate_default", 0.95
        )
        cap_res_excl = label_cap_res_lines(path_names, dest_regions)
        for i, val in enumerate(cap_res_excl):
            tx_df.loc[i, f"CapRes_Excl_{cap_res_num}"] = val

    policy_nums.sort()
    derate_cols = [f"DerateCapRes_{n}" for n in policy_nums]
    excl_cols = [f"CapRes_Excl_{n}" for n in policy_nums]

    return tx_df[original_cols + derate_cols + excl_cols]  # .fillna(0)


@auto_fill_settings()
def add_emission_policies(transmission_df, settings=None):
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

    # Bug where multiple regions for a case will return this as a df, even if the policy
    # for this case applies to all regions (code below expects a Series)
    ycp_shape = year_case_policy.shape
    if ycp_shape[0] == 1 and len(ycp_shape) > 1:
        year_case_policy = year_case_policy.squeeze()  # convert to series

    zones = settings["model_regions"]
    zone_num_map = settings["zone_num_map"]

    zone_cols = ["Region description", "Network_zones"] + list(policies.columns)
    zone_df = pd.DataFrame(columns=zone_cols)
    zone_df["Region description"] = zones
    zone_df["Network_zones"] = zone_df["Region description"].map(zone_num_map)

    # Add code here to make DistrZones something else!
    # If there is only one region, assume that the policy is applied across all regions.
    if isinstance(year_case_policy, pd.Series):
        logger.info(
            "Only one zone was found in the emissions policy file."
            " The same emission policies are being applied to all zones."
        )
        for col, value in year_case_policy.iteritems():
            if col == "CO_2_Max_Mtons":
                zone_df.loc[:, col] = 0
                if value > 0:
                    zone_df.loc[0, col] = value
            else:
                zone_df.loc[:, col] = value
    else:
        for region, col in product(
            year_case_policy["region"].unique(), year_case_policy.columns
        ):
            zone_df.loc[zone_df["Region description"] == region, col] = (
                year_case_policy.loc[year_case_policy.region == region, col].values[0]
            )

    zone_df = zone_df.drop(columns="region")

    network_df = pd.concat([zone_df, transmission_df], axis=1)

    return network_df


def add_misc_gen_values(
    gen_clusters: pd.DataFrame,
    resource_col: str = "Resource",
) -> pd.DataFrame:
    """Add parameter values from the operational constraints table to resources in a generator clusters DataFrame.
    This function loads miscellaneous generator parameter values from the DataManager's
    "operational_constraints" table and assigns them to the appropriate resources in the
    `gen_clusters` DataFrame. The table should contain at least a column matching `resource_col`
    (default "Resource"), and optionally a "region" column. If the "region" column is missing,
    values are applied across all regions. If both "all" and specific region values are provided
    for a resource, the specific region value takes precedence.

    Parameters
    ----------
    gen_clusters : pd.DataFrame
        DataFrame containing generator clusters, with columns "region" and `resource_col`.
    resource_col : str, optional
        Name of the column with resource names in both `gen_clusters` and the operational
        constraints table, by default "Resource".

    Returns
    -------
    pd.DataFrame
        Modified version of `gen_clusters` with new parameter values assigned to resources.

    Notes
    -----
    - Issues a warning if parameter values are missing for any resource in any region.
    - Issues a warning if resources in `gen_clusters` are not found in the input data.
    """
    misc_values = get_data("operational_constraints")
    misc_values[resource_col] = snake_case_col(misc_values[resource_col])

    regions = gen_clusters["region"].unique()

    context = f"Assigning misc generator values from the operational constraints table."
    try:
        region_col = find_region_col(misc_values.columns, context)
    except ValueError:
        region_col = "region"
        misc_values["region"] = "all"

    if "all" in misc_values[region_col].str.lower().to_list():
        df_list = [misc_values]
        for region in regions:
            _df = misc_values.loc[misc_values[region_col].str.lower() == "all", :]
            _df.loc[:, "region"] = region
            df_list.append(_df)

        misc_values = pd.concat(df_list, ignore_index=True)

    # If a user has specific a value for "all" regions plus a value for a specific region,
    # keep the specific region value.
    misc_values = misc_values.loc[
        misc_values[region_col].str.lower() != "all", :
    ].drop_duplicates(subset=[region_col, resource_col], keep="first")

    # regions = [
    #     r for r in misc_values[region_col].fillna("all").unique() if r.lower() != "all"
    # ]

    # missing_regions = [r for r in gen_clusters["region"].unique() if r not in regions]
    # # wrong_regions = [r for r in regions if r not in settings["model_regions"]]
    # if missing_regions:
    #     raise ValueError(
    #         f"The operational data {data_name} is missing regions {missing_regions}."
    #     )

    for tech, _df in misc_values.groupby(resource_col):
        num_tech_regions = len(
            gen_clusters.loc[
                gen_clusters[resource_col].str.contains(tech, case=False, regex=False)
            ].drop_duplicates(subset=["region"])
        )
        num_values = len(_df)
        if num_values < num_tech_regions:
            logger.warning(
                f"The operational constraints table has {num_values} region(s) for the resource "
                f"'{tech}', but the resource is in {num_tech_regions} regions. Check "
                "your input file to ensure values are provided for all appropriate regions."
            )
    generic_resources = []
    for gen_resource in gen_clusters[resource_col].unique():
        for r in regions[::-1]:
            if r in gen_resource:
                gen_resource = gen_resource.replace(r + "_", "")
                generic_resources.append(snake_case_str(gen_resource))
                continue
    generic_resources = set(generic_resources)
    missing_resources = []
    for resource in generic_resources:
        match = False
        for misc_resource in misc_values[resource_col].unique():
            if misc_resource in resource.lower():
                match = True
                continue
        if not match:
            missing_resources.append(resource)

    if missing_resources:
        logger.warning(
            f"The resources {missing_resources} are not included in your operational data "
            f"operational constraints table. This is a warning in case they should have parameters in that table."
        )

    misc_values = misc_values.reset_index(drop=True)
    value_cols = [
        col for col in misc_values.columns if col not in [region_col, resource_col]
    ]

    for idx, row in misc_values.iterrows():
        row_cols = row[value_cols].dropna().index
        gen_clusters.loc[
            (gen_clusters["region"] == row[region_col])
            & (
                snake_case_col(gen_clusters[resource_col]).str.contains(
                    row[resource_col], case=False
                )
            ),
            row_cols,
        ] = row[row_cols].values
    return gen_clusters


@auto_fill_settings()
def reduce_time_domain(
    resource_profiles, load_profiles, settings=None, variable_resources_only=True
):
    demand_segments = load_demand_segments(settings)
    num_hours = len(load_profiles)

    if settings.get("reduce_time_domain", False) is True and num_hours <= 8760:
        days = settings["time_domain_days_per_period"]
        time_periods = settings["time_domain_periods"]
        include_peak_day = settings["include_peak_day"]
        load_weight = settings["demand_weight_factor"]

        results, representative_point, _ = kmeans_time_clustering(
            resource_profiles=resource_profiles,
            load_profiles=load_profiles,
            days_in_group=days,
            num_clusters=time_periods,
            include_peak_day=include_peak_day,
            load_weight=load_weight,
            variable_resources_only=variable_resources_only,
            n_init=settings.get("tdr_n_init", 100),
        )

        reduced_resource_profile = results["resource_profiles"]
        reduced_resource_profile.index.name = "Resource"
        reduced_resource_profile.index = range(1, len(reduced_resource_profile) + 1)
        reduced_load_profile = results["load_profiles"]
        time_series_mapping = results["time_series_mapping"]

        time_index = pd.Series(data=reduced_load_profile.index + 1, name="Time_Index")
        sub_weights = pd.Series(
            data=[x * (days * 24) for x in results["ClusterWeights"]],
            name="Sub_Weights",
        )
        hours_per_period = pd.Series(data=[days * 24], name="Timesteps_per_Rep_Period")
        subperiods = pd.Series(data=[time_periods], name="Rep_Periods")
        reduced_load_output = pd.concat(
            [
                demand_segments,
                subperiods,
                hours_per_period,
                sub_weights,
                time_index,
                reduced_load_profile.round(0),
            ],
            axis=1,
        )

        return (
            reduced_resource_profile,
            reduced_load_output,
            time_series_mapping,
            representative_point,
        )

    if num_hours > 8760:
        logger.warning(
            f"Time series length is {num_hours} hours. Cannot select representative "
            "periods on anything longer than 8760 hours at this time."
        )
    time_index = pd.Series(data=range(1, num_hours + 1), name="Time_Index")
    sub_weights = pd.Series(data=[len(time_index)], name="Sub_Weights")
    hours_per_period = pd.Series(data=[num_hours], name="Timesteps_per_Rep_Period")
    subperiods = pd.Series(data=[1], name="Rep_Periods")

    # Not actually reduced
    load_output = pd.concat(
        [
            demand_segments,
            subperiods,
            hours_per_period,
            sub_weights,
            time_index,
            load_profiles.reset_index(drop=True),
        ],
        axis=1,
    )
    resource_profiles.index = time_index

    return resource_profiles, load_output, None, None


def network_line_loss(transmission: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """Add line loss percentage for each network line between regions.

    Parameters
    ----------
    transmission : pd.DataFrame
        One network line per row with a column "distance_mile"
    settings : dict
        User-defined settings with a parameter "tx_line_loss_100_miles"

    Returns
    -------
    pd.DataFrame
        Same as input but with the new column 'Line_Loss_Percentage'
    """
    if "tx_line_loss_100_miles" not in settings:
        raise KeyError(
            "The parameter 'tx_line_loss_100_miles' is required in your settings file."
        )
    loss_per_100_miles = settings["tx_line_loss_100_miles"]
    if "distance_mile" in transmission.columns:
        distance_col = "distance_mile"
    elif "distance_km" in transmission.columns:
        distance_col = "distance_km"
        loss_per_100_miles *= 0.62137
        logger.debug("Line loss per 100 miles was converted to km.")
    else:
        raise KeyError("No distance column is available in the transmission dataframe")

    transmission["Line_Loss_Percentage"] = (
        transmission[distance_col] / 100 * loss_per_100_miles
    )

    return transmission


def network_reinforcement_cost(
    transmission: pd.DataFrame, settings: dict
) -> pd.DataFrame:
    """Add transmission line reinforcement investment costs (per MW-mile-year)

    Parameters
    ----------
    transmission : pd.DataFrame
        One network line per row with columns "transmission_path_name" and
        "distance_mile".
    settings : dict
        User-defined settings with dictionary "transmission_investment_cost.tx" with
        keys "capex_mw_mile", "wacc", and "investment_years".

    Returns
    -------
    pd.DataFrame
        Same as input but with the new column 'Line_Reinforcement_Cost_per_MW_yr'
    """

    cost_dict = (
        settings.get("transmission_investment_cost", {})
        .get("tx", {})
        .get("capex_mw_mile")
    )
    if not cost_dict:
        raise KeyError(
            "No value for transmission reinforcement costs is included in the settings."
            " These costs are included under transmission_investment_costs.tx."
            "capex_mw_mile.<model_region>. See the `test_settings.yml` file for an "
            "example."
        )
    origin_region_cost = (
        transmission["transmission_path_name"].str.split("_to_").str[0].map(cost_dict)
    )
    dest_region_cost = (
        transmission["transmission_path_name"].str.split("_to_").str[-1].map(cost_dict)
    )

    # Average the costs per mile between origin and destination regions
    line_capex = (origin_region_cost + dest_region_cost) / 2
    line_wacc = (
        settings.get("transmission_investment_cost", {}).get("tx", {}).get("wacc")
    )
    if not line_wacc:
        raise KeyError(
            "No value for the transmission weighted average cost of capital (wacc) is "
            "included in the settings."
            "This numeric value is included under transmission_investment_costs.tx."
            "wacc. See the `test_settings.yml` file for an "
            "example."
        )
    line_inv_period = (
        settings.get("transmission_investment_cost", {})
        .get("tx", {})
        .get("investment_years")
    )
    if not line_inv_period:
        raise KeyError(
            "No value for the transmission investment period is included in the settings."
            "This numeric value is included under transmission_investment_costs.tx."
            "investment_years. See the `test_settings.yml` file for an example."
        )
    line_inv_cost = (
        investment_cost_calculator(
            line_capex,
            line_wacc,
            line_inv_period,
            settings.get("interest_compound_method", "discrete"),
        )
        * transmission["distance_mile"]
    )

    transmission["Line_Reinforcement_Cost_per_MWyr"] = line_inv_cost.round(0)

    return transmission


@auto_fill_settings()
def network_max_reinforcement(
    transmission: pd.DataFrame, settings: dict = None
) -> pd.DataFrame:
    """Add the maximum amount that transmission lines between regions can be reinforced
    in a planning period.

    Parameters
    ----------
    transmission : pd.DataFrame
        One network line per row with the column "Line_Max_Flow_MW"
    settings : dict
        User-defined settings with the parameter "Line_Max_Reinforcement_MW"

    Returns
    -------
    pd.DataFrame
        A copy of the input transmission constraint dataframe with a new column
        `Line_Max_Reinforcement_MW`.
    """

    max_expansion = settings.get("tx_expansion_per_period", 0)
    expansion_mw = settings.get("tx_expansion_mw_per_period", 0)

    if not max_expansion and max_expansion != 0:
        raise KeyError(
            "No value for the transmission expansion allowed in this model period is "
            "included in the settings."
            "This numeric value is included under tx_expansion_per_period. See the "
            "`test_settings.yml` file for an example."
        )
    # if isinstance(max_expansion, dict):
    #     expansion_method = settings.get("tx_expansion_method")
    #     if not expansion_method:
    #         raise KeyError(
    #         "The transmission expansion parameter 'tx_expansion_per_period' is a "
    #         "dictionary. There should also be a settings parameter 'tx_expansion_method' "
    #         "with a dictionary of model region: <type> (either 'capacity' or 'fraction' "
    #         "but it isn't in your settings file."
    #     )
    #     for region, value in max_expansion.items():
    #         existing_tx = transmission.query("Region description == @region")["Line_Max_Flow_MW"]
    #         if expansion_method[region].lower() == "capacity":
    #             transmission.loc[transmission["Region description"] == region,
    #             "Line_Max_Reinforcement_MW"] = value
    #         elif expansion_method[region].lower() == "fraction":
    #             transmission.loc[transmission["Region description"] == region,
    #             "Line_Max_Reinforcement_MW"] = value * existing_tx
    #         else:
    #             raise KeyError(
    #             "The transmission expansion method parameter (tx_expansion_method) "
    #             "should have values of 'capacity' or 'fraction' for each model region. "
    #             f"The value provided was '{expansion_method[region]}'."
    #         )

    # else:
    line_max_flow = transmission["Line_Max_Flow_MW"]
    calculated_values = line_max_flow * max_expansion
    transmission.loc[:, "Line_Max_Reinforcement_MW"] = calculated_values.where(
        calculated_values >= expansion_mw, expansion_mw
    )

    transmission["Line_Max_Reinforcement_MW"] = transmission[
        "Line_Max_Reinforcement_MW"
    ].round(0)

    return transmission


def set_int_cols(df: pd.DataFrame, cols: list = None) -> pd.DataFrame:
    """Set values of some dataframe columns to integers.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    cols : list, optional
        Columns to set as integer, by default None. If none, will use
        `powergenome.GenX.INT_COLS`.

    Returns
    -------
    pd.DataFrame
        Input dataframe with some columns set as integer.
    """
    if not cols:
        cols = INT_COLS

    cols = [c for c in cols if c in df.columns]

    for col in cols:
        df[col] = df[col].fillna(0).astype(int)
    return df


def round_col_values(
    df: pd.DataFrame, col_round_val: Dict[str, int] = None
) -> pd.DataFrame:
    """Round values in columns to a specific sigfig.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    col_round_val : Dict, optional
        Dictionary with key values of column labels and integer values of the number of
        sigfigs, by default None.

    Returns
    -------
    pd.DataFrame
        Same dataframe as input but with rounded values in specified columns.
    """
    if not col_round_val:
        col_round_val = COL_ROUND_VALUES

    col_round_val = {k: v for k, v in col_round_val.items() if k in df.columns}

    for col, value in col_round_val.items():
        df[col] = df[col].fillna(0).round(value)
    return df


def calculate_partial_CES_values(gen_clusters, fuels, settings):
    gens = gen_clusters.copy()
    if settings.get("partial_ces"):
        assert set(fuels["Fuel"]) == set(gens["Fuel"])
        fuel_emission_map = fuels.copy()
        fuel_emission_map = fuel_emission_map.set_index("Fuel")

        gens["co2_emission_rate"] = gens["Heat_Rate_MMBTU_per_MWh"] * gens["Fuel"].map(
            fuel_emission_map["CO2_content_tons_per_MMBtu"]
        )

        # Make the partial CES credit equal to 1 ton minus the emissions rate, but
        # don't include coal plants

        partial_ces = 1 - gens["co2_emission_rate"]

        gens.loc[
            ~(gens["Resource"].str.contains("coal"))
            & (gens["STOR"] == 0)
            & (gens["FLEX"] == 0),
            # & ~(gens["Resource"].str.contains("battery"))
            # & ~(gens["Resource"].str.contains("load_shifting")),
            "CES",
        ] = partial_ces.round(3)
    # else:
    #     gen_clusters = add_genx_model_tags(gen_clusters, settings)

    gens["Zone"] = gens["Zone"]
    gens["Cap_Size"] = gens["Cap_size"]
    gens["Fixed_OM_Cost_per_MWyr"] = gens["Fixed_OM_Cost_per_MWyr"]
    gens["Fixed_OM_Cost_per_MWhyr"] = gens["Fixed_OM_Cost_per_MWhyr"]
    gens["Inv_Cost_per_MWyr"] = gens["Inv_Cost_per_MWyr"]
    gens["Inv_Cost_per_MWhyr"] = gens["Inv_Cost_per_MWhyr"]
    gens["Var_OM_Cost_per_MWh"] = gens["Var_OM_Cost_per_MWh"]
    # gens["Var_OM_Cost_per_MWh_In"] = gens["Var_OM_Cost_per_MWh_in"]
    gens["Start_Cost_per_MW"] = gens["Start_Cost_per_MW"]
    gens["Start_Fuel_MMBTU_per_MW"] = gens["Start_fuel_MMBTU_per_MW"]
    gens["Heat_Rate_MMBTU_per_MWh"] = gens["Heat_Rate_MMBTU_per_MWh"]
    gens["Min_Power"] = gens["Min_Power"]
    # gens["Self_Disch"] = gens["Self_disch"]
    # gens["Eff_Up"] = gens["Eff_up"]
    # gens["Eff_Down"] = gens["Eff_down"]
    # gens["Ramp_Up_Percentage"] = gens["Ramp_Up_percentage"]
    # gens["Ramp_Dn_Percentage"] = gens["Ramp_Dn_percentage"]
    # gens["Up_Time"] = gens["Up_time"]
    # gens["Down_Time"] = gens["Down_time"]
    # gens["Max_Flexible_Demand_Delay"] = gens["Max_DSM_delay"]
    gens["technology"] = gens["Resource"]
    gens["Resource"] = (
        gens["region"] + "_" + gens["technology"] + "_" + gens["cluster"].astype(str)
    )

    return gens


def check_min_power_against_variability(gen_clusters, resource_profile):
    min_gen_levels = resource_profile.min()

    assert len(min_gen_levels) == len(
        gen_clusters
    ), "The number of hourly resource profiles does not match the number of resources"
    # assert (
    #     gen_clusters["Min_Power"].isna().any() is False
    # ), (
    #     "At least one Min_Power value in 'gen_clusters' is null before checking against"
    #     " resource variability"
    # )

    min_gen_levels.index = gen_clusters.index

    gen_clusters["Min_Power"] = gen_clusters["Min_Power"].combine(min_gen_levels, min)

    # assert (
    #     gen_clusters["Min_Power"].isna().any() is False
    # ), (
    #     "At least one Min_Power value in 'gen_clusters' is null. Values were fine "
    #     "before checking against resoruce variability"
    # )

    return gen_clusters


def set_must_run_generation(
    gen_variability: pd.DataFrame, must_run_techs: List[str] = None
) -> pd.DataFrame:
    """Set the generation of must run resources -- those tagged as "MUST_RUN" in
     generators data -- to 1 in all hours.

    Parameters
    ----------
    gen_variability : pd.DataFrame
        Columns are names of resources, rows are floats from 0-1
    must_run_techs : List[str], optional
        Names of the resources selected as must run, by default None

    Returns
    -------
    pd.DataFrame
        Modified version of gen_variability with any must run techs having a value of 1
        in all rows.

    Examples
    --------
    In-memory dataframe:

    >>> gen_variability = pd.DataFrame({
        "gen_1": [0.5, 0.6, 0.7],
        "gen_2": [0.8, 0.9, 1.0],
        "gen_3": [0.0, 0.0, 0.0]
    })
    >>> must_run_techs = ["gen_3"]
    >>> expected_output = pd.DataFrame({
        "gen_1": [0.5, 0.6, 0.7],
        "gen_2": [0.8, 0.9, 1.0],
        "gen_3": [1.0, 1.0, 1.0]
    })
    >>> assert set_must_run_generation(gen_variability, must_run_techs).equals(expected_output)
    """
    for tech in must_run_techs or []:
        if tech not in gen_variability.columns:
            logger.warning(
                f"Trying to set {tech} as a must run resource (hourly generation is) "
                "always 1), but it was not found in the generation variability dataframe."
            )
            continue
        gen_variability.loc[:, tech] = 1.0

    return gen_variability


def calc_emissions_ces_level(network_df, load_df, settings):
    # load_cols = [col for col in load_df.columns if "Load" in col]
    total_load = load_df.sum().sum()

    emissions_limit = settings["emissions_ces_limit"]

    if emissions_limit is not None:
        try:
            ces_value = 1 - (emissions_limit / total_load)
        except TypeError:
            print(emissions_limit, total_load)
            raise TypeError
        network_df["CES"] = ces_value
        network_df["CES"] = network_df["CES"].round(3)

        return network_df
    else:
        return network_df


def fix_min_power_values(
    resource_df: pd.DataFrame,
    gen_profile_df: pd.DataFrame,
    min_power_col: str = "Min_Power",
) -> pd.DataFrame:
    """Fix potentially erroneous min power values for resources with variable generation
    profiles. Any min power values that are higher than the lowest hourly generation
    will be adjusted down to match the lowest hourly generation.


    Parameters
    ----------
    resource_df : pd.DataFrame
        Records of generators/resources. Row order should match column order of
        `gen_profile_df`.
    gen_profile_df : pd.DataFrame
        Hourly generation values for all generators/resources. Column order should match
        row order in `resource_df`.
    min_power_col : str
        Column in `resource_df` that stores the minimum generation power of each
        resource. Default value is "Min_Power".

    Returns
    -------
    pd.DataFrame
        A modified version of `resource_df`. Any rows with minimum power larger than
        hourly generation are adjusted down to match the smallest hourly generation.
    """
    if min_power_col not in resource_df.columns:
        raise ValueError(
            f"When variable generation values against resource min power, the column "
            f"{min_power_col} was not found in the resource dataframe."
        )

    if resource_df.shape[0] != gen_profile_df.shape[1]:
        raise ValueError(
            "When trying to fix min power values, the number of resource dataframe rows"
            f" ({resource_df.shape[0]} rows) does not match the number of variable "
            f"profiles columns ({gen_profile_df.shape[1]} columns)."
        )

    resource_df = resource_df.reset_index(drop=True)
    resource_df.loc[:, "unadjusted_min_power"] = resource_df.loc[:, min_power_col]
    _gen_profile = gen_profile_df.copy(deep=True)
    _gen_profile
    gen_profile_min = _gen_profile.min().reset_index(drop=True)
    mask = (resource_df[min_power_col].fillna(0) > gen_profile_min).values

    logger.debug(
        f"{sum(mask)} resources have {min_power_col} larger than hourly generation."
    )

    resource_df.loc[mask, min_power_col] = gen_profile_min[mask].round(3)

    return resource_df


@auto_fill_settings()
def min_cap_req(settings: dict = None) -> pd.DataFrame:
    """Create a dataframe of minimum capacity requirements for GenX

    Parameters
    ----------
    settings : dict
        Dictionary with user settings. Should include the key `MinCapReg` with nested
        keys of `MinCapTag_*`, then further nested keys `description` and `min_mw`. The
        `MinCapTag_*` should also be listed as values under `model_tag_names`. Any
        technologies eligible for each of the `MinCapTag_*` should have `model_tag_values`
        of 1.

    Returns
    -------
    pd.DataFrame
        A dataframe with minimum capacity constraints formatted for GenX. If `MinCapReq`
        is not included in the settings dictionary it will return None.

    Raises
    ------
    KeyError
        If a `MinCapTag_*` is included under `MinCapReq` but not included in `model_tag_names`
        the function will raise an error.
    """

    c_num = []
    description = []
    min_mw = []

    # if settings.get("MinCapReq"):
    for cap_tag, values in (settings.get("MinCapReq", {}) or {}).items():
        if cap_tag not in settings.get("model_tag_names", []):
            raise KeyError(
                f"The minimum capacity tag {cap_tag} is listed in the settings "
                "'MinCapReq' but not under 'model_tag_names'. You must add it to "
                "'model_tag_names' for the column to appear in Generators_data.csv."
            )

        # It's easy to forget to add all the necessary column names to the
        # generators_columns list in settings.
        if cap_tag not in settings.get("generator_columns", []) and isinstance(
            settings.get("generator_columns"), list
        ):
            settings["generator_columns"].append(cap_tag)

        c_num.append(cap_tag.split("_")[1])
        description.append(values.get("description"))
        min_mw.append(values.get("min_mw"))

    min_cap_df = pd.DataFrame()
    min_cap_df["MinCapReqConstraint"] = c_num
    min_cap_df["Constraint_Description"] = description
    min_cap_df["Min_MW"] = min_mw

    if not min_cap_df.empty:
        return min_cap_df
    else:
        return None


@auto_fill_settings()
def max_cap_req(settings: dict = None) -> pd.DataFrame:
    """Create a dataframe of maximum capacity requirements for GenX

    Parameters
    ----------
    settings : dict
        Dictionary with user settings. Should include the key `MinCapReg` with nested
        keys of `MaxCapTag_*`, then further nested keys `description` and `min_mw`. The
        `MaxCapTag_*` should also be listed as values under `model_tag_names`. Any
        technologies eligible for each of the `MaxCapTag_*` should have `model_tag_values`
        of 1.

    Returns
    -------
    pd.DataFrame
        A dataframe with maximum capacity constraints formatted for GenX. If `MaxCapReq`
        is not included in the settings dictionary it will return None.

    Raises
    ------
    KeyError
        If a `MaxCapTag_*` is included under `MaxCapReq` but not included in `model_tag_names`
        the function will raise an error.
    """

    c_num = []
    description = []
    max_mw = []

    # if settings.get("MaxCapReq"):
    for cap_tag, values in (settings.get("MaxCapReq", {}) or {}).items():
        if cap_tag not in settings.get("model_tag_names", []):
            raise KeyError(
                f"The maximum capacity tag {cap_tag} is listed in the settings "
                "'MaxCapReq' but not under 'model_tag_names'. You must add it to "
                "'model_tag_names' for the column to appear in Generators_data.csv."
            )

        # It's easy to forget to add all the necessary column names to the
        # generators_columns list in settings.
        if cap_tag not in settings.get("generator_columns", []) and isinstance(
            settings.get("generator_columns"), list
        ):
            settings["generator_columns"].append(cap_tag)

        c_num.append(cap_tag.split("_")[1])
        description.append(values.get("description"))
        max_mw.append(values.get("max_mw"))

    max_cap_df = pd.DataFrame()
    max_cap_df["MaxCapReqConstraint"] = c_num
    max_cap_df["Constraint_Description"] = description
    max_cap_df["Max_MW"] = max_mw

    if not max_cap_df.empty:
        return max_cap_df
    else:
        return None


def check_resource_tags(df: pd.DataFrame) -> pd.DataFrame:
    """Check complete generators dataframe to make sure each resource is assigned one,
    and only one, resource tag.

    Parameters
    ----------
    df : pd.DataFrame
        Resource clusters. Should have columns "technology" and "region" in addition
        to the resource tag columns expected by GenX.

    Returns
    -------
    pd.DataFrame
        An unaltered version of the input dataframe.
    """
    tags = [t for t in RESOURCE_TAGS if t in df.columns]
    df_copy = df.loc[:, ["technology", "region"] + tags].copy()
    df_copy[tags] = df_copy[tags].where(df_copy[tags] == 0, 1)
    if not (df_copy[tags].sum(axis=1) == 1).all():
        for idx, row in df_copy.iterrows():
            num_tags = row[tags].sum()
            if num_tags == 0:
                logger.warning(
                    "\n*************************\n"
                    f"The resource {row['technology']} in region {row['region']} does "
                    "not have any assigned resource tags. Check the 'model_tag_values' and "
                    "'regional_tag_values' parameters in your settings file to make sure"
                    "it is assigned one resource tag type from this list:\n\n"
                    f"{RESOURCE_TAGS}\n"
                )
            if num_tags > 1:
                s = row[tags]
                _tags = list(s[s == 1].index)
                logger.warning(
                    "\n*************************\n"
                    f"The resource {row['technology']} in region {row['region']} is "
                    f"assigned {num_tags} resource tags ({_tags}). Check the 'model_tag_values'"
                    " and 'regional_tag_values' parameters in your settings file to make"
                    " sure it is assigned only one resource tag type from this list:\n\n"
                    f"{RESOURCE_TAGS}\n"
                )

        raise ValueError(
            "Use the warnings above to fix the resource tags in your settings file."
        )
    return df


@auto_fill_settings()
def hydro_energy_to_power(
    df: pd.DataFrame,
    default_factor: float = None,
    regional_factors: Dict[str, float] = None,
) -> pd.DataFrame:
    """Calculate the hydro energy to power ratio. Uses average hydro inflow rate and
    multiplied by a factor to calculate the rated number of hours of reservoir hydro
    storage at peak discharge power output. Value minimum is 1.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of resources. Hydro resources should be identified with a "HYDRO" tag
        value of 1
    default_factor : float, optional
        Hydro factor used to scale average inflow rate, by default None
    regional_factors : Dict[str, float], optional
        Specific regional hydro factors used to scale average inflow rate, by default {}

    Returns
    -------
    pd.DataFrame
        Copy of the input dataframe with the new column "Hydro_Energy_to_Power_Ratio"
    """
    if "HYDRO" not in df.columns:
        logger.warning(
            "Generators do not have a column 'HYDRO', so no hydro energy to power ratio is calculated."
        )
        return df
    if not default_factor and not regional_factors:
        logger.warning(
            "No hydro factors have been included in the settings, so no hydro energy to power ratio is calculated."
        )
        return df
    hydro_mask = df["HYDRO"] == 1
    avg_inflow = (
        make_generator_variability(df).mean().reset_index(drop=True) * default_factor
    ).loc[hydro_mask]
    df.loc[hydro_mask, "Hydro_Energy_to_Power_Ratio"] = avg_inflow.where(
        avg_inflow > 1, 1
    )

    for region, factor in (regional_factors or {}).items():
        region_mask = df["region"] == region
        if region_mask.any():
            avg_inflow = (
                make_generator_variability(df).mean().reset_index(drop=True) * factor
            ).loc[hydro_mask & region_mask]
            df.loc[(df["HYDRO"] == 1) & region_mask, "Hydro_Energy_to_Power_Ratio"] = (
                avg_inflow.where(avg_inflow > 1, 1)
            )
    df["Hydro_Energy_to_Power_Ratio"] = df["Hydro_Energy_to_Power_Ratio"].fillna(0)
    return df


def rename_gen_cols(
    df: pd.DataFrame, rename_cols: Dict[str, str] = None
) -> pd.DataFrame:
    """Rename columns in the generators data file.

    By default the only rename so far is "existing_mwh" to "Existing_Cap_MWh".

    Parameters
    ----------
    df : pd.DataFrame
        Final dataframe of generating resources.
    rename_cols : Dict[str, str], optional
        Additional rename pairs, by default None

    Returns
    -------
    pd.DataFrame
        Identical to input dataframe except for renamed columns.
    """

    rename = {
        "capacity_mw": "Existing_Cap_MW",
        "capacity_mwh": "Existing_Cap_MWh",
    }
    if rename_cols:
        rename.update(rename_cols)

    df = df.rename(columns=rename, errors="ignore")

    return df


def add_co2_costs_to_o_m(df: pd.DataFrame) -> pd.DataFrame:
    """Add CO2 pipeline annuity/FOM to fixed O&M and cost per MWh to variable O&M
    using the column names for GenX

    Parameters
    ----------
    df : pd.DataFrame
        Generators dataframe. May include columns "co2_cost_mwh", "co2_pipeline_annuity_mw",
        and "co2_o_m_mw". Must include columns "Var_OM_Cost_per_MWh",
        "Inv_Cost_per_MWyr", and "Fixed_OM_Cost_per_MWyr"

    Returns
    -------
    pd.DataFrame
        Modified version of original df with CO2 pipeline annuity/O&M added to plant
        costs
    """
    if "co2_cost_mwh" in df.columns:
        df["Var_OM_Cost_per_MWh"] += df["co2_cost_mwh"].fillna(0)
    if "co2_pipeline_annuity_mw" in df.columns:
        df["Inv_Cost_per_MWyr"] += df["co2_pipeline_annuity_mw"].fillna(0)
    if "co2_o_m_mw" in df.columns:
        df["Fixed_OM_Cost_per_MWyr"] += df["co2_o_m_mw"].fillna(0)
    if "co2_pipeline_capex_mw" in df.columns:
        df["capex_mw"] += df["co2_pipeline_capex_mw"].fillna(0)

    return df


def cap_retire_within_period(
    gens: pd.DataFrame, first_year: int, last_year: int, capacity_col: str
) -> pd.Series:
    retired_cap = (
        gens.query("retirement_year <= @last_year and retirement_year >= @first_year")
        .groupby("Resource", as_index=False)[[capacity_col, "capacity_mwh"]]
        .sum()
    ).rename(
        columns={
            capacity_col: "Min_Retired_Cap_MW",
            "capacity_mwh": "Min_Retired_Energy_Cap_MW",
        }
    )
    retired_cap["Min_Retired_Charge_Cap_MW"] = 0

    return retired_cap


def check_vre_profiles(
    gen_df: pd.DataFrame,
    gen_var_df: pd.DataFrame,
    vre_cols: List[str] = ["VRE", "HYDRO"],
):
    """Check generation profiles of VRE resources to ensure they are variable. Alert the
    user with a logger warning if any are not.

    Parameters
    ----------
    gen_df : pd.DataFrame
        Dataframe of generators. Must have column "Resource".
    gen_var_df : pd.DataFrame
        Dataframe of hourly generation for each resource. Column names are from the
        "Resource" column of `gen_df`.
    vre_cols : List[str]
        List of boolean columns indicating inclusion in the group of variable resources.
        By default, ["VRE", "HYDRO"].
    """
    var_gen_names = []
    vre_cols = [c for c in vre_cols if c in gen_df.columns]
    if vre_cols:
        for c in vre_cols:
            var_gen_names.extend(gen_df.loc[gen_df[c] == 1, "Resource"].to_list())

        vre_std = gen_var_df[var_gen_names].std()
        if (vre_std == 0).any():
            non_variable = vre_std.loc[vre_std == 0].index.to_list()
            logger.warning(
                f"The variable resources {non_variable} have non-variable generation profiles."
            )


def update_newbuild_canretire(df: pd.DataFrame) -> pd.DataFrame:
    """Update the New_Build and Can_Retire columns in generator data.

    If Can_Retire column doesn't exist, creates it based on New_Build values:
    - Can_Retire = 1 if New_Build != -1, else 0
    - New_Build = 1 if New_Build == 1, else 0

    Parameters
    ----------
    df : pd.DataFrame
        Generator data with New_Build column

    Returns
    -------
    pd.DataFrame
        Modified dataframe with updated Can_Retire and New_Build columns
    """
    if "New_Build" not in df.columns:
        logger.warning("New_Build column not found in generator data")
        return df

    if not df["New_Build"].isin([0, 1, -1]).all():
        logger.warning("New_Build column contains values other than 0, 1, or -1")
        return df

    logger.warning("Upgrading the Can_Retire and New_Build interface")
    df["Can_Retire"] = (df["New_Build"] != -1).astype(int)
    df["New_Build"] = (df["New_Build"] == 1).astype(int)

    return df


def filter_empty_columns(df: pd.DataFrame) -> List[str]:
    """Find columns that have at least one valid value.

    A column is considered valid if it has at least one value that is:
    - Non-zero
    - Not None (Python None)
    - Not "No_fuel" (string)

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe to check

    Returns
    -------
    List[str]
        List of column names that have at least one non-zero and non-None/"No_fuel" value
    """
    # Drop columns that are known to have lists or array values
    drop_cols = ["plant_id_eia", "unit_id_eia", "unit_id_pg", "generator_id", "profile"]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Check for non-None values
    notnull_mask = df.notna().sum() > 0

    # Check for non-"No_fuel" values without forcing mixed dtype casts.
    # Using mask with a string fill value can trigger pandas downcast errors
    # when numeric columns contain inf values.
    def has_non_no_fuel_value(col: pd.Series) -> bool:
        valid = col[col.notna()]
        return valid.ne("No_fuel").any() if not valid.empty else False

    string_notnone_mask = df.apply(has_non_no_fuel_value, axis=0)

    # Check for non-zero values (treat NaN as False)
    nonzero_mask = df.fillna(0).astype(bool).sum() > 0

    # Combine all masks
    valid_cols = nonzero_mask & notnull_mask & string_notnone_mask

    return list(valid_cols[valid_cols].index)


def create_resource_df(df: pd.DataFrame, resource_tag: str) -> pd.DataFrame:
    """Create a dataframe with resource-specific columns for a specific resource type.

    Filters generator data for a specific resource type and removes columns
    specific to other resource types. For storage and thermal resources,
    renames the resource tag column to 'Model'.

    Parameters
    ----------
    df : pd.DataFrame
        Generator data with resource type columns
    resource_tag : str
        Resource type tag (e.g., 'STOR', 'THERM', etc.)

    Returns
    -------
    pd.DataFrame
        Filtered and restructured data for the specific resource type.
        Returns empty DataFrame if resource_tag not in columns.

    """
    logger.info(f"Creating resource data file for {resource_tag}")

    if resource_tag not in df.columns:
        logger.warning(f"Resource tag {resource_tag} not found in dataframe columns")
        return pd.DataFrame()

    # Filter rows for this resource type
    resource_df = df[df[resource_tag] > 0].copy()

    # Get columns to keep for this resource type
    resource_specific_cols = set(RESOURCE_COLUMNS[resource_tag])

    # Find columns with non-zero values
    valid_cols = (
        set(filter_empty_columns(resource_df))
        | set(DEFAULT_COLS)
        | set(resource_specific_cols)
    )
    logger.debug(f"Found {len(valid_cols)} valid columns for {resource_tag}")

    # Get all columns specific to other resource types
    # i.e. columns that do not belong to this resource type
    other_resource_cols = {
        col
        for tag, cols in RESOURCE_COLUMNS.items()
        if tag != resource_tag
        for col in cols
    }
    # Add the resource tag column to the set of other resource columns
    other_resource_cols.add(resource_tag)

    # Find columns to remove: columns that are in other_resource_cols
    # but NOT in resource_specific_cols
    cols_to_remove = (other_resource_cols - resource_specific_cols) & set(df.columns)

    # List of columns to keep
    cols_to_keep = [
        col for col in df.columns if col in valid_cols and col not in cols_to_remove
    ]

    # In the case of STOR and THERM, the resource tag column is also used as
    # 'model' type (e.g., SYM or ASYM for storage, and COMMIT or NOCOMMIT for thermal)
    # Therefore, we rename the resource tag column to 'Model'
    if resource_tag in {"STOR", "THERM"}:
        logger.debug(f"Renaming {resource_tag} column to 'Model'")
        resource_df = resource_df.rename(columns={resource_tag: "Model"})
        cols_to_keep.append("Model")

    # Final check that ALL the resource specific columns are present
    missing_cols = resource_specific_cols - set(cols_to_keep)
    if missing_cols:
        logger.warning(f"Missing columns for {resource_tag}: {missing_cols}")

    # Make sure the first columns are DEFAULT_COLS defined at the top of this file
    remaining_cols = [col for col in cols_to_keep if col not in DEFAULT_COLS]
    final_cols = [col for col in DEFAULT_COLS if col in cols_to_keep] + remaining_cols

    # Return the filtered and restructured dataframe
    return resource_df[final_cols]


def create_policy_df(df: pd.DataFrame, policy_info: Dict[str, str]) -> pd.DataFrame:
    """Create a dataframe with policy-specific columns for a specific policy type.

    Filters generator data for a specific policy type and removes columns
    specific to other policy types.

    Parameters
    ----------
    df : pd.DataFrame
        Generator data with policy type columns
    policy_info : Dict[str, str]
        Policy type configuration

    Returns
    -------
    pd.DataFrame
        Dataframe containing the policy tag columns
    """
    logger.info(f"Creating policy data file for {policy_info['newtag']}")

    if df.empty:
        logger.warning(f"DataFrame is empty, skipping policy {policy_info['newtag']}")
        return pd.DataFrame()

    # Check if the Resource column exists
    if "Resource" not in df.columns:
        raise KeyError("The 'Resource' column is missing from the dataframe.")

    # Check that ESR_, MinCapTag_, and MaxCapTag_ columns only have values equal to 0 or 1
    # and CapRes_ columns only have values between 0 and 1
    for col in df.columns:
        if (
            col.startswith("ESR_")
            or col.startswith("MinCapTag_")
            or col.startswith("MaxCapTag_")
        ):
            if not df[col].isin([0, 1]).all():
                raise ValueError(f"The column {col} can only have values 0 or 1.")
        elif col.startswith("CapRes_"):
            if not (df[col] >= 0).all() or not (df[col] <= 1).all():
                raise ValueError(
                    f"The column {col} can only have values between 0 and 1."
                )

    # Check if any columns start with the policy tag
    if not any(col.startswith(policy_info["oldtag"]) for col in df.columns):
        logger.debug(
            f"No columns start with {policy_info['oldtag']}, skipping policy {policy_info['newtag']}"
        )
        return pd.DataFrame()

    # Slice dataframe to include only Resource column and columns starting with oldtag
    policy_cols = [
        col
        for col in df.columns
        if col == "Resource" or col.startswith(policy_info["oldtag"])
    ]
    policy_df = df[policy_cols].copy()

    # Keep only rows with at least one non-zero value in policy columns
    policy_data_cols = policy_df.select_dtypes(
        include=["number"]
    ).columns  # All numeric columns
    policy_df = policy_df[policy_df[policy_data_cols].gt(0).any(axis=1)]

    # Rename columns replacing oldtag with newtag
    rename_dict = {
        col: col.replace(policy_info["oldtag"], policy_info["newtag"])
        for col in policy_df.columns
        if col.startswith(policy_info["oldtag"])
    }
    policy_df.rename(columns=rename_dict, inplace=True)

    # Remove policy columns from original dataframe
    df.drop(
        columns=[col for col in policy_data_cols if col in df.columns], inplace=True
    )

    return policy_df


def create_multistage_df(df: pd.DataFrame, multistage_cols: List[str]) -> pd.DataFrame:
    """Create multistage data file from generator data.

    Parameters
    ----------
    df : pd.DataFrame
        Generator data
    multistage_cols : List[str]
        List of column names for multistage data

    Returns
    -------
    pd.DataFrame
        DataFrame containing Resource column and multistage data
    """
    logger.info("Creating multistage data file")

    # Select Resource column and multistage columns
    cols_to_keep = ["Resource"] + [col for col in multistage_cols if col in df.columns]
    multistage_df = df[cols_to_keep].copy()

    # Remove multistage columns from original dataframe
    df.drop(columns=[col for col in multistage_cols if col in df.columns], inplace=True)

    return multistage_df


def split_generators_data(
    folder_structure: FolderStructure, gen_data: pd.DataFrame
) -> List[GenXInputData]:
    """Split generator data DataFrame into resource-specific files.

    Parameters
    ----------
    gen_data : pd.DataFrame
        DataFrame containing generator data

    Returns
    -------
    List[GenXResourceData]
        List of GenXResourceData objects containing the resource-specific data
    """

    # Paths to the folders to write the data to
    resource_folder = folder_structure.resource
    policy_assignments_folder = folder_structure.policy_assignments

    # Create a copy of the generator data
    generators_df = gen_data.copy()

    # Drop the R_ID column if it exists, GenX will assign IDs internally
    if "R_ID" in generators_df.columns:
        generators_df = generators_df.drop(columns=["R_ID"])

    # Check if New_Build and Can_Retire columns exist
    if (
        "New_Build" in generators_df.columns
        and "Can_Retire" not in generators_df.columns
    ):
        generators_df = update_newbuild_canretire(generators_df)

    # Process each POLICY_TAGS (defined at the top of this file) and return a dataframe
    # with the policy files to be written out
    resource_data = []
    for policy_tag, policy_info in POLICY_TAGS:
        policy_df = create_policy_df(generators_df, policy_info)
        if not policy_df.empty:
            out_file = POLICY_TAGS_FILENAMES[policy_tag]
            resource_data.append(
                GenXInputData(
                    tag=f"RES_{policy_tag}",
                    folder=policy_assignments_folder,
                    file_name=out_file,
                    dataframe=policy_df,
                )
            )

    # Process multistage data
    multistage_df = create_multistage_df(generators_df, MULTISTAGE_COLS)
    if not multistage_df.empty:
        out_file = "Resource_multistage_data.csv"
        resource_data.append(
            GenXInputData(
                tag="MULTISTAGE",
                folder=resource_folder,
                file_name=out_file,
                dataframe=multistage_df,
            )
        )

    # Process each RESOURCE_TAGS defined at the top of this file
    for resource_tag in RESOURCE_TAGS:
        out_file = RESOURCE_FILENAMES[resource_tag]
        resource_df = create_resource_df(generators_df, resource_tag)
        if not resource_df.empty:
            resource_data.append(
                GenXInputData(
                    tag=resource_tag,
                    folder=resource_folder,
                    file_name=out_file,
                    dataframe=resource_df,
                )
            )
        else:
            logger.info(f"No data found for resource tag {resource_tag}")

    # Return a list of GenXResourceData objects
    return resource_data


def process_genx_data(
    case_folder: Path, genx_data_dict: Dict[str, pd.DataFrame]
) -> List[GenXInputData]:
    """Process genx data dictionary and return a list of GenXResourceData objects.

    Parameters
    ----------
    case_folder : Path
        Path to the case folder
    genx_data_dict : Dict[str, pd.DataFrame]
        Dictionary of resource-specific data

    Returns
    -------
    List[GenXResourceData]
        List of GenXResourceData objects to be written out
    """

    resource_folder = case_folder / "resources"
    policy_assignments_folder = resource_folder / "policy_assignments"
    system_folder = case_folder / "system"
    policy_folder = case_folder / "policies"

    investment_incentive_df = genx_data_dict.get("investment_incentive", pd.DataFrame())
    production_incentive_df = genx_data_dict.get("production_incentive", pd.DataFrame())
    resource_investment_incentive_df = genx_data_dict.get(
        "resource_investment_incentive", pd.DataFrame()
    )
    resource_production_incentive_df = genx_data_dict.get(
        "resource_production_incentive", pd.DataFrame()
    )

    # Create folder structure
    folders = FolderStructure(
        resource=resource_folder,
        policy_assignments=policy_assignments_folder,
        system=system_folder,
        policy=policy_folder,
    )

    # List of GenXInputData objects to be written out
    genx_data = []

    # Process generator data separately
    genx_data.extend(split_generators_data(folders, genx_data_dict["gen_data"]))

    if not resource_investment_incentive_df.empty:
        genx_data.append(
            GenXInputData(
                tag="RES_INV_INCENTIVE",
                folder=policy_assignments_folder,
                file_name=INCENTIVE_FILENAMES["resource_investment"],
                dataframe=resource_investment_incentive_df,
            )
        )

    if not resource_production_incentive_df.empty:
        genx_data.append(
            GenXInputData(
                tag="RES_PROD_INCENTIVE",
                folder=policy_assignments_folder,
                file_name=INCENTIVE_FILENAMES["resource_production"],
                dataframe=resource_production_incentive_df,
            )
        )

    # Add all other data to the list
    genx_data.extend(
        [
            # System folder data
            GenXInputData(
                tag="GENERATORS_VARIABILITY",
                folder=system_folder,
                file_name="Generators_variability.csv",
                dataframe=genx_data_dict.get("gen_variability", pd.DataFrame()),
            ),
            GenXInputData(
                tag="DEMAND",
                folder=system_folder,
                file_name="Demand_data.csv",
                dataframe=genx_data_dict.get("demand_data", pd.DataFrame()),
            ),
            GenXInputData(
                tag="PERIOD_MAP",
                folder=system_folder,
                file_name="Period_map.csv",
                dataframe=genx_data_dict.get("period_map", pd.DataFrame()),
            ),
            GenXInputData(
                tag="REP_PERIOD",
                folder=system_folder,
                file_name="Representative_Period.csv",
                dataframe=genx_data_dict.get("rep_period", pd.DataFrame()),
            ),
            GenXInputData(
                tag="NETWORK",
                folder=system_folder,
                file_name="Network.csv",
                dataframe=genx_data_dict.get("network", pd.DataFrame()),
            ),
            GenXInputData(
                tag="OP_RESERVES",
                folder=system_folder,
                file_name="Operational_reserves.csv",
                dataframe=genx_data_dict.get("op_reserves", pd.DataFrame()),
            ),
            GenXInputData(
                tag="FUELS",
                folder=system_folder,
                file_name="Fuels_data.csv",
                dataframe=genx_data_dict.get("fuels", pd.DataFrame()),
            ),
            # Policy folder data
            GenXInputData(
                tag="ESR",
                folder=policy_folder,
                file_name="Energy_share_requirement.csv",
                dataframe=genx_data_dict.get("esr", pd.DataFrame()),
            ),
            GenXInputData(
                tag="CAP_RESERVES",
                folder=policy_folder,
                file_name="Capacity_reserve_margin.csv",
                dataframe=genx_data_dict.get("cap_reserves", pd.DataFrame()),
            ),
            GenXInputData(
                tag="CO2_CAP",
                folder=policy_folder,
                file_name="CO2_cap.csv",
                dataframe=genx_data_dict.get("co2_cap", pd.DataFrame()),
            ),
            GenXInputData(
                tag="MIN_CAP",
                folder=policy_folder,
                file_name="Minimum_capacity_requirement.csv",
                dataframe=genx_data_dict.get("min_cap", pd.DataFrame()),
            ),
            GenXInputData(
                tag="MAX_CAP",
                folder=policy_folder,
                file_name="Maximum_capacity_requirement.csv",
                dataframe=genx_data_dict.get("max_cap", pd.DataFrame()),
            ),
        ]
    )

    # Add incentive policy tables only if present (keep legacy tests stable)
    if not investment_incentive_df.empty:
        genx_data.append(
            GenXInputData(
                tag="INV_INCENTIVE",
                folder=policy_folder,
                file_name=INCENTIVE_FILENAMES["investment_policy"],
                dataframe=investment_incentive_df,
            )
        )

    if not production_incentive_df.empty:
        genx_data.append(
            GenXInputData(
                tag="PROD_INCENTIVE",
                folder=policy_folder,
                file_name=INCENTIVE_FILENAMES["production_policy"],
                dataframe=production_incentive_df,
            )
        )

    return genx_data


def process_genx_data_old_format(
    case_folder: Path, genx_data_dict: Dict[str, pd.DataFrame]
) -> List[GenXInputData]:
    """Process genx data dictionary and return a list of GenXResourceData objects. Use
    for GenX versions prior to 0.4.0.

    Parameters
    ----------
    case_folder : Path
        Path to the case folder
    genx_data_dict : Dict[str, pd.DataFrame]
        Dictionary of resource-specific data

    Returns
    -------
    List[GenXResourceData]
        List of GenXResourceData objects to be written out
    """

    # Add 'None' column to fuels data if it doesn't exist -- causes error in old GenX otherwise
    if (
        "fuels" in genx_data_dict
        and "None" not in genx_data_dict.get("fuels", pd.DataFrame()).columns
    ):
        genx_data_dict["fuels"]["None"] = 0

    if "demand_data" in genx_data_dict:
        genx_data_dict["demand_data"].columns = genx_data_dict[
            "demand_data"
        ].columns.str.replace("Demand_", "Load_")

    # List of GenXInputData objects to be written out
    genx_data = []

    # Add all other data to the list
    genx_data.extend(
        [
            GenXInputData(
                tag="GENERATORS_DATA",
                folder=case_folder,
                file_name="Generators_data.csv",
                dataframe=genx_data_dict.get("gen_data", pd.DataFrame()),
            ),
            GenXInputData(
                tag="GENERATORS_VARIABILITY",
                folder=case_folder,
                file_name="Generators_variability.csv",
                dataframe=genx_data_dict.get("gen_variability", pd.DataFrame()),
            ),
            GenXInputData(
                tag="DEMAND",
                folder=case_folder,
                file_name="Load_data.csv",
                dataframe=genx_data_dict.get("demand_data", pd.DataFrame()),
            ),
            GenXInputData(
                tag="PERIOD_MAP",
                folder=case_folder,
                file_name="Period_map.csv",
                dataframe=genx_data_dict.get("period_map", pd.DataFrame()),
            ),
            GenXInputData(
                tag="REP_PERIOD",
                folder=case_folder,
                file_name="Representative_Period.csv",
                dataframe=genx_data_dict.get("rep_period", pd.DataFrame()),
            ),
            GenXInputData(
                tag="NETWORK",
                folder=case_folder,
                file_name="Network.csv",
                dataframe=genx_data_dict.get("network", pd.DataFrame()),
            ),
            GenXInputData(
                tag="OP_RESERVES",
                folder=case_folder,
                file_name="Operational_reserves.csv",
                dataframe=genx_data_dict.get("op_reserves", pd.DataFrame()),
            ),
            GenXInputData(
                tag="FUELS",
                folder=case_folder,
                file_name="Fuels_data.csv",
                dataframe=genx_data_dict.get("fuels", pd.DataFrame()),
            ),
            GenXInputData(
                tag="ESR",
                folder=case_folder,
                file_name="Energy_share_requirement.csv",
                dataframe=genx_data_dict.get("esr", pd.DataFrame()),
            ),
            GenXInputData(
                tag="CAP_RESERVES",
                folder=case_folder,
                file_name="Capacity_reserve_margin.csv",
                dataframe=genx_data_dict.get("cap_reserves", pd.DataFrame()),
            ),
            GenXInputData(
                tag="CO2_CAP",
                folder=case_folder,
                file_name="CO2_cap.csv",
                dataframe=genx_data_dict.get("co2_cap", pd.DataFrame()),
            ),
            GenXInputData(
                tag="MIN_CAP",
                folder=case_folder,
                file_name="Minimum_capacity_requirement.csv",
                dataframe=genx_data_dict.get("min_cap", pd.DataFrame()),
            ),
            GenXInputData(
                tag="MAX_CAP",
                folder=case_folder,
                file_name="Maximum_capacity_requirement.csv",
                dataframe=genx_data_dict.get("max_cap", pd.DataFrame()),
            ),
        ]
    )

    return genx_data
