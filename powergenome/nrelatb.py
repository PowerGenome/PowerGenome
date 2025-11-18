"""
Functions to fetch and modify new-build resource data
"""

import collections
import copy
import logging
import operator
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd
from joblib import Parallel, delayed

from powergenome.cluster.renewables import (
    assign_site_cluster,
    calc_cluster_values,
    modify_renewable_group,
)
from powergenome.database import get_data
from powergenome.financials import investment_cost_calculator
from powergenome.params import DATA_PATHS, SETTINGS, build_resource_clusters
from powergenome.price_adjustment import inflation_price_adjustment
from powergenome.resource_clusters import (
    ClusterBuilder,
    ResourceGroup,
    Table,
    map_technologies,
)
from powergenome.settings import apply_all_tag_to_regions
from powergenome.util import (
    add_row_to_csv,
    calculate_file_hash,
    hash_string_sha256,
    reverse_dict_of_lists,
    snake_case_col,
)

idx = pd.IndexSlice
logger = logging.getLogger(__name__)


def fetch_resource_costs(
    settings: dict,
    resource_data_year: int = None,
) -> pd.DataFrame:
    """Get resource cost data from the DataManager, filter where applicable.


    Parameters
    ----------
    settings : dict
        User-defined parameters from a settings file. Required keys:
        `new_resources` (list): List of new resource technologies to include,
        `target_usd_year` (int): Target dollar year for price adjustments,
        `data_location` (Path): Location of inflation adjustment data,
        `dollar_year_table` (str): Name of the table containing dollar year data.
        Optional keys:
        `modified_new_resources` (dict): Dictionary of modified resource technologies,
        `resource_financial_case` (str): Financial case for cost data, defaults to "Market".
    resource_data_year : int, optional
        Year of data vintage. Not the same as planning or technology year. If not
        specified, defaults to None, which will not filter by data year.

    Returns
    -------
    pd.DataFrame
        Power plant cost data with columns:
        ['technology', 'cap_recovery_years', 'cost_case', 'financial_case',
       'basis_year', 'tech_detail', 'fixed_o_m_mw', 'variable_o_m_mwh', 'capex', 'cf',
       'fuel', 'lcoe', 'wacc_real']
    """
    logger.debug("Loading resource cost data")

    col_names = [
        "technology",
        "tech_detail",
        "cost_case",
        "parameter",
        "basis_year",
        "parameter_value",
        "dollar_year",
    ]

    fin_case = settings.get("resource_financial_case", "Market")

    # Fetch cost data from sqlite and create dataframe. Only get values for techs/cases
    # listed in the settings file.
    all_rows = []
    wacc_rows = []
    tech_list = []
    techs = settings["new_resources"]
    mod_techs = []
    if settings.get("modified_new_resources"):
        for _, m in settings.get("modified_new_resources").items():
            mod_techs.append([m["technology"], m["tech_detail"], m["cost_case"], None])

    cost_params = (
        "capex_mw",
        "fixed_o_m_mw",
        "variable_o_m_mwh",
        "capex_mwh",
        "fixed_o_m_mwh",
    )
    # add_pv_wacc = True
    cols = ["technology", "tech_detail", "financial_case", "cost_case", "data_year"]
    # valid_inputs = db_col_values(data_location, data_name, cols)
    for tech in techs + mod_techs:
        tech, tech_detail, cost_case, _ = tech
        filters = [
            [
                ("technology", "=", tech),
                ("tech_detail", "=", tech_detail),
                ("financial_case", "=", fin_case),
                ("cost_case", "=", cost_case),
                ("parameter", "IN", cost_params),
            ]
        ]
        if resource_data_year:
            # If a resource data year is specified, filter by that as well.
            filters[0].append(("data_year", "=", resource_data_year))

        all_rows.append(get_data("resource_cost", filters=filters))
        # all_rows.extend(pg_engine.execute(s, cost_params).fetchall())

        if (tech, cost_case) not in tech_list:
            # ATB2020 summary file provides a single WACC for each technology and a single
            # tech detail of "*", so need to fetch this separately from other cost params.
            # Only need to fetch once per technology.
            wacc_filters = [
                [
                    ("technology", "=", tech),
                    ("financial_case", "=", fin_case),
                    ("cost_case", "=", cost_case),
                    ("parameter", "=", "wacc_real"),
                ]
            ]
            if resource_data_year:
                # If a resource data year is specified, filter by that as well.
                wacc_filters[0].append(("data_year", "=", resource_data_year))
            wacc_rows.append(get_data("resource_cost", filters=wacc_filters))
            # wacc_rows.extend(pg_engine.execute(wacc_s).fetchall())

        tech_list.append((tech, cost_case))

    df = pd.concat(all_rows, ignore_index=True)[col_names]
    wacc_df = pd.concat(wacc_rows, ignore_index=True)[
        ["technology", "cost_case", "basis_year", "parameter_value"]
    ].rename(columns={"parameter_value": "wacc_real"})

    # Transform from tidy to wide dataframe, which makes it easier to fill generator
    # rows with the correct values.
    resource_costs = (
        df.drop_duplicates()
        .set_index(
            [
                "technology",
                "tech_detail",
                "cost_case",
                "dollar_year",
                "basis_year",
                "parameter",
            ]
        )
        .unstack(level=-1)
    )
    resource_costs.columns = resource_costs.columns.droplevel(0)
    resource_costs = (
        resource_costs.reset_index()
        .merge(wacc_df, on=["technology", "cost_case", "basis_year"], how="left")
        .drop_duplicates()
    )
    resource_costs = resource_costs.fillna(0)

    usd_columns = [
        "fixed_o_m_mw",
        "fixed_o_m_mwh",
        "variable_o_m_mwh",
        "capex_mw",
        "capex_mwh",
    ]
    for col in usd_columns:
        if col not in resource_costs.columns:
            resource_costs[col] = 0

    target_usd_year = settings["target_usd_year"]
    if not resource_costs.empty:
        resource_costs[usd_columns] = resource_costs.apply(
            lambda row: inflation_price_adjustment(
                row[usd_columns],
                base_year=row["dollar_year"],
                target_year=target_usd_year,
                data_location=settings["data_location"],
                table_name=settings["dollar_year_table"],
            ),
            axis=1,
        )

    return resource_costs


def fetch_heat_rates(data_year: int = None) -> pd.DataFrame:
    """Get heat rate projections for power plants from the DataManager

    Parameters
    ----------
    data_year : int
        Year of data vintage. Not the same as planning or technology year. Optional,
        defaults to None.

    Returns
    -------
    pd.DataFrame
        Power plant heat rate data by year with columns:
        ['technology', 'tech_detail', 'cost_case', 'basis_year', 'heat_rate']
    """
    if data_year:
        filters = [[("data_year", "=", data_year)]]
        heat_rates = get_data("resource_heat_rate", filters=filters)
    else:
        heat_rates = get_data("resource_heat_rate")
    # heat_rates = heat_rates.loc[heat_rates["data_year"] == data_year, :]

    if heat_rates.empty:
        s = (
            f"Your settings file has parameter `data_year` of {data_year}"
            f", which isn't in the resource heat rate table."
        )
        raise ValueError(s)

    return heat_rates


def single_generator_row(
    resource_costs_hr: pd.DataFrame,
    new_gen_type: str,
    model_year_range: Union[Tuple[int], List[int]],
) -> pd.DataFrame:
    """Create a data row with costs and performace for a single technology

    Parameters
    ----------
    resource_costs_hr : pd.DataFrame
        Data from the tables of both resources costs and heat rates
    new_gen_type : str
        type of generating resource
    model_year_range : Union[Tuple[int], List[int]]
        All of the years that should be averaged over

    Returns
    -------
    pd.DataFrame
        A single row dataframe with average cost and performance values over the study
        period.
    """

    technology, tech_detail, cost_case, size_mw = new_gen_type
    numeric_cols = [
        "basis_year",
        "fixed_o_m_mw",
        "fixed_o_m_mwh",
        "variable_o_m_mwh",
        "capex_mw",
        "capex_mwh",
        "wacc_real",
        "heat_rate",
    ]
    s = resource_costs_hr.loc[
        (resource_costs_hr["technology"] == technology)
        & (resource_costs_hr["tech_detail"] == tech_detail)
        & (resource_costs_hr["cost_case"] == cost_case)
        & (resource_costs_hr["basis_year"].isin(model_year_range)),
        numeric_cols,
    ].mean()
    cols = ["technology", "cost_case", "tech_detail"] + numeric_cols
    row = pd.DataFrame([technology, cost_case, tech_detail] + s.to_list(), index=cols).T

    row["Cap_Size"] = size_mw

    return row


def regional_capex_multiplier(
    df: pd.DataFrame,
    region: str,
    region_map: Dict[str, str],
    tech_map: Dict[str, str],
    regional_multipliers: pd.DataFrame,
) -> pd.DataFrame:
    """
    Adjusts investment costs in the input DataFrame by applying regional capital expenditure
    (capex) multipliers based on technology and region.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing technology and investment cost columns.
    region : str
        Name of the region for which to apply cost multipliers.
    region_map : Dict[str, str]
        Mapping from region names to cost region identifiers.
    tech_map : Dict[str, str]
        Mapping from new technology names to reference technology names used for multipliers.
    regional_multipliers : pd.DataFrame
        DataFrame of regional cost multipliers indexed by cost region and technology.

    Returns
    -------
    pd.DataFrame
        DataFrame with updated investment cost columns and a new column for the applied regional cost multiplier.

    Notes
    -----
    - If a technology name matches more than one entry in the DataFrame, only the first match receives a valid multiplier.
    - Technologies without a specific multiplier receive the average multiplier for the region.
    - Issues with ambiguous technology mapping are logged as warnings.
    """
    df = df.copy()
    cost_region = region_map[region]
    tech_multiplier = regional_multipliers.loc[
        regional_multipliers["region"] == cost_region, "value"
    ]
    avg_multiplier = tech_multiplier.mean()

    tech_multiplier = tech_multiplier.fillna(avg_multiplier)

    tech_multiplier_map = {}
    for new_tech, reference_tech in tech_map.items():
        if df["technology"].str.contains(new_tech, case=False, regex=False).sum() > 0:
            full_new_tech = df.loc[
                df["technology"]
                .str.contains(new_tech, case=False, regex=False)
                .idxmax(),
                "technology",
            ]
            tech_multiplier_map[full_new_tech] = tech_multiplier.at[reference_tech]
        if df["technology"].str.contains(new_tech).sum() > 1:
            s = f"""
    ***************************
    There is an issue with assigning regional cost multipliers. In your settings file
    under the parameter 'cost_multiplier_technology_map`, the technology '{reference_tech}'
    has a matching reference technology '{new_tech}'. This name matches more than one new
    resource listed in the settings parameter 'new_resources'. Only the first matching tech in
    'new_resources' will get a valid regional cost multiplier; the rest will have values of
    0, which will lead to annual investment costs of $0.
        """
            logger.warning(s)
    df["Inv_Cost_per_MWyr"] *= df["technology"].map(tech_multiplier_map)
    df["Inv_Cost_per_MWhyr"] *= df["technology"].map(tech_multiplier_map)
    df["regional_cost_multiplier"] = df["technology"].map(tech_multiplier_map)

    return df


def add_modified_generators(
    settings: dict,
    resource_costs_hr: pd.DataFrame,
    model_year_range: Union[Tuple[int], List[int]],
) -> pd.DataFrame:
    """Create a modified version of a resource.

    For each parameter (capex, heat_rate, etc) that users want modified they should
    specify a list of [<operator>, <value>]. The operator can be add, mul, truediv, or
    sub (substract). This is used to modify individual parameters of the resource.

    Parameters
    ----------
    settings : dict
        User-defined parameters from a settings file
    resource_costs_hr : pd.DataFrame
        Cost and heat rate data for resources
    model_year_range : Union[Tuple[int], List[int]]
        A list or range of years to average resource values from.

    Returns
    -------
    pd.DataFrame
        Row or rows of modified resources. Each row includes the columns:
        ['technology', 'cost_case', 'tech_detail', 'basis_year', 'fixed_o_m_mw',
       'fixed_o_m_mwh', 'variable_o_m_mwh', 'capex', 'capex_mwh', 'cf', 'fuel',
       'lcoe', 'o_m', 'wacc_real', 'heat_rate', 'Cap_Size'].
    """

    # copy settings so popped keys aren't removed permenantly
    _settings = copy.deepcopy(settings)

    allowed_operators = ["add", "mul", "truediv", "sub"]

    mod_tech_list = []
    for name, mod_tech in _settings["modified_new_resources"].items():
        technology = mod_tech.pop("technology")
        tech_detail = mod_tech.pop("tech_detail")
        cost_case = mod_tech.pop("cost_case")
        size_mw = mod_tech.pop("size_mw")

        new_gen_type = (technology, tech_detail, cost_case, size_mw)

        gen = single_generator_row(resource_costs_hr, new_gen_type, model_year_range)
        gen["technology"] = mod_tech.pop("new_technology")
        gen["tech_detail"] = mod_tech.pop("new_tech_detail", "")
        gen["cost_case"] = mod_tech.pop("new_cost_case")

        for parameter, op_list in mod_tech.items():
            if isinstance(op_list, float) | isinstance(op_list, int):
                gen[parameter] = op_list
            else:
                assert len(op_list) == 2, (
                    "Two values, an operator and a numeric value, are needed in the parameter\n"
                    f"'{parameter}' for technology '{name}' in 'modified_new_resources'."
                )
                op, op_value = op_list

                assert parameter in gen.columns, (
                    f"'{parameter}' is not a valid parameter for new resources. Check '{name}'\n"
                    "in 'modified_new_resources' of the settings file."
                )
                assert op in allowed_operators, (
                    f"The key {parameter} for technology {name} needs a valid operator from the list\n"
                    f"{allowed_operators}\n"
                    "in the format [<operator>, <value>] to modify the properties of an existing generator.\n"
                )

                f = operator.attrgetter(op)
                gen[parameter] = f(operator)(gen[parameter], op_value)

        mod_tech_list.append(gen)

    mod_gens = pd.concat(mod_tech_list, ignore_index=True)

    return mod_gens


def build_new_resources(resource_costs, resource_hr, settings, cluster_builder=None):
    """Add rows for new generators in each region

    Parameters
    ----------
    resource_costs : DataFrame
        All cost parameters for new resources. Should include:
        ['technology', 'cost_case', 'financial_case', 'basis_year', 'tech_detail',
        'capex', 'capex_mwh', 'fixed_o_m_mw', 'fixed_o_m_mwh', 'variable_o_m_mwh',
        'wacc_real']
    resource_hr : DataFrame
        The technology, tech_detail, and heat_rate of new resources.
    settings : dict
        User-defined parameters from a settings file
    cluster_builder : ClusterBuilder
        ClusterBuilder object. Reuse to save time. None by default.

    Returns
    -------
    DataFrame
        New generating resources in every region. Contains the columns:
        ['technology', 'basis_year', 'Fixed_OM_Cost_per_MWyr',
       'Fixed_OM_Cost_per_MWhyr', 'Var_OM_Cost_per_MWh', 'capex', 'capex_mwh',
       'Inv_Cost_per_MWyr', 'Inv_Cost_per_MWhyr', 'Heat_Rate_MMBTU_per_MWh',
       'Cap_Size', 'region']
    """
    logger.debug("Creating new resources for each region.")
    new_gen_types = settings["new_resources"]
    model_year = settings["model_year"]
    try:
        first_planning_year = settings["model_first_planning_year"]
        model_year_range = range(first_planning_year, model_year + 1)
    except KeyError:
        model_year_range = list(range(model_year + 1))

    regions = settings["model_regions"]

    resource_costs_hr = resource_costs.merge(
        resource_hr,
        on=["technology", "tech_detail", "cost_case", "basis_year"],
        how="left",
    )

    if new_gen_types:
        new_gen_df = pd.concat(
            [
                single_generator_row(resource_costs_hr, new_gen, model_year_range)
                for new_gen in new_gen_types
            ],
            ignore_index=True,
        )
    else:
        new_gen_df = pd.DataFrame(
            columns=["region", "technology", "tech_detail", "cost_case"]
        )
    # Add user-defined technologies
    # This should probably be separate from resource techs, and the regional cost multipliers
    # should be its own function.
    if settings.get("additional_technologies_fn"):
        if isinstance(settings.get("additional_new_gen"), list):
            # user_costs, user_hr = load_user_defined_techs(settings)
            user_tech = load_user_defined_techs(settings)
            # new_gen_df = pd.concat([new_gen_df, user_costs], ignore_index=True, sort=False)
            new_gen_df = pd.concat(
                [new_gen_df, user_tech], ignore_index=True, sort=False
            )
            # resource_hr = pd.concat([resource_hr, user_hr], ignore_index=True, sort=False)
        else:
            logger.warning(
                "A filename for additional technologies was included but no technologies"
                " were specified in the settings file."
            )

    if settings.get("modified_new_resources"):
        modified_gens = add_modified_generators(
            settings, resource_costs_hr, model_year_range
        )
        new_gen_df = pd.concat(
            [new_gen_df, modified_gens], ignore_index=True, sort=False
        )

    new_gen_df = new_gen_df.rename(
        columns={
            "heat_rate": "Heat_Rate_MMBTU_per_MWh",
            "fixed_o_m_mw": "Fixed_OM_Cost_per_MWyr",
            "fixed_o_m_mwh": "Fixed_OM_Cost_per_MWhyr",
            "variable_o_m_mwh": "Var_OM_Cost_per_MWh",
        }
    )

    # This is now generalized for changes to resource values for any technology type.
    for tech, _tech_modifiers in (settings.get("resource_modifiers") or {}).items():
        tech_modifiers = copy.deepcopy(_tech_modifiers)
        assert isinstance(tech_modifiers, dict), (
            "The settings parameter 'resource_modifiers' must be a nested list.\n"
            "Each top-level key is a short name of the technology, with a nested"
            " dictionary of items below it."
        )
        assert (
            "technology" in tech_modifiers
        ), "Each nested dictionary in resource_modifiers must have a 'technology' key."
        assert (
            "tech_detail" in tech_modifiers
        ), "Each nested dictionary in resource_modifiers must have a 'tech_detail' key."

        technology = tech_modifiers.pop("technology")
        tech_detail = tech_modifiers.pop("tech_detail")

        allowed_operators = ["add", "mul", "truediv", "sub"]

        for key, op_list in tech_modifiers.items():
            if isinstance(op_list, float) | isinstance(op_list, int):
                new_gen_df.loc[
                    (new_gen_df.technology == technology)
                    & (new_gen_df.tech_detail == tech_detail),
                    key,
                ] = op_list
            else:
                assert len(op_list) == 2, (
                    "Two values, an operator and a numeric value, are needed in the parameter\n"
                    f"'{key}' for technology '{tech}' in 'resource_modifiers'."
                )
                op, op_value = op_list

                assert op in allowed_operators, (
                    f"The key {key} for technology {tech} needs a valid operator from the list\n"
                    f"{allowed_operators}\n"
                    "in the format [<operator>, <value>] to modify the properties of an existing generator.\n"
                )

                f = operator.attrgetter(op)
                new_gen_df.loc[
                    (new_gen_df.technology == technology)
                    & (new_gen_df.tech_detail == tech_detail),
                    key,
                ] = f(operator)(
                    new_gen_df.loc[
                        (new_gen_df.technology == technology)
                        & (new_gen_df.tech_detail == tech_detail),
                        key,
                    ],
                    op_value,
                )

    new_gen_df["technology"] = (
        new_gen_df[["technology", "tech_detail", "cost_case"]]
        .astype(str)
        .agg("_".join, axis=1)
    )

    new_gen_df["cap_recovery_years"] = settings["resource_cap_recovery_years"]

    if new_gen_df.empty:
        results = new_gen_df.copy()
    else:
        for tech, years in (
            settings.get("alt_resource_cap_recovery_years") or {}
        ).items():
            new_gen_df.loc[
                new_gen_df["technology"].str.lower().str.contains(tech.lower()),
                "cap_recovery_years",
            ] = years

        new_gen_df["Inv_Cost_per_MWyr"] = investment_cost_calculator(
            capex=new_gen_df["capex_mw"],
            wacc=new_gen_df["wacc_real"],
            cap_rec_years=new_gen_df["cap_recovery_years"],
            compound_method=settings.get("interest_compound_method", "discrete"),
        )

        new_gen_df["Inv_Cost_per_MWhyr"] = investment_cost_calculator(
            capex=new_gen_df["capex_mwh"],
            wacc=new_gen_df["wacc_real"],
            cap_rec_years=new_gen_df["cap_recovery_years"],
            compound_method=settings.get("interest_compound_method", "discrete"),
        )

        # Set no capacity limit on new resources that aren't renewables.
        new_gen_df["Max_Cap_MW"] = -1
        new_gen_df["Max_Cap_MWh"] = -1

        regional_cost_multipliers = get_data("regional_cost_factor").set_index(
            "technology"
        )

        if settings.get("cost_multiplier_region_map"):
            rev_mult_region_map = reverse_dict_of_lists(
                settings["cost_multiplier_region_map"]
            )
        else:
            rev_mult_region_map = {
                region: region for region in settings["model_regions"]
            }
        if settings.get("cost_multiplier_technology_map"):
            rev_mult_tech_map = reverse_dict_of_lists(
                settings["cost_multiplier_technology_map"]
            )
        else:
            tech_list = (
                new_gen_df[["technology", "tech_detail", "cost_case"]]
                .astype(str)
                .agg("_".join, axis=1)
            )

        df_list = []
        settings = apply_all_tag_to_regions(settings)

        df_list = Parallel(n_jobs=settings.get("clustering_n_jobs", 1))(
            delayed(parallel_region_renewables)(
                copy.deepcopy(settings),
                new_gen_df,
                regional_cost_multipliers,
                rev_mult_region_map,
                rev_mult_tech_map,
                region,
                cluster_builder,
            )
            for region in regions
        )

        results = pd.concat(df_list, ignore_index=True, sort=False)

        int_cols = [
            "Fixed_OM_Cost_per_MWyr",
            "Fixed_OM_Cost_per_MWhyr",
            "Inv_Cost_per_MWyr",
            "Inv_Cost_per_MWhyr",
            # "cluster",
        ]
        int_cols = [c for c in int_cols if c in results.columns]
        results = results.fillna(0)
        results[int_cols] = results[int_cols].astype(int)
        results["Var_OM_Cost_per_MWh"] = results["Var_OM_Cost_per_MWh"].astype(float)

    return results


def parallel_region_renewables(
    settings: dict,
    new_gen_df: pd.DataFrame,
    regional_cost_multipliers: pd.DataFrame,
    rev_mult_region_map: Dict[str, List[str]],
    rev_mult_tech_map: Dict[str, List[str]],
    region: str,
    cluster_builder: ClusterBuilder = None,
) -> pd.DataFrame:
    """Wrapper function to run regional capex and add renewable clusters in parallel

    Parameters
    ----------
    settings : dict
        Can have keys "renewables_clusters" and "region_aggregations"
    new_gen_df : pd.DataFrame
        Rows are new-build resources specified by the user
    regional_cost_multipliers : pd.DataFrame
        Cost multiplier for each technology type in different regions
    rev_mult_region_map : Dict[str, List[str]]
        Mapping of cost regions to model regions
    rev_mult_tech_map : Dict[str, List[str]]
        Mapping of technologies from cost map to technologies in new_gen_df
    region : str
        Name of the model region
    cluster_builder
        ClusterBuilder object. Reuse to save time. None by default.

    Returns
    -------
    pd.DataFrame
        New-build resources in a single region. Includes the regionally corrected cost
        and renewable resource clusters as specified by the user.
    """
    _df = new_gen_df.copy()
    _df["region"] = region
    _df = regional_capex_multiplier(
        _df,
        region,
        rev_mult_region_map,
        rev_mult_tech_map,
        regional_cost_multipliers,
    )
    _df = add_renewables_clusters(
        _df,
        region,
        copy.deepcopy(settings),
        cluster_builder,
        cache_results=True,
        use_cache=True,
    )

    return _df


def load_resource_group_data(
    rg: ResourceGroup, cache=True
) -> Tuple[pd.DataFrame, Union[pd.Series, None]]:
    """Load metadata for the specified resource group.

    Metadata is information on individual renewable sites such as the site ID, capacity,
    interconnection cost, etc. If the resource group has the attribute "site_map", then
    a mapping of site IDs to generation profile IDs is also returned.

    Parameters
    ----------
    rg : ResourceGroup
        A resource group object
    cache : bool, optional
        A flag indicating whether to cache the data, by default True

    Returns
    -------
    Tuple[pd.DataFrame, Union[pd.Series, None]]
        A tuple of the metadata dataframe and the site map as a Series with site ID as
        the index
    """
    data = rg.metadata.read(cache=cache)
    data.columns = snake_case_col(data.columns)
    if "metro_region" in data.columns and "region" not in data.columns:
        data["region"] = data.loc[:, "metro_region"]
    if "cpa_mw" in data.columns and "mw" not in data.columns:
        data["mw"] = data.loc[:, "cpa_mw"]
    data = data.loc[data["mw"] > 0, :]
    if rg.group.get("profiles"):
        profile_path = Path(rg.group["profiles"])
    else:
        profile_path = None
    if rg.group.get("site_map") and profile_path is not None:
        table = Table(profile_path.parent / rg.group["site_map"])
        cols = table.columns
        df = table.read().set_index(cols[0])
        site_map = df[df.columns[0]]
    else:
        site_map = None

    return data, site_map


def flatten_cluster_def(
    scenario: Union[dict, list, str, int, float], detail_suffix: str
) -> str:
    """Turn a nested dictionary of clustering instructions into a string.

    Parameters
    ----------
    scenario : Union[dict, list, str, int, float]
        Either a dictionary, list, str, or numeric object -- base level must be a string
        or numeric.
    detail_suffix : str
        A string used to separate individual objects

    Returns
    -------
    str
        Flattened string representation
    """
    "Return cluster definition as a string for unique filenames"
    if isinstance(scenario, dict):
        for k, v in scenario.items():
            detail_suffix += flatten_cluster_def(k, "")
            detail_suffix += flatten_cluster_def(v, "")
    elif isinstance(scenario, list):
        for l in scenario:
            detail_suffix += flatten_cluster_def(l, "")
    else:
        detail_suffix += f"{scenario}_"

    return detail_suffix


def add_renewables_clusters(
    df: pd.DataFrame,
    region: str,
    settings: dict,
    cluster_builder: ClusterBuilder = None,
    cache_results: bool = False,
    use_cache: bool = False,
) -> pd.DataFrame:
    """
    Add renewables clusters

    Parameters
    ----------
    df
        New generation technologies.
            - `technology`: Resource technology in the format
                <technology>_<tech_detail>_<cost_case>. Must be unique.
            - `region`: Model region.
    region
        Model region.
    settings
        Dictionary with the following keys:
            - `renewables_clusters`: Determines the clusters built for the region.
            - `region_aggregations`: Maps the model region to IPM regions.


    Returns
    -------
    pd.DataFrame
        Copy of the input dataframe joined to rows for renewables clusters
        on matching resource technology and model region.

    Raises
    ------
    ValueError
        Resource technologies are not unique.
    ValueError
        Renewables clusters do not match resource technologies.
    ValueError
        Renewables clusters match multiple resource technologies.
    """
    if not cluster_builder:
        cluster_builder = build_resource_clusters(
            settings.get("RESOURCE_GROUPS"), settings.get("RESOURCE_GROUP_PROFILES")
        )
    if not df["technology"].is_unique:
        raise ValueError(
            f"Resource technologies are not unique: {df['technology'].to_list()}"
        )
    resource_map = {
        x: map_technologies(x.split("_")[0], x.split("_")[1]) for x in df["technology"]
    }
    mask = df["technology"].isin(
        [tech for tech, match in resource_map.items() if match]
    ) & (df["region"] == region)
    cdfs = []
    if region in (settings.get("region_aggregations", {}) or {}):
        regions = settings.get("region_aggregations", {})[region]
        regions.append(region)  # Add model region, sometimes listed in RG file
    else:
        regions = [region]
    for scenario in copy.deepcopy(settings).get("renewables_clusters", []) or []:
        if scenario["region"] != region:
            continue
        # Match cluster technology to resource technologies
        technologies = [
            k
            for k, v in resource_map.items()
            if v and all([scenario.get(ki) == vi for ki, vi in v.items()])
        ]
        if not technologies:
            s = (
                f"You have a renewables_cluster for technology '{scenario.get('technology')} "
                f"in region '{scenario.get('region')}', but no comparable new-build technology "
                "was specified in your settings file."
            )

            logger.warning(s)
            continue
        if len(technologies) > 1:
            raise ValueError(
                f"Renewables clusters match multiple resource technologies: {scenario}"
            )
        technology = technologies[0]
        _scenario = scenario.copy()
        # ClusterBuilder.get_clusters() does not take region as an argument
        _scenario.pop("region")

        # Assume not preclustering renewables unless set to True in settings or the
        # old parameters are used.
        precluster = False
        precluster_keys = ["max_clusters", "max_lcoe"]
        if settings.get("precluster_renewables") is True:
            precluster = True
        if any([k in precluster_keys for k in _scenario.keys()]):
            precluster = True

            # Create name suffex with unique id info like turbine_type and pref_site
        new_tech_suffix = "_" + "_".join(
            [
                str(v)
                for k, v in _scenario.items()
                if k
                not in [
                    "region",
                    "technology",
                    "max_clusters",
                    "min_capacity",
                    "filter",
                    "bin",
                    "group",
                    "cluster",
                    "group_modifiers",
                ]
            ]
        )
        detail_suffix = flatten_cluster_def(
            {k: v for (k, v) in _scenario.items() if k != "group_modifiers"}, "_"
        )

        # Get the data file paths and calculate their hashes
        drop_keys = [
            "min_capacity",
            "filter",
            "bin",
            "group",
            "cluster",
            "group_modifiers",
        ]
        group_kwargs = dict(
            [(k, v) for k, v in _scenario.items() if k not in drop_keys]
        )
        resource_groups = cluster_builder.find_groups(
            existing=False,
            **group_kwargs,
        )
        if resource_groups:
            profiles_path = resource_groups[0].group.get("profiles")
            metadata_path = resource_groups[0].group.get("metadata")

            profiles_hash = calculate_file_hash(
                Path(profiles_path) if profiles_path else None
            )
            metadata_hash = calculate_file_hash(
                Path(metadata_path) if metadata_path else None
            )

            data_file_hash = f"{metadata_hash}_{profiles_hash}"
        else:
            data_file_hash = "no_data"

        unique_hash = hash_string_sha256(
            f"{region}_{technology}_{detail_suffix}_UTC{settings.get('utc_offset', 0)}_weather_year{settings.get('weather_year','all')}_file_{data_file_hash}"
        )
        cache_cluster_fn = unique_hash + "_cluster_data.parquet"
        cache_site_assn_fn = unique_hash + "_site_assn.parquet"

        sub_folder = settings.get("RESOURCE_GROUPS") or SETTINGS.get("RESOURCE_GROUPS")
        sub_folder = str(sub_folder).replace("/", "_").replace("\\", "_")
        cache_folder = Path(
            settings["input_folder"] / "cluster_assignments" / sub_folder
        )
        if cache_results:
            add_row_to_csv(
                cache_folder / "hash_map.csv",
                headers=[
                    "name",
                    "hash",
                    "metadata_path",
                    "profiles_path",
                    "metadata_sha256",
                    "profiles_sha256",
                ],
                new_row=[
                    f"{region}_{technology}_{detail_suffix}_UTC{settings.get('utc_offset', 0)}_file_{data_file_hash}",
                    unique_hash,
                    str(metadata_path),
                    str(profiles_path),
                    metadata_hash,
                    profiles_hash,
                ],
            )
        cache_cluster_fpath = cache_folder / cache_cluster_fn
        cache_site_assn_fpath = cache_folder / cache_site_assn_fn
        if precluster is False:
            if (
                cache_cluster_fpath.exists()
                and cache_site_assn_fpath.exists()
                and use_cache
            ):
                clusters = pd.read_parquet(cache_cluster_fpath)
                data = pd.read_parquet(cache_site_assn_fpath)
            else:
                # Resource groups already found above
                # ...existing code...
                if not resource_groups:
                    raise ValueError(
                        f"Parameters do not match any resource groups: {group_kwargs}"
                    )
                if len(resource_groups) > 1:
                    meta = [rg.group for rg in resource_groups]
                    raise ValueError(
                        f"Parameters match multiple resource groups: {meta}"
                    )
                renew_data, site_map = load_resource_group_data(
                    resource_groups[0], cache=False
                )
                data = assign_site_cluster(
                    renew_data=renew_data,
                    profile_path=resource_groups[0].group.get("profiles"),
                    regions=regions,
                    site_map=site_map,
                    utc_offset=settings.get("utc_offset", 0),
                    weather_year=settings.get("weather_year"),
                    **_scenario,
                )
                if data.empty:
                    continue
                clusters = (
                    data.groupby("cluster", as_index=False)
                    .apply(calc_cluster_values, _scenario.get("group"))
                    .rename(columns={"mw": "Max_Cap_MW"})
                    .assign(technology=technology, region=region)
                )

                cache_folder.mkdir(parents=True, exist_ok=True)
                if not cache_cluster_fpath.exists() and cache_results:
                    clusters.to_parquet(cache_cluster_fpath)
                if not cache_site_assn_fpath.exists() and cache_results:
                    cols = ["cpa_id", "cluster"]
                    data[cols].to_parquet(cache_site_assn_fpath)
            if settings.get("extra_outputs"):
                # fn = f"{region}_{technology}{new_tech_suffix}_site_cluster_assignments.csv"
                Path(settings["extra_outputs"]).mkdir(parents=True, exist_ok=True)
                cols = ["cpa_id", "cluster"]
                fn = f"{region}_{technology}{new_tech_suffix}_site_cluster_assignments.csv"
                data.loc[:, cols].to_csv(
                    Path(settings["extra_outputs"]) / fn, index=False
                )
        else:
            if cache_cluster_fpath.exists() and use_cache:
                clusters = pd.read_parquet(cache_cluster_fpath)
                data = None
            else:
                clusters = (
                    cluster_builder.get_clusters(
                        **_scenario,
                        ipm_regions=regions,
                        existing=False,
                        utc_offset=settings.get("utc_offset", 0),
                        weather_year=settings.get("weather_year"),
                    )
                    .rename(columns={"mw": "Max_Cap_MW"})
                    .assign(technology=technology, region=region)
                )
                clusters["cluster"] = range(1, 1 + len(clusters))
                data = None
        cache_folder.mkdir(parents=True, exist_ok=True)
        if not cache_cluster_fpath.exists() and cache_results:
            clusters.to_parquet(cache_cluster_fpath)
        if not cache_site_assn_fpath.exists() and data is not None and cache_results:
            cols = ["cpa_id", "cluster"]
            data[cols].to_parquet(cache_site_assn_fpath)
        if _scenario.get("min_capacity"):
            # Warn if total capacity less than expected
            capacity = clusters["Max_Cap_MW"].sum()
            if capacity < _scenario["min_capacity"]:
                logger.warning(
                    f"Selected technology {_scenario['technology']} capacity"
                    + f" in region {region}"
                    + f" less than minimum ({capacity} < {_scenario['min_capacity']} MW)"
                )
        row = df[df["technology"] == technology].to_dict("records")[0]
        clusters["technology"] = clusters["technology"] + new_tech_suffix
        kwargs = {k: v for k, v in row.items() if k not in clusters}
        cdfs.append(
            clusters.assign(**kwargs).pipe(
                modify_renewable_group, _scenario.get("group_modifiers")
            )
        )
    return pd.concat([df[~mask]] + cdfs, sort=False)


def load_user_defined_techs(settings: dict) -> pd.DataFrame:
    """Load user-defined technologies from a CSV file. Returns cost columns and heat
    rate.

    Parameters
    ----------
    settings : dict
        User-defined parameters from a settings file. It must have the key
        'additional_technologies_fn'. The value can either be a string (name of a single
        file) or a dictionary. If the value is a dictionary it should have integer keys
        corresponding to model years and corresponding string values (file name).

        settings['additional_technologies_fn'] = 'user_techs.csv'
        OR
        settings['additional_technologies_fn'] = {
            2030: 'user_techs_2030.csv',
            2045: 'user_techs_2045.csv'
        }

    Returns
    -------
    pd.DataFrame
        A dataframe of user-defined resources with cost and heat rate columns.
    """
    if isinstance(settings["additional_technologies_fn"], collections.abc.Mapping):
        fn = settings["additional_technologies_fn"][settings["model_year"]]
    else:
        fn = settings["additional_technologies_fn"]

    # Search the extra inputs folder first, then the legacy additional_techs folder
    # in repo
    if (Path(settings["input_folder"]) / fn).exists():
        user_techs = pd.read_csv(Path(settings["input_folder"]) / fn)
    else:
        logger.warning(
            "The file with your user defined technologies is not in the user input "
            "folder. Reading the file from PowerGenome/data/additional_technolgies "
            "instead. This may be depreciated in a future version, please move "
            f"{fn} to the folder {settings['input_folder']}."
        )
        user_techs = pd.read_csv(DATA_PATHS["additional_techs"] / fn)

    user_techs = user_techs.loc[
        (user_techs["technology"].isin(settings["additional_new_gen"]))
        & (user_techs["planning_year"] == settings["model_year"]),
        :,
    ]

    user_techs = user_techs.fillna(0)

    if "tech_detail" not in user_techs.columns:
        user_techs["tech_detail"] = ""
    if "cost_case" not in user_techs.columns:
        user_techs["cost_case"] = ""
    if "Cap_Size" not in user_techs.columns:
        user_techs["Cap_Size"] = 1

    if "dollar_year" in user_techs.columns:
        for idx, row in user_techs.iterrows():
            for col in [
                "capex_mw",
                "capex_mwh",
                "fixed_o_m_mw",
                "fixed_o_m_mwh",
                "variable_o_m_mwh",
            ]:
                user_techs.loc[idx, col] = inflation_price_adjustment(
                    row[col],
                    row["dollar_year"],
                    settings["target_usd_year"],
                    data_location=settings["data_location"],
                    table_name=settings["dollar_year_table"],
                )

    cols = [
        "technology",
        "tech_detail",
        "cost_case",
        "capex_mw",
        "capex_mwh",
        "fixed_o_m_mw",
        "fixed_o_m_mwh",
        "variable_o_m_mwh",
        "wacc_real",
        "heat_rate",
        "Cap_Size",
        "dollar_year",
    ]

    return user_techs[cols]
