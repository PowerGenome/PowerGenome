import collections
import csv
import hashlib
import itertools
import logging
import os
import re
import subprocess
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

os.environ["USE_PYGEOS"] = "0"
import geopandas as gpd
import pandas as pd
import pudl
import requests
import sqlalchemy as sa
import yaml
from flatten_dict import flatten
from ruamel.yaml import YAML

logger = logging.getLogger(__name__)


def load_settings(path: Union[str, Path]) -> dict:
    """Load a YAML file or a dictionary of YAML files with settings parameters

    Parameters
    ----------
    path : Union[str, Path]
        Name of the settings file or folder

    Returns
    -------
    dict
        All parameters listed in the YAML file(s)
    """

    path = Path(path)
    if path.is_file():
        with open(path, "r") as f:
            #     settings = yaml.safe_load(f)
            yaml = YAML(typ="safe")
            settings = yaml.load(f)
    elif path.is_dir():
        settings = {}
        for sf in path.glob("*.yml"):
            yaml = YAML(typ="safe")
            s = yaml.load(sf)
            if s:
                settings.update(s)
    else:
        raise FileNotFoundError(
            "Path is not recognized. Check that your path is valid."
        )

    if settings.get("input_folder"):
        settings["input_folder"] = path.parent / settings["input_folder"]

    settings = apply_all_tag_to_regions(settings)
    settings = sort_nested_dict(settings)

    for key in ["PUDL_DB", "PG_DB"]:
        # Add correct connection string prefix if it isn't there
        if settings.get(key):
            settings[key] = sqlalchemy_prefix(settings[key])

    for key in [
        "EFS_DATA",
        "RESOURCE_GROUPS",
        "DISTRIBUTED_GEN_DATA",
        "RESOURCE_GROUP_PROFILES",
    ]:
        if settings.get(key):
            settings[key] = Path(settings[key])

    return fix_param_names(settings)


def sort_nested_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Given a nested dictionary, iterate through all levels to sort keys by length.
    Dictionary values can be more nested dictionaries, strings, numbers, or lists.

    Parameters
    ----------
    d : Dict[str, Any]
        The nested dictionary to be sorted.

    Returns
    -------
    Dict[str, Any]
        The sorted dictionary where keys are sorted by length at each level.

    """
    sorted_dict = dict()

    for key, value in sorted(d.items(), key=lambda x: len(str(x[0]))):
        if isinstance(value, dict):
            sorted_dict[key] = sort_nested_dict(value)
        else:
            sorted_dict[key] = value

    return sorted_dict


def sqlalchemy_prefix(db_path: str) -> str:
    """Check the database path and add sqlite prefix if needed

    Parameters
    ----------
    db_path : str
        Path to the sqlite database. May or may not include sqlite://// (OS specific)

    Returns
    -------
    str
        SqlAlchemy connection string
    """
    if os.name == "nt":
        # if user is using a windows system
        sql_prefix = "sqlite:///"
    else:
        sql_prefix = "sqlite:////"

    if not db_path:
        return None
    if sql_prefix in db_path:
        return db_path
    else:
        return sql_prefix + str(Path(db_path))


def apply_all_tag_to_regions(settings: dict) -> dict:
    """Make copies of renewables_clusters dicts with region "all"

    If a renewables clustering object doesn't already existing for a region/technology
    then make a copy for use. This is helpful with large numbers of regions when
    the clustering parameters can be applied everywhere.

    Parameters
    ----------
    settings : dict
        All user-specified settings from YAML files

    Returns
    -------
    dict
        Copy of the input settings with renewables_clusters objects for all regions

    Raises
    ------
    KeyError
        The dictionary is missing the tag "region"
    KeyError
        The dictionary with region "all" is missing the tag "technology"
    """

    settings_all = dict()
    all_regions = settings["model_regions"]

    # Keeps a list of which regions should be modified by "all" (are not specifically tagged)
    techs_tagged_w_all = []
    techs_tagged_by_region = dict()

    i = 0
    to_delete = []

    # These are the keys in settings which will not be used to determine whether 'all' should apply to that region
    identifier_keys = ["technology", "pref_site", "turbine_type"]

    for d in settings.get("renewables_clusters", []) or []:
        if "region" not in d:
            raise KeyError("Entry missing 'region' tag.")

        reg = d["region"]

        keys = sorted(d.keys())
        tech = ""
        for key in keys:
            if key in identifier_keys:
                if tech != "":
                    tech += "_"
                tech += str(d[key])

        # Update the dict stating that this technology is specified for this region
        if tech in techs_tagged_by_region:
            techs_tagged_by_region[tech].append(reg)
        elif reg.lower() == "all":
            techs_tagged_by_region[tech] = []
        else:
            techs_tagged_by_region[tech] = [reg]

        if reg.lower() == "all":
            settings_all[tech] = d

            if "technology" not in d:
                raise KeyError(f"""Entry for {reg} missing 'technology' tag.""")

            if tech in techs_tagged_w_all:
                s = f"""
                Multiple 'all' tags applied to technology {tech}. Only last one will be used.
                """
                logger.warning(s)

            else:
                techs_tagged_w_all.append(tech)

            to_delete.append(i)

        # Keeps track of the "all" tags so that they can be deleted later in the function
        i += 1

    for i in reversed(to_delete):
        del settings["renewables_clusters"][i]

    for tech in techs_tagged_w_all:
        for reg in all_regions:
            if reg not in techs_tagged_by_region[tech]:
                temp_entry = settings_all[tech].copy()
                temp_entry["region"] = reg

                settings["renewables_clusters"].append(temp_entry)

    return settings


def fix_param_names(settings: dict) -> dict:
    fix_params = {
        "historical_load_region_maps": "historical_load_region_map",
        "demand_response_resources": "flexible_demand_resources",
        "data_years": "eia_data_years",
    }
    for k, v in fix_params.items():
        if k in settings:
            settings[v] = settings[k]
            s = f"""
            The settings parameter named {k} has been changed to {v}. Please correct it in
            your settings file.

            """
            logger.warning(s)
    return settings


def findkeys(node: Union[dict, list], kv: str):
    """
    Return all values in a dictionary from a matching key
    https://stackoverflow.com/a/19871956
    """
    if isinstance(node, list):
        for i in node:
            for x in findkeys(i, kv):
                yield x
    elif isinstance(node, dict):
        if kv in node:
            yield node[kv]
        for j in node.values():
            for x in findkeys(j, kv):
                yield x


def check_atb_scenario(settings: dict, pg_engine: sa.engine.base.Engine):
    """Check the

    Parameters
    ----------
    settings : dict
        Parameters and values from the YAML settings file.
    pg_engine : sa.engine.base.Engine
        Connection to the PG sqlite database.

    Raises
    ------
    KeyError
        Raises an error if an ATB technology scenario in the settings file doesn't match
        the list of available values for that year of ATB data.
    """
    atb_year = settings.get("atb_data_year")

    s = f"""
    SELECT DISTINCT cost_case
    FROM technology_costs_nrelatb
    WHERE
        atb_year == {atb_year}
    """

    atb_cases = [c[0] for c in pg_engine.execute(s).fetchall()]

    techs = []
    for l in findkeys(settings, "atb_new_gen"):
        techs.extend(l)

    cases = [tech[2] for tech in techs]

    for l in findkeys(settings, "atb_cost_case"):
        cases.append(l)

    bad_case_names = []
    for case in cases:
        if case not in atb_cases:
            bad_case_names.append(case)
    if bad_case_names:
        bad_names = list(set(bad_case_names))
        raise KeyError(
            f"There is an error with the ATB tech scenario key in your settings file."
            f" You are using ATB data from {atb_year}, which has cost cases of:\n\n "
            f"{atb_cases}\n\n"
            "Under either 'atb_new_gen' or 'modified_atb_new_gen' you have cost cases "
            f"of:\n\n{bad_names}\n\n "
            "Try searching your settings file for these "
            "values and replacing them with valid cost cases for your ATB year."
        )


def check_settings(settings: dict, pg_engine: sa.engine) -> None:
    """Check for user errors in the settings file.

    The YAML settings file is loaded as a dictionary object. It has many different parts
    that need to have consistent values. This function checks a few (but not all!) of
    the parameters for common errors or misspelled words.

    Parameters
    ----------
    settings : dict
        Parameters and values from the YAML settings file.
    pg_engine : sa.engine
        Connection to the PG sqlite database.
    """
    if settings.get("atb_data_year"):
        check_atb_scenario(settings, pg_engine)
    ipm_region_list = pd.read_sql_table("regions_entity_epaipm", pg_engine)[
        "region_id_epaipm"
    ].to_list()

    cost_mult_regions = list(
        itertools.chain.from_iterable(
            settings.get("cost_multiplier_region_map", {}).values()
        )
    )

    aeo_fuel_regions = list(
        itertools.chain.from_iterable(settings.get("aeo_fuel_region_map", {}).values())
    )

    atb_techs = settings.get("atb_new_gen", []) or []
    atb_mod_techs = settings.get("modified_atb_new_gen", {}) or {}
    add_new_techs = settings.get("additional_new_gen", []) or []
    cost_mult_techs = []
    for k, v in settings.get("cost_multiplier_technology_map", {}).items():
        for t in v:
            cost_mult_techs.append(t)

    # Make sure atb techs are spelled correctly and are in the cost_multiplier_technology_map
    for tech in atb_techs:
        tech, tech_detail, cost_case, _ = tech

        s = f"""
        SELECT technology, tech_detail
        from technology_costs_nrelatb
        where
            technology == "{tech}"
            AND tech_detail == "{tech_detail}"
        """
        if len(pg_engine.execute(s).fetchall()) == 0:
            s = f"""
    *****************************
    The technology {tech} - {tech_detail} listed in your settings file under 'atb_new_gen'
    does not match any NREL ATB technologies. Check your settings file to ensure it is
    spelled correctly"
    *****************************
    """
            logger.warning(s)

        if f"{tech}_{tech_detail}" not in cost_mult_techs:
            s = f"""
    *****************************
    The ATB technology "{tech}_{tech_detail}" listed in your settings file under 'atb_new_gen'
    is not fully specified in the 'cost_multiplier_technology_map' settings parameter.
    Part of the <tech>_<tech_detail> string might be included, but it is best practice to
    include the full name in this format. Check your settings file.
        """
            logger.warning((s))

    for mod_tech in atb_mod_techs.values():
        mt_name = f"{mod_tech['new_technology']}_{mod_tech['new_tech_detail']}"
        if mt_name not in cost_mult_techs:
            s = f"""
    *****************************
    The modified ATB technology "{mt_name}" listed in your settings file under
    'modified_atb_new_gen' is not fully specified in the 'cost_multiplier_technology_map'
    settings parameter. Part of the <new_technology>_<new_tech_detail> string might be
    included, but it is best practice to include the full name in this format. Check
    your settings file.
        """
            logger.warning((s))

    for add_tech in add_new_techs:
        if add_tech not in cost_mult_techs:
            s = f"""
    *****************************
    The additional user-specified technology "{add_tech}" listed in your settings file under
    'additional_new_gen' is not fully specified in the 'cost_multiplier_technology_map'
    settings parameter. Part of the name string might be included, but it is best practice
    to include the full name in this format. Check your settings file.
        """
            logger.warning((s))

    for agg_region, ipm_regions in (settings.get("region_aggregations") or {}).items():
        for ipm_region in ipm_regions:
            if ipm_region not in ipm_region_list:
                s = f"""
    *****************************
    There is no IPM region {ipm_region}, which is listed in {agg_region}"
    *****************************
    """
                logger.warning(s)

    for model_region in settings["model_regions"]:
        if model_region not in cost_mult_regions:
            s = f"""
    *****************************
    The model region {model_region} is not included in the settings parameter `cost_multiplier_region_map`"
    *****************************
            """
            logger.warning(s)

        if model_region not in aeo_fuel_regions:
            s = f"""
    *****************************
    The model region {model_region} is not included in the settings parameter `aeo_fuel_region_map`"
    *****************************
            """
            logger.warning(s)

    gen_col_count = collections.Counter(settings.get("generator_columns", []))
    duplicate_cols = [c for c, num in gen_col_count.items() if num > 1]
    if duplicate_cols:
        raise KeyError(
            f"The settings parameter 'generator_columns' has duplicates of {duplicate_cols}."
            " Remove the duplicates and try again."
        )

    if settings.get("eia_aeo_year") or settings.get("fuel_eia_aeo_year"):
        fuel_aeo_year = settings.get("fuel_eia_aeo_year") or settings.get(
            "eia_aeo_year"
        )
        for k, v in settings.get("eia_series_scenario_names", {}).items():
            if "REF" in v and str(fuel_aeo_year) not in v:
                logger.warning(
                    "The settings EIA fuel scenario (eia_series_scenario_names) key "
                    f"{k} has a value of {v}, which does not match the aeo data year "
                    f"{fuel_aeo_year}. It has been changed to REF{fuel_aeo_year}."
                )
                settings["eia_series_scenario_names"][k] = f"REF{fuel_aeo_year}"

    if settings.get("eia_aeo_year") or settings.get("load_eia_aeo_year"):
        load_aeo_year = settings.get("load_eia_aeo_year") or settings.get(
            "eia_aeo_year"
        )
        growth_scenario = settings.get("growth_scenario", "")
        if "REF" in growth_scenario and str(load_aeo_year) not in growth_scenario:
            logger.warning(
                "The settings EIA demand growth scenario (growth_scenario) key "
                f"value is {growth_scenario}, which does not match the aeo data year "
                f"{load_aeo_year}. It has been changed to REF{load_aeo_year}."
            )
            settings["growth_scenario"] = f"REF{load_aeo_year}"

    if not settings.get("interest_compound_method"):
        logger.info(
            "The default interest compounding method for calculating annuities has "
            "changed from continuous to discrete. This method can be set with the parameter "
            "'interest_compound_method', using values `discrete` or `continuous`.\n"
            "This message will be removed after version 0.7.0."
        )


def init_pudl_connection(
    freq: str = "AS",
    start_year: int = None,
    end_year: int = None,
    pudl_db: str = None,
    pg_db: str = None,
) -> Tuple[sa.engine.base.Engine, pudl.output.pudltabl.PudlTabl]:
    """Initiate a connection object to the sqlite PUDL database and create a pudl
    object that can quickly access parts of the database.

    Parameters
    ----------
    freq : str, optional
        The time frequency that data should be averaged over in the `pudl_out` object,
        by default "YS" (annual data).

    Returns
    -------
    sa.Engine, pudl.pudltabl
        A sqlalchemy engine for connecting to the PUDL database, and a pudl PudlTabl
        object for quickly accessing parts of the database. `pudl_out` is used
        to access unit heat rates.
    """
    from powergenome.params import SETTINGS

    if not pudl_db:
        pudl_db = SETTINGS["PUDL_DB"]
    if not pg_db:
        if SETTINGS.get("PG_DB"):
            pg_db = SETTINGS["PG_DB"]
        else:
            logger.warning(
                "No path to a `PG_DB` database was provided or found in the .env file. Using "
                "the `PUDL_DB` path instead."
            )
            pg_db = SETTINGS["PUDL_DB"]
    pudl_engine = sa.create_engine(pudl_db)
    if start_year is not None:
        start_year = pd.to_datetime(start_year, format="%Y")
    if end_year is not None:
        end_year = pd.to_datetime(end_year, format="%Y")
    """
    pudl_out = pudl.output.pudltabl.PudlTabl(
        freq=freq, pudl_engine=pudl_engine, start_date=start_year, end_date=end_year
        #freq=freq, pudl_engine=pudl_engine, start_date=start_year, end_date=end_year, ds=""
    )
    """
    pudl_out = pudl.output.pudltabl.PudlTabl(
        freq=freq,
        pudl_engine=pudl_engine,
        start_date=start_year,
        end_date=end_year,
        ds=pudl.workspace.datastore.Datastore(),
    )
    pg_engine = sa.create_engine(pg_db)
    # if SETTINGS.get("PG_DB"):
    #     pg_engine = sa.create_engine(SETTINGS["PG_DB"])
    # else:
    #     logger.warning(
    #         "No path to a `PG_DB` database was found in the .env file. Using the "
    #         "`PUDL_DB` path instead."
    #     )
    #     pg_engine = sa.create_engine(SETTINGS["PUDL_DB"])

    return pudl_engine, pudl_out, pg_engine


def reverse_dict_of_lists(d: Dict[str, list]) -> Dict[str, List[str]]:
    """Reverse the mapping in a dictionary of lists so each list item maps to the key

    Parameters
    ----------
    d : Dict[str, List[str]]
        A dictionary with string keys and lists of strings.

    Returns
    -------
    Dict[str, str]
        A reverse mapped dictionary where the item of each list becomes a key and the
        original keys are mapped as values.
    """
    if isinstance(d, collections.abc.Mapping):
        rev = {v: k for k in d for v in d[k]}
    else:
        rev = dict()
    return rev


def map_agg_region_names(
    df: pd.DataFrame,
    region_agg_map: Dict[str, List[str]],
    original_col_name: str,
    new_col_name: str,
) -> pd.DataFrame:
    """Add a column that maps original region names to aggregated model region names.

    A dataframe with un-aggregated region names (e.g. EPA IPM regions) will have a new
    column added. Aggregated model region names will be used in the new column. If a
    model region is not part of an aggregation it will be left as-is in the new column.

    Parameters
    ----------
    df : pd.DataFrame
        Original dataframe with column 'original_col_name'
    region_agg_map : Dict[str, List[str]]
        Mapping of model region names (keys) to a list of aggregated base regions
    original_col_name : str
        Name of the original column with region names.
    new_col_name : str
        Name for the column with mapped model region values.

    Returns
    -------
    pd.DataFrame
        A modified version of the original dataframe with the new column "new_col_name"
        that has values of model regions.
    """

    df[new_col_name] = df.loc[:, original_col_name]

    df.loc[df[original_col_name].isin(region_agg_map.keys()), new_col_name] = df.loc[
        df[original_col_name].isin(region_agg_map.keys()), original_col_name
    ].map(region_agg_map)

    return df


def snake_case_col(col: pd.Series) -> pd.Series:
    "Remove special characters and convert to snake case"
    clean = (
        col.str.lower()
        .str.replace(r"[^0-9a-zA-Z\-]+", " ", regex=True)
        .str.replace("-", "")
        .str.strip()
        .str.replace(" ", "_")
    )
    return clean


def snake_case_str(s: str) -> str:
    "Remove special characters and convert to snake case"
    if s:
        clean = (
            re.sub(r"[^0-9a-zA-Z\-]+", " ", s)
            .lower()
            .replace("-", "")
            .strip()
            .replace(" ", "_")
        )
        return clean


def get_git_hash():
    try:
        git_head_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .strip()
            .decode("ascii")
        )
    except FileNotFoundError:
        git_head_hash = "Git hash unknown"

    return git_head_hash


def download_save(url: str, save_path: Union[str, Path]):
    """
    Download a file that isn't zipped and save it to a given path

    Parameters
    ----------
    url : str
        Valid url to download the zip file
    save_path : str or path object
        Destination to save the file

    """

    r = requests.get(url)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_bytes(r.content)


def update_dictionary(d: dict, u: dict) -> dict:
    """
    Update keys in an existing dictionary (d) with values from u
    Sort keys in updated dictionary by their lengths from shortest to longest

    Parameters
    ----------
    d : dict
        The existing dictionary to be updated.
    u : dict
        The dictionary containing the new values to update the existing dictionary.

    Returns
    -------
    dict
        The updated dictionary with keys sorted by length.
    """
    if not (isinstance(d, collections.abc.Mapping) or d is None) or not isinstance(
        u, collections.abc.Mapping
    ):
        raise TypeError("Inputs must be dictionaries")

    for k, v in sorted(u.items(), key=lambda item: len(str(item[0]))):
        if isinstance(d, collections.abc.Mapping):
            if isinstance(v, collections.abc.Mapping):
                r = update_dictionary(d.get(k, {}), v)
                d[k] = r
            else:
                d[k] = u[k]
        else:
            d = {k: u[k]}
    return dict(sorted(d.items(), key=lambda item: len(str(item[0]))))


def remove_fuel_scenario_name(df, settings):
    _df = df.copy()
    scenarios = settings["eia_series_scenario_names"].keys()
    for s in scenarios:
        _df.columns = _df.columns.str.replace(f"_{s}", "")

    return _df


def remove_fuel_gen_scenario_name(df, settings):
    _df = df.copy()
    scenarios = settings["eia_series_scenario_names"].keys()
    for s in scenarios:
        _df["Fuel"] = _df["Fuel"].str.replace(f"_{s}", "")

    return _df


def write_results_file(
    df: pd.DataFrame,
    folder: Path,
    file_name: str,
    include_index: bool = False,
    float_format: str = None,
    multi_period: bool = True,
):
    """Write a finalized dataframe to one of the results csv files.

    Parameters
    ----------
    df : DataFrame
        Data for a single results file
    folder : Path-like
        A Path object representing the folder for a single case/scenario
    file_name : str
        Name of the file.
    include_index : bool, optional
        If pandas should include the index when writing to csv, by default False
    float_format: str
        Parameter passed to pandas .to_csv
    multi_period : bool, optional
        If results should be formatted for multi-period, by default True
    """
    if not multi_period:
        folder = folder / "Inputs"

    folder.mkdir(exist_ok=True, parents=True)

    path_out = folder / file_name
    df.to_csv(path_out, index=include_index, float_format=float_format)


def write_case_settings_file(settings, folder, file_name):
    """Write a finalized dictionary to YAML file.

    Parameters
    ----------
    settings : dict
        A dictionary with settings
    folder : Path-like
        A Path object representing the folder for a single case/scenario
    file_name : str
        Name of the file.
    """
    folder.mkdir(exist_ok=True, parents=True)
    path_out = folder / file_name

    # yaml = YAML(typ="unsafe")
    _settings = deepcopy(settings)
    # for key, value in _settings.items():
    #     if isinstance(value, Path):
    #         _settings[key] = str(value)
    # yaml.register_class(Path)
    # stream = file(path_out, 'w')
    with open(path_out, "w") as f:
        yaml.dump(_settings, f)


def find_centroid(gdf):
    """Find the centroid of polygons, even when in a geographic CRS

    If the crs is geographic (uses lat/lon) then it is converted to a projection before
    calculating the centroid.

    The projected CRS used here is:

    <Projected CRS: EPSG:2163>
    Name: US National Atlas Equal Area
    Axis Info [cartesian]:
    - X[east]: Easting (metre)
    - Y[north]: Northing (metre)
    Area of Use:
    - name: USA
    - bounds: (167.65, 15.56, -65.69, 74.71)
    Coordinate Operation:
    - name: US National Atlas Equal Area
    - method: Lambert Azimuthal Equal Area (Spherical)
    Datum: Not specified (based on Clarke 1866 Authalic Sphere)
    - Ellipsoid: Clarke 1866 Authalic Sphere
    - Prime Meridian: Greenwich

    Parameters
    ----------
    gdf : GeoDataFrame
        A gdf with a geometry column.

    Returns
    -------
    GeoSeries
        A GeoSeries of centroid Points.
    """

    crs = gdf.crs

    if crs.is_geographic:
        _gdf = gdf.to_crs("EPSG:2163")
        centroid = _gdf.centroid
        centroid = centroid.to_crs(crs)
    else:
        centroid = gdf.centroid

    return centroid


def regions_to_keep(
    model_regions: List[str], region_aggregations: dict = {}
) -> Tuple[list, dict]:
    """Create a list of all IPM regions that are used in the model, either as single
    regions or as part of a user-defined model region. Also includes the aggregate
    regions defined by user.

    Parameters
    ----------
    settings : dict
        User-defined parameters from a settings YAML file with keys "model_regions" and
        "region_aggregations".

    Returns
    -------
    list
        All of the IPM regions and user defined model regions.
    """
    # Settings has a dictionary of lists for regional aggregations.
    region_agg_map = reverse_dict_of_lists(region_aggregations)

    # IPM regions to keep - single in model_regions plus those aggregated by the user
    keep_regions = [
        x
        for x in model_regions + list(region_agg_map)
        if x not in region_agg_map.values()
    ]
    return keep_regions, region_agg_map


def build_case_id_name_map(settings: dict) -> dict:
    """Make a dictionary mapping of case IDs and case names from a CSV file

    Parameters
    ----------
    settings : dict
        Settings parameters. Must include `input_folder` and `case_id_description_fn`

    Returns
    -------
    dict
        Mapping of case id to case name
    """
    case_id_name_df = pd.read_csv(
        Path(settings["input_folder"]) / settings["case_id_description_fn"],
        index_col=0,
    ).squeeze("columns")
    case_id_name_df = case_id_name_df.str.replace(" ", "_")
    case_id_name_map = case_id_name_df.to_dict()

    return case_id_name_map


def make_iterable(item: Union[int, str, Iterable]) -> Iterable:
    """Return an iterable version of the one or more items passed.

    Parameters
    ----------
    item : Union[int, str, Iterable]
       Item that may or may not already be iterable

    Returns
    -------
    Iterable
        An iterable version of the item
    """
    if isinstance(item, str):
        i = iter([item])
    else:
        try:
            # check if it's iterable
            i = iter(item)
        except TypeError:
            i = iter([item])
    return i


def assign_model_planning_years(_settings: dict, year: int) -> dict:
    """Make sure "model_year" and "model_first_planning_year" appear as scalars.

    These can originally be set in any of these forms, in either the default
    settings or in the settings_management dictionary:

    model_year: 2040 and model_first_planning_year: 2031
    model_year: [2040, 2050] and model_first_planning_year: [2031, 2041]
    model_periods: (2031, 2040)
    model_periods: [(2031, 2040), (2041, 2050)]

    This function looks up the right values for the current year and assigns
    them as scalars (the first form above).

    Parameters
    ----------
    _settings : dict
        Model settings dictionary. Must have either "model_periods", "model_year"
        AND "model_first_planning_year", or "model_first_planning_year" as keys.
    year : int
        Model year.

    Returns
    -------
    dict
        Modified settings with scaler versions of "model_year" and "model_first_planning_year".

    Raises
    ------
    ValueError
        model_periods is not a series of tuples
    ValueError
        model_periods tuples are not all length 2
    ValueError
        model_year and model_first_planning_year must all be integer
    KeyError
        None of the required keys found
    ValueError
        The model year from scenario definitions is not in the settings
    """
    if "model_periods" in _settings:
        model_periods = make_iterable(_settings["model_periods"])
        if not all([isinstance(t, tuple) for t in model_periods]):
            raise ValueError(
                "The settings parameter 'model_periods' must be a list of tuples. It is "
                f"currently {_settings['model_periods']}"
            )
        if not all(len(t) == 2 for t in model_periods):
            raise ValueError(
                "The tuples in settings parameter 'model_periods' must all be 2 years. "
                f"The values found are {_settings['model_periods']}"
            )
        model_planning_period_dict = {
            year: (start_year, year)
            for (start_year, year) in make_iterable(_settings["model_periods"])
        }
    elif "model_year" in _settings and "model_first_planning_year" in _settings:
        model_year = make_iterable(_settings["model_year"])
        first_planning_year = make_iterable(_settings["model_first_planning_year"])
        if not all(isinstance(y, int) for y in model_year) and all(
            isinstance(y, int) for y in first_planning_year
        ):
            raise ValueError(
                "Both 'model_year' and 'model_first_planning_year' parameters must be "
                f"integers or lists of integers. The values found are {model_periods} and "
                f"{first_planning_year}."
            )
        model_planning_period_dict = {
            year: (start_year, year)
            for year, start_year in zip(
                make_iterable(_settings["model_year"]),
                make_iterable(_settings["model_first_planning_year"]),
            )
        }
    elif "model_first_planning_year" in _settings:
        # we also allow leaving out the model_year tag and just specifying
        # model_first_planning_year
        model_planning_period_dict = {
            year: (
                _settings["model_first_planning_year"],
                _settings["model_first_planning_year"],
            )
        }
    else:
        raise KeyError(
            "To build a dictionary of scenario settings your settings file should include "
            "either the key 'model_periods' (a list of 2-element lists) or the keys "
            "'model_year' and 'model_first_planning_year' (each a list of years)."
        )

    # remove any model period data already there
    for key in ["model_periods", "model_year", "model_first_planning_year"]:
        try:
            del _settings[key]
        except KeyError:
            pass

    if year not in model_planning_period_dict:
        raise ValueError(
            f"The year {year} is in your scenario definition file for case {_settings.get('case_id')} "
            "but was not found in the 'model_year' or 'model_periods' settings parameters. "
            "Either it is missing in the main settings file or was removed in the "
            "'settings_management' section."
        )
    # assign the scalar values
    _settings["model_first_planning_year"] = model_planning_period_dict[year][0]
    _settings["model_year"] = model_planning_period_dict[year][1]

    return _settings


def build_scenario_settings(
    settings: dict, scenario_definitions: pd.DataFrame
) -> Dict[int, Dict[Union[int, str], dict]]:
    """Build a nested dictionary of settings for each planning year/scenario

    Parameters
    ----------
    settings : dict
        The full settings file, including the "settings_management" section with
        alternate values for each scenario
    scenario_definitions : pd.DataFrame
        Values from the csv file defined in the settings file "scenario_definitions_fn"
        parameter. This df has columns corresponding to categories in the
        "settings_management" section of the settings file, with row values defining
        specific case/scenario names.

    Returns
    -------
    dict
        A nested dictionary. The first set of keys are the planning years, the second
        set of keys are the case ID values associated with each case.
    """

    # don't allow duplicate rows in the scenario definitions table, since they
    # could give unexpected results
    dups = scenario_definitions[["case_id", "year"]].duplicated()
    if dups.sum() > 0:
        raise ValueError(
            "The following cases and years are repeated in your scenario definitions file:\n\n"
            + scenario_definitions[dups].to_string(index=False)
        )

    if settings.get("case_id_description_fn"):
        case_id_name_map = build_case_id_name_map(settings)
    else:
        case_id_name_map = None

    all_category_levels = set()
    active_category_levels = set()
    scenario_settings = {}
    missing_flag = object()
    for i, scenario_row in scenario_definitions.iterrows():
        year, case_id = scenario_row[["year", "case_id"]]

        _settings = deepcopy(settings)
        _settings["case_id"] = case_id

        # first apply any settings under "all_years", then any settings for this year
        for settings_year in ["all_years", year]:

            planning_year_settings_management = (
                settings.get("settings_management", {}).get(settings_year) or {}
            )

            # update settings from all_cases entry if available (these settings
            # are applied to all cases for this year, and don't use the category
            # names or levels from the scenario definitions table)
            if "all_cases" in planning_year_settings_management:
                new_parameter = planning_year_settings_management["all_cases"]
                _settings = update_dictionary(_settings, new_parameter)

            modified_settings = {}
            for category, level in scenario_row.drop(["case_id", "year"]).items():
                # category is a column from the scenario definitions table, e.g. ccs_capex
                # level is the selection for this category for this case/year, e.g., "mid" or "none"

                new_parameter = planning_year_settings_management.get(category, {}).get(
                    level, missing_flag
                )

                # Remember category/levels that were selected and that actually
                # had an effect.
                all_category_levels.add((year, category, level))
                if new_parameter is not missing_flag:
                    # note: user could set None or {} as the setting, to indicate
                    # this flag should use the default settings as-is
                    active_category_levels.add((year, category, level))
                if new_parameter in [missing_flag, None, {}]:
                    continue

                _settings = update_dictionary(_settings, new_parameter)

                # report any conflicts between these settings and previous ones
                for key in flatten(new_parameter).keys():
                    if key in modified_settings:
                        raise ValueError(
                            f"The setting {key} is modified by both the "
                            f"`{modified_settings[key]}` flag and the "
                            f"`{category}={level}` flag in the scenario "
                            f"definition for case {case_id}, {year}."
                        )
                    else:
                        # remember this setting for later
                        modified_settings[key] = f"{category}={level}"

        # make sure model year data appears in standard form
        assign_model_planning_years(_settings, year)

        if case_id_name_map:
            _settings["case_name"] = case_id_name_map[case_id]

        scenario_settings.setdefault(year, {})[case_id] = _settings

    # Report any settings in the scenario definitions that had no effect. Values
    # can be changed via either the "all_years" key or a specific year, so we
    # have to wait till the end to decide which tags had no effect.
    missing_category_levels = all_category_levels - active_category_levels
    if missing_category_levels:
        missing = (
            pd.DataFrame(
                missing_category_levels,
                columns=["year", "category", "level"],
            )
            .pivot(index="year", columns="category", values="level")
            .fillna("")
            .reset_index()
        )
        logger.warning(
            "The following parameter value(s) in your scenario definitions file "
            "are not included in the 'settings_management' dictionary for the "
            "specified year(s). Settings will not be modified to reflect these "
            "entries:\n\n"
            + missing.to_string(index=False)
            + "\n\nYou can place empty keys for these in settings_management "
            "dictionary to avoid this message."
        )

    return scenario_settings


def remove_feb_29(df: pd.DataFrame) -> pd.DataFrame:
    """Remove Feb 29 from a wide format leap-year dataseries

    Parameters
    ----------
    df : pd.DataFrame
        A wide format dataframe with 8784 columns

    Returns
    -------
    pd.DataFrame
        The same dataframe but without the 24 hours in Feb 29 and only 8760 rows.
    """
    idx_start = df.index.min()
    idx_name = df.index.name
    df["datetime"] = pd.date_range(start="2012-01-01", freq="H", periods=8784)

    df = df.loc[~((df.datetime.dt.month == 2) & (df.datetime.dt.day == 29)), :]
    df.index = range(idx_start, idx_start + 8760)
    df.index.name = idx_name

    return df.drop(columns=["datetime"])


def load_ipm_shapefile(settings: dict, path: Union[str, Path] = None):
    """
    Load the shapefile of IPM regions

    Parameters
    ----------
    settings : dict
        User-defined parameters from a settings YAML file. This is where any region
        aggregations would be defined.
    path : Union[str, Path]
        Path, loction, or URL of the IPM shapefile/geojson to load. Default value is
        a simplified geojson stored in the PowerGenome data folder.

    Returns
    -------
    geodataframe
        Regions to use in the study with the matching geometry for each.
    """
    if not path:
        from powergenome.params import IPM_GEOJSON_PATH

        path = IPM_GEOJSON_PATH
    keep_regions, region_agg_map = regions_to_keep(
        settings["model_regions"], settings.get("region_aggregations", {}) or {}
    )
    try:
        ipm_regions = gpd.read_file(path, engine="pyogrio")
    except ImportError:
        ipm_regions = gpd.read_file(path, engine="fiona")
    ipm_regions = ipm_regions.rename(columns={"IPM_Region": "region"})

    if settings.get("user_region_geodata_fn"):
        logger.info("Appending user regions to IPM Regions")
        user_regions = gpd.read_file(
            Path(settings["input_folder"]) / settings["user_region_geodata_fn"]
        )
        if "region" not in user_regions.columns:
            region_col = [c for c in user_regions.columns if "region" in c.lower()][0]
            user_regions = user_regions.rename(columns={region_col: "region"})
            logger.warning(
                "The user supplied region geodata file does not include the "
                "property 'region' for any of the region polygons! Automatically detecting "
                "the correct column but this may cause errors."
            )
        user_regions = user_regions.to_crs(ipm_regions.crs)
        ipm_regions = ipm_regions.append(user_regions)

    model_regions_gdf = ipm_regions.loc[ipm_regions["region"].isin(keep_regions)]
    model_regions_gdf = map_agg_region_names(
        model_regions_gdf, region_agg_map, "region", "model_region"
    ).reset_index(drop=True)

    return model_regions_gdf


def deep_freeze(thing):
    """
    https://stackoverflow.com/a/66729248/3393071
    """
    from collections.abc import Collection, Hashable, Mapping

    from frozendict import frozendict

    if thing is None or isinstance(thing, str):
        return thing
    elif isinstance(thing, Mapping):
        return frozendict({k: deep_freeze(v) for k, v in thing.items()})
    elif isinstance(thing, Collection):
        return tuple(deep_freeze(i) for i in thing)
    elif not isinstance(thing, Hashable):
        raise TypeError(f"unfreezable type: '{type(thing)}'")
    else:
        return thing


def deep_freeze_args(func):
    """
    https://stackoverflow.com/a/66729248/3393071
    """
    import functools

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        return func(*deep_freeze(args), **deep_freeze(kwargs))

    return wrapped


def find_region_col(cols: Union[pd.Index, List[str]], context: str = None) -> str:
    """Find the column name that identifies regions.

    DataFrame, geospatial objects, etc might have different names for the region column.
    To retain some flexibility, only require that the region column has the string
    "region" in it (case insensitive).

    Raise an error if more than one column contains the string "region". If `context` is
    provided, include it in the error message for users.

    Parameters
    ----------
    cols : Iterable[str]
        DataFrame columns or other iterable sequence.
    context : str, optional
        Information about the sequence of names that can help a user understand what
        type of object might have multiple names containing "region", by default None

    Returns
    -------
    str
        Name of the column that identifies regions.

    Raises
    ------
    ValueError
        More than one column contains the string "region".
    ValueError
        No column contains the string "region".
    """

    region_col = [c for c in cols if "region" in c.lower()]
    if len(region_col) > 1:
        s = (
            "When attempting to identify the appropriate region columns, more than one "
            f"column in this dataframe includes the string 'region' ({region_col})."
        )
        if context:
            s += f"\n\nContext: {context}"

        raise ValueError(s)
    elif len(region_col) == 0:
        s = (
            "No columns contain the required string 'region'. The DataFrame columns "
            f"are ({cols})."
        )
        if context:
            s += f"\n\nContext: {context}"

        raise ValueError(s)
    else:
        return region_col[0]


def remove_leading_zero(id: Union[str, int]) -> Union[str, int]:
    """Remove leading zero from IDs that are otherwise integers.

    There is a discrepency between some generator IDs in PUDL and 860m where they are
    listed with a leading zero in one and an integer in the other. To better match,
    strip zeros from IDs that would be an integer without them.

    Parameters
    ----------
    id : Union[str, int]
        An integer or string identifier

    Returns
    -------
    Union[str, int]
        Either the original ID (if integer or non-numeric string) or an integer version
        of the ID with leading zeros removed
    """
    if isinstance(id, int):
        return id
    elif id.isnumeric():
        id = id.lstrip("0")
    return id


def hash_string_sha256(input_string: str) -> str:
    """Create a reproducible hash of an input string. Use for creating cache filenames.

    Parameters
    ----------
    input_string : str
        String representing the data to hash

    Returns
    -------
    str
        Hexdigest hash of the input string.
    """
    # For simplicity, require string inputs.
    if not isinstance(input_string, str):
        raise TypeError("The input value cannot be hashed if it is not a string.")
    # Encode the string into bytes
    input_bytes = input_string.encode("utf-8")

    # Create a SHA-256 hash object
    hasher = hashlib.sha256()

    # Pass the bytes to the hasher
    hasher.update(input_bytes)

    # Generate the hexadecimal representation of the digest
    hex_digest = hasher.hexdigest()

    return hex_digest


def add_row_to_csv(file: Path, new_row: List[str], headers: List[str] = None) -> None:
    """Add a row of data to an existing CSV file. If the file does not exist, create it
    with headers and the first row of data.

    Parameters
    ----------
    file : Path
        Path to the CSV file
    new_row : List[str]
        Data to add as a new row in the CSV
    headers : List[str], optional
        Header names, by default None. Required if the file does not exist.

    Raises
    ------
    ValueError
        The file does not exist and no headers were provided.
    """
    file = Path(file)
    # Check if file exists
    if not file.exists():
        if headers is None:
            raise ValueError(
                f"No headers provided. The file {file} does not exist, so headers are "
                "required to create the file."
            )
        file.parent.mkdir(parents=True, exist_ok=True)
        with file.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)  # write headers first time only

    with file.open("r") as f:
        reader = csv.reader(f)
        data = list(reader)  # this contains all the rows in your CSV file

    if new_row not in data:  # check if row already exists in data
        with file.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(new_row)  # add the new row to the CSV file
