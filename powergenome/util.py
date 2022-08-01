import collections
from copy import deepcopy
import functools
import itertools
import logging
import re
import subprocess
from typing import Dict, List, Tuple, Union

import pandas as pd
import geopandas as gpd
import pudl
import requests
import sqlalchemy as sa

from flatten_dict import flatten
import yaml
from ruamel.yaml import YAML
from pathlib import Path
from frozendict import frozendict

from powergenome.params import IPM_GEOJSON_PATH, SETTINGS

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

    return fix_param_names(settings)


def fix_param_names(settings: dict) -> dict:

    fix_params = {"historical_load_region_maps": "historical_load_region_map"}
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
        end_year = pd.to_datetime(end_year + 1, format="%Y")
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

    https://stackoverflow.com/a/32357112
    """
    for k, v in u.items():
        if isinstance(d, collections.abc.Mapping):
            if isinstance(v, collections.abc.Mapping):
                r = update_dictionary(d.get(k, {}), v)
                d[k] = r
            else:
                d[k] = u[k]
        else:
            d = {k: u[k]}
    return d


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
    """
    sub_folder = folder / "Inputs"
    sub_folder.mkdir(exist_ok=True, parents=True)

    path_out = sub_folder / file_name
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
        squeeze=True,
    )
    case_id_name_df = case_id_name_df.str.replace(" ", "_")
    case_id_name_map = case_id_name_df.to_dict()

    return case_id_name_map


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

    model_planning_period_dict = {
        year: (start_year, year)
        for year, start_year in zip(
            settings["model_year"], settings["model_first_planning_year"]
        )
    }

    case_id_name_map = build_case_id_name_map(settings)

    scenario_settings = {}
    for year in scenario_definitions["year"].unique():
        scenario_settings[year] = {}
        planning_year_settings_management = settings["settings_management"][year]

        # Create a dictionary with keys of things that change (e.g. ccs_capex) and
        # values of nested dictionaries that give case_id: scenario name
        planning_year_scenario_definitions_dict = (
            scenario_definitions.loc[scenario_definitions.year == year]
            .set_index("case_id")
            .to_dict()
        )
        planning_year_scenario_definitions_dict.pop("year")

        for case_id in scenario_definitions.query("year==@year")["case_id"].unique():
            _settings = deepcopy(settings)

            if "all_cases" in planning_year_settings_management:
                new_parameter = planning_year_settings_management["all_cases"]
                _settings = update_dictionary(_settings, new_parameter)

            # Add the scenario definition values to the settings files
            # e.g.
            # case_id	year	demand_response	growth	tx_expansion	ng_price
            # p1	    2030	moderate	            normal	high	reference
            case_scenario_definitions = scenario_definitions.loc[
                (scenario_definitions.case_id == case_id)
                & (scenario_definitions.year == year),
                :,
            ]
            for col in scenario_definitions.columns:
                _settings[col] = case_scenario_definitions.squeeze().at[col]

            modified_settings = []
            for (
                category,
                case_value_dict,
            ) in planning_year_scenario_definitions_dict.items():
                # key is the category e.g. ccs_capex, case_value_dict is p1: mid
                try:
                    case_value = case_value_dict[case_id]
                    new_parameter = (
                        planning_year_settings_management[category][case_value] or {}
                    )

                    try:
                        settings_keys = list(flatten(new_parameter).keys())
                    except AttributeError:
                        settings_keys = {}

                    for key in settings_keys:
                        assert (
                            key not in modified_settings
                        ), f"The settings key {key} is modified twice in case id {case_id}"

                        modified_settings.append(key)

                    if new_parameter is not None:
                        _settings = update_dictionary(_settings, new_parameter)
                    # print(_settings[list(new_parameter.keys())[0]])

                except KeyError:
                    pass

            _settings["model_first_planning_year"] = model_planning_period_dict[year][0]
            _settings["model_year"] = model_planning_period_dict[year][1]
            _settings["case_name"] = case_id_name_map[case_id]
            scenario_settings[year][case_id] = _settings

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


def load_ipm_shapefile(settings: dict, path: Union[str, Path] = IPM_GEOJSON_PATH):
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
    keep_regions, region_agg_map = regions_to_keep(settings)

    ipm_regions = gpd.read_file(path)
    ipm_regions = ipm_regions.rename(columns={"IPM_Region": "region"})

    if settings.get("user_region_geodata_fn"):
        logger.info("Appending user regions to IPM Regions")
        user_regions = gpd.read_file(
            Path(settings["input_folder"]) / settings["user_region_geodata_fn"]
        )
        if "region" not in user_regions.columns:
            raise KeyError(
                "The user supplied region geodata file does not include the "
                "property 'region' for any of the region polygons! User region "
                "geodata can not be appropriately mapped to model regions."
            )
        user_regions = user_regions.to_crs(ipm_regions.crs)
        ipm_regions = ipm_regions.append(user_regions)

    model_regions_gdf = ipm_regions.loc[ipm_regions["region"].isin(keep_regions)]
    model_regions_gdf = map_agg_region_names(
        model_regions_gdf, region_agg_map, "region", "model_region"
    ).reset_index(drop=True)

    return model_regions_gdf
