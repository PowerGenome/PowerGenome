import collections
from copy import deepcopy
import subprocess

import pandas as pd
import pudl
import requests
import sqlalchemy as sa

import yaml
from ruamel.yaml import YAML
from pathlib import Path

from powergenome.params import SETTINGS


def load_settings(path):

    with open(path, "r") as f:
        #     settings = yaml.safe_load(f)
        yaml = YAML(typ="safe")
        settings = yaml.load(f)

    return settings


def init_pudl_connection(freq="YS"):

    pudl_engine = sa.create_engine(
        SETTINGS["pudl_db"]
    )  # pudl.init.connect_db(SETTINGS)
    pudl_out = pudl.output.pudltabl.PudlTabl(freq=freq, pudl_engine=pudl_engine)

    return pudl_engine, pudl_out


def reverse_dict_of_lists(d):

    return {v: k for k in d for v in d[k]}


def map_agg_region_names(df, region_agg_map, original_col_name, new_col_name):

    df[new_col_name] = df.loc[:, original_col_name]

    df.loc[df[original_col_name].isin(region_agg_map.keys()), new_col_name] = df.loc[
        df[original_col_name].isin(region_agg_map.keys()), original_col_name
    ].map(region_agg_map)

    return df


def snake_case_col(col):
    "Remove special characters and convert to snake case"
    clean = (
        col.str.lower()
        .str.replace(r"[^0-9a-zA-Z\-]+", " ")
        .str.replace("-", "")
        .str.strip()
        .str.replace(" ", "_")
    )
    return clean


def snake_case_str(s):
    "Remove special characters and convert to snake case"
    clean = (
        s.lower()
        .replace(r"[^0-9a-zA-Z\-]+", " ")
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


def download_save(url, save_path):
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


def shift_wrap_profiles(df, offset):
    "Shift hours to a local offset and append first rows to end"

    wrap_rows = df.iloc[:offset, :]

    shifted_wrapped_df = pd.concat([df.iloc[offset:, :], wrap_rows], ignore_index=True)
    return shifted_wrapped_df


def update_dictionary(d, u):
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
        _df["Fuel"] = _df["Fuel"].str.replace(f"_{s}", "")

    return _df


def write_results_file(df, folder, file_name, include_index=False):
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
    """
    sub_folder = folder / "Inputs"
    sub_folder.mkdir(exist_ok=True, parents=True)

    path_out = sub_folder / file_name

    df.to_csv(path_out, index=include_index)


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
