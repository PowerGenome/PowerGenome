import subprocess
import yaml

import pudl
from powergenome.params import SETTINGS
import sqlalchemy as sa
import requests


def load_settings(path):

    with open(path, 'r') as f:
        settings = yaml.safe_load(f)

    return settings


def init_pudl_connection(freq='YS'):

    pudl_engine = sa.create_engine(SETTINGS['pudl_db']) # pudl.init.connect_db(SETTINGS)
    pudl_out = pudl.output.pudltabl.PudlTabl(freq=freq, pudl_engine=pudl_engine)

    return pudl_engine, pudl_out


def reverse_dict_of_lists(d):

    return {v: k for k in d for v in d[k]}


def map_agg_region_names(df, region_agg_map, original_col_name, new_col_name):

    df[new_col_name] = df.loc[:, original_col_name]

    df.loc[
        df[original_col_name].isin(region_agg_map.keys()),
        new_col_name
    ] = df.loc[
            df[original_col_name].isin(region_agg_map.keys()),
            original_col_name
        ].map(region_agg_map)

    return df


def snake_case_col(col):
    "Remove special characters and convert to snake case"
    clean = (
        col.str.lower()
        .str.replace('[^0-9a-zA-Z\-]+', ' ')
        .str.replace('-', '')
        .str.strip()
        .str.replace(' ', '_')
    )
    return clean


def snake_case_str(s):
    "Remove special characters and convert to snake case"
    clean = (
        s.lower()
        .replace('[^0-9a-zA-Z\-]+', ' ')
        .replace('-', '')
        .strip()
        .replace(' ', '_')
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
