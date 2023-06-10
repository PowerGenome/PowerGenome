"""
Parameters and settings
"""
import os
from pathlib import Path
from typing import Union

from dotenv import find_dotenv, load_dotenv

from powergenome import __file__
from powergenome.resource_clusters import ClusterBuilder
from powergenome.util import sqlalchemy_prefix

# Not convinced this is the best way to set folder paths but it works!
powergenome_path = Path(__file__).parent
project_path = powergenome_path.parent

load_dotenv(dotenv_path=powergenome_path / ".env")


DATA_PATHS = {}
DATA_PATHS["results"] = project_path / "results"
DATA_PATHS["powergenome"] = project_path / "powergenome"
DATA_PATHS["data"] = project_path / "data"
DATA_PATHS["atb_storage_costs"] = DATA_PATHS["data"] / "NREL_ATB_battery_costs.csv"
DATA_PATHS["ipm_shapefiles"] = DATA_PATHS["data"] / "IPM Regions v617 04-05-17"
DATA_PATHS["tests"] = project_path / "tests"
DATA_PATHS["test_data"] = DATA_PATHS["tests"] / "data"
DATA_PATHS["settings"] = project_path / "settings"
DATA_PATHS["eia"] = DATA_PATHS["data"] / "eia"
DATA_PATHS["eia_860m"] = DATA_PATHS["eia"] / "860m"
DATA_PATHS["cost_multipliers"] = DATA_PATHS["data"] / "cost_multipliers"
DATA_PATHS["cache"] = DATA_PATHS["data"] / "cache"
DATA_PATHS["cache"].mkdir(exist_ok=True)
DATA_PATHS["additional_techs"] = DATA_PATHS["data"] / "additional_technologies"
DATA_PATHS["coal_fgd"] = DATA_PATHS["data"] / "coal_fgd" / "fgd_output.csv"
DATA_PATHS["cpi_data"] = DATA_PATHS["data"] / "cpi_data" / "cpi_data.csv"

IPM_SHAPEFILE_PATH = DATA_PATHS["ipm_shapefiles"] / "IPM_Regions_201770405.shp"
IPM_GEOJSON_PATH = DATA_PATHS["data"] / "ipm_regions_simple.geojson"

SETTINGS = {}
SETTINGS["PUDL_DB"] = sqlalchemy_prefix(os.environ.get("PUDL_DB"))
SETTINGS["PG_DB"] = sqlalchemy_prefix(os.environ.get("PG_DB"))
SETTINGS["EFS_DATA"] = os.environ.get("EFS_DATA")
SETTINGS["RESOURCE_GROUPS"] = os.environ.get("RESOURCE_GROUPS")
SETTINGS["DISTRIBUTED_GEN_DATA"] = os.environ.get("DISTRIBUTED_GEN_DATA")
SETTINGS["RESOURCE_GROUP_PROFILES"] = os.environ.get("RESOURCE_GROUP_PROFILES")


def build_resource_clusters(group_path: Union[str, Path] = None):
    if not group_path:
        group_path = SETTINGS.get("RESOURCE_GROUPS")
    if not group_path:
        cluster_builder = ClusterBuilder([])
    else:
        cluster_builder = ClusterBuilder.from_json(
            Path(group_path, ".").glob("**/*.json")
        )
    return cluster_builder
