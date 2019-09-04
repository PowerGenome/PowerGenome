"""Test functions in generation.py"""
import logging
import pytest
import pandas as pd
import numpy as np
import src
from src.generators import (
    group_technologies,
    fill_missing_tech_descriptions,
    label_small_hydro,
    label_retirement_year,
)
from src.params import DATA_PATHS
from src.util import load_settings, reverse_dict_of_lists, map_agg_region_names

logger = logging.getLogger(src.__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    # More extensive test-like formatter...
    "%(asctime)s [%(levelname)8s] %(name)s:%(lineno)s %(message)s",
    # This is the datetime format string.
    "%Y-%m-%d %H:%M:%S",
)
handler.setFormatter(formatter)
logger.addHandler(handler)


@pytest.fixture(scope="module")
def generation_fuel_eia923_data():
    gen_fuel = pd.read_csv(
        DATA_PATHS["test_data"] / "generation_fuel_eia923.csv",
        parse_dates=["report_date"],
    )
    return gen_fuel


@pytest.fixture(scope="module")
def generators_eia860_data():
    gens_860 = pd.read_csv(
        DATA_PATHS["test_data"] / "generators_eia860.csv",
        parse_dates=["report_date", "planned_retirement_date"],
    )
    return gens_860


@pytest.fixture(scope="module")
def generators_entity_eia_data():
    gen_entity = pd.read_csv(
        DATA_PATHS["test_data"] / "generators_entity_eia.csv",
        parse_dates=["operating_date"],
    )
    return gen_entity


@pytest.fixture(scope="module")
def boiler_generator_assn_eia_860_data():
    bga = pd.read_csv(DATA_PATHS["test_data"] / "boiler_generator_assn_eia_860.csv")
    return bga


@pytest.fixture(scope="module")
def hr_by_unit_data():
    hr_by_unit = pd.read_csv(
        DATA_PATHS["test_data"] / "hr_by_unit.csv", parse_dates=["report_date"]
    )
    return hr_by_unit


@pytest.fixture(scope="module")
def plant_region_map_ipm_data():
    plant_region_map = pd.read_csv(DATA_PATHS["test_data"] / "plant_region_map_ipm.csv")
    return plant_region_map


@pytest.fixture(scope="module")
def test_settings():
    settings = load_settings(DATA_PATHS["test_data"] / "pudl_data_extraction.yml")
    return settings


def test_group_technologies(generators_eia860_data, test_settings):
    df = generators_eia860_data.loc[
        generators_eia860_data.report_date.dt.year == 2017, :
    ]
    # df = df.query("report_date.dt.year==2017")
    df = df.drop_duplicates(subset=["plant_id_eia", "generator_id"])

    grouped_by_tech = group_technologies(df, test_settings)
    techs = grouped_by_tech["technology_description"].unique()
    capacities = grouped_by_tech.groupby("technology_description")["capacity_mw"].sum()
    expected_hydro_cap = 48.1
    hydro_cap = capacities["Conventional Hydroelectric"]
    expected_peaker_cap = 354.8
    peaker_cap = capacities["Peaker"]

    assert len(df) == len(grouped_by_tech)
    assert df["capacity_mw"].sum() == grouped_by_tech["capacity_mw"].sum()
    assert "Peaker" in techs
    assert np.allclose(hydro_cap, expected_hydro_cap)
    assert np.allclose(peaker_cap, expected_peaker_cap)


def test_fill_missing_tech_descriptions(generators_eia860_data):
    filled = fill_missing_tech_descriptions(generators_eia860_data)

    assert len(generators_eia860_data) == len(
        filled.dropna(subset=["technology_description"])
    )


def test_label_small_hyro(
    generators_eia860_data, test_settings, plant_region_map_ipm_data
):
    region_agg_map = reverse_dict_of_lists(test_settings["region_aggregations"])
    model_region_map_df = map_agg_region_names(
        df=plant_region_map_ipm_data,
        region_agg_map=region_agg_map,
        original_col_name="region",
        new_col_name="model_region",
    )
    df = pd.merge(
        generators_eia860_data, model_region_map_df, on="plant_id_eia", how="left"
    )
    logger.info(df[["plant_id_eia", "technology_description", "model_region"]].head())

    # df["model_region"] = df["region"].map(reverse_dict_of_lists)

    df = label_small_hydro(df, test_settings)
    print(df.query("plant_id_eia==34"))
    logger.info(df["technology_description"].unique())

    assert "Small Hydroelectric" in df["technology_description"].unique()
    assert np.allclose(
        df.loc[df.technology_description == "Small Hydroelectric", "capacity_mw"], 12.1
    )


def test_label_retirement_year(
    generators_eia860_data, generators_entity_eia_data, test_settings
):
    gens = pd.merge(
        generators_eia860_data,
        generators_entity_eia_data,
        on=["plant_id_eia", "generator_id"],
        how="left",
    )
    df = label_retirement_year(gens, test_settings)

    assert df.loc[df["retirement_year"].isnull(), :].empty is True
