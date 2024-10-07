"Test functions for interregional transmission lines"

import os
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from powergenome.external_data import insert_user_tx_costs, load_user_tx_costs
from powergenome.params import DATA_PATHS
from powergenome.transmission import agg_transmission_constraints
from powergenome.util import (
    build_scenario_settings,
    init_pudl_connection,
    load_settings,
)

if os.name == "nt":
    # if user is using a windows system
    sql_prefix = "sqlite:///"
else:
    sql_prefix = "sqlite:////"
pudl_engine, pudl_out, pg_engine = init_pudl_connection(
    pudl_db=sql_prefix + str(DATA_PATHS["test_data"] / "pudl_test_data.db"),
    pg_db=sql_prefix + str(DATA_PATHS["test_data"] / "pg_misc_tables.sqlite3"),
)


@pytest.fixture(scope="module")
def CA_AZ_settings():
    settings = load_settings(
        DATA_PATHS["powergenome"].parent / "example_systems" / "CA_AZ" / "settings"
    )
    settings["input_folder"] = Path(
        DATA_PATHS["powergenome"].parent
        / "example_systems"
        / "CA_AZ"
        / settings["input_folder"]
    )
    settings["RESOURCE_GROUPS"] = DATA_PATHS["test_data"] / "resource_groups_base"
    scenario_definitions = pd.read_csv(
        settings["input_folder"] / settings["scenario_definitions_fn"]
    )
    scenario_settings = build_scenario_settings(settings, scenario_definitions)

    return scenario_settings[2030]["p1"]


def test_load_user_tx(tmp_path):
    cols = [
        "start_region",
        "dest_region",
        "total_interconnect_annuity_mw",
        "total_interconnect_cost_mw",
        "total_line_loss_frac",
        "dollar_year",
    ]
    tx_line = namedtuple("tx_line", cols)
    lines = [
        tx_line("CA_S", "WECC_AZ", 1000, 100000, 0.07, 2018),
        tx_line("CA_N", "CA_N", 2000, 200000, 0.06, 2018),
    ]
    user_tx = pd.DataFrame(lines)
    user_tx.to_csv(tmp_path / "tx_lines.csv", index=False)

    model_regions = ["CA_N", "CA_S", "WECC_AZ"]
    target_usd_year = 2020

    user_tx_costs = load_user_tx_costs(
        tmp_path / "tx_lines.csv", model_regions, target_usd_year
    )

    assert all(
        user_tx_costs["total_interconnect_annuity_mw"]
        > user_tx["total_interconnect_annuity_mw"]
    )

    user_tx_costs = load_user_tx_costs(
        tmp_path / "tx_lines.csv", model_regions, target_usd_year=None
    )

    assert np.allclose(
        user_tx_costs["total_interconnect_annuity_mw"],
        user_tx["total_interconnect_annuity_mw"],
    )


def test_insert_user_tx_costs(tmp_path, CA_AZ_settings):
    cols = [
        "start_region",
        "dest_region",
        "total_interconnect_annuity_mw",
        "total_interconnect_cost_mw",
        "total_line_loss_frac",
        "dollar_year",
    ]
    tx_line = namedtuple("tx_line", cols)
    lines = [
        tx_line("CA_S", "WECC_AZ", 1000, 100000, 0.07, 2018),
        tx_line("CA_N", "CA_N", 2000, 200000, 0.06, 2018),
    ]
    user_tx = pd.DataFrame(lines)
    user_tx.to_csv(tmp_path / "tx_lines.csv", index=False)
    model_regions = ["CA_N", "CA_S", "WECC_AZ"]
    target_usd_year = 2020

    user_tx = load_user_tx_costs(
        tmp_path / "tx_lines.csv", model_regions, target_usd_year
    )

    tx_constraints = agg_transmission_constraints(
        pg_engine,
        CA_AZ_settings,
    )

    combined_tx = insert_user_tx_costs(tx_constraints, user_tx)

    assert len(combined_tx) == 2

    req_cols = [
        "Network_Lines",
        "z1",
        "z2",
        "z3",
        "Line_Reinforcement_Cost_per_MWyr",
        "Line_Reinforcement_Cost_per_MW",
        "Line_Loss_Percentage",
    ]
    for col in req_cols:
        assert col in combined_tx.columns
        assert combined_tx[col].notnull().all()
