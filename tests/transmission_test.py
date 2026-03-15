"Test functions for interregional transmission lines"

import os
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import sqlalchemy as sa

from powergenome.external_data import insert_user_tx_costs, load_user_tx_costs
from powergenome.params import DATA_PATHS
from powergenome.transmission import agg_transmission_constraints, calc_network_upgrade_costs
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


# --------------------------------------------------------------------------- #
# Fixtures and helpers for calc_network_upgrade_costs tests                   #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def network_upgrade_engine(tmp_path_factory):
    """Create a self-contained SQLite database for calc_network_upgrade_costs tests.

    Topology (two model regions):
      - CA_N  = [WEC_CALN, WEC_BANC]   (aggregated, MST needed)
      - WECC_AZ                          (single base region)

    Connectivity (transmission_single_epaipm):
      WEC_BANC  <-> WECC_AZ  (establishes CA_N <-> WECC_AZ as a connected pair)
      WEC_CALN  <-> WECC_AZ  (a second, higher-cost path)
      WEC_BANC  <-> WEC_CALN (intra CA_N connection)

    Costs (transmission_cost_nrel_reeds):
      WEC_BANC  <-> WEC_CALN : capital=100_000, annuity= 4_844, loss=0.02 (intra CA_N)
      WEC_BANC  <-> WECC_AZ  : capital=250_000, annuity=12_110, loss=0.05 (cheapest inter)
      WEC_CALN  <-> WECC_AZ  : capital=300_000, annuity=14_532, loss=0.06 (costlier inter)

    Demand (load_curves_nrel_efs):
      WEC_BANC : total =  50 MWh in year 2019
      WEC_CALN : total = 100 MWh in year 2019
      WECC_AZ  : total =  75 MWh in year 2019
    """
    import sqlite3

    tmp_dir = tmp_path_factory.mktemp("tx_cost_test")
    db_path = str(tmp_dir / "test_network_upgrade.sqlite3")
    con = sqlite3.connect(db_path)

    # ---- transmission constraints (capacity) ----
    pd.DataFrame(
        {
            "region_from": [
                "WEC_BANC",
                "WECC_AZ",
                "WEC_CALN",
                "WECC_AZ",
                "WEC_BANC",
                "WEC_CALN",
            ],
            "region_to": [
                "WECC_AZ",
                "WEC_BANC",
                "WECC_AZ",
                "WEC_CALN",
                "WEC_CALN",
                "WEC_BANC",
            ],
            "firm_ttc_mw": [500.0, 500.0, 400.0, 400.0, 2750.0, 2750.0],
        }
    ).to_sql("transmission_single_epaipm", con, index=False, if_exists="replace")

    # ---- transmission cost table (symmetric A->B and B->A entries) ----
    pd.DataFrame(
        [
            ("WEC_BANC", "WEC_CALN", 100_000.0, 4_844.0, 0.02, 2018),
            ("WEC_CALN", "WEC_BANC", 100_000.0, 4_844.0, 0.02, 2018),
            ("WEC_BANC", "WECC_AZ", 250_000.0, 12_110.0, 0.05, 2018),
            ("WECC_AZ", "WEC_BANC", 250_000.0, 12_110.0, 0.05, 2018),
            ("WEC_CALN", "WECC_AZ", 300_000.0, 14_532.0, 0.06, 2018),
            ("WECC_AZ", "WEC_CALN", 300_000.0, 14_532.0, 0.06, 2018),
        ],
        columns=[
            "region_from",
            "region_to",
            "capital_cost_mw",
            "annum_cost_mw",
            "line_loss_frac",
            "dollar_year",
        ],
    ).to_sql(
        "transmission_cost_nrel_reeds", con, index=False, if_exists="replace"
    )

    # ---- demand table ----
    pd.DataFrame(
        [
            (1, "WEC_BANC", 50.0, 2019),
            (1, "WEC_CALN", 100.0, 2019),
            (1, "WECC_AZ", 75.0, 2019),
        ],
        columns=["time_index", "region", "load_mw", "year"],
    ).to_sql("load_curves_nrel_efs", con, index=False, if_exists="replace")

    con.close()

    if os.name == "nt":
        prefix = "sqlite:///"
    else:
        prefix = "sqlite:////"
    return sa.create_engine(prefix + db_path)


def test_calc_network_upgrade_costs(network_upgrade_engine):
    """calc_network_upgrade_costs returns correct total cost and losses.

    CA_N = [WEC_CALN, WEC_BANC] (aggregated)
    WECC_AZ = single region

    Expected totals for the CA_N <-> WECC_AZ pair
    -----------------------------------------------
    Direct connection  : WEC_BANC -> WECC_AZ  (cheapest at 250_000 $/MW capital)
    Intra-regional CA_N: single MST edge WEC_BANC <-> WEC_CALN (100_000 $/MW)
                         Demand weights: (50+100) / (50+100) = 1.0
                         Weighted cost  = 1.0 * 100_000 = 100_000
                         Weighted loss  = 1.0 * 0.02    = 0.02
    Intra-regional WECC_AZ: 0 (single-region)

    Total capital  = 250_000 + 100_000 + 0 = 350_000
    Total annuity  =  12_110 +   4_844 + 0 =  16_954
    Total loss     =    0.05 +    0.02 + 0 =    0.07
    """
    settings = {
        "model_regions": ["CA_N", "WECC_AZ"],
        "region_aggregations": {
            "CA_N": ["WEC_CALN", "WEC_BANC"],
        },
    }

    result = calc_network_upgrade_costs(network_upgrade_engine, settings)

    assert len(result) == 1, "Expected exactly one connected model region pair"

    row = result.iloc[0]
    assert set([row["start_region"], row["dest_region"]]) == {"CA_N", "WECC_AZ"}

    assert "total_interconnect_cost_mw" in result.columns
    assert result["total_interconnect_cost_mw"].iloc[0] == pytest.approx(350_000.0)

    assert "total_interconnect_annuity_mw" in result.columns
    assert result["total_interconnect_annuity_mw"].iloc[0] == pytest.approx(16_954.0)

    assert result["total_line_loss_frac"].iloc[0] == pytest.approx(0.07)

    assert "dollar_year" in result.columns
    assert result["dollar_year"].iloc[0] == 2018


def test_calc_network_upgrade_costs_no_aggregations(network_upgrade_engine):
    """When no region_aggregations are defined, intra-regional costs are zero."""
    settings = {
        "model_regions": ["WEC_BANC", "WECC_AZ"],
        # no region_aggregations
    }

    result = calc_network_upgrade_costs(network_upgrade_engine, settings)

    assert len(result) == 1
    row = result.iloc[0]
    assert set([row["start_region"], row["dest_region"]]) == {"WEC_BANC", "WECC_AZ"}

    # Only the direct connection cost; no intra-regional contribution
    assert result["total_interconnect_cost_mw"].iloc[0] == pytest.approx(250_000.0)
    assert result["total_line_loss_frac"].iloc[0] == pytest.approx(0.05)


def test_calc_network_upgrade_costs_missing_cost_column(tmp_path):
    """KeyError is raised when the cost table lacks required columns."""
    import sqlite3

    db_path = str(tmp_path / "bad_test.sqlite3")
    con = sqlite3.connect(db_path)
    pd.DataFrame(
        {
            "region_from": ["WEC_BANC"],
            "region_to": ["WECC_AZ"],
            # missing: line_loss_frac
        }
    ).to_sql("bad_cost_table", con, index=False)
    con.close()

    if os.name == "nt":
        prefix = "sqlite:///"
    else:
        prefix = "sqlite:////"
    bad_engine = sa.create_engine(prefix + db_path)

    settings = {"model_regions": ["WEC_BANC", "WECC_AZ"]}
    with pytest.raises(KeyError, match="line_loss_frac"):
        calc_network_upgrade_costs(
            bad_engine, settings, cost_table="bad_cost_table"
        )


