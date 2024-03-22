"""
Create a slimmed-down version of the pg_misc database for testing.
"""

import sqlite3

import pandas as pd

from powergenome.params import DATA_PATHS
from powergenome.util import init_pudl_connection


def create_testing_db():
    pudl_engine, pudl_out, pg_engine = init_pudl_connection(
        start_year=2018, end_year=2020
    )

    pg_test_conn = sqlite3.connect(DATA_PATHS["test_data"] / "pg_misc_tables.sqlite3")

    # Read tables. Filter data where necessary
    s = """
        SELECT time_index, region_id_epaipm, load_mw, year
        FROM load_curves_ferc
        WHERE region_id_epaipm in ('WECC_AZ', 'WEC_CALN', 'WEC_BANC', 'WEC_LADW')
    """
    load_curves_ferc = pd.read_sql_query(s, pg_engine)

    s = """
        SELECT time_index, region, sector, subsector, load_mw, year
        FROM load_curves_nrel_efs
        WHERE region in ('WECC_AZ', 'WEC_CALN', 'WEC_BANC', 'WEC_LADW')
    """
    load_curves_efs = pd.read_sql_query(s, pg_engine)

    s = """
        SELECT *
        FROM offshore_spur_costs_nrelatb
        WHERE atb_year = 2022
        AND basis_year >= 2020
    """
    offshore_spur = pd.read_sql_query(s, pg_engine)

    s = """
        SELECT *
        FROM technology_costs_nrelatb
        WHERE atb_year = 2022
        AND parameter in ('capex_mw', 'capex_mwh', 'variable_o_m_mwh', 'fixed_o_m_mw', 'fixed_o_m_mwh', 'wacc_real')
    """
    tech_costs = pd.read_sql_query(s, pg_engine)

    s = """
        SELECT *
        FROM technology_heat_rates_nrelatb
        WHERE atb_year = 2022
    """
    tech_hr = pd.read_sql_query(s, pg_engine)

    plant_region_map = pd.read_sql_table("plant_region_map_epaipm", pg_engine)
    regions = pd.read_sql_table("regions_entity_epaipm", pg_engine)
    transmission = pd.read_sql_table("transmission_single_epaipm", pg_engine).drop(
        columns=["id", "tariff_mills_kwh"]
    )

    # Write tables to test db
    load_curves_ferc.to_sql(
        "load_curves_ferc", pg_test_conn, if_exists="replace", index=False
    )
    load_curves_efs.to_sql(
        "load_curves_nrel_efs", pg_test_conn, if_exists="replace", index=False
    )
    offshore_spur.to_sql(
        "offshore_spur_costs_nrelatb", pg_test_conn, if_exists="replace", index=False
    )
    tech_costs.to_sql(
        "technology_costs_nrelatb", pg_test_conn, if_exists="replace", index=False
    )
    tech_hr.to_sql(
        "technology_heat_rates_nrelatb", pg_test_conn, if_exists="replace", index=False
    )
    plant_region_map.to_sql(
        "plant_region_map_epaipm", pg_test_conn, if_exists="replace", index=False
    )
    regions.to_sql(
        "regions_entity_epaipm", pg_test_conn, if_exists="replace", index=False
    )
    transmission.to_sql(
        "transmission_single_epaipm", pg_test_conn, if_exists="replace", index=False
    )


if __name__ == "__main__":
    create_testing_db()
