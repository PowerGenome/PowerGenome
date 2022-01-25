import sqlite3

import pandas as pd
from powergenome.params import DATA_PATHS
from powergenome.util import init_pudl_connection


def create_testing_db():
    pudl_engine, pudl_out, pg_engine = init_pudl_connection(
        start_year=2018, end_year=2020
    )
    pudl_test_conn = sqlite3.connect(DATA_PATHS["test_data"] / "pudl_test_data.db")

    sql = """
        SELECT *
        FROM boiler_generator_assn_eia860
        WHERE plant_id_eia IN (116, 151, 149, 34, 113, 117, 141, 160)
    """
    bga = pd.read_sql_query(sql, pudl_engine, parse_dates="report_date")
    bga = bga.query("report_date.dt.year >= 2018")
    bga.to_sql(
        "boiler_generator_assn_eia860", pudl_test_conn, index=False, if_exists="replace"
    )

    sql = """
        SELECT *
        FROM generation_fuel_eia923
        WHERE plant_id_eia IN (116, 151, 149, 34, 113, 117, 141, 160)
    """
    gen_fuel = pd.read_sql_query(sql, pudl_engine, parse_dates="report_date")
    gen_fuel = gen_fuel.query("report_date.dt.year >= 2018")
    gen_fuel.to_sql(
        "generation_fuel_eia923", pudl_test_conn, index=False, if_exists="replace"
    )

    sql = """
        SELECT *
        FROM generators_eia860
        WHERE plant_id_eia IN (116, 151, 149, 34, 113, 117, 141, 160)
    """
    gen860 = pd.read_sql_query(sql, pudl_engine, parse_dates="report_date")
    gen860 = gen860.query("report_date.dt.year >= 2018")
    gen860.to_sql("generators_eia860", pudl_test_conn, index=False, if_exists="replace")

    sql = """
        SELECT *
        FROM generators_entity_eia
        WHERE plant_id_eia IN (116, 151, 149, 34, 113, 117, 141, 160)
    """
    gen_entity = pd.read_sql_query(sql, pudl_engine)
    gen_entity.to_sql(
        "generators_entity_eia", pudl_test_conn, index=False, if_exists="replace"
    )

    hr_by_unit = pudl_out.hr_by_unit().query(
        "plant_id_eia.isin([116, 151, 149, 34, 113, 117, 141, 160])"
    )
    hr_by_unit.to_sql("hr_by_unit", pudl_test_conn, index=False, if_exists="replace")

    sql = """
        SELECT *
        FROM plant_region_map_epaipm
        WHERE plant_id_eia IN (116, 151, 149, 34, 113, 117, 141, 160)
    """
    plant_region = pd.read_sql_query(sql, pg_engine)
    plant_region.to_sql(
        "plant_region_map_epaipm", pudl_test_conn, index=False, if_exists="replace"
    )


if __name__ == "__main__":
    create_testing_db()
