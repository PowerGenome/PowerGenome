import pandas as pd
from src.util import init_pudl_connection
from src.params import DATA_PATHS
import sqlite3


def create_testing_db():
    pudl_engine, pudl_out = init_pudl_connection()
    test_conn = sqlite3.connect(DATA_PATHS["test_data"] / 'test_data.db')

    sql = """
        SELECT *
        FROM boiler_generator_assn_eia860
        WHERE plant_id_eia IN (116, 151, 149, 34, 113, 117)
    """
    bga = pd.read_sql_query(sql, pudl_engine)
    bga.to_sql("boiler_generator_assn_eia860", test_conn, index=False)

    sql = """
        SELECT *
        FROM generation_fuel_eia923
        WHERE plant_id_eia IN (116, 151, 149, 34, 113, 117)
    """
    gen_fuel = pd.read_sql_query(sql, pudl_engine)
    gen_fuel.to_sql("generation_fuel_eia923", test_conn, index=False)

    sql = """
        SELECT *
        FROM generators_eia860
        WHERE plant_id_eia IN (116, 151, 149, 34, 113, 117)
    """
    gen860 = pd.read_sql_query(sql, pudl_engine)
    gen860.to_sql("generators_eia860", test_conn, index=False)

    sql = """
        SELECT *
        FROM generators_entity_eia
        WHERE plant_id_eia IN (116, 151, 149, 34, 113, 117)
    """
    gen_entity = pd.read_sql_query(sql, pudl_engine)
    gen_entity.to_sql("generators_entity_eia", test_conn, index=False)

    hr_by_unit = pudl_out.hr_by_unit().query("plant_id_eia.isin([116, 151, 149, 34, 113, 117])")
    hr_by_unit.to_sql("hr_by_unit", test_conn, index=False)

    sql = """
        SELECT *
        FROM plant_region_map_ipm
        WHERE plant_id_eia IN (116, 151, 149, 34, 113, 117)
    """
    plant_region = pd.read_sql_query(sql, pudl_engine)
    plant_region.to_sql("plant_region_map_ipm", test_conn, index=False)


if __name__ == "__main__":
    create_testing_db()
