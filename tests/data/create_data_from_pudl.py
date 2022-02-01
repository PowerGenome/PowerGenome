import sqlite3
import sqlalchemy as sa

import pandas as pd
from powergenome.params import DATA_PATHS
from powergenome.util import init_pudl_connection

GENS860_COLS = [
    "report_date",
    "plant_id_eia",
    "generator_id",
    # "associated_combined_heat_power",
    # "balancing_authority_code_eia",
    # "bypass_heat_recovery",
    "capacity_mw",
    # "county",
    "current_planned_operating_date",
    "energy_source_code_1",
    # "ferc_cogen_status",
    # "iso_rto_code",
    # "latitude",
    # "longitude",
    "minimum_load_mw",
    # "operating_date",
    "operational_status_code",
    # "original_planned_operating_date",
    # "state",
    "summer_capacity_mw",
    "technology_description",
    # "unit_id_pudl",
    "winter_capacity_mw",
    "fuel_type_code_pudl",
    # "zip_code",
    "planned_retirement_date",
    "time_cold_shutdown_full_load_code",
    "switch_oil_gas",
    "planned_new_capacity_mw",
    "energy_source_code_2",
    "region",
]
GEN_FUEL_COLS = [
    "report_date",
    "plant_id_eia",
    "energy_source_code",
    "fuel_consumed_for_electricity_mmbtu",
    "fuel_consumed_for_electricity_units",
    "fuel_consumed_mmbtu",
    "fuel_consumed_units",
    "fuel_mmbtu_per_unit",
    "net_generation_mwh",
    "prime_mover_code",
    "fuel_type_code_pudl",
]
ENTITY_COLS = ["plant_id_eia", "generator_id", "prime_mover_code", "operating_date"]


def create_testing_db():
    pudl_engine, pudl_out, pg_engine = init_pudl_connection(
        start_year=2018, end_year=2020
    )
    pudl_test_conn = sqlite3.connect(DATA_PATHS["test_data"] / "pudl_test_data.db")

    plant_region = pd.read_sql_table("plant_region_map_epaipm", pg_engine)
    # gens_860 = pudl_out.gens_eia860()
    s = "SELECT * from generators_eia860 where strftime('%Y',report_date)='2020'"
    gens_860 = pd.read_sql_query(s, pudl_engine, parse_dates=["report_date"])
    # gens_860 = gens_860.loc[gens_860.report_date.dt.year == 2020, :]
    gens_860 = pd.merge(gens_860, plant_region, on="plant_id_eia", how="inner")
    gens_860 = gens_860.loc[:, GENS860_COLS]
    gens_860 = gens_860.groupby(
        ["region", "technology_description"], as_index=False
    ).head(10)
    gens_860 = gens_860.drop(columns="region")
    eia_plant_ids = gens_860["plant_id_eia"].unique()

    gen_entity = pd.read_sql_table("generators_entity_eia", pudl_engine)
    gen_entity = gen_entity.loc[
        gen_entity["plant_id_eia"].isin(eia_plant_ids), ENTITY_COLS
    ]

    bga = pudl_out.bga_eia860()
    bga = bga.loc[
        (bga.report_date.dt.year == 2020) & (bga.plant_id_eia.isin(eia_plant_ids)), :
    ]

    s = "SELECT * from generation_fuel_eia923 where strftime('%Y',report_date)='2020'"
    gen_fuel = pd.read_sql_query(s, pudl_engine, parse_dates=["report_date"])
    gen_fuel = gen_fuel.loc[
        gen_fuel.plant_id_eia.isin(eia_plant_ids),
        GEN_FUEL_COLS,
    ]
    s = "SELECT * from generation_fuel_nuclear_eia923 where strftime('%Y',report_date)='2020'"
    gen_fuel_nuc = pd.read_sql_query(s, pudl_engine, parse_dates=["report_date"])
    gen_fuel_nuc = gen_fuel_nuc.loc[
        gen_fuel_nuc.plant_id_eia.isin(eia_plant_ids),
        GEN_FUEL_COLS,
    ]
    # gen_fuel = pd.concat([gen_fuel, gen_fuel_nuc], ignore_index=True)

    s = "SELECT * from generation_eia923 where strftime('%Y',report_date)='2020'"
    gen_923 = pd.read_sql_query(s, pudl_engine, parse_dates=["report_date"])
    gen_923 = gen_923.loc[
        gen_923.plant_id_eia.isin(eia_plant_ids),
        :,
    ]

    s = "SELECT * from boiler_fuel_eia923 where strftime('%Y',report_date)='2020'"
    boiler_fuel = pd.read_sql_query(s, pudl_engine, parse_dates=["report_date"])
    boiler_fuel = boiler_fuel.loc[
        boiler_fuel.plant_id_eia.isin(eia_plant_ids),
        :,
    ]

    plant_entity = pd.read_sql_table("plants_entity_eia", pudl_engine)
    plant_entity = plant_entity.loc[plant_entity["plant_id_eia"].isin(eia_plant_ids), :]
    plants_eia_860 = pd.read_sql_table("plants_eia860", pudl_engine)
    plants_eia_860 = plants_eia_860.loc[
        plants_eia_860["plant_id_eia"].isin(eia_plant_ids), :
    ]

    plants_eia = pd.read_sql_table("plants_eia", pudl_engine)
    plants_eia = plants_eia.loc[plants_eia["plant_id_eia"].isin(eia_plant_ids), :]

    utilities_eia = pd.read_sql_table("utilities_eia", pudl_engine)
    utilities_entity = pd.read_sql_table("utilities_entity_eia", pudl_engine)
    utilities_860 = pd.read_sql_table("utilities_eia860", pudl_engine)

    plant_region.to_sql(
        "plant_region_map_epaipm", pudl_test_conn, index=False, if_exists="replace"
    )
    gens_860.to_sql(
        "generators_eia860", pudl_test_conn, index=False, if_exists="replace"
    )
    gen_entity.to_sql(
        "generators_entity_eia", pudl_test_conn, index=False, if_exists="replace"
    )
    bga.to_sql(
        "boiler_generator_assn_eia860", pudl_test_conn, index=False, if_exists="replace"
    )
    gen_fuel.to_sql(
        "generation_fuel_eia923", pudl_test_conn, index=False, if_exists="replace"
    )
    gen_fuel_nuc.to_sql(
        "generation_fuel_nuclear_eia923",
        pudl_test_conn,
        index=False,
        if_exists="replace",
    )
    gen_923.to_sql(
        "generation_eia923", pudl_test_conn, index=False, if_exists="replace"
    )
    boiler_fuel.to_sql(
        "boiler_fuel_eia923", pudl_test_conn, index=False, if_exists="replace"
    )
    plant_entity.to_sql(
        "plants_entity_eia", pudl_test_conn, index=False, if_exists="replace"
    )
    plants_eia_860.to_sql(
        "plants_eia860", pudl_test_conn, index=False, if_exists="replace"
    )
    plants_eia.to_sql("plants_eia", pudl_test_conn, index=False, if_exists="replace")
    utilities_eia.to_sql(
        "utilities_eia", pudl_test_conn, index=False, if_exists="replace"
    )
    utilities_entity.to_sql(
        "utilities_entity_eia", pudl_test_conn, index=False, if_exists="replace"
    )
    utilities_860.to_sql(
        "utilities_eia860", pudl_test_conn, index=False, if_exists="replace"
    )


if __name__ == "__main__":
    create_testing_db()
