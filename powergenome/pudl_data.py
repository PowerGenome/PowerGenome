"Download and process PUDL data."

import logging
from pathlib import Path
from typing import List

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)


def download_pudl_generator_data(report_years: List[int], save_path: str) -> None:
    """Download and process PUDL generator data for specified report years, and save the
    results to a parquet file.

    This function retrieves generator data from pre-processed PUDL datasets hosted on an
    S3 bucket. It processes the data to include heat rates for generators, filling in
    missing values where possible by joining data from multiple sources. The final
    processed data is saved as a parquet file.

    Parameters
    ----------
    report_years : List[int]
        A list of years for which generator data should be downloaded and processed.
    save_path : str
        The directory path where the resulting parquet file will be saved.
    """
    con = duckdb.connect()

    # Check if all report_years exist in the data files
    available_years = con.execute(
        """
        SELECT DISTINCT EXTRACT(YEAR from report_date) as report_year
        FROM read_parquet('https://s3.us-west-2.amazonaws.com/pudl.catalyst.coop/nightly/out_eia__yearly_generators.parquet')
        """
    ).fetchall()
    available_years = {row[0] for row in available_years}

    missing_years = [year for year in report_years if year not in available_years]
    if set(missing_years) == set(report_years):
        raise ValueError(
            f"None of the requested EIA data years ({report_years}) are available in PUDL. "
            "Please check the available years in the dataset."
        )
    if missing_years:
        logger.warning(
            f"The following EIA data years are not available yet: {missing_years}. "
            "Please check the available years in the dataset."
        )

    # Download pre-processed annual data. Does not include heat rates for units without boilers.
    report_years_str = ", ".join(str(year) for year in report_years)
    con.execute(
        f"""
        CREATE OR REPLACE TABLE out_eia__yearly_generators AS
    SELECT
        plant_id_eia AS plant_id,
        generator_id,
        unit_id_pudl AS unit_id,
        energy_source_code_1 as primary_energy_source,
        prime_mover_code,
        EXTRACT(YEAR from report_date) as report_year,
        technology_description as technology,
        capacity_mw,
        summer_capacity_mw,
        winter_capacity_mw,
        energy_storage_capacity_mwh,
        minimum_load_mw,
        EXTRACT(YEAR from current_planned_generator_operating_date) as planned_generator_operating_year,
        EXTRACT(YEAR from planned_generator_retirement_date) as planned_generator_retirement_year,
        unit_heat_rate_mmbtu_per_mwh as heat_rate_mmbtu_per_mwh,
        fuel_cost_per_mmbtu,
        net_generation_mwh,
        operational_status_code,
        operational_status,
        total_mmbtu,
        data_maturity
    FROM read_parquet('https://s3.us-west-2.amazonaws.com/pudl.catalyst.coop/nightly/out_eia__yearly_generators.parquet')
    WHERE report_year in ({report_years_str})
    AND data_maturity = 'final'
    """
    )

    # Download fuel consumption and generation data to calculate heat rates.
    con.execute(
        f"""
        CREATE or REPLACE TABLE out_eia923__yearly_generation_fuel_by_generator AS
        SELECT
            plant_id_eia AS plant_id,
            generator_id,
            unit_id_pudl AS unit_id,
            EXTRACT(YEAR from report_date) AS report_year,
            fuel_consumed_for_electricity_mmbtu,
            net_generation_mwh,
            fuel_consumed_for_electricity_mmbtu / net_generation_mwh AS heat_rate_mmbtu_per_mwh
        FROM read_parquet('https://s3.us-west-2.amazonaws.com/pudl.catalyst.coop/nightly/out_eia923__yearly_generation_fuel_by_generator.parquet')
        WHERE report_year in ({report_years_str});

        """
        # GROUP BY plant_id_eia, generator_id, report_year;
    )

    # Join the two tables to get heat rates for all generators.
    con.execute(
        """
        CREATE OR REPLACE TABLE filled_generators AS
        SELECT
            g.plant_id,
            g.generator_id,
            g.unit_id,
            g.primary_energy_source,
            g.prime_mover_code,
            g.report_year,
            g.technology,
            g.capacity_mw,
            g.summer_capacity_mw,
            g.winter_capacity_mw,
            g.energy_storage_capacity_mwh,
            g.minimum_load_mw,
            g.planned_generator_retirement_year,
            g.planned_generator_operating_year,
            COALESCE(g.heat_rate_mmbtu_per_mwh, f.heat_rate_mmbtu_per_mwh) AS heat_rate_mmbtu_per_mwh,
            g.fuel_cost_per_mmbtu,
            COALESCE(g.net_generation_mwh, f.net_generation_mwh) AS net_generation_mwh,
            COALESCE(g.total_mmbtu, f.fuel_consumed_for_electricity_mmbtu) AS total_mmbtu,
            g.operational_status_code,
            g.operational_status,
        FROM out_eia__yearly_generators AS g
        LEFT JOIN out_eia923__yearly_generation_fuel_by_generator AS f
        ON g.plant_id = f.plant_id
        AND g.generator_id = f.generator_id
        AND g.unit_id = f.unit_id
        AND g.report_year = f.report_year;
        """
    )

    # Save the filled generators table to a parquet file.
    con.execute(
        f"""
        COPY (SELECT * FROM filled_generators LIMIT 100) TO '{Path(save_path) / "generator_data.parquet"}' (FORMAT PARQUET);
        """
    )
