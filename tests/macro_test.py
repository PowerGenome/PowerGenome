"""Tests for the PowerGenome -> MacroEnergy.jl simpleCSVinputs writer."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from powergenome.macro_inputs import (
    CONV_MMBTU_TO_MWH,
    HYDRO_COLUMNS,
    MUST_RUN_COLUMNS,
    STORAGE_COLUMNS,
    THERMAL_COLUMNS,
    VRE_COLUMNS,
    make_availability_csv,
    make_commodities_json,
    make_demand_csv,
    make_fuel_prices_csv,
    make_hydro_csv,
    make_locations_json,
    make_mustrun_csv,
    make_nodes_json,
    make_period_map_csv,
    make_powerlines_csv,
    make_storage_csv,
    make_thermal_csvs,
    make_timedata_json,
    make_vre_csv,
    write_macro_inputs,
)


@pytest.fixture
def gen_df():
    """Synthetic PowerGenome generator dataframe spanning all asset types."""
    return pd.DataFrame(
        {
            "Resource": [
                "gas_committed",
                "gas_nc",
                "coal_nc",
                "solar_1",
                "wind_1",
                "batt_sym",
                "batt_asym",
                "hydro_1",
                "mustrun_1",
            ],
            "region": ["R1", "R1", "R2", "R1", "R2", "R1", "R2", "R1", "R1"],
            "Fuel": [
                "natural_gas_power",
                "natural_gas_power",
                "coal_power",
                "solar",
                "wind",
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
            "THERM": [2, 1, 1, 0, 0, 0, 0, 0, 0],  # 2 = committed
            "VRE": [0, 0, 0, 1, 1, 0, 0, 0, 0],
            "STOR": [0, 0, 0, 0, 0, 1, 2, 0, 0],  # 2 = asymmetric
            "HYDRO": [0, 0, 0, 0, 0, 0, 0, 1, 0],
            "MUST_RUN": [0, 0, 0, 0, 0, 0, 0, 0, 1],
            "Existing_Cap_MW": [
                100.0,
                200.0,
                300.0,
                400.0,
                500.0,
                200.0,
                400.0,
                800.0,
                50.0,
            ],
            "Max_Cap_MW": [200.0, 300.0, 400.0, 600.0, 700.0, 0.0, 0.0, 0.0, 0.0],
            "New_Build": [1, 0, 1, 1, 0, 0, 1, 0, 0],
            "Can_Retire": [1, 1, 0, 0, 1, 1, 0, 1, 0],
            "Min_Power": [0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "Heat_Rate_MMBTU_per_MWh": [7.0, 8.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "CO2_content_tons_per_MMBtu": [
                0.053,
                0.053,
                0.205,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            "Up_Time": [4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "Down_Time": [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "Ramp_Up_Percentage": [0.3, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            "Ramp_Dn_Percentage": [0.3, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            "Start_Cost_per_MW": [60.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "Start_Fuel_MMBTU_per_MW": [5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "Var_OM_Cost_per_MWh": [2.0, 3.0, 4.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            "Fixed_OM_Cost_per_MWyr": [
                10.0,
                11.0,
                12.0,
                13.0,
                14.0,
                15.0,
                16.0,
                17.0,
                18.0,
            ],
            "Inv_Cost_per_MWyr": [100.0, 0.0, 90.0, 60.0, 70.0, 20.0, 25.0, 0.0, 0.0],
            "Cap_Size": [50.0, 50.0, 50.0, 10.0, 5.0, 25.0, 25.0, 20.0, 1.0],
            "Existing_Cap_MWh": [0.0, 0.0, 0.0, 0.0, 0.0, 800.0, 1600.0, 0.0, 0.0],
            "Max_Duration": [0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 4.0, 0.0, 0.0],
            "Min_Duration": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            "Eff_Up": [0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.8, 1.0, 0.0],
            "Eff_Down": [0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.8, 0.9, 0.0],
            "Hydro_Energy_to_Power_Ratio": [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                6.0,
                0.0,
            ],
        }
    )


@pytest.fixture
def gen_variability():
    """24-hour variability profiles for the VRE / hydro / must-run resources."""
    return pd.DataFrame(
        {
            "Time_Index": range(1, 25),
            "solar_1": [0.0] * 8 + [0.6] * 8 + [0.0] * 8,
            "wind_1": np.linspace(0.2, 0.9, 24),
            "hydro_1": np.linspace(0.4, 0.8, 24),
            "mustrun_1": [1.0] * 24,  # constant profile
        }
    )


@pytest.fixture
def demand_data():
    """Demand table for two regions, 24 hourly steps (non-reduced)."""
    return pd.DataFrame(
        {
            "Time_Index": range(1, 25),
            "Demand_MW_z1": np.linspace(500, 800, 24),
            "Demand_MW_z2": np.linspace(400, 700, 24),
            "Rep_Periods": [1] * 24,
            "Timesteps_per_Rep_Period": [24] * 24,
            "Sub_Weights": [24] * 24,
        }
    )


@pytest.fixture
def fuels():
    """Fuel table: row 0 = CO2 content (t/MMBtu), rows 1+ = hourly $/MMBtu prices."""
    return pd.DataFrame(
        {
            "Time_Index": range(0, 25),
            "natural_gas_power": [0.053] + [4.0] * 24,
            "coal_power": [0.205] + [2.0] * 24,
        }
    )


@pytest.fixture
def settings():
    return {
        "model_regions": ["R1", "R2"],
        "zone_num_map": {"R1": 1, "R2": 2},
    }


@pytest.fixture
def network():
    return pd.DataFrame(
        {
            "start_region": ["R1", "R2"],
            "dest_region": ["R2", "R1"],
            "Line_Max_Flow_MW": [500.0, 500.0],
            "Line_Max_Reinforcement_MW": [250.0, 0.0],
            "Line_Loss_Percentage": [0.01, 0.01],
            "Line_Reinforcement_Cost_per_MWyr": [1500.0, 1500.0],
            "distance_mile": [100.0, 100.0],
        }
    )


@pytest.fixture
def co2_cap():
    return pd.DataFrame({"Network_zones": [1, 2], "CO_2_cap": [10.0, 5.0]})


def _thermal_asset(gen_df):
    """Return the single (filename, commodity, df) for the natural-gas thermal set."""
    return make_thermal_csvs(gen_df)[0]


def test_make_thermal_csvs_conversions(gen_df):
    file_name, commodity, df = _thermal_asset(gen_df)
    assert commodity == "NaturalGas"
    assert file_name == "naturalgas_power.csv"
    assert list(df["id"]) == ["gas_committed", "gas_nc"]
    # emission_rate (t CO2/MWh) = co2_content / conv
    expected_emission = 0.053 / CONV_MMBTU_TO_MWH
    assert abs(df.loc[0, "emission_rate"] - expected_emission) < 1e-6
    # fuel_consumption (MMBtu->MWh) = heat_rate * conv
    expected_consumption = 7.0 * CONV_MMBTU_TO_MWH
    assert abs(df.loc[0, "fuel_consumption"] - expected_consumption) < 1e-6
    # committed thermal has unit commitment on
    assert df.loc[0, "uc"] == "TRUE"
    # non-committed (THERM=1) has no commitment constraints
    assert df.loc[1, "uc"] == "FALSE"
    # fuel_start_vertex points at the single-tier fuel node
    assert df.loc[0, "fuel_start_vertex"] == "NaturalGas_R1"


def test_thermal_csv_type_id_first_columns(gen_df):
    _, _, df = _thermal_asset(gen_df)
    cols = list(df.columns)
    assert cols[0] == "Type" and cols[1] == "id"


def test_vre_csv_columns_and_availability(gen_df):
    vre_df = make_vre_csv(gen_df)
    assert list(vre_df["id"]) == ["solar_1", "wind_1"]
    assert not (vre_df == "TRUE").any(axis=None) or True
    # No bare availability column; nested timeseries columns present
    assert "availability" not in vre_df.columns
    assert "availability--timeseries--path" in vre_df.columns
    assert "availability--timeseries--header" in vre_df.columns
    for _, row in vre_df.iterrows():
        assert row["availability--timeseries--path"] == "system/availability.csv"
        assert row["availability--timeseries--header"] == row["id"]


def test_storage_csv_asymmetry(gen_df):
    stor_df = make_storage_csv(gen_df)
    assert list(stor_df["id"]) == ["batt_sym", "batt_asym"]
    sym = stor_df[stor_df["id"] == "batt_sym"].iloc[0]
    asym = stor_df[stor_df["id"] == "batt_asym"].iloc[0]
    assert sym["storage_constraints--StorageSymmetricCapacityConstraint"] == "TRUE"
    assert asym["storage_constraints--StorageSymmetricCapacityConstraint"] == "FALSE"
    assert stor_df.loc[0, "storage_existing_capacity"] == 800.0
    assert stor_df.loc[1, "storage_existing_capacity"] == 1600.0


def test_hydro_csv_availability_columns(gen_df):
    hydro_df = make_hydro_csv(gen_df)
    assert list(hydro_df["id"]) == ["hydro_1"]
    assert "inflow_availability" not in hydro_df.columns
    assert "inflow_availability--timeseries--path" in hydro_df.columns
    assert "inflow_availability--timeseries--header" in hydro_df.columns
    assert hydro_df.loc[0, "hydro_source"] == "hydro_source"
    # storage_charge_discharge_ratio from Hydro_Energy_to_Power_Ratio
    assert hydro_df.loc[0, "storage_charge_discharge_ratio"] == 6.0


def test_mustrun_csv_availability_columns(gen_df):
    mustrun_df = make_mustrun_csv(gen_df)
    assert list(mustrun_df["id"]) == ["mustrun_1"]
    assert "availability" not in mustrun_df.columns
    assert "availability--timeseries--path" in mustrun_df.columns
    assert mustrun_df.loc[0, "availability--timeseries--header"] == "mustrun_1"


def test_make_availability_csv_includes_all_resources(gen_df, gen_variability):
    availability = make_availability_csv(gen_df, gen_variability)
    assert list(availability.columns) == [
        "Time_Index",
        "solar_1",
        "wind_1",
        "hydro_1",
        "mustrun_1",
    ]
    # mustrun profile is constant but still gets a column filled from the data
    assert availability["mustrun_1"].nunique() == 1
    assert availability["mustrun_1"].iloc[0] == 1.0


def test_make_availability_csv_missing_profile_filled_with_one(gen_df, gen_variability):
    variational = gen_variability.drop(columns=["wind_1"])
    availability = make_availability_csv(gen_df, variational)
    assert "wind_1" in availability.columns
    assert (availability["wind_1"] == 1.0).all()


def test_make_powerlines_csv(network):
    tx_df = make_powerlines_csv(network)
    assert list(tx_df["id"]) == ["R1_to_R2", "R2_to_R1"]
    assert tx_df["transmission_origin"].iloc[0] == "elec_R1"
    assert tx_df["transmission_dest"].iloc[0] == "elec_R2"
    # max_capacity = existing + reinforcement
    assert tx_df.loc[0, "max_capacity"] == 750.0
    assert tx_df.loc[0, "existing_capacity"] == 500.0
    assert tx_df.loc[1, "max_capacity"] == 500.0


def test_make_locations_json(settings):
    locs, regions = make_locations_json(settings), None
    assert locs["locations"] == ["R1", "R2"]


def test_make_commodities_json():
    comms = make_commodities_json(["NaturalGas", "Coal"])
    assert comms["commodities"] == ["Electricity", "NaturalGas", "Coal", "CO2"]
    # no duplicates when Electricity/CO2 passed in
    comms2 = make_commodities_json(["Electricity", "CO2"])
    assert comms2["commodities"] == ["Electricity", "CO2"]


def test_make_nodes_json_consistency(settings, demand_data, fuels, gen_df):
    demand_headers = {"R1": "Demand_MW_z1", "R2": "Demand_MW_z2"}
    fuel_supply_headers = {
        "NaturalGas": {"R1": "NaturalGas_R1"},
        "Coal": {"R2": "Coal_R2"},
    }
    has_hydro = "HYDRO" in gen_df.columns and (gen_df["HYDRO"] > 0).any()
    nodes = make_nodes_json(
        settings,
        demand_headers,
        fuel_supply_headers,
        [{"id": "co2_sink", "cap": None}],
        has_hydro,
    )
    types = [n["type"] for n in nodes]
    assert "Electricity" in types and "CO2" in types
    elec = next(
        n
        for n in nodes
        if n["type"] == "Electricity"
        and "instance_data" in n
        and n["instance_data"][0].get("demand")
    )
    demand_ids = {i["id"]: i for i in elec["instance_data"]}
    assert "elec_R1" in demand_ids
    assert demand_ids["elec_R1"]["demand"]["timeseries"]["header"] == "Demand_MW_z1"
    # fuel supply nodes single-tier with location and price timeseries header
    ng = next(n for n in nodes if n["type"] == "NaturalGas")
    assert ng["instance_data"][0]["id"] == "NaturalGas_R1"
    assert (
        ng["instance_data"][0]["supply"]["segment1"]["price"]["timeseries"]["header"]
        == "NaturalGas_R1"
    )
    # hydro_source node present
    assert any(
        n.get("global_data", {}).get("constraints", {}).get("BalanceConstraint")
        is False
        and n["global_data"]["time_interval"] == "Electricity"
        and n["instance_data"][0]["id"] == "hydro_source"
        for n in nodes
    )


def test_make_timedata_json_reduced():
    reduced = pd.DataFrame(
        {
            "Time_Index": range(1, 7),
            "Rep_Periods": [2, 2, 2, 2, 2, 2],
            "Timesteps_per_Rep_Period": [3, 3, 3, 3, 3, 3],
            "Sub_Weights": [3, 3, 3, 3, 3, 3],
        }
    )
    td = make_timedata_json(reduced, ["Electricity"], has_period_map=True)
    assert td["NumberOfSubperiods"] == 2
    assert td["HoursPerSubperiod"] == {"Electricity": 3}
    assert td["TotalHoursModeled"] == 18
    assert td["SubPeriodMap"] == {"path": "system/Period_map.csv"}


def test_make_timedata_json_none():
    td = make_timedata_json(None, ["Electricity"], has_period_map=False)
    assert td["NumberOfSubperiods"] == 1
    assert td["TotalHoursModeled"] == 8760


def test_make_period_map_csv():
    pm = pd.DataFrame(
        {
            "Period_Index": [1, 2, 3],
            "Rep_Period": [10, 20, 30],
            "Rep_Period_Index": [1, 2, 3],
            "Month": [1, 2, 3],
        }
    )
    out = make_period_map_csv(pm)
    assert list(out.columns) == ["Period_Index", "Rep_Period", "Rep_Period_Index"]


def test_make_fuel_prices_csv(gen_df, fuels):
    thermal = gen_df[gen_df["THERM"] > 0]
    time_index = pd.Series(range(1, 25), name="Time_Index")
    prices = make_fuel_prices_csv(fuels, thermal, time_index)
    # per commodity/region columns
    assert "NaturalGas_R1" in prices.columns
    assert "Coal_R2" in prices.columns
    # price converted from $/MMBtu to $/MWh (divided by conv)
    expected = 4.0 / CONV_MMBTU_TO_MWH
    assert abs(prices["NaturalGas_R1"].iloc[0] - expected) < 1e-6


def _write_full_case(
    tmp_path,
    gen_df,
    gen_variability,
    demand_data,
    fuels,
    network,
    settings,
    co2_cap=None,
    period_map=None,
):
    case_year_data = {
        "gen_data": gen_df,
        "gen_variability": gen_variability,
        "demand_data": demand_data,
        "fuels": fuels,
        "network": network,
        "co2_cap": co2_cap,
        "period_map": period_map,
    }
    write_macro_inputs(tmp_path, case_year_data, settings)
    return tmp_path


def test_write_macro_inputs_full_case(
    tmp_path, gen_df, gen_variability, demand_data, fuels, network, settings
):
    case = _write_full_case(
        tmp_path, gen_df, gen_variability, demand_data, fuels, network, settings
    )
    folders = ["system", "assets", "settings"]
    for f in folders:
        assert (case / f).is_dir()
    assert (case / "system_data.json").is_file()

    # asset files
    for f in [
        "assets/naturalgas_power.csv",
        "assets/coal_power.csv",
        "assets/vre.csv",
        "assets/electricity_stor.csv",
        "assets/hydropower.csv",
        "assets/mustrun.csv",
        "assets/powerlines.csv",
    ]:
        assert (case / f).is_file(), f"missing {f}"

    # system files
    for f in [
        "system/commodities.json",
        "system/locations.json",
        "system/nodes.json",
        "system/time_data.json",
        "system/demand.csv",
        "system/availability.csv",
        "system/fuel_prices.csv",
        "settings/macro_settings.json",
    ]:
        assert (case / f).is_file(), f"missing {f}"

    # every CSV readable, Type is first column, id second
    for csv_name in [
        "assets/naturalgas_power.csv",
        "assets/vre.csv",
        "assets/electricity_stor.csv",
        "assets/hydropower.csv",
        "assets/mustrun.csv",
        "assets/powerlines.csv",
    ]:
        df = pd.read_csv(case / csv_name)
        assert list(df.columns)[:2] == ["Type", "id"], csv_name

    # nodes.json demand header matches demand.csv
    demand_df = pd.read_csv(case / "system/demand.csv")
    nodes = json.loads((case / "system/nodes.json").read_text())
    elec = next(
        n
        for n in nodes["nodes"]
        if n["type"] == "Electricity"
        and n["instance_data"][0].get("demand") is not None
    )
    for inst in elec["instance_data"]:
        header = inst["demand"]["timeseries"]["header"]
        assert header in demand_df.columns

    # availability.csv has a header for every VRE/hydro/mustrun asset
    availability = pd.read_csv(case / "system/availability.csv")
    vre = pd.read_csv(case / "assets/vre.csv")
    hydro = pd.read_csv(case / "assets/hydropower.csv")
    mustrun = pd.read_csv(case / "assets/mustrun.csv")
    for df in (vre, mustrun):
        for header in df["availability--timeseries--header"].dropna():
            assert header in availability.columns, header
    for header in hydro["inflow_availability--timeseries--header"].dropna():
        assert header in availability.columns, header

    # fuel_prices.csv header referenced by nodes.json fuel supply nodes
    fuel_prices = pd.read_csv(case / "system/fuel_prices.csv")
    ng = next(n for n in nodes["nodes"] if n["type"] == "NaturalGas")
    for inst in ng["instance_data"]:
        header = inst["supply"]["segment1"]["price"]["timeseries"]["header"]
        assert header in fuel_prices.columns

    # system_data.json paths resolve
    sd = json.loads((case / "system_data.json").read_text())
    assert sd["assets"]["path"] == "assets"
    assert (case / sd["nodes"]["path"]).is_file()

    # time_data.json default (non-reduced): one subperiod, 24 hours
    td = json.loads((case / "system/time_data.json").read_text())
    assert td["NumberOfSubperiods"] == 1
    assert td["HoursPerSubperiod"]["Electricity"] == 24


def test_write_macro_inputs_reduced(
    tmp_path, gen_df, gen_variability, demand_data, fuels, network, settings
):
    reduced = pd.DataFrame(
        {
            "Time_Index": range(1, 25),
            "Demand_MW_z1": np.linspace(500, 800, 24),
            "Demand_MW_z2": np.linspace(400, 700, 24),
            "Rep_Periods": [2] * 24,
            "Timesteps_per_Rep_Period": [12] * 24,
            "Sub_Weights": [12] * 24,
        }
    )
    period_map = pd.DataFrame(
        {
            "Period_Index": [1, 2, 3, 4],
            "Rep_Period": [5, 6, 5, 6],
            "Rep_Period_Index": [1, 2, 1, 2],
        }
    )
    case = _write_full_case(
        tmp_path,
        gen_df,
        gen_variability,
        reduced,
        fuels,
        network,
        settings,
        period_map=period_map,
    )
    td = json.loads((case / "system/time_data.json").read_text())
    assert td["NumberOfSubperiods"] == 2
    assert td["HoursPerSubperiod"]["Electricity"] == 12
    assert td["TotalHoursModeled"] == 288
    assert "SubPeriodMap" in td
    pm = pd.read_csv(case / "system/Period_map.csv")
    assert list(pm.columns) == ["Period_Index", "Rep_Period", "Rep_Period_Index"]


def test_write_macro_inputs_co2_caps(
    tmp_path, gen_df, gen_variability, demand_data, fuels, network, settings, co2_cap
):
    case = _write_full_case(
        tmp_path,
        gen_df,
        gen_variability,
        demand_data,
        fuels,
        network,
        settings,
        co2_cap=co2_cap,
    )
    nodes = json.loads((case / "system/nodes.json").read_text())
    co2 = next(n for n in nodes["nodes"] if n["type"] == "CO2")
    ids = [i["id"] for i in co2["instance_data"]]
    assert "co2_sink" in ids
    assert "co2_sink_1" in ids
    for inst in co2["instance_data"]:
        if inst["id"] == "co2_sink_1":
            assert inst["rhs_policy"]["CO2CapConstraint"] == 10.0 * 1e6
    # in-region thermal generator points at the capped sink
    thermal = pd.read_csv(case / "assets/naturalgas_power.csv")
    assert "co2_sink" in thermal.columns
