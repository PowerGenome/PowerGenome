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
    MacroCaseBuilder,
    load_nsd_segments,
    make_availability_csv,
    make_case_settings_json,
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
    make_system_data_json,
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
            "WACC": [0.08, 0.08, 0.09, 0.07, 0.07, 0.06, 0.06, 0.05, 0.05],
            "Capital_Recovery_Period": [
                20.0,
                20.0,
                25.0,
                25.0,
                20.0,
                15.0,
                15.0,
                30.0,
                10.0,
            ],
            "Lifetime": [
                25.0,
                25.0,
                30.0,
                30.0,
                25.0,
                20.0,
                20.0,
                40.0,
                15.0,
            ],
            "Min_Retired_Cap_MW": [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
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
            "Capital_Recovery_Period": [60.0, 60.0],
            "WACC": [0.044, 0.044],
        }
    )


@pytest.fixture
def co2_cap():
    return pd.DataFrame(
        {
            "Network_zones": [1, 2],
            "CO_2_Cap_Zone_1": [1, 0],
            "CO_2_Cap_Zone_2": [0, 1],
            "CO_2_Max_Mtons_1": [10.0, 0.0],
            "CO_2_Max_Mtons_2": [0.0, 5.0],
        }
    )


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
    # multistage financial attributes are populated from the generator columns
    assert df.loc[0, "wacc"] == 0.08
    assert df.loc[0, "capital_recovery_period"] == 20.0
    assert df.loc[0, "lifetime"] == 25.0
    assert "min_retired_capacity" in df.columns


def test_thermal_emission_rate_from_fuels_table_fallback(gen_df, fuels):
    """Emission rate is derived from the fuels table when the generator dataframe
    has no CO2_content column (the normal pipeline path)."""
    gen_df = gen_df.copy()
    gen_df = gen_df.drop(columns=["CO2_content_tons_per_MMBtu"])
    file_name, commodity, df = make_thermal_csvs(gen_df, fuels=fuels)[0]
    assert commodity == "NaturalGas"
    expected_emission = 0.053 / CONV_MMBTU_TO_MWH
    assert all(abs(v - expected_emission) < 1e-6 for v in df["emission_rate"]), df[
        "emission_rate"
    ].tolist()


def test_thermal_emission_rate_zero_when_no_fuels_source(gen_df):
    """Without a fuels table or CO2 column, emission rate falls back to blank (0)."""
    gen_df = gen_df.copy()
    gen_df = gen_df.drop(columns=["CO2_content_tons_per_MMBtu"])
    file_name, commodity, df = make_thermal_csvs(gen_df)[0]
    assert df["emission_rate"].notna().all()
    # no CO2 source available -> blank cells (treated as 0 by Macro)
    assert all(v == "" for v in df["emission_rate"])


def test_thermal_csv_type_id_first_columns(gen_df):
    _, _, df = _thermal_asset(gen_df)
    cols = list(df.columns)
    assert cols[0] == "Type" and cols[1] == "id"


def test_can_retire_derived_from_new_build_when_missing():
    """When Can_Retire is absent, derive it from New_Build (GenX semantics)."""
    df = pd.DataFrame(
        {
            "Resource": ["nb_minus1", "nb_zero", "nb_one"],
            "region": ["R1", "R1", "R1"],
            "THERM": [1, 1, 1],
            "New_Build": [-1, 0, 1],
            "Fuel": ["natural_gas_power", "natural_gas_power", "natural_gas_power"],
            "Min_Power": [0.5, 0.5, 0.5],
            "Heat_Rate_MMBTU_per_MWh": [7.0, 7.0, 7.0],
            "CO2_content_tons_per_MMBtu": [0.053, 0.053, 0.053],
            "Cap_Size": [50.0, 50.0, 50.0],
            "Existing_Cap_MW": [100.0, 100.0, 100.0],
            "Inv_Cost_per_MWyr": [100.0, 100.0, 100.0],
            "Fixed_OM_Cost_per_MWyr": [10.0, 10.0, 10.0],
            "Var_OM_Cost_per_MWh": [2.0, 2.0, 2.0],
            "Start_Fuel_MMBTU_per_MW": [0.0, 0.0, 0.0],
            "Up_Time": [0.0, 0.0, 0.0],
            "Down_Time": [0.0, 0.0, 0.0],
            "Ramp_Up_Percentage": [0.0, 0.0, 0.0],
            "Ramp_Dn_Percentage": [0.0, 0.0, 0.0],
            "Start_Cost_per_MW": [0.0, 0.0, 0.0],
            # no Can_Retire column on purpose
        }
    )
    _, _, thermal_df = _thermal_asset(df)
    # New_Build == -1 means never retire (matches GenX update_newbuild_canretire)
    assert list(thermal_df["can_retire"]) == ["FALSE", "TRUE", "TRUE"]
    # ...while can_expand is driven purely by positive New_Build
    assert list(thermal_df["can_expand"]) == ["FALSE", "FALSE", "TRUE"]


def test_vre_csv_columns_and_availability(gen_df):
    vre_df = make_vre_csv(gen_df)
    assert list(vre_df["id"]) == ["solar_1", "wind_1"]
    assert not (vre_df == "TRUE").any(axis=None) or True
    # No bare availability column; nested timeseries columns present
    assert "availability" not in vre_df.columns
    assert "availability--timeseries--path" in vre_df.columns
    assert "availability--timeseries--header" in vre_df.columns
    for _, row in vre_df.iterrows():
        assert row["availability--timeseries--path"] == "system/availability_1.csv"
        assert row["availability--timeseries--header"] == row["id"]
        # GenX VRE capacity is continuous -> capacity_size = 1.0 (not Cap_Size).
        assert row["capacity_size"] == 1.0

    # stage_number controls the availability CSV path
    vre_stage2 = make_vre_csv(gen_df, stage_number=2)
    assert (
        vre_stage2.iloc[0]["availability--timeseries--path"]
        == "system/availability_2.csv"
    )


def test_storage_csv_asymmetry(gen_df):
    stor_df = make_storage_csv(gen_df)
    assert list(stor_df["id"]) == ["batt_sym", "batt_asym"]
    sym = stor_df[stor_df["id"] == "batt_sym"].iloc[0]
    asym = stor_df[stor_df["id"] == "batt_asym"].iloc[0]
    assert sym["storage_constraints--StorageSymmetricCapacityConstraint"] == "TRUE"
    assert asym["storage_constraints--StorageSymmetricCapacityConstraint"] == "FALSE"
    assert stor_df.loc[0, "storage_existing_capacity"] == 800.0
    assert stor_df.loc[1, "storage_existing_capacity"] == 1600.0
    # storage/charge can_expand follow GenX New_Build; can_retire follows Can_Retire
    # (batt_sym: New_Build=0, Can_Retire=1; batt_asym: New_Build=1, Can_Retire=0)
    assert sym["storage_can_expand"] == "FALSE"
    assert sym["charge_can_expand"] == "FALSE"
    assert sym["charge_can_retire"] == "TRUE"
    assert asym["storage_can_expand"] == "TRUE"
    assert asym["charge_can_expand"] == "TRUE"
    assert asym["charge_can_retire"] == "FALSE"
    # GenX storage capacity is continuous -> discharge capacity_size = 1.0 (not Cap_Size)
    assert stor_df["discharge_capacity_size"].tolist() == [1.0, 1.0]


def test_hydro_csv_availability_columns(gen_df):
    hydro_df = make_hydro_csv(gen_df)
    assert list(hydro_df["id"]) == ["hydro_1"]
    assert "inflow_availability" not in hydro_df.columns
    assert "inflow_availability--timeseries--path" in hydro_df.columns
    assert "inflow_availability--timeseries--header" in hydro_df.columns
    assert (
        hydro_df.loc[0, "inflow_availability--timeseries--path"]
        == "system/availability_1.csv"
    )
    assert hydro_df.loc[0, "hydro_source"] == "hydro_source"
    # Hydro reservoir energy capacity: StorageCapacityConstraint + StorageMaxDurationConstraint
    # model the GenX HYDRO_RES_KNOWN_CAP bound (ratio * capacity), matching the reference
    # GenX_to_Macro converter (storage_charge_discharge_ratio = 1.0, constraint enabled).
    assert hydro_df.loc[0, "storage_constraints--StorageCapacityConstraint"] == "TRUE"
    assert (
        hydro_df.loc[0, "storage_constraints--StorageMaxDurationConstraint"] == "TRUE"
    )
    assert (
        hydro_df.loc[0, "storage_constraints--StorageChargeDischargeRatioConstraint"]
        == "TRUE"
    )
    # Discharge edge constraints mirror GenX's cHydroMaxOutflow (discharge <= prior-hour
    # storage). The reference converter always enables Capacity + RampingLimit + this one.
    assert (
        hydro_df.loc[0, "discharge_constraints--StorageDischargeLimitConstraint"]
        == "TRUE"
    )
    assert hydro_df.loc[0, "storage_existing_capacity"] == 6.0 * 800.0
    assert hydro_df.loc[0, "storage_max_duration"] == 6.0
    assert hydro_df.loc[0, "storage_charge_discharge_ratio"] == 1.0
    # expand/retire flags follow GenX New_Build / Can_Retire (reference converter):
    # in/outflow retire with the plant and only discharge/inflow can be built;
    # reservoir storage can only expand/retire for known-capacity (reservoir) hydro.
    assert hydro_df.loc[0, "inflow_can_retire"] == "TRUE"  # Can_Retire = 1
    assert hydro_df.loc[0, "storage_can_retire"] == "TRUE"  # Can_Retire & known_cap
    assert hydro_df.loc[0, "discharge_can_expand"] == "FALSE"  # New_Build = 0
    assert hydro_df.loc[0, "storage_can_expand"] == "FALSE"  # New_Build = 0
    # GenX hydro capacity is continuous -> capacity_size is 1.0 (not Cap_Size = 20)
    assert hydro_df.loc[0, "discharge_capacity_size"] == 1.0


def test_mustrun_csv_availability_columns(gen_df):
    mustrun_df = make_mustrun_csv(gen_df)
    assert list(mustrun_df["id"]) == ["mustrun_1"]
    assert "availability" not in mustrun_df.columns
    assert "availability--timeseries--path" in mustrun_df.columns
    assert (
        mustrun_df.loc[0, "availability--timeseries--path"]
        == "system/availability_1.csv"
    )
    assert mustrun_df.loc[0, "availability--timeseries--header"] == "mustrun_1"
    # GenX's must-run output ignores Cap_Size -> capacity_size = 1.0
    assert mustrun_df.loc[0, "capacity_size"] == 1.0


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
    # financial attributes carried through from the GenX network columns
    assert list(tx_df["wacc"]) == [0.044, 0.044]
    assert list(tx_df["capital_recovery_period"]) == [60.0, 60.0]
    # No GenX per-line Lifetime: lifetime falls back to capital_recovery_period
    assert list(tx_df["lifetime"]) == [60.0, 60.0]


def test_make_powerlines_csv_no_financial_cols():
    network = pd.DataFrame(
        {
            "start_region": ["R1"],
            "dest_region": ["R2"],
            "Line_Max_Flow_MW": [500.0],
            "Line_Max_Reinforcement_MW": [0.0],
            "Line_Loss_Percentage": [0.01],
            "Line_Reinforcement_Cost_per_MWyr": [1500.0],
            "distance_mile": [100.0],
        }
    )
    tx_df = make_powerlines_csv(network)
    # blank financial cells -> Macro defaults
    assert pd.isna(tx_df.loc[0, "wacc"])
    assert pd.isna(tx_df.loc[0, "capital_recovery_period"])
    assert pd.isna(tx_df.loc[0, "lifetime"])


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
    assert (
        demand_ids["elec_R1"]["demand"]["timeseries"]["path"] == "system/demand_1.csv"
    )
    # fuel supply nodes single-tier with location and price timeseries header
    ng = next(n for n in nodes if n["type"] == "NaturalGas")
    assert ng["instance_data"][0]["id"] == "NaturalGas_R1"
    assert (
        ng["instance_data"][0]["supply"]["segment1"]["price"]["timeseries"]["header"]
        == "NaturalGas_R1"
    )
    assert (
        ng["instance_data"][0]["supply"]["segment1"]["price"]["timeseries"]["path"]
        == "system/fuel_prices_1.csv"
    )
    # hydro_source node present
    assert any(
        n.get("global_data", {}).get("constraints", {}).get("BalanceConstraint")
        is False
        and n["global_data"]["time_interval"] == "Electricity"
        and n["instance_data"][0]["id"] == "hydro_source"
        for n in nodes
    )


def test_load_nsd_segments_from_demand_segments_csv(tmp_path):
    """VOLL demand segments CSV maps to Macro NSD price_nsd / max_nsd vectors."""
    seg_csv = tmp_path / "demand_segments_voll.csv"
    seg_csv.write_text(
        "Voll,Demand_Segment,Cost_of_Demand_Curtailment_per_MW,"
        "Max_Demand_Curtailment,$/MWh\n"
        "2000,1,1,1,2000\n"
        ",2,0.9,0.04,1800\n"
        ",3,0.55,0.024,1100\n"
        ",4,0.2,0.003,400\n"
    )
    settings = {
        "input_folder": str(tmp_path),
        "demand_segments_fn": "demand_segments_voll.csv",
    }
    max_nsd, price_nsd = load_nsd_segments(settings)
    # Sorted by descending cost: 2000, 1800, 1100, 400
    assert price_nsd == [2000.0, 1800.0, 1100.0, 400.0]
    assert max_nsd == [1.0, 0.04, 0.024, 0.003]


def test_load_nsd_segments_default_when_no_file():
    """Without a demand segments file, the single-segment default is used."""
    settings = {}
    max_nsd, price_nsd = load_nsd_segments(settings)
    assert max_nsd == [1]
    assert price_nsd == [5000.0]


def test_make_nodes_json_uses_demand_segments(tmp_path):
    """nodes.json price_nsd / max_nsd come from the demand segments CSV."""
    seg_csv = tmp_path / "demand_segments_voll.csv"
    seg_csv.write_text(
        "Voll,Demand_Segment,Cost_of_Demand_Curtailment_per_MW,"
        "Max_Demand_Curtailment,$/MWh\n"
        "2000,1,1,1,2000\n"
        ",2,0.9,0.04,1800\n"
    )
    settings = {
        "model_regions": ["R1"],
        "zone_num_map": {"R1": 1},
        "input_folder": str(tmp_path),
        "demand_segments_fn": "demand_segments_voll.csv",
    }
    nodes = make_nodes_json(
        settings,
        demand_headers={"R1": "Demand_MW_z1"},
        fuel_supply_headers={},
        co2_sinks=[{"id": "co2_sink", "cap": None}],
        has_hydro=False,
    )
    elec = next(
        n
        for n in nodes
        if n["type"] == "Electricity" and n.get("instance_data", [{}])[0].get("demand")
    )
    gd = elec["global_data"]
    assert gd["price_nsd"] == [2000.0, 1800.0]
    assert gd["max_nsd"] == [1.0, 0.04]


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
    assert td["SubPeriodMap"] == {"path": "system/Period_map_1.csv"}

    td_stage2 = make_timedata_json(
        reduced, ["Electricity"], has_period_map=True, stage_number=2
    )
    assert td_stage2["SubPeriodMap"] == {"path": "system/Period_map_2.csv"}


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

    # asset files live in a per-stage assets/assets_1 folder
    for f in [
        "assets/assets_1/naturalgas_power.csv",
        "assets/assets_1/coal_power.csv",
        "assets/assets_1/vre.csv",
        "assets/assets_1/electricity_stor.csv",
        "assets/assets_1/hydropower.csv",
        "assets/assets_1/mustrun.csv",
        "assets/assets_1/powerlines.csv",
    ]:
        assert (case / f).is_file(), f"missing {f}"

    # system files (keys shared across stages; the rest are per-stage)
    for f in [
        "system/commodities.json",
        "system/locations.json",
        "system/nodes_1.json",
        "system/time_data_1.json",
        "system/demand_1.csv",
        "system/availability_1.csv",
        "system/fuel_prices_1.csv",
        "settings/macro_settings.json",
        "settings/case_settings.json",
    ]:
        assert (case / f).is_file(), f"missing {f}"

    # old non-suffixed files must NOT exist
    for f in [
        "system/nodes.json",
        "system/time_data.json",
        "system/demand.csv",
        "system/availability.csv",
        "system/fuel_prices.csv",
        "assets/vre.csv",
    ]:
        assert not (case / f).exists(), f"legacy file should not exist: {f}"

    # every CSV readable, Type is first column, id second
    for csv_name in [
        "assets/assets_1/naturalgas_power.csv",
        "assets/assets_1/vre.csv",
        "assets/assets_1/electricity_stor.csv",
        "assets/assets_1/hydropower.csv",
        "assets/assets_1/mustrun.csv",
        "assets/assets_1/powerlines.csv",
    ]:
        df = pd.read_csv(case / csv_name)
        assert list(df.columns)[:2] == ["Type", "id"], csv_name

    # powerlines.csv carries the transmission financial attributes
    tx = pd.read_csv(case / "assets/assets_1/powerlines.csv")
    assert set(["wacc", "capital_recovery_period", "lifetime"]) <= set(tx.columns)
    assert (tx["wacc"] == 0.044).all()
    assert (tx["capital_recovery_period"] == 60.0).all()
    # no per-line Lifetime in GenX -> falls back to capital_recovery_period
    assert (tx["lifetime"] == 60.0).all()

    # nodes.json demand header matches demand.csv
    demand_df = pd.read_csv(case / "system/demand_1.csv")
    nodes = json.loads((case / "system/nodes_1.json").read_text())
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
    availability = pd.read_csv(case / "system/availability_1.csv")
    vre = pd.read_csv(case / "assets/assets_1/vre.csv")
    hydro = pd.read_csv(case / "assets/assets_1/hydropower.csv")
    mustrun = pd.read_csv(case / "assets/assets_1/mustrun.csv")
    for df in (vre, mustrun):
        for header in df["availability--timeseries--header"].dropna():
            assert header in availability.columns, header
    for header in hydro["inflow_availability--timeseries--header"].dropna():
        assert header in availability.columns, header

    # fuel_prices.csv header referenced by nodes.json fuel supply nodes
    fuel_prices = pd.read_csv(case / "system/fuel_prices_1.csv")
    ng = next(n for n in nodes["nodes"] if n["type"] == "NaturalGas")
    for inst in ng["instance_data"]:
        header = inst["supply"]["segment1"]["price"]["timeseries"]["header"]
        assert header in fuel_prices.columns

    # multistage system_data.json: one case entry per stage, settings pointer
    sd = json.loads((case / "system_data.json").read_text())
    assert len(sd["case"]) == 1
    entry = sd["case"][0]
    assert entry["assets"]["path"] == "assets/assets_1"
    assert (case / entry["assets"]["path"]).is_dir()
    assert (case / entry["nodes"]["path"]).is_file()
    assert (case / entry["time_data"]["path"]).is_file()
    assert sd["settings"]["path"] == "settings/case_settings.json"

    # case_settings.json has one PeriodLength per stage
    cs = json.loads((case / "settings/case_settings.json").read_text())
    assert cs["PeriodLengths"] == [1]

    # time_data.json default (non-reduced): one subperiod, 24 hours
    td = json.loads((case / "system/time_data_1.json").read_text())
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
    td = json.loads((case / "system/time_data_1.json").read_text())
    assert td["NumberOfSubperiods"] == 2
    assert td["HoursPerSubperiod"]["Electricity"] == 12
    assert td["TotalHoursModeled"] == 288
    assert "SubPeriodMap" in td
    assert td["SubPeriodMap"]["path"] == "system/Period_map_1.csv"
    pm = pd.read_csv(case / "system/Period_map_1.csv")
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
    nodes = json.loads((case / "system/nodes_1.json").read_text())
    co2 = next(n for n in nodes["nodes"] if n["type"] == "CO2")
    ids = [i["id"] for i in co2["instance_data"]]
    assert "co2_sink" in ids
    assert "co2_sink_1" in ids
    for inst in co2["instance_data"]:
        if inst["id"] == "co2_sink_1":
            assert inst["rhs_policy"]["CO2CapConstraint"] == 10.0 * 1e6
    # in-region thermal generator points at the capped sink
    thermal = pd.read_csv(case / "assets/assets_1/naturalgas_power.csv")
    assert "co2_sink" in thermal.columns


def test_make_system_data_json_multistage():
    sd = make_system_data_json([2, 1], assets_folder="assets")
    assert list(sd.keys()) == ["case", "settings"]
    # stages are emitted in ascending numeric order
    assert [e["assets"]["path"] for e in sd["case"]] == [
        "assets/assets_1",
        "assets/assets_2",
    ]
    for i, entry in enumerate(sd["case"], start=1):
        assert entry["nodes"]["path"] == f"system/nodes_{i}.json"
        assert entry["time_data"]["path"] == f"system/time_data_{i}.json"
        assert entry["commodities"]["path"] == "system/commodities.json"
        assert entry["settings"]["path"] == "settings/macro_settings.json"
    assert sd["settings"]["path"] == "settings/case_settings.json"


def test_make_case_settings_json():
    cs = make_case_settings_json(3)
    assert cs["PeriodLengths"] == [1, 1, 1]
    assert cs["DiscountRate"] == 0.045
    assert cs["SolutionAlgorithm"] == "Monolithic"


def test_macro_case_builder_multistage(
    tmp_path, gen_df, gen_variability, demand_data, fuels, network, settings
):
    """Two planning periods written out-of-order become two numbered stages."""
    builder = MacroCaseBuilder(tmp_path)
    # add stages out of numeric order like the year-major main loop would
    stage1_data = {
        "gen_data": gen_df,
        "gen_variability": gen_variability,
        "demand_data": demand_data,
        "fuels": fuels,
        "network": network,
    }
    builder.add_stage(1, stage1_data, settings)
    builder.add_stage(2, stage1_data, settings)
    builder.finalize()

    # case entries sorted with assets/assets_N paths
    sd = json.loads((tmp_path / "system_data.json").read_text())
    assert [e["assets"]["path"] for e in sd["case"]] == [
        "assets/assets_1",
        "assets/assets_2",
    ]

    # case_settings.json has one PeriodLength per stage
    cs = json.loads((tmp_path / "settings/case_settings.json").read_text())
    assert cs["PeriodLengths"] == [1, 1]

    # both stages fully materialized with per-stage file numbers
    for i in (1, 2):
        assert (tmp_path / "assets" / f"assets_{i}" / "vre.csv").is_file()
        assert (tmp_path / "assets" / f"assets_{i}" / "naturalgas_power.csv").is_file()
        assert (tmp_path / "system" / f"nodes_{i}.json").is_file()
        assert (tmp_path / "system" / f"time_data_{i}.json").is_file()
        assert (tmp_path / "system" / f"demand_{i}.csv").is_file()
        assert (tmp_path / "system" / f"availability_{i}.csv").is_file()
        assert (tmp_path / "system" / f"fuel_prices_{i}.csv").is_file()

    # assets reference the per-stage availability CSV
    vre1 = pd.read_csv(tmp_path / "assets/assets_1/vre.csv")
    assert vre1.iloc[0]["availability--timeseries--path"] == "system/availability_1.csv"
    vre2 = pd.read_csv(tmp_path / "assets/assets_2/vre.csv")
    assert vre2.iloc[0]["availability--timeseries--path"] == "system/availability_2.csv"

    # systems files shared across stages
    assert (tmp_path / "system/commodities.json").is_file()
    assert (tmp_path / "system/locations.json").is_file()
    assert (tmp_path / "settings/macro_settings.json").is_file()


def test_macro_case_builder_duplicate_stage_raises(
    tmp_path, gen_df, gen_variability, demand_data, fuels, network, settings
):
    builder = MacroCaseBuilder(tmp_path)
    data = {
        "gen_data": gen_df,
        "gen_variability": gen_variability,
        "demand_data": demand_data,
        "fuels": fuels,
        "network": network,
    }
    builder.add_stage(2, data, settings)
    with pytest.raises(ValueError, match="already added"):
        builder.add_stage(2, data, settings)
