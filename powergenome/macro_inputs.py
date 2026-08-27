"""Generate MacroEnergy.jl simpleCSVinputs-format case inputs from PowerGenome data.

When enabled (via the ``macro_output`` setting or the ``--macro`` CLI flag),
PowerGenome writes a Macro (https://github.com/macroenergy/MacroEnergy.jl) case
folder in the *simpleCSVinputs* format instead of the GenX ``Inputs`` format.
The structure and column conventions follow the example set at
``macroenergy/MacroEnergyExamples.jl/examples/multisector_3zone_simpleCSVinputs``
and the semantic mapping from GenX data follows
``EmilDimanchev/GenX_to_Macro``.

Only the simpleCSV format is produced (no Macro JSON asset files).
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from powergenome.external_data import load_demand_segments

logger = logging.getLogger(__name__)

# Conversion factor MMBtu -> MWh (1 MWh = 3.412 MMBtu)
CONV_MMBTU_TO_MWH = 0.29307107

# Macro asset type names used in the simpleCSV "Type" column
THERMAL_TYPE = "ThermalPower"
THERMALCCS_TYPE = "ThermalPowerCCS"
VRE_TYPE = "VRE"
STORAGE_TYPE = "Battery"
HYDRO_TYPE = "HydroRes"
MUST_RUN_TYPE = "MustRun"
TRANSMISSION_TYPE = "TransmissionLink"

# Map a (possibly region/scenario-prefixed) fuel name to a Macro commodity name.
# Substrings are matched in order; the first hit wins.
FUEL_COMMODITIES = {
    "natural_gas": "NaturalGas",
    "naturalgas": "NaturalGas",
    "coal": "Coal",
    "uranium": "Uranium",
    "biomass": "Biomass",
    "wood": "Biomass",
    "distillate": "Distillate",
    "oil": "Oil",
    "refuse": "Refuse",
    "municipal": "MunicipalSolidWaste",
    "geothermal": "Geothermal",
    "hydrogen": "Hydrogen",
}

# Financial attributes added by the multistage GenX_to_Macro conversion
# (get_wacc_and_crp / Resource_multistage_data.csv). Macro has built-in
# defaults for these, so blank cells fall back gracefully.
FINANCIAL_COLUMNS = [
    "wacc",
    "capital_recovery_period",
    "lifetime",
    "min_retired_capacity",
]

# simpleCSV column layouts (order is preserved when writing CSVs).
# Taken from the multisector_3zone_simpleCSVinputs example assets.
THERMAL_COLUMNS = [
    "Type",
    "id",
    "co2_sink",
    "timedata",
    "fuel_commodity",
    "fuel_start_vertex",
    "uc",
    "elec_constraints--MinFlowConstraint",
    "elec_constraints--MinDownTimeConstraint",
    "elec_constraints--CapacityConstraint",
    "elec_constraints--MinUpTimeConstraint",
    "elec_constraints--RampingLimitConstraint",
    "location",
    "can_expand",
    "min_down_time",
    "fuel_consumption",
    "fixed_om_cost",
    "existing_capacity",
    "min_up_time",
    "capacity_size",
    "ramp_down_fraction",
    "emission_rate",
    "variable_om_cost",
    "annualized_investment_cost",
    "startup_fuel_consumption",
    "ramp_up_fraction",
    "min_flow_fraction",
    "startup_cost",
    "can_retire",
    *FINANCIAL_COLUMNS,
]

# Extra simpleCSV columns present only on CCS (ThermalPowerCCS) assets. The
# ``edges--co2_captured_edge--*`` nesting maps into
# ``data[:edges][:co2_captured_edge][:end_vertex]`` / ``[:variable_om_cost]``,
# which Macro's ThermalPowerCCS requires for the captured-CO2 edge.
CCS_COLUMNS = [
    "capture_rate",
    "edges--co2_captured_edge--end_vertex",
    "edges--co2_captured_edge--variable_om_cost",
]

# Name of the single (location-less, uncapped) CO2Captured sink node shared by
# every CCS asset's captured-CO2 edge. Matches the GenX_to_Macro "co2_sink_injection"
# convention.
CO2_CAPTURED_NODE = "co2_sink_injection"

VRE_COLUMNS = [
    "Type",
    "id",
    "elec_can_expand",
    "elec_can_retire",
    "elec_constraints--MaxCapacityConstraint",
    "max_capacity",
    "location",
    "fixed_om_cost",
    "annualized_investment_cost",
    "availability--timeseries--path",
    "availability--timeseries--header",
    "existing_capacity",
    "capacity_size",
    *FINANCIAL_COLUMNS,
]

STORAGE_COLUMNS = [
    "Type",
    "id",
    "discharge_can_retire",
    "storage_constraints--StorageMinDurationConstraint",
    "storage_constraints--StorageCapacityConstraint",
    "storage_constraints--StorageMaxDurationConstraint",
    "storage_constraints--StorageSymmetricCapacityConstraint",
    "storage_constraints--BalanceConstraint",
    "discharge_constraints--StorageDischargeLimitConstraint",
    "discharge_constraints--CapacityConstraint",
    "storage_can_retire",
    "storage_can_expand",
    "location",
    "storage_max_duration",
    "discharge_annualized_investment_cost",
    "discharge_efficiency",
    "storage_fixed_om_cost",
    "discharge_fixed_om_cost",
    "storage_annualized_investment_cost",
    "discharge_variable_om_cost",
    "charge_efficiency",
    "storage_min_duration",
    "charge_variable_om_cost",
    "storage_existing_capacity",
    "discharge_min_flow_fraction",
    "discharge_existing_capacity",
    "discharge_can_expand",
    "charge_can_retire",
    "charge_can_expand",
    "discharge_capacity_size",
    *FINANCIAL_COLUMNS,
]

HYDRO_COLUMNS = [
    "Type",
    "id",
    "discharge_can_retire",
    "inflow_can_retire",
    "storage_long_duration",
    "storage_constraints--MinStorageOutflowConstraint",
    "storage_constraints--LongDurationStorageImplicitMinMaxConstraint",
    "storage_constraints--StorageCapacityConstraint",
    "storage_constraints--StorageMaxDurationConstraint",
    "storage_constraints--StorageChargeDischargeRatioConstraint",
    "storage_constraints--BalanceConstraint",
    "discharge_constraints--CapacityConstraint",
    "discharge_constraints--RampingLimitConstraint",
    "discharge_constraints--StorageDischargeLimitConstraint",
    "hydro_source",
    "storage_can_expand",
    "discharge_can_expand",
    "inflow_can_expand",
    "storage_can_retire",
    "location",
    "discharge_ramp_down_fraction",
    "inflow_availability--timeseries--path",
    "inflow_availability--timeseries--header",
    "discharge_existing_capacity",
    "discharge_efficiency",
    "inflow_efficiency",
    "discharge_fixed_om_cost",
    "discharge_ramp_up_fraction",
    "storage_min_outflow_fraction",
    "storage_existing_capacity",
    "storage_max_duration",
    "discharge_capacity_size",
    "storage_charge_discharge_ratio",
    *FINANCIAL_COLUMNS,
]

MUST_RUN_COLUMNS = [
    "Type",
    "id",
    "can_retire",
    "fixed_om_cost",
    "can_expand",
    "location",
    "existing_capacity",
    "availability--timeseries--path",
    "availability--timeseries--header",
    "capacity_size",
    *FINANCIAL_COLUMNS,
]

TRANSMISSION_COLUMNS = [
    "Type",
    "id",
    "transmission_constraints--MaxCapacityConstraint",
    "max_capacity",
    "transmission_origin",
    "loss_fraction",
    "existing_capacity",
    "annualized_investment_cost",
    "commodity",
    "distance",
    "transmission_dest",
    *FINANCIAL_COLUMNS,
]


# ---------------------------------------------------------------------------
# Small helper functions
# ---------------------------------------------------------------------------


def _is_true(value) -> bool:
    """Treat a value as True if it is a positive number or a truthy string."""
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in ("true", "t", "1", "yes")
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    if np.isnan(v):
        return False
    return v > 0


def _num(value, default=None):
    """Return ``value`` as a float, or ``default`` when it is NaN/None."""
    if value is None:
        return default
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if np.isnan(v):
        return default
    return v


def _clean_fuel_name(fuel: Optional[str]) -> Optional[str]:
    """Return a normalized base fuel name from a PowerGenome full fuel name."""
    if fuel is None or (isinstance(fuel, float) and np.isnan(fuel)):
        return None
    name = re.sub(r"[_\-\s]+", "_", str(fuel).strip()).lower()
    return name or None


def _fuel_commodity(fuel: Optional[str]) -> Optional[str]:
    """Map a PowerGenome fuel name to a Macro commodity name (or None)."""
    name = _clean_fuel_name(fuel)
    if not name:
        return None
    for fragment, commodity in FUEL_COMMODITIES.items():
        if fragment in name:
            return commodity
    return None


def _format_bool(value) -> str:
    """Serialize a value to a simpleCSV boolean cell (TRUE/FALSE/empty)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if _is_true(value):
        return "TRUE"
    return "FALSE"


def _financial_attrs(row: pd.Series) -> dict:
    """Return the multistage financial attributes, when the generator has them.

    Reads the PowerGenome/GenX columns (WACC, Capital_Recovery_Period,
    Lifetime, Min_Retired_Cap_MW) and maps them onto the Macro simpleCSV
    financial columns. Missing columns produce no entry (blank cell -> Macro
    default).

    ``min_retired_capacity`` is capped at the resource's existing capacity so
    that a GenX multistage file that requests more retirement than the plant
    has does not make the Macro model infeasible.
    """
    mapping = {
        "WACC": "wacc",
        "Capital_Recovery_Period": "capital_recovery_period",
        "Lifetime": "lifetime",
        "Min_Retired_Cap_MW": "min_retired_capacity",
    }
    out = {}
    for pg_col, macro_col in mapping.items():
        if pg_col in row.index and row.get(pg_col) is not None:
            out[macro_col] = row.get(pg_col)
    existing = _num(_gen_value(row, "Existing_Cap_MW", np.nan), np.nan)
    min_retired = _num(out.get("min_retired_capacity"), 0.0)
    if (
        min_retired is not None
        and existing is not None
        and not np.isnan(existing)
        and min_retired > existing
    ):
        out["min_retired_capacity"] = existing
    return out


def _availability_filename(stage_number: int) -> str:
    return f"system/availability_{stage_number}.csv"


# ---------------------------------------------------------------------------
# Asset CSV builders
# ---------------------------------------------------------------------------


def _gen_value(series: pd.Series, name: str, default=None):
    """Return a value from a generator row if the column exists."""
    if name in series.index:
        return series.get(name, default)
    return default


def _is_committed(row: pd.Series) -> bool:
    """Detect unit commitment for a thermal generator row."""
    commit = _gen_value(row, "Commit")
    if commit is not None:
        return _is_true(commit)
    model = _gen_value(row, "Model")
    if model is not None:
        model_str = str(model).lower()
        return "commit" in model_str or _num(model, 0) >= 2
    therm = _gen_value(row, "THERM")
    if therm is not None:
        return _num(therm, 0) >= 2
    return False


def _storage_is_asymmetric(row: pd.Series) -> bool:
    model = _gen_value(row, "Model")
    if model is not None and isinstance(model, str):
        return "asym" in model.lower()
    stor = _gen_value(row, "STOR")
    if stor is not None:
        return _num(stor, 0) >= 2
    return False


def _prep_gen_df(gen_df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that would produce invalid macro assets and add defaults."""
    if gen_df is None or gen_df.empty:
        return pd.DataFrame()
    df = gen_df.copy()
    # All resource rows need an id and a region
    if "Resource" not in df.columns:
        return pd.DataFrame()
    for col in ("region", "New_Build", "Existing_Cap_MW"):
        if col not in df.columns:
            df[col] = 0
    # GenX stores "never retire" as New_Build == -1 and derives Can_Retire
    # from it (see GenX.update_newbuild_canretire). Replicate that so Macro
    # retires the same assets GenX would.
    if "Can_Retire" not in df.columns:
        df["Can_Retire"] = (df["New_Build"] != -1).astype(int)
    for col in ("Inv_Cost_per_MWyr", "Fixed_OM_Cost_per_MWyr"):
        if col not in df.columns:
            df[col] = 0.0
    return df


def _thermal_asset_filename(commodity: str) -> str:
    return f"{commodity.lower()}_power.csv"


def make_thermal_csvs(
    gen_df: pd.DataFrame,
    settings: dict = None,
    fuels: pd.DataFrame = None,
) -> List[tuple]:
    """Build one thermal simpleCSV asset file per base fuel commodity.

    CO2 emission rates are taken from the ``fuels`` table (the GenX
    ``Fuels_data`` format, where row 0 holds per-fuel CO2 content in tonnes
    per MMBtu) when present, otherwise from a ``CO2_content_tons_per_MMBtu``
    column on the generator dataframe.

    Returns a list of ``(file_name, commodity, DataFrame)`` tuples. Each file
    contains the thermal generators (THERM tag) that burn the corresponding fuel.
    """
    gen_df = _prep_gen_df(gen_df)
    if gen_df.empty:
        return []
    res = gen_df[gen_df["THERM"] > 0]
    if res.empty:
        return []

    conv = CONV_MMBTU_TO_MWH
    by_commodity: Dict[str, pd.DataFrame] = {}

    for _, row in res.iterrows():
        commodity = _fuel_commodity(_gen_value(row, "Fuel"))
        if commodity is None:
            logger.warning(
                "Could not map fuel %r to a Macro commodity; skipping generator %s",
                _gen_value(row, "Fuel"),
                _gen_value(row, "Resource"),
            )
            continue
        by_commodity.setdefault(commodity, []).append(row)

    out = []
    for commodity, rows in by_commodity.items():
        records = []
        file_has_ccs = False
        for row in rows:
            committed = _is_committed(row)
            heat_rate = _num(_gen_value(row, "Heat_Rate_MMBTU_per_MWh", np.nan), 0.0)
            fuel_consumption = heat_rate * conv
            fuel_name = _gen_value(row, "Fuel")
            co2_content = _num(
                _gen_value(row, "CO2_content_tons_per_MMBtu", np.nan), np.nan
            )
            if not _num(co2_content) and fuels is not None:
                # CO2 content lives in the fuels table (row 0 of each fuel column)
                _, co2_content = _fuel_price_by_fullname(fuels, fuel_name)
            emission_rate = _num(co2_content, 0.0) / conv if _num(co2_content) else 0.0
            region = _gen_value(row, "region")
            resource = _gen_value(row, "Resource")
            min_flow = _num(_gen_value(row, "Min_Power", 0.0), 0.0)

            is_ccs = _num(_gen_value(row, "CO2_Capture_Fraction", 0.0), 0.0) > 0
            file_has_ccs = file_has_ccs or is_ccs

            record = {
                "Type": THERMAL_TYPE,
                "id": resource,
                "co2_sink": _gen_value(row, "co2_sink", "co2_sink"),
                "timedata": commodity,
                "fuel_commodity": commodity,
                "fuel_start_vertex": f"{commodity}_{region}",
                "uc": _format_bool(committed),
                "elec_constraints--MinFlowConstraint": _format_bool(min_flow > 0),
                "elec_constraints--MinDownTimeConstraint": _format_bool(committed),
                "elec_constraints--CapacityConstraint": "TRUE",
                "elec_constraints--MinUpTimeConstraint": _format_bool(committed),
                "elec_constraints--RampingLimitConstraint": "TRUE",
                "location": region,
                "can_expand": _format_bool(_gen_value(row, "New_Build")),
                "min_down_time": _num(_gen_value(row, "Down_Time", np.nan)),
                "fuel_consumption": fuel_consumption if heat_rate else "",
                "fixed_om_cost": _num(
                    _gen_value(row, "Fixed_OM_Cost_per_MWyr", np.nan)
                ),
                "existing_capacity": _num(_gen_value(row, "Existing_Cap_MW", np.nan)),
                "min_up_time": _num(_gen_value(row, "Up_Time", np.nan)),
                "capacity_size": _num(_gen_value(row, "Cap_Size", np.nan)),
                "ramp_down_fraction": _gen_value(row, "Ramp_Dn_Percentage", np.nan),
                "emission_rate": emission_rate if _num(co2_content) else "",
                "variable_om_cost": _num(
                    _gen_value(row, "Var_OM_Cost_per_MWh", np.nan)
                ),
                "annualized_investment_cost": _num(
                    _gen_value(row, "Inv_Cost_per_MWyr", np.nan)
                ),
                "startup_fuel_consumption": _num(
                    _gen_value(row, "Start_Fuel_MMBTU_per_MW", np.nan), 0.0
                )
                * conv,
                "ramp_up_fraction": _gen_value(row, "Ramp_Up_Percentage", np.nan),
                "min_flow_fraction": min_flow if min_flow > 0 else "",
                "startup_cost": _num(_gen_value(row, "Start_Cost_per_MW", np.nan)),
                "can_retire": _format_bool(_gen_value(row, "Can_Retire")),
                **_financial_attrs(row),
            }

            # CCS assets split total emissions into a residual flow on the CO2
            # edge (co2_sink, at the capped sink like any other thermal plant)
            # and a captured flow on the co2_captured_edge, which ends at the
            # location-less, uncapped CO2Captured sink node.
            if is_ccs:
                cf = _num(_gen_value(row, "CO2_Capture_Fraction", 0.0), 0.0)
                record["Type"] = THERMALCCS_TYPE
                record["emission_rate"] = (
                    (1 - cf) * co2_content / conv if _num(co2_content) else ""
                )
                record["capture_rate"] = (
                    cf * co2_content / conv if _num(co2_content) else ""
                )
                record["edges--co2_captured_edge--end_vertex"] = CO2_CAPTURED_NODE
                record["edges--co2_captured_edge--variable_om_cost"] = _num(
                    _gen_value(row, "CCS_Disposal_Cost_per_Metric_Ton", np.nan)
                )
            else:
                record["capture_rate"] = ""
                record["edges--co2_captured_edge--end_vertex"] = ""
                record["edges--co2_captured_edge--variable_om_cost"] = ""

            records.append(record)
        columns = THERMAL_COLUMNS + (CCS_COLUMNS if file_has_ccs else [])
        out.append(
            (
                _thermal_asset_filename(commodity),
                commodity,
                pd.DataFrame(records, columns=columns),
            )
        )
    return out


def make_vre_csv(gen_df: pd.DataFrame, stage_number: int = 1) -> pd.DataFrame:
    """Build the VRE simpleCSV asset file."""
    gen_df = _prep_gen_df(gen_df)
    if gen_df.empty:
        return pd.DataFrame(columns=VRE_COLUMNS)
    res = gen_df[gen_df["VRE"] > 0]
    records = []
    for _, row in res.iterrows():
        region = _gen_value(row, "region")
        # GenX uses Max_Cap_MW <= 0 (or NaN) to mean "no explicit upper bound".
        # Writing 0 here locks an existing resource out entirely, so fall back to
        # the existing capacity instead (new-build VRE always carries a positive
        # Max_Cap_MW, so it is never affected by this fallback).
        max_cap = _num(_gen_value(row, "Max_Cap_MW", np.nan))
        if max_cap is None or max_cap <= 0:
            max_cap = _num(_gen_value(row, "Existing_Cap_MW", np.nan), 0.0)
        records.append(
            {
                "Type": VRE_TYPE,
                "id": _gen_value(row, "Resource"),
                "elec_can_expand": _format_bool(_gen_value(row, "New_Build")),
                "elec_can_retire": _format_bool(_gen_value(row, "Can_Retire")),
                "elec_constraints--MaxCapacityConstraint": "TRUE",
                "max_capacity": max_cap,
                "location": region,
                "fixed_om_cost": _num(
                    _gen_value(row, "Fixed_OM_Cost_per_MWyr", np.nan)
                ),
                "annualized_investment_cost": _num(
                    _gen_value(row, "Inv_Cost_per_MWyr", np.nan)
                ),
                "availability--timeseries--path": _availability_filename(stage_number),
                "availability--timeseries--header": _gen_value(row, "Resource"),
                "existing_capacity": _num(_gen_value(row, "Existing_Cap_MW", np.nan)),
                # GenX treats VRE capacity as continuous (no discrete unit sizes), so
                # the reference converter hardcodes capacity_size = 1.0 here; writing
                # GenX's Cap_Size would instead treat every MW as its own "unit".
                "capacity_size": 1.0,
                **_financial_attrs(row),
            }
        )
    return pd.DataFrame(records, columns=VRE_COLUMNS)


def make_storage_csv(gen_df: pd.DataFrame) -> pd.DataFrame:
    """Build the Battery simpleCSV asset file."""
    gen_df = _prep_gen_df(gen_df)
    if gen_df.empty:
        return pd.DataFrame(columns=STORAGE_COLUMNS)
    res = gen_df[gen_df["STOR"] > 0]
    records = []
    for _, row in res.iterrows():
        asymmetric = _storage_is_asymmetric(row)
        records.append(
            {
                "Type": STORAGE_TYPE,
                "id": _gen_value(row, "Resource"),
                "discharge_can_retire": _format_bool(_gen_value(row, "Can_Retire")),
                "storage_constraints--StorageMinDurationConstraint": "TRUE",
                "storage_constraints--StorageCapacityConstraint": "TRUE",
                "storage_constraints--StorageMaxDurationConstraint": "TRUE",
                "storage_constraints--StorageSymmetricCapacityConstraint": _format_bool(
                    not asymmetric
                ),
                "storage_constraints--BalanceConstraint": "TRUE",
                "discharge_constraints--StorageDischargeLimitConstraint": "TRUE",
                "discharge_constraints--CapacityConstraint": "TRUE",
                "storage_can_retire": _format_bool(_gen_value(row, "Can_Retire")),
                "storage_can_expand": _format_bool(_gen_value(row, "New_Build")),
                "location": _gen_value(row, "region"),
                "storage_max_duration": _num(_gen_value(row, "Max_Duration", np.nan)),
                "discharge_annualized_investment_cost": _num(
                    _gen_value(row, "Inv_Cost_per_MWyr", np.nan)
                ),
                "discharge_efficiency": _num(_gen_value(row, "Eff_Down", np.nan)),
                "storage_fixed_om_cost": _num(
                    _gen_value(row, "Fixed_OM_Cost_per_MWhyr", np.nan)
                ),
                "discharge_fixed_om_cost": _num(
                    _gen_value(row, "Fixed_OM_Cost_per_MWyr", np.nan)
                ),
                "storage_annualized_investment_cost": _num(
                    _gen_value(row, "Inv_Cost_per_MWhyr", np.nan)
                ),
                "discharge_variable_om_cost": _num(
                    _gen_value(row, "Var_OM_Cost_per_MWh", np.nan)
                ),
                "charge_efficiency": _num(_gen_value(row, "Eff_Up", np.nan)),
                "storage_min_duration": _num(_gen_value(row, "Min_Duration", np.nan)),
                "charge_variable_om_cost": _num(
                    _gen_value(row, "Var_OM_Cost_per_MWh_In", np.nan)
                ),
                # Existing_Cap_MWh is 0.0 for new-build-only storage and NaN when
                # absent; never fall through to Max_Cap_MWh (GenX's -1 "no max"
                # sentinel is not an existing capacity and makes Macro infeasible).
                "storage_existing_capacity": _num(
                    _gen_value(row, "Existing_Cap_MWh", np.nan), 0.0
                ),
                "discharge_min_flow_fraction": _num(
                    _gen_value(row, "Min_Power", 0.0), 0.0
                ),
                "discharge_existing_capacity": _num(
                    _gen_value(row, "Existing_Cap_MW", np.nan)
                ),
                "discharge_can_expand": _format_bool(_gen_value(row, "New_Build")),
                "charge_can_retire": _format_bool(_gen_value(row, "Can_Retire")),
                "charge_can_expand": _format_bool(_gen_value(row, "New_Build")),
                # GenX storage capacity is continuous; the reference converter
                # hardcodes storage/charge/discharge capacity_size = 1.0 (Macro's
                # default), rather than GenX's Cap_Size cluster size.
                "discharge_capacity_size": 1.0,
                **_financial_attrs(row),
            }
        )
    return pd.DataFrame(records, columns=STORAGE_COLUMNS)


def make_hydro_csv(gen_df: pd.DataFrame, stage_number: int = 1) -> pd.DataFrame:
    """Build the HydroRes simpleCSV asset file."""
    gen_df = _prep_gen_df(gen_df)
    if gen_df.empty:
        return pd.DataFrame(columns=HYDRO_COLUMNS)
    res = gen_df[gen_df["HYDRO"] > 0]
    records = []
    for _, row in res.iterrows():
        ratio = _num(_gen_value(row, "Hydro_Energy_to_Power_Ratio", 1.0), 1.0)
        existing_cap = _num(_gen_value(row, "Existing_Cap_MW", np.nan))
        # Reservoir hydro (GenX HYDRO_RES_KNOWN_CAP, matching the reference
        # converter's known-capacity set) is any hydro with a positive
        # energy-to-power ratio; only these carry reservoir storage/duration limits
        # and only these can expand or retire storage (run-of-river cannot).
        known_cap = ratio > 0
        new_build = _is_true(_gen_value(row, "New_Build", False))
        can_retire = _is_true(_gen_value(row, "Can_Retire", False))
        records.append(
            {
                "Type": HYDRO_TYPE,
                "id": _gen_value(row, "Resource"),
                "discharge_can_retire": _format_bool(can_retire),
                # The inflow edge mirrors GenX's hydro resource: it retires together
                # with the plant, but inflow capacity is never built.
                "inflow_can_retire": _format_bool(can_retire),
                "storage_long_duration": "FALSE",
                "storage_constraints--MinStorageOutflowConstraint": "FALSE",
                "storage_constraints--LongDurationStorageImplicitMinMaxConstraint": "FALSE",
                "storage_constraints--StorageCapacityConstraint": _format_bool(
                    known_cap
                ),
                "storage_constraints--StorageMaxDurationConstraint": _format_bool(
                    known_cap
                ),
                "storage_constraints--StorageChargeDischargeRatioConstraint": _format_bool(
                    known_cap
                ),
                "storage_constraints--BalanceConstraint": "TRUE",
                "discharge_constraints--CapacityConstraint": "TRUE",
                "discharge_constraints--RampingLimitConstraint": "TRUE",
                "discharge_constraints--StorageDischargeLimitConstraint": "TRUE",
                "hydro_source": "hydro_source",
                "storage_can_expand": _format_bool(new_build and known_cap),
                "discharge_can_expand": _format_bool(new_build),
                "inflow_can_expand": _format_bool(new_build),
                "storage_can_retire": _format_bool(can_retire and known_cap),
                "location": _gen_value(row, "region"),
                "discharge_ramp_down_fraction": _num(
                    _gen_value(row, "Ramp_Dn_Percentage", 1.0), 1.0
                ),
                "inflow_availability--timeseries--path": _availability_filename(
                    stage_number
                ),
                "inflow_availability--timeseries--header": _gen_value(row, "Resource"),
                "discharge_existing_capacity": existing_cap,
                "discharge_efficiency": _num(_gen_value(row, "Eff_Down", 1.0), 1.0),
                "inflow_efficiency": _num(_gen_value(row, "Eff_Up", 1.0), 1.0),
                "discharge_fixed_om_cost": _num(
                    _gen_value(row, "Fixed_OM_Cost_per_MWyr", np.nan)
                ),
                "discharge_ramp_up_fraction": _num(
                    _gen_value(row, "Ramp_Up_Percentage", 1.0), 1.0
                ),
                "storage_min_outflow_fraction": _num(
                    _gen_value(row, "Min_Power", np.nan)
                ),
                "storage_existing_capacity": (
                    ratio * existing_cap if known_cap else np.nan
                ),
                "storage_max_duration": ratio if known_cap else np.nan,
                # GenX hydro capacity is continuous; the reference converter hardcodes
                # discharge capacity_size = 1.0 for hydro.
                "discharge_capacity_size": 1.0,
                "storage_charge_discharge_ratio": 1.0,
                **_financial_attrs(row),
            }
        )
    return pd.DataFrame(records, columns=HYDRO_COLUMNS)


def make_mustrun_csv(gen_df: pd.DataFrame, stage_number: int = 1) -> pd.DataFrame:
    """Build the MustRun simpleCSV asset file."""
    gen_df = _prep_gen_df(gen_df)
    if gen_df.empty:
        return pd.DataFrame(columns=MUST_RUN_COLUMNS)
    res = gen_df[gen_df["MUST_RUN"] > 0]
    records = []
    for _, row in res.iterrows():
        records.append(
            {
                "Type": MUST_RUN_TYPE,
                "id": _gen_value(row, "Resource"),
                "can_retire": _format_bool(_gen_value(row, "Can_Retire")),
                "fixed_om_cost": _num(
                    _gen_value(row, "Fixed_OM_Cost_per_MWyr", np.nan)
                ),
                "can_expand": _format_bool(_gen_value(row, "New_Build")),
                "location": _gen_value(row, "region"),
                "existing_capacity": _num(_gen_value(row, "Existing_Cap_MW", np.nan)),
                "availability--timeseries--path": _availability_filename(stage_number),
                "availability--timeseries--header": _gen_value(row, "Resource"),
                # The reference converter notes that "GenX internally assumes
                # capacity_size = 1.0 for must run generators"; GenX's must-run output
                # is pP_Max * eTotalCap (continuous, ignoring Cap_Size entirely).
                "capacity_size": 1.0,
                **_financial_attrs(row),
            }
        )
    return pd.DataFrame(records, columns=MUST_RUN_COLUMNS)


def make_powerlines_csv(network: pd.DataFrame) -> pd.DataFrame:
    """Build the TransmissionLink simpleCSV asset file from the GenX network df."""
    if network is None or network.empty:
        return pd.DataFrame(columns=TRANSMISSION_COLUMNS)
    records = []
    for _, row in network.iterrows():
        start = _gen_value(row, "start_region")
        dest = _gen_value(row, "dest_region")
        if start is None or dest is None:
            continue
        max_flow = _num(_gen_value(row, "Line_Max_Flow_MW", 0.0), 0.0)
        reinforcement = (
            _num(_gen_value(row, "Line_Max_Reinforcement_MW", np.nan), 0.0) or 0.0
        )
        financial = _financial_attrs(row)
        # GenX provides no per-line Lifetime for transmission. Fall back to
        # the capital_recovery_period so Macro does not treat the line as a
        # 1-year asset (which would force premature retirement of new builds).
        if "lifetime" not in financial and "capital_recovery_period" in financial:
            financial["lifetime"] = financial["capital_recovery_period"]
        records.append(
            {
                "Type": TRANSMISSION_TYPE,
                "id": f"{start}_to_{dest}",
                "transmission_constraints--MaxCapacityConstraint": "TRUE",
                "max_capacity": max_flow + reinforcement,
                "transmission_origin": f"elec_{start}",
                "loss_fraction": _num(
                    _gen_value(row, "Line_Loss_Percentage", 0.0), 0.0
                ),
                "existing_capacity": max_flow,
                "annualized_investment_cost": _num(
                    _gen_value(row, "Line_Reinforcement_Cost_per_MWyr", np.nan)
                ),
                "commodity": "Electricity",
                "distance": _num(_gen_value(row, "distance_mile", np.nan)),
                "transmission_dest": f"elec_{dest}",
                **financial,
            }
        )
    return pd.DataFrame(records, columns=TRANSMISSION_COLUMNS)


# ---------------------------------------------------------------------------
# System (JSON + CSV) builders
# ---------------------------------------------------------------------------


def make_commodities_json(commodities: List[str]) -> dict:
    """Return commodities.json content for an ordered commodity list."""
    # Electricity and CO2 are always needed
    ordered = ["Electricity"]
    for c in commodities:
        if c not in ordered:
            ordered.append(c)
    for c in ("CO2",):
        if c not in ordered:
            ordered.append(c)
    return {"commodities": ordered}


def load_nsd_segments(settings: dict) -> tuple:
    """Load non-served-demand (VOLL) segments from the GenX demand-segments CSV.

    Maps each GenX demand segment to a Macro NSD segment, matching the price
    GenX itself derives: ``price_nsd[s] = Cost_of_Demand_Curtailment_per_MW[s]
    × Voll[1]`` (each segment's cost is a fraction of the base value of lost
    service). ``Voll`` is a single base value (the first non-null entry), and
    the ``$/MWh`` column is purely informational:

    - ``price_nsd[s]`` ← ``Cost_of_Demand_Curtailment_per_MW[s] * Voll[1]``
    - ``max_nsd[s]`` ← ``Max_Demand_Curtailment`` (fraction of demand)

    Falls back to the ``$/MWh`` column when the ``Voll`` / ``Cost_...``
    columns are absent, and to a single segment when no demand-segments file
    is configured. The fallback maximum curtailment fraction and value of
    lost service can be overridden from settings (``macro_default_max_nsd``,
    default 1, and ``macro_default_voll``, default $10000/MWh).

    Returns
    -------
    (max_nsd, price_nsd) : tuple of list[float]
    """
    default_max_nsd = float(settings.get("macro_default_max_nsd", 1))
    default_voll = float(settings.get("macro_default_voll", 10000.0))
    fn = settings.get("demand_segments_fn")
    if not fn:
        return [default_max_nsd], [default_voll]
    try:
        df = load_demand_segments(settings)
    except Exception:
        logger.warning(
            "Could not load demand segments file '%s'; "
            "using default single-segment VOLL.",
            fn,
        )
        return [default_max_nsd], [default_voll]

    max_col = next(
        (
            c
            for c in ("Max_Demand_Curtailment", "Max_Demand_Curtailment_MW")
            if c in df.columns
        ),
        None,
    )
    if max_col is None:
        logger.warning(
            "Demand segments file missing required columns; "
            "using default single-segment VOLL."
        )
        return [default_max_nsd], [default_voll]

    # GenX prices each segment as Cost_of_Demand_Curtailment_per_MW × Voll[1].
    # "Voll" is a single base value (first non-null entry); the per-segment
    # column holds the fraction of that base value served by each segment.
    voll_col = "Voll" if "Voll" in df.columns else None
    frac_col = (
        "Cost_of_Demand_Curtailment_per_MW"
        if "Cost_of_Demand_Curtailment_per_MW" in df.columns
        else None
    )
    if (
        voll_col is not None
        and frac_col is not None
        and not df[voll_col].dropna().empty
    ):
        voll = df[voll_col].dropna().iloc[0]
        price_series = pd.to_numeric(df[frac_col], errors="coerce") * float(voll)
    else:
        # Fallback: use the actual $/MWh column when present
        price_col = next(
            (c for c in ("$/MWh", "Cost_per_MWh") if c in df.columns), None
        )
        if price_col is None:
            logger.warning(
                "Demand segments file missing required columns; "
                "using default single-segment VOLL."
            )
            return [default_max_nsd], [default_voll]
        price_series = pd.to_numeric(df[price_col], errors="coerce")

    df = df.assign(_price=price_series)
    # Sort by descending cost (GenX convention: segment 1 = highest VOLL)
    df = df.sort_values("_price", ascending=False).reset_index(drop=True)
    max_nsd = [float(v) for v in df[max_col].tolist()]
    price_nsd = [float(v) for v in df["_price"].tolist()]
    return max_nsd, price_nsd


def make_nodes_json(
    settings: dict,
    demand_headers: Dict[str, str],
    fuel_supply_headers: Dict[str, Dict[str, str]],
    co2_sinks: List[dict],
    has_hydro: bool,
    stage_number: int = 1,
    has_ccs: bool = False,
) -> list:
    """Build the full nodes.json structure.

    Parameters
    ----------
    settings : dict
        PowerGenome settings (model_regions, zone_num_map).
    demand_headers : dict
        Mapping of region -> demand.csv header name.
    fuel_supply_headers : dict
        Mapping of commodity -> {region: fuel_prices.csv header}.
    co2_sinks : list of dict
        Sink node entries (id, cap tonnes or None for uncapped generic sink).
    has_hydro : bool
        Whether a hydro_source node should be added.
    stage_number : int
        Per-stage suffix used in the demand / fuel price timeseries paths.
    has_ccs : bool
        Whether any CCS (ThermalPowerCCS) asset exists; when true a location-less
        CO2Captured sink node (``co2_sink_injection``) is added so captured CO2
        flows out of the system (matching GenX, where captured emissions do not
        count toward the caps).
    """
    demand_path = f"system/demand_{stage_number}.csv"
    fuel_path = f"system/fuel_prices_{stage_number}.csv"
    nodes = []

    # Load VOLL / non-served-demand segments from the GenX demand-segments CSV
    max_nsd, price_nsd = load_nsd_segments(settings)

    # Electricity demand nodes (one block)
    electricity_global = {
        "time_interval": "Electricity",
        "max_nsd": max_nsd,
        "price_nsd": price_nsd,
        "constraints": {
            "BalanceConstraint": True,
            "MaxNonServedDemandConstraint": True,
            "MaxNonServedDemandPerSegmentConstraint": True,
        },
    }
    demand_instances = []
    for region, header in demand_headers.items():
        demand_instances.append(
            {
                "id": f"elec_{region}",
                "location": region,
                "demand": {"timeseries": {"path": demand_path, "header": header}},
            }
        )
    if demand_instances:
        nodes.append(
            {
                "type": "Electricity",
                "global_data": electricity_global,
                "instance_data": demand_instances,
            }
        )

    # Fuel supply nodes (one block per commodity, single-tier with location)
    for commodity, regions in sorted(fuel_supply_headers.items()):
        if not regions:
            continue
        instances = []
        for region in sorted(regions):
            instances.append(
                {
                    "id": f"{commodity}_{region}",
                    "location": region,
                    "supply": {
                        "segment1": {
                            "price": {
                                "timeseries": {
                                    "path": fuel_path,
                                    "header": fuel_supply_headers[commodity][region],
                                }
                            }
                        }
                    },
                }
            )
        nodes.append(
            {
                "type": commodity,
                "global_data": {
                    "time_interval": commodity,
                    "constraints": {"BalanceConstraint": True},
                },
                "instance_data": instances,
            }
        )

    # CO2 sink node(s)
    if co2_sinks:
        co2_instances = []
        for sink in co2_sinks:
            instance = {"id": sink["id"], "constraints": {"BalanceConstraint": False}}
            if sink.get("cap") is not None:
                instance["constraints"]["CO2CapConstraint"] = True
                instance["rhs_policy"] = {"CO2CapConstraint": sink["cap"]}
            else:
                instance["constraints"]["CO2CapConstraint"] = False
                instance["rhs_policy"] = {"CO2CapConstraint": 0}
            co2_instances.append(instance)
        nodes.append(
            {
                "type": "CO2",
                "global_data": {"time_interval": "CO2"},
                "instance_data": co2_instances,
            }
        )

    # hydro_source node (balance node for hydro inflow)
    if has_hydro:
        nodes.append(
            {
                "type": "Electricity",
                "global_data": {
                    "time_interval": "Electricity",
                    "constraints": {"BalanceConstraint": False},
                },
                "instance_data": [{"id": "hydro_source"}],
            }
        )

    # CO2Captured sink node shared by all CCS assets' captured-CO2 edges
    if has_ccs:
        nodes.append(
            {
                "type": "CO2Captured",
                "global_data": {
                    "time_interval": "CO2Captured",
                    "constraints": {"BalanceConstraint": False},
                },
                "instance_data": [{"id": CO2_CAPTURED_NODE}],
            }
        )

    return nodes


def make_timedata_json(
    demand_data: pd.DataFrame,
    commodities: List[str],
    has_period_map: bool,
    stage_number: int = 1,
) -> dict:
    """Build time_data.json from the (possibly reduced) demand data."""
    if demand_data is None or demand_data.empty:
        rep_periods = 1
        hours_per_subperiod = 8760
        total_hours = 8760
    else:
        rep_periods = (
            int(demand_data["Rep_Periods"].iloc[0])
            if "Rep_Periods" in demand_data.columns
            else 1
        )
        hours_per_subperiod = (
            int(demand_data["Timesteps_per_Rep_Period"].iloc[0])
            if "Timesteps_per_Rep_Period" in demand_data.columns
            else 8760
        )
        total_hours = (
            int(demand_data["Sub_Weights"].sum())
            if "Sub_Weights" in demand_data.columns
            else hours_per_subperiod
        )

    time_data = {
        "HoursPerSubperiod": {c: hours_per_subperiod for c in commodities},
        "HoursPerTimeStep": {c: 1 for c in commodities},
        "NumberOfSubperiods": rep_periods,
        "TotalHoursModeled": total_hours,
    }
    if has_period_map and rep_periods > 1:
        time_data["SubPeriodMap"] = {"path": f"system/Period_map_{stage_number}.csv"}
    return time_data


def make_macro_settings_json(settings: Optional[dict] = None) -> dict:
    """Return macro_settings.json content matching the example set.

    Each flag can be overridden from settings (``macro_constraint_scaling``,
    ``macro_write_subcommodities``, ``macro_auto_create_nodes``,
    ``macro_auto_create_locations``); defaults match the
    ``multisector_3zone_simpleCSVinputs`` example set.
    """
    settings = settings or {}
    return {
        "ConstraintScaling": bool(settings.get("macro_constraint_scaling", True)),
        "WriteSubcommodities": bool(settings.get("macro_write_subcommodities", True)),
        "AutoCreateNodes": bool(settings.get("macro_auto_create_nodes", False)),
        "AutoCreateLocations": bool(settings.get("macro_auto_create_locations", True)),
    }


def make_system_data_json(stages: List[int], assets_folder: str = "assets") -> dict:
    """Return multistage system_data.json content pointing at per-stage folders.

    Follows the layout produced by ``GenX_to_Macro``'s multistage driver: one
    ``case`` entry per stage (sorted by stage number), each referencing its own
    assets / time_data / nodes paths, plus a shared settings path.
    """
    case_entries = []
    for stage in sorted(stages):
        case_entries.append(
            {
                "commodities": {"path": "system/commodities.json"},
                "locations": {"path": "system/locations.json"},
                "settings": {"path": "settings/macro_settings.json"},
                "assets": {"path": f"{assets_folder}/assets_{stage}"},
                "time_data": {"path": f"system/time_data_{stage}.json"},
                "nodes": {"path": f"system/nodes_{stage}.json"},
            }
        )
    return {
        "case": case_entries,
        "settings": {"path": "settings/case_settings.json"},
    }


def _planning_period_lengths(settings: dict) -> list:
    """Return the per-period lengths (years) from PowerGenome period settings.

    Derives one length per planning period from ``model_periods`` (a list of
    ``(first_year, last_year)`` tuples) or from the paired
    ``model_first_planning_year`` / ``model_year`` lists; each span is
    inclusive (``last - first + 1``). Returns an empty list when no planning
    period information is present.
    """
    if not settings:
        return []
    if settings.get("model_periods"):
        periods = settings["model_periods"]
        # A single period may be stored as a flat [first, last]
        if periods and not isinstance(periods[0], (list, tuple)):
            periods = [periods]
        return [max(int(last) - int(first) + 1, 1) for first, last in periods]
    first = settings.get("model_first_planning_year")
    last = settings.get("model_year")
    if first is None or last is None:
        return []
    if not isinstance(first, (list, tuple)):
        first = [first]
    if not isinstance(last, (list, tuple)):
        last = [last]
    return [
        max(int(last_year) - int(first_year) + 1, 1)
        for first_year, last_year in zip(first, last)
    ]


def make_case_settings_json(
    n_stages: int,
    settings: Optional[dict] = None,
    period_lengths: Optional[list] = None,
) -> dict:
    """Return the settings/case_settings.json content for a multistage case.

    ``PeriodLengths`` has one entry per stage. Each period's length is
    derived from PowerGenome's planning period definitions
    (``model_periods`` or the paired ``model_first_planning_year`` /
    ``model_year``; each span inclusive). An explicit ``period_lengths``
    list takes precedence, then an explicit ``macro_period_lengths`` setting,
    then derivation, then a fallback of 1 year per stage. Discount rate and
    solution algorithm mirror GenX_to_Macro's multistage settings file.

    Overridable from settings: ``macro_period_lengths`` (list, one entry per
    stage), ``macro_discount_rate`` (float), and ``macro_solution_algorithm``
    (e.g. ``"Monolithic"`` or ``"Nested"``).
    """
    settings = settings or {}
    if period_lengths is None:
        period_lengths = settings.get("macro_period_lengths")
    if period_lengths is None:
        derived = _planning_period_lengths(settings)
        period_lengths = (derived + [1] * n_stages)[:n_stages]
    return {
        "PeriodLengths": list(period_lengths),
        "DiscountRate": settings.get("macro_discount_rate", 0.045),
        "SolutionAlgorithm": settings.get("macro_solution_algorithm", "Monolithic"),
    }


def make_demand_csv(demand_data: pd.DataFrame) -> pd.DataFrame:
    """Return the demand.csv contents (Time_Index + Demand_MW_z* columns)."""
    if demand_data is None or demand_data.empty:
        return pd.DataFrame()
    demand_cols = [c for c in demand_data.columns if str(c).startswith("Demand_MW_z")]
    cols = []
    if "Time_Index" in demand_data.columns:
        cols.append("Time_Index")
    cols.extend(demand_cols)
    return demand_data[cols].reset_index(drop=True)


def make_availability_csv(gen_df, gen_variability, time_index=None) -> pd.DataFrame:
    """Return availability.csv with a column for every VRE/hydro/mustrun resource.

    Every VRE, hydro, and must-run resource gets a column so the ``*--timeseries
    --header`` availability references in the asset files always resolve. When a
    resource has no profile in ``gen_variability`` (or its profile is constant),
    the column is filled with a constant 1.0.
    """
    if gen_df is None or gen_df.empty:
        return pd.DataFrame()
    resources: List[str] = []
    for tag in ("VRE", "HYDRO", "MUST_RUN"):
        if tag in gen_df.columns:
            for resource in gen_df.loc[gen_df[tag] > 0, "Resource"]:
                if resource not in resources:
                    resources.append(resource)
    if not resources:
        return pd.DataFrame()

    if time_index is None:
        if gen_variability is not None and "Time_Index" in gen_variability.columns:
            time_index = gen_variability["Time_Index"]
        elif gen_variability is not None and len(gen_variability):
            time_index = pd.Series(range(1, len(gen_variability) + 1))
        else:
            time_index = pd.Series(range(1, 8761))

    out = pd.DataFrame({"Time_Index": time_index.to_numpy()})
    for resource in resources:
        if gen_variability is not None and resource in gen_variability.columns:
            out[resource] = gen_variability[resource].to_numpy()
        else:
            out[resource] = 1.0
    return out


def _fuel_price_by_fullname(fuels: pd.DataFrame, fuel_name: str):
    """Return (price series, co2 content) for a full fuel name from the fuels table."""
    if fuels is None or fuel_name not in fuels.columns:
        return None, None
    col = fuels[fuel_name]
    # Row 0 is CO2 content; rows 1.. are prices
    co2_content = col.iloc[0]
    prices = col.iloc[1:]
    return prices.astype(float), float(co2_content)


def make_fuel_prices_csv(
    fuels: pd.DataFrame,
    thermal_resources: pd.DataFrame,
    time_index: pd.Series,
    default_price: float = 0.0,
) -> pd.DataFrame:
    """Build fuel_prices.csv with per-commodity/region price columns.

    Prices are converted from $/MMBtu to $/MWh by dividing by the
    MMBtu->MWh conversion factor. When a fuel is not present in the fuels
    table, its price falls back to ``default_price`` in $/MWh (a constant;
    overridable from settings via ``macro_default_fuel_price``).
    """
    cols = {"Time_Index": time_index.to_numpy()}
    commodity_regions: Dict[str, Dict[str, tuple]] = {}
    for _, row in thermal_resources.iterrows():
        commodity = _fuel_commodity(_gen_value(row, "Fuel"))
        if commodity is None:
            continue
        region = _gen_value(row, "region")
        fuel = _gen_value(row, "Fuel")
        if region not in commodity_regions.setdefault(commodity, {}):
            prices, _ = _fuel_price_by_fullname(fuels, fuel)
            if prices is None:
                # fall back to a constant price ($/MWh) if the fuel is not in
                # the fuels table; convert to $/MMBtu so the shared conversion
                # below yields ``default_price`` $/MWh again
                prices = pd.Series(
                    float(default_price) * CONV_MMBTU_TO_MWH,
                    index=range(len(time_index)),
                )
            commodity_regions[commodity][region] = (fuel, prices)

    for commodity, regions in commodity_regions.items():
        for region, (fuel, prices) in regions.items():
            # prices rows are per hour; convert MMBtu -> MWh
            converted = prices.to_numpy() / CONV_MMBTU_TO_MWH
            header = f"{commodity}_{region}"
            cols[header] = converted[: len(time_index)]
    return pd.DataFrame(cols)


def make_period_map_csv(period_map: pd.DataFrame) -> pd.DataFrame:
    """Convert the PG time_series_mapping to the Macro Period_map.csv format."""
    if period_map is None or period_map.empty:
        return pd.DataFrame()
    out = period_map.copy()
    if "Rep_Period_Index" in out.columns and "Rep_Period" not in out.columns:
        out["Rep_Period"] = out["Rep_Period_Index"]
    keep = [
        c
        for c in ("Period_Index", "Rep_Period", "Rep_Period_Index")
        if c in out.columns
    ]
    return out[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _co2_sinks_for(
    gen_df: pd.DataFrame, settings: dict, co2_cap: pd.DataFrame
) -> List[dict]:
    """Build the CO2 sink node entries and tag thermal generators' ``co2_sink``.

    Emulates GenX ``co2_cap!`` mass-cap semantics. Each cap ``p`` in the (GenX
    ``CO2_cap.csv``) table has a ``CO_2_Cap_Zone_p`` flag column and a
    ``CO_2_Max_Mtons_p`` column. The cap binds across the flagged zones, with RHS

    .. math::
       \\sum_{z \\in \\mathcal{Z}^{cap}_p} \\text{CO2MaxMtons}_{z,p} \\times 10^6

    We create one sink node ``co2_sink_<p>`` per cap carrying that RHS through
    its ``CO2CapConstraint``, and point every thermal generator located in a
    capped zone at the first cap that flags its zone. Generators in zones that
    belong to no cap keep the generic uncapped ``co2_sink``.

    Returns the list of ``(id, cap)`` entries for the CO2 node block in nodes.json.
    """
    sinks: List[dict] = [{"id": "co2_sink", "cap": None}]
    if co2_cap is None or co2_cap.empty:
        return sinks
    zone_num_map = settings.get("zone_num_map", {})
    zone_to_region = {int(zone): reg for reg, zone in zone_num_map.items()}
    max_cols = sorted(
        (c for c in co2_cap.columns if str(c).startswith("CO_2_Max_Mtons_")),
        key=lambda c: str(c).rsplit("_", 1)[-1],
    )
    zone_to_sink: Dict[int, str] = {}
    for max_col in max_cols:
        cap_num = str(max_col).rsplit("_", 1)[-1]
        flag_col = f"CO_2_Cap_Zone_{cap_num}"
        if flag_col not in co2_cap.columns:
            continue
        try:
            flagged = co2_cap[co2_cap[flag_col].astype(float) == 1]
        except (TypeError, ValueError):
            flagged = co2_cap[co2_cap[flag_col] == 1]
        if flagged.empty:
            continue
        rhs = sum(_num(value, 0.0) for value in flagged[max_col].tolist()) * 1e6
        sink_id = f"co2_sink_{cap_num}"
        sinks.append({"id": sink_id, "cap": rhs})
        for _, row in flagged.iterrows():
            zone = row.get("Network_zones")
            if pd.isna(zone):
                continue
            # First cap that flags a zone wins (caps are zone-disjoint in practice).
            zone_to_sink.setdefault(int(zone), sink_id)

    if "co2_sink" not in gen_df.columns:
        gen_df["co2_sink"] = "co2_sink"
    for zone, sink_id in zone_to_sink.items():
        region = zone_to_region.get(zone)
        if region is None:
            continue
        mask = (gen_df["region"] == region) & (gen_df["THERM"] > 0)
        gen_df.loc[mask, "co2_sink"] = sink_id
    return sinks


class MacroCaseBuilder:
    """Accumulate per-stage Macro simpleCSVinputs output for one case.

    A single Macro case (the "multistage" layout from GenX_to_Macro) is made of
    one ``case`` entry per planning period (stage). Because PowerGenome's main
    loop iterates periods out of order for a given case, this class buffers the
    stage data and only writes the shared, case-level files on ``finalize``.
    """

    def __init__(self, case_root):
        self.case_root = Path(case_root)
        self.stage_numbers: List[int] = []
        self.commodities: List[str] = []
        self._stage_data = {}  # stage_number -> (case_year_data, settings)

    @property
    def _system_folder(self):
        return self.case_root / "system"

    @property
    def _settings_folder(self):
        return self.case_root / "settings"

    def add_stage(self, stage_number, case_year_data, settings) -> None:
        """Buffer one planning period (stage) for writing on finalize."""
        if stage_number in self._stage_data:
            raise ValueError(
                f"Stage {stage_number} already added to Macro case {self.case_root}"
            )
        self.stage_numbers.append(stage_number)
        self._stage_data[stage_number] = (case_year_data, settings)

    def finalize(self) -> None:
        """Write all per-stage files and the case-level system_data.json."""
        stages = sorted(self.stage_numbers)
        system_folder = self._system_folder
        settings_folder = self._settings_folder
        system_folder.mkdir(parents=True, exist_ok=True)
        settings_folder.mkdir(parents=True, exist_ok=True)

        if not stages:
            return

        # Buffer commodities across stages (shared commodities.json).
        has_ccs = False
        for stage in stages:
            case_year_data, stage_settings = self._stage_data[stage]
            gen_df = case_year_data.get("gen_data")
            if gen_df is None or gen_df.empty:
                continue
            if (
                "CO2_Capture_Fraction" in gen_df.columns
                and (gen_df["CO2_Capture_Fraction"].fillna(0) > 0).any()
            ):
                has_ccs = True
            fuels = case_year_data.get("fuels")
            for file_name, commodity, _ in make_thermal_csvs(
                gen_df, stage_settings, fuels
            ):
                if commodity and commodity not in self.commodities:
                    self.commodities.append(commodity)

        # Electricity and CO2 are always present in a Macro case.
        for name in ("Electricity", "CO2"):
            if name not in self.commodities:
                self.commodities.append(name)
        # CCS assets add a CO2Captured commodity so Macro builds a
        # ``time_data[:CO2Captured]`` timeseries (via the CO2 supertype fallback).
        if has_ccs and "CO2Captured" not in self.commodities:
            self.commodities.append("CO2Captured")

        # Per-stage files.
        for stage in stages:
            case_year_data, settings = self._stage_data[stage]
            self._write_stage(stage, case_year_data, settings)

        # Shared / case-level files.
        with open(system_folder / "commodities.json", "w") as f:
            json.dump(make_commodities_json(self.commodities), f, indent=4)
        locations = self._stage_data[stages[0]][1].get("model_regions", [])
        with open(system_folder / "locations.json", "w") as f:
            json.dump({"locations": locations}, f, indent=4)
        # Case-level settings are written from the first stage's settings
        # (all stages share the same scenario settings object).
        case_settings = self._stage_data[stages[0]][1]
        # Derive each stage's period length from its own planning years
        # (model_first_planning_year / model_year), unless the user supplied
        # an explicit macro_period_lengths override.
        if case_settings.get("macro_period_lengths") is None:
            stage_lengths = []
            for stage in stages:
                stage_lengths_for = _planning_period_lengths(self._stage_data[stage][1])
                stage_lengths.append(stage_lengths_for[-1] if stage_lengths_for else 1)
        else:
            stage_lengths = None
        with open(settings_folder / "macro_settings.json", "w") as f:
            json.dump(make_macro_settings_json(case_settings), f, indent=2)
        with open(settings_folder / "case_settings.json", "w") as f:
            json.dump(
                make_case_settings_json(
                    len(stages), case_settings, period_lengths=stage_lengths
                ),
                f,
                indent=2,
            )
        with open(self.case_root / "system_data.json", "w") as f:
            json.dump(
                make_system_data_json(stages, assets_folder="assets"), f, indent=4
            )

    def _write_stage(self, stage_number, case_year_data, settings) -> None:
        """Write every file belonging to a single stage."""
        system_folder = self._system_folder
        assets_folder = self.case_root / "assets" / f"assets_{stage_number}"
        assets_folder.mkdir(parents=True, exist_ok=True)

        gen_df = case_year_data.get("gen_data")
        gen_variability = case_year_data.get("gen_variability")
        demand_data = case_year_data.get("demand_data")
        network = case_year_data.get("network")
        fuels = case_year_data.get("fuels")
        period_map = case_year_data.get("period_map")
        co2_cap = case_year_data.get("co2_cap")

        # ---- assets ----
        thermal_files = []
        thermal_resources = pd.DataFrame()
        co2_sinks = []
        if gen_df is not None and not gen_df.empty:
            # Pin CO2 sinks (also tags in-region thermal gens to capped sinks)
            co2_sinks = _co2_sinks_for(gen_df, settings, co2_cap)
            thermal_resources = gen_df[gen_df["THERM"] > 0]
            thermal_files = make_thermal_csvs(gen_df, settings, fuels)

            vre_df = make_vre_csv(gen_df, stage_number=stage_number)
            if not vre_df.empty:
                vre_df.to_csv(assets_folder / "vre.csv", index=False)
            stor_df = make_storage_csv(gen_df)
            if not stor_df.empty:
                stor_df.to_csv(assets_folder / "electricity_stor.csv", index=False)
            hydro_df = make_hydro_csv(gen_df, stage_number=stage_number)
            if not hydro_df.empty:
                hydro_df.to_csv(assets_folder / "hydropower.csv", index=False)
            mustrun_df = make_mustrun_csv(gen_df, stage_number=stage_number)
            if not mustrun_df.empty:
                mustrun_df.to_csv(assets_folder / "mustrun.csv", index=False)
        else:
            co2_sinks = [{"id": "co2_sink", "cap": None}]

        for file_name, _, df in thermal_files:
            df.to_csv(assets_folder / file_name, index=False)

        if network is not None and not network.empty:
            tx_df = make_powerlines_csv(network)
            if not tx_df.empty:
                tx_df.to_csv(assets_folder / "powerlines.csv", index=False)

        # ---- system CSVs ----
        if demand_data is not None and not demand_data.empty:
            demand_df = make_demand_csv(demand_data)
            if not demand_df.empty:
                demand_df.to_csv(
                    system_folder / f"demand_{stage_number}.csv", index=False
                )

        time_index = None
        if gen_variability is not None and "Time_Index" in gen_variability.columns:
            time_index = gen_variability["Time_Index"]
        elif demand_data is not None and "Time_Index" in demand_data.columns:
            time_index = demand_data["Time_Index"]
        if time_index is None:
            time_index = pd.Series(range(1, 8761))

        availability_df = make_availability_csv(
            gen_df, gen_variability, time_index=time_index
        )
        if not availability_df.empty:
            availability_df.to_csv(
                system_folder / f"availability_{stage_number}.csv", index=False
            )
        fuel_prices_df = make_fuel_prices_csv(
            fuels,
            thermal_resources,
            time_index,
            default_price=settings.get("macro_default_fuel_price", 0.0),
        )
        if not fuel_prices_df.empty:
            fuel_prices_df.to_csv(
                system_folder / f"fuel_prices_{stage_number}.csv",
                index=False,
                float_format="%.6f",
            )

        period_map_df = make_period_map_csv(period_map)
        has_period_map = not period_map_df.empty
        if has_period_map:
            period_map_df.to_csv(
                system_folder / f"Period_map_{stage_number}.csv", index=False
            )

        # ---- demand header / fuel supply header maps ----
        zone_num_map = settings.get("zone_num_map", {})
        demand_headers: Dict[str, str] = {}
        for region in settings.get("model_regions", []):
            if demand_data is None or demand_data.empty:
                break
            zone = zone_num_map.get(region)
            header = f"Demand_MW_z{zone}"
            if any(c == header for c in demand_data.columns):
                demand_headers[region] = header

        fuel_supply_headers: Dict[str, Dict[str, str]] = {}
        if not thermal_resources.empty:
            for _, row in thermal_resources.iterrows():
                commodity = _fuel_commodity(_gen_value(row, "Fuel"))
                region = _gen_value(row, "region")
                if commodity is None or region is None:
                    continue
                fuel_supply_headers.setdefault(commodity, {})[
                    region
                ] = f"{commodity}_{region}"

        co2_sinks = co2_sinks or [{"id": "co2_sink", "cap": None}]

        has_hydro = bool(
            gen_df is not None
            and "HYDRO" in gen_df.columns
            and (gen_df["HYDRO"] > 0).any()
        )
        has_ccs = bool(
            gen_df is not None
            and "CO2_Capture_Fraction" in gen_df.columns
            and (gen_df["CO2_Capture_Fraction"].fillna(0) > 0).any()
        )

        # ---- per-stage JSON files ----
        with open(system_folder / f"nodes_{stage_number}.json", "w") as f:
            json.dump(
                {
                    "nodes": make_nodes_json(
                        settings,
                        demand_headers,
                        fuel_supply_headers,
                        co2_sinks,
                        has_hydro,
                        stage_number=stage_number,
                        has_ccs=has_ccs,
                    )
                },
                f,
                indent=4,
            )
        with open(system_folder / f"time_data_{stage_number}.json", "w") as f:
            json.dump(
                make_timedata_json(
                    demand_data,
                    self.commodities,
                    has_period_map,
                    stage_number=stage_number,
                ),
                f,
                indent=4,
            )
