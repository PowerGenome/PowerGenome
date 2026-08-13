# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog (<https://keepachangelog.com/en/1.0.0/>),
and this project adheres to Semantic Versioning.

## [Unreleased]

### Added

- Optional MacroEnergy.jl (Macro) simpleCSV input output mode. When enabled (via the `macro_output: true` setting or the `run_powergenome` `--macro` flag), PowerGenome writes Macro-format case inputs (only the `simpleCSVinputs` format) instead of GenX `Inputs/Inputs_pN` files. Outputs match the `macroenergy/MacroEnergyExamples.jl/examples/multisector_3zone_simpleCSVinputs` structure: a top-level `system_data.json`, `assets/` asset CSVs (thermal, VRE, storage, hydro, must-run, transmission), and `system/` + `settings/` support files (demand, availability, fuel prices, period map, locations/commodities/nodes/time data/macro settings). The semantic GenX-to-Macro mapping follows `EmilDimanchev/GenX_to_Macro`; cross-sector assets (hydrogen, liquid fuels) are not yet emitted.
- Macro CO2/CCS support. Generators with a `CO2_Capture_Fraction` > 0 (e.g. CCS-equipped NGCC) are written as `ThermalPowerCCS` assets that split emissions into a residual flow to the capped CO2 sink (`emission_rate = (1 - capture_fraction) x full rate`, matching GenX's post-capture `CO2` coefficient) and a captured flow on `edges--co2_captured_edge--end_vertex=co2_sink_injection` (a location-less `CO2Captured` sink node, so captured CO2 does not count toward the caps — mirroring GenX's captured-emissions handling). CO2 caps are now ingested from GenX's mass-cap schema (`CO_2_Cap_Zone_<n>` flags + `CO_2_Max_Mtons_<n>`), creating one `co2_sink_<n>` per cap with RHS = sum of flagged-zone maxima and pointing each in-zone thermal generator at the cap that flags it.
- Multistage (multi-period) Macro simpleCSV output. In a multi-period PowerGenome case, each planning period is written as one Macro stage, matching the multistage `GenX_to_Macro` conversion (`lbonaldo/GenX_to_Macro`, `lb/multistage` branch): `system_data.json` contains a `case` array with one entry per period (`assets/assets_N/`, `system/time_data_N.json`, `system/nodes_N.json`, etc., sorted by period), plus a `settings/case_settings.json` with `PeriodLengths` (one entry per stage), `DiscountRate`, and `SolutionAlgorithm`. Financial attributes (`wacc`, `capital_recovery_period`, `lifetime`, `min_retired_capacity`) are emitted on thermal, VRE, storage, hydro, and must-run assets (read from the equivalent GenX columns when present).
- Macro hydro reservoir constraints now match GenX. The hydropower writer enables `StorageCapacityConstraint`, `StorageMaxDurationConstraint`, and `StorageChargeDischargeRatioConstraint` on the reservoir storage edge (bounding stored energy by `Hydro_Energy_to_Power_Ratio` x capacity, per GenX's `cHydroMaxEnergy` for known-capacity reservoirs) and `discharge_constraints--StorageDischargeLimitConstraint` on the discharge edge (mirroring GenX's `cHydroMaxOutflow`: discharge cannot exceed prior-hour storage), consistent with the reference `GenX_to_Macro` converter. This closes a parity gap where Macro shed far less unserved energy than GenX for hydro-heavy systems.
- Simplified capacity reserve credit specification with `capacity_reserve_values` and auto-expansion. Users can now specify technology credits in a flat format (single constraint) or nested format (multiple constraints) that automatically populate `regional_tag_values`, reducing configuration boilerplate for complex multi-constraint systems. Explicit `regional_tag_values` entries take precedence, allowing mixed automatic and manual specification.
- weather_year filter support for renewable generation profiles (tidy format). Accepts a single int or a list of ints; when multiple years are provided, profiles are concatenated and a continuous per-site time_index is rebuilt.
- weather_year filter support for hourly demand profiles. When present, demand is filtered to the requested year and the per-region time_index is rebuilt to be sequential starting at 1.
- Comprehensive distributed generation (DG) test suite covering capacity interpolation/extrapolation, multi-weather-year profiles, aggregation, timezone shifting, and hourly generation.
- Comprehensive error handling test suite for `_parse_interconnect_capex` covering all TypeError, ValueError, and KeyError paths (9 distinct error conditions validated).
- Interpolation/extrapolation summary logging for DG capacity: single consolidated message listing regions interpolated, backward/forward extrapolated, exact matches, and missing.
- Flexible interconnection cost setting `interconnect_capex_mw` supporting scalar, technology-only, region-only, region->technology, and technology->region nested mappings with shortest-first substring precedence.
- Automatic annuity calculation for existing rows that already have a non-zero `interconnect_capex_mw` but zero/blank `interconnect_annuity`.
- Partial-year fill for DG capacity: when some regions have the requested year and others do not, interpolate/extrapolate only the missing regions.
- Auto-generation of regional cost multiplier mappings: `cost_multiplier_region_map` and `cost_multiplier_technology_map` are now optional. When not provided, the system automatically creates mappings using substring matching between new resource names and regional cost factor technologies.
- Validation that all technologies in `new_resources` are covered by regional cost corrections with appropriate warnings/info messages.
- Support for aggregated regions in regional cost multipliers: when a model region aggregates multiple base regions, the average cost multiplier across base regions is used.
- Simplified fuel workflow: automatic aggregation of base region fuel prices for aggregated regions when `region_aggregations` is defined without `fuel_region_map`.
- Full fuel names automatically constructed as `{region}_{scenario}_{fuel}` (e.g., `CA_N_reference_coal`) when legacy mapping parameters are not provided.

### Changed

- Macro output: transmission `powerlines.csv` now includes the financial attributes `wacc`, `capital_recovery_period`, and `lifetime` when the GenX network dataframe carries `WACC`/`Capital_Recovery_Period` columns (written in multi-period mode). GenX provides no per-line transmission `Lifetime`, so `lifetime` falls back to `capital_recovery_period` to avoid Macro treating lines as 1-year assets. `min_retired_capacity` is also emitted when `Min_Retired_Cap_MW` is present.
- Macro output: VRE, storage-discharge, hydro-discharge, and must-run asset `capacity_size` are now written as 1.0 (matching the reference `GenX_to_Macro` converter, which hardcodes these to 1.0 because GenX treats them as continuous resources). Thermal `capacity_size` continues to use the generator's `Cap_Size`, as in the reference converter.
- Macro hydro output: `inflow_can_retire` is set from the generator's Can_Retire flag, `discharge_can_expand` from New_Build, and `storage_can_expand`/`storage_can_retire` from New_Build/Can_Retire ANDed with a known-capacity check (`Hydro_Energy_to_Power_Ratio > 0`). `storage_charge_discharge_ratio` is written as the constant 1.0 and `discharge_capacity_size` as 1.0, matching the reference converter.
- BREAKING: Standardized renewable generation profile inputs to tidy format only. Legacy wide-format (one column per site) is no longer supported. Tidy schema is site_id, time_index, value, and optional weather_year.
- Profile IO refactored to use the centralized DataManager loader (DNF filters + column projection) to avoid full-file reads of large tidy files; no pre-scan is performed. A warning is issued if requested weather_years are unavailable.
- Site identifiers are no longer coerced to strings during profile loading and clustering; native types are preserved end-to-end.
- Renewable cluster cache keys now include weather_year to avoid cross-year reuse.
- DG profiles timezone offset semantics: a negative `tz_offset` now shifts earlier hours later in the array (e.g., `tz_offset = -2` makes hour 3 match the original hour 1).
- DG profiles across multiple weather years now build a unique sequential `time_index` across (weather_year, time_index) pairs and reset the index to start at 1.
- DG profile aggregation uses capacity-weighted averages and requires a `model_year`. Aggregation now validates that capacity exists for all declared component regions.
- BREAKING: Refactored `calculate_transmission_inv_cost` and `add_transmission_inv_cost` to use explicit arguments instead of a monolithic `settings` dictionary. The function now extracts WACC and capital recovery years directly from dataframe columns (`wacc_real`, `cap_recovery_years`) for annuity calculations. Update call sites accordingly (see settings docs for examples). Legacy spur mileage logic only used when `interconnect_capex_mw` is not provided.
- Existing `interconnect_capex_mw` values are preserved (never overwritten); annuities are now computed for both newly assigned and pre-existing capex rows lacking `interconnect_annuity`.
- Hourly DG generation now passes `year` into profile loading to ensure correct capacity-weighted aggregation without relying on global settings state.
- Regional cost multiplier application now supports automatic averaging for aggregated regions and provides better logging for technologies without explicit mappings.
- Fuel price workflow simplified: legacy AEO mapping parameters (`fuel_series_scenario_names`, `fuel_series_names`, `fuel_series_region_names`, `fuel_region_map`) are now optional. When not provided, fuel prices are expected directly in the fuel price table for all base regions, and PowerGenome automatically averages base region prices for aggregated regions.

### Deprecated

- `transmission_investment_cost` (spur/offshore/tx mileage-based logic) is deprecated. Provide `interconnect_capex_mw` instead. Legacy logic will emit warnings and be removed in a future release.

### Removed

- Support for wide-format generation profile files (CSV/Parquet with one column per site). All generation profiles must now be in tidy format.

### Security

- None.
