# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog (<https://keepachangelog.com/en/1.0.0/>),
and this project adheres to Semantic Versioning.

## [Unreleased]

### Added

- Simplified capacity reserve credit specification with `capacity_reserve_values` and auto-expansion. Users can now specify technology credits in a flat format (single constraint) or nested format (multiple constraints) that automatically populate `regional_tag_values`, reducing configuration boilerplate for complex multi-constraint systems. Explicit `regional_tag_values` entries take precedence, allowing mixed automatic and manual specification.
- weather_year filter support for renewable generation profiles (tidy format). Accepts a single int or a list of ints; when multiple years are provided, profiles are concatenated and a continuous per-site time_index is rebuilt.
- weather_year filter support for hourly demand profiles. When present, demand is filtered to the requested year and the per-region time_index is rebuilt to be sequential starting at 1.
- Supplemental demand (`supplemental_demand_table`) region names may now be either a base region or a model region; base-region names are mapped to their aggregated model region automatically.
- Supplemental demand coverage validation: when the supplemental table has a `weather_year` column and the load data's weather years are known, every weather year must be covered by a specific-year row or an `all` row, otherwise a descriptive error is raised.
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

- File hashing for renewable cluster cache keys (`calculate_file_hash`) is now memoized within a top-level run and uses 1 MB read blocks. Previously each of the 16+ regions re-hashed the full multi-GB profile parquet files, spending >80% of a warm-cache pipeline run (~4.4 min of a 5.5 min run) re-reading tens of gigabytes. Hashes are keyed on path + mtime + size and are reset at the start of each `run_powergenome` invocation so modified files are still detected.
- Large tidy profile reads (renewable resource profiles, demand) now reshape to wide inside DuckDB (`read_tidy_profiles_wide`) instead of loading the full tidy file into pandas and calling `DataFrame.pivot`. For single weather-year loads an ordered single-thread scan writes directly into a pre-allocated NumPy plate (bounded peak memory, preserves the source value dtype such as `float32`); multi-year concatenation and non-integer site ids fall back to a SQL `PIVOT` that is chunked by site (default 5000 sites per slice) so the fallback's peak memory is bounded too (an unbounded PIVOT measured >50 GB for ~69k sites). This keeps peak memory well under ~8 GB and removes the superlinear pivot cost for tens of thousands of sites.
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

- Supplemental demand is now applied at the base load-data stage (long format) inside `make_load_curves`, before per-weather-year hours are renumbered 1..N and before base regions are aggregated. The old wide-format block-tiling approach (which assumed every weather year had `hours_per_year` hours) has been removed; `weather_year: all` rows now expand to one copy per weather year actually present in the load data, so leap years and other unequal-length weather years are handled correctly.
- Supplemental demand is no longer applied at the end of `make_final_load_curves` for the standard load pipeline; it is applied inside `make_load_curves` instead. The user-supplied WIDE load path (`load_usr_demand_profiles`) still gets supplemental demand applied in wide format, but weather-year-specific rows are rejected there with a descriptive error.

### Deprecated

- `transmission_investment_cost` (spur/offshore/tx mileage-based logic) is deprecated. Provide `interconnect_capex_mw` instead. Legacy logic will emit warnings and be removed in a future release.

### Removed

- Support for wide-format generation profile files (CSV/Parquet with one column per site). All generation profiles must now be in tidy format.

### Security

- None.
