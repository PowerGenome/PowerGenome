# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog (<https://keepachangelog.com/en/1.0.0/>),
and this project adheres to Semantic Versioning.

## [Unreleased]

### Added

- weather_year filter support for renewable generation profiles (tidy format). Accepts a single int or a list of ints; when multiple years are provided, profiles are concatenated and a continuous per-site time_index is rebuilt.
- weather_year filter support for hourly demand profiles. When present, demand is filtered to the requested year and the per-region time_index is rebuilt to be sequential starting at 1.
- Comprehensive distributed generation (DG) test suite covering capacity interpolation/extrapolation, multi-weather-year profiles, aggregation, timezone shifting, and hourly generation.
- Interpolation/extrapolation summary logging for DG capacity: single consolidated message listing regions interpolated, backward/forward extrapolated, exact matches, and missing.
- Partial-year fill for DG capacity: when some regions have the requested year and others do not, interpolate/extrapolate only the missing regions.
- Auto-generation of regional cost multiplier mappings: `cost_multiplier_region_map` and `cost_multiplier_technology_map` are now optional. When not provided, the system automatically creates mappings using substring matching between new resource names and regional cost factor technologies.
- Validation that all technologies in `new_resources` are covered by regional cost corrections with appropriate warnings/info messages.
- Support for aggregated regions in regional cost multipliers: when a model region aggregates multiple base regions, the average cost multiplier across base regions is used.

### Changed

- BREAKING: Standardized renewable generation profile inputs to tidy format only. Legacy wide-format (one column per site) is no longer supported. Tidy schema is site_id, time_index, value, and optional weather_year.
- Profile IO refactored to use the centralized DataManager loader (DNF filters + column projection) to avoid full-file reads of large tidy files; no pre-scan is performed. A warning is issued if requested weather_years are unavailable.
- Site identifiers are no longer coerced to strings during profile loading and clustering; native types are preserved end-to-end.
- Renewable cluster cache keys now include weather_year to avoid cross-year reuse.
- DG profiles timezone offset semantics: a negative `tz_offset` now shifts earlier hours later in the array (e.g., `tz_offset = -2` makes hour 3 match the original hour 1).
- DG profiles across multiple weather years now build a unique sequential `time_index` across (weather_year, time_index) pairs and reset the index to start at 1.
- DG profile aggregation uses capacity-weighted averages and requires a `model_year`. Aggregation now validates that capacity exists for all declared component regions.
- Hourly DG generation now passes `year` into profile loading to ensure correct capacity-weighted aggregation without relying on global settings state.
- Regional cost multiplier application now supports automatic averaging for aggregated regions and provides better logging for technologies without explicit mappings.

### Deprecated

- None.

### Removed

- Support for wide-format generation profile files (CSV/Parquet with one column per site). All generation profiles must now be in tidy format.

### Security

- None.
