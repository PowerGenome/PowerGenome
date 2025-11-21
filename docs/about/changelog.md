# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to Semantic Versioning.

For the complete, detailed changelog, see the [CHANGELOG.md](https://github.com/PowerGenome/PowerGenome/blob/master/CHANGELOG.md) file in the repository.

## [Unreleased]

### Added

- **Multi-weather-year support**: Filter renewable generation profiles and hourly demand by weather year(s)
- **Comprehensive DG test suite**: Coverage for capacity interpolation/extrapolation, multi-weather-year profiles, aggregation
- **Auto-generated cost multiplier mappings**: Optional `cost_multiplier_region_map` and `cost_multiplier_technology_map` with automatic substring matching
- **Partial-year DG capacity fill**: Interpolate/extrapolate only missing regions when some have the requested year
- **Regional cost aggregation**: Automatic averaging of cost multipliers for aggregated model regions

### Changed

- **BREAKING**: Standardized renewable generation profiles to tidy format only (no more wide format)
  - Tidy schema: `site_id`, `time_index`, `value`, optional `weather_year`
- **Profile IO refactored**: Uses centralized DataManager with DNF filters
- **DG timezone semantics**: Negative `tz_offset` shifts earlier hours later in array
- **Multi-year DG profiles**: Build sequential `time_index` across weather years
- **DG aggregation**: Uses capacity-weighted averages and requires `model_year`

### Removed

- Support for wide-format generation profile files (CSV/Parquet with one column per site)

## Version History

For detailed version history, including patch releases and full changelogs, visit:

- [GitHub Releases](https://github.com/PowerGenome/PowerGenome/releases)
- [Full CHANGELOG](https://github.com/PowerGenome/PowerGenome/blob/master/CHANGELOG.md)

## Migration Guides

### Migrating to v0.8.0

#### Renewable Profile Format

**Old (Wide Format)**:

```csv
time_index,site_001,site_002,site_003
1,0.45,0.52,0.38
2,0.42,0.48,0.35
```

**New (Tidy Format)**:

```csv
site_id,time_index,value,weather_year
site_001,1,0.45,2012
site_001,2,0.42,2012
site_002,1,0.52,2012
site_002,2,0.48,2012
```

See [Distributed Generation](../explanation/distributed-generation.md) for complete migration details.

## Contributing

Found a bug or have a feature request? See our [Contributing](contributing.md) guide to get involved!
