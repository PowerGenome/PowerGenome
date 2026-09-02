# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to Semantic Versioning.

For the complete, detailed changelog, see the [CHANGELOG.md](https://github.com/PowerGenome/PowerGenome/blob/main/CHANGELOG.md) file in the repository.

## [Unreleased]

### Added

- **Multi-weather-year support**: Filter renewable generation profiles and hourly demand by weather year(s)
- **Comprehensive DG test suite**: Coverage for capacity interpolation/extrapolation, multi-weather-year profiles, aggregation
- **Flexible interconnection cost specification**: New `interconnect_capex_mw` setting replaces the old per-mile spur line system (`capacity_limit_spur_fn`)
  - Supports scalar, region-only, technology-only, and nested patterns
  - Case-insensitive substring matching with shortest-first precedence for technologies
  - Automatic annuity calculation for both new and existing resources
- **Auto-generated cost multiplier mappings**: Optional `cost_multiplier_region_map` and `cost_multiplier_technology_map` with automatic substring matching
- **Partial-year DG capacity fill**: Interpolate/extrapolate only missing regions when some have the requested year
- **Regional cost aggregation**: Automatic averaging of cost multipliers for aggregated model regions
- **Simplified fuel workflow**: Automatic aggregation of base region fuel prices for aggregated regions

### Changed

- **BREAKING**: Standardized renewable generation profiles to tidy format only (no more wide format)
  - Tidy schema: `site_id`, `time_index`, `value`, optional `weather_year`
- **BREAKING**: Refactored `calculate_transmission_inv_cost` and `add_transmission_inv_cost` to use explicit arguments
  - Functions now read WACC and capital recovery years from dataframe columns
  - Legacy spur mileage logic only activated when `interconnect_capex_mw` not provided
- **Profile IO refactored**: Uses centralized DataManager with DNF filters
- **DG timezone semantics**: Negative `tz_offset` shifts earlier hours later in array
- **Multi-year DG profiles**: Build sequential `time_index` across weather years
- **DG aggregation**: Uses capacity-weighted averages and requires `model_year`
- **Fuel workflow simplified**: Legacy mapping parameters (`fuel_series_*`, `fuel_region_map`) now optional
  - Fuel price tables should contain all base regions
  - Aggregated region prices calculated automatically by averaging base regions

### Deprecated

- **Spur line mileage system**: `transmission_investment_cost` with `capacity_limit_spur_fn` file deprecated in favor of `interconnect_capex_mw`
  - Legacy system still functional but emits warnings
  - Will be removed in a future release

### Removed

- Support for wide-format generation profile files (CSV/Parquet with one column per site)

## Version History

For detailed version history, including patch releases and full changelogs, visit:

- [GitHub Releases](https://github.com/PowerGenome/PowerGenome/releases)
- [Full CHANGELOG](https://github.com/PowerGenome/PowerGenome/blob/main/CHANGELOG.md)

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

#### Fuel Price Workflow

**Old (Legacy AEO with mappings)**:

Settings required multiple mapping parameters:

```yaml
fuel_data_year: 2025
fuel_series_scenario_names:
  reference: REF2025
fuel_series_names:
  coal: STC
  naturalgas: NG
fuel_series_region_names:
  pacific: PCF
fuel_region_map:
  pacific: [CA_N, CA_S]
```

Fuel price table had AEO codes:

```csv
fuel,region,year,price,scenario
STC,PCF,2030,2.5,REF2025
```

**New (Simplified with complete regional coverage)**:

Only basic parameters needed:

```yaml
fuel_scenarios:
  coal: reference
  naturalgas: reference
tech_fuel_map:
  Conventional Steam Coal: coal
  Natural Gas Fired Combined Cycle: naturalgas
fuel_price_table: fuel_prices.csv
```

Fuel price table contains **all base regions**:

```csv
fuel,region,year,price,scenario
coal,CA_N,2030,2.5,reference
coal,CA_S,2030,2.6,reference
naturalgas,CA_N,2030,4.2,reference
naturalgas,CA_S,2030,4.3,reference
```

PowerGenome automatically:

- Constructs full fuel names: `CA_N_reference_coal`
- Averages prices for aggregated regions (e.g., `CA` from `CA_N` + `CA_S`)

Legacy mapping parameters still supported for backward compatibility.

See [Fuel Settings](../reference/settings/fuels.md) for complete documentation.

#### Interconnection Costs

**Old (Per-mile spur line system)**:

Settings used mileage-based costs:

```yaml
transmission_investment_cost:
  spur:
    capex_mw_mile:
      CA_N: 3000
      CA_S: 3200
    wacc: 0.069
    investment_years: 60
```

Extra input file `capacity_limit_spur_fn` provided spur distances:

```csv
resource,region,spur_miles
offshore_wind_fixed,CA_N,45
solar_pv,CA_N,12
```

**New (Direct per-MW cost specification)**:

Simplified settings with flexible patterns:

```yaml
interconnect_capex_mw:
  default: 120000
  offshore_wind: 200000  # substring match, case-insensitive
  CA_N:
    battery: 60000
    solar: 105000
```

No external file needed. Supports:

- Scalar (applies to all resources)
- Region-only or technology-only mappings
- Nested region→technology or technology→region patterns
- Shortest-first substring precedence for technology matching

Annuities calculated automatically using plant financial parameters (`wacc_real`, `cap_recovery_years`).

Legacy spur system still supported for backward compatibility but will be removed in a future release.

See [Settings Documentation](../reference/settings/index.md) for complete `interconnect_capex_mw` specification.

## Contributing

Found a bug or have a feature request? See our [Contributing](contributing.md) guide to get involved!
