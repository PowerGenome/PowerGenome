# Model Definition Settings

These core parameters define the overall model scope, planning periods, and currency adjustments.

## Planning Years

### `model_periods`

**Type**: List of 2-element lists `[[first_year, last_year], ...]`
**Required**: Yes

Defines planning periods as pairs of `[first_year, last_year]`. The last year is used for demand profiles and technology cost vintages; costs are averaged over the full span of each period.

```yaml
# Single period
model_periods:
  - [2025, 2030]

# Multi-period study
model_periods:
  - [2025, 2030]
  - [2031, 2040]
  - [2041, 2050]
```

This creates planning periods:

- 2025–2030 (6 years)
- 2031–2040 (10 years)
- 2041–2050 (10 years)

!!! note "How periods are used"

    - **Demand**: Uses the last year of each period (e.g., 2030 demand for 2025–2030)
    - **New-build costs**: Averaged across the full period (e.g., 2025–2030 capital costs are mean values over those years)
    - **Retirements**: A generator is retired if its expected retirement year falls before the end of a period

### Legacy format

Planning periods can also be defined as two separate parallel lists:

```yaml
model_year: [2030, 2040, 2050]
model_first_planning_year: [2025, 2031, 2041]
```

Each position in `model_year` is paired with the corresponding entry in `model_first_planning_year`. `model_periods` is preferred because it defines each period in one place, which reduces the risk of list-length mismatches.

## Currency Conversion

### `target_usd_year`

**Type**: Integer
**Required**: Yes
**Example**: `2023`

All monetary values are converted to this base year USD using CPI data.

```yaml
target_usd_year: 2023
```

PowerGenome loads CPI data from `data/cpi_data/` to perform conversions. Technology costs from ATB (in their native years) and EIA data (various years) are all normalized to this target year.

## Time Zone

### `utc_offset`

**Type**: Integer
**Required**: No
**Default**: 0
**Example**: `-8`

Hour offset from UTC for the model timezone. All time-series data is stored in UTC and converted using this offset.

```yaml
utc_offset: -8  # Pacific Time (UTC-8)
```

US time zones:

- Pacific: `-8`
- Mountain: `-7`
- Central: `-6`
- Eastern: `-5`

## Example Configuration

Complete model definition for a multi-period study:

```yaml
# Planning horizon
model_periods:
  - [2025, 2030]
  - [2031, 2040]
  - [2041, 2050]

# Currency
target_usd_year: 2023

# Time zone
utc_offset: -8
```

## Related Settings

- [Regions](regions.md): Define model regions and aggregations
- [Scenario Management](scenario-management.md): Multi-scenario parameter swapping
- [Resource Tags](resource-tags.md): Detailed tag configuration
- [Data Tables](data-tables.md): Input data source configuration
