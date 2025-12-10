# Model Definition Settings

These core parameters define the overall model scope, planning periods, and currency adjustments.

## Planning Years

### `model_year`

**Type**: List of integers
**Required**: Yes
**Example**: `[2030, 2040, 2050]`

The primary planning year(s) for the model. All costs are calculated for this year, and it determines which technology cost projections to use.

```yaml
model_year: [2030, 2040, 2050]
```

### `model_first_planning_year`

**Type**: List of integers
**Required**: Yes
**Example**: `[2025, 2031, 2041]`

The first year in a multi-period planning horizon. Used to calculate period lengths and determine which existing generators are retired.

```yaml
model_year: [2030, 2040, 2050]
model_first_planning_year: [2025, 2031, 2041]
```

This creates planning periods:

- 2025-2030 (5 years)
- 2031-2040 (10 years)
- 2041-2050 (10 years)

!!! note "How periods are used"

    - **Demand**: Uses the final `model_year` in each period (e.g., 2030 demand for 2025-2030)
    - **New-build costs**: Averaged across the full period (e.g., 2025-2030 capital costs are mean values over those years)
    - **Retirements**: Evaluated against `model_first_planning_year`/`model_year` windows

To shorten/extend the horizon, edit the paired lists. For example, to model 2030, 2035, and 2045 only, set `model_year: [2030, 2035, 2045]` and `model_first_planning_year: [2025, 2031, 2036]`.

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
model_year: [2030, 2040, 2050]
model_first_planning_year: [2025, 2031, 2041]

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
