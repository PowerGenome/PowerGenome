# Distributed Generation

PowerGenome models **distributed generation (DG)** — primarily rooftop solar — as behind-the-meter generation that reduces the net load seen by the bulk power system. This page explains how the DG system works and what data it needs.

For reference documentation on the related settings parameters, see [Demand and Load Settings](../reference/settings/demand.md).

---

## What is distributed generation here?

In PowerGenome, "distributed generation" refers to small-scale generators sited at customer meters, mainly rooftop photovoltaics. These resources:

- Are never dispatched by the optimizer (they generate whenever sunlight is available)
- Reduce the net electricity demand that grid-scale resources must serve
- Are modeled via two inputs: **how much capacity** exists, and **what normalized profile** it follows

The pipeline subtracts DG generation from hourly gross load before passing demand to GenX.

---

## Data model

DG data is stored in two tables that DataManager loads at runtime.

### Capacity table (`distributed_capacity_table`)

Rows: one per region per year
Required columns:

| Column | Description |
|---|---|
| `region` | Model region name or base region name |
| `capacity_mw` | Installed DG capacity in MW |
| `year` | Planning year |

Optional column: `scenario` (for multi-scenario files).

If data is not available for a requested planning year, PowerGenome linearly interpolates between the nearest years with data. If the year is outside the range of available data, the nearest year's values are used.

### Profiles table (`distributed_profiles_table`)

Rows: one per region per hour per weather year
Required columns:

| Column | Description |
|---|---|
| `region` | Model region name or base region name |
| `weather_year` | Year the weather data comes from |
| `time_index` | Hour index (1 to 8760) |
| `value` | Normalized generation (0–1, where 1 = peak output) |

`value` must be normalized so the maximum is 1. Hourly generation (MWh) is computed by multiplying `value × capacity_mw`.

---

## How DG affects load

During load profile construction, PowerGenome:

1. Loads gross hourly demand from `demand_table`
2. Loads DG capacity for the planning year
3. Loads normalized DG profiles for the appropriate `weather_year`
4. Multiplies capacity × profile to get hourly generation
5. Subtracts hourly DG generation from gross demand → **net load**

Net load is what ends up in GenX's `Load_data.csv`.

---

## Settings configuration

### Minimal setup

```yaml
# Data location
data_location: /path/to/data

# DG tables (files in data_location)
distributed_capacity_table: distributed_capacity.parquet
distributed_profiles_table: distributed_profiles.parquet

# Optional: pick a specific weather year for profiles
# Defaults to model_year if not set
weather_year: 2012
```

### Multi-scenario setup

If your data contains a `scenario` column, filter it per scenario:

```yaml
distributed_capacity_table:
  table_name: distributed_capacity.parquet
  scenario: high_electrification

distributed_profiles_table:
  table_name: distributed_profiles.parquet
  scenario: high_electrification
```

Alternatively, keep separate files per scenario and swap them via
`settings_management` across scenarios.

### Region aggregation

If your data uses sub-regional base regions (e.g., IPM regions) while your
model combines them into aggregated model regions, PowerGenome automatically
aggregates capacity when `region_aggregations` is defined in settings.

---

## Migrating from the old system

Before the DataManager refactor, DG used a single Parquet file with an
`DISTRIBUTED_GEN_DATA` environment variable. The new system replaces this
with two tables in `data_location`. This section explains how to convert.

### What changed

| Old system | New system |
|---|---|
| `distributed_gen_fn` | `distributed_capacity_table` + `distributed_profiles_table` |
| `distributed_gen_scenario` | `scenario` key in table config |
| `DISTRIBUTED_GEN_DATA` env var | `data_location` in settings |
| Population-weighted aggregation | Direct region-level data |

### Converting your data

The script below converts the old NREL-format Parquet file to the new format:

```python
import pandas as pd
from pathlib import Path

# Configuration
old_data_path = Path("/path/to/DISTRIBUTED_GEN_DATA")
old_file = "dgen_profiles.parquet"
scenario = "reference"  # your scenario name
output_path = Path("/path/to/new/data")
output_path.mkdir(parents=True, exist_ok=True)

# Load old data and filter to scenario
old_data = pd.read_parquet(old_data_path / old_file)
old_data = old_data[old_data["scenario"] == scenario]

# Create capacity table (peak generation = installed capacity proxy)
capacity = (
    old_data
    .groupby(["region", "year"])["distpv_MWh"]
    .max()
    .reset_index()
    .rename(columns={"distpv_MWh": "capacity_mw"})
)

# Create profiles table (normalize by peak)
profiles = []
for (region, year), group in old_data.groupby(["region", "year"]):
    peak = group["distpv_MWh"].max()
    if peak > 0:
        profile = group.copy()
        profile["value"] = profile["distpv_MWh"] / peak
        profile["weather_year"] = year
        profiles.append(profile[["region", "weather_year", "time_index", "value"]])

profiles_df = pd.concat(profiles, ignore_index=True)

# Save
capacity.to_parquet(output_path / "distributed_capacity.parquet", index=False)
profiles_df.to_parquet(output_path / "distributed_profiles.parquet", index=False)
```

### Updating settings

Remove old settings:

```yaml
# Remove these:
distributed_gen_fn: dgen_profiles.parquet
distributed_gen_scenario: reference
# DISTRIBUTED_GEN_DATA env var in .env
```

Add new settings:

```yaml
distributed_capacity_table: distributed_capacity.parquet
distributed_profiles_table: distributed_profiles.parquet
weather_year: 2012  # optional
```

---

## Troubleshooting

**No DG subtraction is happening**

Confirm both `distributed_capacity_table` and `distributed_profiles_table` are set in settings. DataManager will log a warning if either table cannot be loaded.

**Wrong year being used for profiles**

Explicitly set `weather_year` in settings to pin the profile year:

```yaml
weather_year: 2012
```

By default, `weather_year` falls back to `model_year`, which may not match your available profile data.

**Region names don't match**

DG data region names must match either the base (IPM) region names or the aggregated model region names in `model_regions`. Check the `region` column in your capacity and profile files against your settings.

---

## Related documentation

- [Demand and Load Settings](../reference/settings/demand.md): Full parameter list for DG settings
- [Configure Data Tables](../how-to/configure-data-tables.md): How to configure DataManager tables
- [Data Pipeline Flow](pipeline.md): How DG fits into the overall load construction step
