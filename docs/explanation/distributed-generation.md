# Distributed Generation

PowerGenome models **distributed generation (DG)** — primarily rooftop solar — either
as behind-the-meter generation that reduces net load or as a generating resource.
This page explains the inputs and both treatments.

For reference documentation on the related settings parameters, see [Demand and Load Settings](../reference/settings/demand.md).

---

## What is distributed generation here?

In PowerGenome, "distributed generation" refers to small-scale generators sited at customer meters, mainly rooftop photovoltaics. These resources:

- By default, reduce the net electricity demand that grid-scale resources must serve
- Can instead be included as generating resources using `dg_as_resource: true`
- Are modeled via two inputs: **how much capacity** exists, and **what normalized profile** it follows

By default, the pipeline subtracts DG generation from hourly gross load before passing
demand to GenX. Resource mode leaves gross demand unchanged; the resource's tags
determine its treatment in GenX.

---

## Data model

DG data is stored in two tables that DataManager loads at runtime.

### Capacity table (`distributed_capacity_table`)

Rows: one per region per year
Required columns:

| Column | Description |
|---|---|
| `region` | Base region name, or model region name when it is not aggregated |
| `capacity_mw` | Installed DG capacity in MW |
| `year` | Planning year |

Optional column: `scenario` (for multi-scenario files).

If **no requested base region** has data for the planning year, PowerGenome
interpolates each region between available years, using the nearest value outside
the range. If some regions have an exact-year value and others do not, the missing
regions are not interpolated. Provide complete coverage for each requested year.

### Profiles table (`distributed_profiles_table`)

Rows: one per region per hour per weather year
Required columns:

| Column | Description |
|---|---|
| `region` | Base region name, or model region name when it is not aggregated |
| `weather_year` | Year the weather data comes from |
| `time_index` | Hour index within the source profile |
| `value` | Capacity factor (0–1, relative to installed capacity) |

Hourly generation in MW is computed as `value × capacity_mw`; the profile's maximum
need not reach 1. Profiles are reordered and indexed from 1. Supply matching hours
and weather years for DG and load. For multiple weather years, list them in
ascending order: DG profiles are sorted by year, and subtraction is positional.

---

## How DG affects load

During load profile construction, PowerGenome:

1. Loads gross hourly demand from `demand_table`
2. Loads DG capacity for the planning year
3. Loads normalized DG profiles for the appropriate `weather_year`
4. Multiplies capacity × profile to get hourly generation
5. Multiplies generation by `1 + avg_distribution_loss` (default 1) and subtracts it from gross demand → **net load**

Net load is what is written to GenX's demand input (`system/Demand_data.csv` in the default v6 format; `Load_data.csv` in the legacy `old_genx_format` mode).

---

## Settings configuration

### Minimal setup

```yaml
# Data location
data_location: /path/to/data

# DG tables (files in data_location)
distributed_capacity_table: distributed_capacity.parquet
distributed_profiles_table: distributed_profiles.parquet

# Optional: restrict profiles to a specific weather year
# If omitted, all available weather years in distributed_profiles_table are used
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
aggregates capacity and computes capacity-weighted profiles when
`region_aggregations` is defined. Provide both inputs for the component regions;
each component with a profile needs a capacity value.

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
| Legacy region mappings | Base-region capacity and capacity-weighted profiles |

### Converting your data

Prepare separate capacity and profile tables using the columns above. Use installed
capacity, not peak hourly generation, as the capacity value; divide generation by
installed capacity to derive the capacity-factor profile. Preserve the actual
weather year independently of the planning year.

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

If `weather_year` is omitted, all available DG weather years are loaded. Ensure
they align with the selected demand profiles.

**Region names don't match**

For aggregated model regions, use their component base-region names in both tables.
For other regions, use the names in `model_regions`.

---

## Related documentation

- [Demand and Load Settings](../reference/settings/demand.md): Full parameter list for DG settings
- [Configure Data Tables](../how-to/configure-data-tables.md): How to configure DataManager tables
- [Data Pipeline Flow](pipeline.md): How DG fits into the overall load construction step
