# Migration Guide: Old to New Distributed Generation System

## Overview

This guide will help you migrate from the old distributed generation system (using parquet files and population weights) to the new DataManager-based system.

## What Changed

### Old System (Before)

```yaml
# Settings
distributed_gen_fn: "dgen_profiles.parquet"
distributed_gen_scenario: "reference"
DISTRIBUTED_GEN_DATA: "/path/to/dgen/data"
utc_offset: 0
```

**Requirements:**

- NREL-format parquet files with columns: `region`, `time_index`, `year`, `scenario`, `distpv_MWh`
- Population weight files (e.g., `ipm_state_pop_weight_*.csv`)
- `DISTRIBUTED_GEN_DATA` environment variable

### New System (After)

```yaml
# Settings
distributed_capacity_table: "distributed_capacity.parquet"
distributed_profiles_table: "distributed_profiles.parquet"
weather_year: 2012  # optional, defaults to model_year
utc_offset: 0
```

**Requirements:**

- Two separate tables: capacity and profiles
- No population weights needed
- No environment variables needed
- Data loaded through DataManager

## Migration Steps

### Step 1: Convert Your Data

Create a script to convert old NREL format to new format:

```python
import pandas as pd
from pathlib import Path

# Configuration
old_data_path = Path("/path/to/DISTRIBUTED_GEN_DATA")
old_file = "dgen_profiles.parquet"
scenario = "reference"  # or your scenario name

# Load old data
print("Loading old data...")
old_data = pd.read_parquet(old_data_path / old_file)

# Filter to your scenario
old_data = old_data[old_data["scenario"] == scenario]

print(f"Processing {len(old_data)} rows...")

# Create capacity table
print("Creating capacity table...")
capacity = (
    old_data
    .groupby(["region", "year"])["distpv_MWh"]
    .max()
    .reset_index()
    .rename(columns={"distpv_MWh": "capacity_mw"})
)

print(f"Capacity table: {len(capacity)} rows, {capacity['region'].nunique()} regions, "
      f"{capacity['year'].nunique()} years")

# Create profiles table
print("Creating profiles table...")
profiles = []

for (region, year), group in old_data.groupby(["region", "year"]):
    # Get peak for normalization
    peak = group["distpv_MWh"].max()

    if peak > 0:
        norm_profile = group.copy()
        norm_profile["value"] = norm_profile["distpv_MWh"] / peak
        norm_profile["weather_year"] = year
        profiles.append(
            norm_profile[["region", "weather_year", "time_index", "value"]]
        )

profiles_df = pd.concat(profiles, ignore_index=True)

print(f"Profiles table: {len(profiles_df)} rows, {profiles_df['region'].nunique()} regions, "
      f"{profiles_df['weather_year'].nunique()} weather years")

# Save new format
output_path = Path("/path/to/new/data")
output_path.mkdir(parents=True, exist_ok=True)

capacity.to_parquet(output_path / "distributed_capacity.parquet", index=False)
profiles_df.to_parquet(output_path / "distributed_profiles.parquet", index=False)

print("\nConversion complete!")
print(f"Files saved to: {output_path}")
print(f"  - distributed_capacity.parquet")
print(f"  - distributed_profiles.parquet")

# Verify the data
print("\nData verification:")
print("\nCapacity summary:")
print(capacity.groupby("year")["capacity_mw"].agg(["count", "sum", "mean"]))
print("\nProfile summary:")
print(profiles_df.groupby("weather_year")["value"].agg(["count", "min", "max", "mean"]))
```

### Step 2: Update Your Settings File

#### Before

```yaml
model_year: 2030
model_regions:
  - region1
  - region2
  - region3

# Old DG settings
distributed_gen_fn: "dgen_profiles.parquet"
distributed_gen_scenario: "reference"
utc_offset: 0

# In .env file:
# DISTRIBUTED_GEN_DATA=/path/to/dgen/data
```

#### After

```yaml
model_year: 2030
weather_year: 2012  # optional, defaults to model_year
model_regions:
  - region1
  - region2
  - region3

# New DG settings
distributed_capacity_table: "distributed_capacity.parquet"
distributed_profiles_table: "distributed_profiles.parquet"
utc_offset: 0

# Remove from .env file:
# DISTRIBUTED_GEN_DATA (no longer needed)
```

### Step 3: Update DataManager Initialization

#### Before

```python
from powergenome.pudl_data_extraction import setup_engines

pudl_engine, pudl_out, pg_engine = setup_engines(settings)

# DG data loaded from DISTRIBUTED_GEN_DATA environment variable
```

#### After

```python
from powergenome.database import initialize_data_manager

# Initialize with your data location
data_location = "/path/to/new/data"
initialize_data_manager(settings, data_location)

# DG data now loaded through DataManager
```

### Step 4: Verify Your Code Still Works

The public API hasn't changed, so existing code should work:

```python
# These functions still work the same way
from powergenome.load_profiles import (
    make_distributed_gen_profiles,
    subtract_distributed_generation
)
from powergenome.generators import add_dg_resources

# Get profiles (normalized 0-1)
dg_profiles = make_distributed_gen_profiles(settings)

# Subtract from load
load_curves_net = subtract_distributed_generation(load_curves, settings)

# Add as generator resources
gen_df = add_dg_resources(settings, gen_df=existing_generators)
```

## Handling Multiple Scenarios

### Old System

```python
# Scenario was a column in the data
distributed_gen_scenario: "high_electrification"
```

### New System

**Option 1: Separate Files**

```yaml
# Use different files for different scenarios
distributed_capacity_table: "distributed_capacity_high.parquet"
distributed_profiles_table: "distributed_profiles_high.parquet"
```

**Option 2: Add Scenario Column and Use Filters**

```yaml
# Keep scenario column in your data and filter
distributed_capacity_table:
  table_name: "distributed_capacity.parquet"
  filters:
    - [["scenario", "=", "high_electrification"]]

distributed_profiles_table:
  table_name: "distributed_profiles.parquet"
  filters:
    - [["scenario", "=", "high_electrification"]]
```

**Option 3: Use Scenario Parameter**

```yaml
# Use the scenario convenience parameter
distributed_capacity_table:
  table_name: "distributed_capacity.parquet"
  scenario: "high_electrification"

distributed_profiles_table:
  table_name: "distributed_profiles.parquet"
  scenario: "high_electrification"
```

## Handling Year Interpolation

### Old System

The old system automatically interpolated between available years.

### New System

Prepare data for the exact years you need.

```python
import pandas as pd
import numpy as np

def interpolate_years(df, year_col, value_cols, target_years):
    """Interpolate data to target years."""

    results = []

    # Get unique combinations of non-year columns
    group_cols = [c for c in df.columns if c not in [year_col] + value_cols]

    for keys, group in df.groupby(group_cols):
        # Interpolate each value column
        for target_year in target_years:
            # Find bounding years
            available_years = sorted(group[year_col].unique())

            if target_year in available_years:
                # Exact match
                row = group[group[year_col] == target_year].iloc[0].to_dict()
            else:
                # Interpolate
                lower = max([y for y in available_years if y < target_year], default=None)
                upper = min([y for y in available_years if y > target_year], default=None)

                if lower is None:
                    # Use earliest year
                    row = group[group[year_col] == available_years[0]].iloc[0].to_dict()
                    row[year_col] = target_year
                elif upper is None:
                    # Use latest year
                    row = group[group[year_col] == available_years[-1]].iloc[0].to_dict()
                    row[year_col] = target_year
                else:
                    # Linear interpolation
                    lower_row = group[group[year_col] == lower].iloc[0]
                    upper_row = group[group[year_col] == upper].iloc[0]

                    weight_upper = (target_year - lower) / (upper - lower)
                    weight_lower = 1 - weight_upper

                    row = lower_row.to_dict()
                    row[year_col] = target_year

                    for col in value_cols:
                        row[col] = (
                            lower_row[col] * weight_lower +
                            upper_row[col] * weight_upper
                        )

            results.append(row)

    return pd.DataFrame(results)

# Example usage:
capacity = pd.read_parquet("distributed_capacity.parquet")
capacity_interpolated = interpolate_years(
    capacity,
    year_col="year",
    value_cols=["capacity_mw"],
    target_years=[2025, 2030, 2035, 2040, 2045, 2050]
)
capacity_interpolated.to_parquet("distributed_capacity_interpolated.parquet")
```

## Troubleshooting

### Issue: "Table not found" error

**Cause:** DataManager not initialized or table not configured

**Solution:**

```python
from powergenome.database import initialize_data_manager, list_tables

# Check initialization
initialize_data_manager(settings, data_location)

# Check available tables
print(list_tables())

# Should see: ['distributed_capacity', 'distributed_profiles', ...]
```

### Issue: No data returned for my year

**Cause:** Year filtering or missing data

**Solution:**

```python
from powergenome.database import get_data

# Check what years are available
capacity = get_data("distributed_capacity", columns=["year"])
print(f"Available years: {sorted(capacity['year'].unique())}")

# Either:
# 1. Update settings to use available year
# 2. Add data for your year
# 3. Interpolate to create data for your year
```

### Issue: Regions don't match

**Cause:** Region names differ between old population weights and new system

**Solution:**

```python
# Check what regions are in your data
capacity = get_data("distributed_capacity", columns=["region"])
print(f"Available regions: {sorted(capacity['region'].unique())}")

# Update region names in data or settings to match
```

### Issue: Generation values look different

**Cause:** Normalization or capacity differences

**Solution:**

```python
# Verify peak generation matches
old_peak = old_data.groupby("region")["distpv_MWh"].max()
new_capacity = get_distributed_gen_capacity(year=2030, regions=regions)

comparison = pd.DataFrame({
    "old_peak": old_peak,
    "new_capacity": new_capacity.set_index("region")["capacity_mw"]
})
comparison["difference"] = comparison["new_capacity"] - comparison["old_peak"]
print(comparison)
```

## Testing Your Migration

Create a test script to verify the migration:

```python
import pandas as pd
from powergenome.database import initialize_data_manager
from powergenome.distributed_gen import (
    get_distributed_gen_capacity,
    get_distributed_gen_profiles,
    get_distributed_gen_hourly_generation,
)

# Settings
settings = {
    "model_year": 2030,
    "weather_year": 2030,
    "model_regions": ["region1", "region2"],
    "distributed_capacity_table": "distributed_capacity.parquet",
    "distributed_profiles_table": "distributed_profiles.parquet",
}

data_location = "/path/to/new/data"

# Initialize
initialize_data_manager(settings, data_location)

# Test 1: Load capacity
print("Test 1: Load capacity")
capacity = get_distributed_gen_capacity(
    year=settings["model_year"],
    regions=settings["model_regions"],
)
print(capacity)
assert not capacity.empty, "Capacity is empty!"
assert all(r in capacity["region"].values for r in settings["model_regions"]), "Missing regions!"

# Test 2: Load profiles
print("\nTest 2: Load profiles")
profiles = get_distributed_gen_profiles(
    weather_year=settings["weather_year"],
    regions=settings["model_regions"],
)
print(profiles.head())
assert not profiles.empty, "Profiles are empty!"
assert len(profiles) == 8760, f"Expected 8760 hours, got {len(profiles)}"
assert all(0 <= profiles[col].max() <= 1 for col in profiles.columns), "Profiles not normalized!"

# Test 3: Calculate hourly generation
print("\nTest 3: Calculate hourly generation")
hourly_gen = get_distributed_gen_hourly_generation(
    year=settings["model_year"],
    weather_year=settings["weather_year"],
    regions=settings["model_regions"],
)
print(hourly_gen.head())
assert not hourly_gen.empty, "Hourly generation is empty!"

# Test 4: Verify peak matches capacity
print("\nTest 4: Verify peak matches capacity")
for region in settings["model_regions"]:
    peak = hourly_gen[region].max()
    cap = capacity.loc[capacity["region"] == region, "capacity_mw"].values[0]
    diff = abs(peak - cap)
    print(f"{region}: peak={peak:.2f}, capacity={cap:.2f}, diff={diff:.2f}")
    assert diff < 0.1, f"Peak doesn't match capacity for {region}!"

print("\n✅ All tests passed!")
```

## Need Help?

If you encounter issues during migration:

1. Check the logs for warning messages
2. Review the documentation in `docs/distributed_generation.md`
3. See examples in `examples/distributed_gen_example.py`
4. Verify your data format matches the requirements
5. Test with a simple example before migrating all data

## Summary Checklist

- [ ] Convert old data to new format (capacity + profiles)
- [ ] Update settings file (remove old params, add new tables)
- [ ] Remove DISTRIBUTED_GEN_DATA from .env file
- [ ] Update DataManager initialization in your code
- [ ] Test with new system
- [ ] Verify results match old system
- [ ] Update documentation/comments in your code
