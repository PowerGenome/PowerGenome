# Demand Settings

These parameters control electricity demand data sources for the model.

## Demand Data

### `demand_table`

**Type**: String or dictionary
**Required**: Yes
**Example**: See below

Table containing hourly electricity demand projections for all model regions and planning periods.

**Simple**:

```yaml
demand_table: load_curves.csv
```

**Advanced** (with scenario selection):

```yaml
demand_table:
  table_name: demand_timeseries.parquet
  scenario: high_electrification
```

**Required columns**:

- `region`: Base region name, or a model region that is not aggregated
- `time_index`: Hour index
- `load_mw`: Demand in MW
- `year`: Planning year

**Optional columns**:

- `scenario`: Demand scenario identifier (reference, high_ev, high_electrification, etc.)
- `weather_year`: Weather data vintage year

**Format**: Tidy/long format with one row per region-time-year observation.

**Example demand CSV**:

```csv
time_index,weather_year,region,load_mw,year,scenario
1,2012,CA_N,15234.5,2030,reference
2,2012,CA_N,14123.2,2030,reference
3,2012,CA_N,13890.4,2030,reference
1,2012,CA_S,12450.8,2030,reference
...
1,2012,CA_N,16890.2,2040,reference
2,2012,CA_N,15678.9,2040,reference
...
```

!!! important "Multi-Period Coverage"
    The `demand_table` should contain hourly demand projections for **all future modeling periods**. For example, if your model runs from 2030 to 2050 in 5-year increments, the table should include demand data for 2030, 2035, 2040, 2045, and 2050.

### Demand Scenarios

Use the `scenario` column to manage different demand futures:

```yaml
demand_table:
  table_name: demand_projections.parquet
  scenario: high_electrification  # Options: reference, high_ev, high_electrification
```

Common scenarios:

- **reference**: Base case load growth
- **high_ev**: Increased electric vehicle adoption
- **high_electrification**: Widespread building/industrial electrification
- **low_growth**: Slower demand growth with efficiency improvements

## Supplemental Demand

### `supplemental_demand_table`

**Type**: String or dictionary
**Required**: No
**Example**: See below

Optional table of additional hourly demand (e.g. data-center forecasts, new industrial loads) to add on top of the baseline demand profiles.

**How it is applied**: Supplemental demand is joined into the **base load data stage** inside `make_load_curves` — in long format, one row per (region × hour), *before* per-weather-year hours are renumbered to 1..N and *before* base regions are aggregated into model regions. Rows sharing the same `(region, weather_year, time_index)` are summed with the base load, so **no block-tiling is used** and **weather years of unequal length** (e.g. an 8 784-hour leap year next to a normal 8 760-hour year) are handled correctly. A `weather_year: all` row expands to one copy for each weather year actually present in the load data, rather than tiling a fixed block size.

**Simple** (file in `data_location` folder, or co-located with a database):

```yaml
supplemental_demand_table: supplemental_demand.csv
```

**Advanced** (with scenario selection):

```yaml
supplemental_demand_table:
  table_name: supplemental_demand.csv
  scenario: high_data_center
```

**Required columns**:

- `region`: Base region **or** model region name. Base region names are mapped to the
  aggregated model region that contains them (per `region_aggregations`); model region
  names are used as-is. Names matching neither a known base region nor a model region
  are logged as a warning and ignored.
- `time_index`: Integer hour index (1-based) **or** the strings `all` / `all_hours`
- `load_mw`: MW of demand to add

**Optional columns**:

- `year`: When present, rows are filtered to the current `model_year`
- `scenario`: Scenario identifier; when present, exactly one scenario must remain after
  loading. If multiple scenarios are found and none has been selected, PowerGenome raises
  a descriptive error listing the available scenario names and showing how to select one
  (see below).
- `weather_year`: When present, controls which weather-year the row applies to:
  `all` applies the row to every weather year present in the load data, a specific
  value (e.g. `2012`) applies it only to that weather year, and a **blank** value
  causes the row to be skipped. There is **no fixed block size**: an `all` row expands
  to one copy for each weather year actually present, and weather years of different
  lengths are supported. When the load data spans multiple weather years (its weather
  years are known via the `weather_year` setting), every weather year must be covered
  by either a specific-value row or an `all` row; a missing weather year raises an
  error naming the uncovered years. A specific `weather_year` that does not match any
  weather year in the load data raises a descriptive error telling you to set the
  `weather_year` setting. No coverage check is performed when this column is absent.

**`time_index` values**:

- **`all` / `all_hours`** – adds the demand increment to every hour of the weather
  year(s) selected by `weather_year` (flat load addition). For each selected weather
  year, one copy is added per hour actually present in that weather year.
- **Integer** – adds the demand increment only to the hour with that index; with
  `weather_year` set to `all`, it is applied to that hour in every weather year.

**Scenario handling**:

If the supplemental demand file contains multiple scenarios and no scenario is specified, PowerGenome raises:

```
ValueError: The supplemental_demand table contains multiple scenarios
('baseline', 'high_data_center') but no scenario has been selected.
Specify which scenario to use in your settings file, for example:

    supplemental_demand_table:
      table_name: supplemental_demand.csv
      scenario: baseline
```

**File co-located with a database**:

When `data_location` points to a database file (`.db`, `.duckdb`), a filename with a
`.csv` or `.parquet` extension is resolved relative to the database's parent directory:

```yaml
data_location: /data/pg_data.db
supplemental_demand_table: supplemental_demand.csv  # → /data/supplemental_demand.csv
```

**Example supplemental demand CSV**:

```csv
region,time_index,load_mw,year,scenario
WEST,all_hours,500,2030,base_data_center
WEST,all_hours,800,2035,base_data_center
EAST,all_hours,200,2030,base_data_center
EAST,1,300,2030,base_data_center
WEST,all_hours,1000,2030,high_data_center
WEST,all_hours,1500,2035,high_data_center
EAST,all_hours,400,2030,high_data_center
```

For a step-by-step guide see [Add Supplemental Hourly Demand](../../how-to/add-supplemental-demand.md).

## Time Domain Reduction

Time domain reduction replaces the hourly load and resource profiles with a small
number of representative periods (usually days or weeks) that are each modeled once
and weighted by how often they stand in for the full horizon. It is off by default.

The backend follows the numerical design of MacroEnergy's
[time domain reduction](https://github.com/macroenergy/MacroEnergyTimeReduction.jl):
profiles are scaled per feature, features are weighted by the square root of their
weight, the period containing the system peak load can be forced into the output,
and each representative period is a real input period. Reduced outputs are
therefore numerically different from earlier releases even with identical settings.

### `reduce_time_domain`

**Type**: Boolean
**Default**: `false`
**Effect**: Enables time domain reduction. When `false`, hourly load and resource
profiles are written unchanged. Reduction is skipped with a warning when the horizon
is longer than 8760 hours (also unchanged).

```yaml
reduce_time_domain: true
```

### `time_domain_periods`

**Type**: Integer
**Required when reducing**
**Effect**: Number of representative periods in the reduced model. The output has
`time_domain_periods * time_domain_days_per_period * 24` rows, and
`time_domain_periods` must not exceed the number of complete periods in the horizon.
When it equals the number of complete periods, every period is its own representative
period and clustering is skipped.

```yaml
time_domain_periods: 52
```

### `time_domain_days_per_period`

**Type**: Integer
**Required when reducing**
**Effect**: Number of 24-hour days in each representative period. With the default
8760-hour horizon, `7` gives 52 candidate weeks and `time_domain_periods: 52` selects
every week.

```yaml
time_domain_days_per_period: 7
```

### `include_peak_day`

**Type**: Boolean
**Required when reducing** (the example settings use `true`)
**Effect**: Reserves one representative-period slot for the period containing the hour
with the highest system-wide demand across all model regions. That period is excluded
from clustering and reconstructed from its own hours, so it is never double counted.
Turn it off to let the clustering decide every representative period. At least two
representative periods are needed only when clustering runs, since one slot is reserved
for the peak and at least one must remain to cluster: a horizon with a single complete
period and `time_domain_periods: 1` succeeds and returns that period.

```yaml
include_peak_day: true
```

### `demand_weight_factor`

**Type**: Number (>= 0)
**Required when reducing** (typical value `1`)
**Effect**: Weight of each demand feature relative to the resource features. Demand
values are multiplied by `sqrt(demand_weight_factor)` after they are scaled. This is
a deliberate change from earlier releases, which multiplied raw demand by the factor
and scaled all features together.

```yaml
demand_weight_factor: 2
```

### `time_domain_reduction_method`

**Type**: String — `kmeans`, `kmedoids`, or `legacy_kmeans`
**Default**: `kmeans`
**Effect**: Clustering backend.

- `kmeans` — scikit-learn k-means with k-means++ initialization. Each cluster's
  representative is the real period nearest its centroid.
- `kmedoids` — alternating (Voronoi) k-medoids over the Euclidean distance matrix.
  Each representative is an actual period, so it is robust to unusual periods.
- `legacy_kmeans` — the previous implementation in `powergenome/time_reduction.py`.
  It remains available for one release and reproduces the old outputs, including its
  global feature scaling and dropped trailing hours.

```yaml
time_domain_reduction_method: kmedoids
```

### `tdr_scaling`

**Type**: String — `normalize` or `standardize`
**Default**: `normalize`
**Effect**: How each feature is scaled before weighting.

- `normalize` — min-max each feature to `[0, 1]` over the candidate period hours.
- `standardize` — z-score each feature over the candidate period hours.

A constant feature becomes a constant row (all zeros under `normalize`), and constant
rows cannot change the distances between candidate periods. Scaling is per feature, so
a demand series in MW and a capacity-factor series in `[0, 1]` carry comparable
influence without tuning `demand_weight_factor`.

```yaml
tdr_scaling: standardize
```

### `tdr_n_restarts`

**Type**: Positive integer
**Default**: `10`
**Effect**: Number of random initializations for k-means and k-medoids. Higher values
find lower-cost clusterings at some runtime cost. The deprecated alias `tdr_n_init` is
accepted when `tdr_n_restarts` is absent; the legacy method keeps reading `tdr_n_init`
with its old default.

```yaml
tdr_n_restarts: 20
```

### `tdr_vre_profile_grouping`

**Type**: String — `none` or `region_technology`
**Default**: `none`
**Effect**: Controls the features used to choose representative periods for variable
renewable resources. It never changes the reduced output profiles: every resource
column is still written with its own profile.

- `none` — one clustering feature per resource, so adding sites of the same technology
  adds clustering weight to that technology.
- `region_technology` — VRE resources are grouped by `(region, technology)` from the
  generator table, and each group contributes exactly one feature: the arithmetic mean
  of the group's raw hourly profiles, computed before scaling. Onshore and offshore
  wind are separate groups because their technologies differ. A group carries a single
  feature weight, not one per member, so adding sites no longer changes how much that
  technology shape influences the clustering. Resources without usable
  region/technology/VRE metadata stay individual features and a single warning reports
  how many fell back.

More grouping modes may be added later.

```yaml
tdr_vre_profile_grouping: region_technology
```

### Trailing hours and `Sub_Weights`

Only complete periods are clustered. When the horizon does not divide evenly, the
hours after the last complete period are excluded from clustering and their weight is
redistributed across the representative periods in proportion to how many periods each
one represents (largest-remainder allocation). `Sub_Weights` therefore always sums
exactly to the input horizon — for example, 8760 hours with 7-day periods are 52
complete weeks (8736 hours) plus 24 trailing hours, and `Sub_Weights` sums to 8760. A
warning names the horizon, the period length, the number of complete periods, the
trailing hours, and how they were redistributed. An evenly divisible horizon produces
no warning.

### Limit on horizon length

Representative periods cannot be selected for horizons longer than 8760 hours. Such
runs are written unchanged with a warning, exactly as before.

## Related Settings

- [Data Tables](data-tables.md): Format specification for all data tables
- [Regions](regions.md): Model region definitions
- [Model Definition](model-definition.md): Planning years configuration
