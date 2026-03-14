# Time Reduction Settings

PowerGenome can reduce a full 8760-hour year to a smaller set of **representative periods** — groups of consecutive days that stand in for the full year. This reduces the size of your optimization problem while preserving the statistical properties of demand and variable generation.

Time domain reduction is applied jointly to load profiles and variable generation profiles (wind, solar) using k-means clustering.

---

## Parameters

### `reduce_time_domain`

**Type**: `bool`
**Default**: `false`

Set to `true` to enable time domain reduction. When `false` (or absent), all 8760 hours are passed to GenX as a single period with weight 8760.

Time reduction only works on data with ≤ 8760 hourly rows. If your data is already shorter (e.g., a previous reduction), the step is skipped automatically.

```yaml
reduce_time_domain: true
```

---

### `time_domain_periods`

**Type**: `int`
**Required when**: `reduce_time_domain: true`

The number of representative periods (clusters) to select. Each period is a block of `time_domain_days_per_period` consecutive days.

Common values:

| Value | Description |
|---|---|
| 4 | Seasonal quarters (very coarse) |
| 12 | Monthly-style reduction |
| 52 | Weekly — one week per calendar week |

```yaml
time_domain_periods: 12
```

---

### `time_domain_days_per_period`

**Type**: `int`
**Required when**: `reduce_time_domain: true`

The number of consecutive days in each representative period. Combined with
`time_domain_periods`, this defines the total number of hours in your reduced
time series:

$$\text{total hours} = \text{time\_domain\_periods} \times \text{time\_domain\_days\_per\_period} \times 24$$

For a model with unit-commitment constraints, using multiple days per period (e.g., 7) is recommended so that start-up and shut-down decisions span realistic multi-day sequences.

```yaml
time_domain_days_per_period: 7   # weekly periods
```

---

### `include_peak_day`

**Type**: `bool`
**Required when**: `reduce_time_domain: true`

When `true`, the day with the highest system-wide demand is always included as one of the representative periods, regardless of which day k-means would select. This ensures the optimization model sees the true system peak.

```yaml
include_peak_day: true
```

---

### `demand_weight_factor`

**Type**: `int` or `float`
**Required when**: `reduce_time_domain: true`

Before k-means clustering, all profiles (load and variable generation) are scaled to the range [0, 1]. Demand profiles are then multiplied by this factor. A value greater than 1 weights demand more heavily than renewable generation when selecting clusters.

- **`1`**: demand and variable generation weighted equally
- **`5`**: demand contributes 5× more to cluster selection (recommended starting point)

```yaml
demand_weight_factor: 5
```

---

### `tdr_n_init`

**Type**: `int`
**Default**: `100`
**Optional**

Number of k-means initializations to run. More initializations reduce the chance of converging to a local minimum, at the cost of longer run time.

```yaml
tdr_n_init: 100
```

---

## Minimal example

```yaml
reduce_time_domain: true
time_domain_periods: 12
time_domain_days_per_period: 7
include_peak_day: true
demand_weight_factor: 5
```

This creates 12 representative weeks, with the peak demand day guaranteed to appear in one of them.

---

## Output format

When time reduction is active, the `system/Demand_data.csv` output (or `Load_data.csv` when using the legacy `old_genx_format`) includes additional
metadata columns that GenX uses to reconstruct the weight of each representative
period:

| Column | Description |
|---|---|
| `Rep_Periods` | Total number of representative periods |
| `Timesteps_per_Rep_Period` | Hours per period (`days × 24`) |
| `Sub_Weights` | Number of hours from the full year that each period represents |
| `Time_Index` | Hour index (1 to `Rep_Periods × Timesteps_per_Rep_Period`) |

`Sub_Weights` must sum to 8760 across all periods for an annual model.

---

## Choosing parameter values

There is no universally correct number of representative periods. Typical guidance:

- **12–52 periods** is a common range for capacity expansion. Fewer periods run faster but miss nuance in correlated wind/solar/demand patterns.
- **7-day periods** are a good starting point. Shorter periods (1–2 days) can cause the optimizer to ignore multi-day operational constraints.
- **Always include the peak day** when your model has capacity reserve constraints.
- **Check convergence**: run with a larger number of periods and verify that model results are stable before reducing further.

---

## Related documentation

- [Demand and Load Settings](demand.md): Load profile parameters
- [Model Definition](model-definition.md): Planning years that determine how many hours are in each period
