# Generator Clustering

PowerGenome groups real-world power plants into representative **clusters** to keep capacity expansion models tractable. Rather than modeling each unit individually (which could mean thousands of rows), generators with similar characteristics are merged into a smaller number of clusters per region and technology type.

---

## Why clustering is needed

Detailed generator inventories from EIA data can contain thousands of individual units. Including each unit as a separate resource in GenX would make the optimization problem very large without meaningfully improving accuracy, because units of the same vintage and technology behave nearly identically.

Clustering reduces these to a manageable number of representative resources while preserving the key features the model needs: capacity, heat rate, cost, and retirement eligibility.

---

## How existing generators are clustered

### Step 1: Filter and group

Generators are first filtered to those that:

- Operate in one of the model regions (after region aggregation)
- Are not yet retired based on their `operating_year` and `retirement_year` (from input data)

They are then grouped by **(model region, technology)** pairs. Each group is clustered independently.

### Step 2: K-means clustering

Within each region–technology group, [k-means clustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) partitions units into `k` clusters where `k` is set by `num_clusters` (or `alt_num_clusters` for per-region overrides).

**Clustering features** (by default):

- `heat_rate_mmbtu_mwh` — thermal efficiency
- `fom_per_mwyr` — fixed O&M cost

Features are standardized (mean 0, unit variance) before clustering so that neither dominates the distance metric purely because of scale.

If a region–technology group has fewer plants than the requested number of clusters, the number of clusters is automatically reduced to match the number of plants.

### Step 3: Cluster aggregation

For each cluster, representative values are computed:

| Attribute | Aggregation method |
|---|---|
| `capacity_mw` | Sum of all plants in cluster |
| `heat_rate_mmbtu_mwh` | Capacity-weighted average |
| `fom_per_mwyr` | Capacity-weighted average |
| `vom_per_mwh` | Capacity-weighted average |
| `minimum_load_mw` | Capacity-weighted average |

### Step 4: Resource naming

Each cluster is assigned a resource label:

```
<region>_<technology_snake_case>_<cluster_number>
```

For example: `CA_N_natural_gas_fired_combined_cycle_1`

This name is used as the resource identifier throughout the output files.

---

## Controlling the number of clusters

### Default cluster count

```yaml
num_clusters: 1  # Default: merge all plants of a type into one cluster
```

A value of `1` means all generators of the same type in a region become one representative unit. This is the most aggressive reduction and is appropriate when you need a compact model.

### Per-region overrides

Use `alt_num_clusters` to set different cluster counts for specific combinations:

```yaml
alt_num_clusters:
  CA_N:
    Natural Gas Fired Combined Cycle: 3   # Three CCGT clusters in CA_N
    Conventional Steam Coal: 1             # All coal in one cluster
  TX:
    Natural Gas Fired Combined Cycle: 5
```

Setting `num_clusters` to `0` or omitting a region/tech combination excludes those generators entirely. Technology names here are matched against the raw data values; they are **not** the ATB-style names.

### Individual units (no clustering)

To preserve every generator as an individual resource, set `num_clusters` equal to the number of plants (or use a large number — PowerGenome reduces it to the number of plants automatically):

```yaml
num_clusters: 999  # Each plant becomes its own resource
```

---

## Technology grouping

`tech_groups` merges similar technologies into a single group before clustering, so they share a resource label:

```yaml
tech_groups:
  Landfill Gas:
    - Landfill Gas
    - Municipal Solid Waste
    - Other Gases
```

This is useful for small fuel types where separating them would create many tiny resources.

---

## Retirement filtering

Existing generators are clustered using every unit operating in the first planning
period. A unit counts as operating when its `operating_year` is on or before the period and
its `retirement_year` is later than the period's end. Both columns come from the
generation input data; there is no `retirement_ages` setting (that code path has been
removed). A blank `retirement_year` satisfies neither test, so the unit is counted as
neither operating nor retired and drops out of the clusters — give every unit a
retirement year and use a far-future value for units expected to keep operating.

!!! note "Myopic multi-period models"
    Existing generators are clustered **once**, with all units operating in the first
    planning period, so group membership is stable across every period. Retirements
    between periods do not change the clusters; instead, capacity that retires within a
    period is removed from the cluster at that period via `Min_Retired_Cap_MW` /
    `Min_Retired_Energy_Cap_MW` (see `cap_retire_within_period`). Set each unit's
    `retirement_year` in the generation input data to control when its capacity drops
    out.

    One inconsistency is fatal: the capacity that a cluster is required to retire in each later period (`Min_Retired_Cap_MW`) is written from the units that a *later* period expects to retire, but GenX compares the total with the capacity the cluster has available in the *first* period (`Existing_Cap_MW`). When units drop out of the cluster between periods those requirements can add up to more than the first period has, and the model is infeasible. PowerGenome floors the requirements to the precision of the matching capacity column and stops with an error naming the resources that still overshoot — see [Debugging](../how-to/debugging.md).

---

## Viewing cluster assignments

When `extra_outputs_path` is configured (or by default in the `extra_outputs` sub-folder of each case), PowerGenome writes a CSV file showing which real plant belonged to which cluster. This is useful for debugging unexpected groupings.

---

## Related documentation

- [Existing Generators Settings](../reference/settings/existing-generators.md): `num_clusters`, `alt_num_clusters`, `tech_groups`
- [Add Custom Technologies](../how-to/add-technologies.md): Adding new-build resources
- [Modify Generator Attributes](../how-to/modify-generator-attributes.md): Applying custom formulas to cluster attributes
- [Architecture Overview](architecture.md): How clustering fits in the full pipeline
