# Generator Clustering

PowerGenome groups real-world power plants into representative **clusters** to keep capacity expansion models tractable. Rather than modeling each unit individually (which could mean thousands of rows), generators with similar characteristics are merged into a smaller number of clusters per region and technology type.

---

## Why clustering is needed

Detailed generator inventories from EIA data can contain thousands of individual units. Including each unit as a separate resource in GenX would make the optimization problem very large without meaningfully improving accuracy, because units of the same vintage and technology behave nearly identically.

Clustering reduces these to a manageable number of representative resources while preserving the key features the model needs: capacity, heat rate, cost, and retirement eligibility.

---

## How existing generators are clustered

### Step 1: Filter and group

Generators in the model regions (after region aggregation) are grouped by
**(model region, technology)** pairs. Each group is clustered independently.
[Retirement status](#retirement-filtering) determines which units contribute capacity,
not cluster membership.

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
| `capacity_mw` | Sum of operating units in cluster |
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

## Missing existing-resource availability profiles

Existing hydro, wind, solar, and other profile-backed resources need usable hourly
availability data. PowerGenome checks missing resource groups, missing regional
metadata, missing sites, and incomplete time series against a combined regional
fallback budget before assigning constant `1.0` availability.

PowerGenome permits a small, **combined regional fallback fleet**: by default,
`existing_resource_profile_fallback_max_demand_fraction: 0.005` allows affected
operating capacity up to 0.5% of mean regional demand **before DG subtraction**.
The denominator includes regional aggregation and supplemental/flexible demand,
uses unrounded hourly values for the selected scenario and years, and precedes time
reduction. Means across weather years are weighted by modeled hours.

At or below the limit, a warning identifies the affected resources and PowerGenome
retains their capacity, tags, and retirement attributes with availability `1.0` in
all modeled hours. Above it, an actionable error stops generation of model inputs.
Partial profile coverage charges the entire failing existing-generator cluster;
resource-group metadata capacity is never used as the affected capacity. All affected
technologies and clusters share the allowance in their model region. Valid profiles
remain unchanged, and retired/zero-capacity clusters consume no allowance.

For example, SD (p32 + p38) with 1584.86 MW mean demand has a 7.92 MW allowance:
4 MW of missing run-of-river profiles warns and falls back. Modeling p32 alone with
206.28 MW mean demand gives only 1.03 MW, so the same 4 MW errors. If those units
retire in 2030, they contribute zero operating capacity at model year 2030, even if
the planning period starts in 2026.

Set the fraction to `0` for strict mode; values must be finite numbers in `[0, 1]`,
not booleans. Positive-capacity fallback requires valid demand even with `--no-load`;
complete profiles add no demand requirement. Explicitly profile-free resource groups
(`group.profiles is None`, such as thermal/storage groups) remain exempt, as do
unsupported dispatchable technologies. New-build behavior is unchanged. Invalid schemas,
unreadable files, ambiguous group matches, and invalid weather-year requests remain
errors rather than fallback conditions.

This tolerance is a modeling choice, **not an industry standard or reliability
guarantee**. Constant availability can overstate usable generation. Its full-output
energy is bounded by the chosen fraction of gross demand over the same hours, but
that bound says nothing about hourly reliability or dispatch. Repair metadata/profile
coverage before considering a larger tolerance.

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

Set scheduled retirements in the generation table's `retirement_year` column. A
blank value or missing column means no planned retirement.

A unit contributes its full capacity if it is operating at the planning period's
**end** (`model_year`): `operating_year` must be on or before that year, and
`retirement_year` must be later or unspecified. Units that come online during a
period are included without prorating their capacity.

!!! note "Myopic multi-period models"
    Use the same generation data across the periods of a case to keep cluster
    membership consistent. PowerGenome records scheduled retirements in later
    periods as minimum retirement requirements for each resource (`Min_Retired_*`
    columns).

---

## Viewing cluster assignments

When `extra_outputs_path` is configured (or by default in the `extra_outputs` sub-folder of each case), PowerGenome writes a CSV file showing which real plant belonged to which cluster. This is useful for debugging unexpected groupings.

---

## Related documentation

- [Existing Generators Settings](../reference/settings/existing-generators.md): `num_clusters`, `alt_num_clusters`, `tech_groups`
- [Add Custom Technologies](../how-to/add-technologies.md): Adding new-build resources
- [Modify Generator Attributes](../how-to/modify-generator-attributes.md): Applying custom formulas to cluster attributes
- [Architecture Overview](architecture.md): How clustering fits in the full pipeline
