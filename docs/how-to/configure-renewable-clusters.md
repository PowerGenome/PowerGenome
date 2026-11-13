# Configure Renewable Resource Clusters

Use this guide to turn candidate wind or solar sites into buildable model
resources, with explicit choices about screening, cost tiers, and profile
similarity. For a runnable exercise using the published Texas example, follow
[Working with Renewable Resources](../tutorials/renewable-resources.md).

## Overview

A resource **group** is an input dataset: a JSON definition connects a technology
to site metadata and generation profiles. A resource **cluster** is an output:
one candidate resource representing one or more sites within a model region.
The setting `group` is different again: it separates sites by categorical
metadata, such as county.

Choose the simplest representation that preserves the differences your study
needs. More clusters increase model size; fewer can hide cost or temporal
variation. Compare results at different resolutions rather than choosing counts
from the number of model regions alone.

| Need | Setting | Effect |
| --- | --- | --- |
| Keep every eligible site | Omit `group`, `bin`, and `cluster` | One resource per site |
| Aggregate all eligible sites | One `bin` with `q: 1` | One resource per model region |
| Separate cost or performance tiers | `bin` | One resource per nonempty bin unless further split |
| Preserve categorical boundaries | `group` | Separate resources for observed categories |
| Preserve differences within tiers/categories | `cluster` | Cluster within each nonempty combination |
| Exclude candidates outside the study scope | `filter` | Removes sites before aggregation |

## Data Requirements

### Required Data Files

Set `RESOURCE_GROUPS` to a directory of JSON definitions (or a list of such
directories). This is the published Texas wind definition,
`example_systems/TX-3-zone/resource_groups/onshorewind_group.json`:

```json
{
  "technology": "landbasedwind",
  "metadata": "onshorewind_lcoe_tx_3_zone.csv",
  "profiles": "onshorewind_profiles_tx_3_zone.csv",
  "site_map": "onshorewind_site_map_tx_3_zone.csv"
}
```

Metadata paths resolve relative to the JSON file. Profiles are sought under
`RESOURCE_GROUP_PROFILES`, if supplied, then alongside the JSON. In the flexible
clustering path, `site_map` resolves alongside the **resolved profiles file**.
Keep that map with profiles if you move them to a separate directory.

`site_map` is optional. The CA/AZ example omits it for both technologies because
its profile files already use `cpa_id` values in `site_id`. That example also
points `RESOURCE_GROUPS` and `RESOURCE_GROUP_PROFILES` at the same directory, so
metadata and profiles stay together.

Relative `RESOURCE_GROUPS` and `RESOURCE_GROUP_PROFILES` paths resolve from the
current working directory. The published Texas settings require running from
the **examples root**, not inside `example_systems/TX-3-zone/`. By contrast,
`data_location` and `input_folder` resolve relative to the parent of the settings
directory.

### Required Columns

These requirements describe flexible site clustering, not legacy preclustered
resource trees. Column names are normalized to snake case when loaded; for
example, `CPA_ID` becomes `cpa_id`.

| Input | Requirement and units |
| --- | --- |
| `cpa_id` | Unique candidate-site identifier |
| `region` or `metro_region` | Region matching a model region or its base regions in `region_aggregations` |
| `capacity_mw` or `cpa_mw` | Available site capacity, MW; nonpositive-capacity rows are dropped |
| Filter/bin/cluster features | Actual metadata columns; no automatic creation of `cf`, `lcoe`, latitude, etc. |
| `lcoe`, if used | Input screening/binning metric, typically $/MWh; verify its definition and dollar year |
| `cf`, if used | Capacity factor as a fraction, not percent; a metadata feature, not recomputed from profiles |
| Interconnection costs, if supplied | `interconnect_capex_mw` in $/MW or `interconnect_annuity` in $/MW-year |

Other feature units come from your dataset. Do not treat `capacity_factor` as an
alias for `cf`, or assume county and coordinate columns exist. Check data
coverage and cost definitions before changing regional aggregations.

### Generation Profiles

Use CSV or Parquet with tidy columns `site_id`, `time_index`, `value`, and
optionally `weather_year`. Values are generation per unit of capacity. IDs match
`cpa_id` directly, or map through `site_map`: its first column is the candidate ID
and its second column the profile ID (the Texas example uses `cpa_id,profile_id`).
Check mapping completeness; candidates missing from a supplied map are dropped.

Set `weather_year` to select the intended year(s), and `utc_offset` to align
profiles with demand. Wide profiles with site IDs as columns are deprecated and
cannot be filtered by `weather_year`. Without a profiles file, the flexible path
uses constant availability of 1; that is not a realistic wind or solar profile.

## Basic Configuration Pattern

### 1. Match the technology and edit one settings definition

Include the matching cost-table technology in the existing `new_resources` list:

```yaml
new_resources:
  - [LandbasedWind, Class3, Moderate, 1]
```

Each entry is `[technology, tech_detail, cost_case, unit_size]`, with unit size in
MW. Keep the other technologies your study needs; replacing this list removes
them. `technology: landbasedwind` in a cluster entry is the resource-group
identifier mapped from the new-build technology, not the full cost-table name.
For offshore wind, also match `turbine_type: fixed` or `floating` to the selected
technology detail and resource JSON.

Edit existing definitions rather than adding an override file with duplicate
top-level keys. A settings directory loads only its immediate `.yml` files,
uses shallow top-level replacement, and has no reliable override order. Lists
and nested dictionaries are not merged across files.

### 2. Apply one filter/bin/cluster recipe

The following uses features in the Texas wind metadata and preserves the
example's solar definition. Replace the existing `renewables_clusters` list in
the study's resource settings:

```yaml
renewables_clusters:
  - region: all
    technology: landbasedwind
    filter:
      - feature: lcoe
        max: 75
    bin:
      - feature: lcoe
        q: 2
        weights: capacity_mw
    cluster:
      - feature: profile
        method: agglomerative
        n_clusters: 2
  - region: all
    technology: utilitypv
    filter:
      - feature: lcoe
        max: 60
```

Here `75` retains the Texas example's illustrative wind screening assumption;
it is **not a recommended economic cutoff**. A high-LCOE site can still be valuable
because of its location or output timing. Omit the filter if all candidates
belong in the study, and document any exclusion you do use. Filter bounds are
inclusive, but the implementation ignores `min: 0` and `max: 0`.

The same illustrative screens appear in both published systems: `max: 75` for
wind and `max: 60` for solar in Texas, and `max: 130` for wind and `max: 50` for
solar in CA/AZ. The different numbers reflect different candidate datasets, not a
recommendation; screen on your own metadata's cost distribution.

Processing order is:

1. Select sites in each model region, including its aggregated base regions.
2. Apply all filters, then optional `min_capacity` selection.
3. Bin numeric features across filtered sites in that model region.
4. Separate categorical groups, if requested.
5. Cluster within each nonempty bin/group combination.
6. Sum site capacity and capacity-weight profiles and supported cost columns.

In this recipe, two capacity-weighted LCOE tiers are each split into up to two
profile clusters: **at most four new wind resources per model region**.
Weighted quantiles aim for similar MW, not exactly equal capacity; indivisible
sites and duplicate values can produce unequal or fewer bins. Aggregation
averages profiles; it does not select a single representative site.

### 3. Set regional scope deliberately

Use `region: all` for a common definition. To vary it, add a complete entry for
an exact model-region name, such as `TX2`; it replaces the `all` definition for
the same resource identifiers in that region. It does not inherit omitted
filters or bin settings. Avoid duplicate entries for the same region/resource.

Use a categorical `group` only to preserve a boundary **inside** a model region;
model regions are already processed separately. Scoping cluster entries is not
a general mechanism for excluding technologies; use `new_gen_not_available` for
that. It excludes new-build technologies from named model regions, renewable ones
included: an excluded renewable is dropped before clustering, and a cluster entry
that targets it in that region is skipped instead of being reported as unmatched.
See
[New-Build Resources — `new_gen_not_available`](../reference/settings/new-build.md#new_gen_not_available).

## Calculating Total Cluster Count

For one binning and one clustering step, the approximate upper bound per model
region is observed category combinations × bins × requested clusters. Empty
combinations, duplicate bin edges, and too few sites reduce the count.

- No `group`, `bin`, or `cluster`: each retained site remains a resource.
- `group` and/or `bin`, without `cluster`: one resource per nonempty combination.
- `bin: [{feature: lcoe, q: 1}]`: aggregate all retained sites into one resource.
- `cluster` only: up to `n_clusters` resources per region.

Do not filter sites merely to meet a resource-count target: that changes the
candidate supply. First reduce bin/cluster counts or unnecessary groups, then
compare capacity, cost ranges, profile diversity, runtime, and model outcomes.

## Run and inspect the result

For the Texas example, run from the examples root (use `settings_wind` instead
of `settings` if following the tutorial's separate-settings exercise):

```bash
validate_powergenome --settings_file example_systems/TX-3-zone/settings
run_powergenome \
  --settings_file example_systems/TX-3-zone/settings \
  --results_folder renewable_test
```

The first planning period's default GenX outputs are:

- `renewable_test/base/Inputs/Inputs_p1/resources/Vre.csv`: resource attributes.
- `renewable_test/base/Inputs/Inputs_p1/system/Generators_variability.csv`: profiles
  named by `Resource`, aligned with modeled time steps.
- `renewable_test/base/Inputs/Inputs_p1/extra_outputs/`: site-to-cluster assignment
  CSVs for checking membership.

Other studies can use different case-directory names. The legacy flat generator
layout is not the default; see [Output Format](../explanation/output-format.md).

Check new-build wind separately from existing wind. Compare `Max_Cap_MW` totals
with eligible site capacity, count resources by region, and inspect
`Inv_Cost_per_MWyr` ($/MW-year), not just the metadata's screening `lcoe`.
Profile columns must match the written `Resource` names. A simple mean over the
written hours is **not an annual capacity factor** when the profile covers only
a sample of hours. See the
[tutorial's output checks](../tutorials/renewable-resources.md#step-5-compare-the-outputs).

## Caching for speed

The generator workflow defaults both controls to `true`; the published Texas
example explicitly sets both to `false`. To enable reads and writes, use:

```yaml
cache_resource_clusters: true
use_resource_clusters_cache: true
```

Caches live under `input_folder/cluster_assignments/`, in a resource-path-specific
subfolder. Reading and writing are separate: set `use_resource_clusters_cache`
to `false` to recompute; `cache_resource_clusters: false` only disables writes.

Keys include region, technology, cluster settings, timezone, weather year, and
metadata/profile file hashes. The site-map file is not separately hashed. After
changing a site map, remove the affected cache files before regenerating them;
disabling reads alone recomputes but does not overwrite existing same-key files.

## Optional refinements

### Binning Methods

Use one bin-count option per entry; provide a numeric `feature` column:

| Option | Behavior |
| --- | --- |
| `q: 3` | Three quantile bins; add `weights: capacity_mw` to balance by MW rather than site count |
| `bins: 3` | Three equal-width feature ranges; `weights` has no effect |
| `bins: [0, 30, 50, 75]` | Explicit feature-value edges |
| `q: [0, 0.5, 1]` | Explicit quantile edges |
| `mw_per_bin: 10000` | Equal-width count from total MW divided by this value, truncated, minimum one |
| `mw_per_q: 10000` | Quantile count from the same calculation |

Explicit outer bin edges are expanded to cover the data range, with warnings;
**bin edges are not screening thresholds**. Multiple bin entries subdivide
sites sequentially, before categorical grouping. Keep one dimension until you
have checked its effect.

### Clustering on Generation Profiles

Use `feature: profile` with `agglomerative` (aliases `agg`, `hierarchical`) as in
the main recipe. For metadata instead of profiles:

- Agglomerative clustering accepts one scalar numeric feature, such as `cf`.
- `kmeans` accepts a list of numeric columns, even for one feature, such as
  `feature: [cf]`; features are standardized. Latitude/longitude can be included
  only if present in your metadata. Use agglomerative for profile clustering.
- Set `n_clusters` explicitly, or use `mw_per_cluster` to derive a count from
  capacity within each bin/group (truncated, minimum one).

### Modify Costs by Group {#step-5-modify-costs-by-group}

Group modifiers are optional, after aggregation. For example, if your metadata
has `county`, put `group: [county]` in the same cluster definition and add:

```yaml
group_modifiers:
  - group: county
    group_value: Example_County
    Inv_Cost_per_MWyr: [mul, 1.1]
```

Use only documented cost adjustments. The group must appear in `group` so its
value enters the cluster label. Matching is by case-insensitive substring of
that label: inspect matches carefully. Scalars replace values; two-item lists
support `add`, `sub`, `mul`, and `truediv`.

Annual technology investment costs have already been calculated here. Modify
`Inv_Cost_per_MWyr` to change that component; changing `capex_mw` alone does not
recalculate it. `interconnect_annuity` is a separate site-cost component added
later. Avoid double counting it. To modify technology capital cost **before**
annualization, use `resource_modifiers` with exact `technology` **and**
`tech_detail` selectors, as described in
[New-Build Resources](../reference/settings/new-build.md#resource_modifiers).

### Selecting a limited candidate supply

`min_capacity` retains the lowest-`lcoe` sites up to the first site meeting the
requested MW, after filters. It is not a minimum build requirement, and can
overshoot because sites are indivisible. It requires an `lcoe` column. Use it
only when deliberately limiting candidate supply, not to choose cluster size.

The legacy `precluster_renewables`/`max_clusters`/`max_lcoe` resource-tree workflow
is separate from this guide. Do not mix its keys into the flexible recipe.

## Troubleshooting

| Symptom | Check first |
| --- | --- |
| No resource group matches | JSON technology/other identifiers, matching `new_resources`, file paths, and current working directory |
| No or missing sites | Region names, nonpositive capacity, filters, site-map coverage, and requested weather years |
| Far more resources than expected | Omitting all aggregation settings preserves individual sites; use one bin to aggregate |
| Fewer clusters than requested | Empty category combinations, duplicate feature/bin values, or fewer sites than requested clusters |
| Unexpected cost or location | Inspect site assignments and inputs; expensive sites can be valuable, so do not automatically tighten screening |
| Changed mappings have no effect | Recompute and remove affected stale cache entries; see caching above |

## Related Settings

- [New-Build Resources](../reference/settings/new-build.md): cost technologies,
  modifiers, and renewable settings.
- [Resource Tags](../reference/settings/resource-tags.md): model classifications.
- [Regions](../reference/settings/regions.md): regional aggregation.
- [Data Tables](../reference/settings/data-tables.md): source-table configuration.

## Next Steps

Run the [Texas wind tutorial](../tutorials/renewable-resources.md), then test a
second aggregation resolution while holding candidate supply fixed. Record
which differences matter for your study before adding groups or more features.
