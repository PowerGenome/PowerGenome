# Configure Renewable Resource Clusters

This guide explains how to set up renewable resource clusters (wind and solar sites) using PowerGenome's flexible clustering system. The clustering approach allows you to divide renewable resources by performance characteristics, geography, and cost attributes.

## Overview

Renewable clustering divides potential wind and solar sites into discrete groups that can be invested in by the capacity expansion model. The clustering system supports four complementary mechanisms:

1. **Filters** - Exclude sites based on criteria (e.g., only sites with LCOE < $50/MWh)
2. **Groups** - Divide sites by categorical features (e.g., wind class, county)
3. **Bins** - Divide sites by continuous features into ranges (e.g., LCOE quartiles)
4. **Clusters** - Apply k-means clustering within each group/bin combination

**Total clusters** = (# groups) × (# bins) × (# clusters per combination)

For example: 3 wind classes × 4 LCOE bins × 2 k-means clusters = **24 total resource clusters**

## When to Use Renewable Clusters

Use renewable clusters when:

- **Geographic diversity** matters (wind/solar in different regions)
- **Cost variation** is significant (want to build cheapest sites first)
- **Performance differences** exist (Class 3 vs Class 7 wind)
- **Resource limits** apply (land availability in each county)

Skip clustering if you have a simple system where all renewable sites are similar.

## Basic Configuration Pattern

Renewable clusters are defined in the `renewable_clusters` section:

```yaml
renewable_clusters:
  - region: all             # Required: region(s) to include
    technology: landbasedwind  # Required: technology name
    filter: [...]          # Optional: exclude sites
    group: [...]           # Optional: categorical divisions
    bin: [...]             # Optional: continuous feature ranges
    cluster: [...]         # Optional: k-means within groups
    group_modifiers: [...]  # Optional: adjust costs by group
```

## Step 1: Start with Filters (Recommended)

**Filters remove sites** that don't meet quality thresholds. If your resource data includes a cost metric like `lcoe`, **start by filtering on it**.

### Why Filter on LCOE?

LCOE (Levelized Cost of Energy) combines capacity factor and capital costs. Filtering on LCOE ensures you only consider economically viable sites, reducing computational complexity.

```yaml
renewable_clusters:
  - region: all
    technology: landbasedwind
    filter:
      - feature: lcoe
        max: 50  # Only sites with LCOE ≤ $50/MWh
```

**Filter structure**: Each filter in the list must have:

- `feature`: Column name from resource data (e.g., `lcoe`, `capacity_factor`, `dist_to_tx_km`)
- `max`: Maximum value (optional, sites with feature > max are excluded)
- `min`: Minimum value (optional, sites with feature < min are excluded)

**Example filters**:

```yaml
filter:
  - feature: lcoe
    max: 50           # Exclude sites with LCOE > $50/MWh
  - feature: capacity_factor
    min: 0.3          # Exclude sites with CF < 30%
  - feature: dist_to_tx_km
    max: 100          # Exclude sites > 100 km from transmission
```

## Step 2: Add Bins for Cost Stratification (Recommended)

After filtering, **bin on LCOE** to divide resources into cost tiers. This allows the model to build the cheapest sites first.

### LCOE Binning Example

```yaml
renewable_clusters:
  - region: all
    technology: landbasedwind
    filter:
      - feature: lcoe
        max: 50
    bin:
      - feature: lcoe
        q: 4          # Create 4 quantile bins (quartiles)
        weights: mw   # Weight by capacity when creating bins
```

**Result**: Sites divided into 4 bins (e.g., $0-$30, $30-$38, $38-$44, $44-$50 LCOE ranges)

**Bin structure**: Each bin entry is a dictionary with:

- `feature`: Column name to bin on (required)
- `bins`: Integer number of equal-width bins OR list of bin edges (e.g., `[0, 30, 50, 75]`)
- `q`: Integer number of quantile bins OR list of quantile edges (e.g., `[0, 0.25, 0.5, 0.75, 1.0]`)
- `weights`: Column to weight bins by (optional, e.g., `mw` to weight by capacity)
- `mw_per_bin`: Alternative to `bins`/`q` - calculate number of bins from total MW (optional)
- `mw_per_q`: Alternative to `bins`/`q` - calculate number of quantiles from total MW (optional)

**Note**: Use either `bins` OR `q`, not both. If `weights` is specified with `q`, weighted quantiles are calculated.

### Binning Methods

**Quantile binning** (recommended for balanced clusters):

```yaml
bin:
  - feature: lcoe
    q: 3            # 3 quantiles (terciles)
    weights: mw     # Equal MW in each bin
```

**Equal-width binning** (may create unbalanced bins):

```yaml
bin:
  - feature: lcoe
    bins: 4  # 4 equal-width bins across LCOE range
```

**Explicit bin edges**:

```yaml
bin:
  - feature: lcoe
    bins: [0, 30, 50, 75]  # Custom bin boundaries
```

**Capacity-based bins**:

```yaml
bin:
  - feature: lcoe
    mw_per_bin: 10000  # ~10 GW per bin, number of bins calculated automatically
    weights: mw
```

### Binning on Other Features

```yaml
# Bin by capacity factor (performance tiers)
bin:
  - feature: capacity_factor
    q: 3
    weights: mw  # Equal capacity in each performance tier

# Bin by distance to transmission
bin:
  - feature: dist_to_tx_km
    bins: [0, 25, 50, 100, 200]  # Explicit distance ranges
```

### Multiple Bins

You can apply multiple binning dimensions sequentially:

```yaml
bin:
  - feature: lcoe
    q: 3
    weights: mw
  - feature: capacity_factor
    q: 2  # Further divide each LCOE bin by CF
```

**Result**: 3 LCOE bins × 2 CF bins = 6 total bin combinations

## Step 3: Use Groups for Categorical Divisions

**Groups divide sites by categorical features** like wind class, county, or interconnection zone.

### Wind Class Grouping

Most common use case - divide by resource quality:

```yaml
renewable_clusters:
  - region: all
    technology: landbasedwind
    group:
      - class  # Divides by wind class (Class 3, Class 4, etc.)
    cluster:
      - feature: cf
        n_clusters: 2
        method: agg
```

**Result**: Each class gets 2 k-means clusters (e.g., Class 3 → 2 clusters, Class 4 → 2 clusters)

### Geographic Grouping

Divide by county or zone to respect land availability:

```yaml
renewable_clusters:
  - region: all
    technology: utilitypv
    group:
      - county  # One or more clusters per county
    cluster:
      - feature: lcoe
        n_clusters: 1  # Single representative site per county
        method: agg
```

### Multiple Groups

Combine categorical features (multiplies cluster count):

```yaml
renewable_clusters:
  - region: all
    technology: landbasedwind
    group:
      - class   # 3 wind classes
      - ipm_region  # 5 IPM regions (use different column than model region)
    cluster:
      - feature: cf
        n_clusters: 2
        method: agg
```

**Total clusters** = 3 classes × 5 regions × 2 k-means = **30 clusters**

## Step 4: Configure K-Means Clustering

After grouping/binning, **k-means clusters aggregate similar sites** within each combination.

### Basic Clustering

```yaml
renewable_clusters:
  - region: all
    technology: landbasedwind
    cluster:
      - feature: cf           # Cluster on capacity factor
        n_clusters: 3         # 3 clusters per group/bin
        method: agg           # Clustering algorithm
```

**Cluster structure**: Each cluster entry is a dictionary with:

- `feature`: Column name(s) to cluster on (required). Can be a string or list of strings. Use `profile` to cluster on generation profiles.
- `method`: Clustering algorithm - `kmeans`, `agglomerative`/`agg`, `max_distance` (required)
- `n_clusters`: Number of clusters (required)
- `mw_per_cluster`: Alternative to `n_clusters` - calculate number of clusters from total MW (optional)

### Clustering on Multiple Features

Specify a list of features to cluster on multiple columns simultaneously:

```yaml
cluster:
  - feature: [latitude, longitude, capacity_factor]
    n_clusters: 2
    method: kmeans
```

This creates 2 clusters based on geographic proximity AND performance similarity.

**Common feature choices**:

- `latitude`, `longitude`: Geographic proximity
- `capacity_factor` or `cf`: Performance similarity
- `lcoe`: Cost similarity
- `elevation`, `slope`: Terrain characteristics
- `profile`: Generation time series (only works with `agglomerative` method)

### Clustering on Generation Profiles

```yaml
cluster:
  - feature: profile        # Cluster on hourly generation patterns
    n_clusters: 3
    method: agglomerative   # Required for profile clustering
```

### Alternative Clustering Methods

```yaml
# Hierarchical/agglomerative clustering
cluster:
  - feature: cf
    n_clusters: 4
    method: agg              # Short for agglomerative

# Max-distance clustering (ensures spatial diversity)
cluster:
  - feature: [latitude, longitude]
    n_clusters: 5
    method: max_distance
```

### Limiting Total Clusters

**Note**: The total number of clusters is the product of groups × bins × n_clusters. To manage cluster count:

1. Reduce the number of groups
2. Reduce the number of bins (`q` or `bins` value)
3. Reduce `n_clusters`
4. Be selective about which regions/technologies to cluster

```yaml
renewable_clusters:
  - region: all
    technology: landbasedwind
    group: [class, ipm_region]  # Could create 3 × 5 = 15 combinations
    bin:
      - feature: lcoe
        q: 2                # × 2 bins = 30 combinations (reduced from 4)
    cluster:
      - feature: cf
        n_clusters: 1       # × 1 cluster = 30 total
        method: agg
```

Reducing bins/clusters can simplify the model while preserving geographic and cost diversity.

## Step 5: Modify Costs by Group

**Group modifiers adjust costs for specific groups** after clustering (e.g., higher interconnection costs in certain regions).

```yaml
renewable_clusters:
  - region: all
    technology: landbasedwind
    group: [class]
    cluster:
      - feature: cf
        n_clusters: 2
        method: agg
    group_modifiers:
      - group: class
        group_value: 3
        capex_mw: [mul, 1.1]   # 10% higher capex for Class 3 wind
        fixed_o_m_mw: [add, 5] # +$5/MW-yr O&M
      - group: class
        group_value: 7
        capex_mw: [mul, 0.95]  # 5% lower capex for Class 7
```

### Modifier Operations

- `[mul, factor]`: Multiply existing value
- `[add, value]`: Add to existing value
- `[sub, value]`: Subtract from existing value
- Scalar: Replace value entirely

```yaml
group_modifiers:
  - group: ipm_region
    group_value: CA_N
    capex_mw: [mul, 1.2]      # 20% cost increase
    interconnect_mw: [add, 50] # +$50/MW interconnection
  - group: ipm_region
    group_value: TX
    capex_mw: 800000          # Set to $800k/MW (replaces value)
```

## Calculating Total Cluster Count

Understanding how cluster count multiplies is critical for managing model size.

### Formula

**Total Clusters** = (# filter-passing groups) × (# bins) × (# k-means clusters)

- If no groups: 1 group assumed
- If no bins: 1 bin assumed
- If no k-means: 1 cluster assumed

### Examples

**Simple k-means only**:

```yaml
cluster:
  - feature: cf
    n_clusters: 5
    method: agg
```

→ **5 total clusters**

**Groups + k-means**:

```yaml
group: [class]  # 3 classes in data
cluster:
  - feature: cf
    n_clusters: 2
    method: agg
```

→ 3 × 2 = **6 total clusters**

**Bins + k-means**:

```yaml
bin:
  - feature: lcoe
    q: 4
cluster:
  - n_clusters: 3
```

→ 4 × 3 = **12 total clusters**

**Groups + bins + k-means**:

```yaml
group: [region]  # 5 regions
bin:
  - feature: lcoe
    q: 3
cluster:
  - n_clusters: 2
```

→ 5 × 3 × 2 = **30 total clusters**

**Groups + bins (no k-means)**:

```yaml
group: [class, county]  # 3 classes, 20 counties
bin:
  - feature: lcoe
    q: 4
```

→ 3 × 20 × 4 × 1 = **240 total clusters** (potentially too many!)

## Complete Examples

### Example 1: Simple Wind Clustering (Recommended Starting Point)

```yaml
renewable_clusters:
  - region: all
    technology: landbasedwind
    # Filter to economically viable sites
    filter:
      - feature: lcoe
        max: 50
      - feature: capacity_mw
        min: 5  # Sites must be ≥ 5 MW

    # Divide into 4 cost tiers
    bin:
      - feature: lcoe
        q: 4          # 4 quantiles (quartiles)
        weights: mw   # Equal capacity in each bin

    # 2 k-means clusters per cost tier (geographic diversity)
    cluster:
      - feature: [latitude, longitude, capacity_factor]
        n_clusters: 2
        method: kmeans
```

**Result**: 4 bins × 2 clusters = **8 wind resource clusters**

### Example 2: Wind with Class Groups

```yaml
renewable_clusters:
  - region: all
    technology: landbasedwind
    filter:
      - feature: lcoe
        max: 60

    # Group by wind class (categorical)
    group: [class]

    # Bin by cost within each class
    bin:
      - feature: lcoe
        q: 3
        weights: mw

    # 1 representative cluster per class/bin
    cluster:
      - feature: cf
        n_clusters: 1
        method: agg

    # Adjust costs for lower-quality wind
    group_modifiers:
      - group: class
        group_value: 3
        capex_mw: [mul, 1.15]  # 15% higher costs
```

**Result**: 3 classes × 3 bins × 1 cluster = **9 wind clusters**

### Example 3: Solar with Geographic Groups

```yaml
renewable_clusters:
  - region: all
    technology: utilitypv
    filter:
      - feature: capacity_factor
        min: 0.18
      - feature: lcoe
        max: 40

    # Group by county for land use limits
    group: [county]

    # No binning (costs similar within counties)

    # Single representative site per county
    cluster:
      - feature: [latitude, longitude]
        n_clusters: 1
        method: kmeans

    # Higher interconnection costs in remote counties
    group_modifiers:
      - group: county
        group_value: Rural_County_A
        interconnect_mw: [add, 100]
      - group: county
        group_value: Rural_County_B
        interconnect_mw: [add, 150]
```

**Result**: 1 cluster per county (e.g., 25 counties = **25 solar clusters**)

### Example 4: Offshore Wind (Performance-Based)

```yaml
renewable_clusters:
  - region: all
    technology: offshorewind
    filter:
      - feature: capacity_factor
        min: 0.35  # Only high-quality sites
      - feature: dist_to_shore_km
        max: 50   # Within 50 km of shore

    # Bin by performance (no geographic groups - limited areas)
    bin:
      - feature: capacity_factor
        bins: 3       # 3 equal-width bins

    # 2 clusters per performance tier
    cluster:
      - feature: [latitude, longitude, water_depth]
        n_clusters: 2
        method: agg
```

**Result**: 3 CF bins × 2 clusters = **6 offshore wind clusters**

### Example 5: Complex Multi-Region System

```yaml
renewable_clusters:
  - region: [TX, OK, KS, NM]  # Great Plains only
    technology: landbasedwind
    filter:
      - feature: lcoe
        max: 55

    # Group by state and wind class
    group: [region, class]

    # Bin by cost within each group
    bin:
      - feature: lcoe
        q: 2           # Keep bins low with multiple groups
        weights: mw

    # Small number of clusters per group/bin
    cluster:
      - feature: cf
        n_clusters: 1
        method: agg

    group_modifiers:
      - group: ipm_region
        group_value: TX_Class7
        capex_mw: [mul, 0.9]   # Best TX wind
      - group: ipm_region
        group_value: NM_Class3
        capex_mw: [mul, 1.1]   # Lower quality NM wind
```

**Result**: 4 regions × ~2 classes × 2 bins × 1 cluster = **~16 clusters**

## Best Practices

### 1. Start Simple, Add Complexity

**Phase 1** - Filter + Bins:

```yaml
filter:
  - feature: lcoe
    max: 50
bin:
  - feature: lcoe
    q: 4
cluster:
  - feature: cf
    n_clusters: 2
    method: agg
```

**Phase 2** - Add groups if needed:

```yaml
filter:
  - feature: lcoe
    max: 50
group: [class]  # Add categorical division
bin:
  - feature: lcoe
    q: 3          # Reduce bins to manage count
cluster:
  - feature: cf
    n_clusters: 2
    method: agg
```

### 2. Always Filter on LCOE First (If Available)

Filtering removes uneconomical sites before clustering, reducing:

- Computational time
- Memory usage
- Irrelevant resource options

```yaml
# Good: Filter before clustering
filter:
  - feature: lcoe
    max: 60
cluster:
  - feature: cf
    n_clusters: 10  # Only clusters good sites
    method: agg

# Avoid: Clustering everything
cluster:
  - feature: cf
    n_clusters: 50  # Includes expensive sites
    method: agg
```

### 3. Use Quantile Binning for Balanced Clusters

Quantile binning ensures each cost tier has similar site counts:

```yaml
bin:
  - feature: lcoe
    q: 4          # Each bin has ~25% of sites (quartiles)
    weights: mw   # Can also weight by capacity
```

Equal-width binning may create bins with very few sites.

### 4. Match Cluster Count to Model Scale

**Small model** (1-5 regions):

- 5-15 clusters per technology
- Simple binning (2-3 bins)
- Single k-means cluster per bin

**Medium model** (5-15 regions):

- 15-40 clusters per technology
- Groups by region or class
- 2-3 bins × 1-2 k-means clusters

**Large model** (15+ regions):

- 40-100 clusters per technology
- Multiple groups (region + class)
- Keep bins and n_clusters low to manage total count
- Consider 1 k-means cluster per group/bin

### 5. Use Group Modifiers Sparingly

Only modify costs when you have specific regional data:

```yaml
# Good: Specific, data-driven adjustment
group_modifiers:
  - group: ipm_region
    group_value: CA_N
    interconnect_mw: [add, 75]  # Known higher interconnection costs

# Avoid: Arbitrary adjustments
group_modifiers:
  - group: ipm_region
    group_value: Region1
    capex_mw: [mul, 1.05]  # Why 5%?
```

### 6. Check Cluster Counts Before Running

Calculate expected clusters:

- Count unique values in grouping columns
- Multiply by bins and k-means clusters
- Ensure total is reasonable (< 100 per technology for most models)

### 7. Balance Geographic and Cost Diversity

**Good balance**:

```yaml
group: [region]      # Geographic diversity
bin:
  - feature: lcoe
    q: 3             # Cost diversity
cluster:
  - n_clusters: 1    # Simple representative
```

**Too geographic-heavy** (ignores costs):

```yaml
group: [region, county, class]  # Very specific locations
# No cost binning - might select expensive sites
```

**Too cost-heavy** (ignores geography):

```yaml
bin:
  - feature: lcoe
    q: 10            # Fine cost resolution
# No groups - might concentrate in one region
```

## Troubleshooting

### Too Many Clusters

**Problem**: 200+ clusters slow down the model

**Solutions**:

1. Reduce number of bins (lower `q` or `bins` value)
2. Use fewer groups
3. Reduce `n_clusters`

```yaml
# Before: 5 regions × 4 bins × 3 clusters = 60
group: [region]
bin:
  - feature: lcoe
    q: 4
cluster:
  - feature: cf
    n_clusters: 3
    method: agg

# After: 5 regions × 2 bins × 2 clusters = 20
group: [region]
bin:
  - feature: lcoe
    q: 2
cluster:
  - feature: cf
    n_clusters: 2
    method: agg
```

### All Clusters in One Region

**Problem**: All selected sites in the same area

**Solutions**:

1. Add geographic groups
2. Include lat/lon in clustering features
3. Use `max_distance` clustering method

```yaml
# Solution 1: Group by region
group: [region]
cluster:
  - feature: cf
    n_clusters: 2
    method: kmeans

# Solution 2: Geographic features
cluster:
  - feature: [latitude, longitude, lcoe]
    n_clusters: 3
    method: kmeans

# Solution 3: Max-distance
cluster:
  - feature: [latitude, longitude]
    method: max_distance
    n_clusters: 5
```

### Expensive Sites Selected

**Problem**: High-cost clusters appear in optimal solution

**Solutions**:

1. Tighten `max_lcoe` filter
2. Add more cost bins
3. Check that bins are actually dividing by cost

```yaml
# Tighter filter
filter:
  - feature: lcoe
    max: 45  # Was 60

# More cost granularity
bin:
  - feature: lcoe
    q: 5     # Was 3
```

### Missing Expected Groups

**Problem**: Fewer clusters than expected

**Cause**: Not all group combinations exist in filtered data

**Solution**: Check resource data to verify group values exist after filtering

## Data Requirements

Renewable clustering requires **resource group profiles** with site-level data.

### Required Data Files

Located in `RESOURCE_GROUP_PROFILES` directory:

```
RESOURCE_GROUP_PROFILES/
├── LandbasedWind_Class1_resource_groups.parquet
├── UtilityPV_Class1_resource_groups.parquet
└── OffshoreWind_Class1_resource_groups.parquet
```

### Required Columns

**Minimum**:

- `region`: Model region name
- `latitude`, `longitude`: Coordinates
- `capacity_mw`: Site capacity
- `capacity_factor`: Average CF
- `profile_id`: Links to generation profile

**Recommended**:

- `lcoe`: For filtering and binning
- `class`: Wind/solar resource class
- `county`: For geographic grouping

**Optional**:

- `dist_to_tx_km`: Distance to transmission
- `elevation`, `slope`: Terrain features
- `interconnect_mw`: Interconnection cost

### Generation Profiles

Hourly profiles in same directory:

```
LandbasedWind_Class1_profile_0001.csv  # 8760 hourly CFs
LandbasedWind_Class1_profile_0002.csv
...
```

Each `profile_id` in resource groups must have a corresponding profile file.

## Related Settings

- **[New-Build Resources](../reference/settings/new-build.md)**: Base technology costs
- **[Resource Tags](../reference/settings/resource-tags.md)**: GenX resource classifications
- **[Regions](../reference/settings/regions.md)**: Model region definitions
- **[Data Tables](../reference/settings/data-tables.md)**: Table configuration

## Next Steps

1. **Verify data**: Check resource group files have required columns
2. **Start simple**: Filter + bin on LCOE with 2-3 k-means clusters
3. **Examine results**: Look at cluster sizes and locations
4. **Iterate**: Add groups or bins based on model needs
5. **Optimize**: Adjust cluster count for runtime vs. resolution trade-off
