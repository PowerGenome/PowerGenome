# Working with Renewable Resources

This tutorial teaches you how to configure wind and solar resources in PowerGenome, including how to select high-quality sites, cluster resources, and customize costs by region.

**Time**: ~45 minutes
**Prerequisites**: Complete [Getting Started](getting-started.md)

## What You'll Learn

- How PowerGenome represents renewable resources
- How to filter sites by resource quality (LCOE, capacity factor)
- How to cluster many sites into representative resources
- How to configure interconnection and transmission costs
- How to customize renewable resources by region

## Understanding Renewable Resource Modeling

PowerGenome models renewables differently than thermal generators:

**Thermal generators** (gas, coal, nuclear)
: Defined by technology, cost case, and size from NREL ATB. Dispatchable with heat rates and fuel costs.

**Renewable resources** (wind, solar)
: Site-specific with generation profiles, location costs (spur lines, transmission), and variable output.

### Resource Groups vs Resource Clusters

**Resource groups**
: Pre-computed renewable resource sites with hourly generation profiles, typically from NREL or other datasets. Each site has characteristics like LCOE, capacity factor, spur line distance.

**Resource clusters**
: Aggregations of similar resource group sites into a single representative resource for the capacity expansion model. Reduces model size while preserving resource quality diversity.

!!! example "Why Cluster?"
    A region might have 1,000 potential wind sites. Clustering groups them into 5-10 representative "bins" based on cost or quality, making the optimization tractable while maintaining diversity.

## Step 1: Understanding Resource Group Data

PowerGenome expects renewable resource data with these columns:

**Required**:

- `region` - Model region name
- `technology` - Technology type (landbasedwind, utilitypv, offshorewind)
- `cluster` - Unique identifier for the resource cluster
- `lcoe` or `cost` - Levelized cost of energy ($/MWh)
- `mw` or `capacity` - Available capacity (MW)

**Optional but recommended**:

- `capacity_factor` - Annual average capacity factor
- `spur_miles` - Distance to nearest transmission (miles)
- `offshore_spur_miles` - Offshore cable distance
- `interconnect_capex_mw` - Connection cost ($/MW)

PowerGenome includes sample resource group data in test systems. Check the structure:

```python
import pandas as pd

# Load example resource groups
resource_groups = pd.read_csv(
    "tests/test_system/test_data/resource_groups/test_resource_groups.csv"
)

print(resource_groups.columns)
print(resource_groups.groupby(['region', 'technology']).size())
```

## Step 2: Basic Renewable Configuration

Add renewables to your new-build resources in `settings/resources.yml`:

```yaml
# Include renewable technologies in new-build options
new_resources:
  - [NaturalGas, CCAvgCF, Moderate, 500]
  - [LandbasedWind, Class3, Moderate, 1]
  - [UtilityPV, Class1, Moderate, 1]
  - [OffShoreWind, Class12, Moderate, 1]
```

Note the `1` MW size - this indicates the unit size for unit commitment. GenX does not apply unit commitment to renewable resources, so the value here is not important.

## Step 3: Configure Renewable Clusters

The `renewables_clusters` parameter controls how resource group sites are filtered and aggregated.

### Simple Example: All Sites

Include all available sites without filtering:

```yaml
renewables_clusters:
  - region: all  # Apply to all regions
    technology: landbasedwind
    bin:
      - feature: lcoe
        weights: capacity_mw
        q: 1  # One bin = all sites aggregated
```

This creates a single wind resource per region with weighted-average characteristics.

### Filter by Cost

Include only economically attractive sites:

```yaml
renewables_clusters:
  - region: all
    technology: landbasedwind
    filter:
      - feature: lcoe
        max: 60  # Only sites with LCOE ≤ $60/MWh
    bin:
      - feature: lcoe
        weights: capacity_mw
        q: 3  # Create 3 cost bins (low/mid/high)
```

This creates 3 wind resources per region representing low, medium, and high cost sites.

### Filter by Capacity Factor

Select high-quality wind resources:

```yaml
renewables_clusters:
  - region: all
    technology: landbasedwind
    filter:
      - feature: cf
        min: 0.35  # Only sites with CF ≥ 35%
      - feature: lcoe
        max: 70
    bin:
      - feature: cf
        weights: capacity_mw
        q: 4  # 4 bins by capacity factor
```

### Multiple Filters

Combine filters to select sites meeting all criteria:

```yaml
renewables_clusters:
  - region: all
    technology: utilitypv
    filter:
      - feature: lcoe
        max: 50
      - feature: spur_miles  # Close to transmission
        max: 10
    bin:
      - feature: lcoe
        weights: capacity_mw
        q: 2
```

## Step 4: Region-Specific Configuration

Different regions may have different resource quality. Configure separately:

```yaml
renewables_clusters:
  # Great Plains: Excellent wind, more bins
  - region: midwest
    technology: landbasedwind
    filter:
      - feature: lcoe
        max: 50  # Abundant low-cost wind
    bin:
      - feature: lcoe
        weights: capacity_mw
        q: 5  # High resolution

  # Northeast: Limited wind, fewer bins
  - region: northeast
    technology: landbasedwind
    filter:
      - feature: lcoe
        max: 80  # Higher cost threshold
    bin:
      - feature: lcoe
        weights: capacity_mw
        q: 2  # Coarser resolution
```

## Step 5: Binning Methods

PowerGenome supports different clustering approaches:

### Quantile Binning (Default)

Divide sites into equal-capacity bins:

```yaml
bin:
  - feature: lcoe
    weights: capacity_mw  # Equal capacity per bin
    q: 4  # 4 bins (quantiles)
```

Each bin has approximately the same total MW capacity.

### Custom Bin Edges

Specify exact bin boundaries:

```yaml
bin:
  - feature: lcoe
    weights: capacity_mw
    bins: [0, 30, 50, 70, 100]  # Create 4 bins with these boundaries
```

Useful for aligning with policy thresholds or known resource quality tiers.

## Step 6: Caching for Speed

Processing large resource group datasets can be slow. PowerGenome will cache renewable clustering results in the `extra_inputs` folder. The cached results are uniquely identified based on the renewable cluster specifications, the `utc_offset` value, and the hash value of resource group files.

Enable caching:

```yaml
# Cache processed clusters
cache_resource_clusters: true

# Use cached clusters (skip processing)
use_resource_clusters_cache: true
```

**How caching works**:

1. First run: Process resource groups → save to cache → use in model
2. Subsequent runs: Load from cache → use in model (much faster)

Delete cache files manually if desired or set `cache_resource_clusters: false`.

## Step 7: Run and Examine Outputs

```bash
run_powergenome \
  --settings_file settings \
  --results_folder renewable_test
```

### Examine Renewable Resources

```python
import pandas as pd

gens = pd.read_csv("renewable_test/p1/Inputs/Inputs_p1/Generators_data.csv")

# Filter to renewables
renewables = gens[gens['technology'].isin(['landbasedwind', 'utilitypv', 'offshorewind'])]

print(f"Total renewable clusters: {len(renewables)}")
print(f"\nBy technology:")
print(renewables.groupby('technology').size())

print(f"\nBy region and technology:")
print(renewables.groupby(['region', 'technology']).size())

# Check cost range
print(f"\nWind investment costs ($/MW-yr):")
wind = renewables[renewables['technology'] == 'landbasedwind']
print(f"  Min: ${wind['Inv_Cost_per_MWyr'].min():,.0f}")
print(f"  Max: ${wind['Inv_Cost_per_MWyr'].max():,.0f}")

# Capacity available
print(f"\nTotal wind capacity available: {wind['Max_Cap_MW'].sum():,.0f} MW")
```

### Check Generation Profiles

```python
# Load variability profiles
profiles = pd.read_csv("renewable_test/p1/Inputs/Inputs_p1/Generators_variability.csv")

# Get wind resources
wind_resources = renewables[renewables['technology'] == 'landbasedwind']['Resource'].tolist()

# Check capacity factors
for resource in wind_resources:
    if resource in profiles.columns:
        cf = profiles[resource].mean()
        print(f"{resource}: CF = {cf:.2%}")
```

## Advanced Topics

### Hierarchical Clustering

For more sophisticated resource aggregation:

```yaml
renewables_clusters:
  - region: all
    technology: landbasedwind
    filter:
      - feature: lcoe
        max: 70
    cluster:
      - feature: cf  # Cluster on a single feature
        method: hierarchical
        n_clusters: 5
```

This uses hierarchical clustering to group sites by both cost and quality.

### Combining with Modified Resources

Adjust renewable costs after clustering:

```yaml
resource_modifiers:
  wind_bonus:
    technology: landbasedwind
    Inv_Cost_per_MWyr: [mul, 0.9]  # 10% investment tax credit equivalent
```

### Regional Technology Restrictions

Exclude renewables from specific regions:

```yaml
new_gen_not_available:
  desert_region:
    - LandbasedWind_*  # No wind (low capacity factors)
  mountain_region:
    - OffShoreWind_*  # No offshore wind (landlocked)
```

## Common Issues

### No Renewable Resources Generated

**Problem**: Model has no wind/solar resources

**Possible causes**:

1. No `renewables_clusters` configured
2. Filters too restrictive (no sites pass)
3. Technology not in `new_resources`
4. Missing resource group data

**Solutions**:

```yaml
# Check data availability
renewables_clusters:
  - region: all
    technology: landbasedwind
    # No filters - include everything
    bin:
      - feature: lcoe
        weights: capacity_mw
        q: 1
```

### Too Many Resources

**Problem**: Hundreds of renewable clusters, model too large

**Solution**: Use fewer bins or stricter filters:

```yaml
renewables_clusters:
  - region: all
    technology: landbasedwind
    filter:
      - feature: lcoe
        max: 50  # Only best sites
    bin:
      - feature: lcoe
        weights: capacity_mw
        q: 2  # Just 2 bins per region
```

## Renewable Clustering Parameters Reference

**`renewables_clusters`** - List of dictionaries, each with:

- **`region`** (required): Model region name or "all"
- **`technology`** (required): landbasedwind, utilitypv, offshorewind, geothermal
- **`filter`** (optional): List of filters to apply
  - **`feature`**: Column name to filter on (lcoe, capacity_factor, spur_miles, etc.)
  - **`min`**: Minimum value (inclusive)
  - **`max`**: Maximum value (inclusive)
- **`bin`** or **`cluster`** (required): How to group sites
  - **`feature`**: Column(s) to bin/cluster on
  - **`weights`**: Column for weighting (usually "mw")
  - **`q`**: Number of bins/clusters
  - **`method`**: *Only in `cluster`* kmeans or agg (hierarchical)

See [New Build Resources Settings](../reference/settings/new-build.md) for complete documentation.

## Next Steps

- **Configure policies**: [Resource Tags](../reference/settings/resource-tags.md) for renewable portfolio standards
- **Multi-region renewables**: [Building a Multi-Region Model](multi-region-model.md)
- **Offshore wind details**: [New Build Settings](../reference/settings/new-build.md)
- **Resource group data**: [Data Tables](../how-to/configure-data-tables.md)

## Further Reading

- NREL ATB: [Annual Technology Baseline](https://atb.nrel.gov/)
- Resource data sources: [NREL reV](https://www.nrel.gov/gis/renewable-energy-potential.html)
- GenX renewable modeling: [GenX Documentation](https://genxproject.github.io/GenX/)
