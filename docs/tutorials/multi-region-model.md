# Building a Multi-Region Model

This tutorial shows you how to build a capacity expansion model with multiple regions connected by transmission. You'll learn how to define regions, configure transmission constraints, and set region-specific parameters.

**Time**: ~45 minutes
**Prerequisites**: Complete [Getting Started](getting-started.md)

## What You'll Build

A 3-region model with:

- Custom regional boundaries by aggregating base regions
- Inter-regional transmission with flow limits and costs
- Region-specific technology availability and costs
- Regional capacity reserve requirements

## Understanding Regions

PowerGenome uses a hierarchical regional structure:

**Base regions**
: The fundamental geographic units (e.g. from the ReEDS model). These are balancing authority-scale regions labeled `p1`, `p2`, `p3`, etc.

**Aggregated regions**
: Groups of multiple base regions combined together (e.g., `p1_2` combines base regions `p1` and `p2`)

**Model regions**
: The regions used in your study - can be either single base regions OR aggregated regions

**Region aggregations**
: The settings parameter that defines how base regions map to model regions

!!! info "Why Aggregate Regions?"
    Aggregating reduces model complexity while preserving key transmission constraints. For example, combining `p1` and `p2` into aggregated region `p1_2` reduces generator clusters and transmission pathways while maintaining major inter-regional flows.

## Step 1: Define Model Regions

Create `settings/regions.yml`:

```yaml
# List of regions in your model
model_regions:
  - northeast
  - midwest
  - south

# How base regions map to model regions
region_aggregations:
  northeast:
    - p129  # Maine
    - p130  # New Hampshire
    - p131  # Vermont
    - p132  # Massachusetts
    - p133  # Rhode Island
    - p134  # Connecticut
  midwest:
    - p74   # Wisconsin
    - p75   # Minnesota (north)
    - p76   # Minnesota (south)
    - p77   # Iowa (north)
    - p78   # Iowa (south)
  south:
    - p89   # Georgia
    - p90   # Alabama
    - p91   # Florida Panhandle
    - p92   # North Carolina
    - p93   # South Carolina
```

**Key points**:

- Each entry in `model_regions` must have a corresponding key in `region_aggregations`
- Base region names (p1, p2, etc.) must match those in your data tables
- Order matters for output organization but not model behavior

## Step 2: Configure Transmission

Transmission configuration in PowerGenome happens in two parts: **data tables** that define network topology and costs, and **settings** that control expansion and spur line costs.

### Transmission Data Tables

Network topology (which regions connect and with what capacity) comes from data tables, not YAML settings. You'll need two CSV files in your `extra_inputs/` folder:

**`ipm_tx_corrections.csv`** - Transmission capacity between regions:

```csv
region_from,region_to,firm_ttc_mw,notes
northeast,midwest,2000,Existing interconnection
midwest,south,3000,Main north-south corridor
northeast,south,1500,Eastern corridor
```

**`network_costs.csv`** - Line costs and losses:

```csv
start_region,dest_region,total_interconnect_annuity_mw,total_interconnect_cost_mw,total_line_loss_frac,dollar_year
northeast,midwest,15000,350000,0.08,2018
midwest,south,12000,280000,0.05,2018
northeast,south,18000,420000,0.10,2018
```

!!! tip "Pre-calculated Transmission Costs"
    PowerGenome can calculate transmission costs from centroid-to-centroid distances, but **we strongly recommend using pre-calculated costs** from least-cost-path analysis. See [Patankar et al. (2023)](https://www.sciencedirect.com/science/article/abs/pii/S2666278723000144) for the methodology.

### Transmission Settings

Add to `settings/transmission.yml` or `settings/data.yml`:

```yaml
# Transmission expansion controls
tx_expansion_per_period: 1.0  # Max expansion as multiple of existing (1.0 = can double)
tx_expansion_mw_per_period: 500  # Minimum expansion increment (MW)
```

### Understanding Transmission Parameters

**`transmission_constraints_table`**
: CSV or Parquet table defining network topology. Columns: `region_from`, `region_to`, `firm_ttc_mw`. PowerGenome loads this from `data_location` and aggregates across base regions if needed.

**`transmission_cost_table`**
: CSV or Parquet table with pre-calculated line costs and losses. Use this instead of distance-based approximations when possible.

**`tx_expansion_per_period`**
: Maximum fractional expansion of existing lines. Value of 1.0 means a 500 MW line can add 500 MW (doubling). Value of 0.5 allows 50% increase.

**`tx_expansion_mw_per_period`**
: Absolute expansion limit in MW. Useful for setting minimum buildable increments (e.g., one 230kV line). PowerGenome uses the *larger* of the two expansion parameters per line.

!!! warning "Network Topology from Data Tables"
    Unlike generator and demand configuration, transmission network structure is **not** defined in YAML settings values. The `transmission_constraints_table` determines which regions are connected.

## Step 3: Regional Technology Availability

Some technologies may not be available in all regions. Add to `settings/resources.yml`:

```yaml
# Technologies NOT available in each region
new_gen_not_available:
  northeast:
    - Coal_*  # No new coal
  south:
    - OffShoreWind_*  # No offshore wind (not coastal in this aggregation)
  midwest:
    - OffShoreWind_*
```

The `*` wildcard matches any tech_detail/cost_case combination.

## Step 4: Regional Cost Adjustments

Construction costs vary by region due to labor, materials, and permitting. Add regional multipliers:

```yaml
# Map model regions to EIA cost regions
cost_multiplier_region_map:
  northeast: TRE
  midwest: NPCCUPNY
  south: SERCSOES

# Point to cost multiplier file
cost_multiplier_fn: AEO_2020_regional_cost_corrections.csv

# Technologies to apply multipliers to
cost_multiplier_technology_map:
  CC - multi shaft: [NaturalGas_CCAvgCF]
  Solar PV - tracking: [UtilityPV_Class1]
  Wind: [LandbasedWind_Class3]
  Battery storage: [Battery_*]
```

!!! note "Cost Multiplier Files"
    PowerGenome includes default cost multiplier files in `data/cost_multipliers/`. These come from EIA's Annual Energy Outlook capital cost assumptions.

## Step 5: Regional Capacity Reserves

Some regions require capacity reserves (planning reserve margin):

```yaml
# Regional capacity reserve requirements
regional_capacity_reserves:
  CapRes_1:
    northeast: 0.15  # 15% reserve margin
  CapRes_2:
    midwest: 0.13    # 13%
  CapRes_3:
    south: 0.12      # 12%

# Transmission de-rate for capacity imports
# (only 95% of transmission capacity counts toward reserves)
cap_res_network_derate_default: 0.95
```

Add `CapRes_1`, `CapRes_2`, and `CapRes_3` to your resource tags so generators can contribute to reserves (see [Resource Tags](../reference/settings/resource-tags.md)). Because the capacity reserve constrains are specific to regions, these specific tag values will be included under `regional_tag_values`.

## Step 6: Regional Demand

Each region needs its own hourly load profile. Specify the demand data table in `settings/data.yml`:

```yaml
demand_table: load_curves.csv  # Or .parquet
```

The `demand_table` should contain hourly load profiles in **tidy format** (one row per region-hour observation):

```csv
time_index,weather_year,region,load_mw,year
1,2012,northeast,15234.5,2030
2,2012,northeast,14123.2,2030
3,2012,northeast,13890.4,2030
1,2012,midwest,18920.3,2030
2,2012,midwest,17845.1,2030
...
```

PowerGenome will load this table from `data_location` and use it to create the GenX `Load_data.csv` file. A "scenario" column can also be included in the demand table. Use the dictionary format with a `scenario` option to select a specific scenario:

```yaml
demand_table:
  table_name: load_curves.parquet
  scenario: base
```

See [Demand Settings](../reference/settings/demand.md) for details on load growth, demand response, and other demand parameters.

## Step 7: Run the Model

```bash
run_powergenome \
  --settings_file settings \
  --results_folder multi_region_test
```

## Examining Multi-Region Outputs

### Generators by Region

```python
import pandas as pd

gens = pd.read_csv("multi_region_test/p1/Inputs/Inputs_p1/Generators_data.csv")

# Capacity by region and technology
regional_capacity = gens.groupby(['region', 'technology'])['Existing_Cap_MW'].sum()
print(regional_capacity)

# New-build availability by region
new_build = gens[gens['New_Build'] == 1].groupby('region')['technology'].unique()
for region, techs in new_build.items():
    print(f"\n{region}: {list(techs)}")
```

### Transmission Network

```python
network = pd.read_csv("multi_region_test/p1/Inputs/Inputs_p1/Network.csv")

print(network[['Network_Lines', 'z1', 'z2', 'Line_Max_Flow_MW',
               'transmission_path_name', 'Line_Loss_Percentage']])

# Check if expansion is allowed
print(f"\nExpansion allowed: {network['Line_Max_Reinforcement_MW'].max() > 0}")
```

### Regional Demand

```python
load = pd.read_csv("multi_region_test/p1/Inputs/Inputs_p1/Load_data.csv")

# Total annual load by region
annual_load = load.sum()
print(annual_load)

# Peak load by region
peak_load = load.max()
print(f"\nPeak loads:\n{peak_load}")
```

## Common Pitfalls

### Region Name Mismatches

**Problem**: `KeyError: 'p22'` when running the model

**Cause**: Region name in settings doesn't match data tables

**Solution**: Check that:

- Base region names in `region_aggregations` match exactly (case-sensitive, format: `p<number>`)
- Model region names are used consistently in all settings parameters
- Data tables use the correct region column name

### Missing Transmission Lines

**Problem**: Model reports missing transmission connections between regions

**Cause**: Region pair not included in `transmission_constraints_table`

**Solution**: Add the missing line to your transmission constraints CSV with appropriate capacity

### Unbalanced Load

**Problem**: Some regions have no load or unrealistic load profiles

**Cause**: Incorrect region mapping in `historical_load_region_map`

**Solution**: Verify that all base regions in your aggregations are listed in the load region maps

## Regional Settings Reference

Parameters that accept model region names:

- `model_regions` - List of regions in study
- `region_aggregations` - How to combine base regions
- `new_gen_not_available` - Technology restrictions by region
- `cost_multiplier_region_map` - Regional cost adjustments
- `regional_capacity_reserves` - Reserve requirements by region
- `regional_tag_values` - Custom tag values by region (see [Resource Tags](../reference/settings/resource-tags.md))
- `regional_no_grouping` - Disable technology grouping in specific regions
- `regional_hydro_factor` - Hydro energy-to-power ratio by region
- `alt_num_clusters` - Override clustering settings by region
- `alt_growth_rate` - Custom load growth rates by region

See [Regions and Geography Settings](../reference/settings/regions.md) for complete documentation.

## Advanced Topics

### Sub-Regional Transmission

To model transmission within a region, split it into multiple model regions:

```yaml
model_regions:
  - northeast_coastal
  - northeast_inland
  - midwest
  - south

region_aggregations:
  northeast_coastal:
    - NENG_CT
  northeast_inland:
    - NENGREST
    - NENG_ME
  # ... rest of regions
```

Then add intra-northeast transmission routes.

### Regional Renewable Zones

Different regions can have different renewable resource quality:

```yaml
renewables_clusters:
  - region: northeast
    technology: landbasedwind
    filter:
      - feature: lcoe
        max: 80  # Higher cost threshold in northeast
    bin:
      - feature: lcoe
        weights: capacity_mw
        q: 3

  - region: midwest
    technology: landbasedwind
    filter:
      - feature: lcoe
        max: 60  # Better wind resources in midwest
    bin:
      - feature: lcoe
        weights: capacity_mw
        q: 4  # More bins for better resolution
```

See [Configure Renewable Clusters](../how-to/configure-renewable-clusters.md).

### Isolated Regions

To model a region without transmission connections, simply exclude it from your transmission data tables:

```yaml
model_regions:
  - hawaii  # Island grid
  - western_us
```

**`extra_inputs/ipm_tx_corrections.csv`**:

```csv
region_from,region_to,firm_ttc_mw,notes
# No rows with hawaii - it's isolated
```

Hawaii will be modeled as an independent grid with no inter-regional flows.

## Next Steps

- **Add renewable resources**: [Working with Renewable Resources](renewable-resources.md)
- **Configure policies**: [Resource Tags](../reference/settings/resource-tags.md) for RPS/CES
- **Multi-period models**: [Scenario Management](../reference/settings/scenario-management.md)
- **Transmission details**: [Transmission Settings Reference](../reference/settings/transmission.md)

## Further Reading

- ReEDS model and regions: [NREL ReEDS Documentation](https://www.nrel.gov/analysis/reeds/)
- PowerGenome region mappings: [Geospatial Mappings Wiki](https://github.com/PowerGenome/PowerGenome/wiki/Geospatial-Mappings)
- EIA cost regions: [AEO Assumptions](https://www.eia.gov/outlooks/aeo/assumptions/)
- GenX transmission modeling: [GenX Documentation](https://genxproject.github.io/GenX/)
