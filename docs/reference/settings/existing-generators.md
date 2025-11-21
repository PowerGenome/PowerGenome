# Existing Generators Settings

These parameters control how existing power plants are processed, clustered, and configured for the model.

## Clustering Parameters

### `num_clusters`

**Type**: Integer or nested dictionary
**Required**: Yes
**Example**: See below

Default number of clusters to create for each technology in each region. Can be specified globally or per region/technology.

**Global** (same for all regions/technologies):

```yaml
num_clusters:
  Conventional Steam Coal: 3
  Nuclear: 1
  Natural Gas Fired Combined Cycle: ~ # Does not cluster plants
```

**Regional** (different per region):

```yaml
alt_num_clusters:
  CA_N:
    Conventional Steam Coal: 5
    Natural Gas Fired Combined Cycle: 10
    Onshore Wind Turbine: 3
  CA_S:
    Natural Gas Fired Combined Cycle: 8
```

**How it works**:

- K-means clustering, by default on heat rate and fixed O&M
- Each cluster represents a group of similar generators
- Reduces thousands of plants to manageable number of resources
- "None" (`~`) skips clustering and returns individual generators

### `alt_num_clusters`

**Type**: Dictionary (region → technology → number)
**Required**: No
**Example**: See below

Overrides `num_clusters` for specific region/technology combinations.

```yaml
num_clusters:
  Nuclear: 2  # Default

alt_num_clusters:
  CA_N:
    Nuclear: 1  # Each nuclear plant is unique
    Biomass: 2  # Only 2 biomass clusters needed
```

This allows fine-grained control without repeating all technologies in the main `num_clusters` parameter.

### `generator_cluster_columns`

**Type**: List of strings
**Required**: No
**Default**: `["heat_rate_mmbtu_mwh", "fixed_o_m_mw"]`
**Example**: See below

Columns (features) used for k-means clustering. By default, generators are clustered based on heat rate and fixed O&M costs.

**Default behavior** (uses heat rate and fixed O&M implicitly):

```yaml
generator_cluster_columns:
  - heat_rate_mmbtu_mwh
  - fixed_o_m_mw
```

**Custom features**:

```yaml
generator_cluster_columns:
  - heat_rate_mmbtu_mwh
  - capacity_mw
  - operating_year
```

**Use cases**:

- Add `capacity_mw` to keep large/small plants separate
- Add `operating_year` to segregate old vs. new plants
- Add `minimum_load_mw` for minimum load constraints
- Use only `heat_rate_mmbtu_mwh` for thermal efficiency clustering

Available columns depend on your generation data schema. Common options:

- `heat_rate_mmbtu_mwh`: Thermal efficiency
- `capacity_mw`: Plant size
- `fixed_o_m_mw`: Fixed O&M costs
- `variable_o_m_mwh`: Variable O&M costs
- `operating_year`: Plant age proxy
- `minimum_load_mw`: Minimum stable operation level

!!! tip "Clustering Quality"
    Clustering features should be normalized or on similar scales for best results. PowerGenome uses the scikit-learn `StandardScaler()` method to normalize features.

## Technology Grouping

### `tech_groups`

**Type**: Dictionary (group name → list of technologies)
**Required**: No
**Example**: See below

Combines multiple similar technologies into a single group before clustering. Useful for technologies that are functionally equivalent.

```yaml
tech_groups:
  Peaker:
    - Natural Gas Fired Combustion Turbine
    - Natural Gas Steam Turbine
  Biomass:
    - Biomass
    - Municipal Solid Waste
    - Landfill Gas
```

**Benefits**:

- Reduces cluster count for minor technology variations
- Groups functionally similar resources
- Simplifies output analysis

**Matching**: Technology names are matched case-insensitively with partial matching.

### `regional_no_grouping`

**Type**: Dictionary (region → list of technologies)
**Required**: No
**Example**: See below

Technologies that should NOT be grouped or clustered in specific regions. Each plant becomes its own resource.

```yaml
regional_no_grouping:
  CA_N:
    - Nuclear
    - Hydroelectric Pumped Storage
  ALL_REGIONS:
    - Geothermal
```

Use this for:

- Unique plants (nuclear reactors)
- Plants with specific operating constraints
- Facilities where aggregation loses critical detail

<!-- Retirement logic via explicit age thresholds removed in v0.8.0-beta. Generator retention handled through input data (retirement_year filtering) or scenario-specific cost/availability modifiers. -->

## Data Table Configuration

### `generation_table`

**Type**: String or dictionary
**Required**: Yes
**Example**: See below

Source of existing generator data. Can be a simple filename or advanced configuration with filters.

**Simple**:

```yaml
generation_table: generators_2030.csv
```

**Advanced**:

```yaml
generation_table:
  table_name: generators.parquet
  filters:
    - - [operating_year, '<=', 2030]
      - [retirement_year, '>', 2030]
  columns: [plant_id, technology, capacity_mw, heat_rate, region]
```

See [Data Tables](data-tables.md) for full filter syntax.

### `plant_region_table`

**Type**: String or dictionary
**Required**: No
**Example**: `"plant_regions.csv"`

Mapping of plants to base regions.

```yaml
plant_region_table: plant_region_map.csv
```

Expected columns:

- `plant_id`: Plant identifier
- `region`: Region name (matching `model_regions`)

## Example Configuration

Complete existing generator settings for a regional model:

```yaml
# Clustering
num_clusters:
  Conventional Steam Coal: 3
  Nuclear: 2
  Hydroelectric Pumped Storage: 2
  Peaker: 1

alt_num_clusters:
  CA_N:
    Nuclear: 1
    Hydroelectric Pumped Storage: 1
  CA_S:
    Nuclear: 1

# Technology grouping
tech_groups:
  Peaker:
    - Natural Gas Fired Combustion Turbine
    - Natural Gas Steam Turbine

# Data source
generation_table:
  table_name: generators.parquet
  filters:
    - - [operating_year, '<=', 2030]
```

## Related Settings

- [Model Definition](model-definition.md): Planning years affect retirement calculations
- [Regions](regions.md): `model_regions` used in clustering
- [Resource Tags](resource-tags.md): Tag assignment for clustered generators
- [Data Tables](data-tables.md): Table configuration details
