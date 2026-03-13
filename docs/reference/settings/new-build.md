# New-Build Resources Settings

These parameters control technology costs, new resource definitions, regional availability, and capacity constraints for candidate generators.

## Technology Cost Sources

### `new_resources`

**Type**: List of lists
**Required**: For ATB-based technologies
**Example**: See below

List of technologies from NREL Annual Technology Baseline (ATB) to include as new-build candidates. Each item is a list: `[technology, tech_detail, cost_case, size_mw]`.

```yaml
new_resources:
  - [NaturalGas, CCCCSAvgCF, Moderate, 500]
  - [NaturalGas, CTAvgCF, Moderate, 100]
  - [Battery, "*", Moderate, 100]
  - [UtilityPV, Class1, Moderate, 100]
  - [LandbasedWind, Class3, Moderate, 100]
  - [OffshoreWind, Class3, Moderate, 100]
  - [Nuclear, Nuclear - Large, Moderate, 1000]
```

**Format**: `[technology, tech_detail, cost_case, size_mw]`

- `technology`: Technology type (e.g., `NaturalGas`, `Battery`, `UtilityPV`)
- `tech_detail`: Specific variant (e.g., `CCAvgCF`, `Class1`, `*` for wildcard)
- `cost_case`: Cost trajectory (`Conservative`, `Moderate`, `Advanced`)
- `size_mw`: Unit size in MW

### `resource_data_year`year`

**Type**: Integer
**Required**: No
**Default**: Latest available
**Example**: `2023`

Which year's ATB data to use for technology costs.

```yaml
resource_data_year: 2023
```

ATB is updated annually with revised cost projections. Specify the data vintage for reproducibility.

### `resource_financial_case`case`

**Type**: String
**Required**: No
**Default**: `"Market"`
**Example**: `"Market"`

ATB financial assumptions case. Options:

- `Market`: Market-based financing
- `R&D`: Optimistic R&D case with policy support

```yaml
resource_financial_case: Market
```

### `resource_cap_recovery_years`

**Type**: Integer
**Required**: No
**Default**: `20`
**Example**: `20`

Default capital recovery period (economic lifetime) for new generators, used to calculate annualized capital costs.

```yaml
resource_cap_recovery_years: 20
```

### `alt_resource_cap_recovery_years`

**Type**: Dictionary (technology → years)
**Required**: No
**Example**: See below

Override capital recovery period for specific technologies (matched using string contains).

```yaml
alt_resource_cap_recovery_years:
  NaturalGas_CCAvg: 15
  NaturalGas_CTAvg: 15
  Battery: 15
  Nuclear: 40
```

## Cost and Performance Adjustments

### `resource_modifiers`

**Type**: Dictionary (tech_name → modifications)
**Required**: No
**Example**: See below

Modify ATB technology parameters. Each entry specifies technology/tech_detail matching and parameter changes.

```yaml
resource_modifiers:
  batteries:
    technology: Battery
    tech_detail: "*"
    Var_OM_Cost_per_MWh: [add, 0.15]
    Var_OM_Cost_per_MWh_In: 0.15
    size_mw: 100
```

Each entry creates a candidate resource with the specified unit size.

Parameter values inside each modifier can also be specified as **year-keyed dictionaries** (see [Settings by Model Year](#settings-by-model-year)) to vary costs across planning periods without a full scenario management setup:

```yaml
resource_modifiers:
  batteries:
    technology: Battery
    tech_detail: "*"
    Var_OM_Cost_per_MWh:
      2030: [add, 0.15]
      2040: [add, 0.10]
    size_mw: 100
```

### `modified_new_resources`

**Type**: Dictionary (new_name → base + modifications)
**Required**: No
**Example**: See below

Create new technology variants by copying and modifying existing ATB technologies.

```yaml
modified_new_resources:
  NGCCS100:
    technology: NaturalGas
    tech_detail: CCCCSAvgCF
    cost_case: Conservative
    size_mw: 500
    new_technology: NaturalGas
    new_tech_detail: CCS100
    new_cost_case: Moderate
    capex_mw: [add, 116000]
    heat_rate: [add, 0.365]
    fixed_o_m_mw: [add, 9670]
    variable_o_m_mwh: [mul, 1.076]

  hydrogen_turbine:
    technology: NaturalGas
    tech_detail: 1-on-1 Combined Cycle (H-Frame)
    cost_case: Moderate
    size_mw: 100
    new_technology: hydrogen
    new_tech_detail: 1-on-1 Combined Cycle (H-Frame)
    new_cost_case: Moderate
```

**Base parameters** (copy from existing ATB technology):

- `technology`: ATB technology to copy from
- `tech_detail`: ATB tech detail to copy from
- `cost_case`: ATB cost case to copy from
- `size_mw`: Unit size

**New parameters** (create new technology with this name):

- `new_technology`: New technology name
- `new_tech_detail`: New tech detail name
- `new_cost_case`: New cost case name

**Modification operators**:

- `[add, value]`: Add to base value
- `[mul, value]`: Multiply base value
- `[sub, value]`: Subtract from base value
- `[truediv, value]`: Divide base value
- Direct value: Override base value

## Settings by Model Year

Year-keyed values are supported across all settings sections, not just new-build parameters.

Use [Year-Keyed Values](year-keyed-values.md) for the complete rules, validation behavior, and additional examples.

Quick summary:

- Exact year values take priority.
- `default` is a fallback key.
- Incomplete year coverage without fallback raises a `ValueError`.

Example in `resource_modifiers`:

```yaml
resource_modifiers:
  utility_pv:
    technology: UtilityPV
    tech_detail: Class1
    capex_mw:
      2030: [mul, 0.9]
      2040: [mul, 0.8]
```

Example with fallback:

```yaml
carbon_tax:
  2030: 25
  default: 50
```

!!! note
    The `settings_management` key itself is never modified by year-keyed resolution; it is always processed by scenario management logic first.

## Regional Availability

### `new_gen_not_available`

**Type**: Dictionary (region → list of technologies)
**Required**: No
**Example**: See below

Exclude specific new-build technologies from certain regions.

```yaml
new_gen_not_available:
  AZ:
    - OffshoreWind_Class1_Moderate_-1
    - Geothermal_HydroFlash_Moderate
  CA_N:
    - Coal_new_Moderate
```

Use cases:

- Physical constraints (no offshore wind in landlocked states)
- Policy constraints (coal banned)
- Resource availability (no geothermal potential)

## Interconnection Costs

### `interconnect_capex_mw`

**Type**: Number or dictionary
**Required**: No
**Default**: None (falls back to legacy `transmission_investment_cost` if provided)
**Example**: See patterns below

Flexible specification of per-MW interconnection capital cost (USD/MW) applied to new and existing resources. Replaces the deprecated spur line mileage system (`transmission_investment_cost` with `capacity_limit_spur_fn` file).

**Important**: All cost values must be in the same dollar year as `target_usd_year` (the target dollar year for cost normalization). PowerGenome does not automatically adjust interconnection costs for inflation.

**Supported Patterns**:

#### 1. Scalar

Apply uniform cost to all resources:

```yaml
interconnect_capex_mw: 150000  # $150k/MW for all resources
```

#### 2. Explicit Schema (Recommended)

Use an explicit fallback with typed override blocks:

```yaml
interconnect_capex_mw:
  fallback_capex_mw: 110000
  by_technology:
    wind: 120000
    offshore_wind: 200000
    battery: 50000
  by_region:
    CA_N: 125000
  by_region_technology:
    CA_N:
      offshore_wind: 210000
```

Precedence (most specific to least specific):

1. `by_region_technology`
2. `by_region`
3. `by_technology`
4. `fallback_capex_mw`

#### 3. Region-Only

Different costs by region (exact region name matching):

```yaml
interconnect_capex_mw:
  fallback_capex_mw: 120000
  by_region:
    CA_N: 140000
    CA_S: 130000
    AZ: 125000
```

#### 4. Technology-Only

Different costs by technology using case-insensitive substring matching with shortest-first precedence:

```yaml
interconnect_capex_mw:
  fallback_capex_mw: 100000
  by_technology:
    wind: 120000          # matches 'LandbasedWind_Class3_Moderate', 'OffshoreWind_Class1_Moderate'
    offshore_wind: 200000 # longer substring overwrites previous 'wind' assignment
    battery: 50000        # matches 'Battery_4Hr_Moderate', 'Battery_*_Moderate'
    solar: 110000         # matches 'UtilityPV_Class1_Moderate'
```

**Precedence rule**: Shorter substrings applied first, then longer (more specific) substrings override previous assignments. This allows general categories with specific exceptions.

#### 5. Region -> Technology Nested

Region-specific technology costs with regions as keys under `by_region_technology`:

```yaml
interconnect_capex_mw:
  fallback_capex_mw: 110000
  by_region_technology:
    CA_N:
      battery: 60000
      offshore_wind: 210000
      solar: 105000
    CA_S:
      solar: 95000
      wind: 115000
  by_region:
    AZ: 125000
```

Within each region, technology substrings follow shortest-first precedence.

**Matching Rules**:

- **Technology matching**: Case-insensitive substring search against the `technology` column
- **Substring precedence**: Shortest substrings processed first; longer substrings override
- **Region matching**: Exact match against model region names (case-sensitive)
- **Explicit schema**: Use only `fallback_capex_mw`, `by_region`, `by_technology`, and `by_region_technology` at the top level.
- **Invalid mixing**: In legacy syntax, mixing region and technology keys at the same top level is invalid.

**Examples of Invalid Configuration**:

```yaml
# ERROR: Cannot mix region and technology keys
interconnect_capex_mw:
  CA_N: 140000      # Region key
  wind: 120000      # Technology key - mixing not allowed!
```

**Legacy Syntax (Deprecated)**:

You may still use the older top-level `default` style for backward compatibility. A deprecation warning is logged at runtime.

**Behavior**:

- **Non-destructive**: Existing non-zero `interconnect_capex_mw` values in the resource dataframe are never overwritten
- **Automatic annuity**: `interconnect_annuity` is computed automatically for:
  - Newly assigned capex rows
  - Pre-existing capex rows with zero/blank annuity
- **Financial parameters**: Annuity calculation uses plant-specific `wacc_real` and `cap_recovery_years` from the resource dataframe
- **Legacy bypass**: When `interconnect_capex_mw` is provided, legacy spur mileage columns (`spur_miles`, `offshore_spur_miles`, `tx_miles`) are ignored (warning logged if present)

**Migration from Legacy System**:

Old approach (deprecated):

```yaml
# Settings
transmission_investment_cost:
  spur:
    capex_mw_mile:
      CA_N: 3000
      CA_S: 3200
    wacc: 0.069
    investment_years: 60

# Required extra_inputs file: capacity_limit_spur_fn
# File: resource_capacity_spur.csv
# resource,region,spur_miles
# offshore_wind_fixed,CA_N,45
# solar_pv,CA_N,12
```

New approach (recommended):

```yaml
interconnect_capex_mw:
  fallback_capex_mw: 120000
  by_technology:
    offshore_wind: 200000  # Direct cost specification
    solar: 105000
```

**Benefits of new system**:

- No external file dependency
- More intuitive (direct $/MW vs. mileage calculation)
- Flexible pattern matching (region, technology, or nested)
- Automatic annuity handling
- Better error messages

### `transmission_investment_cost`

**Type**: Dictionary
**Required**: No
**Deprecated**: Use `interconnect_capex_mw` instead
**Example**: See legacy documentation

Legacy spur line mileage-based interconnection cost system. Contains nested dictionaries for `spur`, `offshore_spur`, and `tx` with keys:

- `capex_mw_mile`: Cost per MW-mile (by region)
- `wacc`: Weighted average cost of capital
- `investment_years`: Capital recovery period

**Deprecation notice**: This parameter is deprecated and will be removed in a future release. It is only used when `interconnect_capex_mw` is not provided, and emits warnings during execution. Migrate to `interconnect_capex_mw` for simplified, more flexible cost specification.

## Renewable Resource Clustering

### `renewables_clusters`

**Type**: List of dictionaries
**Required**: For renewable resources (wind, solar, geothermal) with spatially-distributed potential
**Example**: See below

Configure clustering of renewable resource sites from external datasets (e.g., NREL reV). PowerGenome filters, bins, and clusters pre-existing resource data to create representative sites for the model.

```yaml
renewables_clusters:
  - region: all
    technology: landbasedwind
    filter:
      - feature: lcoe_interconnect_adj
        max: 60
    bin:
      - feature: lcoe_interconnect_adj
        weights: capacity_mw
        mw_per_bin: 10000
    cluster:
      - feature: cf
        n_clusters: 2
        method: agg

  - region: p4
    technology: utilitypv
    filter:
      - feature: lcoe_interconnect_adj
        max: 35
    bin:
      - feature: lcoe_interconnect_adj
        weights: capacity_mw
        mw_per_bin: 50000
    cluster:
      - feature: lcoe_interconnect_adj
        n_clusters: 2
        method: agg

  - region: all
    technology: geothermal
    type: egs
    filter:
      - feature: class
        max: 6
    group:
      - class
    group_modifiers:
      - group: class
        group_value: 6
        Inv_Cost_per_MWyr: [add, 64820]

  - region: all
    technology: offshorewind
    turbine_type: floating
    pref_site: 0
    bin:
      - feature: lcoe_interconnect_adj
        weights: capacity_mw
        q: 2
```

**Common parameters**:

- `region`: Model region name or `all` (applies to all regions)
- `technology`: Technology type (lowercase: `landbasedwind`, `utilitypv`, `offshorewind`, `geothermal`)

**Filtering** (`filter`):

- `feature`: Attribute to filter on (`lcoe`, `cf`, `lcoe_interconnect_adj`, `class`)
- `max`: Maximum value threshold (exclude resources above this)
- `min`: Minimum value threshold (exclude resources below this)

**Binning** (`bin`):

Groups resources into bins before clustering:

- `feature`: Attribute to bin by (`lcoe`, `lcoe_interconnect_adj`, `cf`)
- `weights`: Weight bins by capacity (`mw`, `capacity_mw`)
- `q`: Number of quantile bins (equal-frequency binning)
- `mw_per_bin`: Fixed MW capacity per bin (alternative to `q`)
- `bins`: Custom bin edges (list of values)

**Clustering** (`cluster`):

Aggregates resources within each bin:

- `feature`: Attribute to cluster by (`cf`, `lcoe`, `lcoe_interconnect_adj`, `profile`)
- `n_clusters`: Number of clusters to create
- `method`: Clustering algorithm (`agg` for agglomerative hierarchical clustering)

**Grouping** (`group` + `group_modifiers`):

Alternative to binning/clustering for technologies with discrete classes:

- `group`: List of features to group by (e.g., `[class]`)
- `group_modifiers`: List of cost adjustments per group value

**Technology-specific parameters**:

- **Offshore wind**: `turbine_type` (`fixed`, `floating`), `pref_site` (0 or 1 for preferred sites)
- **Geothermal**: `type` (`egs`, `geohydrobinary`)

**Workflow**:

1. Load resource sites from external data (NREL reV, ReEDS)
2. Filter sites by quality thresholds
3. Bin remaining sites by cost/performance
4. Cluster sites within each bin to reduce model size
5. Create aggregated resource with weighted-average characteristics

See [Configure Renewable Clusters How-To](../../how-to/configure-renewable-clusters.md) for detailed examples.

## Cost Multipliers

Regional capital cost multipliers adjust technology costs by region to account for differences in labor costs, permitting, and construction conditions. PowerGenome uses the `regional_cost_factor_table` to apply these adjustments.

See [Regional Configuration - Cost Multipliers](regions.md#cost-multipliers) for details on `cost_multiplier_region_map` and `cost_multiplier_technology_map` parameters.

## Example Configuration

Complete new-build configuration:

```yaml
# New-build technologies
resource_data_year: 2023
resource_financial_case: Market
resource_cap_recovery_years: 20

new_resources:
  - [NaturalGas, CCCCSAvgCF, Conservative, 500]
  - [NaturalGas, CTAvgCF, Moderate, 100]
  - [Battery, "*", Moderate, 100]
  - [UtilityPV, Class1, Moderate, 100]
  - [LandbasedWind, Class3, Moderate, 100]
  - [Nuclear, Nuclear - Large, Moderate, 1000]

# Technology modifications
resource_modifiers:
  batteries:
    technology: Battery
    tech_detail: "*"
    Var_OM_Cost_per_MWh: [add, 0.15]

# Regional restrictions
new_gen_not_available:
  AZ:
    - OffshoreWind_Class1_Moderate
  CA_N:
    - Coal_newAvgCF_Moderate

# Capacity limits
new_build_max_capacity:
  CA_N:
    Nuclear_Nuclear_Moderate: 2000
    UtilityPV_Class1_Moderate: 10000

# Renewable sites
renewable_clusters:
  - region: CA_N
    technology: LandbasedWind
    cluster: 1
    max_capacity: 2000
    profile_id: CA_N_wind_class3

RESOURCE_GROUP_PROFILES: /data/nrel_profiles
```

## Related Settings

- [Model Definition](model-definition.md): `model_year` affects cost projections
- [Regions](regions.md): Regional availability and cost multipliers
- [Fuels](fuels.md): Fuel costs for thermal technologies
- [Resource Tags](resource-tags.md): Tag assignment for new resources
