# Regional Configuration

Regional settings define the geographic structure of your model—how base regions aggregate into model regions, and region-specific customizations.

!!! info "Understanding PowerGenome Regions"
    PowerGenome uses a hierarchical regional structure based on NREL's Regional Energy Deployment System (ReEDS) regions:

    - **Base regions**: The fundamental geographic units from the ReEDS model (labeled `p1`, `p2`, `p3`, etc.)
    - **Aggregated regions**: Groups of multiple base regions combined together (e.g., `p1_2` combines `p1` and `p2`)
    - **Model regions**: The regions in your study - can be either single base regions OR aggregated regions

    The `region_aggregations` parameter defines how base regions map to model regions. While PowerGenome examples use ReEDS regions, you can adapt this structure to any geography (countries, provinces, utility territories, etc.) by providing appropriately structured data tables.

## Core Regional Parameters

### `model_regions`

**Type**: List of strings
**Required**: Yes
**Example**: `["CA_N", "CA_S", "AZ", "NM"]`

Names of the regions that will appear in output files. These can be:

- Single base regions (e.g., `p10`, `p22`)
- Aggregated region names (defined in `region_aggregations`)

```yaml
model_regions: [p1_2, p3, northwest]
```

Here `p1_2` and `northwest` are aggregated regions and `p3` is a base region.

All regional parameters use these names as keys.

### `region_aggregations`

**Type**: Dictionary (model region → list of base regions)
**Required**: Only if using aggregations
**Example**: See below

Maps model region names to the base regions they contain. Use this to combine multiple base regions into larger study areas.

```yaml
region_aggregations:
  p1_2:      # Aggregated region
    - p1
    - p2
  p3: [p3]   # Pass-through (optional, can omit)
  northwest:  # Custom name for aggregated region
    - p4
    - p5
    - p6
```

**Important**:

- Model region names in `region_aggregations` must appear in `model_regions`
- Base region names must match those in your generation/demand data tables
- If `model_regions` contains a name NOT in `region_aggregations`, it's treated as a pass-through (1:1 mapping to a base region)

## Regional Capacity Reserves

### `regional_capacity_reserves`

**Type**: Nested dictionary
**Required**: No
**Example**: See below

Defines capacity reserve requirements by zone. Top-level keys are `CapRes_<num>` tags, second level keys are model regions.

```yaml
regional_capacity_reserves:
  CapRes_1:
    CA_N: 0.15  # 15% reserve margin
    CA_S: 0.15
  CapRes_2:
    AZ: 0.12
```

Configure resource contributions with `capacity_reserve_values`,
`model_tag_values`, or `regional_tag_values`; see
[Resource Tags](resource-tags.md#capacity-reserve-tags).

### `cap_res_network_derate_default`

**Type**: Float (0-1)
**Required**: No
**Default**: 0.95
**Example**: `0.95`

Derating factor for transmission imports used to meet capacity reserves. A value of 0.95 means transmission can only contribute 95% of its capacity toward reserve requirements.

```yaml
cap_res_network_derate_default: 0.95
```

## Technology-Specific Regional Settings

### `new_gen_not_available`

**Type**: Dictionary (region name or `all`/`all_regions` → list of technology name patterns)
**Required**: No
**Default**: None

Excludes new-build technologies from individual model regions. `new_resources` is
not region-scoped — every entry is built in every model region — so this is the
supported way to keep a technology out of a subset of regions.

Patterns are case-insensitive `fnmatch` globs matched against the full resource
name `"<technology>_<tech_detail>_<cost_case>"`, the `technology` column of the
resource files. A pattern with no wildcard is an exact-name match, so
`OffshoreWind_Class1` matches nothing; write `OffshoreWind_*`. The keys `all` and
`all_regions` (any casing) apply to every model region and are additive with
per-region keys.

```yaml
new_gen_not_available:
  AZ:
    - OffshoreWind_*      # No offshore wind in AZ
    - Geothermal_*        # No geothermal potential
  CA_N:
    - Coal_new*           # Coal banned in CA
  all:
    - Nuclear_*           # No new nuclear anywhere
```

Exclusions cover renewable technologies as well as conventional ones. See
[New-Build Resources — `new_gen_not_available`](new-build.md#new_gen_not_available)
for the full pattern semantics, the validation behaviour, and the interaction with
`renewables_clusters`.

## Cost Multipliers

### `cost_multiplier_region_map`

**Type**: Dictionary (cost region name → list of model regions)
**Required**: No (only if `regional_cost_factor_table` doesn't contain all base regions)
**Example**: See below

Maps cost region names from the `regional_cost_factor_table` to model regions. Use this when your cost multiplier table uses different region names than your base/model regions.

```yaml
cost_multiplier_region_map:
  CA_N_SPUR: [CA_N]
  CA_S_SPUR: [CA_S]
  Southwest: [AZ, NM]  # One cost region covers multiple model regions
```

**When to use**:

- Your `regional_cost_factor_table` doesn't include all base regions
- Cost table uses aggregate region names (e.g., "Southwest" covers multiple model regions)
- Using generic cost tables (e.g., from NREL) with different naming conventions

**When NOT needed**: If `regional_cost_factor_table` already contains all your base or model regions with matching names.

### `cost_multiplier_technology_map`

**Type**: Dictionary (cost table technology name → list of user technology names)
**Required**: No
**Example**: See below

Maps technology names from the `regional_cost_factor_table` to user-defined technology names. Use this to apply cost multipliers from one technology to custom technologies created via `modified_new_resources`.

```yaml
cost_multiplier_technology_map:
  NaturalGas_2-on-1 Combined Cycle (F-Frame): [hydrogen_turbine, synthetic_fuel_turbine]  # Multiple custom techs use NGCC costs
  LandbasedWind: [Biomass_Dedicated]  # Use wind cost multipliers for biomass
```

**Common use cases**:

- Custom technologies from `modified_new_resources` that should use cost multipliers from a base ATB technology
- Technologies from `additional_technologies_fn` that need regional cost adjustments
- Applying one technology's multipliers to another (e.g., using wind multipliers for solar)

## UTC Offset

### `utc_offset`

**Type**: Integer
**Required**: No
**Default**: 0
**Example**: `-8`

Hour offset applied when shifting input time profiles to the model timezone.

```yaml
utc_offset: -8  # Pacific Time (UTC-8)
```

US time zones:

- Pacific: `-8`
- Mountain: `-7`
- Central: `-6`
- Eastern: `-5`

## Legacy Regional Mappings (Optional)

### `fuel_region_map`

**Type**: Dictionary (AEO region → list of model regions)
**Required**: No (legacy workflow only)
**Example**: See below

!!! warning "Legacy Parameter"
    This parameter is only needed for the legacy AEO fuel workflow. In the simplified workflow (recommended), provide fuel prices for all base regions directly in your fuel price table instead.

Maps EIA Annual Energy Outlook (AEO) fuel price regions to model regions when using the legacy AEO data workflow.

```yaml
fuel_region_map:
  pacific: [CA_N, CA_S]
  mountain: [AZ, NM]
  west_south_central: [TX]
```

Common AEO region names:

- `pacific`: Pacific
- `mountain`: Mountain
- `west_south_central`: West South Central
- `east_north_central`: East North Central
- `new_england`: New England

See [Fuel Settings](fuels.md) for details on the simplified vs. legacy fuel workflows.

## Example Configuration

Complete regional settings for a California + Southwest model:

```yaml
# Base model regions
model_regions: [CA_N, CA_S, AZ, NM]

# Aggregate IPM regions
region_aggregations:
  CA_N: [CA_IID, CA_LADWP, CA_BANC]
  CA_S: [CA_SCE, CA_SDGE]

# Regional capacity reserves
regional_capacity_reserves:
  CapRes_1:
    CA_N: 0.15
    CA_S: 0.15
    AZ: 0.12
    NM: 0.12

cap_res_network_derate_default: 0.95

# Regional technology exclusions (patterns match <technology>_<tech_detail>_<cost_case>)
new_gen_not_available:
  CA_N: [Coal_new*, OffshoreWind_*]
  CA_S: [Coal_new*]
  AZ: [OffshoreWind_*, Geothermal_*]
  NM: [OffshoreWind_*, Geothermal_*]

# Timezone
utc_offset: -8
```

## Related Settings

- [Model Definition](model-definition.md): Planning years and model scope
- [Existing Generators](existing-generators.md): Region-specific clustering
- [New-Build Resources](new-build.md): Regional availability constraints
- [Transmission](transmission.md): Inter-regional network topology
