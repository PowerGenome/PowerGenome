# Add Custom Technologies

This guide shows how to add new generation technologies, modify existing ones, and configure renewable resource clusters.

## Technology Sources

PowerGenome gets technology data from several sources:

1. **NREL ATB (Annual Technology Baseline)**: Standard new-build technologies
2. **Existing generators**: From data tables (existing fleet)
3. **Additional technologies**: User-defined technologies not in ATB
4. **Modified resources**: Variants of ATB technologies with adjusted parameters

## Add ATB Technologies

### List Available Technologies

ATB technologies follow naming convention: `{Technology}_{TechDetail}_{FinancialCase}`

**Common technologies**:

- `NaturalGas_2-on-1 Combined Cycle (F-Frame)_Moderate`: Combined cycle gas turbine
- `NaturalGas_Combustion Turbine (F-Frame)_Moderate`: Combustion turbine (peaking)
- `UtilityPV_Class1_Moderate`: Utility-scale solar PV
- `LandbasedWind_Class3_Moderate`: Onshore wind
- `OffShoreWind_Class12_Moderate`: Floating offshore wind
- `Utility-Scale Battery Storage_Lithium Ion_Moderate`: Battery storage
- `Nuclear_Nuclear - Large_Moderate`: Large nuclear power
- `Geothermal_HydroBinary_Moderate`: Geothermal hydrothermal

**Cost cases**:

- `Conservative`: High cost
- `Moderate`: Mid cost
- `Advanced`: Low cost (R&D success)

### Include in Model

Add technologies to `new_resources`:

```yaml
new_resources:
  - [NaturalGas, 2-on-1 Combined Cycle (F-Frame), Moderate, 500]
  - [NaturalGas, Combustion Turbine (F-Frame), Moderate, 100]
  - [UtilityPV, Class1, Moderate, 100]
  - [LandbasedWind, Class3, Moderate, 1]
  - [Utility-Scale Battery Storage, Lithium Ion, Moderate, 1]
```

Specify unit sizes in `new_resources` (fourth element of each list):

```yaml
new_resources:
  - [NaturalGas, 2-on-1 Combined Cycle (F-Frame), Moderate, 500]  # 500 MW per unit
  - [NaturalGas, Combustion Turbine (F-Frame), Moderate, 100]  # 100 MW per unit
  - [UtilityPV, Class1, Moderate, 1]
  - [LandbasedWind, Class3, Moderate, 1]
  - [Utility-Scale Battery Storage, Lithium Ion, Moderate, 1]
  - [Nuclear, Nuclear - Large, Moderate, 1000]
```

### Set Cost Case

Choose financial assumptions and cost trajectories:

```yaml
resource_data_year: 2023  # ATB vintage year
resource_financial_case: Market  # Market or R&D
resource_cap_recovery_years: 20  # Economic lifetime
```

**Cost cases** are specified per-technology in `new_resources`:

- `Conservative`: High cost trajectory
- `Moderate`: Mid cost trajectory
- `Advanced`: Low cost trajectory (R&D success)

Example:

```yaml
new_resources:
  - [UtilityPV, Class1, Conservative, 100]  # High cost
  - [UtilityPV, Class1, Moderate, 100]      # Mid cost
  - [UtilityPV, Class1, Advanced, 100]      # Low cost
```

### Assign Resource Tags

Define dispatch behavior:

```yaml
model_tag_names:
  - THERM
  - VRE
  - STOR
  - MUST_RUN

model_tag_values:
  THERM:
    NaturalGas_: 1
  VRE:
    UtilityPV: 1
    LandbasedWind: 1
  STOR:
    Utility-Scale Battery Storage: 1
  MUST_RUN:
    Geothermal: 1
```

## Modify ATB Technology Costs

### Resource Modifiers

Adjust parameters of existing ATB technologies:

```yaml
resource_modifiers:
  utility_pv:
    technology: UtilityPV
    tech_detail: Class1
    capex_mw: [mul, 1.2]  # Multiply by 1.2 (20% higher)
    fixed_o_m_mw: 18000  # Direct override to $18,000/MW-year

  landbased_wind:
    technology: LandbasedWind
    tech_detail: Class3
    capex_mw: [mul, 0.9]  # 10% cost reduction
    variable_o_m_mwh: 0  # Zero variable O&M

  batteries:
    technology: Utility-Scale Battery Storage
    tech_detail: Lithium Ion
    capex_mw: [add, 55000]  # Add $100/kW (in 2004 USD) for interconnection
    Var_OM_Cost_per_MWh: [add, 0.15]
    Var_OM_Cost_per_MWh_In: 0.15  # Set directly (not in ATB)
    wacc_real: 0.04675  # NREL ATB Market WACC for UtilityPV
```

**Modifier behavior**:

- **Operators in lists**: `[operator, value]` format
  - `[mul, 1.2]`: Multiply by 1.2 (20% higher)
  - `[add, 50000]`: Add $50,000
  - `[sub, 10000]`: Subtract $10,000
  - `[truediv, 2]`: Divide by 2
- **Direct values**: Set parameter directly (no operator)
  - Use for parameters not in ATB database
  - Use for absolute overrides

### Regional Cost Multipliers

Apply regional construction cost differences:

**regional_cost_multipliers.csv**:

```csv
technology,region,value
NaturalGas_2-on-1 Combined Cycle (F-Frame),p1,1.140882
NaturalGas_Combustion Turbine (F-Frame),p1,1.132064
LandbasedWind_Class3,p2,1.097588
Utility-Scale Battery Storage_Lithium Ion,p1,1.072245
```

Multipliers apply on top of `resource_modifiers`.

## Create Technology Variants

### Modified New Resources

Create renamed copies of ATB technologies with modified parameters:

```yaml
modified_new_resources:
  # Hydrogen turbine based on natural gas combined cycle
  hydrogen_turbine:
    technology: NaturalGas
    tech_detail: 1-on-1 Combined Cycle (H-Frame)
    cost_case: Moderate
    size_mw: 100
    new_technology: hydrogen
    new_tech_detail: turbine
    new_cost_case: Moderate
    heat_rate: [mul, 1.1]  # 10% worse heat rate
    capex_mw: [add, 50000]  # Add $50k/MW for hydrogen infrastructure

  # High-cost solar variant
  utility_pv_high:
    technology: UtilityPV
    tech_detail: Class1
    cost_case: Moderate
    size_mw: 100
    new_technology: UtilityPV
    new_tech_detail: Class1_High
    new_cost_case: High
    capex_mw: [mul, 1.5]  # 50% more expensive
```

**Required fields**:

- `technology`, `tech_detail`, `cost_case`: Source technology
- `new_technology`, `new_tech_detail`, `new_cost_case`: New technology name
- `size_mw`: Unit size

**Optional modifiers**: Use `[operator, value]` format same as `resource_modifiers`

**Assign tags**:

```yaml
model_tag_values:
  THERM:
    hydrogen: 1  # Matches new_technology name
  VRE:
    UtilityPV: 1
  Commit:
    hydrogen: 1
  New_Build:
    hydrogen: 1
```

## Configure Renewable Clusters

### Renewable Resource Groups

Define renewable resource clustering using the `renewables_clusters` parameter:

```yaml
renewables_clusters:
  - region: all  # Apply to all regions, or specify individual regions
    technology: landbasedwind
    filter:
      - feature: lcoe
        max: 75  # Maximum LCOE threshold
    bin:
      - feature: lcoe
        weights: capacity_mw  # Weight binning by capacity
        q: 2  # Create 2 bins based on quartiles

  - region: p1
    technology: utilitypv
    filter:
      - feature: lcoe_interconnect_adj
        max: 30
    bin:
      - feature: lcoe_interconnect_adj
        weights: capacity_mw
        mw_per_bin: 50000  # Create bins of 50 GW each
    cluster:
      - feature: cf  # Cluster within bins by capacity factor
        n_clusters: 2
        method: agg  # Agglomerative clustering

  - region: all
    technology: offshorewind
    turbine_type: floating
    pref_site: 0  # Include all sites (1 for preferred sites only)
    bin:
      - feature: lcoe_interconnect_adj
        weights: capacity_mw
        q: 2
```

**Common parameters**:

- `region`: Model region name or `all`
- `technology`: Technology type (lowercase: `landbasedwind`, `utilitypv`, `offshorewind`)
- `filter`: List of feature filters to exclude resources
  - `feature`: Attribute to filter on (`lcoe`, `cf`, `lcoe_interconnect_adj`)
  - `max`: Maximum value threshold
- `bin`: List of binning operations
  - `feature`: Attribute to bin by
  - `weights`: Weight bins by capacity (`mw`, `capacity_mw`)
  - `q`: Number of quantile bins
  - `mw_per_bin`: Fixed MW per bin
- `cluster`: List of clustering operations within bins
  - `feature`: Attribute to cluster by (`cf`, `profile`, `lcoe`)
  - `n_clusters`: Number of clusters
  - `method`: Clustering method (`agg` for agglomerative)

**Technology-specific parameters**:

- Offshore wind: `turbine_type` (`fixed`, `floating`), `pref_site` (0 or 1)
- Geothermal: `type` (`egs`, `geohydrobinary`)

### Advanced: Group Modifiers

Modify costs for specific resource groups:

```yaml
renewables_clusters:
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
        Inv_Cost_per_MWyr: [add, 64820]  # Additional cost for class 6
```

**Note**: Renewable resource profiles and capacity data come from external datasets (NREL reV, ReEDS). PowerGenome does not generate these profiles—it clusters pre-existing resource sites based on cost and performance characteristics.

## Restrict Technology Availability

### Regional Restrictions

Prohibit technologies in specific regions:

```yaml
new_gen_not_available:
  AZ:
    - OffShoreWind  # No offshore wind in Arizona
    - Nuclear
  CA_N:
    - Coal  # No new coal anywhere in California
  CA_S:
    - Coal
```

### Technology Minimum Load

Set minimum stable generation level:

```yaml
min_cap_req:
  NaturalGas_2-on-1 Combined Cycle (F-Frame)_Moderate: 0.3  # 30% minimum load
  Nuclear_Nuclear - Large_Moderate: 0.9  # 90% minimum load
  Coal_New_Moderate: 0.4
```

### Ramp Rates

Define maximum ramp rates (fraction per hour):

```yaml
ramp_up_rates:
  NaturalGas_2-on-1 Combined Cycle (F-Frame)_Moderate: 0.5  # 50% per hour
  Coal_New_Moderate: 0.2  # 20% per hour
  Nuclear_Nuclear - Large_Moderate: 0.05  # 5% per hour

ramp_down_rates:
  NaturalGas_2-on-1 Combined Cycle (F-Frame)_Moderate: 0.5
  Coal_New_Moderate: 0.2
  Nuclear_Nuclear - Large_Moderate: 0.05
```

## Example: Add Hydrogen Technology

Complete example adding hydrogen turbines:

### 1. Create Additional Technology

**additional_technologies.csv**:

```csv
technology,tech_detail,model_year,capex_mw,fixed_o_m_mw,variable_o_m_mwh,heat_rate_mmbtu_mwh,fuel,wacc_real,capital_recovery_period,unit_size_mw
Hydrogen,Turbine,2030,900000,18000,4,9.5,hydrogen,0.055,25,150
Hydrogen,Turbine,2040,750000,15000,4,9.0,hydrogen,0.055,25,150
Hydrogen,Turbine,2050,600000,12000,4,8.5,hydrogen,0.055,25,150
```

### 2. Configure in Settings

**settings/custom_technologies.yml**:

```yaml
# Include additional tech file
additional_technologies_fn: additional_technologies.csv

# Add to new resources - use list format
new_resources:
  - [Hydrogen, Turbine, Moderate, 150]

# Assign tags
model_tag_values:
  THERM:
    Hydrogen: 1
  Commit:
    Hydrogen: 1
  New_Build:
    Hydrogen: 1

# Map to fuel
tech_fuel_map:
  Hydrogen: hydrogen

# Operational constraints
min_cap_req:
  Hydrogen_Turbine_Moderate: 0.2
ramp_up_rates:
  Hydrogen_Turbine_Moderate: 0.8
ramp_down_rates:
  Hydrogen_Turbine_Moderate: 0.8
```

### 3. Add Fuel Prices

**settings/fuels.yml**:

```yaml
user_fuel_price:
  hydrogen:
    2030:
      CA_N: 25  # $/MMBtu
      CA_S: 25
      AZ: 28
    2040:
      CA_N: 18
      CA_S: 18
      AZ: 20
    2050:
      CA_N: 12
      CA_S: 12
      AZ: 14
```

### 4. Set Regional Availability

**settings/tech_availability.yml**:

```yaml
new_gen_not_available:
  AZ:
    - Hydrogen  # Not available in Arizona (no infrastructure)
```

## Troubleshooting

### Technology Not Appearing

**Problem**: Technology defined but not in output

**Check**:

1. Technology in `new_resources`?
2. Technology size specified (fourth element in list)?
3. Tags defined in `model_tag_values`?
4. Technology restricted in `new_gen_not_available`?

### Renewable Clustering Issues

**Problem**: No renewable resources in output

**Solution**:

- Verify `renewables_clusters` parameter is configured
- Check filter thresholds (e.g., `max: 75` for LCOE)
- Ensure renewable resource data is available in data source
- Check that technology names match data (lowercase: `landbasedwind`, `utilitypv`, `offshorewind`)

### Cost Data Missing

**Problem**: `KeyError: 'capex_mw'`

**Solution**:

- For ATB technologies: Verify `resource_data_year` has that technology
- For additional technologies: Check all required columns in CSV
- Check technology name spelling matches exactly

## Best Practices

1. **Use wildcards**: `Battery_*` matches all battery durations
2. **Start with ATB**: Prefer ATB technologies when available
3. **Document assumptions**: Comment custom cost adjustments
4. **Validate tags**: All technologies need complete tag assignments
5. **Test incrementally**: Add one technology at a time
6. **Regional variation**: Use `resource_modifiers` for regional costs
7. **Version data**: Track ATB year, profile vintage in documentation

## Next Steps

- [Configure Renewable Clusters](configure-renewable-clusters.md): Detailed renewable siting
- [Settings Reference - New-Build Resources](../reference/settings/new-build.md): Complete parameter documentation
- [Settings Reference - Resource Tags](../reference/settings/resource-tags.md): Tag configuration
