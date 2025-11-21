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

- `NaturalGas_CCAvgCF_Moderate`: Combined cycle gas turbine
- `NaturalGas_CTAvgCF_Moderate`: Combustion turbine (peaking)
- `UtilityPV_Class1_Moderate`: Utility-scale solar PV
- `LandbasedWind_Class3_Moderate`: Onshore wind
- `OffshoreWind_Class1_Moderate`: Offshore wind
- `Battery_*_Moderate`: Battery storage (2hr, 4hr, 8hr, 10hr)
- `Nuclear_Nuclear_Moderate`: Nuclear power
- `Geothermal_HydroFlash_Moderate`: Geothermal

**Cost cases**:

- `Conservative`: High cost
- `Moderate`: Mid cost
- `Advanced`: Low cost (R&D success)

### Include in Model

Add technologies to `new_resources`:

```yaml
new_resources:
  - [NaturalGas, CCAvgCF, Moderate, 500]
  - [NaturalGas, CTAvgCF, Moderate, 100]
  - [UtilityPV, Class1, Moderate, 100]
  - [LandbasedWind, Class3, Moderate, 100]
  - [Battery, "*", Moderate, 100]  # Wildcard for all battery durations
```

Specify unit sizes in `new_resources` (fourth element of each list):

```yaml
new_resources:
  - [NaturalGas, CCAvgCF, Moderate, 500]  # 500 MW per unit
  - [NaturalGas, CTAvgCF, Moderate, 100]  # 100 MW per unit
  - [UtilityPV, Class1, Moderate, 100]
  - [LandbasedWind, Class3, Moderate, 100]
  - [Battery, "4Hr", Moderate, 100]
  - [Battery, "8Hr", Moderate, 50]
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
  NaturalGas_CCAvgCF_Moderate:
    THERM: 1
    VRE: 0
    STOR: 0
    MUST_RUN: 0

  UtilityPV_Class1_Moderate:
    THERM: 0
    VRE: 1
    STOR: 0
    MUST_RUN: 0

  Battery_*:  # Applies to all battery durations
    THERM: 0
    VRE: 0
    STOR: 1
    MUST_RUN: 0
```

## Modify ATB Technology Costs

### Resource Modifiers

Adjust parameters of existing ATB technologies:

```yaml
resource_modifiers:
  UtilityPV_Class1_Moderate:
    capex_mw:
      2030: 1.2  # Multiply by 1.2 (20% higher)
      2040: 1.1  # 10% higher in 2040
    fixed_o_m_mw:
      2030: 18000  # Override to $18,000/MW-year

  LandbasedWind_Class3_Moderate:
    capex_mw:
      2030: 0.9  # 10% cost reduction
    variable_o_m_mwh:
      2030: 0  # Zero variable O&M

  Battery_4Hr_Moderate:
    capex_mwh:  # Storage energy cost
      2030: 250000  # $250/kWh = $250,000/MWh
    capex_mw:  # Storage power cost
      2030: 150000  # $150/kW = $150,000/MW
```

**Modifier behavior**:

- Values < 1: Cost reduction (0.9 = 10% cheaper)
- Values > 1: Cost increase (1.2 = 20% more expensive)
- Absolute values: Direct override (not multiplication)

### Regional Cost Multipliers

Apply regional construction cost differences:

```yaml
cost_multiplier_fn: regional_cost_multipliers.csv
cost_multiplier_region_map:
  CA_N: PCA_WECC
  CA_S: PCA_WECC
  AZ: PCA_WECC
```

**regional_cost_multipliers.csv**:

```csv
region,technology,multiplier
PCA_WECC,UtilityPV,1.25
PCA_WECC,LandbasedWind,1.15
PCA_WECC,Battery_4Hr,1.1
PCA_EAST,UtilityPV,0.95
PCA_EAST,OffshoreWind,1.3
```

Multipliers apply on top of `resource_modifiers`.

## Create Technology Variants

### Modified New Resources

Create renamed copies with different parameters:

```yaml
modified_new_resources:
  # High-cost solar variant
  UtilityPV_Class1_High:
    base_resource: UtilityPV_Class1_Moderate
    capex_mw:
      2030: 1.5  # 50% more expensive
      2040: 1.4

  # Advanced battery with longer duration
  Battery_12Hr_Advanced:
    base_resource: Battery_8Hr_Moderate
    capex_mwh:
      2030: 200000  # Lower energy cost
    duration: 12  # 12-hour storage

  # Offshore wind with high capacity factor
  OffshoreWind_HighCF:
    base_resource: OffshoreWind_Class1_Moderate
    capex_mw:
      2030: 1.1  # Slightly higher cost
    capacity_factor:
      2030: 0.5  # 50% CF instead of ATB default
```

**Include in model**:

```yaml
new_resources:
  UtilityPV_Class1_High: 100
  Battery_12Hr_Advanced: 50
  OffshoreWind_HighCF: 200
```

**Assign tags**:

```yaml
model_tag_values:
  UtilityPV_Class1_High:
    THERM: 0
    VRE: 1
    STOR: 0

  Battery_12Hr_Advanced:
    THERM: 0
    VRE: 0
    STOR: 1
    LDS: 1  # Long-duration storage
```

## Add Non-ATB Technologies

### Additional Technologies CSV

Define custom technologies not in ATB:

**settings**:

```yaml
additional_technologies_fn: additional_technologies.csv
```

**additional_technologies.csv**:

```csv
technology,tech_detail,model_year,capex_mw,fixed_o_m_mw,variable_o_m_mwh,heat_rate_mmbtu_mwh,fuel,wacc_real,capital_recovery_period,unit_size_mw
Biomass,Dedicated,2030,3500000,45000,5,10.5,biomass,0.05,20,50
Biomass,Dedicated,2040,3200000,42000,5,10.2,biomass,0.05,20,50
Hydrogen,Turbine,2030,800000,15000,3,8.5,hydrogen,0.06,25,100
Hydrogen,Turbine,2040,650000,12000,3,8.2,hydrogen,0.06,25,100
CCS_Retrofit,Coal,2030,1500000,60000,8,11.0,coal,0.05,30,300
```

**Required columns**:

- `technology`: Technology name
- `tech_detail`: Technology variant
- `model_year`: Model year
- `capex_mw`: Capital cost ($/MW)
- `fixed_o_m_mw`: Fixed O&M ($/MW-year)
- `variable_o_m_mwh`: Variable O&M ($/MWh)
- `unit_size_mw`: Unit size (MW)

**Optional columns**:

- `heat_rate_mmbtu_mwh`: Thermal efficiency (thermal only)
- `fuel`: Fuel type
- `wacc_real`: Real weighted average cost of capital
- `capital_recovery_period`: Economic lifetime (years)
- `capacity_factor`: Default capacity factor

### Include Additional Technologies

```yaml
# Include in model
new_resources:
  Biomass_Dedicated: 50
  Hydrogen_Turbine: 100
  CCS_Retrofit_Coal: 300

# Assign resource tags
model_tag_values:
  Biomass_Dedicated_Moderate:
    THERM: 1
    VRE: 0
    STOR: 0
    MUST_RUN: 0

  Hydrogen_Turbine_Moderate:
    THERM: 1
    VRE: 0
    STOR: 0
    MUST_RUN: 0

  CCS_Retrofit_Coal_Moderate:
    THERM: 1
    VRE: 0
    STOR: 0
    MUST_RUN: 0

# Map to fuel prices
tech_fuel_map:
  Biomass_Dedicated: biomass
  Hydrogen_Turbine: hydrogen
  CCS_Retrofit_Coal: coal
```

## Configure Renewable Clusters

### Renewable Resource Groups

Define specific wind/solar sites with pre-computed profiles:

```yaml
renewable_clusters:
  LandbasedWind_Class3_Moderate:
    - region: CA_N
      cluster: 1
      profile_id: LandbasedWind_Class3_1
      capacity_mw: 500
      capex_mw:
        2030: 1600000
    - region: CA_N
      cluster: 2
      profile_id: LandbasedWind_Class3_2
      capacity_mw: 800
      capex_mw:
        2030: 1650000
    - region: CA_S
      cluster: 1
      profile_id: LandbasedWind_Class3_3
      capacity_mw: 1200
      capex_mw:
        2030: 1550000

  UtilityPV_Class1_Moderate:
    - region: CA_N
      cluster: 1
      profile_id: UtilityPV_Class1_1
      capacity_mw: 2000
    - region: CA_S
      cluster: 1
      profile_id: UtilityPV_Class1_2
      capacity_mw: 3500
```

**Parameters**:

- `region`: Model region
- `cluster`: Cluster ID (unique within region/technology)
- `profile_id`: Filename prefix for generation profile
- `capacity_mw`: Maximum capacity at this site
- `capex_mw`: Site-specific capital cost (optional)

### Generation Profiles

Profiles must exist in `RESOURCE_GROUP_PROFILES` folder:

```yaml
RESOURCE_GROUP_PROFILES: /data/nrel_profiles
```

**File structure**:

```
/data/nrel_profiles/
├── LandbasedWind_Class3_1.csv
├── LandbasedWind_Class3_2.csv
├── LandbasedWind_Class3_3.csv
├── UtilityPV_Class1_1.csv
└── UtilityPV_Class1_2.csv
```

**Profile format** (8760 rows, one column):

```csv
0.32
0.35
0.41
...
```

Values are capacity factors (0-1).

### Capacity Limits and Spur Costs

Limit new-build capacity and add interconnection costs:

```yaml
capacity_limit_spur_fn: capacity_limits.csv
```

**capacity_limits.csv**:

```csv
region,technology,cluster,max_capacity,spur_miles,spur_capex_mw_mile
CA_N,LandbasedWind_Class3_Moderate,1,500,15,30000
CA_N,LandbasedWind_Class3_Moderate,2,800,25,30000
CA_S,UtilityPV_Class1_Moderate,1,3500,5,25000
```

**Parameters**:

- `max_capacity`: Maximum MW at site
- `spur_miles`: Transmission distance to grid
- `spur_capex_mw_mile`: Spur line cost ($/MW/mile)

## Restrict Technology Availability

### Regional Restrictions

Prohibit technologies in specific regions:

```yaml
new_gen_not_available:
  AZ:
    - OffshoreWind_*  # No offshore wind in Arizona
    - Geothermal_*
  CA_N:
    - Coal_*  # No new coal anywhere in California
  CA_S:
    - Coal_*
```

### Technology Minimum Load

Set minimum stable generation level:

```yaml
min_cap_req:
  NaturalGas_CCAvgCF_Moderate: 0.3  # 30% minimum load
  Nuclear_Nuclear_Moderate: 0.9  # 90% minimum load
  Coal_NewAvgCF_Moderate: 0.4
```

### Ramp Rates

Define maximum ramp rates (fraction per hour):

```yaml
ramp_up_rates:
  NaturalGas_CCAvgCF_Moderate: 0.5  # 50% per hour
  Coal_NewAvgCF_Moderate: 0.2  # 20% per hour
  Nuclear_Nuclear_Moderate: 0.05  # 5% per hour

ramp_down_rates:
  NaturalGas_CCAvgCF_Moderate: 0.5
  Coal_NewAvgCF_Moderate: 0.2
  Nuclear_Nuclear_Moderate: 0.05
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

# Add to new resources
new_resources:
  Hydrogen_Turbine: 150

# Assign tags
model_tag_values:
  Hydrogen_Turbine_Moderate:
    THERM: 1
    VRE: 0
    STOR: 0
    FLEX: 0
    HYDRO: 0
    MUST_RUN: 0

# Map to fuel
tech_fuel_map:
  Hydrogen_Turbine: hydrogen

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
    - Hydrogen_Turbine_Moderate  # Not available in Arizona (no infrastructure)
```

## Troubleshooting

### Technology Not Appearing

**Problem**: Technology defined but not in output

**Check**:

1. Technology in `new_resources`?
2. Technology size specified (fourth element in list)?
3. Tags defined in `model_tag_values`?
4. Technology restricted in `new_gen_not_available`?

### Profile Not Found

**Problem**: `FileNotFoundError: LandbasedWind_Class3_1.csv`

**Solution**:

- Check `RESOURCE_GROUP_PROFILES` path
- Verify filename matches `profile_id` in `renewable_clusters`
- Ensure profile has 8760 rows

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
