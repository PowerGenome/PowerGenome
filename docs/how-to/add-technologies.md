# Add Custom Technologies

This guide shows how to add new generation technologies, modify existing ones, and configure renewable resource clusters.

## Technology Sources

PowerGenome gets technology data from several sources:

1. **NREL ATB (Annual Technology Baseline)**: Standard new-build technologies
2. **Existing generators**: From data tables (existing fleet)
3. **Additional technologies**: Your own cost tables, stacked next to the ATB
   table in `data.yml`
4. **Modified resources**: Variants of ATB technologies with adjusted parameters

## Add ATB Technologies

### List Available Technologies

New resource names follow `{Technology}_{TechDetail}_{CostCase}`. The available
names depend on the configured resource cost table and data vintage.

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
resource_data_year: 2024  # Data vintage; must exist in the configured cost table
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
    Fixed_OM_Cost_per_MWyr: 18000  # Direct override to $18,000/MW-year

  landbased_wind:
    technology: LandbasedWind
    tech_detail: Class3
    capex_mw: [mul, 0.9]  # 10% cost reduction
    Var_OM_Cost_per_MWh: 0  # Zero variable O&M

  batteries:
    technology: Utility-Scale Battery Storage
    tech_detail: Lithium Ion
    capex_mw: [add, 55000]  # Add $55,000/MW
    Var_OM_Cost_per_MWh: [add, 0.15]
    Var_OM_Cost_per_MWh_In: 0.15  # Set directly (not in ATB)
    wacc_real: 0.04675  # Override the real WACC
```

`resource_modifiers` runs after O&M columns are renamed to their GenX names;
`modified_new_resources` runs earlier and uses source names such as
`fixed_o_m_mw` and `variable_o_m_mwh`.

Both published examples set battery variable O&M this way in
`settings/resources.yml`: `Var_OM_Cost_per_MWh: [add, 0.15]` and
`Var_OM_Cost_per_MWh_In: 0.15`.

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
Configure this file with `regional_cost_factor_table`; it must cover the
technologies and regions in the study.

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

The Texas example's only entry, `hydrogen_turbine`, shows the minimum required
fields: a 100 MW copy of `NaturalGas / 1-on-1 Combined Cycle (H-Frame) /
Moderate` renamed to `hydrogen / 1-on-1 Combined Cycle (H-Frame)`, with no cost
modifiers. See
[`settings/resources.yml`](https://github.com/PowerGenome/PowerGenome-examples/blob/7e34f714469e68fca173cdb849f74c47be779420/example_systems/TX-3-zone/settings/resources.yml).

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
    pref_site: 0  # Match resource groups whose metadata has pref_site: 0
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
  - `weights`: Weight quantiles by a site-data column such as `capacity_mw`
  - `q`: Number of quantile bins
  - `mw_per_bin`: Fixed MW per bin
- `cluster`: List of clustering operations within bins
  - `feature`: Attribute to cluster by (`cf`, `profile`, `lcoe`)
  - `n_clusters`: Number of clusters
  - `method`: Clustering method (`agg` for agglomerative)

**Technology-specific parameters**:

- Offshore wind: `turbine_type` (`fixed`, `floating`), and optionally `pref_site`
  to match the corresponding resource-group metadata
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

!!! warning "Legacy setting — not applied"
    `new_gen_not_available` is only validated for region-name consistency; it is **not**
    applied to exclude technologies. Non-renewable `new_resources` are built in
    every model region. Renewable resource clusters are scoped per region through
    `renewables_clusters` (each entry has a `region` key).

```yaml
# Legacy — not applied in this version
new_gen_not_available:
  AZ:
    - OffShoreWind  # No offshore wind in Arizona
    - Nuclear
  CA_N:
    - Coal  # No new coal anywhere in California
  CA_S:
    - Coal
```

### Operating Constraints

Configure minimum output and ramping in the operational constraints input table,
not with `min_cap_req`, `ramp_up_rates`, or `ramp_down_rates` settings.

## Add a Technology With Its Own Cost Table

ATB will not have every technology you want to model. Rather than editing the ATB
table, add your own cost and heat rate tables and list them next to the ATB tables
in `data.yml`. Every table listed under a key is stacked into a single table before
it is read, so a technology from your own file is indistinguishable from an ATB
technology to the rest of the pipeline.

The walkthrough below adds a 300 MW advanced nuclear (SMR-300) resource. Each
fragment matches the
[Texas 3-zone example system](https://github.com/PowerGenome/PowerGenome-examples/tree/7e34f714469e68fca173cdb849f74c47be779420/example_systems/TX-3-zone),
where the steps were run end to end.

### 1. Create the Cost Table

New-build costs come from the `resource_cost` table, which is tidy: one row per
technology, cost case, basis year, and parameter.

| Column | Description |
| --- | --- |
| `technology` | First element of the `new_resources` entry |
| `tech_detail` | Second element |
| `cost_case` | Third element |
| `basis_year` | Year the value applies to |
| `parameter` | `capex_mw`, `fixed_o_m_mw`, `variable_o_m_mwh`, `capex_mwh` and `fixed_o_m_mwh` (storage), or `wacc_real` |
| `parameter_value` | Value |
| `financial_case` | Must match `resource_financial_case` (default `Market`) |
| `data_year` | Data vintage; must match `resource_data_year` when it is set |
| `dollar_year` | Dollar year of the values, adjusted to `target_usd_year` |
| `cap_recovery_years` | Capital recovery period, usually `resource_cap_recovery_years` |
| `units` | Informational only |

Write a `capex_mw`, `fixed_o_m_mw`, `variable_o_m_mwh`, and `wacc_real` row for every
basis year in the modeling period, plus `capex_mwh`/`fixed_o_m_mwh` for storage.

**data/custom_technology_costs.csv** (abbreviated — the real file has four rows for
each basis year from 2025 to 2045):

```csv
data_year,basis_year,cap_recovery_years,cost_case,financial_case,technology,tech_detail,parameter,parameter_value,units,dollar_year
2024,2025,20,Moderate,Market,AdvancedNuclear,SMR-300,capex_mw,6500000.0,,2024
2024,2025,20,Moderate,Market,AdvancedNuclear,SMR-300,fixed_o_m_mw,105000.0,,2024
2024,2025,20,Moderate,Market,AdvancedNuclear,SMR-300,variable_o_m_mwh,2.5,,2024
2024,2025,20,Moderate,Market,AdvancedNuclear,SMR-300,wacc_real,0.055,,2024
2024,2026,20,Moderate,Market,AdvancedNuclear,SMR-300,capex_mw,6418750.0,,2024
```

A planning period uses the average of the rows whose `basis_year` falls inside it,
so a declining cost path is a normal way to write cost projections.

!!! warning "Mismatched keys drop rows silently"
    Costs are selected with an exact match on `technology`, `tech_detail`,
    `cost_case`, `financial_case`, and (when `resource_data_year` is set) `data_year`.
    A typo in one of those columns, or a `data_year` of 2022 in a system configured
    for 2024, leaves the technology with zero costs instead of raising an error.

### 2. Add Heat Rates for Thermal Technologies

**data/custom_technology_heat_rates.csv** (abbreviated):

```csv
technology,tech_detail,cost_case,basis_year,heat_rate,data_year
AdvancedNuclear,SMR-300,Moderate,2025,10.5,2024
AdvancedNuclear,SMR-300,Moderate,2026,10.5,2024
AdvancedNuclear,SMR-300,Moderate,2027,10.5,2024
```

Heat rates use the same keys as costs. Skip this table for VRE and storage
technologies, which have no heat rate.

### 3. List Both Tables in `data.yml`

Table settings accept a list as well as a single filename. The entries are combined
with a `UNION ALL BY NAME`, so they must have the same columns — and each entry can
be either a filename or a full table configuration with its own `columns`,
`filters`, or `scenario`.

**settings/data.yml**:

```yaml
resource_cost_table:
  - technology_costs_atb_tx_3_zone.csv
  - custom_technology_costs.csv
resource_heat_rate_table:
  - technology_heat_rates_nrelatb.csv
  - custom_technology_heat_rates.csv
```

Both files are resolved in `data_location`. List entries are stacked in order, so
your table can also be listed first if a later ATB entry should win.

### 4. List and Tag the Technology

**settings/resources.yml** — the fourth element is the unit size in MW:

```yaml
new_resources:
  - [Nuclear, Nuclear - Large, Moderate, 1000]
  - [AdvancedNuclear, SMR-300, Moderate, 300]
```

**settings/fuels.yml** — assign the fuel that `tech_fuel_map` gives the technology:

```yaml
tech_fuel_map:
  AdvancedNuclear: uranium
```

If the fuel is not in the configured fuel price table, give it a price with
`user_fuel_price` (see [Fuels settings](../reference/settings/fuels.md#user_fuel_price)),
e.g. `user_fuel_price: {hydrogen: 16.0}`.

**settings/resource_tags.yml** — tag names match as substrings of the technology
name, ignoring case and underscores, and longer keys are applied last. The existing
`Nuclear` entries already cover `AdvancedNuclear`, so the specific entries below
mainly guard against `Nuclear` exclusions — without the `New_Build` entry the new
technology inherits `Nuclear: -1` and is not flagged as a new build:

```yaml
model_tag_values:
  THERM:
    AdvancedNuclear: 1
  New_Build:
    AdvancedNuclear: 1
  CapRes_1:
    AdvancedNuclear: 0.9
```

### 5. Verify

```bash
run_powergenome --settings_file example_systems/TX-3-zone/settings --results_folder tx_test
```

The new resource shows up once per model region in
`tx_test/base/Inputs/Inputs_p1/resources/Thermal.csv` and again, with the later
planning period costs, in `Inputs_p2`:

```
Resource                              region  Cap_Size  Fixed_OM_Cost_per_MWyr  Var_OM_Cost_per_MWh  Heat_Rate_MMBTU_per_MWh  Fuel         basis_year  capex_mw  Model  New_Build
TX1_advancednuclear_smr300_moderate_0  TX1         300.0                 105000                 2.5                     10.5  TX1_uranium      2027.5   6296875      1          1
TX2_advancednuclear_smr300_moderate_0  TX2         300.0                 105000                 2.5                     10.5  TX2_uranium      2027.5   6296875      1          1
TX3_advancednuclear_smr300_moderate_0  TX3         300.0                 105000                 2.5                     10.5  TX3_uranium      2027.5   6296875      1          1
```

`capex_mw` is the average of the 2025–2030 rows, and `basis_year` is the midpoint
of the years that were averaged.

!!! note "Alternative: wide-format CSV via `additional_technologies_fn`"
    The `additional_technologies_fn` + `additional_new_gen` pair still works. It
    reads a wide CSV (one column per cost parameter, one row per planning year) from
    `input_folder` instead of the data tables, and only rows named in
    `additional_new_gen` are used. Prefer the data-table route above: the costs live
    with the rest of your data, a list entry can be filtered per scenario, and no
    extra settings key is needed.

## Regional Availability of New Resources

`new_resources` builds each listed non-renewable technology in **every** model
region, so it cannot be used to keep a technology out of a single region. Renewable
resources are scoped per region through `renewables_clusters` (each entry has a
`region` key). The legacy `new_gen_not_available` key is only validated, not applied.

```yaml
new_resources:
  - [LandbasedWind, Class3, Moderate, 100]

renewables_clusters:
  - region: CA_N
    technology: landbasedwind
    min_capacity: 25000
    cluster:
      - feature: cf
        n_clusters: 2
        method: agg
```

## Troubleshooting

### Technology Not Appearing

**Problem**: Technology defined but not in output

**Check**:

1. Technology in `new_resources`, or in `additional_new_gen` if it comes from an
   `additional_technologies_fn` CSV?
2. Technology size given as the fourth list element (or as `Cap_Size` in that CSV)?
3. Cost rows findable in the `resource_cost` table — same `technology`, `tech_detail`,
   `cost_case`, `financial_case`, and `data_year`?
4. Tags defined in `model_tag_values`?
5. For renewables, is a cluster definition present for the target region?

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

- For ATB technologies: verify `resource_data_year` has that technology
- For your own cost table: verify the file is listed under `resource_cost_table` and
  has one `capex_mw`, `fixed_o_m_mw`, `variable_o_m_mwh`, and `wacc_real` row per
  basis year
- Check that `technology`, `tech_detail`, `cost_case`, `financial_case`, and
  `data_year` match the settings exactly — mismatched keys drop rows without an error
- Thermal technologies also need rows in `resource_heat_rate_table`

## Best Practices

1. **Start with the configured cost table**: Prefer existing technologies when available
2. **Document assumptions**: Comment custom cost adjustments
3. **Validate tags**: Check resource classifications
4. **Test incrementally**: Add one technology at a time
5. **Regional variation**: Use `regional_cost_factor_table` for regional costs
6. **Version data**: Track ATB year, profile vintage in documentation

## Next Steps

- [Configure Renewable Clusters](configure-renewable-clusters.md): Detailed renewable siting
- [Settings Reference - New-Build Resources](../reference/settings/new-build.md): Complete parameter documentation
- [Settings Reference - Resource Tags](../reference/settings/resource-tags.md): Tag configuration
