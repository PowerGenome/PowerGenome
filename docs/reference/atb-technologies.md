# ATB Technology Names

PowerGenome uses technology names from NREL's Annual Technology Baseline (ATB) for new-build generation resources. This page documents the valid `technology` and `tech_detail` names for each ATB version.

!!! info "Technology Naming Convention"
    ATB technologies use a two-part naming structure: `<technology>_<tech_detail>`. The `technology` indicates the generation type (e.g., `LandbasedWind`), while `tech_detail` specifies resource quality, configuration, or capture technology (e.g., `Class3`, `Moderate`).

## Cost Scenarios

Most ATB technologies support three cost scenarios:

- **Conservative**: High-cost projection
- **Moderate**: Mid-range cost projection (default)
- **Advanced**: Low-cost projection with optimistic technology improvements

These are specified separately in settings (e.g., `atb_cost_case: Moderate`).

---

## ATB 2024

NREL's [2024 Annual Technology Baseline](https://atb.nrel.gov/electricity/2024/technologies) provides cost and performance data for the following technologies:

### Biopower

**Technology**: `Biopower`
**Tech Details**: `Dedicated`

Dedicated biomass power plants.

### Concentrating Solar Power (CSP)

**Technology**: `CSP`
**Tech Details**: `Class8`, `Class3`, `Class2`

!!! warning
    PowerGenome includes capex and FOM for CSP, but **users must provide generation profiles**.

### Coal

**Technology**: `Coal`
**Tech Details**:

- `New` - Supercritical pulverized coal without capture
- `IGCC` - Integrated gasification combined cycle without capture
- `IGCC-90%-CCS` - IGCC with 90% carbon capture
- `95%-CCS` - Supercritical PC with 95% capture
- `99%-CCS` - Supercritical PC with 99% capture

!!! note
    CCS options do **not** include CO₂ pipeline or storage costs. Add these separately using `co2_pipeline_cost_fn`.

### Commercial PV

**Technology**: `CommPV`
**Tech Details**: `Class1` through `Class10`

!!! warning
    PowerGenome includes capex and FOM, but **users must provide generation profiles**.

### Geothermal

**Technology**: `Geothermal`
**Tech Details**:

- `NFEGSFlash` - Near-field EGS flash
- `NFEGSBinary` - Near-field EGS binary
- `HydroFlash` - Hydrothermal flash
- `HydroBinary` - Hydrothermal binary
- `DeepEGSFlash` - Deep EGS flash
- `DeepEGSBinary` - Deep EGS binary

!!! warning
    Users must provide resource potential and temperature data.

### Hydropower

**Technology**: `Hydropower`
**Tech Details**:

- **Non-powered dams (NPD)**: `NPD1` through `NPD8`
- **New stream-reach development (NSD)**: `NSD1` through `NSD4`

See [ATB documentation](https://atb.nrel.gov/electricity/2024/hydropower) for resource class descriptions.

### Land-Based Wind

**Technology**: `LandbasedWind`
**Tech Details**: `Class1` through `Class10`

- **Class 1**: Highest quality wind resource
- **Class 4**: Representative of majority of US projects
- **Classes 1-7**: Same capex
- **Classes 8-10**: Higher capex for lower-quality resources

### Natural Gas

**Technology**: `NaturalGas`
**Tech Details**:

- `Combustion Turbine (F-Frame)` - Simple cycle gas turbine
- `2-on-1 Combined Cycle (F-Frame)` - Standard combined cycle
- `2-on-1 Combined Cycle (H-Frame)` - High-efficiency combined cycle
- `1-on-1 Combined Cycle (H-Frame)` - Single-train H-frame CC
- `2-on-1 Combined Cycle (F-Frame) 97% CCS`
- `2-on-1 Combined Cycle (F-Frame) 95% CCS`
- `2-on-1 Combined Cycle (H-Frame) 97% CCS`
- `2-on-1 Combined Cycle (H-Frame) 95% CCS`
- `1-on-1 Combined Cycle (H-Frame) 95% CCS`
- `1-on-1 Combined Cycle (H-Frame) 97% CCS`
- `Fuel Cell` - Solid oxide fuel cell
- `Fuel Cell 98% CCS` - Fuel cell with carbon capture

!!! note
    CCS options do **not** include CO₂ pipeline or storage costs.

### Offshore Wind

**Technology**: `OffShoreWind`
**Tech Details**:

- `Class1` through `Class7` - Fixed-bottom offshore wind
- `Class8` through `Class14` - Floating offshore wind

**Recommendations**:

- **Class 3**: Best represents fixed-bottom resources
- **Class 12**: Best represents California Wind Energy Areas (floating)

!!! warning "Floating Wind Costs"
    ATB2024 floating turbine costs (Class8-14) start in **2030**. No cost data available for years before 2030.

### Residential PV

**Technology**: `ResPV`
**Tech Details**: `Class1` through `Class10`

Each class has different capacity factor but identical capex.

### Utility-Scale PV

**Technology**: `UtilityPV`
**Tech Details**: `Class1` through `Class10`

Each class has different capacity factor but identical capex.

### Nuclear

**Technology**: `Nuclear`
**Tech Details**:

- `Nuclear - Small` - Small modular reactors (~300 MW)
- `Nuclear - Large` - Large reactors (~1,000 MW)

### Battery Storage

**Technology**: `Utility-Scale Battery Storage`
**Tech Details**: `Lithium Ion`

Utility-scale systems with separate MW (power) and MWh (energy) costs. PowerGenome data includes only utility-scale costs. Fixed O&M is 2.5% of capex.

### Pumped Storage Hydropower

**Technology**: `Pumped Storage Hydropower`
**Tech Details**: `NatlClass 1` through `NatlClass 15`

See [ATB2024 documentation](https://atb.nrel.gov/electricity/2024/pumped_storage_hydropower) for resource categorization.

---

## ATB 2022

NREL's [2022 Annual Technology Baseline](https://atb.nrel.gov/electricity/2022/technologies) includes:

!!! danger "IMPORTANT: Natural Gas CCS Warning"
    In ATB2022, NREL uses NETL data for natural gas technologies. **Advanced and Moderate NGCCCCS do NOT represent combined cycle plants**. By 2030, Advanced technology is a solid oxide fuel cell. Moderate costs/heat rates are averages of Conservative/Advanced. **Use Conservative if you want a combined cycle plant with carbon capture.**

### Biopower

**Technology**: `Biopower`
**Tech Details**: `Dedicated`, `CofireOld`, `CofireNew`

### CSP

**Technology**: `CSP`
**Tech Details**: `Class5`, `Class3`, `Class1`

### Coal

**Technology**: `Coal`
**Tech Details**:

- `newAvgCF` - New supercritical PC without capture
- `IGCCAvgCF` - IGCC without capture
- `CCS90AvgCF` - Coal with 90% carbon capture

### Commercial PV

**Technology**: `CommPV`
**Tech Details**: `Class1` through `Class10`

Each class has different capacity factor but identical capex.

### Geothermal

**Technology**: `Geothermal`
**Tech Details**: `NFEGSFlash`, `NFEGSBinary`, `HydroFlash`, `HydroBinary`, `DeepEGSFlash`, `DeepEGSBinary`

### Hydropower

**Technology**: `Hydropower`
**Tech Details**: `NSD1` through `NSD4`, `NPD1` through `NPD8`

### Land-Based Wind

**Technology**: `LandbasedWind`
**Tech Details**: `Class1` through `Class10`

All wind classes in ATB2022 have identical capex. **Class 4** represents the majority of US projects. **Class 1** is the highest quality resource.

### Natural Gas

**Technology**: `NaturalGas`
**Tech Details**:

- `CTAvgCF` - Combustion turbine
- `CCAvgCF` - Combined cycle without capture
- `CCCCSAvgCF` - Combined cycle/fuel cell with 90% capture

!!! danger
    See warning above about CCS technologies.

### Offshore Wind

**Technology**: `OffShoreWind`
**Tech Details**:

- `Class1` through `Class7` - Fixed-bottom (each has different capex)
- `Class8` through `Class14` - Floating (each has different capex)

**Recommendations**:

- **Class 3**: Best for near-term fixed-bottom deployment
- **Class 12**: Best for California Wind Energy Areas

### Residential PV

**Technology**: `ResPV`
**Tech Details**: `Class1` through `Class10`

### Utility-Scale PV

**Technology**: `UtilityPV`
**Tech Details**: `Class1` through `Class10`

### Battery Storage

**Technology**: `Battery`
**Tech Details**: `*` (wildcard - duration determined by model)

ATB2022 reports batteries in 2-10 hour configurations. NREL provides energy ($/kWh) and power ($/kW) costs separately. PowerGenome provides both, allowing models to determine optimal storage duration. Fixed O&M is 2.5% of capex for both components.

### Pumped Storage Hydropower

**Technology**: `Pumped Storage Hydropower`
**Tech Details**: `NatlClass 1` through `NatlClass 15`

### Distributed Wind

**Technology**: `DistributedWind`
**Tech Details**:

- `LargeScaleClass1-10`
- `MidsizeScaleClass1-10`
- `CommercialScaleClass1-10`
- `ResidentialScaleClass1-10`

!!! warning
    Costs included, but users must provide generation profiles.

---

## ATB 2021

NREL's [2021 Annual Technology Baseline](https://atb.nrel.gov/electricity/2021/technologies) includes:

!!! danger "IMPORTANT: Natural Gas CCS Warning"
    Same as ATB2022: Advanced and Moderate NGCCCCS do NOT represent combined cycle plants. **Use Conservative for combined cycle with carbon capture.**

### Biopower

**Technology**: `Biopower`
**Tech Details**: `Dedicated`, `CofireOld`, `CofireNew`

### CSP

**Technology**: `CSP`
**Tech Details**: `Class5`, `Class3`, `Class1`

### Coal

**Technology**: `Coal`
**Tech Details**:

- `newHighCF`, `newAvgCF` - New supercritical PC
- `IGCCHighCF`, `IGCCAvgCF` - IGCC
- `CCS90HighCF`, `CCS90AvgCF` - 90% carbon capture
- `CCS36HighCF`, `CCS36AvgCF` - 36% carbon capture

### Commercial PV

**Technology**: `CommPV`
**Tech Details**: `Class1` through `Class10`

### Geothermal

**Technology**: `Geothermal`
**Tech Details**: `NFEGSFlash`, `NFEGSBinary`, `HydroFlash`, `HydroBinary`, `DeepEGSFlash`, `DeepEGSBinary`

### Hydropower

**Technology**: `Hydropower`
**Tech Details**: `NSD1` through `NSD4`, `NPD1` through `NPD4`

### Land-Based Wind

**Technology**: `LandbasedWind`
**Tech Details**: `Class1` through `Class10`

All classes have identical capex (change from ATB2020). **Class 4** represents majority of US projects.

### Natural Gas

**Technology**: `NaturalGas`
**Tech Details**:

- `CTHighCF`, `CTAvgCF` - Combustion turbine
- `CCHighCF`, `CCAvgCF` - Combined cycle
- `CCCCSHighCF`, `CCCCSAvgCF` - Combined cycle/fuel cell with 90% CCS

!!! danger
    See warning above about CCS technologies.

### Offshore Wind

**Technology**: `OffShoreWind`
**Tech Details**:

- `Class1` through `Class7` - Fixed-bottom
- `Class8` through `Class14` - Floating

**Recommendations**:

- **Class 3**: Near-term fixed-bottom deployment
- **Class 13**: Floating resources

### Residential PV

**Technology**: `ResPV`
**Tech Details**: `Class1` through `Class10`

### Utility-Scale PV

**Technology**: `UtilityPV`
**Tech Details**: `Class1` through `Class10`

### Battery Storage

**Technology**: `Battery`
**Tech Details**: `*`

Reported in 2-10 hour configurations with separate energy/power costs. Fixed O&M is 2.5% of capex.

---

## ATB 2020

NREL's [2020 Annual Technology Baseline](https://atb-archive.nrel.gov/electricity/2020/) uses different cost scenario names:

!!! note "Cost Scenario Names"
    ATB2020 originally used `High`, `Mid`, and `Low` instead of Conservative/Moderate/Advanced. NREL has since updated files to use the newer naming convention.

### Biopower

**Technology**: `Biopower`
**Tech Details**: `Dedicated`, `CofireOld`, `CofireNew`

### CSP

**Technology**: `CSP`
**Tech Details**: `Class5`, `Class3`, `Class1`

### Coal

**Technology**: `Coal`
**Tech Details**:

- `newHighCF`, `newAvgCF` - New supercritical PC
- `IGCCHighCF`, `IGCCAvgCF` - IGCC
- `CCS90HighCF`, `CCS90AvgCF` - 90% capture
- `CCS30HighCF`, `CCS30AvgCF` - 30% capture

### Commercial PV

**Technology**: `CommPV`
**Tech Details**: `Seattle`, `LosAngeles`, `KansasCity`, `Daggett`, `Chicago`

Each city has different capacity factor.

### Geothermal

**Technology**: `Geothermal`
**Tech Details**: `NFEGSFlash`, `NFEGSBinary`, `HydroFlash`, `HydroBinary`, `DeepEGSFlash`, `DeepEGSBinary`

### Hydropower

**Technology**: `Hydropower`
**Tech Details**: `NSD1` through `NSD4`, `NPD1` through `NPD8`

### Land-Based Wind

**Technology**: `LandbasedWind`
**Tech Details**: `LTRG1` through `LTRG10`

Each TRG (Terrestrial Reference Gradation) has different capex representing different turbine technologies. **LTRG4** represents majority of US projects. **LTRG1** is highest quality.

### Natural Gas

**Technology**: `NaturalGas`
**Tech Details**:

- `CTHighCF`, `CTAvgCF` - Combustion turbine
- `CCHighCF`, `CCAvgCF` - Combined cycle
- `CCCCSHighCF`, `CCCCSAvgCF` - Combined cycle with 90% CCS

### Offshore Wind

**Technology**: `OffShoreWind`
**Tech Details**:

- `OTRG1` through `OTRG7` - Fixed-bottom (each has different capex)
- `OTRG8` through `OTRG15` - Floating

**Recommendations**:

- **OTRG3**: Near-term fixed-bottom deployment
- **OTRG13**: Floating resources

### Residential PV

**Technology**: `ResPV`
**Tech Details**: `Seattle`, `LosAngeles`, `KansasCity`, `Daggett`, `Chicago`

### Utility-Scale PV

**Technology**: `UtilityPV`
**Tech Details**: `Seattle`, `LosAngeles`, `KansasCity`, `Daggett`, `Chicago`

### Battery Storage

**Technology**: `Battery`
**Tech Details**: `*`

Reported in 2 and 4-hour configurations with separate energy/power costs. Fixed O&M is 2.5% of capex.

---

## Usage in Settings

To use ATB technologies in your PowerGenome configuration:

```yaml
# Specify ATB version and cost scenario
atb_data_year: 2024
atb_cost_case: Moderate  # Conservative, Moderate, or Advanced

# List available new-build technologies
atb_new_gen:
  - LandbasedWind_Class3_Moderate
  - UtilityPV_Class1_Moderate
  - NaturalGas_CCAvgCF_Moderate
  - Battery_*_Moderate  # Wildcard for all battery durations
```

The format is `<technology>_<tech_detail>_<cost_case>`.

See [Adding Technologies](../how-to/add-technologies.md) for complete configuration details.

## Related Resources

- [NREL ATB Website](https://atb.nrel.gov/)
- [Adding Technologies Guide](../how-to/add-technologies.md)
- [Technology Settings Reference](settings/new-build.md)
- [Resource Modifiers](settings/new-build.md#resource_modifiers)
