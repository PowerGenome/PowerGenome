# Output File Schemas

PowerGenome generates a folder of CSV files for each scenario/year case. These files conform to the [GenX](https://genxproject.github.io/GenX.jl/) input format.

!!! note "v6 vs legacy format"
    The output structure differs between GenX v6+ and the legacy format. Set `genx_v6: true` in your settings to use the v6 layout (default). The legacy format uses `Generators_data.csv` and `Load_data.csv` instead of separate resource-type files.

---

## Folder layout

```
results/case_2030/
├── Demand_data.csv                          # Hourly or representative-period load
├── Fuels_data.csv                           # Fuel prices by time period
├── Generators_variability.csv               # Capacity-factor profiles
├── Network.csv                              # Transmission topology and limits
├── Period_map.csv                           # Maps representative periods to full year
├── Representative_Period.csv                # Weight of each representative period
├── Operational_reserves.csv                 # Reserve requirement parameters
│
├── resources/                               # One file per resource type
│   ├── Thermal.csv
│   ├── Vre.csv
│   ├── Storage.csv
│   ├── Must_run.csv
│   ├── Flex_demand.csv
│   ├── Hydro.csv
│   ├── Resource_energy_share_requirement.csv
│   ├── Resource_capacity_reserve_margin.csv
│   ├── Resource_minimum_capacity_requirement.csv
│   └── Resource_maximum_capacity_requirement.csv
│
├── Energy_share_requirement.csv             # RPS/CES zone-level targets
├── Capacity_reserve_margin.csv             # Planning reserve requirements
├── CO2_cap.csv                             # Emission cap constraints
├── Minimum_capacity_requirement.csv        # Minimum build mandates
└── Maximum_capacity_requirement.csv        # Maximum build limits
```

---

## System files

### `Demand_data.csv` / `Load_data.csv`

Hourly (or representative-period) electricity demand for each model zone.

| Column | Description |
|---|---|
| `Time_Index` | Hour index (1-based) |
| `Demand_MW_z{N}` | Load in MW for zone N (one column per zone) |
| `Sub_Weights` | Weight (hours) assigned to this representative period row |

### `Fuels_data.csv`

Fuel prices for each planning period.

| Column | Description |
|---|---|
| `Fuel` | Fuel name matching resource fuel assignments |
| `{zone}` | Price in $/MMBtu for that zone (one column per zone) |
| `CO2_content_tons_per_MMBtu` | Emission factor |

### `Generators_variability.csv`

Capacity factor time series for variable and hydro resources.

| Column | Description |
|---|---|
| `Time_Index` | Hour index (1-based) |
| `{resource_name}` | Capacity factor for that resource (one column per VRE/hydro resource) |

### `Network.csv`

Transmission network topology and line parameters.

| Column | Description |
|---|---|
| `Network_Lines` | Line ID |
| `z{N}` | Zone incidence (+1 sending, -1 receiving, 0 not connected) |
| `Line_Max_Flow_MW` | Maximum forward flow |
| `Line_Min_Flow_MW` | Maximum reverse flow (negative) |
| `Line_Loss_Percentage` | Fractional transmission loss |
| `Line_Max_Reinforcement_MW` | Maximum allowed capacity expansion |
| `Line_Reinforcement_Cost_per_MWyr` | Annualized expansion cost ($/MW-year) |
| `CapRes_Excl_{N}` | Capacity reserve zone exclusion flag |

### `Period_map.csv`

Maps each representative period to the full-year hours it represents (only present when time-domain reduction is used).

| Column | Description |
|---|---|
| `Period_Index` | Row index |
| `Rep_Period` | Representative period number |
| `Rep_Period_Index` | Hour within the representative period |
| `Time_Index` | Full-year hour this row represents |

### `Representative_Period.csv`

Weight (number of hours of the full year) each representative period represents.

| Column | Description |
|---|---|
| `Rep_Period` | Representative period number |
| `Sub_Weights` | Hours this period represents in the full year |

---

## Resource files (v6 format)

Each file contains only resources of that type. Columns common to all resource files:

| Column | Description |
|---|---|
| `Resource` | Unique resource name |
| `Zone` | Integer zone index |
| `Existing_Cap_MW` | Installed capacity at start of planning period |
| `New_Build` | 1 = new build allowed, 0 = fixed |
| `Max_Cap_MW` | Maximum buildable capacity (-1 = unlimited) |
| `Min_Cap_MW` | Minimum required capacity |
| `Inv_Cost_per_MWyr` | Annualized investment cost ($/MW-year) |
| `Fixed_OM_Cost_per_MWyr` | Fixed O&M cost ($/MW-year) |
| `Var_OM_Cost_per_MWh` | Variable O&M cost ($/MWh) |
| `Reg_Cost` | Regulation cost ($/MW) |
| `Rsv_Cost` | Spinning reserve cost ($/MW) |

### `Thermal.csv`

Additional columns for thermal generators (tag `THERM = 1` or `2`):

| Column | Description |
|---|---|
| `Heat_Rate_MMBTU_per_MWh` | Average heat rate |
| `Fuel` | Fuel name |
| `Start_Cost_per_MW` | Cold-start cost ($/MW) |
| `Start_Fuel_MMBTU_per_MW` | Fuel consumed per start (MMBtu/MW) |
| `Min_Power` | Minimum stable output fraction |
| `Ramp_Up_Percentage` | Maximum hourly ramp-up fraction |
| `Ramp_Dn_Percentage` | Maximum hourly ramp-down fraction |
| `Up_Time` | Minimum up-time (hours) |
| `Down_Time` | Minimum down-time (hours) |
| `CO2_Capture_Fraction` | Fraction of CO₂ captured (for CCS resources) |

### `Vre.csv`

Variable renewable energy resources (`VRE = 1`):

| Column | Description |
|---|---|
| `Num_VRE_Bins` | Number of capacity bins for this VRE type |

Profiles are written to `Generators_variability.csv`.

### `Storage.csv`

Battery or other storage resources (`STOR = 1` or `2`):

| Column | Description |
|---|---|
| `Existing_Cap_MWh` | Installed energy capacity |
| `Max_Cap_MWh` | Maximum energy capacity (-1 = unlimited) |
| `Min_Cap_MWh` | Minimum energy capacity |
| `Inv_Cost_per_MWhyr` | Annualized energy capacity investment cost |
| `Fixed_OM_Cost_per_MWhyr` | Fixed O&M per MWh of energy capacity |
| `Eff_Up` | Charging efficiency |
| `Eff_Down` | Discharging efficiency |
| `Self_Disch` | Self-discharge rate per hour |
| `Min_Duration` | Minimum storage duration (hours) |
| `Max_Duration` | Maximum storage duration (hours) |
| `Max_Charge_Cap_MW` | Maximum charge power capacity (-1 = unlimited) |
| `Existing_Charge_Cap_MW` | Installed charge power capacity |

### `Must_run.csv`

Must-run resources (`MUST_RUN = 1`): same common columns, no additional columns required.

### `Flex_demand.csv`

Flexible demand resources (`FLEX = 1`):

| Column | Description |
|---|---|
| `Max_Flexible_Demand_Advance` | Max hours demand can be shifted earlier |
| `Max_Flexible_Demand_Delay` | Max hours demand can be shifted later |
| `Flexible_Demand_Energy_Eff` | Round-trip efficiency of demand shifting |

### `Hydro.csv`

Hydropower resources (`HYDRO = 1`):

| Column | Description |
|---|---|
| `Hydro_Energy_Share_Max` | Maximum fraction of annual energy in any period |
| `Hydro_Energy_Share_Min` | Minimum fraction of annual energy in any period |

---

## Policy files

### `Energy_share_requirement.csv`

Zone-level RPS or CES targets.

| Column | Description |
|---|---|
| `ESR_{N}` | Minimum clean energy fraction for requirement N |

Resource contributions are set in `Resource_energy_share_requirement.csv`.

### `Capacity_reserve_margin.csv`

Planning reserve requirements per zone.

| Column | Description |
|---|---|
| `CapRes_{N}` | Reserve margin fraction for zone N |

### `CO2_cap.csv`

Emission caps by zone.

| Column | Description |
|---|---|
| `CO_2_Cap_Zone_{N}` | Type of cap (1 = absolute, 2 = rate-based, 3 = fraction) |
| `CO_2_Cap_Limit_{N}` | Cap value (metric tonnes or fraction) |

### `Minimum_capacity_requirement.csv` / `Maximum_capacity_requirement.csv`

Min/max capacity build requirements per zone and technology group.

| Column | Description |
|---|---|
| `MinCapReqConstraint` / `MaxCapReqConstraint` | Constraint ID |
| `Constraint_Description` | Human-readable label |
| `Min_MW` / `Max_MW` | Required capacity in MW |

---

## Resource policy attribute files

These live in `resources/` and link individual resources to policy constraints.

### `Resource_energy_share_requirement.csv`

| Column | Description |
|---|---|
| `Resource` | Resource name |
| `ESR_{N}` | Fraction of generation counting toward ESR N (0–1; or >1 for above-market CES credit) |

### `Resource_capacity_reserve_margin.csv`

| Column | Description |
|---|---|
| `Resource` | Resource name |
| `CapRes_{N}` | Contribution fraction for reserve zone N |

### `Resource_minimum_capacity_requirement.csv` / `Resource_maximum_capacity_requirement.csv`

| Column | Description |
|---|---|
| `Resource` | Resource name |
| `MinCapTag_{N}` / `MaxCapTag_{N}` | 1 if resource counts toward constraint N, else 0 |
