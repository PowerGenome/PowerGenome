# Output File Schemas

PowerGenome generates a folder of CSV files for each scenario/year case. These files conform to the [GenX](https://genxproject.github.io/GenX.jl/) input format.

!!! note "Macro output"
    When Macro output is enabled (see [Output Formats](../settings/macro-output.md)), PowerGenome also writes a
    MacroEnergy.jl `simpleCSVinputs` case alongside the GenX files. This page documents the GenX layout; the
    Macro layout and its GenX-to-Macro mapping are described in the [Output File Format explanation](../../explanation/output-format.md).

!!! note "v6 vs legacy format"
    The default is the split v6 layout. Set `old_genx_format: true` for the
    legacy layout, which puts `Generators_data.csv` and `Load_data.csv` directly
    in the period folder.

---

## Folder layout

```
results/<case_id>/Inputs/Inputs_p<N>/
├── system/
│   ├── Demand_data.csv
│   ├── Fuels_data.csv
│   ├── Generators_variability.csv
│   ├── Network.csv
│   ├── Period_map.csv
│   ├── Representative_Period.csv
│   └── Operational_reserves.csv
├── resources/                               # One file per resource type
│   ├── Thermal.csv
│   ├── Vre.csv
│   ├── Storage.csv
│   ├── Must_run.csv
│   ├── Flex_demand.csv
│   ├── Hydro.csv
│   └── policy_assignments/
│       ├── Resource_energy_share_requirement.csv
│       ├── Resource_capacity_reserve_margin.csv
│       ├── Resource_minimum_capacity_requirement.csv
│       └── Resource_maximum_capacity_requirement.csv
└── policies/
    ├── Energy_share_requirement.csv
    ├── Capacity_reserve_margin.csv
    ├── CO2_cap.csv
    ├── Minimum_capacity_requirement.csv
    └── Maximum_capacity_requirement.csv
```

Without scenario definitions, the period folder is `results/Inputs/Inputs_p<N>/`.
Files that are empty or correspond to disabled features are omitted.

---

## System files

### `Demand_data.csv` / `Load_data.csv`

Hourly (or representative-period) electricity demand for each model zone.

| Column | Description |
|---|---|
| `Time_Index` | Hour index (1-based) |
| `Demand_MW_z{N}` | Load in MW for zone N (one column per zone) |
| `Rep_Periods` | Number of representative periods |
| `Timesteps_per_Rep_Period` | Hours per representative period |
| `Sub_Weights` | Hours represented by each period, stored in the first rows |

The legacy file uses `Load_MW_z{N}` instead of `Demand_MW_z{N}`.

### `Fuels_data.csv`

Fuel prices and emission factors for each planning period.

| Column | Description |
|---|---|
| `Time_Index` | `0` for the emission-factor row; `1` onward for hourly prices |
| `{fuel_name}` | CO₂ content (tonnes/MMBtu) in row 0, then hourly price ($/MMBtu) |

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
| `Network_zones` | Zone label (for example `z1`); rows also carry line data |
| `Start_Zone`, `End_Zone` | Numeric zone identifiers at each end of a line |
| `Line_Max_Flow_MW` | Maximum forward flow |
| `Line_Loss_Percentage` | Fractional transmission loss |
| `Line_Max_Reinforcement_MW` | Maximum allowed capacity expansion |
| `Line_Reinforcement_Cost_per_MWyr` | Annualized expansion cost ($/MW-year) |
| `CapRes_Excl_{N}` | Capacity reserve zone exclusion flag |

### `Period_map.csv`

Maps each representative period to the full-year hours it represents (only present when time-domain reduction is used).

| Column | Description |
|---|---|
| `Period_Index` | Row index |
| `Rep_Period_Index` | Index of the selected representative period |
| `Rep_Period` | Original period number selected as the representative |
| `Month` | Month associated with the original period |

### `Representative_Period.csv`

Original periods selected as representatives. Weights are in `Demand_data.csv`.

| Column | Description |
|---|---|
| `slot` | Original period label, such as `p42` |

---

## Resource files (v6 format)

Each file contains only resources of that type. Columns common to all resource files:

| Column | Description |
|---|---|
| `Resource` | Unique resource name |
| `Zone` | Integer zone index |
| `Existing_Cap_MW` | Existing capacity operating at the end of the planning period |
| `New_Build` | 1 = new build allowed, 0 = fixed |
| `Max_Cap_MW` | Maximum buildable capacity (-1 = unlimited) |
| `Min_Cap_MW` | Minimum required capacity |
| `Inv_Cost_per_MWyr` | Annualized investment cost ($/MW-year) |
| `Fixed_OM_Cost_per_MWyr` | Fixed O&M cost ($/MW-year) |
| `Var_OM_Cost_per_MWh` | Variable O&M cost ($/MWh) |
| `Reg_Cost` | Regulation cost ($/MW) |
| `Rsv_Cost` | Spinning reserve cost ($/MW) |

### `Thermal.csv`

Additional columns for thermal generators (tag `THERM > 0`, written as `Model`):

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

Battery or other storage resources (`STOR > 0`, written as `Model`):

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
| `Hydro_Energy_to_Power_Ratio` | Hydro energy-to-power ratio |
| `Eff_Up` / `Eff_Down` | Charging and discharging efficiencies, where applicable |
| `LDS` | Long-duration storage indicator, where applicable |

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
| `CO_2_Cap_Zone_{N}` | Zone membership for cap N |
| `CO_2_Max_Mtons_{N}` | Absolute cap N in million tonnes, when supplied |

Columns are taken from the configured emission policy table, so other supported
`CO_2` policy columns may appear.

### `Minimum_capacity_requirement.csv` / `Maximum_capacity_requirement.csv`

Min/max capacity build requirements per zone and technology group.

| Column | Description |
|---|---|
| `MinCapReqConstraint` / `MaxCapReqConstraint` | Constraint ID |
| `Constraint_Description` | Human-readable label |
| `Min_MW` / `Max_MW` | Required capacity in MW |

---

## Resource policy attribute files

These live in `resources/policy_assignments/` and link individual resources to
policy constraints. Rows without a positive contribution to any constraint in
the file are omitted.

### `Resource_energy_share_requirement.csv`

| Column | Description |
|---|---|
| `Resource` | Resource name |
| `ESR_{N}` | Eligibility for ESR N (0 or 1) |

### `Resource_capacity_reserve_margin.csv`

| Column | Description |
|---|---|
| `Resource` | Resource name |
| `Derating_factor_{N}` | Contribution fraction for reserve constraint N (0–1) |

### `Resource_minimum_capacity_requirement.csv` / `Resource_maximum_capacity_requirement.csv`

| Column | Description |
|---|---|
| `Resource` | Resource name |
| `Min_Cap_{N}` / `Max_Cap_{N}` | 1 if resource counts toward constraint N, else 0 |
