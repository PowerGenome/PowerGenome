# Output File Format

PowerGenome writes input files for the [GenX](https://github.com/GenXProject/GenX.jl) capacity expansion model and,
optionally, for [MacroEnergy.jl](https://github.com/macroenergy/MacroEnergy.jl) (Macro) in its `simpleCSVinputs`
format. This page describes the folder structure, file names, and the key content of each output.

For the full GenX documentation on input file schemas, see the [GenX documentation](https://genxproject.github.io/GenX.jl/stable/).
For Macro, see the [MacroEnergy Examples](https://github.com/macroenergy/MacroEnergyExamples.jl) repository.

---

## Folder structure

Each (case_id, planning period) pair produces a folder. Files shown below are
examples; empty or disabled outputs are omitted.

```
results/
└── <case_id>/
    └── Inputs/
        └── Inputs_p<N>/           ← one per planning period
            ├── system/
            │   ├── Demand_data.csv
            │   ├── Fuels_data.csv
            │   ├── Generators_variability.csv
            │   └── Network.csv
            ├── resources/          ← one CSV per resource type
            │   ├── Thermal.csv
            │   ├── Vre.csv
            │   ├── Storage.csv
            │   ├── Hydro.csv
            │   ├── ...
            │   └── policy_assignments/
            │       ├── Resource_energy_share_requirement.csv
            │       ├── Resource_capacity_reserve_margin.csv
            │       └── ...
            ├── policies/
            │   ├── Capacity_reserve_margin.csv
            │   ├── CO2_cap.csv
            │   ├── Energy_share_requirement.csv
            │   ├── Maximum_capacity_requirement.csv
            │   └── Minimum_capacity_requirement.csv
            └── powergenome_case_settings.yml
```

When running without scenario definitions, `case_id = Inputs` and the folder is simply `results/Inputs/Inputs_p1/`.

---

## System files

### `Demand_data.csv`

Hourly electricity demand for each model zone. One row per hour, with columns:

| Column | Description |
|---|---|
| `Rep_Periods` | Number of periods in the file |
| `Timesteps_per_Rep_Period` | Hours per period |
| `Sub_Weights` | Hours each period represents |
| `Time_Index` | Sequential hour index |
| `Demand_MW_z<N>` | Load in MW for zone N |

All supplied hours are written as one period, so `Rep_Periods` is `1`,
`Timesteps_per_Rep_Period` is the number of hours, and `Sub_Weights` equals that
hour count.

### `Fuels_data.csv`

One column per fuel plus `Time_Index`. The row with index `0` contains CO₂
content (tonnes/MMBtu); subsequent rows contain hourly prices ($/MMBtu).

### `Generators_variability.csv`

Normalized capacity factor (0–1) for each variable renewable resource, by hour. One column per VRE resource, one row per timestep.

### `Network.csv`

Transmission network topology. One row per network line (inter-regional connection):

| Column | Description |
|---|---|
| `Network_Lines` | Line identifier |
| `Network_zones` | Zone label (for example `z1`); rows also carry line data |
| `Start_Zone`, `End_Zone` | Numeric zone identifiers at each end of a line |
| `Line_Max_Flow_MW` | Maximum power flow in MW |
| `Line_Loss_Percentage` | Fraction of power lost in transmission |
| `Line_Max_Reinforcement_MW` | Maximum expansion allowed (if `tx_expansion_per_period` or `tx_expansion_mw_per_period` is set) |
| `Line_Reinforcement_Cost_per_MWyr` | Annualized cost to expand line by 1 MW |

---

## Resource files

Resources are split into typed CSVs based on their primary model tag:

| File | Resources included |
|---|---|
| `Thermal.csv` | `THERM > 0` |
| `Vre.csv` | `VRE = 1` |
| `Storage.csv` | `STOR > 0` |
| `Hydro.csv` | `HYDRO = 1` |
| `Flex_demand.csv` | `FLEX = 1` |
| `Must_run.csv` | `MUST_RUN = 1` |

Long-duration storage is identified by the `LDS` column in storage or hydro files,
not a separate resource file. The writer does not currently split electrolyzers
into a separate file.

All resource files share a common set of columns, with additional columns present depending on the resource type.

### Key shared columns

| Column | Description |
|---|---|
| `Resource` | Unique resource name |
| `Zone` | Zone number (integer) |
| `Existing_Cap_MW` | Existing capacity operating at the end of the planning period |
| `New_Build` | 1 if the resource is a candidate for new investment |
| `Cap_Size` | Unit size in MW (for integer investment decisions) |
| `Inv_Cost_per_MWyr` | Annualized investment cost ($/MW/yr) |
| `Fixed_OM_Cost_per_MWyr` | Fixed O&M cost ($/MW/yr) |
| `Var_OM_Cost_per_MWh` | Variable O&M cost ($/MWh) |
| `Heat_Rate_MMBTU_per_MWh` | Thermal efficiency (thermal resources only) |
| `Fuel` | Fuel name (must match a column in `Fuels_data.csv`) |
| `Min_Power` | Minimum stable generation as fraction of capacity |

### Storage columns (additional)

| Column | Description |
|---|---|
| `Existing_Cap_MWh` | Installed energy capacity |
| `Inv_Cost_per_MWhyr` | Investment cost for energy component |
| `Eff_Up` | Charging efficiency |
| `Eff_Down` | Discharging efficiency |
| `Max_Duration` | Maximum hours of storage at rated power |
| `Min_Duration` | Minimum hours |

---

## Policy files

### `Energy_share_requirement.csv`

Defines Renewable Portfolio Standard (RPS) or Clean Energy Standard (CES) constraints
by zone:

| Column | Description |
|---|---|
| `ESR_<N>` | Minimum fraction of load that eligible resources must serve |

The companion file `resources/policy_assignments/Resource_energy_share_requirement.csv` indicates which resources are eligible for each constraint (1 = eligible, 0 = not).

### `Capacity_reserve_margin.csv`

Required reserve margin by zone and constraint:

| Column | Description |
|---|---|
| `CapRes_<N>` | Reserve requirement as a fraction of peak demand for each zone |

### `CO2_cap.csv`

Emission cap constraints. Each row is one constraint; columns reference zones.

### `Minimum_capacity_requirement.csv` / `Maximum_capacity_requirement.csv`

Technology-specific build floors and ceilings. One row per constraint, matched by resource name patterns.

---

## Legacy format (old GenX)

If your GenX version uses the older single-file format (`Generators_data.csv`, `Load_data.csv`), set:

```yaml
old_genx_format: true
```

PowerGenome then writes `Generators_data.csv` combining all resource types and
`Load_data.csv` rather than `Demand_data.csv`. These files are directly in the
period folder, not the v6 subfolders.

---

## Macro `simpleCSVinputs` format

When enabled (see [Macro Output Settings](../reference/settings/macro-output.md)), PowerGenome additionally writes
a Macro case in the `simpleCSVinputs` format — the same structure used by Macro's
[`multisector_3zone_simpleCSVinputs` example](https://github.com/macroenergy/MacroEnergyExamples.jl/tree/main/examples/multisector_3zone_simpleCSVinputs).
Only the simpleCSV format is produced (not Macro's JSON asset format). The semantic mapping of generators,
transmission, demand, fuel supply, and CO2 caps follows the [GenX-to-Macro converter](https://github.com/EmilDimanchev/GenX_to_Macro)
(`lb/multistage` branch for multi-period runs).

With `macro_output: true`, both formats are written in the same `run_powergenome`
call unless GenX output is disabled. Because all intermediate data processing
(storage clustering, market data) is shared, writing GenX and Macro inputs together in one run is
**faster than running PowerGenome twice** (once per model). Use `--no-genx` / `genx_output: false` to produce Macro
inputs only.

### Folder structure

The Macro case is written at `results/<case_id>/` with scenario definitions, or
at `results/` without them. It contains one stage per planning period:

```
<macro case folder>/
├── system_data.json          # Top-level manifest: commodities, locations, settings, assets, time_data, nodes
├── settings/
│   ├── case_settings.json    # PeriodLengths, DiscountRate, SolutionAlgorithm, ...
│   └── macro_settings.json   # ConstraintScaling, WriteSubcommodities, AutoCreateNodes, ...
├── system/
│   ├── commodities.json      # Electricity, NaturalGas, CO2, Uranium, Coal, ...
│   ├── locations.json        # Model region names
│   ├── nodes_1.json          # Stage 1 demand nodes, fuel supply nodes, CO2 sinks
│   ├── time_data_1.json      # Stage 1 time settings
│   ├── demand_1.csv
│   ├── availability_1.csv
│   └── fuel_prices_1.csv
└── assets/
    └── assets_1/
        ├── <base fuel>_power.csv
        ├── vre.csv
        ├── electricity_stor.csv
        ├── hydropower.csv
        ├── mustrun.csv
        └── powerlines.csv
```

Later stages use the same names with their stage number. Individual files depend
on the data and options used.

### Resource mapping highlights

| GenX concept | Macro output |
|---|---|
| `Existing_Cap_MW` | `existing_capacity` / `discharge_existing_capacity` (thermal, VRE, storage, hydro) |
| `Heat_Rate_MMBTU_per_MWh` | `fuel_consumption` = heat rate × 0.29307107 MWh/MMBtu |
| CO2 emissions | `emission_rate` = fuel CO2 content / 0.29307107, plus a `co2_sink` node |
| `Eff_Up` / `Eff_Down` | `charge_efficiency` / `discharge_efficiency` (storage, hydro) |
| `New_Build` / `Can_Retire` | `can_expand` / `can_retire` |
| `Line_Max_Flow_MW` + reinforcement | `existing_capacity` + `max_capacity` on a `TransmissionLink` |
| VRE variability | Resource columns in `system/availability_<stage>.csv` |
| Demand data | `Demand_MW_z<zone>` columns in `system/demand_<stage>.csv` and electricity nodes |
| Fuel prices | per-(fuel, region) supply node + `system/fuel_prices_<stage>.csv` |
| CO2 caps | CO2 sink node with `CO2CapConstraint` |

The Macro writer is electricity-centric; it does not yet emit cross-sector
hydrogen, liquid-fuel, or demand-side assets.

---

## Related documentation

- [Resource Tags Settings](../reference/settings/resource-tags.md): How `THERM`, `VRE`, `STOR` etc. are assigned
- [Transmission Settings](../reference/settings/transmission.md): Network parameters
- [Macro Output Settings](../reference/settings/macro-output.md): How to enable GenX and/or Macro output
- [Data Pipeline Flow](pipeline.md): How outputs are assembled from pipeline stages
