# Output File Format

PowerGenome writes input files for the [GenX](https://github.com/GenXProject/GenX.jl) capacity expansion model and,
optionally, for [MacroEnergy.jl](https://github.com/macroenergy/MacroEnergy.jl) (Macro) in its `simpleCSVinputs`
format. This page describes the folder structure, file names, and the key content of each output.

For the full GenX documentation on input file schemas, see the [GenX documentation](https://genxproject.github.io/GenX.jl/stable/).
For Macro, see the [MacroEnergy Examples](https://github.com/macroenergy/MacroEnergyExamples.jl) repository.

---

## Folder structure

Each (case_id, planning period) pair produces a folder:

```
results/
└── <case_id>/
    └── Inputs/
        └── Inputs_p<N>/           ← one per planning period
            ├── system/
            │   ├── Demand_data.csv
            │   ├── Fuels_data.csv
            │   ├── Generators_variability.csv
            │   ├── Network.csv
            │   ├── Period_map.csv
            │   └── Representative_Period.csv
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

Hourly electricity demand for each model zone. One row per hour (or representative period timestep), with columns:

| Column | Description |
|---|---|
| `Rep_Periods` | Number of representative periods |
| `Timesteps_per_Rep_Period` | Hours per representative period |
| `Sub_Weights` | Weight (hours) each period represents in the full year |
| `Time_Index` | Sequential hour index |
| `Demand_MW_z<N>` | Load in MW for zone N |

When time domain reduction is off, there is one period covering all 8760 hours with `Sub_Weights = 8760`.

### `Fuels_data.csv`

Fuel prices by time step. One row per hour with one column per fuel type, plus a `Time_Index` column.

### `Generators_variability.csv`

Normalized capacity factor (0–1) for each variable renewable resource, by hour. One column per VRE resource, one row per timestep.

### `Network.csv`

Transmission network topology. One row per network line (inter-regional connection):

| Column | Description |
|---|---|
| `Network_Lines` | Line identifier |
| `z<N>` | Zone membership (+1 sending, −1 receiving, 0 not in line) |
| `Line_Max_Flow_MW` | Maximum power flow in MW |
| `Line_Loss_Percentage` | Fraction of power lost in transmission |
| `Line_Max_Reinforcement_MW` | Maximum expansion allowed (if `tx_expansion_per_mw` is set) |
| `Line_Reinforcement_Cost_per_MWyr` | Annualized cost to expand line by 1 MW |

### `Period_map.csv` and `Representative_Period.csv`

Present only when time domain reduction is active. Map representative periods back to the original 8760-hour year for chronological analysis.

---

## Resource files

Resources are split into typed CSVs based on their primary model tag:

| File | Resources included |
|---|---|
| `Thermal.csv` | `THERM = 1` |
| `Vre.csv` | `VRE = 1` |
| `Storage.csv` | `STOR > 0` |
| `Hydro.csv` | `HYDRO = 1` |
| `Flex_demand.csv` | `FLEX = 1` |
| `Must_run.csv` | `MUST_RUN = 1` |
| `Long_duration_storage.csv` | `LDS = 1` |
| `Electrolyzer.csv` | `ELECTROLYZER = 1` |

All resource files share a common set of columns, with additional columns present depending on the resource type.

### Key shared columns

| Column | Description |
|---|---|
| `Resource` | Unique resource name |
| `Zone` | Zone number (integer) |
| `Existing_Cap_MW` | Installed capacity at start of planning period |
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

Defines Renewable Portfolio Standard (RPS) or Clean Energy Standard (CES) constraints. One row per constraint:

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

PowerGenome then writes `Generators_data.csv` combining all resource types, and `Load_data.csv` rather than `Demand_data.csv`.

---

## Macro `simpleCSVinputs` format

When enabled (see [Macro Output Settings](../reference/settings/macro-output.md)), PowerGenome additionally writes
a Macro case in the `simpleCSVinputs` format — the same structure used by Macro's
[`multisector_3zone_simpleCSVinputs` example](https://github.com/macroenergy/MacroEnergyExamples.jl/tree/main/examples/multisector_3zone_simpleCSVinputs).
Only the simpleCSV format is produced (not Macro's JSON asset format). The semantic mapping of generators,
transmission, demand, fuel supply, and CO2 caps follows the [GenX-to-Macro converter](https://github.com/EmilDimanchev/GenX_to_Macro)
(`lb/multistage` branch for multi-period runs).

Both formats are written in the same `run_powergenome` call by default. Because all intermediate data processing
(storage clustering, time reduction, market data) is shared, writing GenX and Macro inputs together in one run is
**faster than running PowerGenome twice** (once per model). Use `--no-genx` / `genx_output: false` to produce Macro
inputs only.

### Folder structure

The Macro case is written to a `macro_out/`-style folder (per planning period when multi-period), containing
`system_data.json` plus three input subfolders:

```
<macro case folder>/
├── system_data.json          # Top-level manifest: commodities, locations, settings, assets, time_data, nodes
├── settings/
│   ├── case_settings.json    # PeriodLengths, DiscountRate, SolutionAlgorithm, ...
│   └── macro_settings.json   # ConstraintScaling, WriteSubcommodities, AutoCreateNodes, ...
├── system/
│   ├── commodities.json      # Electricity, NaturalGas, CO2, Uranium, Coal, ...
│   ├── locations.json        # Model region names
│   ├── nodes.json            # Electricity demand nodes, fuel supply nodes, CO2 sinks, hydro source
│   ├── time_data.json        # HoursPerSubperiod, HoursPerTimeStep, NumberOfSubperiods, ...
│   ├── Period_map.csv        # Sub-period mapping (only when time reduction is active)
│   ├── demand.csv            # Time_Index, Demand_MW_z<zone>...
│   ├── availability.csv      # Time_Index, <resource>...  (non-constant VRE/hydro/must-run profiles)
│   └── fuel_prices.csv       # Time_Index, <fuel>_<region>... price timeseries per fuel node
└── assets/
    ├── <base fuel>_power.csv # Type=ThermalPower (one file per base fuel)
    ├── vre.csv               # Type=VRE
    ├── electricity_stor.csv  # Type=Battery
    ├── hydropower.csv        # Type=HydroRes
    ├── mustrun.csv           # Type=MustRun
    └── powerlines.csv        # Type=TransmissionLink
```

### Resource mapping highlights

| GenX concept | Macro output |
|---|---|
| `Existing_Cap_MW` | `existing_capacity` / `discharge_existing_capacity` (thermal, VRE, storage, hydro) |
| `Heat_Rate_MMBTU_per_MWh` | `fuel_consumption` = heat rate × 0.29307107 MWh/MMBtu |
| CO2 emissions | `emission_rate` = fuel CO2 content / 0.29307107, plus a `co2_sink` node |
| `Eff_Up` / `Eff_Down` | `charge_efficiency` / `discharge_efficiency` (storage, hydro) |
| `New_Build` / `Can_Retire` | `can_expand` / `can_retire` |
| `Line_Max_Flow_MW` + reinforcement | `existing_capacity` + `max_capacity` on a `TransmissionLink` |
| VRE variability | `availability` column in `system/availability.csv` |
| Demand data | `Demand_MW_z<zone>` columns in `system/demand.csv` + electricity nodes in `nodes.json` |
| Fuel prices | per-(fuel, region) supply node + `system/fuel_prices.csv` |
| CO2 caps | CO2 sink node with `CO2CapConstraint` |

Cross-sector assets (hydrogen, liquid fuels, CCS detail, demand-side) are not yet emitted; PowerGenome is
electricity-centric and the Macro writer is scoped to the electricity system in v1.

---

## Related documentation

- [Resource Tags Settings](../reference/settings/resource-tags.md): How `THERM`, `VRE`, `STOR` etc. are assigned
- [Time Reduction Settings](../reference/settings/time-reduction.md): Controls `Rep_Periods` and `Sub_Weights`
- [Transmission Settings](../reference/settings/transmission.md): Network parameters
- [Macro Output Settings](../reference/settings/macro-output.md): How to enable GenX and/or Macro output
- [Data Pipeline Flow](pipeline.md): How outputs are assembled from pipeline stages
