# Input Table Schemas

PowerGenome reads source data from files (CSV or Parquet) or a database. The DataManager maps each settings key to a standardized internal table name. This page describes the required columns and data types for each table.

!!! tip "Finding your data files"
    Set `data_location` in your settings YAML to the folder containing these files. See [Configure Data Tables](../../how-to/configure-data-tables.md) for the full configuration syntax, including column projection and row filtering.

---

## Table reference

| Settings key | Internal name | Typical file |
| --- | --- | --- |
| `generation_table` | `generation` | `generators.csv` |
| `plant_region_table` | `plant_region` | `plant_region_map.csv` |
| `resource_cost_table` | `resource_cost` | `technology_costs.csv` |
| `resource_heat_rate_table` | `resource_heat_rate` | `technology_heat_rates.csv` |
| `operational_constraints_table` | `operational_constraints` | `operational_constraints.csv` |
| `transmission_constraints_table` | `transmission_constraints` | `transmission.csv` |
| `transmission_cost_table` | `transmission_cost` | `network_costs.csv` |
| `fuel_price_table` | `fuel_price` | `fuel_prices.csv` |
| `dollar_year_table` | `dollar_year` | `cpi_data.csv` |
| `regional_cost_factor_table` | `regional_cost_factor` | `regional_cost_multipliers.csv` |
| `demand_table` | `demand` | `load_curves.csv` |
| `supplemental_demand_table` | `supplemental_demand` | `supplemental_demand.csv` |
| `emission_policies_table` | `emission_policies` | `emission_policies.csv` |
| `demand_segments_table` | `demand_segments` | `demand_segments.csv` |
| `distributed_capacity_table` | `distributed_capacity` | `distributed_capacity.csv` |
| `distributed_profiles_table` | `distributed_profiles` | `distributed_profiles.csv` |

`distributed_profile_table` is also accepted for `distributed_profiles`.
See [Configure Data Tables](../../how-to/configure-data-tables.md) for aliases
and configuration options.

---

## Existing generators (`generation`)

Contains one row per generator unit. Matched to model regions via the `plant_region` table.

| Column | Type | Required | Notes |
| --- | --- | --- | --- |
| `technology` | string | Yes | EIA technology name (e.g. `Natural Gas Fired Combustion Turbine`) |
| `plant_id` | integer | Yes | EIA plant ID |
| `generator_id` | string | Recommended | Identifies the unit in cluster-assignment outputs |
| `capacity_mw` | float | Yes | Nameplate capacity in MW |
| `capacity_mwh` | float | Yes | Storage energy capacity in MWh; values may be blank for non-storage |
| `operating_year` | integer | Yes | Year plant came online; used to calculate age |
| `retirement_year` | integer | No | Year the unit leaves service; controls capacity listed in `Min_Retired_*_Cap_MW` and if a unit contributes to the characteristics (heat rate, unit size, etc.) of a resource. A blank value (or a missing column) means the unit has no planned retirement, so it operates through the whole horizon |
| `historical_capacity_factor` | float | No | Source-data attribute |
| `heat_rate_mmbtu_mwh` | float | Yes | Average heat rate; supply appropriate values for thermal generators |
| `vom_per_mwh` | float | Yes | Variable O&M cost ($/MWh) |
| `fom_per_mwyr` | float | Yes | Fixed O&M cost ($/MW-year) |
| `energy_source_code` | string | No | Source fuel code (e.g. `NG`, `BIT`, `SUN`) |
| `minimum_load_mw` | float | No | Source-data attribute; operational constraints set `Min_Power` |

---

## Plant–region map (`plant_region`)

Maps EIA plant IDs to model regions. Only plants listed here will be included in the model.

| Column | Type | Required | Notes |
| --- | --- | --- | --- |
| `plant_id` | integer | Yes | EIA plant ID; must match `generation` table |
| `region` | string | Yes | Base region or unaggregated model region name |

---

## New-build technology costs (`resource_cost`)

Long-format table of cost and financial parameters for new-build resources. Each parameter is its own row.

| Column | Type | Required | Notes |
| --- | --- | --- | --- |
| `technology` | string | Yes | ATB technology name |
| `tech_detail` | string | Yes | Technology detail (e.g. `Lithium Ion`) |
| `cost_case` | string | Yes | ATB cost case (e.g. `Advanced`, `Moderate`, `Conservative`) |
| `financial_case` | string | Yes | Financial assumption set (e.g. `R&D`, `Market`) |
| `parameter` | string | Yes | Cost parameter name (e.g. `capex_mw`, `fixed_o_m_mw`, `wacc_real`) |
| `parameter_value` | float | Yes | Value for this parameter |
| `units` | string | No | Units string for reference |
| `data_year` | integer | Yes | ATB vintage year |
| `basis_year` | integer | Yes | Technology deployment year costs apply to |
| `cap_recovery_years` | integer | No | Source attribute; the new-build workflow uses the corresponding settings |
| `dollar_year` | integer | Yes | Dollar-year for cost figures |

---

## Technology heat rates (`resource_heat_rate`)

Heat rate by technology used for new-build thermal resources.

| Column | Type | Required | Notes |
| --- | --- | --- | --- |
| `technology` | string | Yes | ATB technology name |
| `tech_detail` | string | Yes | ATB detail variant |
| `cost_case` | string | Yes | ATB cost case |
| `basis_year` | integer | Yes | Deployment year |
| `data_year` | integer | Yes | ATB vintage year |
| `heat_rate` | float | Yes | Heat rate in MMBtu/MWh |

---

## Operational constraints (`operational_constraints`)

Per-technology or per-resource overrides for GenX operational parameters.
Matching values are applied to generator rows before output formatting.

| Column | Type | Required | Notes |
| --- | --- | --- | --- |
| `Resource` | string | Yes | Resource name or technology name pattern |
| `region` | string | No | Model region or `all` for all regions; omitted means all regions |
| `Min_Power` | float | No | Minimum power output fraction |
| `Self_Disch` | float | No | Storage self-discharge rate |
| `Eff_Up` | float | No | Charging efficiency |
| `Eff_Down` | float | No | Discharging efficiency |
| `Min_Duration` | float | No | Minimum storage duration (hours) |
| `Max_Duration` | float | No | Maximum storage duration (hours) |
| `Ramp_Up_Percentage` | float | No | Maximum ramp-up rate (fraction of capacity per hour) |
| `Ramp_Dn_Percentage` | float | No | Maximum ramp-down rate |
| `Up_Time` | integer | No | Minimum up-time (hours) |
| `Down_Time` | integer | No | Minimum down-time (hours) |

Non-null parameter values are applied to matching rows. The v6 output writer
then splits those rows into resource files and may omit empty columns.

---

## Transmission constraints (`transmission_constraints`)

Existing inter-regional transfer capacities. Used to build `Network.csv`.

| Column | Type | Required | Notes |
| --- | --- | --- | --- |
| `region_from` | string | Yes | Source IPM or base region |
| `region_to` | string | Yes | Destination IPM or base region |
| `firm_ttc_mw` | float | Yes | Firm total transfer capacity in MW |
| `notes` | string | No | Optional description |

---

## Transmission costs (`transmission_cost`)

Capital costs for new inter-regional transmission lines.

| Column | Type | Required | Notes |
| --- | --- | --- | --- |
| `start_region` | string | Yes | Model region at one end of the line |
| `dest_region` | string | Yes | Model region at the other end |
| `total_interconnect_cost_mw` | float | Yes | Total build cost ($/MW) |
| `total_interconnect_annuity_mw` | float | Yes | Annualized build cost ($/MW-year) |
| `total_line_loss_frac` | float | No | Fractional line loss |
| `start_id` | integer | No | Source node ID |
| `dest_id` | integer | No | Destination node ID |
| `dollar_year` | integer | Yes | Dollar-year for cost figures |
| `mw-km_per_mw` | float | No | Route length in MW-km per MW of capacity |
| `notes` | string | No | Optional description |

---

## Fuel prices (`fuel_price`)

Time series of fuel prices by fuel type and region.

| Column | Type | Required | Notes |
| --- | --- | --- | --- |
| `year` | integer | Yes | Planning year |
| `fuel` | string | Yes | Fuel name; must match `tech_fuel_map` values in settings |
| `region` | string | Yes | Fuel price region |
| `price` | float | Yes | Fuel price (typically $/MMBtu) |
| `scenario` | string | No | Fuel price scenario; can be filtered in the table configuration |
| `data_year` | integer | Yes | Data vintage; include even for one vintage |
| `dollar_year` | integer | For inflation adjustment | Dollar-year for price figures |

---

## Dollar-year / CPI (`dollar_year`)

Conversion factors for adjusting costs between dollar years.

| Column | Type | Required | Notes |
| --- | --- | --- | --- |
| `year` | integer | Yes | Calendar year |
| `period` | string | No | Period label (legacy column, may be omitted) |
| `value` | float | Yes | CPI index value |

---

## Regional cost multipliers (`regional_cost_factor`)

Adjustments to capital costs by technology and region.

| Column | Type | Required | Notes |
| --- | --- | --- | --- |
| `technology` | string | Yes | ATB technology name |
| `region` | string | Yes | Model region |
| `value` | float | Yes | Multiplier (1.0 = no adjustment) |

---

## Demand / load profiles (`demand`)

Hourly load time series. One row per hour per region.

| Column | Type | Required | Notes |
| --- | --- | --- | --- |
| `time_index` | integer | Yes | Hour index within the source profile |
| `region` | string | Yes | Base region or unaggregated model region |
| `load_mw` | float | Yes | Load in MW |
| `year` | integer | Yes | Planning year |
| `weather_year` | integer | No | Meteorological year for the profile |

---

## Supplemental demand (`supplemental_demand`)

Adds hourly load before region aggregation. It uses `region`, `time_index`, and
`load_mw`; optional `year` and `weather_year` select the applicable planning and
weather years. See [Add Supplemental Demand](../../how-to/add-supplemental-demand.md)
for coverage and aggregation rules.

---

## Emission policies and demand segments

`emission_policies` uses `year`, `region`, optional `case_id` and `copy_case_id`,
and policy columns such as `CO_2_Cap_Zone_1` and `CO_2_Max_Mtons_1`. If
`case_id` is omitted, the current case is used; an `all` case can provide
fallback values.

`demand_segments` contains GenX demand-curtailment inputs, such as `Voll`,
`Demand_Segment`, `Cost_of_Demand_Curtailment_per_MW`, and
`Max_Demand_Curtailment`. These columns are included in the demand output.

---

## Distributed capacity (`distributed_capacity`)

Installed distributed generation capacity by region and year.

| Column | Type | Required | Notes |
| --- | --- | --- | --- |
| `region` | string | Yes | Base region or unaggregated model region |
| `year` | integer | Yes | Planning year |
| `capacity_mw` | float | Yes | Total installed DG capacity in MW |

---

## Distributed generation profiles (`distributed_profiles`)

Normalized capacity factor profiles for distributed generation.

| Column | Type | Required | Notes |
| --- | --- | --- | --- |
| `time_index` | integer | Yes | Hour index within the source profile |
| `region` | string | Yes | Base region or unaggregated model region |
| `weather_year` | integer | Yes | Meteorological year for the profile |
| `value` | float | Yes | Capacity factor (0–1) |

When regions are aggregated, provide component-region capacities and profiles.
See [Distributed Generation](../../explanation/distributed-generation.md) for
interpolation and aggregation behavior.
