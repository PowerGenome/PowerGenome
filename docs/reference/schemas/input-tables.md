# Input Table Schemas

PowerGenome reads source data from files (CSV or Parquet) or a database. The DataManager maps each settings key to a standardized internal table name. This page describes the required columns and data types for each table.

!!! tip "Finding your data files"
    Set `data_location` in your settings YAML to the folder containing these files. See [Configure Data Tables](../../how-to/configure-data-tables.md) for the full configuration syntax, including column projection and row filtering.

---

## Table reference

| Settings key | Internal name | Typical file |
|---|---|---|
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
| `distributed_capacity_table` | `distributed_capacity` | `distributed_capacity.csv` |
| `distributed_profiles_table` | `distributed_profiles` | `distributed_profiles.csv` |

---

## Existing generators (`generation`)

Contains one row per generator unit. Matched to model regions via the `plant_region` table.

| Column | Type | Required | Notes |
|---|---|---|---|
| `technology` | string | Yes | EIA technology name (e.g. `Natural Gas Fired Combustion Turbine`) |
| `plant_id` | integer | Yes | EIA plant ID |
| `generator_id` | string | Yes | EIA generator ID |
| `capacity_mw` | float | Yes | Nameplate capacity in MW |
| `capacity_mwh` | float | No | Storage energy capacity in MWh; leave blank for non-storage |
| `operating_year` | integer | Yes | Year plant came online; used to calculate age |
| `retirement_year` | integer | No | Scheduled retirement year; takes precedence over age-based retirement |
| `historical_capacity_factor` | float | No | Used to impute missing heat-rate data |
| `heat_rate_mmbtu_mwh` | float | No | Average heat rate; required for thermal generators |
| `vom_per_mwh` | float | No | Variable O&M cost ($/MWh) |
| `fom_per_mwyr` | float | No | Fixed O&M cost ($/MW-year) |
| `energy_source_code` | string | Yes | EIA fuel code (e.g. `NG`, `BIT`, `SUN`) |
| `minimum_load_mw` | float | No | Minimum stable load in MW |

---

## Plant–region map (`plant_region`)

Maps EIA plant IDs to model regions. Only plants listed here will be included in the model.

| Column | Type | Required | Notes |
|---|---|---|---|
| `plant_id` | integer | Yes | EIA plant ID; must match `generation` table |
| `region` | string | Yes | Model region name |

---

## New-build technology costs (`resource_cost`)

Long-format table of cost and financial parameters for new-build resources. Each parameter is its own row.

| Column | Type | Required | Notes |
|---|---|---|---|
| `technology` | string | Yes | ATB technology name |
| `tech_detail` | string | Yes | ATB detail variant (e.g. `Moderate`, `Conservative`) |
| `cost_case` | string | Yes | ATB cost case (e.g. `R&D`, `Market`) |
| `financial_case` | string | Yes | Financial assumption set (e.g. `R&D`, `Market`) |
| `parameter` | string | Yes | Cost parameter name (e.g. `CAPEX`, `Fixed_OM_Cost_per_kWyr`) |
| `parameter_value` | float | Yes | Value for this parameter |
| `units` | string | No | Units string for reference |
| `data_year` | integer | Yes | ATB vintage year |
| `basis_year` | integer | Yes | Technology deployment year costs apply to |
| `cap_recovery_years` | integer | No | Assumed capital recovery period |
| `dollar_year` | integer | Yes | Dollar-year for cost figures |

---

## Technology heat rates (`resource_heat_rate`)

Heat rate by technology used for new-build thermal resources.

| Column | Type | Required | Notes |
|---|---|---|---|
| `technology` | string | Yes | ATB technology name |
| `tech_detail` | string | Yes | ATB detail variant |
| `cost_case` | string | Yes | ATB cost case |
| `basis_year` | integer | Yes | Deployment year |
| `data_year` | integer | Yes | ATB vintage year |
| `heat_rate` | float | Yes | Heat rate in MMBtu/MWh |

---

## Operational constraints (`operational_constraints`)

Per-technology or per-resource overrides for GenX operational parameters. Values here are written directly to `Generators_data.csv`.

| Column | Type | Required | Notes |
|---|---|---|---|
| `Resource` | string | Yes | Resource name or technology name pattern; `all` matches any resource |
| `region` | string | Yes | Model region or `all` for all regions |
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

Any column that matches a `Generators_data.csv` column name is forwarded to that output file.

---

## Transmission constraints (`transmission_constraints`)

Existing inter-regional transfer capacities. Used to build `Network.csv`.

| Column | Type | Required | Notes |
|---|---|---|---|
| `region_from` | string | Yes | Source IPM or base region |
| `region_to` | string | Yes | Destination IPM or base region |
| `firm_ttc_mw` | float | Yes | Firm total transfer capacity in MW |
| `notes` | string | No | Optional description |

---

## Transmission costs (`transmission_cost`)

Capital costs for new inter-regional transmission lines.

| Column | Type | Required | Notes |
|---|---|---|---|
| `start_region` | string | Yes | Model region at one end of the line |
| `dest_region` | string | Yes | Model region at the other end |
| `interconnect_cost_mw` | float | Yes | Total build cost ($/MW) |
| `interconnect_annuity_mw` | float | Yes | Annualized build cost ($/MW-year) |
| `line_loss_frac` | float | Yes | Fractional line loss |
| `start_id` | integer | No | Source node ID |
| `dest_id` | integer | No | Destination node ID |
| `dollar_year` | integer | Yes | Dollar-year for cost figures |
| `mw-km_per_mw` | float | No | Route length in MW-km per MW of capacity |
| `notes` | string | No | Optional description |

---

## Fuel prices (`fuel_price`)

Time series of fuel prices by fuel type and region.

| Column | Type | Required | Notes |
|---|---|---|---|
| `year` | integer | Yes | Planning year |
| `fuel` | string | Yes | Fuel name; must match `tech_fuel_map` values in settings |
| `region` | string | Yes | Model region |
| `price` | float | Yes | Fuel price (typically $/MMBtu) |
| `scenario` | string | No | AEO scenario name; used with `fuel_price_scenario` setting |
| `data_year` | integer | No | AEO data vintage |
| `dollar_year` | integer | Yes | Dollar-year for price figures |

---

## Dollar-year / CPI (`dollar_year`)

Conversion factors for adjusting costs between dollar years.

| Column | Type | Required | Notes |
|---|---|---|---|
| `year` | integer | Yes | Calendar year |
| `period` | string | No | Period label (legacy column, may be omitted) |
| `value` | float | Yes | CPI index value |

---

## Regional cost multipliers (`regional_cost_factor`)

Adjustments to capital costs by technology and region.

| Column | Type | Required | Notes |
|---|---|---|---|
| `technology` | string | Yes | ATB technology name |
| `region` | string | Yes | Model region |
| `value` | float | Yes | Multiplier (1.0 = no adjustment) |

---

## Demand / load profiles (`demand`)

Hourly load time series. One row per hour per region.

| Column | Type | Required | Notes |
|---|---|---|---|
| `time_index` | integer | Yes | Hour index within the year (0–8759) |
| `region` | string | Yes | Model region |
| `load_mw` | float | Yes | Load in MW |
| `year` | integer | Yes | Planning year |
| `weather_year` | integer | No | Meteorological year for the profile |

---

## Distributed capacity (`distributed_capacity`)

Installed distributed generation capacity by region and year.

| Column | Type | Required | Notes |
|---|---|---|---|
| `region` | string | Yes | Model region |
| `year` | integer | Yes | Planning year |
| `capacity_mw` | float | Yes | Total installed DG capacity in MW |

---

## Distributed generation profiles (`distributed_profiles`)

Normalized capacity factor profiles for distributed generation.

| Column | Type | Required | Notes |
|---|---|---|---|
| `time_index` | integer | Yes | Hour index within the year (0–8759) |
| `region` | string | Yes | Model region |
| `weather_year` | integer | No | Meteorological year for the profile |
| `value` | float | Yes | Capacity factor (0–1) |
