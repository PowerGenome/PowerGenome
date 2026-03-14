# Multi-Scenario Architecture

This page explains *how* PowerGenome's scenario management system works internally. For step-by-step instructions on setting up scenarios, see [Run Multi-Scenario Studies](../how-to/run-scenarios.md).

---

## The problem scenarios solve

A single capacity expansion study typically involves multiple assumptions about uncertain inputs — technology costs, fuel prices, demand growth, policy stringency. Rather than running PowerGenome once per assumption set by hand, the scenario system lets you define a matrix of cases and process them in a single invocation.

---

## Two types of variation

PowerGenome distinguishes between two kinds of parameter variation:

### 1. Planning-year variation (year-keyed values)

When a parameter naturally varies over time — e.g., capital costs fall as technology matures — you encode this directly in the settings using a year-keyed dictionary:

```yaml
atb_cost_case:
  2030: Moderate
  2040: Advanced
```

No scenario definitions CSV is needed. PowerGenome resolves these automatically for each planning year. See [Year-Keyed Values](../reference/settings/year-keyed-values.md) for details.

### 2. Case variation (scenario management)

When you want to compare *alternative futures* — e.g., low vs. high gas price, with vs. without a carbon tax — you define cases in a CSV file and describe parameter swaps in `settings_management`:

```yaml
# settings_management.yml
settings_management:
  all_years:
    gas_price:
      low:
        fuel_scenarios:
          naturalgas: low_price
      high:
        fuel_scenarios:
          naturalgas: high_price
```

```csv
# scenario_definitions.csv
case_id,year,gas_price
low_gas,2030,low
high_gas,2030,high
```

---

## The scenario matrix

The scenario definitions CSV defines the full set of runs. Each row is one (case, year) pair:

| Column | Description |
|---|---|
| `case_id` | Unique identifier for a scenario run |
| `year` | Planning year — must match a period end year from `model_periods` |
| *(other columns)* | Dimension names that correspond to `settings_management` keys |

Every `case_id` must have one row per model year. For a model with 3 years and 4 cases, the CSV has 12 rows.

---

## How settings are constructed for each case

For each (case_id, year) pair PowerGenome:

1. **Starts from the base settings** loaded from the settings folder
2. **Applies year-keyed resolution** — any parameter that is a year-keyed dict is resolved to its value for this planning year
3. **Applies `all_years` swaps** — entries in `settings_management.all_years` are applied to every year
4. **Applies year-specific swaps** — entries in `settings_management.<year>` are then applied
5. **Within each year block**, dimension columns from the scenario CSV row are matched to sub-keys, and the corresponding parameter dict is deep-merged into settings

The result is an independent, fully resolved settings dictionary for that specific run. Each case's settings do not affect any other case.

### Deep merge behavior

Parameter swap dictionaries are deep-merged. This means nested structures like `resource_modifiers` or `new_gen_not_available` are updated key-by-key rather than replaced wholesale:

```yaml
# Base settings
resource_modifiers:
  UtilityPV_Class1_Moderate:
    capex_mw: 1.0

# Scenario swap: solar_cost = low
resource_modifiers:
  UtilityPV_Class1_Moderate:
    capex_mw: 0.8   # overrides only capex_mw; other keys are preserved
```

However, **lists are replaced entirely** — not merged. If a swap sets `new_resources` to a new list, the base `new_resources` list is discarded.

---

## The `all_cases` selector

Within a year block, `all_cases` applies to every run regardless of case dimension values:

```yaml
settings_management:
  2040:
    all_cases:
      carbon_tax: 75  # Applied to every 2040 run
    tech_cost:
      low:
        atb_cost_case: Advanced
```

This is useful for policy parameters that change year-over-year but don't vary across scenarios.

---

## Output folder structure

Each (case_id, year) pair writes to its own folder:

```
results/
└── <case_id>/
    └── Inputs/
        └── Inputs_p<period_number>/
            ├── system/
            │   ├── Demand_data.csv
            │   ├── Fuels_data.csv
            │   └── Network.csv
            ├── resources/
            │   └── ...
            ├── policies/
            │   └── ...
            └── powergenome_case_settings.yml
```

`Inputs_p1` corresponds to the first planning period, `Inputs_p2` to the second, and so on. The `powergenome_case_settings.yml` file records the exact resolved settings used for that run — useful for debugging or reproducing results.

---

## Running without scenario definitions

If `scenario_definitions_fn` is absent from settings, PowerGenome creates one implicit "case" called `Inputs` for each period end year in `model_periods`. This is equivalent to a scenario CSV with one row per year:

```csv
case_id,year
Inputs,2030
Inputs,2040
```

Year-keyed values in settings are still resolved per year.

---

## Related documentation

- [Run Multi-Scenario Studies](../how-to/run-scenarios.md): Step-by-step instructions
- [Scenario Management Settings](../reference/settings/scenario-management.md): Complete parameter reference
- [Year-Keyed Values](../reference/settings/year-keyed-values.md): Per-year parameter resolution
- [Data Pipeline Flow](pipeline.md): How scenario settings flow through the pipeline
