# Scenario Management Settings

This reference defines scenario selectors, file paths, and supported swap shapes.
For a complete runnable workflow, see [Run Multiple Scenarios](../../how-to/run-scenarios.md).
For merge rationale and execution order, see
[Multi-Scenario Architecture](../../explanation/scenarios.md).

## Scenario Definition File

### `scenario_definitions_fn`

**Type:** string filename or path, relative to `input_folder`.
**Required:** no; enables the explicit scenario CSV when set.

```yaml
scenario_definitions_fn: scenario_inputs.csv
```

| CSV column | Requirement |
| --- | --- |
| `case_id` | Mandatory case identifier; repeat it across that case’s planning years. |
| `year` | Mandatory integer end year from `model_year` or `model_periods`. |
| Other columns | Dimension names; cells select levels under `settings_management`. |

Each case/year pair must be unique. Include each planning year once per complete
case and order years chronologically within cases: row order sets period numbers.
The CSV lists only requested combinations; PowerGenome does not expand a full
factorial design. The [how-to](../../how-to/run-scenarios.md#2-create-scenario-definitions)
contains a four-row example with two cases and two years.

## Settings Management

### `settings_management`

**Type:** mapping of year selector → dimension → level → settings to modify.
**Default:** no swaps. A CSV with only `case_id,year` is valid without this mapping.

```yaml
settings_management:
  all_years:
    carbon_policy:
      unchanged: {}
      tax_50:
        carbon_tax: 50
  2040:
    all_cases:
      tx_expansion_per_period: 0.5
```

| Selector | Meaning |
| --- | --- |
| Integer year, such as `2040` | Apply to rows for that planning year. |
| `all_years` | Apply to every CSV row before the specific-year block. |
| `all_cases` inside either block | Direct setting/value pairs applied before that block’s dimensions. |
| Dimension and level | Match a CSV column and cell value; names and value types must agree. |
| `{}` or `null` (`~`) as a selected level | Recognized no-op; keep the base settings unchanged. |

Swaps recursively merge dictionaries but replace scalars and entire lists.
Within each block, two selected dimensions modifying the same leaf raise an
error. The conflict check excludes `all_cases` and resets between `all_years`
and a specific year. Do not split this top-level mapping across files: loading
settings directories uses shallow updates with no guaranteed file order.
See [merge precedence](../../explanation/scenarios.md#how-settings-are-constructed-for-each-case).

## Multi-Period Scenarios

Keep `model_year` and `model_first_planning_year` lists aligned and chronological,
or use start/end pairs in `model_periods` (see [Model Definition](model-definition.md)).
A year-specific swap does not persist into later periods.

A selected swap can contain year-keyed values, avoiding repeated year blocks.
If the base value is a scalar, first change it to a year-keyed mapping in its
original settings file: a dictionary swap cannot merge into an existing scalar.
For example, use this base value and scenario mapping (each defined only once):

```yaml
carbon_tax: {2030: 0, 2040: 0}
settings_management:
  all_years:
    carbon_policy:
      rising_tax:
        carbon_tax:
          2030: 25
          2040: 50
  2040:
    carbon_policy:
      rising_tax:
        carbon_tax: 75
```

For a CSV selecting `rising_tax` in both years, the resolved tax is 25 in 2030
and 75 in 2040. `all_years` is processed first; the 2040 scalar replaces its
whole year-keyed dictionary before year resolution. Year-keyed mappings use
integer years and an optional `default`, not `all_years`. Cover every year being
resolved or provide `default`; see [Year-Keyed Values](year-keyed-values.md).

## Parameter Swap Examples

These are alternative fragments to put under a selected level, not a second
complete settings file. Use values supported by your configured input tables.
The example names below match the bundled test system where applicable.

### Technology Costs

```yaml
resource_modifiers:
  batteries:
    technology: Utility-Scale Battery Storage
    tech_detail: Lithium Ion
    capex_mw: [mul, 0.8]
    capex_mwh: [mul, 0.8]
```

The arbitrary short-name key (`batteries`) identifies a modifier. Include exact
`technology` and `tech_detail` selectors matching a resource in `new_resources`;
`*` is not a wildcard. Numeric parameter values set an absolute value; pairs use
`[operator, value]`, with `add`, `mul`, `sub`, or `truediv`. They can also be
year-keyed. Reusing an existing modifier key preserves its other fields.

An ATB trajectory (`Advanced`, `Moderate`, or `Conservative`) is the `cost_case`
entry of each `new_resources` row, not a standalone trajectory setting. To change
it with a swap, provide the **entire** desired resource list in the shape
`[technology, tech_detail, cost_case, size_mw]`. `resource_financial_case` selects
a financial case, not an ATB trajectory. See [New-Build Resources](new-build.md).

### Technology Availability

A list replacement such as this allows only these listed generic candidates:

```yaml
new_resources:
  - [NaturalGas, 1-on-1 Combined Cycle (H-Frame), Moderate, 500]
  - [Utility-Scale Battery Storage, Lithium Ion, Moderate, 1]
```

Any other base `new_resources` entries are removed. Resources can also be
produced by `renewables_clusters`, which must be considered separately.
`new_resources` is not region-scoped: omissions apply everywhere. The legacy
`new_gen_not_available` does not exclude resources, and `ALL_REGIONS` is not a
recognized region key. For region-scoped renewable candidates, use
`renewables_clusters`. Two dimensions replacing `new_resources` in one block
conflict; define a combined candidate list instead.

### Fuel prices and demand

```yaml
fuel_scenarios:
  naturalgas: reference
```

Fuel scenario names must exist in the configured fuel-price data; a `high_price`
label is not automatically available. See [Fuel Settings](fuels.md).

Demand curves are read from the DataManager demand table as supplied. Legacy
`growth_scenario`, `load_eia_aeo_year`, `historical_load_region_map`,
`future_load_region_map`, and `alt_growth_rate` do not apply growth; there is no
`default_growth_rate`. Supply different demand data for a demand sensitivity.
Distributed generation capacity and profiles also come from configured tables,
not a capacity-growth settings value; see
[Distributed Generation](../../explanation/distributed-generation.md).

### Retirement assumptions

```yaml
generation_table:
  table_name: generation_early_retirement.parquet
```

Use a real table containing the desired `retirement_year` values. For myopic
multi-period studies, select the same generation table in **every period of a
case**, for example with an `all_years` swap. Fixed membership is needed for
consistent generator clusters; do not swap or filter generation data by period.
See [Retirement filtering](../../explanation/clustering.md#retirement-filtering)
and [Data Tables](data-tables.md) for table selection syntax.

### Transmission and regional reserves

```yaml
tx_expansion_per_period: 1.0
tx_expansion_mw_per_period: 400
regional_capacity_reserves:
  CapRes_1:
    p1_2: 0.15
    p3: 0.15
    p4: 0.15
```

The allowed reinforcement is the larger of the fraction of existing line
capacity and the specified MW amount. `1.0` allows adding the existing capacity
again; the MW amount is a minimum allowance, not a cap. There are no transmission
settings named `max_network_reinforcement_mw`, `tx_expansion_per_mw`, or
`enforce_constraints`. See [Transmission](transmission.md).

Reserve margins use **constraint → region → fraction**, not a flat region map:
`0.15` means 15%, not `1.15`. A selected level can change several nested fields
at once, with unaffected entries retained by deep merge. See
[Configure Capacity Reserves](../../how-to/configure-capacity-reserves.md).

## Input/Output Folders

### `input_folder`

**Type:** string path.
**Default when omitted:** `extra_inputs`, relative to the working directory.

```yaml
input_folder: extra_inputs
```

When explicitly configured as a relative path, it resolves from the parent of
the settings directory (or from the parent of a single settings file).
`scenario_definitions_fn` resolves under it. Relative `data_location` follows
the same settings-based rule, while relative `RESOURCE_GROUPS` and
`RESOURCE_GROUP_PROFILES` use the current working directory.

### Results Folder

Specify the destination with `--results_folder`; it is not a YAML setting.
With default GenX output and a scenario CSV, the path is
`results/<case_id>/Inputs/Inputs_pN/` when the destination is `results`.
Each period contains `resources/`, `system/`, `policies/`, and the resolved
`powergenome_case_settings.yml`. Without a scenario CSV, the path is
`results/Inputs/Inputs_pN/`. See the [CLI Reference](../cli.md) for output flags.

## Scenario Copying

### `copy_case_id`

**Type:** optional column in the `emission_policies_table` input, not a YAML
setting or a column that copies scenario settings.

```csv
case_id,year,region,copy_case_id,ESR_1
baseline,2030,p1_2,,0.50
carbon_50,2030,p1_2,baseline,
```

The target copies policy values from the source case with the same **year and
region**. Provide corresponding rows for other years and regions as needed.
Policy columns must use output names such as `ESR_1`, not informal `RPS` or `CES`
labels. The bundled fixture instead uses `case_id: all` rows to share its caps
across cases. See [Energy Share Requirements](../../how-to/energy-share-requirements.md).

## Validation

| Condition | Behavior |
| --- | --- |
| Duplicate case/year | Error. |
| Unknown dimension/level for a row | Warning and no swap unless matched by either applicable block. |
| Two selected dimensions change one leaf in a block | Error. |
| CSV years differ from configured years | Runner warns; a row with an unconfigured year subsequently fails planning-year resolution. |
| Missing year within a case | Not automatically rejected; check completeness yourself. |
| Missing year-keyed coverage without `default` | Error during year resolution. |
| Requested `--case-id` absent from CSV | Error. |

Run `validate_powergenome --settings_file settings` for settings/data checks,
then inspect resolved scenario settings and run one case before the full set.
The [how-to validation example](../../how-to/run-scenarios.md#validate-scenario-definitions)
checks CSV completeness explicitly. Reruns do not automatically skip completed cases.

## Related Settings

- [Model Definition](model-definition.md): Planning-year configuration.
- [Year-Keyed Values](year-keyed-values.md): Coverage and fallback rules.
- [New-Build Resources](new-build.md): Candidate lists and resource modifiers.
