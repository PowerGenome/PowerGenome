# Multi-Scenario Architecture

PowerGenome separates a study into **cases** (alternative assumptions) and
**planning periods** (time within each case). It generates inputs for those
case/year pairs; it does not run the capacity-expansion optimization.

This page explains why settings resolve as they do. Follow
[Run Multiple Scenarios](../how-to/run-scenarios.md) for a runnable example, or
use [Scenario Management Settings](../reference/settings/scenario-management.md)
for accepted shapes and parameter-specific patterns.

## Two types of variation

Use **year-keyed values** for changes over time, such as an increasing carbon tax
within one case. Use **scenario management** for alternatives, such as comparing
a tax with no tax. These mechanisms can be combined: a selected scenario swap
can itself contain year-keyed values.

A year-keyed dictionary needs integer years, optionally a `default` fallback,
and coverage of the planning years being resolved. It is not a scenario
selector. `all_years` is instead a selector inside `settings_management`; it is
not a fallback key for ordinary year-keyed values. See
[Year-Keyed Values](../reference/settings/year-keyed-values.md).

## The scenario matrix

Each CSV row selects one case/year pair. Extra columns select named categories
and levels in `settings_management`. The CSV is an explicit list of combinations,
not a request to generate a Cartesian product. A case ID stays the same across
its planning periods.

Duplicate case/year pairs are rejected. A complete case should contain every
planning year, but completeness is not enforced automatically. Period numbers
are assigned by counting each case in CSV row order, independently of year
resolution. Thus a row for 2040 can be labeled period 1 if it precedes that
case’s 2030 row. Chronological rows and correctly paired planning-year lists
prevent this mismatch.

Execution order is different from period numbering: the runner groups rows by
year in first-seen order, then processes cases within that year in CSV order.
It executes sequentially. With a chronological matrix, all first-period cases
run before any second-period cases.

## How settings are constructed for each case

Base settings loading and scenario swapping have different merge rules:

- **Loading a directory:** immediate `*.yml` files are combined using shallow
  dictionary updates. A duplicate top-level key replaces the previous value;
  file order is not guaranteed. Splitting `settings_management` across files
  does not combine the mappings.
- **Applying a scenario:** each row starts with an independent deep copy of the
  loaded base settings. Nested dictionaries in swaps merge recursively;
  scalar values and lists replace existing values.

For each CSV row, `build_scenario_settings` applies these stages:

1. Assign the case ID and its row-derived period number.
2. Apply `settings_management.all_years.all_cases`.
3. Apply matching dimension/level swaps in `settings_management.all_years`,
   following the CSV column order.
4. Apply `settings_management.<year>.all_cases`.
5. Apply matching dimension/level swaps in `settings_management.<year>`,
   following the CSV column order.
6. Resolve planning-year lists or periods to scalar start/end years, then
   resolve year-keyed dictionaries. Expand reserve and output-tag settings.

The entire `all_years` block is processed before the entire specific-year block.
In particular, a year-specific `all_cases` value can override a value selected
by a dimension under `all_years`. It is not true that all dimension selections
always take precedence over all `all_cases` values.

### Deep merge behavior

Suppose a base resource modifier has both operating-cost fields:

```yaml
resource_modifiers:
  batteries:
    technology: Utility-Scale Battery Storage
    tech_detail: Lithium Ion
    Var_OM_Cost_per_MWh: [add, 0.15]
    Var_OM_Cost_per_MWh_In: 0.15
```

A swap with the same short-name key can change just one field:

```yaml
settings_management:
  all_years:
    storage_cost:
      higher:
        resource_modifiers:
          batteries:
            Var_OM_Cost_per_MWh: [add, 0.30]
```

The technology selectors and `Var_OM_Cost_per_MWh_In` survive. The list
`[add, 0.15]` is replaced by `[add, 0.30]`; it is not extended. Similarly,
setting `new_resources` replaces the entire candidate list, not just one entry.
A dictionary swap needs an absent, null, or dictionary base value; it cannot
merge into an existing scalar or list. When adding a year-keyed dictionary to
a swap, give the base setting a compatible shape first.

Two selected dimensions in the **same block** cannot modify the same leaf
setting, even to the same value. This conflict check restarts for the
specific-year block and does not include `all_cases`, allowing the deliberate
precedence described above. Use a combined dimension when two choices both
need to replace the same list; do not rely on column order to break the conflict.

Each row starts afresh. A swap in 2030 does not carry forward to 2040. Use
`all_years` or an explicit later-year swap when the change must persist.

## The `all_cases` selector

`all_cases` contains setting/value pairs directly, without a dimension or level.
It applies to every row covered by its enclosing block. It is useful for a
shared year-specific assumption; a matching dimension in that same block may
still override it. See the
[selector format](../reference/settings/scenario-management.md#settings_management)
for a compact example.

## Multi-period consistency

Settings isolation lets different cases use different input tables. It does not
make arbitrary between-period changes safe. For myopic multi-period studies,
keep the same generation table in every period of a case so existing-generator
cluster membership stays fixed. Retirement assumptions belong in that data’s
`retirement_year` column, not in period-by-period table filtering. See
[Retirement filtering](clustering.md#retirement-filtering).

## Output folder structure

With a scenario CSV, default GenX files go to
`results/<case_id>/Inputs/Inputs_pN/{resources,system,policies}/` when the results
folder is named `results`. `N` is the row-derived period number, not a calendar
year. The sibling `powergenome_case_settings.yml` records resolved settings for
checking the generated inputs. See the
[how-to output tree](../how-to/run-scenarios.md#run-the-scenario-set) for an example.

## Running without scenario definitions

Without `scenario_definitions_fn`, the runner creates an implicit case named
`Inputs` and resolves settings for the configured planning years in list order.
Year-keyed values still work, but `settings_management` swaps are not applied.
There are no dimension selections, and `--case-id` is ignored.

The default GenX path omits the case-ID directory:
`results/Inputs/Inputs_pN/`. Adding a scenario CSV is therefore not just another
way to name the same output folder.

## Related documentation

- [Run Multiple Scenarios](../how-to/run-scenarios.md): Create and test two cases.
- [Scenario Management Settings](../reference/settings/scenario-management.md):
  Schemas, validation behavior, and advanced combinations.
- [Data Pipeline Flow](pipeline.md): How resolved settings feed data processing.
