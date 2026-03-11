# Use Year-Keyed Settings Values

This guide shows how to set values by planning year directly in settings files.

Use this pattern when you want different values in different model years
without building a full scenario matrix.

## Before You Start

You should already have:

- A working settings folder
- Planning years defined (`model_year` and `model_first_planning_year`)
- A parameter you want to vary by year

If you need full multi-case scenario logic, use [Run Multi-Scenario Studies](run-scenarios.md).

## 1. Start With A Working Baseline

Example planning years:

```yaml
model_year: [2030, 2040]
model_first_planning_year: [2020, 2031]
```

## 2. Convert A Scalar Value To A Year-Keyed Value

Suppose you currently have:

```yaml
carbon_tax: 25
```

Convert it to year-keyed form:

```yaml
carbon_tax:
  2030: 25
  2040: 50
```

PowerGenome will resolve one value per planning year when building case settings.

## 3. Use Fallback Values With `all` Or `default`

Use a fallback when you do not want to list every year explicitly.

```yaml
carbon_tax:
  2030: 25
  default: 40
```

Behavior:

- Year `2030` uses `25` (exact year match).
- Any other planning year uses `40` (`default` fallback).

You can also use `all` as a fallback key.

## 4. Apply Year-Keyed Values In Nested Settings

You can use this pattern inside nested dictionaries too.

```yaml
resource_modifiers:
  utility_pv:
    technology: UtilityPV
    tech_detail: Class1
    capex_mw:
      2030: [mul, 0.90]
      2040: [mul, 0.80]
```

## 5. Validate Coverage Rules

For each year-keyed dictionary, PowerGenome expects either:

- Explicit entries for all planning years, or
- A fallback key (`all` or `default`)

If only some years are covered and no fallback exists, PowerGenome raises a `ValueError`.

Example that fails for planning years `[2030, 2040]`:

```yaml
capex_mw:
  2030: [mul, 0.95]
```

Fix by adding missing years or a fallback:

```yaml
capex_mw:
  2030: [mul, 0.95]
  default: [mul, 1.0]
```

## 6. Keep `settings_management` For Scenario Dimensions

Year-keyed values handle variation by planning year.

Use `settings_management` when you need differences across scenario
columns (policy, fuel case, technology assumptions, and so on).

A common pattern:

- Baseline year variation via year-keyed values
- Scenario-specific deltas via `settings_management`

## See Also

- [Year-Keyed Values (Reference)](../reference/settings/year-keyed-values.md)
- [Configure Settings](configure-settings.md)
- [Run Multi-Scenario Studies](run-scenarios.md)
