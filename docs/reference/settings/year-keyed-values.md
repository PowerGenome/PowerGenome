# Year-Keyed Settings Values

Year-keyed values let you assign different values to different planning years without creating a full multi-scenario matrix.

This pattern is available for any settings parameter value, including nested dictionaries such as `resource_modifiers`, regional settings, and policy values.

!!! warning "Advanced usage"
    Year-keyed values are powerful, but they also add validation rules and precedence behavior. Read this page before using them broadly.

## When To Use This

Use year-keyed values when:

- You need values that change by planning year.
- You have a single case (or a few cases) and do not need full `settings_management` switching.
- You want changes to be resolved automatically per planning period.

Use `settings_management` when:

- You need different values across many scenario dimensions (`case_id`, policy, fuel cases, etc.).
- You need explicit scenario matrices in `scenario_definitions_fn`.

## Format

A value is treated as year-keyed when all keys are either:

- Integer years in the range `1900` to `2200`, or
- Special fallback keys: `all` or `default`

And at least one integer year key is present.

```yaml
carbon_tax:
  2030: 25
  2040: 50
```

Nested example:

```yaml
resource_modifiers:
  utility_pv:
    technology: UtilityPV
    tech_detail: Class1
    capex_mw:
      2030: [mul, 0.90]
      2040: [mul, 0.80]
```

## Selection Rules

For each planning year, PowerGenome resolves one value using this priority:

1. Exact year match (for example, `2030`)
2. `all` fallback
3. `default` fallback

If no match is found, PowerGenome raises a `ValueError`.

Example with fallback:

```yaml
capex_mw:
  2030: [mul, 0.92]
  default: [mul, 1.0]
```

- In `2030`, value is `[mul, 0.92]` (exact year wins).
- In `2040`, value is `[mul, 1.0]` (uses `default`).

## Coverage Validation

When PowerGenome builds per-year case settings, year-keyed values are validated against the planning years in your run.

A year-keyed dict must satisfy one of these:

- Every planning year has an explicit key, or
- It includes `all` or `default` as a catch-all fallback

If only some planning years are covered and no fallback exists, PowerGenome raises a `ValueError`.

## Common Patterns

Exact years only:

```yaml
reserve_margin:
  2030: 0.15
  2040: 0.18
  2050: 0.20
```

Exact year plus broad fallback:

```yaml
co2_pipeline_capex_mw:
  2030: 120000
  all: 100000
```

`2030` uses `120000`; other years use `100000`.

Constant across all years (prefer scalar):

```yaml
# Preferred
carbon_tax: 25

# Also valid, but more verbose
# carbon_tax:
#   2030: 25
#   all: 25
```

## Interaction With Scenario Management

Year-keyed resolution is applied to normal settings values when per-year scenario settings are built.

The `settings_management` structure itself is not transformed by year-keyed resolution.

If you need both scenario and year variation:

1. Use year-keyed values for baseline by-year behavior.
2. Use `settings_management` for scenario-specific overrides.

See [Scenario Management](scenario-management.md) for multi-case workflows.
