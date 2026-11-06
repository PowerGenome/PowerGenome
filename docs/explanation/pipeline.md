# Data Pipeline Flow

This page follows a `run_powergenome` execution from settings to output files.
For component responsibilities, see [Architecture Overview](architecture.md).

```text
Settings and validation → DataManager → Gross demand → Generators and fuels
→ Net demand and time reduction → Transmission and policies → Output writers
```

## Stage 1: Settings loading and validation

PowerGenome loads the settings directory's immediate `.yml` files. Top-level
values replace one another rather than recursively merging, so avoid duplicate
keys across files.

The scenario CSV, when configured, defines case/period combinations. Otherwise,
the planning-year settings define one sequence of periods. Scenario swaps and
year-keyed values resolve the configuration for each pass. Settings validation
runs before data initialization; errors stop the pipeline.

**Product**: base settings plus resolved case/period settings. See
[Scenario Management](scenarios.md) for resolution rules and
[Validate Settings](../how-to/validate-settings.md) for validation coverage.

## Stage 2: DataManager initialization

DataManager creates tables or views over the configured sources, mapping settings
keys to standard names such as `generation`, `demand`, and `fuel_price`.
Data-dependent validation checks those sources against the settings.

At each case/period boundary, `update_data_manager()` refreshes configurations
for the resolved settings, including row filters. It does not require each
processing module to reopen and interpret filenames.

**Product**: standardized data access. See
[Configure Data Tables](../how-to/configure-data-tables.md).

## Stage 3: Gross demand, generators, and fuels

When load processing is enabled, unrounded gross demand is constructed before
generator clustering. It also supplies the denominator for existing-resource
profile fallback checks.

`GeneratorClusters.create_all_generators()` then combines:

- **Existing units**: label operating/retirement status, group by region and
  technology, cluster using the configured features and counts, and aggregate
  capacity and operating attributes. Non-operating units remain in cluster
  membership but do not contribute operating capacity.
- **New resources**: select technology cost/performance rows, apply resource
  modifiers and renamed variants, and build renewable resources from the
  configured sites and profiles.

Fuel mappings, regional cost factors, startup costs, and other attributes are
applied as required by the resource workflow.

**Products**: resource attributes (`gen_data`), hourly resource variability,
and fuel-price inputs—not yet the final split GenX files.

## Stage 4: Net demand and time reduction

`make_final_load_curves()` uses the earlier gross demand, including configured
supplemental/flexible demand. Distributed generation is subtracted unless it is
represented as an explicit resource with `dg_as_resource`.

When `reduce_time_domain: true`, load and generation variability are reduced
together into representative groups of days. Weights and mapping data describe
how those periods represent the original series. Without reduction, all input
hours pass through as one period. Reduction is skipped for series longer than
8760 hours.

**Products**: aligned demand and variability profiles, weights, and time-series
mapping. An unweighted reduced sum is not annual energy. See
[Time Reduction Settings](../reference/settings/time-reduction.md).

## Stage 5: Transmission and policies

Transmission processing reads `transmission_constraints`, aggregates capacities
to model regions, and adds costs/losses from `transmission_cost`. It then
calculates expansion bounds and other model-facing network columns.

Policy processing assembles requirements and resource assignments from the
configured policy tables and settings—for example, energy-share requirements,
CO₂ constraints, capacity reserves, and minimum/maximum capacity requirements.

**Products**: network and policy DataFrames. See
[Transmission Settings](../reference/settings/transmission.md) and
[Resource Tags](../reference/settings/resource-tags.md).

## Stage 6: Output formatting and writing

Model-specific formatting casts/rounds selected columns, checks resource tags,
and arranges the processed DataFrames into files. The default GenX layout is:

```text
<results_folder>/[<case_id>/]Inputs/Inputs_p<period>/
    system/       # Demand, fuel, network, and variability files
    resources/    # Resource-type files and policy assignments
    policies/     # Policy requirements
    extra_outputs/
    powergenome_case_settings.yml
```

The case subdirectory is used when running a scenario CSV; without one, `Inputs`
is directly under the results folder. Resource files for empty types and files
for unused features are omitted. `old_genx_format: true` selects legacy combined
files. The saved settings record the resolved configuration for that period.

See [Output File Schemas](../reference/schemas/output-files.md) for the canonical
layout and column definitions, and [Macro Output](../reference/settings/macro-output.md)
for that writer's scope.

## Execution flags

Skip-stage flags are mainly useful for focused development and diagnostics:

| Flag | Skips |
| --- | --- |
| `--no-current-gens` | Existing generator clustering; new-build resources still run. |
| `--no-gens` | The entire generator stage. |
| `--no-load` | Load construction and time reduction. |
| `--no-transmission` | Transmission and policy files. |

Not every partial-stage combination produces complete model inputs. The default
GenX writer expects generator data, so `--no-gens` alone may fail to produce a
usable output. See the [CLI Reference](../reference/cli.md) rather than using
skip flags as a first-run recipe.
