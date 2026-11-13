# Building a Multi-Region Model

In this tutorial, inspect the published Texas study's three-region network,
change its transmission expansion limits, and verify the difference in the
generated inputs. PowerGenome prepares these inputs; GenX decides whether to
build transmission.

**Prerequisites**: Complete [Getting Started](getting-started.md). Keep the
PowerGenome environment activated and work from `powergenome-tutorial/`, where
`example_systems/TX-3-zone/` and the unchanged `tx_output/` baseline already exist.
Do not change into the Texas subdirectory: resource paths resolve from the
examples root.

## Step 1: Understand the configured regions

Open `example_systems/TX-3-zone/settings/model_definition.yml`. It contains:

```yaml
model_regions: [TX1, TX2, TX3]
region_aggregations:
  TX1: [p60, p61, p62]
  TX2: [p63, p64]
  TX3: [p65, p67]
```

The seven `p...` names are ReEDS balancing areas, the **base regions** in the
source data. `TX1`, `TX2`, and `TX3` are **model regions** combining them.

PowerGenome sums demand and existing capacity into the model regions and
aggregates transmission between them. Connections internal to `TX1`, such as
`p60`–`p61`, are no longer inter-regional lines. Model regions are sorted
alphabetically; that order determines GenX zone numbers.

Do not change the region names in this exercise: all source tables and renewable
groups must cover the selected geography. For another study, start with the
[System Design tool](https://gschivley.github.io/PowerGenome-tools/web/) and check
its settings against your available data. See [Regions](../reference/settings/regions.md)
for the complete regional configuration.

## Step 2: Inspect the network sources

Open `example_systems/TX-3-zone/settings/data.yml` and find these entries:

```yaml
transmission_constraints_table: transmission_capacity_tx_3_zone.csv
transmission_cost_table: network_costs_tx_3_zone.csv
```

They name files under `data_location: data`, relative to
`example_systems/TX-3-zone/`:

- **Constraints**: directed base-region capacities in `region_from`, `region_to`,
  and `firm_ttc_mw`. PowerGenome aggregates these into model-region connections.
- **Costs**: model-region pairs in `start_region` and `dest_region`, with capital
  cost, annualized cost, dollar year, and optional line loss.

The constraints table defines routes; neither cost-table entries nor YAML
expansion settings create routes. To add a candidate route with no existing
capacity, include its base-region pair in the constraints table with
`firm_ttc_mw: 0`, and supply the corresponding model-region pair's costs in the
cost table. Confirm that the route appears in the generated `Network.csv`.
To exclude a connection, remove all constraints-table entries that map to it.

See the [input schemas](../reference/schemas/input-tables.md) before supplying
replacement tables. The published example is not a calibrated network.

## Step 3: Change expansion limits

Replace the contents of
`example_systems/TX-3-zone/settings/transmission.yml` with:

```yaml
tx_expansion_per_period: 0.5
tx_expansion_mw_per_period: 100
```

Do not add duplicate keys in another settings file. These values replace the
published limits of `1.0` and `400`.

For each line, PowerGenome allows the **larger** of:

- Existing capacity multiplied by `tx_expansion_per_period`.
- `tx_expansion_mw_per_period`, in MW.

A 500 MW line can therefore add up to 250 MW with these settings. A candidate
route with no existing capacity can add up to 100 MW. Neither parameter is a
minimum build size or a discrete construction increment. Limits apply per
planning period; they do not require GenX to build anything.

## Step 4: Generate the changed inputs

Run from `powergenome-tutorial/`:

```bash
validate_powergenome --settings_file example_systems/TX-3-zone/settings
run_powergenome \
  --settings_file example_systems/TX-3-zone/settings \
  --results_folder multi_region_test
```

Keep `tx_output/` from Getting Started unchanged for comparison. Investigate
errors or new warnings in `multi_region_test/log.txt`. Both planning periods
(2030 and 2045) are written under `multi_region_test/base/Inputs/`.

## Step 5: Verify both planning periods

Run this Python example from the same directory:

```python
from pathlib import Path

import pandas as pd

for period in ("Inputs_p1", "Inputs_p2"):
    relative = Path("base/Inputs") / period / "system/Network.csv"
    baseline = pd.read_csv(Path("tx_output") / relative)
    changed = pd.read_csv(Path("multi_region_test") / relative)
    key = "transmission_path_name"
    baseline = baseline.set_index(key).sort_index()
    changed = changed.set_index(key).sort_index()

    pd.testing.assert_index_equal(baseline.index, changed.index)
    pd.testing.assert_series_equal(
        baseline["Line_Max_Flow_MW"], changed["Line_Max_Flow_MW"]
    )
    expected = (changed["Line_Max_Flow_MW"] * 0.5).clip(lower=100).round()
    pd.testing.assert_series_equal(
        changed["Line_Max_Reinforcement_MW"], expected, check_names=False,
        check_dtype=False,
    )
    assert (
        changed["Line_Max_Reinforcement_MW"]
        < baseline["Line_Max_Reinforcement_MW"]
    ).any()
    print(period)
    print(changed[["Start_Zone", "End_Zone", "Line_Max_Flow_MW",
                   "Line_Max_Reinforcement_MW"]])
```

The assertions confirm that the routes and existing capacities are unchanged,
while expansion bounds reflect the new settings in both periods. Inspect
`Line_Reinforcement_Cost_per_MWyr` and `Line_Loss_Percentage` in the same file
when reviewing the cost and loss assumptions.

Restore `tx_expansion_per_period: 1.0` and `tx_expansion_mw_per_period: 400` in
`example_systems/TX-3-zone/settings/transmission.yml` before continuing another
tutorial.

## Next steps

- [Working with Renewable Resources](renewable-resources.md): change wind
  resource resolution using the same published Texas dataset.
- [Transmission settings](../reference/settings/transmission.md): capacity
  selection, costs, and expansion parameters.
- [Configure Capacity Reserves](../how-to/configure-capacity-reserves.md):
  requirements and resource credits for regional adequacy constraints.
- [New-build settings](../reference/settings/new-build.md): regional cost
  mappings and excluding new-build technologies from specific regions.
