# Building a Multi-Region Model

In this tutorial, inspect the bundled study's three-region network, change its
transmission expansion limits, and verify the difference in the generated inputs.
PowerGenome prepares these inputs; GenX decides whether to build transmission.

**Prerequisites**: Complete [Getting Started](getting-started.md). Keep the
PowerGenome environment activated and work from the copied `powergenome-study`
directory, where `settings/`, `test_data/`, and `test_output/` already exist.

## Step 1: Understand the configured regions

Open `settings/model_definition.yml`. The bundled study already contains:

```yaml
model_regions: [p1_2, p3, p4]
region_aggregations:
  p1_2: [p1, p2]
```

`p1`, `p2`, `p3`, and `p4` are **base regions** in the source data.
`p1_2` is a **model region** combining two base regions. The other model regions,
`p3` and `p4`, pass through unchanged and need no aggregation entry.

PowerGenome sums demand and existing capacity into the model regions and
aggregates transmission between them. Connections internal to `p1_2` are no
longer inter-regional lines. Model regions are sorted alphabetically; that order
determines GenX zone numbers.

Do not change the region names in this exercise: all source tables and renewable
groups must cover the selected geography. For another study, start with the
[System Design tool](https://gschivley.github.io/PowerGenome-tools/web/) and check
its settings against your available data. See [Regions](../reference/settings/regions.md)
for the complete regional configuration.

## Step 2: Inspect the network sources

Open `settings/data.yml` and find these existing entries:

```yaml
transmission_constraints_table: reeds_ba_tx_NARIS_avg_test_data.csv
transmission_cost_table: network_costs_test_data.csv
```

They name files under the configured `data_location: test_data`:

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
replacement tables. The bundled data is for testing, not a calibrated network.

## Step 3: Change expansion limits

Replace the contents of the copied study's `settings/transmission.yml` with:

```yaml
tx_expansion_per_period: 0.5
tx_expansion_mw_per_period: 100
```

Do not add duplicate keys in another settings file. These values replace the
bundled limits of `1.0` and `400`.

For each line, PowerGenome allows the **larger** of:

- Existing capacity multiplied by `tx_expansion_per_period`.
- `tx_expansion_mw_per_period`, in MW.

A 500 MW line can therefore add up to 250 MW with these settings. A candidate
route with no existing capacity can add up to 100 MW. Neither parameter is a
minimum build size or a discrete construction increment. Limits apply per
planning period; they do not require GenX to build anything.

## Step 4: Generate the changed inputs

Run from the copied study directory:

```bash
validate_powergenome --settings_file settings
run_powergenome --settings_file settings --results_folder multi_region_test
```

Keep `test_output/` from Getting Started unchanged for comparison. Investigate
new errors or warnings in `multi_region_test/log.txt`; the fixture's known fuel
coverage warnings do not make it suitable for economic analysis.

## Step 5: Verify both planning periods

Run this Python example from the same directory:

```python
from pathlib import Path

import pandas as pd

for period in ("Inputs_p1", "Inputs_p2"):
    relative = Path("Inputs") / period / "system/Network.csv"
    baseline = pd.read_csv(Path("test_output") / relative)
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
`settings/transmission.yml` before continuing another tutorial.

## Next steps

- [Working with Renewable Resources](renewable-resources.md): change wind
  resource resolution using the same bundled dataset.
- [Transmission settings](../reference/settings/transmission.md): capacity
  selection, costs, and expansion parameters.
- [Configure Capacity Reserves](../how-to/configure-capacity-reserves.md):
  requirements and resource credits for regional adequacy constraints.
- [New-build settings](../reference/settings/new-build.md): regional cost
  mappings and the limitations of legacy availability settings.
