# Configure Energy Share Requirements

**Energy Share Requirements (ESR)** are constraints that require a minimum fraction of load to be served by eligible resources. They are how PowerGenome models **Renewable Portfolio Standards (RPS)** and **Clean Energy Standards (CES)**.

This guide walks through setting up ESR constraints and assigning resource eligibility.

---

## Prerequisites

- A working PowerGenome settings folder (see [Getting Started](../tutorials/getting-started.md))
- An `extra_inputs` folder referenced by `input_folder` in settings

---

## How ESR constraints work

An ESR constraint says: "At least X% of load in zone Z must be served by eligible resources for constraint N."

Two inputs define each constraint:

1. **`emission_policies.csv`** — sets the minimum fraction per zone per constraint
2. **`model_tag_values` (ESR tags)** — marks which resources are eligible for which constraint

PowerGenome writes these to `Energy_share_requirement.csv` (zone requirements) and `Resource_energy_share_requirement.csv` (resource eligibility) in the GenX output.

---

## Step 1: Create the emission policies CSV

In your `extra_inputs` folder, create `emission_policies.csv` with `case_id`,
`year`, `region`, and policy columns. `case_id` may be omitted when every row
applies to the current case, but include it when using `copy_case_id`.

**Column naming**:

- `ESR_<N>` — energy share requirement constraint N (fraction, e.g. `0.5` = 50%)
- `CO_2_Cap_Zone_<N>` — marks zone as part of CO₂ cap constraint N (`1` = included)
- `CO_2_Max_Mtons_<N>` — CO₂ cap in million metric tons for constraint N

### Single-region example (RPS applied to all zones equally)

```csv
case_id,year,region,ESR_1
all,2030,all,0.50
all,2040,all,0.70
all,2050,all,0.90
```

A single policy row for a case and year applies its values to every model
region. Here `all` makes that intent clear.

### Multi-region example (different RPS per zone)

```csv
case_id,year,region,ESR_1,ESR_2
all,2030,CA_N,0.60,0.80
all,2030,CA_S,0.60,0.80
all,2030,AZ,0.15,0.00
all,2040,CA_N,0.80,0.90
all,2040,CA_S,0.80,0.90
all,2040,AZ,0.25,0.00
```

### Published multi-region example

The CA/AZ example uses two model regions with a different RPS and CES in each:

```csv
case_id,year,region,ESR_1,ESR_2,ESR_3
base,2030,CA,0.20,0.25,0
base,2030,AZ,0.087,0.087,0
base,2045,CA,0.15,0.15,0
base,2045,AZ,0.087,0.087,0
```

The region names match `model_regions` (`CA` and `AZ`), not the ReEDS base
regions behind them. The Texas example has the same shape for `TX1`-`TX3` but
leaves every ESR column at `0`, which is a useful starting point when you want
the table structure without a binding policy.

### Multi-case example

Use `case_id = all` as a fallback for cases with no explicit rows. Use a specific
`case_id` to vary policies across scenarios:

```csv
case_id,year,region,ESR_1
all,2030,all,0.50
high_policy,2030,all,0.70
low_policy,2030,all,0.30
```

The fallback is selected only when the case ID is absent from the entire policy
table, not separately for each year or region. Include every needed year for an
explicit case.

### Copy policies from another case

Use `copy_case_id` to reuse another case's policies rather than duplicating rows:

```csv
case_id,year,region,copy_case_id,ESR_1
baseline,2030,all,,0.50
alternative,2030,all,baseline,
```

`alternative` copies the policy values from `baseline` for the same year and
region. Keep `case_id`, `year`, and `region` as the first three columns.

---

## Step 2: Point settings to the file

```yaml
# settings/policies.yml
input_folder: extra_inputs
emission_policies_table: emission_policies.csv
```

`emission_policies_table` is loaded through the DataManager, so the value may use
any supported DataManager table configuration rather than only a CSV path. The
legacy `emission_policies_fn` key is still accepted as an alias; both published
examples currently use it.

---

## Step 3: Define which resources are eligible

In `model_tag_values` in settings, add ESR tags. The tag names must match the column names in your CSV (`ESR_1`, `ESR_2`, etc.):

```yaml
model_tag_names:
  - THERM
  - VRE
  - MUST_RUN
  - STOR
  - HYDRO
  - ESR_1    # RPS: renewable energy requirement
  - ESR_2    # CES: clean energy standard

model_tag_values:
  ESR_1:
    LandbasedWind: 1
    OffShoreWind: 1
    UtilityPV: 1
    ResPV: 1
    Hydroelectric: 1
    Geothermal: 1
    NaturalGas: 0
    Coal: 0
    Nuclear: 0

  ESR_2:
    LandbasedWind: 1    # Eligible for CES
    OffShoreWind: 1
    UtilityPV: 1
    Nuclear: 1          # Nuclear counts as clean
    NaturalGas_CCS: 1   # Gas with CCS may count as clean
    NaturalGas: 0
    Coal: 0
```

Only resources with tag value `1` count toward the corresponding constraint.
Both published examples use this split: `ESR_1` covers renewables, `ESR_2` adds
nuclear and CCS, and `ESR_3` stays at zero.

Tag keys in `model_tag_values` are included even if they are absent from
`model_tag_names`. Check the resource technology names in your data and adjust
the examples above to match.

---

## Step 4: Clean Energy Standard (CES) credits

PowerGenome's GenX output validates `ESR_*` resource tags as binary values.
It does not calculate emission-rate-based fractional CES credits from an
`ESR_CES_numerator` setting. Use binary eligibility or configure a different
policy mechanism in the downstream model.

---

## Verification

After running PowerGenome, check the output:

**`policies/Energy_share_requirement.csv`** — should have one row per zone and one `ESR_<N>` column with the fraction you specified.

**`resources/policy_assignments/Resource_energy_share_requirement.csv`** — should have one row per resource with `ESR_<N>` = 1 for eligible resources and 0 for others.

Cross-check that the technologies you expect to be eligible are marked `1`.

---

## Troubleshooting

**ESR columns missing from output**

Confirm `emission_policies_table` is set in settings and the file is in `input_folder`. If the CSV has no `ESR_*` columns, no ESR output is produced.

**Resources not marked as eligible**

Check that each key in `model_tag_values` is a substring of the resource
`technology` field. Matching is case-insensitive and ignores underscores. Inspect
the resource CSVs and adjust the keys accordingly.

**Wrong fraction applied to a region**

If a case/year has only one policy row, PowerGenome applies it to every zone,
regardless of its `region` value. Use multiple per-region rows when zones need
different fractions.

---

## Related documentation

- [Resource Tags Settings](../reference/settings/resource-tags.md): Full `ESR_*` tag reference
- [Run Multi-Scenario Studies](run-scenarios.md): Varying policies across scenarios
- [Output File Format](../explanation/output-format.md): Structure of `Energy_share_requirement.csv`
