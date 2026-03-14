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

In your `extra_inputs` folder, create `emission_policies.csv`. The required columns are `case_id`, `year`, and `region`. Additional columns define policy constraints.

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

Using `region = all` applies the same policy fraction to every model region.

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

### Multi-case example

Use `case_id = all` when the same policy applies to all scenario cases. Use a specific `case_id` to vary policies across scenarios:

```csv
case_id,year,region,ESR_1
all,2030,all,0.50
high_policy,2030,all,0.70
low_policy,2030,all,0.30
```

### Copy policies from another case

Use `copy_case_id` to reuse another case's policies rather than duplicating rows:

```csv
case_id,year,region,copy_case_id,ESR_1
baseline,2030,all,,0.50
alternative,2030,all,baseline,
```

`alternative` will use the same ESR_1 value as `baseline`.

---

## Step 2: Point settings to the file

```yaml
# settings/policies.yml
input_folder: extra_inputs
emission_policies_fn: emission_policies.csv
```

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

!!! note "Tag names must be in model_tag_names"
    Any tag name you use in `model_tag_values` must also be listed in `model_tag_names`. If you add `ESR_1` to `model_tag_values` but not `model_tag_names`, PowerGenome will raise an error.

---

## Step 4: Clean Energy Standard (CES) credits

For a CES — where resources earn fractional credits based on their emission rate relative to coal — enable the `ESR_CES_numerator` option:

```yaml
ESR_CES_numerator: true
```

When enabled, resources earn a credit equal to `(1 - emission_rate / coal_emission_rate)` instead of a binary 0/1 flag. A zero-emission resource earns a credit of 1.0; a resource emitting half of coal earns 0.5.

---

## Verification

After running PowerGenome, check the output:

**`policies/Energy_share_requirement.csv`** — should have one row per zone and one `ESR_<N>` column with the fraction you specified.

**`resources/policy_assignments/Resource_energy_share_requirement.csv`** — should have one row per resource with `ESR_<N>` = 1 for eligible resources and 0 for others.

Cross-check that the technologies you expect to be eligible are marked `1`.

---

## Troubleshooting

**ESR columns missing from output**

Confirm `emission_policies_fn` is set in settings and the file is in `input_folder`. If the CSV has no `ESR_*` columns, no ESR output is produced.

**Resources not marked as eligible**

Check that the technology name in `model_tag_values` is a substring of the resource names in your output. Technology matching is case-insensitive. Use `Generators_data.csv` (or the equivalent resources CSV) to see actual resource names, then adjust tag key strings accordingly.

**Wrong fraction applied to a region**

If using `region = all`, the same value is applied everywhere. Switch to per-region rows if zones need different fractions.

---

## Related documentation

- [Resource Tags Settings](../reference/settings/resource-tags.md): Full `ESR_*` tag reference
- [Run Multi-Scenario Studies](run-scenarios.md): Varying policies across scenarios
- [Output File Format](../explanation/output-format.md): Structure of `Energy_share_requirement.csv`
