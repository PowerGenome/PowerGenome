---
title: Configure Investment and Production Incentives
---

This guide shows how to define investment and production incentives in settings and how those settings translate to GenX input files.

### What You Get

- policies/Investment_incentive.csv
- policies/Production_incentive.csv
- resources/policy_assignments/Resource_investment_incentive.csv
- resources/policy_assignments/Resource_production_incentive.csv

### Prerequisites

- PowerGenome version with incentives support.
- A working study folder using the tests/test_system layout as a template.

### Step 1 — Create incentives.yml

Add a new file in your settings folder named incentives.yml. Example:

```yaml
investment_incentives:
  Inv_Incentive_1:
    value: 0.30
    description: ITC for VRE
    technologies:
      - wind
      - solar
  Inv_Incentive_2:
    value: 0.20
    technologies:
      - nuclear
      - geothermal

production_incentives:
  Prod_Incentive_1:
    value: 26
    type: MWh
    description: PTC for wind
    technologies:
      - wind
  Prod_Incentive_2:
    value: 85
    type: Tonne_CO2
    description: 45Q for CCS
    technologies:
      - CCS
```

Notes:

- Use names Inv_Incentive_N and Prod_Incentive_N with N starting at 1 and increasing by 1 without gaps.
- Production incentives must have exactly one `type`: either MWh or Tonne_CO2 (case-insensitive; spaces allowed).
- `description` is optional; if absent, the policy name is used for PolicyDescription.
- `technologies` is a list of strings matched against technology names using case-insensitive substring match after removing spaces and underscores.

### Step 2 — Run PowerGenome

Run your pipeline as usual. For example:

```bash
run_powergenome --settings_file settings --results_folder output_dir
```

The writers will emit four incentive files into your case folder:

- policies/Investment_incentive.csv (Policy_ID, PolicyDescription, Value)
- policies/Production_incentive.csv (Policy_ID, PolicyDescription, Value, Production_Type)
- resources/policy_assignments/Resource_investment_incentive.csv (Resource, Inv_Incentive_1, …)
- resources/policy_assignments/Resource_production_incentive.csv (Resource, Prod_Incentive_1, …)

### Step 3 — Verify outputs

- Policy IDs: Ordered by numeric suffix (e.g., Inv_Incentive_1 → 1). Numbering must be consecutive with no gaps.
- Production type: Exactly one per policy; normalized to MWh or Tonne_CO2.
- Resource assignment: Only resources that qualify for at least one incentive appear (rows with all-zero policy flags are dropped).

### Current Limitations

- Eligibility is global (no regional scoping) and based on technology string matching.
- Overlapping matches are allowed; a resource can qualify for multiple incentives.
