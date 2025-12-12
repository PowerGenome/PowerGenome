---
title: Incentives Settings Reference
---

### Overview

PowerGenome supports two incentive types:

- Investment incentives: fractional values applied to capital costs.
- Production incentives: per-unit credits with a required basis (MWh or Tonne_CO2).

Both are configured in a standalone incentives.yml and translated into GenX policy files and resource assignment matrices.

### Settings Schema

Top-level keys:

- investment_incentives: map of Inv_Incentive_N → policy.
- production_incentives: map of Prod_Incentive_N → policy.

Policy object fields:

- value: number (required). Fraction (investment) or credit (production).
- technologies: list[str] (required). Eligibility via case-insensitive substring match on technology, after removing spaces and underscores.
- description: string (optional). Used for PolicyDescription in policy tables. Defaults to the policy key.
- type: string (production only). Exactly one of MWh or Tonne_CO2 (case-insensitive; spaces allowed).

Naming and numbering rules:

- Keys must follow Inv_Incentive_N and Prod_Incentive_N.
- N must start at 1 and be consecutive (no gaps, no duplicates).

Example:

```yaml
investment_incentives:
  Inv_Incentive_1:
    value: 0.3
    description: ITC for VRE
    technologies: [wind, solar]
  Inv_Incentive_2:
    value: 0.2
    technologies: [nuclear, geothermal]

production_incentives:
  Prod_Incentive_1:
    value: 26
    type: MWh
    description: PTC for wind
    technologies: [wind]
  Prod_Incentive_2:
    value: 85
    type: Tonne_CO2
    description: 45Q for CCS
    technologies: [CCS]
```

### Output Files and Columns

Policies (case_folder/Inputs/Inputs_pX/policies/):

- Investment_incentive.csv
  - Policy_ID: integer (from N in Inv_Incentive_N; ordered by N).
  - PolicyDescription: string (`description` if set, else key).
  - Value: number (fractional credit).

- Production_incentive.csv
  - Policy_ID: integer (from N in Prod_Incentive_N; ordered by N).
  - PolicyDescription: string (`description` if set, else key).
  - Value: number (credit magnitude).
  - Production_Type: one of MWh or Tonne_CO2.

Resource assignments (case_folder/Inputs/Inputs_pX/resources/policy_assignments/):

- Resource_investment_incentive.csv
  - Resource: string.
  - Inv_Incentive_N columns: binary flags (1 if eligible).
  - Only includes resources qualifying for at least one investment incentive.

- Resource_production_incentive.csv
  - Resource: string.
  - Prod_Incentive_N columns: binary flags (1 if eligible).
  - Only includes resources qualifying for at least one production incentive.

### Matching Behavior

- Technology strings are normalized (remove spaces/underscores; case-insensitive) and matched via substring containment. For example:
  - "wind" matches "Onshore Wind".
  - "CCS" matches "Natural Gas CCS100".

### Validation and Errors

- Missing value or invalid technologies type raise errors.
- Production incentives require exactly one valid type (MWh or Tonne_CO2).
- Numbering must be consecutive starting at 1; gaps or duplicates raise errors.
