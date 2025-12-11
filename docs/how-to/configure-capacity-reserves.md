# Configure Capacity Reserve Requirements

This guide shows how to specify capacity reserve credit values for different technologies and constraints in PowerGenome. It covers both simple and complex scenarios using the `capacity_reserve_values` and `regional_capacity_reserves` settings.

## Overview

Capacity reserve requirements define how much capacity different technologies contribute to meeting reserve obligations. PowerGenome supports specifying these values in a user-friendly way that automatically populates the underlying `regional_tag_values` structure.

## Basic Setup: Uniform Technology Credits

For systems where technologies get the **same credit values across all capacity reserve constraints**, use the **flat format**:

```yaml
regional_capacity_reserves:
  CapRes_1:
    p1: 0.164
    p2: 0.164
    p3: 0.164
  CapRes_2:
    p1: 0.090
    p2: 0.090
    p3: 0.090

capacity_reserve_values:
  Conventional Steam Coal: 0.9
  Natural Gas Fired Combined Cycle: 0.9
  Onshore Wind Turbine: 0.15
  Utility Solar Photovoltaic: 0.25
  Hydroelectric: 1.0
  Nuclear: 1.0
```

In this format:

- All technologies listed in `capacity_reserve_values` receive the same credit values
- These values apply to **all** capacity reserve constraints in `regional_capacity_reserves`
- Each region gets populated automatically with the technology credits

**Result** (internally):

```yaml
regional_tag_values:
  p1:
    CapRes_1:
      Conventional Steam Coal: 0.9
      Natural Gas Fired Combined Cycle: 0.9
      Onshore Wind Turbine: 0.15
      Utility Solar Photovoltaic: 0.25
      Hydroelectric: 1.0
      Nuclear: 1.0
    CapRes_2:
      Conventional Steam Coal: 0.9
      Natural Gas Fired Combined Cycle: 0.9
      Onshore Wind Turbine: 0.15
      Utility Solar Photovoltaic: 0.25
      Hydroelectric: 1.0
      Nuclear: 1.0
  p2:
    CapRes_1:
      Conventional Steam Coal: 0.9
      ...
    CapRes_2:
      Conventional Steam Coal: 0.9
      ...
  # Similar structure for p3
```

## Advanced Setup: Multiple Constraints

For systems where technologies have **different credit values depending on the constraint**, use the **nested format**:

```yaml
regional_capacity_reserves:
  CapRes_1:
    p1: 0.164
    p2: 0.164
    p3: 0.164
  CapRes_2:
    p4: 0.180
    p5: 0.180
    p6: 0.180
  CapRes_3:
    p1: 0.090
    p2: 0.090
    p3: 0.090
    p4: 0.090
    p5: 0.090
    p6: 0.090

capacity_reserve_values:
  CapRes_1:
    Conventional Steam Coal: 0.85
    Natural Gas Fired Combined Cycle: 0.85
    Onshore Wind Turbine: 0.05
    Hydroelectric: 1.0
    Nuclear: 1.0
  CapRes_2:
    Conventional Steam Coal: 0.95
    Natural Gas Fired Combined Cycle: 0.95
    Onshore Wind Turbine: 0.25
    Utility Solar Photovoltaic: 0.40
    Hydroelectric: 1.0
    Nuclear: 1.0
  CapRes_3:
    Conventional Steam Coal: 0.9
    Natural Gas Fired Combined Cycle: 0.9
    Onshore Wind Turbine: 0.15
    Utility Solar Photovoltaic: 0.25
    Hydroelectric: 1.0
    Nuclear: 1.0
```

In this format:

- Each constraint has its own credit values for technologies
- Constraints can have different requirements in different regions
- Technologies can have different credits depending on the constraint

**Result** (internally):

```yaml
regional_tag_values:
  p1:
    CapRes_1:
      Conventional Steam Coal: 0.85
      Natural Gas Fired Combined Cycle: 0.85
      ...
    CapRes_3:
      Conventional Steam Coal: 0.9
      ...
  p4:
    CapRes_2:
      Conventional Steam Coal: 0.95
      ...
    CapRes_3:
      Conventional Steam Coal: 0.9
      ...
```

## Mixing Automatic and Manual Configuration

You can mix the automatic expansion with manual `regional_tag_values` overrides for specific regions or constraints:

```yaml
capacity_reserve_values:
  Tech1: 0.9
  Tech2: 0.9

regional_capacity_reserves:
  CapRes_1:
    p1: 0.164
    p2: 0.164

# Manual overrides for special cases
regional_tag_values:
  p1:
    CapRes_1:
      Tech1: 0.95  # Different credit for p1
```

**Result** (merged):

- `p1/CapRes_1/Tech1`: `0.95` (explicit override)
- `p1/CapRes_1/Tech2`: `0.9` (auto-expanded)
- `p2/CapRes_1/Tech1`: `0.9` (auto-expanded)
- `p2/CapRes_1/Tech2`: `0.9` (auto-expanded)

Explicit entries always take precedence over auto-generated values, allowing you to override specific cells while benefiting from automation elsewhere.

## Scenario-Specific Values

You can use the scenario management system to vary capacity reserve values across different cases:

```yaml
# Base settings
capacity_reserve_values:
  Coal: 0.9
  Natural Gas: 0.9

settings_management:
  all_years:
    reserve_policy:
      conservative:
        capacity_reserve_values:
          Coal: 0.8      # Lower credit in conservative scenario
          Natural Gas: 0.8
      aggressive:
        capacity_reserve_values:
          Coal: 1.0      # Higher credit in aggressive scenario
          Natural Gas: 1.0
```

Then in your scenario definitions file, set `reserve_policy` to `conservative` or `aggressive` for each case. PowerGenome will automatically expand the scenario-specific `capacity_reserve_values` for each case.

## Technology Name Matching

Technology names in `capacity_reserve_values` must match the technology names in your generator data exactly (case-sensitive). Common technology names in PowerGenome include:

- `Conventional Steam Coal`
- `Natural Gas Fired Combined Cycle`
- `Natural Gas Fired Combustion Turbine`
- `Onshore Wind Turbine`
- `Utility Solar Photovoltaic`
- `Distributed Solar Photovoltaic`
- `Hydroelectric`
- `Nuclear`
- `Battery Storage`

You can also create custom new-build resources with their own names, which will work with `capacity_reserve_values` as long as the names match.

## Tips and Best Practices

1. **Use the flat format for uniform credits**: If all your technologies have the same credit values across all constraints, the flat format is simpler and easier to maintain.

2. **Use the nested format for varying credits**: If technology credits differ by constraint, use the nested format to specify each constraint separately.

3. **Keep constraint names consistent**: Use the naming scheme `CapRes_1`, `CapRes_2`, `CapRes_3`, etc. and stick with it throughout your settings.

4. **Document constraint meanings**: Add comments explaining what each constraint represents:

   ```yaml
   regional_capacity_reserves:
     CapRes_1:  # Primary reserve requirement (MW)
       p1: 0.164
   ```

5. **Validate your values**: Ensure technology credit values are between 0 and 1. Values represent the fraction of installed capacity that counts toward the reserve requirement:
   - `1.0`: Fully counts (e.g., baseload coal, nuclear)
   - `0.5`: Half credit (e.g., wind, which has low capacity factor)
   - `0.0`: Zero credit (doesn't contribute)

6. **Use scenario variations for sensitivity analysis**: Explore how reserve policies affect the optimal mix by creating different cases with varying `capacity_reserve_values`.

## Troubleshooting

**Settings not taking effect?**

- Check that `capacity_reserve_values` and `regional_capacity_reserves` are both present in your settings.
- Verify that technology names match exactly (spaces, capitalization, hyphens).
- Ensure `model_regions` includes all regions referenced in `regional_capacity_reserves`.

**Different constraints getting same credits unexpectedly?**

- This is the intended flat format behavior. Use nested format if constraints should have different credits per technology.

**Explicit `regional_tag_values` being ignored?**

- Remember that explicit entries in `regional_tag_values` take precedence. If you want automatic values, don't specify them in `regional_tag_values`.
