# Modify Generator Attributes with Custom Formulas

This guide shows how to apply custom formulas to existing generator attributes, such as increasing fixed O&M costs based on plant age.

## Overview

The `resource_attr_modifiers` parameter allows you to define custom formulas that modify generator attributes based on other data fields. This is useful for:

- **Age-based cost adjustments**: Increase O&M costs as plants get older
- **Capacity-based modifications**: Scale costs with unit size
- **Custom adjustments**: Combine a rate and a column from the generator data

## Basic Usage

### Structure

```yaml
resource_attr_modifiers:
  <technology_name>:
    - attribute: <column_to_modify>
      formula:
        op: <operation>  # "add", "mul", "sub", "truediv", or "replace"
        rate: <numeric_value>
        multiplier: <column_name>
```

**Parameters**:

- `<technology_name>`: Technology to modify (case-insensitive substring match)
- `attribute`: Column in generator dataframe to modify
- `formula.op`: Operation to apply (see below)
- `formula.rate`: Numeric rate/coefficient
- `formula.multiplier`: Column name containing values to multiply by rate

### Operations

| Operation | Behavior | Formula |
|-----------|----------|---------|
| `add` | Add to existing value | `new_value = old_value + (rate × multiplier)` |
| `mul` | Multiply existing value | `new_value = old_value × (rate × multiplier)` |
| `sub` | Subtract from existing value | `new_value = old_value - (rate × multiplier)` |
| `truediv` | Divide existing value | `new_value = old_value / (rate × multiplier)` |
| `replace` | Replace existing value | `new_value = rate × multiplier` |

## Examples

### Increase O&M with Plant Age

Increase fixed O&M costs by $126/MW-yr for each year of plant age:

```yaml
resource_attr_modifiers:
  conventional steam coal:
    - attribute: fom_per_mwyr
      formula:
        op: add
        rate: 126  # $/MW-yr per year of age
        multiplier: age
```

**How it works**:

1. Matches all generators with "conventional steam coal" in technology name
2. For each matched generator: `fom_per_mwyr = fom_per_mwyr + (126 × age)`
3. Example: 20-year-old plant adds $2,520/MW-yr to base O&M

### Multiple Attributes for One Technology

Apply multiple formulas to the same technology:

```yaml
resource_attr_modifiers:
  natural gas fired combined cycle:
    - attribute: fom_per_mwyr
      formula:
        op: add
        rate: 50  # $/MW-yr per year
        multiplier: age
    - attribute: heat_rate_mmbtu_mwh
      formula:
        op: add
        rate: 0.005  # MMBtu/MWh per year of age
        multiplier: age
```

This increases both fixed O&M and heat rate as plants age.

### Replace Instead of Add

Set a value based entirely on a formula (ignoring existing value):

```yaml
resource_attr_modifiers:
  biomass:
    - attribute: fom_per_mwyr
      formula:
        op: replace
        rate: 100  # Illustrative coefficient
        multiplier: capacity_mw
```

With `replace`, the original `fom_per_mwyr` value is ignored and replaced by
`100 × capacity_mw`. Check units and input columns for your own formula.

### Multiple Technologies

Modify different technologies with different formulas:

```yaml
resource_attr_modifiers:
  conventional steam coal:
    - attribute: fom_per_mwyr
      formula:
        op: add
        rate: 126
        multiplier: age
  natural gas:
    - attribute: fom_per_mwyr
      formula:
        op: add
        rate: 75
        multiplier: age
  nuclear:
    - attribute: fom_per_mwyr
      formula:
        op: add
        rate: 200
        multiplier: age
```

Each technology gets its own age-based adjustment rate.

## Available Attributes

Common attributes you can modify:

| Attribute | Description | Units |
|-----------|-------------|-------|
| `fom_per_mwyr` | Fixed O&M cost | $/MW-yr |
| `vom_per_mwh` | Variable O&M cost | $/MWh |
| `heat_rate_mmbtu_mwh` | Heat rate | MMBtu/MWh |
| `minimum_load_mw` | Minimum load | MW |

Available multiplier columns depend on your data schema. Common ones:

- `age`: Plant age in years
- `capacity_mw`: Nameplate capacity
- `operating_year`: Year plant started operation
- Custom columns from your data tables

!!! warning "Column Requirements"
    Both the `attribute` and `multiplier` columns must exist in the generator dataframe. If either is missing, PowerGenome will raise a `KeyError`.

## Technology Matching

Technology names are matched **case-insensitively** using **substring matching**:

```yaml
resource_attr_modifiers:
  coal:  # Matches "Conventional Steam Coal", "Coal IGCC", etc.
    - attribute: fom_per_mwyr
      formula:
        op: add
        rate: 126
        multiplier: age
```

**Matching rules**:

- Case-insensitive: `"Coal"` matches `"conventional steam coal"`
- Substring: `"gas"` matches `"Natural Gas Fired Combined Cycle"`
- Underscores ignored, but spaces are not: `"natural_gas"` matches
  `"NaturalGas"`, not `"Natural Gas"`
- Keys are regular expressions; escape regex metacharacters when matching them literally.
- An unmatched technology does not raise an error; it changes no rows.

Be specific enough to avoid unintended matches. For example:

- ✅ `"conventional steam coal"`: Matches only coal steam plants
- ⚠️ `"coal"`: Matches all coal technologies (steam, IGCC, etc.)
- ❌ `"steam"`: Matches coal steam AND gas steam turbines

## Execution Order

`resource_attr_modifiers` is applied to individual rows immediately after
loading the generation table and calculating `age` as the model year minus
`operating_year`. It runs before region mapping and clustering, so later
cluster-averaged values and region mapping columns are not available.

## Validation

PowerGenome validates formulas at runtime:

**Missing attribute**:

```yaml
resource_attr_modifiers:
  coal:
    - attribute: nonexistent_column  # ❌ KeyError
      formula:
        op: add
        rate: 100
        multiplier: age
```

**Missing multiplier**:

```yaml
resource_attr_modifiers:
  coal:
    - attribute: fom_per_mwyr
      formula:
        op: add
        rate: 100
        multiplier: nonexistent_column  # ❌ KeyError
```

**Invalid operation**:

```yaml
resource_attr_modifiers:
  coal:
    - attribute: fom_per_mwyr
      formula:
        op: multiply  # ❌ Should be "mul"
        rate: 2
        multiplier: age
```

**Missing keys**:

```yaml
resource_attr_modifiers:
  coal:
    - formula:  # ❌ Missing "attribute" key
        op: add
        rate: 100
        multiplier: age
```

All errors raise exceptions during pipeline execution.

## Complete Example

Full configuration applying age-based degradation to multiple technologies:

```yaml
# Existing generator settings
num_clusters:
  Conventional Steam Coal: 3
  Natural Gas Fired Combined Cycle: 5
  Nuclear: 1

tech_groups:
  Combined Cycle:
    - Natural Gas Fired Combined Cycle
    - Natural Gas Steam Turbine

# Age-based cost adjustments
resource_attr_modifiers:
  conventional steam coal:
    - attribute: fom_per_mwyr
      formula:
        op: add
        rate: 126  # $/MW-yr per year of age
        multiplier: age
    - attribute: heat_rate_mmbtu_mwh
      formula:
        op: add
        rate: 0.003  # MMBtu/MWh per year of age
        multiplier: age

  natural gas fired combined cycle:
    - attribute: fom_per_mwyr
      formula:
        op: add
        rate: 75
        multiplier: age
    - attribute: heat_rate_mmbtu_mwh
      formula:
        op: add
        rate: 0.002  # MMBtu/MWh per year of age
        multiplier: age

  nuclear:
    - attribute: fom_per_mwyr
      formula:
        op: add
        rate: 200  # Higher aging costs for nuclear
        multiplier: age
```

**Result**: Older plants have higher O&M costs and worse heat rates, reflecting maintenance needs and efficiency degradation over time.

## Debugging

To verify formula application:

1. **Check generator output**: Examine the relevant `resources/*.csv` file for modified values (`Generators_data.csv` with `old_genx_format: true`)
2. **Compare with/without modifiers**: Run with `resource_attr_modifiers` commented out
3. **Validate formulas**: Test with simple cases before complex scenarios

Example validation:

```python
import pandas as pd
from powergenome.generators import apply_custom_gen_formula

gens = pd.DataFrame({
    "technology": ["Conventional Steam Coal"],
    "age": [20],
    "fom_per_mwyr": [10000],
})
modifiers = {
    "coal": [{
        "attribute": "fom_per_mwyr",
        "formula": {"op": "add", "rate": 126, "multiplier": "age"},
    }],
}
print(apply_custom_gen_formula(gens, modifiers)["fom_per_mwyr"])
```

The result is `12520` $/MW-yr.

## Related Settings

- [Existing Generators](../reference/settings/existing-generators.md): Clustering and technology grouping
- [Resource Modifiers](../reference/settings/new-build.md#resource_modifiers): Modify **new-build** resource costs
- [Regional Cost Factors](../reference/settings/regions.md#cost-multipliers): Regional cost adjustments
- [Model Tag Values](../reference/settings/resource-tags.md#model_tag_values): Case-insensitive substring matching for technology names

## See Also

- [Add Custom Technologies](add-technologies.md): Create new technology definitions
- [Configure Settings](configure-settings.md): YAML syntax and structure
- [Existing Generators Reference](../reference/settings/existing-generators.md): Complete parameter documentation
