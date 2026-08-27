# How-To Guides

These guides provide step-by-step instructions for specific tasks. Each guide assumes you have completed the [Getting Started](../tutorials/getting-started.md) tutorial.

## Configuration

- **[Configure Settings](configure-settings.md)**: Organize settings files, use modular YAML, and manage complex configurations
- **[Use Year-Keyed Settings Values](use-year-keyed-settings.md)**: Resolve parameter values automatically by planning year across settings files
- **[Configure Renewable Resource Clusters](configure-renewable-clusters.md)**: Set up wind and solar clusters using filters, bins, groups, and agglomerative or k-means clustering
- **[Configure Capacity Reserve Requirements](configure-capacity-reserves.md)**: Specify technology credit values for capacity reserve constraints
- **[Run Multi-Scenario Studies](run-scenarios.md)**: Set up and execute multi-scenario runs with parameter variations
- **[Add Supplemental Hourly Demand](add-supplemental-demand.md)**: Inject additional hourly demand (data-center forecasts, new industrial loads) on top of baseline demand profiles

## Validation and Debugging

- **[Validate Settings Before Running](validate-settings.md)**: Check settings and data for common configuration mistakes before running the pipeline
- **[Debugging](debugging.md)**: Diagnose and fix common errors

## Customization

- **[Add Custom Technologies](add-technologies.md)**: Define user technologies, modify standard resources, and configure cost parameters
- **[Modify Generator Attributes](modify-generator-attributes.md)**: Apply custom formulas to existing generator attributes based on plant age, capacity, or other data fields

## Quick Reference

| Task | Guide | Difficulty |
|------|-------|-----------|
| Split settings into multiple files | [Configure Settings](configure-settings.md) | Easy |
| Set parameter values by planning year | [Use Year-Keyed Settings Values](use-year-keyed-settings.md) | Medium |
| Add a new fuel type | [Add Technologies](add-technologies.md) | Easy |
| Apply age-based O&M costs | [Modify Generator Attributes](modify-generator-attributes.md) | Easy |
| Specify capacity reserve credits | [Configure Capacity Reserves](configure-capacity-reserves.md) | Easy |
| Set up renewable resource clusters | [Configure Renewable Clusters](configure-renewable-clusters.md) | Medium |
| Run sensitivity analysis | [Run Scenarios](run-scenarios.md) | Medium |
| Filter renewable sites by LCOE | [Configure Renewable Clusters](configure-renewable-clusters.md) | Medium |
| Add data-center or industrial load | [Add Supplemental Hourly Demand](add-supplemental-demand.md) | Easy |
| Validate settings before a run | [Validate Settings](validate-settings.md) | Easy |

## Contributing Guides

Found a task that's not covered? Consider contributing a how-to guide! See our [Contributing](../about/contributing.md) page for guidelines.
