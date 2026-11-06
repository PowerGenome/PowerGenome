# Tutorials

Start with a complete small study, then change one part at a time. These lessons
use bundled test data so you can inspect actual outputs without first assembling
a research dataset. PowerGenome prepares inputs; it does not solve the model.

## Learning path

### 1. [Getting Started](getting-started.md)

Install with uv, copy the bundled study outside the source checkout, validate it,
and generate inputs for two planning periods. Learn where settings, data, logs,
and GenX outputs live.

**Prerequisites**: Git, uv, a terminal, and basic command-line familiarity.

### 2. [Building a Multi-Region Model](multi-region-model.md)

Continue in the copied study. Inspect its base-to-model-region aggregation,
change transmission expansion limits, and verify both periods' network outputs
against the first run.

**Prerequisites**: Getting Started and its unchanged `test_output/` baseline.

### 3. [Working with Renewable Resources](renewable-resources.md)

Use the bundled wind data to change renewable resource resolution and compare
resource counts, capacity limits, and generation profiles. Follow the page's
isolated-copy setup so previous tutorial changes do not affect the comparison.

**Prerequisites**: The installed environment and bundled source data from Getting
Started. This lesson can be taken independently of the transmission exercise.

## Move to your own study

The fixture has short profiles and known warnings; it demonstrates the workflow,
not realistic system economics. A real study needs complete compatible sources
for existing plants, technology costs, renewable profiles, demand, fuels,
transmission, and any configured policies.

Use the [System Design tool](https://gschivley.github.io/PowerGenome-tools/web/)
to create starting settings, then [Configure Data Tables](../how-to/configure-data-tables.md)
and [validate the study](../how-to/validate-settings.md). Changing region names
or planning years alone does not supply the corresponding data.

## Need help?

- [Debugging](../how-to/debugging.md): first checks and common failures.
- [Reference](../reference/index.md): exact parameters and schemas.
- [groups.io](https://groups.io/g/powergenome): community questions.
- [GitHub Issues](https://github.com/PowerGenome/PowerGenome/issues): bug reports.
