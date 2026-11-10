# Tutorials

Start with a complete small study, then change one part at a time. These lessons
use the published Texas three-zone study from
[PowerGenome-examples](https://github.com/PowerGenome/PowerGenome-examples).
Inspect actual outputs without assembling a research dataset or cloning
PowerGenome's source. PowerGenome prepares inputs; it does not solve the model.

## Learning path

### 1. [Getting Started](getting-started.md)

Install a published release with uv, download a pinned examples snapshot,
validate the Texas study, and generate inputs for 2030 and 2045. Learn where
settings, data, logs, and GenX outputs live.

**Prerequisites**: uv, curl, tar, a terminal, and basic command-line familiarity.

### 2. [Building a Multi-Region Model](multi-region-model.md)

Continue with the downloaded Texas study. Inspect its base-to-model-region
aggregation, change transmission expansion limits, and verify both periods'
network outputs against the first run.

**Prerequisites**: Getting Started and its unchanged `tx_output/` baseline.

### 3. [Working with Renewable Resources](renewable-resources.md)

Aggregate the Texas example's wind candidates into cost-and-profile clusters
and compare resource counts, capacity limits, and generation profiles while
preserving solar. Follow the separate-settings setup to isolate the change.

**Prerequisites**: The installed environment and downloaded examples from Getting
Started. This lesson can be taken independently of the transmission exercise.

## Move to your own study

The example has short synthetic profiles and can emit warnings; it demonstrates
the workflow, not realistic system economics. Browse
[PowerGenome-examples](https://github.com/PowerGenome/PowerGenome-examples) for
other published cases. A real study needs complete compatible sources
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
