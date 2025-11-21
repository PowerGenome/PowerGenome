# Tutorials

Welcome to the PowerGenome tutorials! These step-by-step guides will teach you how to use PowerGenome effectively.

## Learning Path

We recommend completing these tutorials in order:

### 1. [Getting Started](getting-started.md)

Install PowerGenome, understand the basic workflow, and run your first model. You'll learn:

- How to install PowerGenome
- The structure of a PowerGenome project
- How to run the command-line interface
- How to examine the output files

**Time**: ~30 minutes
**Prerequisites**: Basic command-line knowledge

### 2. [Building a Multi-Region Model](multi-region-model.md)

Create a model with multiple regions and transmission constraints. You'll learn:

- How to define model regions
- How to aggregate base regions
- How to configure transmission between regions
- How to set regional parameters

**Time**: ~45 minutes
**Prerequisites**: Complete "Getting Started"

### 3. [Working with Renewable Resources](renewable-resources.md)

Configure wind and solar resources with generation profiles. You'll learn:

- How renewable resource groups work
- How to filter and cluster renewable sites
- How to configure interconnection costs
- How to customize resource characteristics by region

**Time**: ~45 minutes
**Prerequisites**: Complete "Getting Started"

## Before You Begin

Make sure you have:

- Python 3.8 or higher installed
- Basic familiarity with YAML file format
- A text editor or IDE for editing configuration files
- Enough disk space for data files (~5-10 GB depending on scope)

## Data Requirements

PowerGenome uses data from multiple sources. The tutorials use small test datasets, but for real studies you'll need:

- Generator data (from EIA via PUDL)
- Technology cost data (NREL ATB)
- Generation profiles for renewables
- Demand time series
- Transmission network data

See the [Data Tables](../how-to/data-tables.md) guide for details on obtaining and configuring data sources.

## Need Help?

If you get stuck:

1. Check the [Debugging](../how-to/debugging.md) guide for common issues
2. Review the [Reference](../reference/index.md) documentation for parameter details
3. Ask questions on [groups.io](https://groups.io/g/powergenome)
4. Open an issue on [GitHub](https://github.com/PowerGenome/PowerGenome/issues)
