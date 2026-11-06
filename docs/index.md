# PowerGenome Documentation

Welcome to the PowerGenome documentation! PowerGenome is a data pipeline tool that generates input files for capacity expansion models, primarily [GenX](https://genxproject.github.io/GenX.jl/), with optional output for [MacroEnergy.jl](https://github.com/macroenergy/MacroEnergy.jl) (see [Output Formats](reference/settings/macro-output.md)).

!!! tip "First time user or setting up a new system?"
    Use the **[PowerGenome System Design tool](https://gschivley.github.io/PowerGenome-tools/web/)** to generate a starting settings folder for your study area. The interactive tool walks you through selecting regions, planning periods, technologies, and policies, then produces a starting settings folder you can download and customise. You must also obtain the data files referenced by those settings. See the [System Design tool documentation](https://gschivley.github.io/PowerGenome-tools/) for a step-by-step walkthrough.

## What is PowerGenome?

PowerGenome simplifies the process of creating power system model inputs by transforming raw energy system data (from sources like EIA and NREL) into model-ready datasets. Instead of manually assembling data for each region and scenario, you define your requirements in settings files and PowerGenome generates all necessary inputs.

!!! info "Other geographies"
    Custom regions and datasets can be used, but some processing conventions reflect US sources such as EIA and NREL ATB. Check the required schemas and technology mappings when adapting a study; changing region labels alone is not enough.

## Key Features

- **Existing Generator Clustering**: Aggregate thousands of power plants into manageable clusters while preserving key characteristics
- **New Build Resources**: Integration with NREL Annual Technology Baseline (ATB) for future technology costs
- **Renewable Resource Groups**: Pre-clustered wind and solar resources with generation profiles
- **Transmission Constraints**: Inter-regional transmission limits and expansion costs
- **Demand Profiles**: Hourly load profiles with electrification scenarios
- **Time Reduction**: Representative period selection to reduce computational complexity
- **Multi-Scenario Management**: Run sensitivity analyses across different assumptions
- **Year-Keyed Settings Values**: Define parameter values by planning year across all settings sections
- **Flexible Data Sources**: DataManager architecture supports CSV, Parquet, and DuckDB

## Documentation Structure

This documentation follows the [Diataxis](https://diataxis.fr/) framework:

### [Tutorials](tutorials/index.md)

Step-by-step lessons to learn PowerGenome fundamentals. Start here if you're new to PowerGenome.

### [How-To Guides](how-to/index.md)

Practical guides for specific tasks. Use these when you know what you want to accomplish.

### [Reference](reference/index.md)

Technical descriptions of settings parameters, data schemas, and the command-line interface.

### [Explanation](explanation/index.md)

Background information on how PowerGenome works and why it's designed the way it is.

## Start Here

1. **Run a small, complete example**: Follow [Getting Started](tutorials/getting-started.md)
   to install with Python 3.11+ and run the bundled test system.
2. **Define your own study**: Generate a settings folder with the
   [System Design tool](https://gschivley.github.io/PowerGenome-tools/web/), then
   [configure the input tables](how-to/configure-data-tables.md). Settings alone
   are not a dataset: the referenced data files must also be available locally.
3. **Check before a full run**: Use [Validate Settings](how-to/validate-settings.md),
   then run `run_powergenome -sf settings -rf output` from your study directory.

PowerGenome **prepares model inputs**; it does not solve the capacity expansion
model. Run GenX or Macro separately after inspecting those inputs.

## Get Help

- **Issues**: Report bugs or request features on [GitHub Issues](https://github.com/PowerGenome/PowerGenome/issues)
- **Discussions**: Join the community on [groups.io](https://groups.io/g/powergenome)
- **Source Code**: Browse the code on [GitHub](https://github.com/PowerGenome/PowerGenome)

## Citation

If you use PowerGenome in your research, please cite:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4426097.svg)](https://doi.org/10.5281/zenodo.4426096)

## License

PowerGenome is released under the MIT License. See the [License](about/license.md) page for details.
