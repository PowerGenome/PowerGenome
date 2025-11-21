# PowerGenome Documentation

Welcome to the PowerGenome documentation! PowerGenome is a data pipeline tool that generates input files for capacity expansion models, primarily GenX.

## What is PowerGenome?

PowerGenome simplifies the process of creating power system model inputs by transforming raw energy data (from sources like EIA and NREL) into model-ready datasets. Instead of manually assembling data for each region and scenario, you define your requirements in settings files and PowerGenome generates all necessary inputs.

!!! info "International Use"
    While PowerGenome's examples reference US data sources (EIA, NREL ATB, PUDL), the tool is **region-agnostic**. Users can supply custom data tables representing any geographic region worldwide. The data pipeline works with any properly-structured input data—European grids, Asian power systems, Latin American networks, etc.

## Key Features

- **Existing Generator Clustering**: Aggregate thousands of power plants into manageable clusters while preserving key characteristics
- **New Build Resources**: Integration with NREL Annual Technology Baseline (ATB) for future technology costs
- **Renewable Resource Groups**: Pre-clustered wind and solar resources with generation profiles
- **Transmission Constraints**: Inter-regional transmission limits and expansion costs
- **Demand Profiles**: Hourly load profiles with electrification scenarios
- **Time Reduction**: Representative period selection to reduce computational complexity
- **Multi-Scenario Management**: Run sensitivity analyses across different assumptions
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

## Quick Start

```bash
# Install from PyPI
pip install powergenome

# Or for development
git clone https://github.com/PowerGenome/PowerGenome.git
cd PowerGenome
pip install -e ".[dev]"

# Run with example settings
run_powergenome --settings_file settings --results_folder output
```

## Get Help

- **Issues**: Report bugs or request features on [GitHub Issues](https://github.com/PowerGenome/PowerGenome/issues)
- **Discussions**: Join the community on [groups.io](https://groups.io/g/powergenome)
- **Source Code**: Browse the code on [GitHub](https://github.com/PowerGenome/PowerGenome)

## Citation

If you use PowerGenome in your research, please cite:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4426097.svg)](https://doi.org/10.5281/zenodo.4426096)

## License

PowerGenome is released under the MIT License. See the [License](about/license.md) page for details.
