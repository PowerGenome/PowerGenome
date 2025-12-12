# Reference Documentation

Technical reference for PowerGenome parameters, data schemas, and command-line interface.

## Settings Parameters

Comprehensive documentation of all settings parameters organized by functional area:

- **[Model Definition](settings/model-definition.md)**: Planning periods, target year, timezone
- **[Regions and Geography](settings/regions.md)**: Model regions, aggregations, capacity reserves
- **[Existing Generators](settings/existing-generators.md)**: Clustering, retirements, hydro configuration
- **[New Build Resources](settings/new-build.md)**: Technology costs, WACC, regional availability
- **[Fuels](settings/fuels.md)**: Fuel prices, emission factors, CCS configuration
- **[Demand and Load](settings/demand.md)**: Load profiles, growth rates, distributed generation
- **[Transmission](settings/transmission.md)**: Network constraints, expansion costs, line loss
- **[Time Reduction](settings/time-reduction.md)**: Representative period selection and weighting
- **[Resource Tags](settings/resource-tags.md)**: Technology categorization for model behavior
- **[Incentives Settings](settings/incentives-settings.md)**: Define investment and production incentives and their output schemas
- **[Multi-Scenario Management](settings/scenario-management.md)**: Parameter variation across cases

## Data Schemas

Documentation of input and output file formats:

- **[Input Tables](schemas/input-tables.md)**: Required columns and data types for source data
- **[Output Files](schemas/output-files.md)**: GenX CSV file formats and specifications

## Command-Line Interface

- **[CLI Reference](cli.md)**: All command-line flags and options

## Additional Resources

- [Example Systems](https://github.com/PowerGenome/PowerGenome/tree/master/example_systems): Working configuration examples
- [Test System](https://github.com/PowerGenome/PowerGenome/tree/master/tests/test_system): Minimal test configuration
- [Settings Documentation](https://github.com/PowerGenome/PowerGenome/blob/master/example_systems/settings_documentation.md): Original comprehensive reference

## API Reference

!!! note "API Documentation Pending"
    Comprehensive Python API documentation will be added once module docstrings are enhanced.
    In the meantime, refer to the source code on [GitHub](https://github.com/PowerGenome/PowerGenome/tree/master/powergenome).

## Using This Reference

- **Finding Parameters**: Use the search function (top of page) or browse by category
- **Data Types**: All parameters show expected data type (string, list, dict, etc.)
- **Examples**: Most parameters include YAML examples
- **Required vs Optional**: Required parameters are clearly marked
- **Defaults**: Default values are shown when applicable
