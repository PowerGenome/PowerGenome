# Explanation

These articles provide background information on how PowerGenome works and why it's designed the way it is.

## Architecture and Design

- **[Architecture Overview](architecture.md)**: High-level system design, key components, and design patterns
- **[Data Pipeline Flow](pipeline.md)**: How data moves through PowerGenome from input to output

## Methodology

- **[Generator Clustering](clustering.md)**: K-means clustering methodology for existing power plants
- **[Cost Calculations](costs.md)**: Technology costs, regional adjustments, and annualization
- **[Time Reduction](time-reduction.md)**: Representative period selection and its impact on accuracy

## Data Management

- **[Regional Mappings](regions.md)**: How base regions map to model regions and cost/fuel regions
- **[Distributed Generation](distributed-generation.md)**: DG capacity and profile handling (includes recent refactor)

## Model Integration

- **[Output Format](output-format.md)**: GenX file structure, resource tagging, and policy constraints

## Understanding vs. Doing

These articles explain **why** and **how** things work, not step-by-step instructions. If you're looking for:

- **How to configure something**: See [How-To Guides](../how-to/index.md)
- **Parameter definitions**: See [Reference](../reference/index.md)
- **Learning PowerGenome**: See [Tutorials](../tutorials/index.md)

## Going Deeper

Want to understand PowerGenome internals?

1. Start with [Architecture Overview](architecture.md) for the big picture
2. Read [Data Pipeline Flow](pipeline.md) to understand execution
3. Dive into specific methodology articles based on your interests
4. Review the [source code](https://github.com/PowerGenome/PowerGenome/tree/master/powergenome) for implementation details

## Contributing Explanations

These articles benefit from diverse perspectives. If you have insights about:

- Trade-offs in design decisions
- Alternative approaches and why they weren't chosen
- Common misconceptions
- Historical context for features

Consider contributing! See our [Contributing](../about/contributing.md) guidelines.
