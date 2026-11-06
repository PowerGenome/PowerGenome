# Changelog

For release-by-release changes, see the repository's
[full changelog](https://github.com/PowerGenome/PowerGenome/blob/main/CHANGELOG.md)
and [GitHub Releases](https://github.com/PowerGenome/PowerGenome/releases).
This page highlights changes that affect study configuration; check the release
you have installed before migrating an existing study.

## Configuration migration notes

| Area | Current workflow | Where to check |
| --- | --- | --- |
| Data sources | Configure CSV/Parquet or database tables through DataManager; table dictionaries support filters and column selection. | [Data Tables Settings](../reference/settings/data-tables.md) |
| Renewable profiles | Prefer tidy `site_id,time_index,value` profiles, with `weather_year` when selecting weather years. Legacy wide profiles load only through the unfiltered compatibility path with a warning. | [Configure Renewable Clusters](../how-to/configure-renewable-clusters.md) |
| Distributed generation | Use separate capacity and profile tables. Missing-year capacity can be interpolated/extrapolated, but partial exact-year regional coverage is not filled automatically. | [Distributed Generation](../explanation/distributed-generation.md) |
| Fuel prices | Supply all constituent base regions and a `data_year` column; aggregate-region prices are averaged from base regions. Legacy AEO mapping parameters remain available when needed. | [Fuel Settings](../reference/settings/fuels.md) |
| Interconnection costs | Prefer direct `interconnect_capex_mw`. The legacy mileage path applies when direct costs are absent; annuities use resource financial parameters. | [New-build Settings](../reference/settings/new-build.md#interconnection-costs) |
| Regional cost factors | Mappings may be inferred from matching source names; explicit maps run from cost-table labels to model regions/resource identifiers. | [New-build Settings](../reference/settings/new-build.md#cost-multipliers) |
| GenX outputs | Default outputs split resource types into `resources/` and system inputs into `system/`. Use `old_genx_format: true` only when the downstream model needs legacy flat files. | [Output File Schemas](../reference/schemas/output-files.md) |

These summaries are not replacement settings folders. Follow the linked
reference, validate the migrated study, and compare generated inputs with the
previous version before interpreting model results.

## Developer-facing changes

Transmission investment-cost helpers now take explicit arguments and use WACC
and capital-recovery years from resource DataFrame columns. Profile I/O and DG
processing use centralized data access. Consult the full changelog and relevant
source/tests before updating Python integrations.

## Contributing

Found an undocumented change? See [Contributing](contributing.md).
