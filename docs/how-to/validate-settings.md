# Validate Settings Before Running

Run validation from your study directory before generating inputs:

```bash
validate_powergenome --settings_file settings
```

`run_powergenome` also validates automatically. Standalone validation helps you
find configuration problems without waiting for generator clustering and output
writing.

!!! warning "Validation is not exhaustive"
    Passing validation does not establish that inputs are complete, scientifically
    appropriate, or ready for optimization. Review the logs and generated files,
    especially for a new study or dataset.

## Validate a known-good configuration first

To confirm the validator itself works, run it against a published example from the
root of a clone of
[PowerGenome-examples](https://github.com/PowerGenome/PowerGenome-examples) rather
than from inside the example system folder. Check out the same pinned snapshot the
[Getting Started](../tutorials/getting-started.md#1-download-the-texas-example)
tutorial uses, so the exercise keeps validating the configuration it was written
for even as the default branch moves on:

```bash
git clone https://github.com/PowerGenome/PowerGenome-examples.git
cd PowerGenome-examples
git checkout 7e34f714469e68fca173cdb849f74c47be779420
validate_powergenome --settings_file example_systems/TX-3-zone/settings
```

Remove nothing and change nothing: these settings are expected to load and pass
the settings phase. If this run reports errors, check your installation and the
data download before debugging your own settings. Swap in
`example_systems/CA_AZ/settings` to exercise a two-region setup with two capacity
reserve constraints.

## Understand the result

- **Errors** stop the pipeline. Fix them before running again.
- **Warnings** allow the pipeline to continue, but can identify consequential
  problems such as missing fuel prices or technology costs becoming zero.

The standalone command groups issues by severity and category. The pipeline
writes them to the terminal and `<results_folder>/log.txt`. Use each issue's
`Detail` values to locate the affected regions, fuels, resources, or years.

| Reported issue | What to check |
| --- | --- |
| Missing required key | Add the key with the correct type; see [Model Definition](../reference/settings/model-definition.md). |
| Inconsistent planning years | Use valid `model_periods`, or equally sized `model_year` and `model_first_planning_year` lists. |
| Missing path or table | Check path resolution, exact setting keys, and unique source filenames; see [Configure Data Tables](configure-data-tables.md). |
| Unknown region | Compare spelling and base/model-region roles in settings and tables. |
| Missing fuel-price coverage | Supply every required fuel, scenario, base region, and year; check `fuel_scenarios` against the table. |
| Missing new-resource cost years | Provide cost rows whose `basis_year` overlaps each planning window. |
| Missing model tags | Ensure each new resource matches the appropriate `model_tag_values`; inspect the written resource files too. |

Do not suppress a warning merely to obtain a successful exit code. For example,
missing fuel prices may become `$0/MMBtu`, materially changing a model's choices.

## Choose the scope

```bash
# Settings-only checks, without loading source tables
validate_powergenome --settings_file settings --skip-data-checks

# Force fresh settings and data checks
validate_powergenome --settings_file settings --no-validation-cache
```

| Flag | Purpose |
| --- | --- |
| `--settings_file PATH` | Settings folder or single YAML file; required. |
| `--skip-data-checks` | Check settings structure and consistency only. |
| `--no-validation-cache` | Ignore cached results; alias `--force-validation`. |
| `--no-fail` | Report errors but exit with code 0. Use only when your automation handles the reported failures separately. |

## What is checked

Validation has two phases:

1. **Settings**: required keys, planning-year shapes, paths, region-keyed settings,
   fuel definitions, and new-resource tag coverage.
2. **Settings against data**: configured tables, regional coverage, transmission
   region names, fuel-price coverage, and new-resource cost-year coverage.

In `run_powergenome`, Phase 1 errors stop execution before DataManager
initialization; Phase 2 errors also stop the run. The standalone validator can
attempt Phase 2 even if Phase 1 reports errors. A table-loading exception can
therefore still require fixing the settings before a complete report is possible.

## Skipping validation when nothing changed

Results are cached under `<input_folder>/validation_cache/` when `input_folder`
is configured. Unchanged settings, validation code, and source fingerprints allow
checks to be replayed rather than recomputed. Cached warnings still print;
cached errors still fail.

Use `--no-validation-cache` on either CLI whenever you need a fresh check. To
turn caching off for a study, set:

```yaml
use_validation_cache: false
```

## Next steps

- [Debugging](debugging.md): investigate runtime failures and unexpected outputs.
- [CLI Reference](../reference/cli.md): validation and input-generation options.
- [Input Table Schemas](../reference/schemas/input-tables.md): required fields.
