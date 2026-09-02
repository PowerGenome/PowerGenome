# Macro Output Settings

PowerGenome can write model-case inputs for two capacity expansion models:
[GenX](https://github.com/GenXProject/GenX.jl) (the default) and
[MacroEnergy.jl](https://github.com/macroenergy/MacroEnergy.jl) (Macro), in Macro's `simpleCSVinputs` format.
This page documents the settings that control which format(s) are written.

GenX is written by default. Enabling Macro output adds the Macro `simpleCSVinputs` files **in addition to** (not
instead of) the GenX inputs, so a single `run_powergenome` call can produce both formats at once. Writing both
formats in one run reuses all the intermediate data processing, which is faster than running PowerGenome once per
model. To write only Macro inputs, disable GenX (see [`genx_output`](#genx_output)).

The same options are available as command-line flags to `run_powergenome`:
[`--macro`](../cli.md#macro), [`--genx`](../cli.md#genx), and [`--no-genx`](../cli.md#no-genx). Flags and settings
values are combined per format (either one enables that format), and all flags and boolean settings are
case-insensitive (`--Macro`, `"TRUE"`, etc.).

## `macro_output`

**Type**: bool
**Default**: `false`
**CLI equivalent**: `--macro`

When `true`, write a Macro `simpleCSVinputs` case (only the simpleCSV inputs format — not the Macro JSON format)
in a `macro_out/`-style folder, matching the structure of the Macro examples (e.g.
`macroenergy/MacroEnergyExamples.jl/examples/multisector_3zone_simpleCSVinputs`). The semantic mapping of
generators, transmission, demand, fuel supply, and CO2 caps follows the GenX-to-Macro converter
(`EmilDimanchev/GenX_to_Macro`; multistage runs follow the `lb/multistage` branch). Cross-sector assets
(hydrogen, liquid fuels, CCS) are not yet emitted.

GenX output is **not** disabled by this setting — see [`genx_output`](#genx_output).

## `genx_output`

**Type**: bool
**Default**: `true`
**CLI equivalents**: `--genx` (enable), `--no-genx` (disable)

Controls whether PowerGenome writes the standard GenX `Inputs/Inputs_pN` files. GenX is the default output
format, so this key is only needed to turn GenX **off** (for example, to write Macro inputs only) or to state it
explicitly.

- `genx_output: false` with `macro_output: true` → Macro inputs only.
- `genx_output: true` with `macro_output: true` → both formats in one run.
- `--genx` forces GenX output on, and `--no-genx` forces it off — each flag overrides the settings value. Because
  `--no-genx` is a hard override, `run_powergenome --macro --no-genx` writes Macro inputs only without editing
  any settings file.

```yaml
# Macro inputs only (settings-file approach)
macro_output: true
genx_output: false
```

## Macro model-run settings

The following settings configure the Macro output beyond format selection. They are respected only when Macro
output is enabled; existing behavior is unchanged when these keys are absent.

| Setting | Type | Default | Writes to | Description |
|---|---|---|---|---|
| `macro_discount_rate` | float | `0.045` | `settings/case_settings.json` | Annual discount rate applied across stages. |
| `macro_solution_algorithm` | str | `"Monolithic"` | `settings/case_settings.json` | Solution algorithm for Macro. |
| `macro_period_lengths` | list[int] | derived | `settings/case_settings.json` | `PeriodLengths`, one entry per stage. Derived from PowerGenome's planning-period definitions (`model_periods`, or the paired `model_first_planning_year`/`model_year`); an explicit value overrides the derived one. |
| `macro_constraint_scaling` | bool | `true` | `settings/macro_settings.json` | Scale constraints for solver numerics. |
| `macro_write_subcommodities` | bool | `true` | `settings/macro_settings.json` | Write sub-commodity accounting rows. |
| `macro_auto_create_nodes` | bool | `false` | `settings/macro_settings.json` | Let Macro automatically create missing nodes. |
| `macro_auto_create_locations` | bool | `true` | `settings/macro_settings.json` | Let Macro automatically create missing locations. |
| `macro_default_max_nsd` | float | `1.0` | NSD fallback | Default non-served-demand quantity when no `demand_segments_voll.csv` is provided. |
| `macro_default_voll` | float | `10000` | NSD fallback | Default value of lost load ($/MWh) when no `demand_segments_voll.csv` is provided. |
| `macro_default_fuel_price` | float | `0.0` | fuel price fallback | Fallback fuel price when a fuel is missing from the fuel price table. |

See [Output File Format](../../explanation/output-format.md) for the resulting Macro `simpleCSVinputs` folder
layout, and the [CLI Reference](../cli.md) for the command-line flags.
