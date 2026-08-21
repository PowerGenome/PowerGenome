import argparse
import logging
import shutil
import sys
from datetime import datetime as dt
from pathlib import Path

import pandas as pd

import powergenome
from powergenome.database import initialize_data_manager, update_data_manager
from powergenome.external_data import make_generator_variability
from powergenome.fuels import fuel_cost_table
from powergenome.generators import GeneratorClusters
from powergenome.GenX import (  # add_co2_costs_to_o_m,; add_misc_gen_values,; check_resource_tags,; fix_min_power_values,; hydro_energy_to_power,; set_must_run_generation,
    add_cap_res_network,
    check_vre_profiles,
    create_policy_req,
    create_regional_cap_res,
    max_cap_req,
    min_cap_req,
    network_line_loss,
    network_max_reinforcement,
    network_reinforcement_cost,
    process_genx_data,
    process_genx_data_old_format,
    reduce_time_domain,
    round_col_values,
    set_int_cols,
)
from powergenome.load_profiles import make_final_load_curves
from powergenome.macro_inputs import MacroCaseBuilder
from powergenome.settings import (
    Settings,
    build_scenario_settings,
    resolve_settings_to_year,
)
from powergenome.transmission import (
    agg_transmission_constraints,
    insert_tx_costs,
    load_tx_costs,
)
from powergenome.util import (  # init_pudl_connection,; check_settings,; load_ipm_shapefile,; remove_fuel_gen_scenario_name,; remove_fuel_scenario_name,
    get_first_planning_years_from_settings,
    get_model_years_from_settings,
    write_case_settings_file,
    write_results_file,
)
from powergenome.validate import (
    _extract_planning_periods,
    report_validation_results,
    validate_settings,
    validate_settings_with_data,
)

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")


def _as_bool(value, default=False):
    """Coerce a settings value to a boolean, accepting case-insensitive forms.

    YAML booleans arrive as Python ``bool``; also handle string forms such as
    ``"true"``, ``"TRUE"``, ``"yes"``, ``"on"``, ``"1"`` (and their negatives).
    ``None`` (the key is absent) falls back to ``default``.
    """
    if value is None:
        return default
    if isinstance(value, (bool, int)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def resolve_output_formats(args, settings):
    """Return ``(macro_enabled, genx_enabled)`` for a case.

    CLI flags and per-case settings are combined so both formats can be written
    in a single run. GenX is the default output when no output flag or output
    setting is present; Macro is written in addition to GenX unless GenX is
    explicitly disabled with ``genx_output: false``.
    """
    macro_enabled = getattr(args, "macro", False) or _as_bool(
        settings.get("macro_output"), default=False
    )
    genx_enabled = getattr(args, "genx", False) or _as_bool(
        settings.get("genx_output"), default=True
    )
    return macro_enabled, genx_enabled


def parse_command_line(argv):
    """
    Parse command line arguments. See the -h option.

    :param argv: arguments on the command line must include caller file name.
    """
    # Accept long option names in any case (e.g. --MACRO, --Genx). Option values
    # are left untouched.
    argv = [
        arg if not arg.startswith("--") else arg[:2] + arg[2:].lower()
        for arg in argv
    ]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-sf",
        "--settings_file",
        dest="settings_file",
        type=str,
        default="example_settings.yml",
        help="Specify a YAML settings file.",
    )
    parser.add_argument(
        "-rf",
        "--results_folder",
        dest="results_folder",
        type=str,
        default=dt.now().strftime("%Y-%m-%d %H.%M.%S"),
        help="Specify the results subfolder to write output",
    )
    parser.add_argument(
        "--no-current-gens",
        dest="current_gens",
        action="store_false",
        help="Don't load and cluster current generators.",
    )
    parser.add_argument(
        "--no-gens",
        dest="gens",
        action="store_false",
        help="Use flag to not calculate generator clusters.",
    )
    parser.add_argument(
        "--no-load",
        dest="load",
        action="store_false",
        help="Calculate hourly load. If False, file will not be written.",
    )
    parser.add_argument(
        "--no-transmission",
        dest="transmission",
        action="store_false",
        help="Calculate transmission constraints. If False, file will not be written.",
    )
    parser.add_argument(
        "-s",
        "--sort-gens",
        dest="sort_gens",
        action="store_true",
        help=(
            "Sort generators alphabetically within region. Existing resources will "
            "still be separate from new resources."
        ),
    )
    parser.add_argument(
        "-c",
        "--case-id",
        dest="case_id",
        nargs="*",
        help=(
            "One or more case IDs to select from the scenario inputs file. Only these "
            "cases will be used."
        ),
    )
    parser.add_argument(
        "-mp",
        "--multi-period",
        dest="multi_period",
        action="store_false",
        help=("Use multi-period output format."),
    )
    parser.add_argument(
        "--macro",
        dest="macro",
        action="store_true",
        help=(
            "Write MacroEnergy.jl simpleCSVinputs-format case inputs in addition "
            "to the default GenX output (single run writes both). Can also be "
            "enabled with 'macro_output: true' in a settings file. Set "
            "'genx_output: false' to write Macro inputs only."
        ),
    )
    parser.add_argument(
        "--genx",
        dest="genx",
        action="store_true",
        help=(
            "Write GenX Inputs files (the default output when no output flag or "
            "output setting is set). Can be combined with --macro to write both "
            "formats in a single run. Can also be controlled with "
            "'genx_output: true' in a settings file."
        ),
    )
    arguments = parser.parse_args(argv[1:])
    return arguments


def main(**kwargs):
    args = parse_command_line(sys.argv)
    args.__dict__.update(kwargs)
    cwd = Path.cwd()

    out_folder = cwd / args.results_folder
    out_folder.mkdir(exist_ok=True)

    # Create a logger to output any messages we might have...
    logger = logging.getLogger(powergenome.__name__)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    stream_formatter = logging.Formatter(
        # More extensive test-like formatter...
        "%(asctime)s [%(levelname)8s] %(name)s:%(lineno)s %(message)s",
        # This is the datetime format string.
        "%H:%M:%S",
    )
    handler.setFormatter(stream_formatter)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    file_formatter = logging.Formatter(
        # More extensive test-like formatter...
        "%(asctime)s [%(levelname)8s] %(name)s:%(lineno)s %(message)s",
        # This is the datetime format string.
        "%Y-%m-%d %H:%M:%S",
    )
    filehandler = logging.FileHandler(out_folder / "log.txt")
    filehandler.setLevel(logging.DEBUG)
    filehandler.setFormatter(file_formatter)
    logger.addHandler(filehandler)

    if not args.multi_period:
        logger.info(
            "As of version 0.6.2 the --multi-period/-mp flag can be used to format inputs "
            "for multi-stage modeling in GenX."
        )

    logger.info("Reading settings file")
    settings = Settings(config_path=args.settings_file)

    logger.info("Running Phase 1 settings validation")
    report_validation_results(validate_settings(settings))

    initialize_data_manager(settings, settings["data_location"])

    logger.info("Running Phase 2 data validation")
    from powergenome.database import _data_manager

    report_validation_results(validate_settings_with_data(settings, _data_manager))

    # Copy the settings file to results folder
    if Path(args.settings_file).is_file():
        shutil.copy(args.settings_file, out_folder)
    else:
        shutil.copytree(
            args.settings_file, out_folder / "pg_settings", dirs_exist_ok=True
        )

    logger.debug("Initiating PUDL connections")

    # pudl_engine, pudl_out, pg_engine = init_pudl_connection(
    #     freq="AS",
    #     start_year=min(settings.get("eia_data_years")),
    #     end_year=max(settings.get("eia_data_years")),
    #     pudl_db=settings.get("PUDL_DB"),
    #     pg_db=settings.get("PG_DB"),
    # )

    # check_settings(settings, pg_engine)

    # Make sure everything in model_regions is either an aggregate region
    # or an IPM region. Will need to change this once we start using non-IPM
    # regions.
    # ipm_regions = pd.read_sql_table("regions_entity_epaipm", pg_engine)[
    #     "region_id_epaipm"
    # ]
    # all_valid_regions = ipm_regions.tolist() + list(
    #     settings.get("region_aggregations", {}) or {}
    # )
    # good_regions = [region in all_valid_regions for region in settings["model_regions"]]

    # if not all(good_regions):
    #     logger.warning(
    #         "One or more model regions is not valid. Check to make sure all regions "
    #         "are either in IPM or region_aggregations in the settings YAML file."
    #     )

    input_folder = Path(settings["input_folder"])

    model_years = get_model_years_from_settings(
        settings.get("model_year"), settings.get("model_periods")
    )
    if not isinstance(model_years, list):
        model_years = [model_years]

    has_scenario_definitions = bool(settings.get("scenario_definitions_fn"))
    settings_dict = settings.to_dict()
    planning_periods = _extract_planning_periods(settings_dict)
    if planning_periods:
        model_years = [period[1] for period in planning_periods]

    if has_scenario_definitions:
        scenario_definitions = pd.read_csv(
            input_folder / settings["scenario_definitions_fn"]
        )

        if args.case_id:
            missing_case_ids = set(args.case_id) - set(scenario_definitions["case_id"])
            if missing_case_ids:
                raise ValueError(
                    f"The requested case IDs {missing_case_ids} are not in your scenario "
                    "inputs file."
                )
            scenario_definitions = scenario_definitions.loc[
                scenario_definitions["case_id"].isin(args.case_id), :
            ]

        if set(scenario_definitions["year"]) != set(model_years):
            logger.warning(
                f"The years included in the scenario definitions file ({set(scenario_definitions['year'])}) "
                f"do not match the configured model planning years ({set(model_years)})"
            )
    else:
        logger.info(
            "No 'scenario_definitions_fn' found in settings. Resolving settings "
            "for each planning year from 'model_year' or 'model_periods' settings parameter."
        )
        if args.case_id:
            logger.warning(
                "The --case-id flag is ignored when no 'scenario_definitions_fn' is "
                "specified in settings."
            )
        scenario_settings = {}
        for period, year in enumerate(model_years, start=1):
            year_settings = resolve_settings_to_year(
                settings, year, all_years=model_years
            )
            year_settings["case_id"] = "Inputs"
            year_settings["case_period"] = period
            scenario_settings[year] = {"Inputs": year_settings}

    num_model_years = len(model_years) if isinstance(model_years, list) else 1
    first_planning_years = get_first_planning_years_from_settings(
        settings.get("model_first_planning_year"), settings.get("model_periods")
    )
    num_first_planning_years = (
        len(first_planning_years) if isinstance(first_planning_years, list) else 1
    )
    assert (
        num_model_years == num_first_planning_years
    ), "The number of years in the settings parameter 'model_year' must be the same as 'model_first_planning_year'"

    if has_scenario_definitions:
        # Build a dictionary of settings for every planning year and scenario
        scenario_settings = build_scenario_settings(settings, scenario_definitions)

    model_regions_gdf = None
    first_year = True
    # Macro cases are written one per case (not per period), with each planning
    # period becoming a stage. Because the loop below iterates years first, the
    # stages of a given case are not contiguous, so buffer them here and
    # finalize (write shared case-level files) once the loop completes.
    macro_writers = {}
    for year, year_settings in scenario_settings.items():
        for case_id, _settings in year_settings.items():
            # Use the factory method to create scenario-specific settings
            # This ensures each scenario has independent settings based on the base configuration
            with Settings.for_scenario(settings, _settings) as scenario_settings_obj:

                # Update DataManager for this specific case
                update_data_manager(updated_settings=scenario_settings_obj)

                if has_scenario_definitions:
                    case_folder = (
                        out_folder
                        / f"{case_id}"
                        / "Inputs"
                        / f"Inputs_p{scenario_settings_obj['case_period']}"
                    )
                else:
                    case_folder = (
                        out_folder
                        / "Inputs"
                        / f"Inputs_p{scenario_settings_obj['case_period']}"
                    )
                case_folder.mkdir(parents=True, exist_ok=True)

                scenario_settings_obj["extra_outputs"] = case_folder / "extra_outputs"
                scenario_settings_obj["extra_outputs"].mkdir(
                    parents=True, exist_ok=True
                )
                if has_scenario_definitions:
                    logger.info(f"\n\nStarting year {year} scenario {case_id}\n\n")
                else:
                    logger.info(f"\n\nStarting year {year}\n\n")

                case_year_data = {}
                if args.gens:
                    gc = GeneratorClusters(
                        current_gens=args.current_gens,
                        sort_gens=args.sort_gens,
                        multi_period=args.multi_period,
                        include_retired_cap=first_year is False,
                    )
                    gen_data = gc.create_all_generators()
                    gen_data["Zone"] = gen_data["region"].map(
                        scenario_settings_obj["zone_num_map"]
                    )
                    case_year_data["gen_data"] = gen_data

                    gen_variability = make_generator_variability(gen_data)
                    gen_variability.index.name = "Time_Index"
                    gen_variability.columns = gen_data["Resource"]
                    check_vre_profiles(gen_data, gen_variability)

                    fuels = fuel_cost_table(
                        fuel_costs=gc.fuel_prices,
                        generators=gc.all_resources,
                        num_hours=len(gen_variability),
                    )
                    fuels.index.name = "Time_Index"
                    fuels = fuels.reset_index(drop=False)
                    case_year_data["fuels"] = fuels

                if args.load:
                    load = make_final_load_curves()
                    load.columns = "Demand_MW_z" + load.columns.map(
                        scenario_settings_obj["zone_num_map"]
                    )
                    if not args.gens:
                        gen_variability = pd.DataFrame(index=load.index)

                    # reduce_time_domain returns unchanged inputs if the settings parameter
                    # "reduce_time_domain" is not set to True.
                    (
                        reduced_resource_profile,
                        reduced_load_profile,
                        time_series_mapping,
                        representative_point,
                    ) = reduce_time_domain(gen_variability, load)
                    case_year_data["demand_data"] = reduced_load_profile
                    reduced_resource_profile.index.name = "Time_Index"
                    reduced_resource_profile = reduced_resource_profile.reset_index(
                        drop=False
                    )
                    case_year_data["gen_variability"] = reduced_resource_profile

                    case_year_data["period_map"] = time_series_mapping
                    case_year_data["rep_period"] = representative_point

                else:
                    if args.gens:
                        # gens were computed but load was skipped — store raw
                        # gen_variability without time domain reduction
                        gen_variability.index = range(1, len(gen_variability) + 1)
                        gen_variability.index.name = "Time_Index"
                        gen_variability = gen_variability.reset_index(drop=False)
                        case_year_data["gen_variability"] = gen_variability

                if args.transmission:
                    tx_costs = load_tx_costs()

                    transmission = agg_transmission_constraints(
                        tx_value_col=scenario_settings_obj.get(
                            "tx_value_col", "firm_ttc_mw"
                        ),
                    ).pipe(insert_tx_costs, tx_costs=tx_costs)

                    network = (
                        transmission.pipe(network_max_reinforcement)
                        .pipe(set_int_cols)
                        .pipe(round_col_values)
                        .pipe(add_cap_res_network)
                    )
                    if args.multi_period:
                        for line in network["Network_Lines"].dropna():
                            network.loc[
                                network["Network_Lines"] == line,
                                "Line_Max_Flow_Possible_MW",
                            ] = 1e6
                            network.loc[
                                network["Network_Lines"] == line,
                                "Capital_Recovery_Period",
                            ] = 60
                            network.loc[network["Network_Lines"] == line, "WACC"] = (
                                0.044
                            )
                    case_year_data["network"] = network

                    if scenario_settings_obj.get("emission_policies_fn"):
                        energy_share_req = create_policy_req(
                            col_str_match="ESR",
                        )
                        co2_cap = create_policy_req(col_str_match="CO_2")
                        case_year_data["esr"] = energy_share_req
                        case_year_data["co2_cap"] = co2_cap

                    min_cap = min_cap_req()
                    case_year_data["min_cap"] = min_cap
                    max_cap = max_cap_req()
                    case_year_data["max_cap"] = max_cap

                    cap_res = create_regional_cap_res()
                    case_year_data["cap_reserves"] = cap_res

                if scenario_settings_obj.get("reserves_fn"):
                    case_year_data["op_reserves"] = pd.read_csv(
                        scenario_settings_obj["input_folder"]
                        / scenario_settings_obj["reserves_fn"]
                    )

                macro_output_enabled, genx_output_enabled = resolve_output_formats(
                    args, scenario_settings_obj
                )

                if macro_output_enabled:
                    logger.info(
                        "\n\nWriting Macro simpleCSVinputs format to %s\n\n",
                        case_folder,
                    )
                    # The Macro case root is the parent of the GenX-style Inputs
                    # folder: <out>/<case_id> for scenario definitions and
                    # <out> for an un-keyed case.
                    macro_root = case_folder.parent.parent
                    writer = macro_writers.get(macro_root)
                    if writer is None:
                        writer = MacroCaseBuilder(macro_root)
                        macro_writers[macro_root] = writer
                    writer.add_stage(
                        scenario_settings_obj["case_period"],
                        case_year_data,
                        scenario_settings_obj,
                    )

                if genx_output_enabled:
                    if (
                        scenario_settings_obj.get("old_genx_format", False)
                        is not True
                    ):
                        genx_data = process_genx_data(case_folder, case_year_data)
                    else:
                        genx_data = process_genx_data_old_format(
                            case_folder, case_year_data
                        )
                    for data in genx_data:
                        if data.dataframe is not None and not data.dataframe.empty:
                            write_results_file(
                                data.dataframe,
                                data.folder,
                                data.file_name,
                            )
                write_case_settings_file(
                    settings=scenario_settings_obj.to_dict(),
                    folder=case_folder,
                    file_name="powergenome_case_settings.yml",
                )
                first_year = False

    # Finalize buffered Macro cases (writes per-stage files and the shared
    # case-level system_data.json / case_settings.json).
    for writer in macro_writers.values():
        writer.finalize()


if __name__ == "__main__":
    main()
