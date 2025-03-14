import argparse
import logging
import shutil
import sys
from datetime import datetime as dt
from pathlib import Path

import pandas as pd

import powergenome
from powergenome.external_data import (
    insert_user_tx_costs,
    load_user_tx_costs,
    make_generator_variability,
)
from powergenome.fuels import fuel_cost_table
from powergenome.generators import GeneratorClusters
from powergenome.GenX import (
    add_cap_res_network,
    add_co2_costs_to_o_m,
    add_misc_gen_values,
    check_resource_tags,
    check_vre_profiles,
    create_policy_req,
    create_regional_cap_res,
    fix_min_power_values,
    hydro_energy_to_power,
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
    set_must_run_generation,
)
from powergenome.load_profiles import make_final_load_curves
from powergenome.transmission import (
    agg_transmission_constraints,
    transmission_line_distance,
)
from powergenome.util import (
    build_scenario_settings,
    check_settings,
    init_pudl_connection,
    load_ipm_shapefile,
    load_settings,
    remove_fuel_gen_scenario_name,
    remove_fuel_scenario_name,
    write_case_settings_file,
    write_results_file,
)

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")


def parse_command_line(argv):
    """
    Parse command line arguments. See the -h option.

    :param argv: arguments on the command line must include caller file name.
    """
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
    settings = load_settings(path=args.settings_file)

    # Copy the settings file to results folder
    if Path(args.settings_file).is_file():
        shutil.copy(args.settings_file, out_folder)
    else:
        shutil.copytree(
            args.settings_file, out_folder / "pg_settings", dirs_exist_ok=True
        )

    logger.debug("Initiating PUDL connections")

    pudl_engine, pudl_out, pg_engine = init_pudl_connection(
        freq="AS",
        start_year=min(settings.get("eia_data_years")),
        end_year=max(settings.get("eia_data_years")),
        pudl_db=settings.get("PUDL_DB"),
        pg_db=settings.get("PG_DB"),
    )

    check_settings(settings, pg_engine)

    # Make sure everything in model_regions is either an aggregate region
    # or an IPM region. Will need to change this once we start using non-IPM
    # regions.
    ipm_regions = pd.read_sql_table("regions_entity_epaipm", pg_engine)[
        "region_id_epaipm"
    ]
    all_valid_regions = ipm_regions.tolist() + list(
        settings.get("region_aggregations", {}) or {}
    )
    good_regions = [region in all_valid_regions for region in settings["model_regions"]]

    if not all(good_regions):
        logger.warning(
            "One or more model regions is not valid. Check to make sure all regions "
            "are either in IPM or region_aggregations in the settings YAML file."
        )

    # Sort zones in the settings to make sure they are correctly sorted everywhere.
    settings["model_regions"] = sorted(settings["model_regions"])
    zones = settings["model_regions"]
    logger.info(f"Sorted model regions are {', '.join(zones)}")
    zone_num_map = {
        zone: f"{number + 1}" for zone, number in zip(zones, range(len(zones)))
    }

    input_folder = Path(args.settings_file).parent / Path(settings["input_folder"]).name
    settings["input_folder"] = input_folder

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

    if set(scenario_definitions["year"]) != set(settings["model_year"]):
        logger.warning(
            f"The years included the secenario definitions file ({set(scenario_definitions['year'])}) "
            f"does not match the settings parameter `model_year` ({settings['model_year']})"
        )
    assert len(settings["model_year"]) == len(
        settings["model_first_planning_year"]
    ), "The number of years in the settings parameter 'model_year' must be the same as 'model_first_planning_year'"

    # Build a dictionary of settings for every planning year and case_id
    scenario_settings = build_scenario_settings(settings, scenario_definitions)

    model_regions_gdf = None
    first_year = True
    for year, year_settings in scenario_settings.items():
        for case_id, _settings in year_settings.items():
            case_folder = (
                out_folder
                / f"{case_id}"
                / "Inputs"
                / f"Inputs_p{_settings['case_period']}"
            )
            case_folder.mkdir(parents=True, exist_ok=True)

            _settings["extra_outputs"] = case_folder / "extra_outputs"
            _settings["extra_outputs"].mkdir(parents=True, exist_ok=True)
            logger.info(f"\n\nStarting year {year} scenario {case_id}\n\n")

            case_year_data = {}
            if args.gens:
                gc = GeneratorClusters(
                    pudl_engine=pudl_engine,
                    pudl_out=pudl_out,
                    pg_engine=pg_engine,
                    settings=_settings,
                    current_gens=args.current_gens,
                    sort_gens=args.sort_gens,
                    multi_period=args.multi_period,
                    include_retired_cap=first_year is False,
                )
                gen_data = gc.create_all_generators()
                gen_data["Zone"] = gen_data["region"].map(zone_num_map)
                case_year_data["gen_data"] = gen_data

                gen_variability = make_generator_variability(gen_data)
                gen_variability.index.name = "Time_Index"
                gen_variability.columns = gen_data["Resource"]
                check_vre_profiles(gen_data, gen_variability)

                fuels = fuel_cost_table(
                    fuel_costs=gc.fuel_prices,
                    generators=gc.all_resources,
                    settings=_settings,
                )
                fuels.index.name = "Time_Index"
                fuels = fuels.reset_index(drop=False)
                case_year_data["fuels"] = fuels

            if args.load:
                load = make_final_load_curves(pg_engine=pg_engine, settings=_settings)
                load.columns = "Load_MW_z" + load.columns.map(zone_num_map)
                if not args.gens:
                    gen_variability = pd.DataFrame(index=load.index)

                # reduce_time_domain returns unchanged inputs if the settings parameter
                # "reduce_time_domain" is not set to True.
                (
                    reduced_resource_profile,
                    reduced_load_profile,
                    time_series_mapping,
                    representative_point,
                ) = reduce_time_domain(gen_variability, load, _settings)
                case_year_data["demand_data"] = reduced_load_profile
                reduced_resource_profile.index.name = "Time_Index"
                reduced_resource_profile = reduced_resource_profile.reset_index(
                    drop=False
                )
                case_year_data["gen_variability"] = reduced_resource_profile

                case_year_data["period_map"] = time_series_mapping
                case_year_data["rep_period"] = representative_point

            else:
                gen_variability.index = range(1, len(reduced_resource_profile) + 1)
                gen_variability.index.name = "Time_Index"
                gen_variability = reduced_resource_profile.reset_index(drop=False)
                case_year_data["gen_variability"] = gen_variability

            if args.transmission:
                if args.gens is False:
                    model_regions_gdf = load_ipm_shapefile(_settings)

            if args.transmission:
                if _settings.get("user_transmission_costs"):
                    user_tx_costs = load_user_tx_costs(
                        _settings["input_folder"]
                        / _settings["user_transmission_costs"],
                        _settings["model_regions"],
                        _settings.get("target_usd_year"),
                    )
                    transmission = agg_transmission_constraints(
                        pg_engine=pg_engine, settings=_settings
                    ).pipe(insert_user_tx_costs, user_costs=user_tx_costs)
                else:
                    model_regions_gdf = gc.model_regions_gdf
                    transmission = (
                        agg_transmission_constraints(
                            pg_engine=pg_engine, settings=_settings
                        )
                        .pipe(
                            transmission_line_distance,
                            ipm_shapefile=model_regions_gdf,
                            settings=_settings,
                            units="mile",
                        )
                        .pipe(network_line_loss, settings=_settings)
                        .pipe(network_reinforcement_cost, settings=_settings)
                    )
                transmission = (
                    transmission.pipe(network_max_reinforcement, settings=_settings)
                    .pipe(set_int_cols)
                    .pipe(round_col_values)
                    .pipe(add_cap_res_network, settings=_settings)
                )
                zones = settings["model_regions"]
                network_zones = [f"z{n+1}" for n in range(len(zones))]
                nz_df = pd.Series(data=network_zones, name="Network_zones")
                network = pd.concat([pd.DataFrame(nz_df), transmission], axis=1)
                if args.multi_period:
                    for line in network["Network_Lines"].dropna():
                        network.loc[
                            network["Network_Lines"] == line,
                            "Line_Max_Flow_Possible_MW",
                        ] = 1e6
                        network.loc[
                            network["Network_Lines"] == line, "Capital_Recovery_Period"
                        ] = 60
                        network.loc[network["Network_Lines"] == line, "WACC"] = 0.044
                case_year_data["network"] = network

                if _settings.get("emission_policies_fn"):
                    energy_share_req = create_policy_req(_settings, col_str_match="ESR")
                    co2_cap = create_policy_req(_settings, col_str_match="CO_2")
                    case_year_data["esr"] = energy_share_req
                    case_year_data["co2_cap"] = co2_cap

                min_cap = min_cap_req(_settings)
                case_year_data["min_cap"] = min_cap
                max_cap = max_cap_req(_settings)
                case_year_data["max_cap"] = max_cap

                cap_res = create_regional_cap_res(_settings)
                case_year_data["cap_reserves"] = cap_res

            if _settings.get("reserves_fn"):
                case_year_data["op_reserves"] = pd.read_csv(
                    _settings["input_folder"] / _settings["reserves_fn"]
                )

            if _settings.get("old_genx_format", False) is not True:
                genx_data = process_genx_data(case_folder, case_year_data)
            else:
                genx_data = process_genx_data_old_format(case_folder, case_year_data)

            for data in genx_data:
                if data.dataframe is not None and not data.dataframe.empty:
                    write_results_file(
                        data.dataframe,
                        data.folder,
                        data.file_name,
                    )

            write_case_settings_file(
                settings=_settings,
                folder=case_folder,
                file_name="powergenome_case_settings.yml",
            )
            first_year = False


if __name__ == "__main__":
    main()
