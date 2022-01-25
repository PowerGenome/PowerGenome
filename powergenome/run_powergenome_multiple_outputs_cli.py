import argparse
import copy
import logging
import shutil
import sys
from datetime import datetime as dt
from pathlib import Path

import pandas as pd

import powergenome
from powergenome.fuels import fuel_cost_table
from powergenome.generators import (
    GeneratorClusters,
    load_ipm_shapefile,
    add_fuel_labels,
    add_genx_model_tags,
)
from powergenome.GenX import (
    add_cap_res_network,
    create_policy_req,
    create_regional_cap_res,
    fix_min_power_values,
    min_cap_req,
    reduce_time_domain,
    add_misc_gen_values,
    network_line_loss,
    network_max_reinforcement,
    network_reinforcement_cost,
    round_col_values,
    set_int_cols,
    calculate_partial_CES_values,
)
from powergenome.load_profiles import make_final_load_curves
from powergenome.transmission import (
    agg_transmission_constraints,
    transmission_line_distance,
)
from powergenome.nrelatb import atb_fixed_var_om_existing
from powergenome.external_data import make_generator_variability
from powergenome.util import (
    build_scenario_settings,
    check_settings,
    init_pudl_connection,
    load_settings,
    remove_fuel_scenario_name,
    remove_fuel_gen_scenario_name,
    update_dictionary,
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
        "-f",
        "--no-fuel",
        dest="fuel",
        action="store_false",
        help=(
            "Create fuel table. If False, file will not be written."
            " Can not be created without the generators."
        ),
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
    arguments = parser.parse_args(argv[1:])
    return arguments


def main():

    args = parse_command_line(sys.argv)
    cwd = Path.cwd()

    out_folder = cwd / args.results_folder
    out_folder.mkdir(exist_ok=True)

    # Create a logger to output any messages we might have...
    logger = logging.getLogger(powergenome.__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        # More extensive test-like formatter...
        "%(asctime)s [%(levelname)8s] %(name)s:%(lineno)s %(message)s",
        # This is the datetime format string.
        "%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    filehandler = logging.FileHandler(out_folder / "log.txt")
    filehandler.setFormatter(formatter)
    logger.addHandler(filehandler)

    logger.info("Reading settings file")
    settings = load_settings(path=args.settings_file)

    # Copy the settings file to results folder
    shutil.copy(args.settings_file, out_folder)

    logger.info("Initiating PUDL connections")

    pudl_engine, pudl_out, pg_engine = init_pudl_connection(
        freq="AS",
        start_year=min(settings.get("data_years")),
        end_year=max(settings.get("data_years")),
    )

    check_settings(settings, pg_engine)

    # Make sure everything in model_regions is either an aggregate region
    # or an IPM region. Will need to change this once we start using non-IPM
    # regions.
    ipm_regions = pd.read_sql_table("regions_entity_epaipm", pg_engine)[
        "region_id_epaipm"
    ]
    all_valid_regions = ipm_regions.tolist() + list(
        settings.get("region_aggregations", {})
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
    logger.info(f"Sorted zones are {', '.join(zones)}")
    zone_num_map = {
        zone: f"{number + 1}" for zone, number in zip(zones, range(len(zones)))
    }

    input_folder = cwd / settings["input_folder"]
    settings["input_folder"] = input_folder

    scenario_definitions = pd.read_csv(
        input_folder / settings["scenario_definitions_fn"]
    )

    assert set(scenario_definitions["year"]) == set(
        settings["model_year"]
    ), "The years included the secenario definitions file must match the settings parameter `model_year`"
    assert len(settings["model_year"]) == len(
        settings["model_first_planning_year"]
    ), "The number of years in the settings parameter 'model_year' must be the same as 'model_first_planning_year'"

    # Build a dictionary of settings for every planning year and case_id
    scenario_settings = build_scenario_settings(settings, scenario_definitions)

    i = 0
    model_regions_gdf = None
    for year in scenario_settings:
        for case_id, _settings in scenario_settings[year].items():
            case_folder = (
                out_folder / f"{year}" / f"{case_id}_{year}_{_settings['case_name']}"
            )

            if i == 0:
                if args.gens:
                    gc = GeneratorClusters(
                        pudl_engine=pudl_engine,
                        pudl_out=pudl_out,
                        pg_engine=pg_engine,
                        settings=_settings,
                        current_gens=args.current_gens,
                        sort_gens=args.sort_gens,
                    )
                    gen_clusters = gc.create_all_generators()
                    if args.fuel and args.gens:
                        fuels = fuel_cost_table(
                            fuel_costs=gc.fuel_prices,
                            generators=gc.all_resources,
                            settings=_settings,
                        )
                        # fuels["fuel_indices"] = range(1, len(fuels) + 1)
                        # fuels = remove_fuel_scenario_name(fuels, _settings)
                        fuels.index.name = "Time_Index"
                        write_results_file(
                            df=remove_fuel_scenario_name(fuels, _settings),
                            folder=case_folder,
                            file_name="Fuels_data.csv",
                            include_index=True,
                        )

                    # gen_clusters = remove_fuel_scenario_name(gen_clusters, _settings)
                    gen_clusters["Zone"] = gen_clusters["region"].map(zone_num_map)
                    gen_clusters = add_misc_gen_values(gen_clusters, _settings)
                    # gen_clusters = set_int_cols(gen_clusters)
                    # gen_clusters = gen_clusters.fillna(value=0)

                    # Save existing resources that aren't demand response for use in
                    # other cases
                    existing_gens = gc.existing_resources.copy()
                    # gen_clusters.loc[
                    #     (gen_clusters["Existing_Cap_MW"] >= 0)
                    #     & (gen_clusters["DR"] == 0),
                    #     :,
                    # ]
                    logger.info(
                        f"Finished first round with year {year} scenario {case_id}"
                    )
                    # if settings.get("partial_ces"):
                    gen_variability = make_generator_variability(gen_clusters)
                    gen_variability.index.name = "Time_Index"
                    gen_variability.columns = (
                        gen_clusters["region"]
                        + "_"
                        + gen_clusters["Resource"]
                        + "_"
                        + gen_clusters["cluster"].astype(str)
                    )
                    gens = calculate_partial_CES_values(
                        gen_clusters, fuels, _settings
                    ).pipe(fix_min_power_values, gen_variability)
                    for col in _settings["generator_columns"]:
                        if col not in gens.columns:
                            gens[col] = 0
                    cols = [c for c in _settings["generator_columns"] if c in gens]

                    write_results_file(
                        df=remove_fuel_gen_scenario_name(
                            gens[cols].fillna(0), _settings
                        )
                        .pipe(set_int_cols)
                        .pipe(round_col_values),
                        folder=case_folder,
                        file_name="Generators_data.csv",
                        include_index=False,
                    )
                    # else:
                    #     write_results_file(
                    #         df=gen_clusters.fillna(0),
                    #         folder=case_folder,
                    #         file_name="Generators_data.csv",
                    #         include_index=False,
                    #     )

                    # write_results_file(
                    #     df=gen_variability,
                    #     folder=case_folder,
                    #     file_name="Generators_variability.csv",
                    #     include_index=True,
                    # )

                    i += 1
                if args.transmission:
                    if args.gens is False:
                        model_regions_gdf = load_ipm_shapefile(_settings)

                # genx_settings = make_genx_settings_file(pudl_engine, _settings)
                # write_case_settings_file(
                #     settings=genx_settings,
                #     folder=case_folder,
                #     file_name="GenX_settings.yml",
                # )

            else:
                logger.info(f"\nStarting year {year} scenario {case_id}")
                if args.gens:

                    gc.settings = _settings
                    # gc.current_gens = False

                    # Change the fuel labels in existing generators to reflect the
                    # correct AEO scenario for each fuel and update GenX tags based
                    # on settings.
                    # gc.existing_resources = existing_gens.pipe(
                    #     add_fuel_labels, gc.fuel_prices, _settings
                    # ).pipe(add_genx_model_tags, _settings)

                    gen_clusters = gc.create_all_generators()
                    # if settings.get("partial_ces"):
                    #     fuels = fuel_cost_table(
                    #         fuel_costs=gc.fuel_prices,
                    #         generators=gc.all_resources,
                    #         settings=_settings,
                    #     )
                    #     gen_clusters = calculate_partial_CES_values(
                    #         gen_clusters, fuels, _settings
                    #     )

                    gen_clusters = add_misc_gen_values(gen_clusters, _settings)
                    gen_clusters = set_int_cols(gen_clusters)
                    # gen_clusters = gen_clusters.fillna(value=0)

                    # gen_clusters = remove_fuel_scenario_name(gen_clusters, _settings)
                    gen_clusters["Zone"] = gen_clusters["region"].map(zone_num_map)

                    fuels = fuel_cost_table(
                        fuel_costs=gc.fuel_prices,
                        generators=gc.all_resources,
                        settings=_settings,
                    )
                    gen_variability = make_generator_variability(gen_clusters)
                    gen_variability.index.name = "Time_Index"
                    gen_variability.columns = (
                        gen_clusters["region"]
                        + "_"
                        + gen_clusters["Resource"]
                        + "_"
                        + gen_clusters["cluster"].astype(str)
                    )
                    gens = calculate_partial_CES_values(
                        gen_clusters, fuels, _settings
                    ).pipe(fix_min_power_values, gen_variability)
                    cols = [c for c in _settings["generator_columns"] if c in gens]
                    write_results_file(
                        df=remove_fuel_gen_scenario_name(
                            gens[cols].fillna(0), _settings
                        )
                        .pipe(set_int_cols)
                        .pipe(round_col_values),
                        folder=case_folder,
                        file_name="Generators_data.csv",
                        include_index=False,
                    )
                    # write_results_file(
                    #     df=gen_clusters.fillna(0),
                    #     folder=case_folder,
                    #     file_name="Generators_data.csv",
                    # )

                    # write_results_file(
                    #     df=gen_variability,
                    #     folder=case_folder,
                    #     file_name="Generators_variability.csv",
                    #     include_index=True,
                    # )

            if args.load:
                load = make_final_load_curves(pg_engine=pg_engine, settings=_settings)
                load.columns = "Load_MW_z" + load.columns.map(zone_num_map)

                (
                    reduced_resource_profile,
                    reduced_load_profile,
                    time_series_mapping,
                ) = reduce_time_domain(gen_variability, load, _settings)
                reduced_resource_profile.index.name = "Time_Index"
                write_results_file(
                    df=reduced_load_profile,
                    folder=case_folder,
                    file_name="Load_data.csv",
                    include_index=False,
                )
                write_results_file(
                    df=reduced_resource_profile,
                    folder=case_folder,
                    file_name="Generators_variability.csv",
                    include_index=True,
                    float_format="%.3f",
                )
                if time_series_mapping is not None:
                    write_results_file(
                        df=time_series_mapping,
                        folder=case_folder,
                        file_name="Period_map.csv",
                        include_index=False,
                    )

            if args.transmission:
                # if not model_regions_gdf:
                #     if args.gens is False:
                #         model_regions_gdf = load_ipm_shapefile(_settings)
                #     else:
                #         model_regions_gdf = gc.model_regions_gdf
                # transmission = agg_transmission_constraints(
                #     pudl_engine=pudl_engine, settings=_settings
                # )
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
                    .pipe(network_max_reinforcement, settings=_settings)
                    .pipe(network_reinforcement_cost, settings=_settings)
                    .pipe(set_int_cols)
                    .pipe(round_col_values)
                    .pipe(add_cap_res_network, settings=_settings)
                )
                zones = settings["model_regions"]
                network_zones = [f"z{n+1}" for n in range(len(zones))]
                nz_df = pd.Series(data=network_zones, name="Network_zones")
                network = pd.concat([pd.DataFrame(nz_df), transmission], axis=1)
                # network["Network_zones"] = network_zones

                if _settings.get("emission_policies_fn"):
                    # network = add_emission_policies(transmission, _settings)
                    energy_share_req = create_policy_req(_settings, col_str_match="ESR")
                    co2_cap = create_policy_req(_settings, col_str_match="CO_2")
                min_cap = min_cap_req(_settings)

                cap_res = create_regional_cap_res(_settings)

                write_results_file(
                    df=network,
                    folder=case_folder,
                    file_name="Network.csv",
                    include_index=False,
                )
                if energy_share_req is not None:
                    write_results_file(
                        df=energy_share_req,
                        folder=case_folder,
                        file_name="Energy_share_requirement.csv",
                        include_index=False,
                    )
                if cap_res is not None:
                    write_results_file(
                        df=cap_res,
                        folder=case_folder,
                        file_name="Capacity_reserve_margin.csv",
                        include_index=True,
                    )
                if co2_cap is not None:
                    co2_cap = co2_cap.set_index("Region_description")
                    co2_cap.index.name = None
                    write_results_file(
                        df=co2_cap,
                        folder=case_folder,
                        file_name="CO2_cap.csv",
                        include_index=True,
                    )
                if min_cap is not None:
                    write_results_file(
                        df=min_cap,
                        folder=case_folder,
                        file_name="Minimum_capacity_requirement.csv",
                        include_index=False,
                    )

            if args.fuel and args.gens:
                fuels = fuel_cost_table(
                    fuel_costs=gc.fuel_prices,
                    generators=gc.all_resources,
                    settings=_settings,
                )
                # fuels = remove_fuel_scenario_name(fuels, _settings)

                # Hack to get around the fact that fuels with different cost names
                # get added and end up as duplicates.
                # fuels = fuels.drop_duplicates(subset=["Fuel"], keep="last")
                # fuels["fuel_indices"] = range(1, len(fuels) + 1)
                fuels.index.name = "Time_Index"
                write_results_file(
                    df=remove_fuel_scenario_name(fuels, _settings)
                    .pipe(set_int_cols)
                    .pipe(round_col_values),
                    folder=case_folder,
                    file_name="Fuels_data.csv",
                    include_index=True,
                )
            if _settings.get("reserves_fn"):
                shutil.copy(
                    _settings["input_folder"] / _settings["reserves_fn"],
                    case_folder / "Inputs",
                )

            if _settings.get("genx_settings_fn"):
                shutil.copy(cwd / _settings["genx_settings_fn"], case_folder / "Inputs")

            if _settings.get("genx_settings_folder"):
                genx_settings_folder = case_folder / "Settings"
                genx_settings_folder.mkdir(exist_ok=True)
                for f in (cwd / _settings.get("genx_settings_folder")).glob("*.yml"):
                    shutil.copy(f, genx_settings_folder)

            write_case_settings_file(
                settings=_settings,
                folder=case_folder,
                file_name="powergenome_case_settings.yml",
            )


if __name__ == "__main__":
    main()
