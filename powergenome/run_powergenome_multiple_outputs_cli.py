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
    add_emission_policies,
    make_genx_settings_file,
    reduce_time_domain,
    add_misc_gen_values,
    network_line_loss,
    network_max_reinforcement,
    network_reinforcement_cost,
    set_int_cols,
    calculate_partial_CES_values,
    calc_emissions_ces_level,
)
from powergenome.load_profiles import make_final_load_curves
from powergenome.transmission import (
    agg_transmission_constraints,
    transmission_line_distance,
)
from powergenome.external_data import make_generator_variability
from powergenome.util import (
    init_pudl_connection,
    load_settings,
    update_dictionary,
    remove_fuel_scenario_name,
    write_results_file,
    write_case_settings_file,
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


def build_case_id_name_map(settings):
    case_id_name_df = pd.read_csv(
        Path(settings["input_folder"]) / settings["case_id_description_fn"],
        index_col=0,
        squeeze=True,
    )
    case_id_name_df = case_id_name_df.str.replace(" ", "_")
    case_id_name_map = case_id_name_df.to_dict()

    return case_id_name_map


def build_scenario_settings(settings, scenario_definitions):
    """Build a nested dictionary of settings for each planning year/scenario

    Parameters
    ----------
    settings : dict
        The full settings file, including the "settings_management" section with
        alternate values for each scenario
    scenario_definitions : DataFrame
        Values from the csv file defined in the settings file "scenario_definitions_fn"
        parameter. This df has columns corresponding to categories in the
        "settings_management" section of the settings file, with row values defining
        specific case/scenario names.

    Returns
    -------
    dict
        A nested dictionary. The first set of keys are the planning years, the second
        set of keys are the case ID values associated with each case.
    """

    model_planning_period_dict = {
        year: (start_year, year)
        for year, start_year in zip(
            settings["model_year"], settings["model_first_planning_year"]
        )
    }

    case_id_name_map = build_case_id_name_map(settings)

    scenario_settings = {}
    for year in scenario_definitions["year"].unique():
        scenario_settings[year] = {}
        planning_year_settings_management = settings["settings_management"][year]

        # Create a dictionary with keys of things that change (e.g. ccs_capex) and
        # values of nested dictionaries that give case_id: scenario name
        planning_year_scenario_definitions_dict = (
            scenario_definitions.loc[scenario_definitions.year == year]
            .set_index("case_id")
            .to_dict()
        )
        planning_year_scenario_definitions_dict.pop("year")

        for case_id in scenario_definitions["case_id"].unique():
            _settings = copy.deepcopy(settings)

            if "all_cases" in planning_year_settings_management:
                new_parameter = planning_year_settings_management["all_cases"]
                _settings = update_dictionary(_settings, new_parameter)

            # Add the scenario definition values to the settings files
            # e.g.
            # case_id	year	demand_response	growth	tx_expansion	ng_price
            # p1	    2030	moderate	            normal	high	reference
            case_scenario_definitions = scenario_definitions.loc[
                (scenario_definitions.case_id == case_id)
                & (scenario_definitions.year == year),
                :,
            ]
            for col in scenario_definitions.columns:
                _settings[col] = case_scenario_definitions.squeeze().at[col]

            modified_settings = []
            for (
                category,
                case_value_dict,
            ) in planning_year_scenario_definitions_dict.items():
                # key is the category e.g. ccs_capex, case_value_dict is p1: mid
                try:
                    case_value = case_value_dict[case_id]
                    new_parameter = planning_year_settings_management[category][
                        case_value
                    ]
                    # print(new_parameter)
                    try:
                        settings_keys = list(new_parameter.keys())
                    except AttributeError:
                        settings_keys = {}

                    for key in settings_keys:
                        assert (
                            key not in modified_settings
                        ), f"The settings key {key} is modified twice in case id {case_id}"

                        modified_settings.append(key)

                    if new_parameter is not None:
                        _settings = update_dictionary(_settings, new_parameter)
                    # print(_settings[list(new_parameter.keys())[0]])

                except KeyError:
                    pass

            _settings["model_first_planning_year"] = model_planning_period_dict[year][0]
            _settings["model_year"] = model_planning_period_dict[year][1]
            _settings["case_name"] = case_id_name_map[case_id]
            scenario_settings[year][case_id] = _settings

    return scenario_settings


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
    pudl_engine, pudl_out = init_pudl_connection(freq="YS")

    # Make sure everything in model_regions is either an aggregate region
    # or an IPM region. Will need to change this once we start using non-IPM
    # regions.
    ipm_regions = pd.read_sql_table("regions_entity_epaipm", pudl_engine)[
        "region_id_epaipm"
    ]
    all_valid_regions = ipm_regions.tolist() + list(settings["region_aggregations"])
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
                        fuels["fuel_indices"] = range(1, len(fuels) + 1)
                        # fuels = remove_fuel_scenario_name(fuels, _settings)
                        write_results_file(
                            df=remove_fuel_scenario_name(fuels, _settings),
                            folder=case_folder,
                            file_name="Fuels_data.csv",
                        )

                    # gen_clusters = remove_fuel_scenario_name(gen_clusters, _settings)
                    gen_clusters["zone"] = gen_clusters["region"].map(zone_num_map)
                    gen_clusters = add_misc_gen_values(gen_clusters, _settings)
                    gen_clusters = set_int_cols(gen_clusters)
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
                    gens = calculate_partial_CES_values(gen_clusters, fuels, _settings)
                    cols = [c for c in _settings["generator_columns"] if c in gens]
                    write_results_file(
                        df=remove_fuel_scenario_name(gens[cols].fillna(0), _settings),
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

                    gen_variability = make_generator_variability(
                        gen_clusters, _settings
                    )
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
                    else:
                        model_regions_gdf = gc.model_regions_gdf
                        transmission = agg_transmission_constraints(
                            pudl_engine=pudl_engine, settings=_settings
                        )
                        transmission = (
                            transmission.pipe(
                                transmission_line_distance,
                                ipm_shapefile=model_regions_gdf,
                                settings=_settings,
                                units="mile",
                            )
                            .pipe(network_line_loss, settings=_settings)
                            .pipe(network_max_reinforcement, settings=_settings)
                            .pipe(network_reinforcement_cost, settings=_settings)
                        )

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
                    gc.current_gens = False

                    # Change the fuel labels in existing generators to reflect the
                    # correct AEO scenario for each fuel and update GenX tags based
                    # on settings.
                    gc.existing_resources = existing_gens.pipe(
                        add_fuel_labels, gc.fuel_prices, _settings
                    ).pipe(add_genx_model_tags, _settings)

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
                    gen_clusters["zone"] = gen_clusters["region"].map(zone_num_map)

                    fuels = fuel_cost_table(
                        fuel_costs=gc.fuel_prices,
                        generators=gc.all_resources,
                        settings=_settings,
                    )
                    gens = calculate_partial_CES_values(gen_clusters, fuels, _settings)
                    cols = [c for c in _settings["generator_columns"] if c in gens]
                    write_results_file(
                        df=remove_fuel_scenario_name(gens[cols].fillna(0), _settings),
                        folder=case_folder,
                        file_name="Generators_data.csv",
                        include_index=False,
                    )
                    # write_results_file(
                    #     df=gen_clusters.fillna(0),
                    #     folder=case_folder,
                    #     file_name="Generators_data.csv",
                    # )

                    gen_variability = make_generator_variability(
                        gen_clusters, _settings
                    )
                    # write_results_file(
                    #     df=gen_variability,
                    #     folder=case_folder,
                    #     file_name="Generators_variability.csv",
                    #     include_index=True,
                    # )

            if args.load:
                load = make_final_load_curves(
                    pudl_engine=pudl_engine, settings=_settings
                )
                load.columns = "Load_MW_z" + load.columns.map(zone_num_map)

                reduced_resource_profile, reduced_load_profile = reduce_time_domain(
                    gen_variability, load, _settings
                )
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
                transmission = transmission.pipe(
                    network_max_reinforcement, settings=_settings
                ).pipe(network_reinforcement_cost, settings=_settings)

                network = add_emission_policies(transmission, _settings)

                # Change the CES limit for cases where it's emissions based
                if "emissions_ces_limit" in _settings:
                    network = calc_emissions_ces_level(network, load, _settings)

                # If single-value for CES, use that value for input to GenX
                # settings creation. This way values that are calculated internally
                # get used.
                if network["CES"].std() == 0:
                    ces = network["CES"].mean()
                else:
                    ces = None

                write_results_file(
                    df=network,
                    folder=case_folder,
                    file_name="Network.csv",
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
                fuels = fuels.drop_duplicates(subset=["Fuel"], keep="last")
                fuels["fuel_indices"] = range(1, len(fuels) + 1)
                write_results_file(
                    df=remove_fuel_scenario_name(fuels, _settings),
                    folder=case_folder,
                    file_name="Fuels_data.csv",
                )

            genx_settings = make_genx_settings_file(
                pudl_engine, _settings, calculated_ces=ces
            )
            write_case_settings_file(
                settings=genx_settings,
                folder=case_folder,
                file_name="GenX_settings.yml",
            )
            write_case_settings_file(
                settings=_settings,
                folder=case_folder,
                file_name="powergenome_case_settings.yml",
            )


if __name__ == "__main__":
    main()
