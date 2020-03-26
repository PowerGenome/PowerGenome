import argparse
import logging
import shutil
import sys
from datetime import datetime as dt
from pathlib import Path

import pandas as pd

import powergenome
from powergenome.fuels import fuel_cost_table
from powergenome.generators import GeneratorClusters, load_ipm_shapefile
from powergenome.load_profiles import make_final_load_curves
from powergenome.transmission import (
    agg_transmission_constraints,
    transmission_line_distance,
)
from powergenome.util import (
    init_pudl_connection,
    load_settings,
    remove_fuel_scenario_name,
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

    if args.gens:
        gc = GeneratorClusters(
            pudl_engine=pudl_engine,
            pudl_out=pudl_out,
            settings=settings,
            current_gens=args.current_gens,
            sort_gens=args.sort_gens,
        )
        gen_clusters = gc.create_all_generators()
        gen_clusters = remove_fuel_scenario_name(gen_clusters, settings)
        gen_clusters["zone"] = gen_clusters["region"].map(zone_num_map)

    if args.load:
        load = make_final_load_curves(pudl_engine=pudl_engine, settings=settings)
        load.columns = "Load_MW_z" + load.columns.map(zone_num_map)

    if args.transmission:
        if args.gens is False:
            model_regions_gdf = load_ipm_shapefile(settings)
        else:
            model_regions_gdf = gc.model_regions_gdf
        transmission = agg_transmission_constraints(
            pudl_engine=pudl_engine, settings=settings
        )
        transmission = transmission.pipe(
            transmission_line_distance,
            ipm_shapefile=model_regions_gdf,
            settings=settings,
            units="mile",
        )

    if args.fuel and args.gens:
        fuels = fuel_cost_table(
            fuel_costs=gc.fuel_prices, generators=gc.all_resources, settings=settings
        )
        fuels["fuel_indices"] = range(1, len(fuels) + 1)
        fuels = remove_fuel_scenario_name(fuels, settings)

    logger.info(f"Write GenX input files to {args.results_folder}")
    if args.gens:
        gen_clusters.to_csv(
            out_folder / f"generator_clusters_{args.results_folder}.csv", index=False
        )

    if args.load:
        load.astype(int).to_csv(out_folder / f"load_curves_{args.results_folder}.csv")

    if args.transmission:
        transmission.to_csv(
            out_folder / f"transmission_constraints_{args.results_folder}.csv",
            float_format="%.1f",
        )

    if args.fuel and args.gens:
        fuels.to_csv(out_folder / f"Fuels_data_{args.results_folder}.csv", index=False)


if __name__ == "__main__":
    main()
