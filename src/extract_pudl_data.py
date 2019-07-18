import argparse
import logging
import sys
from datetime import datetime as dt

import pudl
from src.util import load_settings, init_pudl_connection
from src.params import DATA_PATHS
from src.generators import create_region_technology_clusters
from src.load_profiles import load_curves
from src.transmission import agg_transmission_constraints

# Create a logger to output any messages we might have...
logger = logging.getLogger(pudl.__name__)
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
        default="pudl_data_extraction.yml",
        help="Specify a YAML settings file.",
    )
    parser.add_argument(
        "-rf",
        "--results_folder",
        dest="results_folder",
        type=str,
        default=dt.now().strftime("%Y-%m-%d %H:%M:%S"),
        help="Specify the results subfolder to write output",
    )
    arguments = parser.parse_args(argv[1:])
    return arguments


def main():

    args = parse_command_line(sys.argv)
    settings = load_settings(path=args.settings_file)
    out_folder = DATA_PATHS["results"] / args.results_folder
    out_folder.mkdir(exist_ok=True)

    pudl_engine, pudl_out = init_pudl_connection(freq="YS")

    gen_clusters = create_region_technology_clusters(
        pudl_engine=pudl_engine, pudl_out=pudl_out, settings=settings
    )

    load = load_curves(pudl_engine=pudl_engine, settings=settings)

    transmission = agg_transmission_constraints(
        pudl_engine=pudl_engine, settings=settings
    )

    gen_clusters.to_csv(out_folder / f"generator_clusters_{args.results_folder}.csv")
    load.to_csv(out_folder / f"load_curves_{args.results_folder}.csv")
    transmission.to_csv(
        out_folder / f"transmission_constraints_{args.results_folder}.csv"
    )


if __name__ == '__main__':
    main()
