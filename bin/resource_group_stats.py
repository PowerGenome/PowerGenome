"""
Create a summary file with the resource group capacity by lcoe bin for every region
"""
import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from powergenome.params import SETTINGS
from powergenome.util import load_settings, map_agg_region_names, regions_to_keep


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
        default=None,
        help="Specify a YAML settings file to aggregate capacity for model regions.",
    )
    parser.add_argument(
        "--sub_region",
        dest="sub_region",
        type=str,
        default=None,
        help="Name of the sub-region column (division below ipm region).",
    )
    arguments = parser.parse_args(argv[1:])
    return arguments


LCOE_BIN_EDGES = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200, 5000]


def resource_summary(df: pd.DataFrame, sub_region: str = None) -> pd.DataFrame:
    # Loop though metros and only include base (highest number level) clusters, which
    # include all capacity.
    by = ["ipm_region", "metro_id"]
    if sub_region:
        by.insert(-1, sub_region)
    region_list = []
    for region, _df in df.groupby(by):
        max_level = _df["level"].max()
        region_list.append(_df.query("level==@max_level"))

    region_by = ["ipm_region", "lcoe_bin"]
    if sub_region:
        region_by.insert(-1, sub_region)
    refined_df = pd.concat(region_list, ignore_index=True)
    refined_df["lcoe_bin"] = pd.cut(refined_df["lcoe"], bins=LCOE_BIN_EDGES)
    stats = refined_df.groupby(region_by)["mw"].sum().unstack()
    stats.columns = stats.columns.astype(str)  # Convert from categorical index to str
    stats["total_mw"] = stats.sum(axis=1)
    stats = stats.query("total_mw > 0")
    return stats.reset_index()


def find_resource_idx(s: str) -> int:
    name_list = s.split("_")
    for resource in ["offshorewind", "landbasedwind", "utilitypv"]:
        if resource in name_list:
            resource_idx = name_list.index(resource)
    return resource_idx


def main():
    args = parse_command_line(sys.argv)
    meta_files = list(Path(SETTINGS["RESOURCE_GROUPS"]).glob("*metadata.csv"))

    # This might fail with different naming structures of resource group files.
    # TODO: #139 Generalize finding resource type in metadata files.
    # name_len = len(meta_files[0].stem.split("_"))
    # if "offshorewind" in meta_files[0].stem:
    #     resource_position = name_len - 13
    # else:
    #     resource_position = name_len - 11
    resource_position = find_resource_idx(meta_files[0].stem)

    if args.settings_file:
        settings = load_settings(args.settings_file)
        keep_regions, region_agg_map = regions_to_keep(
            settings["model_regions"], settings.get("region_aggregations")
        )
    else:
        settings = {}

    df_list = []
    for m_f in meta_files:
        resource = m_f.stem.split("_")[resource_position]
        # from IPython import embed

        # embed()
        if resource == "existing":
            continue
        if resource == "offshorewind":
            anchor = None
            i = 0
            while not anchor and i <= len(m_f.stem.split("_")):
                i += 1
                if m_f.stem.split("_")[resource_position + i] in ["fixed", "floating"]:
                    anchor = m_f.stem.split("_")[resource_position + i]
            # anchor = m_f.stem.split("_")[resource_position + 4]
            pref = m_f.stem.split("_")[resource_position + i + 1]
            # pref = m_f.stem.split("_")[resource_position + 5]
            resource = f"{resource}_{anchor}_{pref}"
        df = pd.read_csv(m_f)

        stats = resource_summary(df, args.sub_region)
        # from IPython import embed

        # embed()
        cols = list(stats.columns)
        stats["resource"] = resource

        # Make sure capacicity numbers match
        assert np.allclose(stats["total_mw"].sum(), df.query("level == 1")["mw"].sum())

        df_list.append(stats[["resource"] + cols])

    results_df = pd.concat(df_list, ignore_index=True)
    results_df = results_df.fillna(0)

    if args.settings_file:
        results_df = map_agg_region_names(
            results_df,
            region_agg_map=region_agg_map,
            original_col_name="ipm_region",
            new_col_name="model_region",
        )
        results_df = results_df.query("ipm_region in @keep_regions")

        by = ["model_region", "resource"]
        if args.sub_region:
            by.insert(-1, args.sub_region)
        results_df = results_df.groupby(by, as_index=False).sum()
        summary_fn = (
            Path(args.settings_file).parent
            / f"{Path(args.settings_file).stem}_resource_summary_stats.csv"
        )
        print(
            "Summary stats for each model region are being saved to the project folder with your settings file."
        )
    else:
        print(
            "Summary stats for all regions are being saved to the RESOURCE_GROUPS folder listed in your .env file."
        )
        summary_fn = Path(SETTINGS["RESOURCE_GROUPS"]) / "resource_summary_stats.csv"

    results_df.to_csv(summary_fn, index=False, float_format="%.1f")


if __name__ == "__main__":
    main()
