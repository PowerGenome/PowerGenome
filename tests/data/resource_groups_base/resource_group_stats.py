"""
Create a summary file with the resource group capacity by lcoe bin for every region
"""

import pandas as pd
import numpy as np
from pathlib import Path
from powergenome.util import regions_to_keep, map_agg_region_names, load_settings

settings_path = None

if settings_path:
    settings = load_settings(settings_path)
    keep_regions, region_agg_map = regions_to_keep(settings)


LCOE_BIN_EDGES = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200, 5000]

meta_files = list(Path.cwd().glob("*metadata.csv"))

name_len = len(meta_files[0].stem.split("_"))
if "offshorewind" in meta_files[0].stem:
    resource_position = name_len - 7
else:
    resource_position = name_len - 5


def resource_summary(df: pd.DataFrame) -> pd.DataFrame:

    # Loop though metros and only include base (highest number level) clusters, which
    # include all capacity.
    region_list = []
    for region, _df in df.groupby(["ipm_region", "metro_id"]):
        max_level = _df["level"].max()
        region_list.append(_df.query("level==@max_level"))

    refined_df = pd.concat(region_list, ignore_index=True)
    refined_df["lcoe_bin"] = pd.cut(refined_df["lcoe"], bins=LCOE_BIN_EDGES)
    stats = refined_df.groupby(["ipm_region", "lcoe_bin"])["mw"].sum().unstack()
    stats.columns = stats.columns.astype(str)  # Convert from categorical index to str
    stats["total_mw"] = stats.sum(axis=1)
    return stats.reset_index()


df_list = []
for m_f in meta_files:

    resource = m_f.stem.split("_")[resource_position]
    if resource == "existing":
        continue
    if resource == "offshorewind":
        anchor = m_f.stem.split("_")[resource_position + 4]
        pref = m_f.stem.split("_")[resource_position + 5]
        resource = f"{resource}_{anchor}_{pref}"
    df = pd.read_csv(m_f)

    stats = resource_summary(df)
    cols = list(stats.columns)
    stats["resource"] = resource

    # Make sure capacicity numbers match
    assert np.allclose(stats["total_mw"].sum(), df.query("level == 1")["mw"].sum())

    df_list.append(stats[["resource"] + cols])

results_df = pd.concat(df_list, ignore_index=True).fillna(0)

if settings_path:
    results_df = map_agg_region_names(
        results_df,
        region_agg_map=region_agg_map,
        original_col_name="ipm_region",
        new_col_name="model_region",
    )

    results_df = results_df.groupby(["model_region", "resource"], as_index=False).sum()


results_df.to_csv("resource_summary_stats.csv", index=False)
