"Different ways to cluster plants"

import numpy as np
import pandas as pd
from sklearn import cluster, preprocessing


def build_cluster_method_dict(settings):
    cluster_method_dict = {}

    for region in settings["model_regions"]:
        cluster_method_dict[region] = {}
        if region not in settings["cluster_by_owner_regions"]:
            for tech in settings["num_clusters"]:
                cluster_method_dict[region][tech] = cluster_kmeans
        else:
            if tech in settings["cluster_by_owner_regions"][region]:
                cluster_method_dict[region][tech] = cluster_by_owner
            else:
                cluster_method_dict[region][tech] = cluster_kmeans

    return cluster_method_dict


def cluster_kmeans(
    grouped: pd.DataFrame, region: str, tech: str, settings: dict
) -> pd.DataFrame:
    if region in settings.get("alt_num_clusters", {}):
        # if tech in settings["alt_num_clusters"][region]:
        n_clusters = settings["alt_num_clusters"][region].get(tech)
    else:
        n_clusters = settings["num_clusters"].get(tech)

    if n_clusters:
        clusters = cluster.KMeans(n_clusters=n_clusters, random_state=6).fit(
            preprocessing.StandardScaler().fit_transform(grouped)
        )
        grouped["cluster"] = clusters.labels_ + 1  # Change to 1-index for julia

    return grouped


def cluster_by_owner(grouped_units, weighted_ownership, plants, region, tech, settings):
    # I'm doing this by generator right now, but don't have generator info in the inputs.
    # Can calculate weighted percent ownership by pudl unit and use that in here.

    capacity_col = settings["capacity_col"]
    start_capacity = grouped_units[capacity_col].sum()

    owner_cols = [
        # "utility_id_eia",
        "plant_id_eia",
        "owner_utility_id_eia",
        "unit_id_pudl",
        # "owner_name",
        # "owner_state",
        "fraction_owned",
        "ownership_code",
    ]

    plants_cols = ["plant_id_eia", "utility_id_eia"]

    gens_ownership = grouped_units.merge(
        weighted_ownership[owner_cols], on=["plant_id_eia", "unit_id_pudl"], how="left"
    ).merge(plants[plants_cols], on=["plant_id_eia"], how="left")

    gens_ownership.loc[
        gens_ownership["owner_utility_id_eia"].isnull(), "owner_utility_id_eia"
    ] = gens_ownership.loc[
        gens_ownership["owner_utility_id_eia"].isnull(), "utility_id_eia"
    ]

    # First take care of plants where we have a utility_id to cluster by

    # Jointly owned plants
    mask = (
        (gens_ownership.ownership_code == "J")
        & (
            gens_ownership.owner_utility_id_eia.isin(
                settings["cluster_by_owner_regions"][region]["utility_ids_to_cluster"]
            )
        )
        & (gens_ownership.technology_description == tech)
    )

    gens_ownership.loc[mask, "plant_id_eia"] = (
        gens_ownership.loc[mask, "plant_id_eia"].astype(str)
        + "_"
        + gens_ownership.loc[mask, "owner_utility_id_eia"].astype(int).astype(str)
    )
    gens_ownership.loc[mask, capacity_col] *= gens_ownership.loc[mask, "fraction_owned"]
    gens_ownership.loc[mask, "cluster"] = (
        gens_ownership.loc[mask, "owner_utility_id_eia"].astype(int).astype(str)
    )

    # Single owner - respondent or someone else (S or W)
    mask = (
        (gens_ownership.ownership_code != "J")
        & (
            gens_ownership.owner_utility_id_eia.isin(
                settings["cluster_by_owner_regions"][region]["utility_ids_to_cluster"]
            )
        )
        & (gens_ownership.technology_description == tech)
    )
    gens_ownership.loc[mask, "cluster"] = (
        gens_ownership.loc[mask, "owner_utility_id_eia"].astype(int).astype(str)
    )

    # Now take care of plants where the cluster will be "other"
    other_mask = (
        (gens_ownership.ownership_code == "J")
        & (
            ~gens_ownership.owner_utility_id_eia.isin(
                settings["cluster_by_owner_regions"][region]["utility_ids_to_cluster"]
            )
        )
        & (gens_ownership.technology_description == tech)
    )
    gens_ownership.loc[other_mask, "plant_id_eia"] = (
        gens_ownership.loc[other_mask, "plant_id_eia"].astype(str) + "_" + "other"
    )
    gens_ownership.loc[other_mask, capacity_col] *= gens_ownership.loc[
        other_mask, "fraction_owned"
    ]
    gens_ownership.loc[other_mask, "cluster"] = "other"

    # Single owner - respondent or someone else (S or W)
    other_mask = (
        (gens_ownership.ownership_code != "J")
        & (
            ~gens_ownership.owner_utility_id_eia.isin(
                settings["cluster_by_owner_regions"][region]["utility_ids_to_cluster"]
            )
        )
        & (gens_ownership.technology_description == tech)
    )
    gens_ownership.loc[other_mask, "cluster"] = "other"

    gens_ownership = gens_ownership.drop_duplicates()

    end_capacity = gens_ownership[capacity_col].sum()

    assert np.allclose(start_capacity, end_capacity)
    # assert gens_ownership.loc[gens_ownership.cluster.isna(), :].empty is True

    return gens_ownership


def weighted_ownership_by_unit(units_model, gens_860, ownership, settings):
    owner_cols = [
        "utility_id_eia",
        "plant_id_eia",
        "generator_id",
        "owner_utility_id_eia",
        "owner_name",
        "owner_state",
        "fraction_owned",
    ]

    units_model_cols = [
        "plant_id_eia",
        "generator_id",
        "unit_id_pudl",
        settings["capacity_col"],
    ]

    gens_cols = ["plant_id_eia", "generator_id", "ownership_code"]

    owner_unit = (
        ownership[owner_cols]
        .merge(
            units_model.reset_index()[units_model_cols],
            on=["plant_id_eia", "generator_id"],
            how="right",
        )
        .merge(gens_860[gens_cols], on=["plant_id_eia", "generator_id"], how="left")
    )

    def w_ownership(df, capacity_col):
        weighted_ownership = np.average(df["fraction_owned"], weights=df[capacity_col])
        return weighted_ownership

    owner_unit.loc[:, "fraction_owned"] = owner_unit.loc[:, "fraction_owned"].fillna(1)
    weighted_ownership = (
        owner_unit.groupby(
            ["plant_id_eia", "unit_id_pudl", "owner_utility_id_eia", "ownership_code"],
            as_index=False,
        )
        .apply(w_ownership, settings["capacity_col"])
        .reset_index()
        .rename(columns={0: "fraction_owned"})
        .fillna(0.5)
    )

    weighted_ownership = weighted_ownership.merge(
        ownership[["plant_id_eia", "utility_id_eia"]], on=["plant_id_eia"], how="left"
    )

    return weighted_ownership
