"Different ways to cluster plants"

import numpy as np


def cluster_by_owner(grouped_units, ownership, region, tech, settings):

    # I'm doing this by generator right now, but don't have generator info in the inputs.
    # Can calculate weighted percent ownership by pudl unit and use that in here.

    capacity_col = settings["capacity_col"]
    start_capacity = grouped_units[capacity_col].sum()

    owner_cols = [
        "utility_id_eia",
        "plant_id_eia",
        "generator_id",
        "owner_utility_id_eia",
        "owner_name",
        "owner_state",
        "fraction_owned",
    ]
    gens_ownership = grouped_units.merge(
        ownership[owner_cols], on=["plant_id_eia", "generator_id"], how="left"
    )

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
        gens_ownership.loc[mask, "plant_id_eia"]
        + "_"
        + gens_ownership.loc[mask, "owner_utility_id_eia"].astype(int).astype(str)
    )
    gens_ownership.loc[mask, capacity_col] *= gens_ownership.loc[mask, "fraction_owned"]
    gens_ownership.loc[mask, "cluster"] = (
        gens_ownership.loc[mask, "owner_utility_id_eia"].astype(int).astype(str)
    )

    # Single owner - respondent or someone else (S or W)
    mask = (
        (gens_ownership.ownership_code.isin(["W", "S"]))
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
        gens_ownership.loc[other_mask, "plant_id_eia"] + "_" + "other"
    )
    gens_ownership.loc[other_mask, capacity_col] *= gens_ownership.loc[
        other_mask, "fraction_owned"
    ]
    gens_ownership.loc[other_mask, "cluster"] = "other"

    # Single owner - respondent or someone else (S or W)
    other_mask = (
        (gens_ownership.ownership_code.isin(["S", "W"]))
        & (
            ~gens_ownership.owner_utility_id_eia.isin(
                settings["cluster_by_owner_regions"][region]["utility_ids_to_cluster"]
            )
        )
        & (gens_ownership.technology_description == tech)
    )
    gens_ownership.loc[other_mask, "cluster"] = "Other"

    end_capacity = gens_ownership[capacity_col].sum()

    assert np.allclose(start_capacity, end_capacity)
    assert gens_ownership.loc[gens_ownership.cluster.isna(), :].empty is True

    return gens_ownership
