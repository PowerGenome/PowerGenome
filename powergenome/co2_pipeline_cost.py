"Load data from file for pipeline/transport/injection costs"

import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

from powergenome.params import DATA_PATHS
from powergenome.price_adjustment import inflation_price_adjustment
from powergenome.util import snake_case_col, snake_case_str

logger = logging.getLogger(__name__)


def merge_co2_pipeline_costs(
    df: pd.DataFrame,
    co2_data_path: Path,
    co2_pipeline_filters: List[dict],
    region_aggregations: Dict[str, List[str]] = None,
    fuel_emission_factors: Dict[str, float] = None,
    target_usd_year: int = None,
) -> pd.DataFrame:
    """Merge columns with CO2 pipeline costs to a dataframe.

    Data come from an external file. Filters from the settings are used to select the
    data that will be merged.

    An example of the YAML settings parameter would be:

    co2_pipeline_filters:
      - technology: NaturalGas
        tech_detail: CCCCSAvgCF
        with_backbone: false
        percentile: 25

    Parameters
    ----------
    df : pd.DataFrame
        Generators data. Should include the rows "technology" and "region", along with a
        heat rate column that has "heat_rate" and "per_mwh" if fuel emission factors
        will be used to convert costs from per tonne to per MWh.
    co2_data_path : Path
        Path to the file with CO2 pipeline cost data
    co2_pipeline_filters : List[dict]
        Filters to apply to the CO2 cost data file
    region_aggregations : Dict[str, List[str]], optional
        Aggregations of multiple base regions to model regions, by default None
    fuel_emission_factors : Dict[str, float], optional
        CO2 emissions per unit of energy for different fuel types, by default None
    target_usd_year : int, optional
        The target year for cost inflation adjustment, by default None

    Returns
    -------
    pd.DataFrame
        The input dataframe (one row per resource) plus columns with cost data for CO2
        pipeline constuction/operation and CO2 disposal
    """
    co2_df = pd.read_csv(co2_data_path)
    co2_df = co2_df.loc[co2_df["parameter"] != "capacity_mw", :]
    if target_usd_year:
        for dollar_year in co2_df["dollar_year"].unique():
            co2_df.loc[(co2_df["dollar_year"] == dollar_year), "parameter_value"] = (
                inflation_price_adjustment(
                    co2_df.loc[co2_df["dollar_year"] == dollar_year, "parameter_value"],
                    dollar_year,
                    target_usd_year,
                )
            )
    for k, v in (region_aggregations or {}).items():
        co2_df.loc[co2_df["region"].isin(v), "region"] = k

    # When regions are aggregated, use the lowest-cost ipm region
    co2_df = co2_df.sort_values("parameter_value").drop_duplicates(
        [
            "region",
            "technology",
            "tech_detail",
            "with_backbone",
            "percentile",
            "parameter",
        ],
        keep="first",
    )

    # co2_df["resource"] = (co2_df["technology"] + "_" + co2_df["tech_detail"]).lower()
    filter_techs = [
        filt["technology"] + "_" + filt["tech_detail"] for filt in co2_pipeline_filters
    ]
    co2_resources = {}
    for co2_r in filter_techs:
        for model_r in df["technology"].unique():
            if snake_case_str(co2_r) in snake_case_str(model_r):
                co2_resources[co2_r] = model_r

    if not co2_resources:
        logger.warning(
            "You have specified technologies for CO2 pipeline costs in the parameter "
            "'co2_pipeline_filters' but none of the technologies are being used in "
            "your system."
        )
        return df

    df_list = []
    for filt in co2_pipeline_filters:
        # tech = filt["technology"].lower() + "_" + filt["tech_detail"].lower()
        query_str = f"technology == '{filt['technology']}'"
        for k, v in filt.items():
            if k != "technology":
                if isinstance(v, str):
                    query_str += f" and {k} == '{v}'"
                else:
                    query_str += f" and {k} == {v}"

        co2_costs = co2_df.query(query_str)
        co2_costs.loc[:, "technology"] = (
            co2_costs["technology"] + "_" + co2_costs["tech_detail"]
        ).map(co2_resources)
        co2_costs_wide = co2_costs.pivot(
            index=["region", "technology"],
            columns="parameter",
            values="parameter_value",
        ).reset_index()
        co2_costs_wide["capture_rate"] = filt.get("capture_rate", 90)
        df_list.append(co2_costs_wide)

    full_co2_costs_wide = pd.concat(df_list)
    # Split incoming df into techs with and without pipeline cost. Remove pipeline cost
    # columns from the "with" techs to avoid duplicates, merge in co2 costs, then concat
    # with the "without" techs
    df = df.reset_index()
    drop_cols = [
        c for c in full_co2_costs_wide.columns if c not in ["region", "technology"]
    ]
    df_co2_techs = df.loc[df["technology"].isin(co2_resources.values()), :]
    df_co2_techs = df_co2_techs.drop(columns=drop_cols, errors="ignore")
    df_co2_techs = pd.merge(
        df_co2_techs, full_co2_costs_wide, on=["region", "technology"], how="left"
    )
    df_non_co2_techs = df.loc[~df["technology"].isin(co2_resources.values()), :]

    df_co2_costs = (
        pd.concat([df_co2_techs, df_non_co2_techs]).set_index("index").sort_index()
    )

    drop_idx = df_co2_costs.loc[
        (df_co2_costs["technology"].isin(co2_resources.values()))
        & (df_co2_costs[co2_costs_wide.columns].isna().any(axis=1))
    ].index

    if not drop_idx.empty:
        for tech, _df in df_co2_costs.loc[drop_idx, :].groupby("technology"):
            regs = _df["region"].to_list()
            f"The CCS resource {tech} is being removed from regions {regs} because cost "
            "data was not included in your CO2 pipeline cost file."
    df_co2_costs = df_co2_costs.drop(index=drop_idx)
    df_co2_costs.loc[:, co2_costs_wide.columns] = df_co2_costs[
        co2_costs_wide.columns
    ].fillna(0)

    mass_cols = [c for c in co2_costs_wide.columns if "tonne" in c]
    if mass_cols and fuel_emission_factors:
        df_co2_costs = mass_to_energy_costs(
            df_co2_costs, mass_cols, fuel_emission_factors
        )

    return df_co2_costs


def mass_to_energy_costs(
    df: pd.DataFrame,
    mass_cols: List[str],
    fuel_emission_factors: Dict[str, float],
    heat_rate_col: str = None,
) -> pd.DataFrame:
    """Convert costs per tonne of CO2 to a MWh basis

    Parameters
    ----------
    df : pd.DataFrame
        Generators dataframe with a fuel column, a heat rate column, and the columns
        provided in the list `mass_cols`
    mass_cols : List[str]
        Names of columns with cost per tonne of CO2 disposal
    fuel_emission_factors : Dict[str, float]
        Emission rate of tonne per unit of energy for fuels used by CCS generators
    heat_rate_col : str, optional
        Name of the column with generator heat rates, by default None

    Returns
    -------
    pd.DataFrame
        Modified copy of the input, with the new column "co2_cost_mwh"
    """
    if not heat_rate_col:
        heat_rate_cols = [
            c for c in df.columns if "heat_rate" in c.lower() and "per_mwh" in c.lower()
        ]
        if len(heat_rate_cols) > 1:
            logger.warning(
                "More than one column with 'heat_rate' and 'per_mwh' was found when "
                "converting CO2 transport costs from per tonne to per MWh. Values are "
                "not being converted."
            )
            return df

        elif not heat_rate_cols:
            logger.warning(
                "Unable to identify a heat rate column when converting CO2 transport costs "
                "from per tonne to per MWh. A heat rate column should contain 'heat_rate' and "
                "'per_mwh' (capitalization does not matter). Values are not being converted."
            )
            return df
        else:
            heat_rate_col = heat_rate_cols[0]

    fuel_col = [c for c in df.columns if c.lower() == "fuel"]
    fuel_col = fuel_col[0]
    for fuel, ef in fuel_emission_factors.items():
        df.loc[
            df[fuel_col].str.contains(fuel, case=False), "tonne_co2_captured_mwh"
        ] = (
            df.loc[df[fuel_col].str.contains(fuel, case=False), heat_rate_col]
            * ef
            * (
                df.loc[df[fuel_col].str.contains(fuel, case=False), "capture_rate"]
                / 100
            )
        )

    df.loc[:, "co2_cost_mwh"] = df[mass_cols].sum(axis=1) * df["tonne_co2_captured_mwh"]
    df.loc[:, "co2_cost_mwh"] = df.loc[:, "co2_cost_mwh"].fillna(0)

    return df
