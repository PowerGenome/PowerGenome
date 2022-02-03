"Functions to cluster or otherwise reduce the number of hours in generation and load profiles"

from sklearn.cluster import KMeans
from sklearn.preprocessing import minmax_scale
import numpy as np
import datetime
import pandas as pd


def kmeans_time_clustering(
    resource_profiles,
    load_profiles,
    days_in_group,
    num_clusters,
    include_peak_day=True,
    load_weight=1,
    variable_resources_only=True,
):
    """Reduce the number of hours in load and resource variability timeseries using
    kmeans clustering.

    This script is adapted from work originally created by Dharik Mallapragada. For more
    information see:
    - Mallapragada, D. S., Papageorgiou, D. J., Venkatesh, A., Lara, C. L., & Grossmann,
    I. E. (2018). Impact of model resolution on scenario outcomes for electricity sector
    system expansion. Energy, 163, 1231–1244.
    https://doi.org/10.1016/j.energy.2018.08.015
    - Mallapragada, D. S., Sepulveda, N. A., & Jenkins, J. D. (2020). Long-run system
    value of battery energy storage in future grids with increasing wind and solar
    generation. Applied Energy, 275, 115390.
    https://doi.org/10.1016/j.apenergy.2020.115390


    Parameters
    ----------
    resource_profiles : DataFrame
        Hourly generation profiles for all resources. Each column is a resource with
        a unique name, each row is a consecutive hour.
    load_profiles : DataFrame
        Hourly demand profiles of load. Each column is a region with a unique name. each
        row is a consecutive hour.
    days_in_group : int
        The number of 24 hour periods included in each group/cluster
    num_clusters : int
        The number of clusters to include in the output
    include_peak_day : bool, optional
        If the days with system peak demand should be included in outputs, by default
        True
    load_weight : int, optional
        A weighting factor for load profiles during clustering, by default 1
    variable_resources_only : bool, optional
        If clustering should only consider resources with variable (non-zero standard
        deviation) profiles, by default True

    Returns
    -------
    (dict, list, list)
        This function returns multiple items. The dict has keys ['load_profiles',
        'resource_profiles', 'ClusterWeights', 'AnnualGenScaleFactor', 'RMSE', and
        'AnnualProfile']

        The first list has strings with the order of periods selected e.g. ['p42','p26',
        'p3', 'p13', 'p32', 'p8'].

        The second list has integer weights of each cluster.
    """
    resource_col_names = resource_profiles.columns
    if variable_resources_only:
        input_std = resource_profiles.describe().loc["std", :]
        var_col_names = [col for col in input_std.index if input_std[col] > 0]
        resource_profiles = resource_profiles.loc[:, var_col_names]
    # Initialize dataframes to store final and intermediate data in

    input_data = pd.concat(
        [
            load_profiles.reset_index(drop=True),
            resource_profiles.reset_index(drop=True),
        ],
        axis=1,
    )
    input_data = input_data.reset_index(drop=True)
    original_col_names = input_data.columns.tolist()
    # CAUTION: Load Column lables should be named with the phrase "Load_"
    load_col_names = load_profiles.columns

    # Columns to be reported in output files
    new_col_names = input_data.columns.tolist() + ["GrpWeight"]
    # Dataframe storing final outputs
    final_output_data = pd.DataFrame(columns=new_col_names)

    # Dataframe storing normalized inputs
    norm_tseries = pd.DataFrame(columns=original_col_names)

    hours_per_year = len(input_data)

    # Normalized all load and renewables data 0 and LoadWeight, All Renewables b/w 0
    # and 1
    norm_tseries = pd.DataFrame(
        data=minmax_scale(input_data), columns=input_data.columns
    )
    norm_tseries.loc[:, load_col_names] *= load_weight

    # Identify hour with maximum system wide load
    hr_maxSysLoad = input_data.loc[:, load_col_names].sum(axis=1).idxmax()
    ################################ pre-processing data to create concatenated column
    # of load, pv and wind data

    # Number of such samples in a year - by avoiding division by float we are excluding
    # a few days in each sample set
    # Hence annual generation for each zone will not exactly match up with raw data
    # num_data_points = round(hours_per_year / 24 / days_in_group)
    num_data_points = int(hours_per_year / 24 // days_in_group)

    DataPointAsColumns = [f"p{j}" for j in range(1, num_data_points + 1)]

    # Create a dictionary storing groups of time periods to average over for each hour
    HourlyGroupings = {
        i: [j for j in range(days_in_group * 24 * (i - 1), days_in_group * 24 * i)]
        for i in range(1, num_data_points + 1)
    }

    #  Create a new dataframe storing aggregated load and renewables time series
    ModifiedDataNormalized = pd.DataFrame(columns=DataPointAsColumns)
    # Original data organized in concatenated column
    ModifiedData = pd.DataFrame(columns=DataPointAsColumns)

    # Creating the dataframe with concatenated columns
    for j in range(num_data_points):

        if j == 1:  # Store  variable names for the concatenated column
            ConcatenatedRowNames = norm_tseries.loc[HourlyGroupings[j + 1], :].melt(
                id_vars=None
            )["variable"]

        ModifiedDataNormalized[DataPointAsColumns[j]] = norm_tseries.loc[
            HourlyGroupings[j + 1], :
        ].melt(id_vars=None)["value"]
        ModifiedData[DataPointAsColumns[j]] = input_data.loc[
            HourlyGroupings[j + 1], :
        ].melt(id_vars=None)["value"]

    # Eliminate grouping including the hour with largest system laod (GW) - this group
    # will be manually included in the outputs
    if include_peak_day:
        # IP()
        GroupingwithPeakLoad = ["p" + str(int(hr_maxSysLoad / 24 / days_in_group + 1))]
        ClusteringInputDF = ModifiedDataNormalized.drop(GroupingwithPeakLoad, axis=1)
    else:
        ClusteringInputDF = ModifiedDataNormalized

    ################################## k-means clustering process
    # create Kmeans clustering model and specify the number of clusters gathered
    # number of replications =100, squared euclidean distance

    if include_peak_day:  # If peak day in cluster, generate one less cluster
        num_clusters = num_clusters - 1

    # K-means clutering with 100 trials with randomly selected starting values
    model = KMeans(
        n_clusters=num_clusters, n_init=100, init="k-means++", random_state=42
    )
    model.fit(ClusteringInputDF.values.transpose())

    # Store clustered data
    # Create an empty list storing weight of each cluster
    EachClusterWeight = [None] * num_clusters

    # Create an empty list storing name of each data point
    EachClusterRepPoint = [None] * num_clusters

    # creating a dataframe for storing the mapping between representative time period and the entire year
    time_series_mapping = pd.DataFrame(columns=["Period_Index", "Rep_Period"])

    for k in range(num_clusters):
        # Number of points in kth cluster (i.e. label=0)
        EachClusterWeight[k] = len(model.labels_[model.labels_ == k])

        # Compute Euclidean distance of each point from centroid of cluster k
        dist = {
            ClusteringInputDF.loc[:, model.labels_ == k].columns[j]: np.linalg.norm(
                ClusteringInputDF.loc[:, model.labels_ == k].values.transpose()[j]
                - model.cluster_centers_[k]
            )
            for j in range(EachClusterWeight[k])
        }

        # Select column name closest with the smallest euclidean distance to the mean
        EachClusterRepPoint[k] = min(dist, key=lambda k: dist[k])

        # Creating a list that matches each week to a representative week
        for j in range(EachClusterWeight[k]):
            time_series_mapping = time_series_mapping.append(
                pd.DataFrame(
                    {
                        "Period_Index": int(
                            ClusteringInputDF.loc[:, model.labels_ == k].columns[j][1:]
                        ),
                        "Rep_Period": k + 1,
                    },
                    index=[0],
                ),
                ignore_index=True,
            )

    # appending the week representing peak load
    time_series_mapping = time_series_mapping.append(
        pd.DataFrame(
            {"Period_Index": int(GroupingwithPeakLoad[0][1:]), "Rep_Period": k + 2},
            index=[0],
        ),
        ignore_index=True,
    )

    # same CSV file that will be used in GenX
    time_series_mapping = time_series_mapping.sort_values(by=["Period_Index"])
    time_series_mapping = time_series_mapping.reset_index(drop=True)

    # extract month corresponding to each time slot
    time_series_mapping["Month"] = 0
    for Period_Index in time_series_mapping["Period_Index"]:
        dayOfYear = days_in_group * Period_Index
        d = datetime.datetime.strptime("{} {}".format(dayOfYear, 2011), "%j %Y")
        time_series_mapping["Month"][Period_Index - 1] = d.month

    # Storing selected groupings in a new data frame with appropriate dimensions
    # (E.g. load in GW)
    ClusterOutputDataTemp = ModifiedData[EachClusterRepPoint]

    # Selecting rows corresponding to Load in excluded subperiods and exclude them from
    # scale factor calculation
    NRowsLoad = len(load_col_names)
    # Excluding grouping with peak hr from scale factor calculation
    if include_peak_day:
        Actualdata = ModifiedData.loc[0 : 24 * days_in_group * NRowsLoad - 1, :].drop(
            GroupingwithPeakLoad, axis=1
        )
    else:
        Actualdata = ModifiedData.loc[0 : 24 * days_in_group * NRowsLoad - 1, :]

    # Scale factor to adjust total generation in original data set to be equal to scaled
    # up total generation in sampled data set
    SampleweeksAnnualTWh = sum(
        [
            ClusterOutputDataTemp.loc[
                0 : 24 * days_in_group * NRowsLoad - 1, EachClusterRepPoint[j]
            ].sum()
            * EachClusterWeight[j]
            for j in range(num_clusters)
        ]
    )
    ScaleFactor = (
        Actualdata.loc[0 : 24 * days_in_group * NRowsLoad - 1, :].sum().sum()
        / SampleweeksAnnualTWh
    )

    # Updated load values in GW
    ClusterOutputDataTemp.loc[0 : 24 * days_in_group * NRowsLoad - 1, :] = (
        ScaleFactor
        * ClusterOutputDataTemp.loc[0 : 24 * days_in_group * NRowsLoad - 1, :]
    )

    # Add the grouping with the peak hour back into the cluster if that is excluded in
    # the clustering
    if include_peak_day:
        EachClusterRepPoint = EachClusterRepPoint + GroupingwithPeakLoad
        EachClusterWeight = EachClusterWeight + [1]
        ClusterOutputData = pd.concat(
            [ClusterOutputDataTemp, ModifiedData[GroupingwithPeakLoad]],
            axis=1,
            sort=False,
        )
    else:
        ClusterOutputData = ClusterOutputDataTemp

    # Store weights for each selected hour  Number of days *24, for each week
    ClusteredWeights = pd.DataFrame(
        EachClusterWeight * np.ones([days_in_group * 24, len(EachClusterWeight)]),
        columns=EachClusterRepPoint,
    )

    # Storing weights in final output data column
    final_output_data["GrpWeight"] = ClusteredWeights.melt(id_vars=None)["value"]

    # Regenerating data organized by time series (columns) and representative time
    # periods (hours)
    for i in range(len(new_col_names) - 1):
        final_output_data[new_col_names[i]] = ClusterOutputData.loc[
            ConcatenatedRowNames == new_col_names[i], :
        ].melt(id_vars=None)["value"]

    # Calculating error metrics and Annual profile
    FullLengthOutputs = final_output_data
    for j in range(len(EachClusterWeight)):
        # Selecting rows of the FinalOutputData dataframe to append
        df_try = final_output_data.truncate(
            before=days_in_group * 24 * j, after=days_in_group * 24 * (j + 1) - 1
        )
        #        print(EachClusterWeight[j])
        if (
            EachClusterWeight[j] > 1
        ):  # Need to duplicate entries only weight is greater than 1
            FullLengthOutputs = FullLengthOutputs.append(
                [df_try] * (EachClusterWeight[j] - 1), ignore_index=True
            )

    #  Root mean square error between the duration curves of each time series
    # Only conisder the points consider in the k-means clustering - ignoring any days
    # dropped off from original data set  due to rounding
    RMSE = {
        col: np.linalg.norm(
            np.sort(input_data.truncate(after=len(FullLengthOutputs) - 1)[col].values)
            - np.sort(FullLengthOutputs[col].values)
        )
        for col in original_col_names
    }

    load_df = final_output_data.loc[:, load_col_names]
    # if variable_only:
    resource_df = pd.DataFrame(
        columns=resource_col_names, index=final_output_data.index
    )
    for col in resource_col_names:
        try:
            resource_df[col] = final_output_data.loc[:, col].values
        except KeyError:
            pass
    resource_df = resource_df.fillna(value=1)

    # load_df["Sub_Weights"] = np.nan
    # load_df.loc[: len(EachClusterWeight) - 1, "Sub_Weights"] = (
    #     np.array(EachClusterWeight) * NumGrpDays * 24
    # )
    # load_df.to_csv("load_time_reduced.csv", index=False)
    # renewable_df = FinalOutputData.loc[
    #     :, [col for col in FinalOutputData.columns if "Load_" not in col]
    # ]
    # renewable_df = renewable_df.drop(columns=["GrpWeight"])
    # renewable_df.insert(loc=0, column="Resource", value=renewable_df.index + 1)
    # renewable_df.to_csv("renewables_time_reduced.csv", index=False)
    EachClusterRepPoint = pd.DataFrame(EachClusterRepPoint, columns=["slot"])
    return (
        {
            "load_profiles": load_df,  # Scaled Output Load and Renewables profiles for the sampled representative groupings
            "resource_profiles": resource_df,
            "ClusterWeights": EachClusterWeight,  # Weight of each for the representative groupings
            "AnnualGenScaleFactor": ScaleFactor,  # Scale factor used to adjust load output to match annual generation of original data
            "RMSE": RMSE,  # Root mean square error between full year data and modeled full year data (duration curves)
            "AnnualProfile": FullLengthOutputs,
            "time_series_mapping": time_series_mapping,
        },
        EachClusterRepPoint,
        EachClusterWeight,
    )  # Modeled duration curves GW
