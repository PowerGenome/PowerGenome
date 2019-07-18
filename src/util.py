import yaml

import pudl


def load_settings(path):

    with open(path, 'r') as f:
        settings = yaml.safe_load(f)

    return settings


def init_pudl_connection(freq='YS'):

    pudl_engine = pudl.init.connect_db()
    pudl_out = pudl.output.pudltabl.PudlTabl(freq=freq)

    return pudl_engine, pudl_out


def reverse_dict_of_lists(d):

    return {v: k for k in d for v in d[k]}


def map_agg_region_names(df, region_agg_map, original_col_name, new_col_name):

    df[new_col_name] = df.loc[:, original_col_name]

    df.loc[
        df[original_col_name].isin(region_agg_map.keys()),
        new_col_name
    ] = df.loc[
            df[original_col_name].isin(region_agg_map.keys()),
            original_col_name
        ].map(region_agg_map)

    return df
