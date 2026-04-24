import pandas as pd

def get_mean(df, col, convert: bool = True, grp_cols: list = []):
    """
    Compute the mean value of a given column of a dataframe.
    For this the data is grouped by the dataset id ["data_id"].

    :param df: The dataframe
    :param col: The column
    :param convert: If the value should be scaled from [0,1] to [0,100]
    :param grp_cols: If the data should be grouped by additional columns
    :return: A dataframe consisting of the data_id (and grp_cols) as well as the "col" column.
            The "col" column consists of the mean values.
    """
    df = df[["data_id", col] + grp_cols].copy()
    if convert:
        df[col] = df[col] * 100  # Convert to percentage
    df_mean = df.groupby(["data_id"] + grp_cols, sort=False, as_index=False).mean()
    return df_mean

def get_sd(df, col, convert: bool = True, grp_cols: list = []):
    """
    Compute the standard deviation value of a given column of a dataframe.
    For this the data is grouped by the dataset id ["data_id"].

    :param df: The dataframe
    :param col: The column
    :param convert: If the value should be scaled from [0,1] to [0,100]
    :param grp_cols: If the data should be grouped by additional columns
    :return: A dataframe consisting of the data_id (and grp_cols) as well as the "col" column.
            The "col" column consists of the sd values.
    """
    df = df[["data_id", col] + grp_cols].copy()
    if convert:
        df[col] = df[col] * 100  # Convert to percentage
    df_sd = df.groupby(["data_id"] + grp_cols, sort=False, as_index=False).std()
    return df_sd


def mean_sd_tables(df_mean, df_sd, decimal:int=2):
    """
    Takes two dataframes of the same shape: One containing the mean values and one the standard deviation.
    Combine both into a single dataframe.

    :param df_mean: The dataframe with the mean values.
    :param df_sd: The dataframe with the standard deviations.
    :param decimal: The decimals to round the mean and std values to.
    :return: The merged table.
    """
    df_mean = df_mean.round(decimal)
    df_sd = df_sd.round(decimal)
    df_mean = df_mean.astype(str)
    df_sd = df_sd.add_suffix("_sd")
    df_sd = df_sd.astype(str)

    df_merged = pd.concat([df_mean, df_sd], axis=1).astype(str)
    grouper = pd.Index(df_mean.columns.tolist() + df_mean.columns.tolist())
    df_merged = df_merged.groupby(grouper, axis=1, sort=False).apply(
        lambda x: x.astype(str).apply('\u00B1'.join, 1))

    return df_merged