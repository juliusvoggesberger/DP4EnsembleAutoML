import math
import os

import numpy as np
import pandas as pd


def _read_csv(filepath: str, data_id: str) -> pd.DataFrame:
    """
    Read in one file and append the data_id to its rows.

    :param filepath: Path to the file.
    :param data_id: ID of the associated task.
    :return: The dataframe.
    """
    try:
        df = pd.read_csv(filepath, converters={
            "precision": lambda x: np.asarray(list(map(float, x.strip("[]").split()))),
            "recall": lambda x: np.asarray(list(map(float, x.strip("[]").split())))}, index_col=0)
        df["data_id"] = [data_id for i in range(df.shape[0])]
    except FileNotFoundError:
        print(f"File {filepath} not found.")
        df = pd.DataFrame()

    return df


def collect_data(parent_folder: str, filename: str = "ensembles"):
    """
    Iterate over all dataset folders and get the data of a specific file.
    If a folder is not empty, get the optimised ensemble and best individual classifier for the
    dataset.

    :param parent_folder: Parent folder holding all run information.
    :param filename: The csv-file of each run that should be collected. Default is "ensembles".

    :return: A pandas dataframe holding the data of the requested files.
    """
    df = pd.DataFrame()

    # Sort by task id
    for file in os.listdir(parent_folder):
        filepath = parent_folder + os.fsdecode(file) + "/"

        df_2 = _read_csv(filepath + filename + ".csv", file.split("_")[0])
        df = pd.concat([df, df_2], ignore_index=True)

    df["seed"] = parent_folder.split("/")[-2].split(" ")[-1]
    return df


def collect_divbo_data(parent_folder: str):
    seeds = ["123", "456", "789", "1010", "2020"]
    try:
        df = pd.read_csv(parent_folder + "evaluation_runs.csv")
        df = df.rename(columns={"dataset": "data_id"})
    except FileNotFoundError:
        print(f"File {parent_folder}DivBO_evaluation_runs.csv not found.")
        df = pd.DataFrame()

    for d_id in df["data_id"].unique():
        df.loc[df["data_id"] == d_id, "seed"] = seeds[:len(df[df["data_id"] == d_id])]
    df["data_id"] = df["data_id"].astype(str)
    return df


def get_ensembles(filepaths: list, metric: str, only_fusion: bool = True,
                  add_column: list = []) -> pd.DataFrame:
    """
    Returns the evaluation result for each evaluated ensemble.

    :param filepaths: List of filepaths to the evaluation runs.
    :param metric: The evaluation metric to return.
    :return: A pandas dataframe.
    """

    df = pd.DataFrame()
    for filepath in filepaths:
        df_2 = collect_data(filepath, "ensembles")[
            ["data_id", "algorithm", metric, "model type", "seed", "k"] + add_column]
        if only_fusion:
            df_2 = df_2[df_2["model type"] == "Fusion"]
        df = pd.concat([df, df_2], ignore_index=True)
    return df


def get_autosklearn(filepaths: list):
    """
    Return a dataframe containing the performance values for all datasets of the given seeds.

    :param filepaths: The filepaths to the different seed runs.
    :return: The dataframe.
    """
    df = pd.DataFrame()

    for filepath in filepaths:
        # Load The auto-sklearn data
        df_as = pd.read_csv(filepath, converters={
            "precision": lambda x: np.asarray(list(map(float, x.strip("[]").split()))),
            "recall": lambda x: np.asarray(list(map(float, x.strip("[]").split())))})
        df_as["seed"] = filepath.split("/")[-1].split(".")[0].split("_")[-1]
        df = pd.concat([df, df_as], ignore_index=True)

    df = df.rename(columns={"task_id": "data_id"})
    df["data_id"] = df["data_id"].astype(int).astype(str)

    return df


def load_runtimes(filepaths):
    """
    Load evaluation data of the runtimes of Auto-CEn

    :param filepaths: The filepaths to the different seed runs.
    :return: A dataframe.
    """
    df = None
    for filepath in filepaths:
        seed = filepath.split("/")[-2].split(" ")[-1]
        if df is None:
            df = collect_data(filepath, "runtime")
            df["seed"] = seed
        else:
            df_2 = collect_data(filepath, "runtime")
            df_2["seed"] = seed
            df = pd.concat([df, df_2], ignore_index=True)

    return df


def load_diversity(filepaths: list, metric: list, is_as: bool = False) -> pd.DataFrame:
    """
    Loads the diversity for either the ensemble or auto-sklearn evaluation.

    :param filepaths: The filepaths to the different seed runs.
    :param metric: The diversity metric.
    :param is_as: If True, loads the evaluation results for auto-sklearn.
    :return: A dataframe.
    """

    df = pd.DataFrame()

    for filepath in filepaths:
        if is_as:
            for file in os.listdir(filepath):
                df_task = _read_csv(filepath + file, file.split("_")[1])  # [["data_id", *metric]]
                if "correlation_norm" in metric and not "correlation_norm" in df_task.columns and "correlation" in df_task.columns:
                    # Try to save it
                    df_task["correlation_norm"] = 1 - df_task["correlation"]
                df_task = df_task[["data_id", *metric]]

                df_task["seed"] = filepath.split("/")[-3].split(" ")[-1]
                # As the diversity is computed pairwise, the size of the dataframe will be a triangular number
                # The size will be the largest summand of the triangular number, as computed below
                df_task["size"] = (math.sqrt(8 * df_task.shape[0] + 1.) - 1.) / 2.
                df = pd.concat([df, df_task if not df_task.empty else None], ignore_index=True)
        else:
            df_2 = collect_data(filepath, "diversity")[["data_id", *metric, "seed"]]
            df = pd.concat([df, df_2], ignore_index=True)
    return df


def load_multiple_data(filepaths: list, metric: str, data_type: str = "ensemble") -> pd.DataFrame:
    """
    Load the evaluation results for datasets with multiple data characteristics.

    :param filepaths: The filepaths to the evaluation results.
    :param metric: The evaluation metric.
    :param data_type: The type of evaluation results. Can be "ensemble", "diversity" or "runtime".
    :return: Four dataframes. One for SD, HD, CI and Multiple data characteristics.
    """

    dfs = [pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()]
    prefixe = ["SD", "HD", "CI", "Multiple"]

    for filepath in filepaths:
        for i, p in enumerate(prefixe):
            df_2 = load_additional_data(filepath, p, data_type, [metric])
            df_2 = df_2[["data_id", metric, "seed"]]
            dfs[i] = pd.concat([dfs[i], df_2], ignore_index=True)
    return dfs


def load_additional_data(filepath: str, prefix: str, data_type: str = "ensemble",
                         metrics: list = None) -> pd.DataFrame:
    """
    Load the data of an evaluation run for specific data characteristics.

    :param filepath: The filepath to the evaluation run.
    :param prefix: The data characteristic to load. Can be "SD", "HD", "CI", "Multiple".
    :param data_type: The files to load. Can be "ensemble", "diversity" or "runtime".
    :param metrics: List of metrics to load.
    :return: A data frame containing the evaluation results.
    """
    if data_type == "ensemble":
        df = get_ensembles([filepath + prefix + "/"], metrics[0])
    elif data_type == "diversity":
        df = load_diversity([filepath + prefix + "/"], metrics)
    elif data_type == "runtime":
        df = load_runtimes([filepath + prefix + "/"])
    return df
