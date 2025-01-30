# extract file in /data folder and save dataframes in /dataframes folder

import os

import pandas as pd


def extract_data(path):
    """
    Extract data from a path and return a dataframe.
    :param path:
    :return: list of dataframes
    """

    dataframes = []

    for d in os.listdir(path):
        file_path = os.path.join(path, d)

        if file_path.endswith('.json'):
            df = pd.read_json(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path, encoding='ISO-8859-1')
        elif file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.jsonl'):
            df = pd.read_json(file_path, lines=True)
        else:
            raise ValueError('Invalid file format.')


        dataframes.append(df)

    # remove all columns with unnamed like name
    for i in range(len(dataframes)):
        dataframes[i] = dataframes[i].loc[:, ~dataframes[i].columns.str.contains('^Unnamed')]

    #all columns names of dataframes there is lowercase
    for i in range(len(dataframes)):
        dataframes[i].columns = dataframes[i].columns.str.lower()

    return dataframes


