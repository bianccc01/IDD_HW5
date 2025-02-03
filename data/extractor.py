# extract file in /data folder and save dataframes in /dataframes folder

import os

import pandas as pd
import numpy as np


def extract_data(path):
    """
    Extract data from a path and return a dataframe.
    :param path:
    :return: list of dataframes
    """
    dataframes = []

    for d in os.listdir(path):
        file_path = os.path.join(path, d)

        try:
            if file_path.endswith('.json'):
                df = pd.read_json(file_path)
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path, encoding='ISO-8859-1')
            elif file_path.endswith('.xls'):
                df = pd.read_excel(file_path)
            elif file_path.endswith('.jsonl'):
                df = pd.read_json(file_path, lines=True)
            else:
                continue

            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

            for col in df.columns:
                if df[col].dtype == np.int64 or df[col].dtype == np.float64:
                    df[col] = df[col].astype(str)

            #if a value is a list, convert it to a string
            for col in df.columns:
                df[col] = df[col].apply(lambda x: str(x) if isinstance(x, list) else x)

            #lowercase all columns
            df.columns = df.columns.str.lower()

            #remove column id
            if 'id' in df.columns:
                df = df.drop(columns=['id'])

            #add file name to dataframe, replace - with _ in file name and remove extension
            df['file_name'] = d.replace('-', '_').split('.')[0]

            dataframes.append(df)

        except Exception as e:
            print(f"Error processing file {d}: {e}")
    print(f"Total dataframes processed successfully: {len(dataframes)}")

    #order dataframes by Number of rows
    dataframes = sorted(dataframes, key=lambda x: x.shape[0], reverse=True)
    return dataframes


