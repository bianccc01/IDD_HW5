import os

from schema_aligment.fm.model import train_classifier
import data.extractor as extraction
import pandas as pd


def main():
    # Load the data
    file_path = '../data'
    dataframes = []

    for df in dataframes:
        print(f"Dataframe {df.shape[0]} rows")

    print(dataframes[0])

    combined_data = pd.concat(dataframes, ignore_index=True)

    # Train the classifier
    train_classifier(combined_data, 'char_dist')

    print('Training done.')


main()


