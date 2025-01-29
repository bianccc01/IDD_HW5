import os

from fm.model import train_classifier
import data.extractor as extraction
import pandas as pd


def main():
    # Load the data
    file_path = '../data'
    dataframes = []

    for d in os.listdir(file_path):
        dataframes.append(extraction.extract_data(file_path + '/' + d))

    print(dataframes[0])

    combined_data = pd.concat(dataframes, ignore_index=True)

    # Train the classifier
    train_classifier(combined_data, 'char_dist')

    print('Training done.')


main()


