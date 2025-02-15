import pandas as pd
import os
import data.extractor as extraction


def merge_csv(method):

    path = f'../data/schema_alignment/created/{method}'
    dataframes = extraction.extract_data(path)

    # concat dataframes
    merged_df = pd.concat(dataframes, ignore_index=True)

    # Salva il risultato
    merged_df.to_csv(os.path.join(f'{path}/merged', "merged_data.csv"), index=False)

    print("Merge compleated and saved in: ", os.path.join(f'{path}/merged', "merged_data.csv"))


merge_csv('fm')