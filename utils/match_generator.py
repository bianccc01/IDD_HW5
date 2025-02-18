import pandas as pd
import random
from scipy import spatial
from itertools import combinations
from sentence_transformers import SentenceTransformer


model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def generate_matches(df, matches, non_matches):
    """
    Generates a dataset with a given number of matching and non-matching pairs based on Levenshtein similarity.

    :param df: DataFrame containing company records.
    :param matches: Number of matching pairs to generate.
    :param non_matches: Number of non-matching pairs to generate.
    :return: DataFrame containing pairs with a match label and all attributes from the input DataFrame.
    """

    # Ensure DataFrame has sufficient rows
    if len(df) < 2:
        raise ValueError("DataFrame must contain at least two records to generate matches/non-matches.")

    pairs = []

    # Creating all possible pairs using company name
    all_pairs = list(combinations(df.index, 2))

    similarity_scores = []
    for idx1, idx2 in all_pairs:

        print(f"Calculating similarity between {idx1} and {idx2}")

        #skip if the same row
        if idx1 == idx2:
            continue

        vec1 = df[['company_name', 'company_country']].loc[idx1]
        vec2 = df[['company_name', 'company_country']].loc[idx2]

        vec1 = ' '.join([str(val) for val in vec1])
        vec2 = ' '.join([str(val) for val in vec2])

        vec1 = model.encode(vec1)
        vec2 = model.encode(vec2)

        similarity = 1 - spatial.distance.cosine(vec1, vec2)
        
        similarity_scores.append((idx1, idx2, similarity))

    # Sort pairs based on similarity
    similarity_scores.sort(key=lambda x: x[2], reverse=True)

    print("Description of similarity scores")
    print(pd.Series([sim for _, _, sim in similarity_scores]).describe())

    # split the pairs into matches and non-matches with alpha
    match_pairs = []
    non_match_pairs = []

    for idx1, idx2, sim in similarity_scores:
        if sim >= 0.65:
            match_pairs.append((idx1, idx2, sim))
        else:
            non_match_pairs.append((idx1, idx2, sim))

    output_data = []

    for idx1, idx2, _ in match_pairs[:matches]:
        row = {'match': 1}
        for col in df.columns:
            row[f'{col}_1'] = df.loc[idx1, col]
            row[f'{col}_2'] = df.loc[idx2, col]
        output_data.append(row)

    for idx1, idx2, _ in non_match_pairs[:non_matches]:
        row = {'match': 0}
        for col in df.columns:
            row[f'{col}_1'] = df.loc[idx1, col]
            row[f'{col}_2'] = df.loc[idx2, col]
        output_data.append(row)

    return pd.DataFrame(output_data)


# Example usage
df = pd.read_csv('../data/record_linkage/benchmark.csv')
new_df = generate_matches(df, 100, 300)
new_df.to_csv('../data/record_linkage/matches.csv', index=False)
