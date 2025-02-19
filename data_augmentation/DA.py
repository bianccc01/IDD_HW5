import random
import nltk
import pandas as pd
from nltk.corpus import wordnet

# Ensure you have the necessary NLTK resources
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('punkt_tab')


def get_synonyms(word):
    """Get synonyms for a given word using WordNet."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace('_', ' '))
    return list(synonyms)


def synonym_replacement(sentence, n=1):
    """Replace n words with synonyms in a sentence."""
    words = nltk.word_tokenize(sentence)
    random_words = random.sample(words, min(n, len(words)))
    new_sentence = words[:]

    for word in random_words:
        synonyms = get_synonyms(word)
        if synonyms:
            new_sentence = [random.choice(synonyms) if w == word else w for w in new_sentence]

    return ' '.join(new_sentence)


def random_insertion(sentence, n=1):
    """Randomly insert synonyms of words into the sentence."""
    words = nltk.word_tokenize(sentence)
    for _ in range(n):
        word = random.choice(words)
        synonyms = get_synonyms(word)
        if (synonyms):
            insert_pos = random.randint(0, len(words))
            words.insert(insert_pos, random.choice(synonyms))
    return ' '.join(words)


def random_deletion(sentence, p=0.1):
    """Randomly delete words from a sentence with probability p."""
    words = nltk.word_tokenize(sentence)
    if len(words) == 1:
        return sentence  # Avoid deleting the only word
    words = [word for word in words if random.random() > p]
    return ' '.join(words)


def word_swap(sentence, n=1):
    """Randomly swap two words in a sentence n times."""
    words = nltk.word_tokenize(sentence)
    for _ in range(n):
        if len(words) < 2:
            return sentence  # Not enough words to swap
        idx1, idx2 = random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
    return ' '.join(words)


def augment_text(sentence):
    """Apply a random augmentation technique to a sentence."""
    # Choose a random augmentation technique
    augmentation_fn = random.choice([synonym_replacement, random_insertion, random_deletion, word_swap])
    return augmentation_fn(sentence)


# Function to duplicate rows and apply augmentation
def duplicate_and_augment_rows(df, n=1):
    """Duplicate rows and apply augmentation to one of the columns randomly."""
    augmented_data = []

    for idx, row in df.iterrows():
        for _ in range(n):  # Duplicate the row n times
            new_row = row.copy()  # Copy the row to modify it

            # Randomly choose a column to augment
            col_to_augment = [
                'company_name_1', 'company_name_2',
                'company_country_1', 'company_country_2',
                'company_industry_1', 'company_industry_2',
                'company_employees_1', 'company_employees_2'
            ]

            for col in col_to_augment:
                if random.random() < 0.5:  # Apply augmentation with 50% probability
                    # Apply one augmentation technique
                    augmented_sentence = augment_text(str(new_row[col]))
                    print(f"Augmented {col} from '{new_row[col]}' to '{augmented_sentence}'")

                    # Update the row with the augmented value
                    new_row[col] = augmented_sentence

            # Append the augmented row to the list
            augmented_data.append(new_row)

    return pd.DataFrame(augmented_data)


# Example Usage
if __name__ == "__main__":

    input_csv = "../data/record_linkage/matches.tsv"  # Change this to your input CSV file
    output_csv = "../data/record_linkage/augmented_matches.csv"  # Change this to your output CSV file
    df = pd.read_csv(input_csv, sep='\t')

    # Duplicate rows and apply augmentation
    augmented_df = duplicate_and_augment_rows(df, n=3)  # Duplicate each row 3 times, for example

    # Save the augmented data to CSV
    augmented_df.to_csv(output_csv, index=False)
    print(f"Augmented data saved to {output_csv}")
