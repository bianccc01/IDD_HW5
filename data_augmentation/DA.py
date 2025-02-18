import random
import nltk
import pandas as pd
from nltk.corpus import wordnet

# Ensure you have the necessary NLTK resources
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')


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


def augment_text(sentence, num_augmentations=4):
    """Generate multiple augmented versions of the input sentence."""
    augmented_sentences = set()
    augmented_sentences.add(sentence)

    while len(augmented_sentences) < num_augmentations:
        choice = random.choice([synonym_replacement, random_insertion, random_deletion, word_swap])
        augmented_sentences.add(choice(sentence))

    return list(augmented_sentences)

# Example Usage
if __name__ == "__main__":
    input_csv = "../data/record_linkage/matches.csv"  # Change this to your input CSV file
    output_csv = "../data/augmented_matches.csv"  # Change this to your output CSV file
    df = pd.read_csv(input_csv)
    augmented_data = []

    for idx, row in df.iterrows():
        for col in df.columns:
            print(f"Augmenting data for column {col} in row {idx}")
            if pd.api.types.is_string_dtype(df[col]):
                augmented_sentences = augment_text(str(row[col]))
                for sentence in augmented_sentences:
                    augmented_data.append({col: sentence})

    augmented_df = pd.DataFrame(augmented_data)
    augmented_df.to_csv(output_csv, index=False)
    print(f"Augmented data saved to {output_csv}")