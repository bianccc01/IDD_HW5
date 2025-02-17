import pandas as pd
import numpy as np
import random

def clean_value(x):
    if pd.isna(x):
        return "Unknown"
    if isinstance(x, (list, pd.Series, np.ndarray)):
        return '|'.join(map(str, x))
    if isinstance(x, str):
        if x.startswith('[') and x.endswith(']'):
            x = x[1:-1]
    return str(x)


def format_record(row):
    """
    Converte una riga del DataFrame in una stringa nel formato:
      COL <nome_colonna> VAL <valore>
    per tutte le colonne presenti.
    """
    parts = []
    for col, value in row.items():
        parts.append(f"COL {col} VAL {value}")
    return " ".join(parts)

def create_ditto_train_file(match_csv, non_match_csv, output_file):
    """
    Crea il file di training per Ditto combinando coppie matching e non-matching.

    :param match_csv: percorso del CSV con le coppie matching (righe in coppia)
    :param non_match_csv: percorso del CSV con le coppie non-matching (righe in coppia)
    :param output_file: percorso del file di output (es. "ditto_train.txt")
    """
    # Carica i CSV
    df_match = pd.read_csv(match_csv)
    df_non_match = pd.read_csv(non_match_csv)

    # Applica la pulizia dei dati a tutte le colonne
    for col in df_match.columns:
        df_match[col] = df_match[col].apply(clean_value)
    for col in df_non_match.columns:
        df_non_match[col] = df_non_match[col].apply(clean_value)

    # Rimuove la colonna file_name se presente
    if 'file_name' in df_match.columns:
        df_match = df_match.drop(columns=['file_name'])
    if 'file_name' in df_non_match.columns:
        df_non_match = df_non_match.drop(columns=['file_name'])

    df_match = df_match.fillna("Unknown")
    df_non_match = df_non_match.fillna("Unknown")

    #take only company_name column
    df_match = df_match[['company_name']]
    df_non_match = df_non_match[['company_name']]

    train_lines = []

    # Processa le coppie matching (etichetta 1)
    for i in range(0, len(df_match), 2):
        if i + 1 < len(df_match):
            record1 = format_record(df_match.iloc[i])
            record2 = format_record(df_match.iloc[i + 1])
            line = f"{record1} \t {record2} \t 1"
            train_lines.append(line)

    # Processa le coppie non-matching (etichetta 0)
    for i in range(0, len(df_non_match), 2):
        if i + 1 < len(df_non_match):
            record1 = format_record(df_non_match.iloc[i])
            record2 = format_record(df_non_match.iloc[i + 1])
            line = f"{record1} \t {record2} \t 0"
            train_lines.append(line)

    # Mescola (shuffle) le linee
    random.shuffle(train_lines)

    # Salva il file di training
    with open(output_file, "w", encoding="utf-8") as f:
        for line in train_lines:
            f.write(line + "\n")

    print(f"✅ File di training salvato in {output_file}")

# Esempio di utilizzo:
create_ditto_train_file(
    "../../data/record_linkage/ditto/train/match.csv",
    "../../data/record_linkage/ditto/train/not_match.csv",
    "../../ditto-master/data/train/ditto_train.txt"
)
