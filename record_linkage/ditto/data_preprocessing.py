import pandas as pd
import numpy as np
import random
import json
from sklearn.model_selection import train_test_split


def clean_value(x):
    if pd.isna(x):
        return "Unknown"
    if isinstance(x, (list, pd.Series, np.ndarray)):
        return '|'.join(map(str, x))
    if isinstance(x, str):
        # Rimuovi le parentesi quadre esterne, se presenti
        x = x.strip("[]")
        # Sostituisci le virgole con pipe
        x = x.replace(", ", "|")
        # Rimuovi eventuali parentesi quadre residue
        x = x.replace('[', '').replace(']', '')
    return str(x)



def format_record(row):
    """
    Converte una riga del DataFrame in una stringa nel formato:
      COL <nome_colonna> VAL <valore>
    per tutte le colonne presenti.
    """
    parts = []
    for col, value in row.items():
        parts.append(f"[COL] {col} [VAL] {value}")
    return " ".join(parts)


def create_ditto_train_test_val(input_csv, train_file, test_file, val_file, jsonl_file, test_size=0.2):
    """
    Carica un CSV contenente coppie consecutive e il valore di match,
    lo splitta in train/test/validation e lo salva in formato Ditto e JSONL.
    """
    df = pd.read_csv(input_csv)
    df = df.fillna("Unknown")
    df = df.applymap(clean_value)

    # Seleziona solo le colonne necessarie
    df = df[['company_name_1', 'company_name_2', 'company_country_1', 'company_country_2',
             'company_employees_1', 'company_employees_2', 'company_website_1', 'company_website_2', 'match']]

    # Split train/test
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df['match'])
    val_df = test_df.copy()  # Validation uguale al test

    def process_and_save(df, output_file):
        lines = []
        for _, row in df.iterrows():
            record1 = format_record({
                'company_name': row['company_name_1'],
                'company_country': row['company_country_1'],
                'company_employees': row['company_employees_1'],
                'company_website': row['company_website_1']
            })
            record2 = format_record({
                'company_name': row['company_name_2'],
                'company_country': row['company_country_2'],
                'company_employees': row['company_employees_2'],
                'company_website': row['company_website_2']
            })
            label = int(row['match'])
            lines.append(f"{record1} \t {record2} \t {label}")

        random.shuffle(lines)
        with open(output_file, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")

    def create_jsonl(df, output_file):
        records = []
        for _, row in df.iterrows():
            record = {
                "record1": {
                    "company_name": row['company_name_1'],
                    "company_country": row['company_country_1'],
                    "company_employees": row['company_employees_1'],
                    "company_website": row['company_website_1']
                },
                "record2": {
                    "company_name": row['company_name_2'],
                    "company_country": row['company_country_2'],
                    "company_employees": row['company_employees_2'],
                    "company_website": row['company_website_2']
                }
            }
            records.append(json.dumps(record))

        with open(output_file, "w", encoding="utf-8") as f:
            for record in records:
                f.write(record + "\n")

    process_and_save(train_df, train_file)
    process_and_save(test_df, test_file)
    process_and_save(val_df, val_file)
    create_jsonl(test_df, jsonl_file)

    print(f"✅ File di training salvato in {train_file}")
    print(f"✅ File di test salvato in {test_file}")
    print(f"✅ File di validazione salvato in {val_file}")
    print(f"✅ File JSONL per la predizione salvato in {jsonl_file}")


# Esempio di utilizzo:
create_ditto_train_test_val(
    "../../data/record_linkage/ditto/train/matches.csv",
    "../../data/record_linkage/ditto/train/ditto_train.txt",
    "../../data/record_linkage/ditto/train/ditto_test.txt",
    "../../data/record_linkage/ditto/train/ditto_val.txt",
    "../../data/record_linkage/ditto/train/ditto_input.jsonl"
)
