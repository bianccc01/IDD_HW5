import pandas as pd
import numpy as np
import random
import jsonlines
from sklearn.model_selection import train_test_split


def clean_value(x):
    if pd.isna(x):
        return "Unknown"
    if isinstance(x, (list, pd.Series, np.ndarray)):
        return '|'.join(map(str, x))
    if isinstance(x, str):
        x = x.strip("[]")
        x = x.replace(", ", "|")
        x = x.replace('[', '').replace(']', '')
    return str(x)


def format_record(row):
    parts = []
    for col, value in row.items():
        parts.append(f"COL {col} VAL {value}")
    return " ".join(parts)


def create_ditto_train_test_val(input_csv, train_file, test_file, val_file, jsonl_file,
                                train_csv, test_csv, val_csv,
                                test_size=0.2, val_size=0.2):
    df = pd.read_csv(input_csv)
    df = df.fillna("Unknown")
    df = df.applymap(clean_value)

    df = df[['company_name_1', 'company_name_2', 'company_country_1', 'company_country_2',
             'company_employees_1', 'company_employees_2', 'company_website_1', 'company_website_2', 'match']]

    train_df, test_val_df = train_test_split(df, test_size=test_size + val_size, stratify=df['match'], random_state=42)
    test_df, val_df = train_test_split(test_val_df, test_size=val_size / (test_size + val_size),
                                       stratify=test_val_df['match'], random_state=42)

    def process_and_save(df, output_txt, output_csv):
        lines = []
        formatted_rows = []

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
            formatted_rows.append([record1, record2, label])

        random.shuffle(lines)
        with open(output_txt, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")

        csv_df = pd.DataFrame(formatted_rows, columns=['record1', 'record2', 'match'])
        csv_df.to_csv(output_csv, index=False)

    def create_jsonl(df, output_file):
        with jsonlines.open(output_file, mode='w') as writer:
            for _, row in df.iterrows():
                record1 = format_record({
                    "company_name": row['company_name_1'],
                    "company_country": row['company_country_1'],
                    "company_employees": row['company_employees_1'],
                    "company_website": row['company_website_1']
                })
                record2 = format_record({
                    "company_name": row['company_name_2'],
                    "company_country": row['company_country_2'],
                    "company_employees": row['company_employees_2'],
                    "company_website": row['company_website_2']
                })
                writer.write([record1, record2])

    process_and_save(train_df, train_file, train_csv)
    process_and_save(test_df, test_file, test_csv)
    process_and_save(val_df, val_file, val_csv)
    create_jsonl(val_df, jsonl_file)

    print(f"✅ File di training salvato in {train_file} e {train_csv}")
    print(f"✅ File di test salvato in {test_file} e {test_csv}")
    print(f"✅ File di validazione salvato in {val_file} e {val_csv}")
    print(f"✅ File JSONL per la predizione salvato in {jsonl_file}")


create_ditto_train_test_val(
    "../../data/record_linkage/ditto/train/matches.csv",
    "../../data/record_linkage/ditto/train/ditto_train.txt",
    "../../data/record_linkage/ditto/train/ditto_test.txt",
    "../../data/record_linkage/ditto/train/ditto_val.txt",
    "../../ditto-master/data/input/ditto_input.jsonl",
    "../../data/record_linkage/ditto/train/ditto_train.csv",
    "../../data/record_linkage/ditto/train/ditto_test.csv",
    "../../data/record_linkage/ditto/train/ditto_val.csv"
)
