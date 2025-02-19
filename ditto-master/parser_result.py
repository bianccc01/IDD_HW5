import jsonlines
import csv
import argparse


def parse_record(s):
    """
    Parsifica una stringa del tipo:
    "COL company_name VAL Sony COL company_country VAL Japan COL company_employees VAL 109,700 COL company_website VAL https://www.sony.com/"
    e restituisce un dizionario con le coppie chiave-valore.
    """
    record = {}
    # Dividiamo la stringa su "COL " (ignorando eventuali parti vuote)
    parts = s.strip().split("COL ")
    for part in parts:
        if not part.strip():
            continue
        # Ogni parte dovrebbe essere del tipo "nome_attributo VAL valore"
        if " VAL " in part:
            key, value = part.split(" VAL ", 1)
            record[key.strip()] = value.strip()
    return record


def jsonl_to_csv(jsonl_file, csv_file):
    """
    Legge il file JSONL e scrive un CSV con le colonne:
    company_name1, company_country1, company_employees1, company_website1,
    company_name2, company_country2, company_employees2, company_website2, match, match_confidence
    """
    fieldnames = [
        'company_name1', 'company_country1', 'company_employees1', 'company_website1',
        'company_name2', 'company_country2', 'company_employees2', 'company_website2',
        'match', 'match_confidence'
    ]

    with jsonlines.open(jsonl_file) as reader, open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for obj in reader:
            left_str = obj.get("left", "")
            right_str = obj.get("right", "")
            left_record = parse_record(left_str)
            right_record = parse_record(right_str)

            row = {
                'company_name1': left_record.get("company_name", ""),
                'company_country1': left_record.get("company_country", ""),
                'company_employees1': left_record.get("company_employees", ""),
                'company_website1': left_record.get("company_website", ""),
                'company_name2': right_record.get("company_name", ""),
                'company_country2': right_record.get("company_country", ""),
                'company_employees2': right_record.get("company_employees", ""),
                'company_website2': right_record.get("company_website", ""),
                'match': obj.get("match", ""),
                'match_confidence': obj.get("match_confidence", "")
            }
            writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converte un file JSONL in CSV")
    parser.add_argument("--jsonl", type=str, required=True, help="Percorso del file JSONL di input")
    parser.add_argument("--csv", type=str, required=True, help="Percorso del file CSV di output")
    args = parser.parse_args()

    jsonl_to_csv(args.jsonl, args.csv)
