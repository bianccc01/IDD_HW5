import pandas as pd
import data.extractor as extraction
import model



def combine_duplicate_columns(df):
    """
    Se per un dato nome di colonna compaiono più colonne,
    le combina in una singola colonna, dove per ogni riga il valore è una lista
    contenente i valori provenienti da ciascuna colonna duplicata.
    Se la colonna compare una sola volta, il valore rimane invariato.
    Per la colonna 'company_name', prende solo il primo valore che incontra.
    """
    new_data = {}
    for col in df.columns.unique():
        sub_df = df.loc[:, df.columns == col]
        if sub_df.shape[1] > 1:
            if col == "company_name":
                new_data[col] = sub_df.apply(lambda row: row.dropna().iloc[0] if not row.dropna().empty else None, axis=1)
            else:
                new_data[col] = sub_df.apply(lambda row: row.tolist(), axis=1)
        else:
            new_data[col] = sub_df.iloc[:, 0]
    new_df = pd.DataFrame(new_data)
    return new_df



def rename_dataframes():
    # Definisco lo schema mediato come lista per mantenere l'ordine desiderato
    schema_mediato = [
        'company_name', 'company_address', 'company_industry', 'company_country',
        'company_market_cap', 'company_revenue', 'company_employees', 'company_website',
        'company_founded', 'company_capital', 'company_legal_form', 'company_status',
        'company_registration_date', 'company_type', 'company_nace_code', 'company_number',
        'company_headquarters', 'company_valuation', 'company_partners',
        'company_representatives', 'company_investors'
    ]

    dataframes = extraction.extract_data('../../data/schema_alignment/test')
    renamed_dataframes = []

    for df in dataframes:
        predictions = model.test_prediction(df)
        #if value for prediction is unknown_column make ugual to key and add key to schema_mediato
        for key, value in predictions.items():
            if value == 'unknown_column':
                predictions[key] = key
                schema_mediato.append(key)
        print(predictions, df['file_name'][0])

        # Rinomina le colonne (mantieni file_name intatto)
        predicted_columns = {
            col: predictions[col] if col in predictions and col != 'file_name' else col
            for col in df.columns
        }
        df = df.rename(columns=predicted_columns)

        # Combina eventuali colonne duplicate
        df = combine_duplicate_columns(df)

        # Seleziona le colonne nello stesso ordine definito in schema_mediato
        columns_to_keep = [col for col in schema_mediato if col in df.columns]
        if 'file_name' in df.columns:
            columns_to_keep.append('file_name')  # Aggiungo file_name alla fine

        df = df[columns_to_keep]
        renamed_dataframes.append(df)

    save_dataframes(renamed_dataframes)


def save_dataframes(dataframes):
    for df in dataframes:
        file_name = df['file_name'][0]
        df.to_csv(f'../../data/schema_alignment/created/fm/{file_name}.csv', index=False)


rename_dataframes()