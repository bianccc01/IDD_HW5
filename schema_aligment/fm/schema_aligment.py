import os

import data.extractor as extraction
import pandas as pd
import data.extractor as extraction
import model


def rename_dataframes():
    schema_mediato = {
        'company_name', 'company_address', 'company_industry', 'company_country',
        'company_market_cap', 'company_revenue', 'company_employees', 'company_website',
        'company_founded', 'company_capital', 'company_legal_form', 'company_status',
        'company_registration_date', 'company_type', 'company_nace_code', 'company_number',
        'company_headquarters', 'company_valuation', 'company_partners',
        'company_representatives', 'company_investors'
    }

    dataframes = extraction.extract_data('../../data/test')
    renamed_dataframes = []

    for df in dataframes:
        predictions = model.test_prediction(df)

        #print predictions and file name
        print(predictions, df['file_name'][0])

        # Mantieni file_name intatto e rinomina solo le altre colonne
        predicted_columns = {col: predictions[col] if col in predictions and col != 'file_name' else col for col in df.columns}
        df = df.rename(columns=predicted_columns)

        # Seleziona solo le colonne presenti nello schema mediato + file_name
        columns_to_keep = list(schema_mediato & set(df.columns))  # Solo colonne valide
        if 'file_name' in df.columns:
            columns_to_keep.append('file_name')  # Mantieni file_name

        df = df[columns_to_keep]
        renamed_dataframes.append(df)

    save_dataframes(renamed_dataframes)  # Salva i dataframe rinominati


def save_dataframes(dataframes):
    #save dataframes in data/renominated in csv format
    for df in dataframes:
        file_name = df['file_name'][0]
        df.to_csv(f'../../data/created/{file_name}.csv', index=False)


rename_dataframes()


