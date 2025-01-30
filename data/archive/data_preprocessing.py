import pandas as pd


def clean_and_merge_company_data():
    # Carica i dataframe
    df_activities = pd.read_csv('wissel-activity-ariregister.rik.ee.csv')
    df_companies = pd.read_csv('wissel-aziende-ariregister.rik.ee.csv')
    df_partners = pd.read_csv('wissel-partners-ariregister.rik.ee.csv')
    df_representatives = pd.read_csv('wissel-rappresentanti-ariregister.rik.ee.csv')

    # Pulizia dei dati
    # Rimuovi eventuali caratteri di newline e spazi extra
    for df in [df_activities, df_companies, df_partners, df_representatives]:
        for col in df.select_dtypes(include=['object']):
            df[col] = df[col].str.strip().str.replace('\n', '')

    # 1. Merge principale tra aziende e attività
    merged_df = pd.merge(
        df_companies,
        df_activities,
        left_on='ID',
        right_on='ID azienda',
        how='left'
    )

    # 2. Aggiungi informazioni dei partner
    partners_grouped = df_partners.groupby('ID azienda').agg({
        'Name': lambda x: ', '.join(x),
        'Participation': lambda x: ', '.join(x),
        'Contribution': lambda x: ', '.join(x)
    }).reset_index()
    partners_grouped.columns = ['ID', 'Partners', 'Partner_Participations', 'Partner_Contributions']

    merged_df = pd.merge(
        merged_df,
        partners_grouped,
        on='ID',
        how='left'
    )

    # 3. Aggiungi informazioni dei rappresentanti
    representatives_grouped = df_representatives.groupby('ID azienda').agg({
        'Name': lambda x: ', '.join(x),
        'Role': lambda x: ', '.join(x),
        'Start Date': lambda x: ', '.join(x)
    }).reset_index()
    representatives_grouped.columns = ['ID', 'Representatives', 'Representative_Roles', 'Representative_Start_Dates']

    merged_df = pd.merge(
        merged_df,
        representatives_grouped,
        on='ID',
        how='left'
    )

    # Riorganizza le colonne in modo logico
    columns_order = [
        'ID', 'Name', 'Code', 'Legal form', 'Status', 'Registration Date',
        'Capital', 'Address',
        'Area of Activity', 'EMTAK Code', 'NACE Code',
        'Partners', 'Partner_Participations', 'Partner_Contributions',
        'Representatives', 'Representative_Roles', 'Representative_Start_Dates',
        'Deletion Date'
    ]

    final_df = merged_df[columns_order].copy()

    return final_df


# Esecuzione e salvataggio
final_dataset = clean_and_merge_company_data()
final_dataset.to_csv('wissel-aziende-ariregister.csv', index=False)