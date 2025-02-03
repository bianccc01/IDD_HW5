import Flexmatcher
import pandas as pd
import data.extractor as extraction
import os

def train_flexmatcher(schema_list, mapping_list):
    fm = Flexmatcher.FlexMatcher(schema_list, mapping_list)
    fm.train()

    return fm

def test():

    dataframes = extraction.extract_data('../../data/train')

    # Verifica che ci siano dataframe validi
    if not dataframes:
        print("Nessun dataframe valido trovato")
        return

    # Stampa info sui dataframe
    for i, df in enumerate(dataframes):
        print(f"Dataframe {i}: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Sample:\n{df.head(1)}\n")

    null_summary = df.isnull().sum()
    print(null_summary)

    data0_mapping = {
        'id': 'company_id',
        'name': 'company_name',
        'code': 'company_code',
        'legal form': 'company_legal_form',
        'status': 'company_status',
        'registration date': 'company_registration_date',
        'capital': 'company_capital',
        'address': 'company_address',
        'area of activity': 'company_area_of_activity',
        'emtak code': 'company_emtak_code',
        'nace code': 'company_nace_code',
        'partners': 'company_partners',
        'partner_participations': 'company_partner_participations',
        'partner_contributions': 'company_partner_contributions',
        'representatives': 'company_representatives',
        'representative_roles': 'company_representative_roles',
        'representative_start_dates': 'company_representative_start_dates',
        'deletion date': 'company_deletion_date'
    }

    data4_mapping = {
        'name': 'company_name',
        'headquarters': 'company_headquarters',
        'number_of_employees': 'company_employees',
        'address': 'company_address',
        'industry': 'company_industry',
        'website': 'company_website',
        'market_cap': 'company_market_cap',
        'telephone': 'company_telephone',
        'revenue': 'company_revenue'
    }

    data7_mapping = {
        'name': 'company_name',
        'valuation': 'company_valuation',
        'datejoined': 'company_date_joined',
        'country': 'company_country',
        'city': 'company_city',
        'industry': 'company_industry',
        'investors': 'company_investors',
        'founded': 'company_founded',
        'stage': 'company_stage',
        'totalraised': 'company_total_raised'
    }

    data1_mapping = {
        'id': 'company_id',
        'name': 'company_name',
        'address': 'company_address',
        'nation': 'company_nation',
        'hhid': 'company_hhid',
        'industry': 'company_industry',
        'sic_code': 'company_sic_code',
        'type': 'company_type',
        'est_of_ownership': 'company_ownership_type'
    }

    data3_mapping = {
        'name': 'company_name',
        'company_number': 'company_number',
        'registered_office_address': 'company_registered_office_address',
        'company_status': 'company_status',
        'company_type': 'company_type',
        'company_creation_date': 'company_creation_date',
        'nature_of_business': 'company_nature_of_business'
    }

    list_mapping = [data0_mapping, data1_mapping, data3_mapping, data4_mapping, data7_mapping]

    try:
        fm = train_flexmatcher(dataframes, list_mapping)
        print("FlexMatcher training finished")
    except Exception as e:
        print(f"Error with training: {str(e)}")









def test_flexmatcher():
    vals1 = [['year', 'Movie', 'imdb_rating'],
             ['2001', 'Lord of the Rings', '8.8'],
             ['2010', 'Inception', '8.7'],
             ['1999', 'The Matrix', '8.7']]
    header = vals1.pop(0)
    data1 = pd.DataFrame(vals1, columns=header)
    # creating the second dataset
    vals2 = [['title', 'produced', 'popularity'],
             ['The Godfather', '1972', '9.2'],
             ['Silver Linings Playbook', '2012', '7.8'],
             ['The Big Short', '2015', '7.8']]
    header = vals2.pop(0)
    data2 = pd.DataFrame(vals2, columns=header)
    # specifying the mappings for the first and second datasets
    data1_mapping = {'year': 'movie_year',
                     'imdb_rating': 'movie_rating',
                     'Movie': 'movie_name'}
    data2_mapping = {'popularity': 'movie_rating',
                     'produced': 'movie_year',
                     'title': 'movie_name'}

    schema_list = [data1, data2]
    mapping_list = [data1_mapping, data2_mapping]

    fm = train_flexmatcher(schema_list, mapping_list)

    vals3 = [['rt', 'id', 'yr'],
             ['8.5', 'The Pianist', '2002'],
             ['7.7', 'The Social Network', '2010']]
    header = vals3.pop(0)
    data3 = pd.DataFrame(vals3, columns=header)

    predicted_mapping = fm.make_prediction(data3)
    print(predicted_mapping)



test()



