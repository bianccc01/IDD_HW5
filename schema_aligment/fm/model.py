import Flexmatcher
import pandas as pd
import pickle as pkl
import data.extractor as extraction
import os




def train_flexmatcher(schema_list, mapping_list):
    fm = Flexmatcher.FlexMatcher(schema_list, mapping_list, sample_size=600)
    fm.train()

    return fm







def test():

    # Modifica il tuo codice per usare il preprocessing
    dataframes = extraction.extract_data('../../data/train')

    # Verifica che ci siano dataframe validi
    if not dataframes:
        print("Nessun dataframe valido trovato")
        return

    #order dataframes by Name of the file
    dataframes = sorted(dataframes, key=lambda x: x.columns[0])

    # Stampa info sui dataframe
    for i, df in enumerate(dataframes):
        print(f"Dataframe {i}: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Sample:\n{df.head(1)}\n")

    null_summary = df.isnull().sum()
    print(null_summary)

    wissel_aziende_ariregister_mapping = {
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

    output_globaldata_mapping = {
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

    DDD_cbinsight_mapping = {
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

    hitHorizons_dataset_mapping = {
        'name': 'company_name',
        'address': 'company_address',
        'nation': 'company_nation',
        'hhid': 'company_hhid',
        'industry': 'company_industry',
        'sic_code': 'company_sic_code',
        'type': 'company_type',
        'est_of_ownership': 'company_ownership_type'
    }

    output_govuk_bigsize_mapping = {
        'name': 'company_name',
        'company_number': 'company_number',
        'registered_office_address': 'company_registered_office_address',
        'company_status': 'company_status',
        'company_type': 'company_type',
        'company_creation_date': 'company_creation_date',
        'nature_of_business': 'company_nature_of_business'
    }


    mapping_dict = {
        'wissel_aziende_ariregister': wissel_aziende_ariregister_mapping,
        'output_globaldata': output_globaldata_mapping,
        'DDD_cbinsight': DDD_cbinsight_mapping,
        'hitHorizons_dataset': hitHorizons_dataset_mapping,
        'output_govuk_bigsize': output_govuk_bigsize_mapping
    }

    list_mapping = []
    for d in dataframes:
        file_name = d['file_name'].iloc[0]
        if file_name in mapping_dict:
            list_mapping.append(mapping_dict[file_name])
        else:
            raise KeyError(f"Mapping for file {file_name} not found")

    #remove file_name column
    for i in range(len(dataframes)):
        dataframes[i] = dataframes[i].drop(columns=['file_name'])

    try:
        fm = train_flexmatcher(dataframes, list_mapping)
        print("FlexMatcher training finished")

        #save the model
        fm.save_model('flexmatcher.pkl')
    except Exception as e:
        print(f"Error with training: {str(e)}")







def test_prediction():
    # load the model
    with open('flexmatcher.pkl.model', 'rb') as f:
        fm = pkl.load(f)
    dataframes = extraction.extract_data('../../data/test')

    #remove file_name column
    for i in range(len(dataframes)):
        dataframes[i] = dataframes[i].drop(columns=['file_name'])

    for d in dataframes:
        predicted_mapping = fm.make_prediction(d)
        print(predicted_mapping)


test()
