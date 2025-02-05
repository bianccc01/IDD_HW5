import Flexmatcher
import pandas as pd
import pickle as pkl
import data.extractor as extraction
import os




def train_flexmatcher(schema_list, mapping_list):
    fm = Flexmatcher.FlexMatcher(schema_list, mapping_list, sample_size=12000)
    fm.train()

    return fm




def train():

    #TODO: Implementare il preprocessing dei dati drop colonne inutili

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

    disfold_mapping = {
        'link': 'company_website',
        'name': 'company_name',
        'headquarters': 'company_headquarters',
        'employees': 'company_employees',
        'ceo': 'company_ceo',
        'market_cap': 'company_market_cap'
    }

    wissel_aziende_ariregister_mapping = {
        'name': 'company_name',
        'code': 'company_vat_code',
        'legal form': 'company_legal_form',
        'status': 'company_status',
        'registration date': 'company_foundation_date',
        'capital': 'company_market_cap',
        'address': 'company_address',
        'area of activity': 'company_products_and_services',
        'nace code': 'company_nace_code'
    }

    output_globaldata_mapping = {
        'name': 'company_name',
        'headquarters': 'company_headquarters',
        'number_of_employees': 'company_employees',
        'address': 'company_address',
        'industry': 'company_industry',
        'website': 'company_website',
        'market_cap': 'company_market_cap',
        'telephone': 'company_phone_number',
        'revenue': 'company_revenue'
    }

    DDD_cbinsight_mapping = {
        'name': 'company_name',
        'valuation': 'company_market_cap',
        'country': 'company_country',
        'city': 'company_address',
        'industry': 'company_industry',
        'founded': 'company_foundation_date'
    }

    hitHorizons_dataset_mapping = {
        'name': 'company_name',
        'address': 'company_address',
        'nation': 'company_country',
        'industry': 'company_industry',
        'sic_code': 'company_sic_code',
        'type': 'company_legal_form',
        'est_of_ownership': 'company_foundation_date',
    }

    output_govuk_bigsize_mapping = {
        'name': 'company_name',
        'company_number': 'company_phone_number',
        'registered_office_address': 'company_address',
        'company_status': 'company_status',
        'company_type': 'company_legal_form',
        'company_creation_date': 'company_foundation_date',
        'nature_of_business': 'company_products_and_services',
    }


    mapping_dict = {
        'wissel_aziende_ariregister': wissel_aziende_ariregister_mapping,
        'output_globaldata': output_globaldata_mapping,
        'DDD_cbinsight': DDD_cbinsight_mapping,
        'hitHorizons_dataset': hitHorizons_dataset_mapping,
        'output_govuk_bigsize': output_govuk_bigsize_mapping,
        'disfold': disfold_mapping
    }



    list_mapping = []
    for i, d in enumerate(dataframes):
        file_name = d['file_name'].iloc[0]
        mapping = mapping_dict.get(file_name)
        dataframes[i] = d[mapping.keys()]
        list_mapping.append(mapping)


    try:
        fm = train_flexmatcher(dataframes, list_mapping)
        print("FlexMatcher training finished")

        #save the model
        fm.save_model('flexmatcher.pkl')
    except Exception as e:
        print(f"Error with training: {str(e)}")

train()


def test_prediction(dataframe):

    # load the model
    with open('flexmatcher.pkl.model', 'rb') as f:
        fm = pkl.load(f)

    # remove file_name column
    file_name = dataframe['file_name'].iloc[0]
    dataframe = dataframe.drop(columns=['file_name'])

    prediction = fm.make_prediction(dataframe)

    #add file_name to the prediction
    prediction['file_name'] = file_name

    return prediction


