import Flexmatcher
import pandas as pd
import data.extractor as extraction

def train_flexmatcher(schema_list, mapping_list):
    fm = Flexmatcher.FlexMatcher(schema_list, mapping_list, sample_size=1000)
    fm.train()

    return fm

def test():

    dataframes = extraction.extract_data('../../data/train')

    dataset12_mapping = {
        'name': 'company_name',
        'world_rank': 'company_rank',
        'annual_revenue_in_usd': 'revenue',
        'annual_net_income_in_usd': 'net_profit',
        'headquarters_region_city': 'headquarters',
        'headquarters_country': 'country',
        'company_business': 'industry',
        'number_of_employees': 'employees_count',
        'company_website': 'website',
        'total_assets_in_usd': 'total_assets',
        'total_liabilities_in_usd': 'total_liabilities',
        'total_equity_in_usd': 'total_equity',
        'ceo': 'chief_executive',
        'id': 'company_id'
    }

    dataset11_mapping = {
        'name': 'company_name',
        'company_number': 'registration_id',
        'registered_office_address': 'headquarters',
        'company_status': 'status',
        'company_type': 'legal_type',
        'company_creation_date': 'founding_date',
        'nature_of_business': 'industry'
    }

    dataset8_mapping = {
        'name': 'company_name',
        'country': 'country',
        'sales': 'revenue',
        'profit': 'net_profit',
        'assets': 'total_assets',
        'market value': 'market_value'
    }

    dataset10_mapping = {
        'name': 'company_name',
        'headquarters': 'headquarters',
        'number_of_employees': 'employees_count',
        'address': 'headquarters',
        'industry': 'industry',
        'website': 'website',
        'market_cap': 'market_value',
        'telephone': 'phone_number',
        'revenue': 'revenue'
    }

    mapping_list = [dataset8_mapping, dataset10_mapping, dataset11_mapping, dataset12_mapping]

    fm = train_flexmatcher(dataframes, mapping_list)

test()


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







