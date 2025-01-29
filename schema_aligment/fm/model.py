import Flexmatcher
import pandas as pd

def train_flexmatcher(schema_list, mapping_list):
    fm = Flexmatcher.FlexMatcher(schema_list, mapping_list, sample_size=1000)

    return fm.train()

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


test_flexmatcher()




