"""
Implement FlexMatcher.

This module is the main module of the FlexMatcher package and implements the
FlexMatcher class.

Todo:
    * Extend the module to work with and without data or column names.
    * Allow users to add/remove classifiers.
    * Combine modules (i.e., create_training_data and training functions).
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import Flexmatcher.classify as clf
import Flexmatcher.utils as utils
from sklearn import linear_model
import numpy as np
import pandas as pd
import pickle
import time


class FlexMatcher:

    """Match a given schema to the mediated schema.

    The FlexMatcher learns to match an input schema to a mediated schema.
    The class considers panda dataframes as databases and their column names as
    the schema. FlexMatcher learn to do schema matching by training on
    instances of dataframes and how their columns are matched against the
    mediated schema.

    Attributes:
        train_data (dataframe): Dataframe with 3 columns. The name of
            the column in the schema, the value under that column and the name
            of the column in the mediated schema it was mapped to.
        col_train_data (dataframe): Dataframe  with 2 columns. The name
            the column in the schema and the name of the column in the mediated
            schema it was mapped to.
        data_src_num (int): Store the number of available data sources.
        classifier_list (list): List of classifiers used in the training.
        classifier_type (string): List containing the type of each classifier.
            Possible values are 'column' and 'value' classifiers.
        prediction_list (list): List of predictions on the training data
            produced by each classifier.
        weights (ndarray): A matrix where cell (i,j) captures how good the j-th
            classifier is at predicting if a column should match the i-th
            column (where columns are sorted by name) in the mediated schema.
        columns (list): The sorted list of column names in the mediated schema.
    """

    def __init__(self, dataframes, mappings, sample_size=300):
        """Prepares the list of classifiers that are being used for matching
        the schemas and creates the training data from the input datafames
        and their mappings.

        Args:
            dataframes (list): List of dataframes to train on.
            mapping (list): List of dictionaries mapping columns of dataframes
                to columns in the mediated schema.
            sample_size (int): The number of rows sampled from each dataframe
                for training.
        """
        print('Create training data ...')
        self.create_training_data(dataframes, mappings, sample_size)
        print('Training data done ...')
        unigram_count_clf = clf.NGramClassifier(ngram_range=(1, 1))
        bigram_count_clf = clf.NGramClassifier(ngram_range=(2, 2))
        unichar_count_clf = clf.NGramClassifier(analyzer='char_wb',
                                                ngram_range=(1, 1))
        bichar_count_clf = clf.NGramClassifier(analyzer='char_wb',
                                               ngram_range=(2, 2))
        trichar_count_clf = clf.NGramClassifier(analyzer='char_wb',
                                                ngram_range=(3, 3))
        quadchar_count_clf = clf.NGramClassifier(analyzer='char_wb',
                                                 ngram_range=(4, 4))
        char_dist_clf = clf.CharDistClassifier()
        self.classifier_list = [unigram_count_clf, bigram_count_clf,
                                unichar_count_clf, bichar_count_clf,
                                trichar_count_clf, quadchar_count_clf,
                                char_dist_clf]
        self.classifier_type = ['value', 'value', 'value', 'value',
                                'value', 'value', 'value']
        if self.data_src_num > 5:
            col_char_dist_clf = clf.CharDistClassifier()
            col_trichar_count_clf = clf.NGramClassifier(analyzer='char_wb',
                                                        ngram_range=(3, 3))
            col_quadchar_count_clf = clf.NGramClassifier(analyzer='char_wb',
                                                         ngram_range=(4, 4))
            col_quintchar_count_clf = clf.NGramClassifier(analyzer='char_wb',
                                                          ngram_range=(5, 5))
            col_word_count_clf = \
                clf.NGramClassifier(analyzer=utils.columnAnalyzer)
            knn_clf = \
                clf.KNNClassifier()
            self.classifier_list = self.classifier_list + \
                                   [col_char_dist_clf, col_trichar_count_clf,
                                    col_quadchar_count_clf, col_quintchar_count_clf,
                                    col_word_count_clf, knn_clf]
            self.classifier_type = self.classifier_type + (['column'] * 6)

    def create_training_data(self, dataframes, mappings, sample_size):
        """Transform dataframes and mappings into training data.

        The method uses the names of columns as well as the data under each
        column as its training data. It also replaces missing values with 'NA'.

        Args:
            dataframes (list): List of dataframes to train on.
            mapping (list): List of dictionaries mapping columns of dataframes
                to columns in the mediated schema.
            sample_size (int): The number of rows sampled from each dataframe
                for training.
        """
        train_data_list = []
        col_train_data_list = []
        for (datafr, mapping) in zip(dataframes, mappings):
            sampled_rows = datafr.sample(min(sample_size, datafr.shape[0]))
            sampled_data = pd.melt(sampled_rows)
            sampled_data.columns = ['name', 'value']
            sampled_data['class'] = \
                sampled_data.apply(lambda row: mapping[row['name']], axis=1)
            train_data_list.append(sampled_data)
            col_data = pd.DataFrame(datafr.columns)
            col_data.columns = ['name']
            col_data['value'] = col_data['name']
            col_data['class'] = \
                col_data.apply(lambda row: mapping[row['name']], axis=1)
            col_train_data_list.append(col_data)
        train_data = pd.concat(train_data_list, ignore_index=True)
        self.train_data = train_data.fillna('NA')
        self.col_train_data = pd.concat(col_train_data_list, ignore_index=True)
        self.col_train_data = \
            self.col_train_data.drop_duplicates().reset_index(drop=True)
        self.data_src_num = len(dataframes)
        self.columns = \
            sorted(list(set.union(*[set(x.values()) for x in mappings])))
        # removing columns that are not present in the dataframe
        # TODO: this should change (It's not ideal to change problem definition
        # without notifying the user)
        available_columns = []
        for (datafr, mapping) in zip(dataframes, mappings):
            for c in datafr.columns:
                available_columns.append(mapping[c])
        self.columns = sorted(list(set(available_columns)))

    def train(self):
        """Train each classifier and the meta-classifier."""
        self.prediction_list = []
        for (clf_inst, clf_type) in zip(self.classifier_list,
                                        self.classifier_type):
            start = time.time()
            # fitting the models and predict for training data
            if clf_type == 'value':
                clf_inst.fit(self.train_data)
                # predicting the training data
                self.prediction_list.append(clf_inst.predict_training())
            elif clf_type == 'column':
                clf_inst.fit(self.col_train_data)
                # predicting the training data
                col_data_prediction = \
                    pd.concat([pd.DataFrame(clf_inst.predict_training()),
                               self.col_train_data], axis=1)
                data_prediction = self.train_data.merge(col_data_prediction,
                                                        on=['name', 'class'],
                                                        how='left')
                data_prediction = np.asarray(data_prediction)
                data_prediction = \
                    data_prediction[:, range(3, 3 + len(self.columns))]
                self.prediction_list.append(data_prediction)
            print(time.time() - start)

        start = time.time()
        self.train_meta_learner()
        print('Meta: ' + str(time.time() - start))

    def train_meta_learner(self):
        """Train the meta-classifier.

        The data used for training the meta-classifier is the probability of
        assigning each point to each column (or class) by each classifier. The
        learned weights suggest how good each classifier is at predicting a
        particular class."""
        # suppressing a warning from scipy that gelsd is broken and gless is
        # being used instead.
        # warnings.filterwarnings(action="ignore", module="scipy",
        #                        message="^internal gelsd")
        coeff_list = []
        for class_ind, class_name in enumerate(self.columns):
            # preparing the dataset for logistic regression
            regression_data = self.train_data[['class']].copy()
            regression_data['is_class'] = \
                np.where(self.train_data['class'] == class_name, True, False)
            # adding the prediction probability from classifiers
            for classifier_ind, prediction in enumerate(self.prediction_list):
                regression_data['classifer' + str(classifier_ind)] = \
                    prediction[:, class_ind]

            # setting up the logistic regression
            stacker = linear_model.LogisticRegression(fit_intercept=True,
                                                      class_weight='balanced', max_iter=1500)
            stacker.fit(regression_data.iloc[:, 2:],
                        regression_data['is_class'])
            coeff_list.append(stacker.coef_.reshape(1, -1))
        self.weights = np.concatenate(tuple(coeff_list))

    def make_prediction(self, data, adaptive_threshold=None, return_confidence=False):
        """Predict the mapping of columns in the input data to the mediated
        schema."""

        # Pre-elaborazione: sostituisce i NaN e limita il numero di righe se necessario
        data = data.fillna('NA').copy(deep=True)

        predicted_mapping = {}
        confidence_details = {}
        default_column = "unknown_column"
        confidence_threshold = adaptive_threshold if adaptive_threshold is not None else 0.17

        for column in data.columns:
            # Prepara i dati per i classificatori:
            # - column_dat contiene i valori della colonna rinominata in 'value'
            # - column_name contiene il nome della colonna ripetuto per tutte le righe
            column_dat = data[[column]].copy()
            column_dat.columns = ['value']
            column_name = pd.DataFrame({'value': [column] * len(column_dat)})

            # Inizializza la matrice dei punteggi per ogni riga e per ogni colonna target
            scores = np.zeros((len(column_dat), len(self.columns)))

            # Aggrega le predizioni da ciascun classificatore
            for clf_ind, clf_inst in enumerate(self.classifier_list):
                # Scegli il tipo di input in base al tipo di classificatore
                if self.classifier_type[clf_ind] == 'value':
                    raw_prediction = clf_inst.predict(column_dat)
                elif self.classifier_type[clf_ind] == 'column':
                    raw_prediction = clf_inst.predict(column_name)
                else:
                    continue  # Salta il classificatore se il tipo non è riconosciuto

                # Normalizzazione: se la somma lungo l'asse delle classi non è 1, la forziamo
                row_sums = raw_prediction.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1  # evita divisioni per 0
                raw_prediction = raw_prediction / row_sums

                # Applica il peso specifico per ogni classe
                for class_ind in range(len(self.columns)):
                    raw_prediction[:, class_ind] *= self.weights[class_ind, clf_ind]

                # Somma le predizioni ponderate
                scores += raw_prediction

            # Calcola i punteggi aggregati medi per ciascuna classe
            flat_scores = scores.sum(axis=0) / len(column_dat)
            total_score = flat_scores.sum()

            # Se il totale è zero, non è possibile determinare una confidenza
            if total_score == 0:
                predicted_mapping[column] = default_column
                if return_confidence:
                    confidence_details[column] = {
                        'confidence_ratio': 0,
                        'margin': 0,
                        'normalized_margin': 0,
                        'combined_confidence': 0
                    }
                continue

            # Ordina i punteggi per identificare il top-1 e il top-2
            sorted_indices = np.argsort(flat_scores)[::-1]
            top1_index = sorted_indices[0]
            top1_score = flat_scores[top1_index]
            top2_score = flat_scores[sorted_indices[1]] if len(flat_scores) > 1 else 0.0

            # Calcola le metriche di confidenza:
            confidence_ratio = top1_score / total_score
            margin = top1_score - top2_score
            normalized_margin = margin / total_score

            # Combina le metriche: i pesi possono essere regolati in base agli esperimenti
            combined_confidence = 0.7 * confidence_ratio + 0.3 * normalized_margin

            # Se la confidenza combinata è inferiore alla soglia, assegna il valore di default
            if combined_confidence < confidence_threshold:
                predicted_mapping[column] = default_column
            else:
                predicted_mapping[column] = self.columns[top1_index]

            # Se richiesto, salva i dettagli della confidenza per la colonna
            if return_confidence:
                confidence_details[column] = {
                    'confidence_ratio': confidence_ratio,
                    'margin': margin,
                    'normalized_margin': normalized_margin,
                    'combined_confidence': combined_confidence
                }

        if return_confidence:
            return predicted_mapping, confidence_details
        else:
            return predicted_mapping

    def save_model(self, name):
        """Serializes the FlexMatcher object into a model file using python's
        pickle library."""
        with open(name + '.model', 'wb') as f:
            pickle.dump(self, f)

