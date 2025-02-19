import pandas as pd
import numpy as np
import recordlinkage
from recordlinkage.index import Block, SortedNeighbourhood
from recordlinkage.preprocessing import clean
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
import re
from typing import Dict, List, Tuple
import json


class PairwiseMatchingAnalyzer:
    def __init__(self, data_path: str, sample_size: int = 1000, random_state: int = 42):
        self.data_path = data_path
        self.sample_size = sample_size
        self.random_state = random_state
        self.results = []
        self.best_strategy = None
        self.similarity_thresholds = {

            'strict': {
                'name': 0.85,
                'country': 0.9,
                'industry': 0.8,
                'city': 0.85
            },
            'moderate': {
                'name': 0.8,
                'country': 0.85,
                'industry': 0.75,
                'city': 0.8
            },
            'relaxed': {
                'name': 0.75,
                'country': 0.8,
                'industry': 0.7,
                'city': 0.75
            }
        }

    def clean_text(self, x: str) -> str:
        if isinstance(x, str):
            x = re.sub(r'[^\w\s&-]', '', x)
            x = re.sub(r'\s+', ' ', x)
            x = x.replace(' ltd ', ' limited ').replace(' inc ', ' incorporated ')
            x = x.replace(' corp ', ' corporation ').replace(' co ', ' company ')
            return x.lower().strip()
        return ''

    def prepare_data(self) -> pd.DataFrame:
        print("Caricamento dati...")
        df = pd.read_csv(self.data_path, low_memory=False)

        if self.sample_size:
            df = df.sample(n=self.sample_size, random_state=self.random_state)

        required_columns = ['company_name', 'company_country', 'company_industry', 'headquarters_region_city']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Colonne mancanti: {missing_columns}")

        print("Statistiche pre-pulizia:")
        self._print_data_stats(df, required_columns)

        for col in required_columns:
            df[col] = df[col].fillna('').astype(str).apply(self.clean_text)

        print("\nStatistiche post-pulizia:")
        self._print_data_stats(df, required_columns)

        return df

    def _print_data_stats(self, df: pd.DataFrame, columns: List[str]) -> None:
        for col in columns:
            empty = (df[col].fillna('') == '').sum()
            unique = df[col].nunique()
            print(f"{col}: {unique} valori unici, {empty} valori vuoti")

    def create_blocking_strategies(self) -> Dict:
        return {
            'single_field': [
                {'name': 'Name Block', 'fields': ['company_name']},
                {'name': 'Country Block', 'fields': ['company_country']},
                {'name': 'Industry Block', 'fields': ['company_industry']},
                {'name': 'City Block', 'fields': ['headquarters_region_city']},
                {'name': 'Website', 'fields': ['company_website']},
                {'name': 'Company SNM-3', 'fields': ['company_name'],
                 'method': 'sorted_neighbourhood', 'window': 3}
            ],
            'multi_field': [
                {'name': 'Name-Employees', 'fields': ['company_name', 'company_employees']},
                {'name': 'Name-Website', 'fields': ['company_name', 'company_website']},

                {'name': 'Country-Industry', 'fields': ['company_country', 'company_industry']},
                {'name': 'Country-Name', 'fields': ['company_country', 'company_name'],
                 'method': 'sorted_neighbourhood', 'window': 3}
            ],
            'complex': [
                {'name': 'Name-Website-Country', 'fields': ['company_name', 'company_website', 'company_country'],
                 'method': 'sorted_neighbourhood', 'window': 3}
            ]
        }

    def evaluate_strategy(self, strategy: Dict, df: pd.DataFrame, threshold_level: str = 'strict') -> Dict:
        start_time = time()

        # Inizializzazione indexer
        indexer = recordlinkage.Index()

        # Configurazione blocking
        if strategy.get('method') == 'sorted_neighbourhood':
            for field in strategy['fields']:
                indexer.sortedneighbourhood(field, window=strategy.get('window', 3))
        else:
            for field in strategy['fields']:
                indexer.block(field)

        # Generazione coppie e calcolo features
        pairs = indexer.index(df)

        # Configurazione comparatore
        compare = recordlinkage.Compare()
        thresholds = self.similarity_thresholds[threshold_level]

        compare.string('company_name', 'company_name',
                       method='jarowinkler', threshold=thresholds['name'],
                       label='name_sim')
        compare.string('company_country', 'company_country',
                       method='jarowinkler', threshold=thresholds['country'],
                       label='country_sim')
        compare.string('company_industry', 'company_industry',
                       method='levenshtein', threshold=thresholds['industry'],
                       label='industry_sim')
        compare.string('headquarters_region_city', 'headquarters_region_city',
                       method='jarowinkler', threshold=thresholds['city'],
                       label='city_sim')

        features = compare.compute(pairs, df)

        # Calcolo punteggio pesato
        weights = {
            'name_sim': 0.45,
            'country_sim': 0.25,
            'industry_sim': 0.20,
            'city_sim': 0.10
        }

        features['weighted_score'] = sum(features[col] * weights[col]
                                         for col in weights.keys())

        # Analisi matches con diverse soglie
        matches_strict = features[features['weighted_score'] > 0.8]
        matches_moderate = features[features['weighted_score'] > 0.7]
        matches_relaxed = features[features['weighted_score'] > 0.6]

        # Preparazione sample matches
        sample_matches = matches_moderate.head(5) if not matches_moderate.empty else pd.DataFrame()
        sample_matches_dict = sample_matches.reset_index().to_dict('records') if not sample_matches.empty else []

        execution_time = time() - start_time
        total_pairs = len(pairs)
        total_possible_pairs = len(df) * (len(df) - 1) / 2

        return {
            'strategy_name': strategy['name'],
            'total_pairs': total_pairs,
            'matches_strict': len(matches_strict),
            'matches_moderate': len(matches_moderate),
            'matches_relaxed': len(matches_relaxed),
            'avg_similarity_strict': matches_strict['weighted_score'].mean() if not matches_strict.empty else 0,
            'avg_similarity_moderate': matches_moderate['weighted_score'].mean() if not matches_moderate.empty else 0,
            'reduction_ratio': 1 - (total_pairs / total_possible_pairs),
            'execution_time': execution_time,
            'sample_matches': sample_matches_dict
        }

    def analyze_all_strategies(self) -> pd.DataFrame:
        print("Inizia analisi delle strategie...")
        df = self.prepare_data()
        strategies = self.create_blocking_strategies()

        for threshold_level in ['strict', 'moderate', 'relaxed']:
            print(f"\nAnalisi con soglia: {threshold_level}")

            for category, category_strategies in strategies.items():
                for strategy in category_strategies:
                    print(f"Valutazione strategia: {strategy['name']}")
                    result = self.evaluate_strategy(strategy, df, threshold_level)
                    result['category'] = category
                    result['threshold_level'] = threshold_level
                    self.results.append(result)

        self.results_df = pd.DataFrame(self.results)
        self.best_strategy = self._identify_best_strategy()

        return self.results_df

    def _identify_best_strategy(self) -> Dict:
        df = self.results_df

        df['score'] = (
                df['avg_similarity_moderate'] * 0.4 +
                df['reduction_ratio'] * 0.3 +
                (1 / (df['execution_time'] + 1)) * 0.2 +
                (df['matches_moderate'] / df['matches_moderate'].max()) * 0.1
        )

        return df.loc[df['score'].idxmax()].to_dict()

    def plot_results(self):
        plt.figure(figsize=(15, 10))

        sns.scatterplot(data=self.results_df,
                        x='reduction_ratio',
                        y='avg_similarity_moderate',
                        size='matches_moderate',
                        hue='category',
                        style='threshold_level',
                        sizes=(100, 1000),
                        alpha=0.6)

        plt.title('Confronto Strategie di Blocking')
        plt.xlabel('Reduction Ratio')
        plt.ylabel('Average Similarity (Moderate Threshold)')

        best = self.best_strategy
        plt.scatter(best['reduction_ratio'],
                    best['avg_similarity_moderate'],
                    color='red',
                    s=200,
                    marker='*',
                    label='Best Strategy')

        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('blocking_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def save_results(self):
        self.results_df.to_csv('blocking_analysis_results.csv', index=False)

        report = {
            'analysis_summary': {
                'total_strategies_evaluated': len(self.results_df),
                'best_strategy': {k: v for k, v in self.best_strategy.items()
                                  if isinstance(v, (str, int, float, bool)) or v is None},
                'execution_timestamp': str(pd.Timestamp.now())
            },
            'detailed_results': self.results_df.to_dict(orient='records'),
            'configuration': {
                'similarity_thresholds': self.similarity_thresholds,
                'sample_size': self.sample_size
            }
        }

        with open('blocking_analysis_report.json', 'w') as f:
            json.dump(report, f, indent=2)

        with open('blocking_analysis_report.txt', 'w') as f:
            f.write("ANALISI STRATEGIE DI BLOCKING\n")
            f.write("=" * 50 + "\n\n")

            f.write("MIGLIORE STRATEGIA:\n")
            for key, value in self.best_strategy.items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    f.write(f"{key}: {value}\n")

            f.write("\nCONFRONTO STRATEGIE:\n")
            comparison = self.results_df.groupby('strategy_name').agg({
                'matches_moderate': 'mean',
                'avg_similarity_moderate': 'mean',
                'reduction_ratio': 'mean',
                'execution_time': 'mean'
            }).round(3)

            f.write(comparison.to_string())


def main():
    analyzer = PairwiseMatchingAnalyzer(
        data_path="../../data/schema_alignment/created/fm/merged/merged_data.csv",
        sample_size=5000
    )

    print("Avvio analisi...")
    results = analyzer.analyze_all_strategies()

    print("\nGenerazione visualizzazioni...")
    analyzer.plot_results()

    print("\nSalvataggio risultati...")
    analyzer.save_results()

    print("\nAnalisi completata. I risultati sono stati salvati in:")
    print("- blocking_analysis_results.csv")
    print("- blocking_analysis_report.txt")
    print("- blocking_analysis_report.json")
    print("- blocking_comparison.png")


if __name__ == "__main__":
    main()