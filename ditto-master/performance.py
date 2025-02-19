# Re-import necessary libraries and perform the task again
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Load the dataset again after the environment reset
file_path = 'data/output/ditto_predictions.csv'
file_path_DA = 'data/output/ditto_predictions_DA.csv'
data = pd.read_csv(file_path)
data_DA = pd.read_csv(file_path_DA)

# Calculate Precision, Recall, and F1 Score
precision = precision_score(data['ground_truth'], data['match'])
recall = recall_score(data['ground_truth'], data['match'])
f1 = f1_score(data['ground_truth'], data['match'])

precision_DA = precision_score(data_DA['ground_truth'], data_DA['match'])
recall_DA = recall_score(data_DA['ground_truth'], data_DA['match'])
f1_DA = f1_score(data_DA['ground_truth'], data_DA['match'])


# Define the metrics and their values
metrics = ['Precision', 'Recall', 'F1 Score']
metrics_DA = ['Precision_DA', 'Recall_DA', 'F1 Score_DA']

values = [precision, recall, f1]
values_DA = [precision_DA, recall_DA, f1_DA]

print('Ditto')
for metric, value in zip(metrics, values):
    print(f'{metric}: {value:.2f}')

print('\nDitto_DA')
for metric, value in zip(metrics_DA, values_DA):
    print(f'{metric}: {value:.2f}')


# Set positions for the bars
bar_width = 0.35
index = np.arange(len(metrics))

# Create the bar plot in pairs
plt.figure(figsize=(10, 6))
plt.bar(index, values, bar_width, color='skyblue', label='Ditto')
plt.bar(index + bar_width, values_DA, bar_width, color='salmon', label='Ditto_DA')

# Add labels and title
plt.title('Performance Metrics Comparison')
plt.ylabel('Score')
plt.xticks(index + bar_width / 2, metrics)
plt.ylim(0, 1)
plt.legend()

# Display the plot
plt.show()

