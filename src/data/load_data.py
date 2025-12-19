"""
Functions to load and download data for the credit scoring model.
"""

import pandas as pd
import requests
import os

def load_data(data_path):
    """
    Load the German Credit dataset.

    Args:
        data_path (str): Path to the data file

    Returns:
        pd.DataFrame: Loaded dataset
    """
    if not os.path.exists(data_path):
        print(f"Downloading data from UCI repository...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
        response = requests.get(url)
        with open(data_path, 'w') as f:
            f.write(response.text)

        # Also download the attribute names
        names_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.doc"
        response = requests.get(names_url)
        with open(data_path.replace('.data', '.doc'), 'w') as f:
            f.write(response.text)

    # Load the data (no header, space-separated)
    column_names = [
        'status', 'duration', 'credit_history', 'purpose', 'credit_amount',
        'savings', 'employment', 'installment_rate', 'personal_status', 'other_debtors',
        'residence_since', 'property', 'age', 'other_installment', 'housing',
        'existing_credits', 'job', 'liable_people', 'telephone', 'foreign_worker', 'target'
    ]

    df = pd.read_csv(data_path, sep=' ', header=None, names=column_names)

    # Convert target: 1 = good, 2 = bad -> 0 = good, 1 = bad
    df['target'] = df['target'].map({1: 0, 2: 1})

    return df

if __name__ == "__main__":
    df = load_data("data/raw/german_credit_data.csv")
    print(f"Dataset shape: {df.shape}")
    print(df.head())
