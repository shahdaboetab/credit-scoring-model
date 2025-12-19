"""
Alternative single-script entry point for training the credit scoring model.
Simplified version of src/main.py for quick training.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os

def load_data():
    """
    Load the German Credit dataset from UCI repository.

    Returns:
        pd.DataFrame: Loaded dataset
    """
    import requests

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
    response = requests.get(url)

    # Define column names
    column_names = [
        'status', 'duration', 'credit_history', 'purpose', 'credit_amount',
        'savings', 'employment', 'installment_rate', 'personal_status', 'other_debtors',
        'residence_since', 'property', 'age', 'other_installment', 'housing',
        'existing_credits', 'job', 'liable_people', 'telephone', 'foreign_worker', 'target'
    ]

    # Create data directory if it doesn't exist
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)

    # Save raw data
    with open('data/raw/german_credit_data.csv', 'w') as f:
        f.write(response.text)

    # Load data
    df = pd.read_csv('data/raw/german_credit_data.csv', sep=' ', header=None, names=column_names)

    # Convert target: 1 = good, 2 = bad -> 0 = good, 1 = bad
    df['target'] = df['target'].map({1: 0, 2: 1})

    return df

def preprocess_data(df):
    """
    Preprocess the data: handle categorical variables, scale numerical features.

    Args:
        df (pd.DataFrame): Raw dataset

    Returns:
        tuple: (X, y, preprocessor, feature_names)
    """
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first', sparse=False), categorical_cols)
        ]
    )

    # Fit and transform
    X_processed = preprocessor.fit_transform(X)

    # Get feature names
    feature_names = numerical_cols + [
        f"{col}_{val}" for col in categorical_cols
        for val in preprocessor.named_transformers_['cat'].categories_[categorical_cols.index(col)][1:]
    ]

    return X_processed, y, preprocessor, feature_names

def train_and_evaluate():
    """
    Train and evaluate all models.
    """
    print("Loading data...")
    df = load_data()
    print(f"Dataset shape: {df.shape}")

    print("Preprocessing data...")
    X, y, preprocessor, feature_names = preprocess_data(df)

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    }

    # Train and evaluate models
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }

        results[name] = metrics

        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall: {metrics['recall']:.3f}")
        print(f"F1-Score: {metrics['f1_score']:.3f}")
        print(f"ROC-AUC: {metrics['roc_auc']:.3f}")

        # Save model
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, f'models/{name.lower().replace(" ", "_")}.pkl')

    # Save preprocessor
    joblib.dump(preprocessor, 'models/preprocessor.pkl')

    # Save processed data
    processed_df = pd.DataFrame(X, columns=feature_names)
    processed_df['target'] = y
    processed_df.to_csv('data/processed/german_credit_processed.csv', index=False)

    print("\nTraining completed!")
    print("Models saved to 'models/' directory")
    print("Processed data saved to 'data/processed/'")

    return results

if __name__ == "__main__":
    results = train_and_evaluate()
    print("\nFinal Results:")
    for model, metrics in results.items():
        print(f"{model}: ROC-AUC = {metrics['roc_auc']:.3f}")
