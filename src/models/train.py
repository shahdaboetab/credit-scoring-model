"""
Training logic for all models in the credit scoring project.
"""

import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def train_logistic_regression(X_train, y_train, C=1.0, random_state=42):
    """
    Train Logistic Regression model.

    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        C (float): Regularization parameter
        random_state (int): Random state

    Returns:
        model: Trained Logistic Regression model
    """
    model = LogisticRegression(C=C, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def train_decision_tree(X_train, y_train, max_depth=10, random_state=42):
    """
    Train Decision Tree model.

    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        max_depth (int): Maximum depth of the tree
        random_state (int): Random state

    Returns:
        model: Trained Decision Tree model
    """
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train, n_estimators=100, max_depth=10, random_state=42):
    """
    Train Random Forest model.

    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        n_estimators (int): Number of trees
        max_depth (int): Maximum depth of each tree
        random_state (int): Random state

    Returns:
        model: Trained Random Forest model
    """
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model.

    Args:
        model: Trained model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target

    Returns:
        dict: Evaluation metrics
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }

    return metrics

def save_model(model, model_path):
    """
    Save a trained model to disk.

    Args:
        model: Trained model
        model_path (str): Path to save the model
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

def load_model(model_path):
    """
    Load a trained model from disk.

    Args:
        model_path (str): Path to the saved model

    Returns:
        model: Loaded model
    """
    return joblib.load(model_path)

if __name__ == "__main__":
    from src.data.load_data import load_data
    from src.data.preprocess import preprocess_data, split_data

    # Load and preprocess data
    df = load_data("data/raw/german_credit_data.csv")
    df_processed, _, _ = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df_processed)

    # Train models
    lr_model = train_logistic_regression(X_train, y_train)
    dt_model = train_decision_tree(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)

    # Evaluate models
    models = {'Logistic Regression': lr_model, 'Decision Tree': dt_model, 'Random Forest': rf_model}
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test)
        print(f"{name} - Accuracy: {metrics['accuracy']:.3f}, ROC-AUC: {metrics['roc_auc']:.3f}")

    # Save models
    save_model(lr_model, "models/logistic_regression.pkl")
    save_model(dt_model, "models/decision_tree.pkl")
    save_model(rf_model, "models/random_forest.pkl")
