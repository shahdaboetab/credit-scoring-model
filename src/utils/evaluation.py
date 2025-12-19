"""
Evaluation utilities for the credit scoring model.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import json
import os

def plot_confusion_matrix(y_true, y_pred, model_name, save_path=None):
    """
    Plot confusion matrix.

    Args:
        y_true (array): True labels
        y_pred (array): Predicted labels
        model_name (str): Name of the model
        save_path (str, optional): Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Good Credit', 'Bad Credit'],
                yticklabels=['Good Credit', 'Bad Credit'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    plt.show()

def plot_roc_curve(y_true, y_pred_proba, model_name, save_path=None):
    """
    Plot ROC curve.

    Args:
        y_true (array): True labels
        y_pred_proba (array): Predicted probabilities
        model_name (str): Name of the model
        save_path (str, optional): Path to save the plot
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {model_name}')
    plt.legend(loc="lower right")

    if save_path:
        plt.savefig(save_path)
        print(f"ROC curve saved to {save_path}")
    plt.show()

def plot_feature_importance(model, feature_names, model_name, save_path=None):
    """
    Plot feature importance for tree-based models.

    Args:
        model: Trained model with feature_importances_ attribute
        feature_names (list): List of feature names
        model_name (str): Name of the model
        save_path (str, optional): Path to save the plot
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = importances.argsort()[::-1]

        plt.figure(figsize=(10, 6))
        plt.title(f'Feature Importances - {model_name}')
        plt.bar(range(len(feature_names)), importances[indices], align='center')
        plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Feature importance plot saved to {save_path}")
        plt.show()
    else:
        print(f"Model {model_name} does not have feature_importances_ attribute")

def save_metrics(metrics_dict, file_path):
    """
    Save evaluation metrics to a JSON file.

    Args:
        metrics_dict (dict): Dictionary of metrics
        file_path (str): Path to save the metrics
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"Metrics saved to {file_path}")

def load_metrics(file_path):
    """
    Load evaluation metrics from a JSON file.

    Args:
        file_path (str): Path to the metrics file

    Returns:
        dict: Loaded metrics
    """
    with open(file_path, 'r') as f:
        return json.load(f)

def print_classification_report(y_true, y_pred, model_name):
    """
    Print detailed classification report.

    Args:
        y_true (array): True labels
        y_pred (array): Predicted labels
        model_name (str): Name of the model
    """
    print(f"\nClassification Report for {model_name}:")
    print(classification_report(y_true, y_pred, target_names=['Good Credit', 'Bad Credit']))

def compare_models_metrics(models_metrics, save_path=None):
    """
    Compare metrics across different models.

    Args:
        models_metrics (dict): Dictionary with model names as keys and metrics as values
        save_path (str, optional): Path to save the comparison plot
    """
    metrics_df = pd.DataFrame(models_metrics).T
    metrics_df.plot(kind='bar', figsize=(12, 6))
    plt.title('Model Comparison')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Model comparison plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    # Example usage
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier

    # Dummy data for testing
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.randint(0, 2, 100)
    y_pred_proba = np.random.random(100)

    plot_confusion_matrix(y_true, y_pred, "Test Model")
    plot_roc_curve(y_true, y_pred_proba, "Test Model")
