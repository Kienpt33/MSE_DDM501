# train_model.py - Model training and saving script
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd
import joblib
import os
import json
from datetime import datetime

# Create directories if they don't exist
MODEL_DIR = 'models'
DATA_DIR = 'data'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Paths to best model and model info
BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.pkl')
MODEL_INFO_PATH = os.path.join(MODEL_DIR, 'model_info.json')


def generate_data(n_samples=1000, n_features=5, n_informative=3, n_redundant=0,
                  n_classes=2, random_state=42):
    """
    Generate synthetic data using make_classification and save to file
    """
    print(f"Generating data with {n_samples} samples and {n_features} features...")

    # Generate data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=n_classes,
        random_state=random_state
    )

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create DataFrame to save
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    df['target'] = y

    # Save dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_path = os.path.join(DATA_DIR, f'dataset_{timestamp}.csv')
    df.to_csv(data_path, index=False)

    # Save dataset info
    data_info = {
        'n_samples': n_samples,
        'n_features': n_features,
        'n_informative': n_informative,
        'n_redundant': n_redundant,
        'n_classes': n_classes,
        'random_state': random_state,
        'data_path': data_path,
        'timestamp': timestamp
    }

    with open(os.path.join(DATA_DIR, f'data_info_{timestamp}.json'), 'w') as f:
        json.dump(data_info, f)

    print(f"Data saved to: {data_path}")
    return X_train, X_test, y_train, y_test, n_features


def train_model(model_name, X_train, y_train, X_test, y_test, param_grid=None):
    """
    Train and fine-tune the model
    """
    print(f"Training model: {model_name}...")

    # Select the model based on the name
    if model_name == 'LogisticRegression':
        base_model = LogisticRegression(max_iter=10000)
        if param_grid is None:
            # Define valid combinations of solver and penalty
            param_grid = [
                {
                    'penalty': ['l2'],
                    'C': [0.01, 0.1, 1, 10, 100],
                    'solver': ['lbfgs']
                },
                {
                    'penalty': ['l1', 'l2'],
                    'C': [0.01, 0.1, 1, 10, 100],
                    'solver': ['liblinear']
                }
            ]
    elif model_name == 'DecisionTree':
        base_model = DecisionTreeClassifier()
        if param_grid is None:
            param_grid = {
                'max_depth': [None, 5, 10, 15],
                'min_samples_split': [2, 5, 10],
                'criterion': ['gini', 'entropy']
            }
    elif model_name == 'RandomForest':
        base_model = RandomForestClassifier()
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [None, 10],
                'min_samples_split': [2, 5]
            }
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")

    # Hyperparameter tuning
    print("Tuning hyperparameters...")
    grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Evaluate the model on the test set
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Generate evaluation report
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"\nAccuracy of {model_name}: {accuracy:.4f}")
    print(f"\nClassification Report:\n{report}")
    print(f"\nConfusion Matrix:\n{conf_matrix}")

    # Save model info
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_info = {
        'model_name': model_name,
        'best_params': grid_search.best_params_,
        'accuracy': float(accuracy),
        'timestamp': timestamp
    }

    # Save the model
    model_path = os.path.join(MODEL_DIR, f'{model_name}_{timestamp}.pkl')
    joblib.dump(best_model, model_path)

    print(f"Model saved at: {model_path}")
    return best_model, accuracy, model_info, model_path



def compare_and_save_best_model(model_info):
    """
    Compare and save the best model
    """
    # Check if best model info exists
    best_info = None
    if os.path.exists(MODEL_INFO_PATH):
        try:
            with open(MODEL_INFO_PATH, 'r') as f:
                best_info = json.load(f)
        except:
            best_info = None

    # If no existing best model or current model is better
    if best_info is None or model_info['accuracy'] > best_info['accuracy']:
        # Save new best model info
        with open(MODEL_INFO_PATH, 'w') as f:
            json.dump(model_info, f)

        # Save new best model
        model_path = os.path.join(MODEL_DIR, f"{model_info['model_name']}_{model_info['timestamp']}.pkl")
        best_model = joblib.load(model_path)
        joblib.dump(best_model, BEST_MODEL_PATH)

        print(
            f"\nModel {model_info['model_name']} selected as best model with accuracy {model_info['accuracy']:.4f}")
        return True
    else:
        print(
            f"\nKeeping current best model ({best_info['model_name']}) with accuracy {best_info['accuracy']:.4f}")
        return False


def main():
    # Generate data
    X_train, X_test, y_train, y_test, n_features = generate_data(
        n_samples=1000,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        n_classes=2
    )

    # Models to evaluate
    models = ['LogisticRegression', 'DecisionTree', 'RandomForest']

    best_model_updated = False

    # Train and evaluate each model
    for model_name in models:
        print(f"\n{'-' * 50}")
        model, accuracy, model_info, _ = train_model(
            model_name, X_train, y_train, X_test, y_test
        )

        # Check and update best model
        if compare_and_save_best_model(model_info):
            best_model_updated = True

    if best_model_updated:
        print("\nBest model has been updated.")
    else:
        print("\nBest model remains unchanged.")


if __name__ == "__main__":
    main()
