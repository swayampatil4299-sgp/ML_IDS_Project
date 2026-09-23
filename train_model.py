"""
train_model.py
Machine Learning Pipeline for Intrusion Detection System (IDS).

1. Loads synthetic network traffic dataset from data/network_traffic.csv.
2. Preprocesses numerical features (StandardScaler) and categorical features (OneHotEncoder).
3. Trains Random Forest Classifier and Logistic Regression.
4. Evaluates both models (Accuracy, Precision, Recall, F1, Confusion Matrix).
5. Identifies the best performing model.
6. Saves the pipeline bundle to models/ids_model_payload.joblib.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

NUMERICAL_COLS = [
    'duration',
    'src_bytes',
    'dst_bytes',
    'failed_logins',
    'traffic_count',
    'logged_in'
]

CATEGORICAL_COLS = [
    'protocol_type',
    'service',
    'flag'
]

def build_preprocessor() -> ColumnTransformer:
    """Builds a scikit-learn ColumnTransformer for numerical and categorical features."""
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, NUMERICAL_COLS),
            ('cat', cat_pipeline, CATEGORICAL_COLS)
        ]
    )
    return preprocessor

def evaluate_model(model_pipeline, X_test, y_test, class_labels):
    """Calculates accuracy, precision, recall, f1, and confusion matrix."""
    y_pred = model_pipeline.predict(X_test)
    
    # Calculate binary classification metrics for 'Intrusion/Attack' as positive class
    pos_label = 'Intrusion/Attack'
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, pos_label=pos_label, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, pos_label=pos_label, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, pos_label=pos_label, zero_division=0))
    }
    cm = confusion_matrix(y_test, y_pred, labels=class_labels)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    return metrics, cm, report

def get_feature_importances(pipeline, num_cols, cat_cols):
    """Extracts feature importances if available, mapped to transformed feature names."""
    try:
        preprocessor = pipeline.named_steps['preprocessor']
        classifier = pipeline.named_steps['classifier']
        
        # Get feature names from preprocessor
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
        cat_features = list(cat_encoder.get_feature_names_out(cat_cols))
        all_features = num_cols + cat_features
        
        if hasattr(classifier, 'feature_importances_'):
            importances = classifier.feature_importances_
            feat_imp = pd.Series(importances, index=all_features).sort_values(ascending=False)
            return feat_imp.head(10).to_dict()
        elif hasattr(classifier, 'coef_'):
            coefs = np.abs(classifier.coef_[0])
            feat_imp = pd.Series(coefs, index=all_features).sort_values(ascending=False)
            return feat_imp.head(10).to_dict()
    except Exception as e:
        print(f"Warning: could not extract feature importances: {e}")
    return {}

def train_and_save_pipeline(data_path: str, output_path: str):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    feature_cols = NUMERICAL_COLS + CATEGORICAL_COLS
    X = df[feature_cols]
    y = df['label']
    
    class_labels = ['Normal', 'Intrusion/Attack']
    
    # Train / Test split (80% train, 20% test, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
    
    # Define models
    rf_pipeline = Pipeline([
        ('preprocessor', build_preprocessor()),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42))
    ])
    
    lr_pipeline = Pipeline([
        ('preprocessor', build_preprocessor()),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    print("\n--- Training Random Forest Classifier ---")
    rf_pipeline.fit(X_train, y_train)
    rf_metrics, rf_cm, rf_report = evaluate_model(rf_pipeline, X_test, y_test, class_labels)
    rf_feat_imp = get_feature_importances(rf_pipeline, NUMERICAL_COLS, CATEGORICAL_COLS)
    print(f"Random Forest - Accuracy: {rf_metrics['accuracy']:.4f}, Precision: {rf_metrics['precision']:.4f}, Recall: {rf_metrics['recall']:.4f}, F1: {rf_metrics['f1']:.4f}")
    
    print("\n--- Training Logistic Regression ---")
    lr_pipeline.fit(X_train, y_train)
    lr_metrics, lr_cm, lr_report = evaluate_model(lr_pipeline, X_test, y_test, class_labels)
    lr_feat_imp = get_feature_importances(lr_pipeline, NUMERICAL_COLS, CATEGORICAL_COLS)
    print(f"Logistic Regression - Accuracy: {lr_metrics['accuracy']:.4f}, Precision: {lr_metrics['precision']:.4f}, Recall: {lr_metrics['recall']:.4f}, F1: {lr_metrics['f1']:.4f}")
    
    # Select best performer based on F1-score
    if rf_metrics['f1'] >= lr_metrics['f1']:
        best_name = "Random Forest"
        best_pipeline = rf_pipeline
    else:
        best_name = "Logistic Regression"
        best_pipeline = lr_pipeline
        
    print(f"\n>>> Best Performer: {best_name} (F1-score: {max(rf_metrics['f1'], lr_metrics['f1']):.4f})")
    
    # Package everything for Streamlit consumption
    payload = {
        'best_model_name': best_name,
        'best_model_pipeline': best_pipeline,
        'models': {
            'Random Forest': {
                'pipeline': rf_pipeline,
                'metrics': rf_metrics,
                'cm': rf_cm.tolist(),
                'report': rf_report,
                'feature_importance': rf_feat_imp
            },
            'Logistic Regression': {
                'pipeline': lr_pipeline,
                'metrics': lr_metrics,
                'cm': lr_cm.tolist(),
                'report': lr_report,
                'feature_importance': lr_feat_imp
            }
        },
        'feature_cols': feature_cols,
        'numerical_cols': NUMERICAL_COLS,
        'categorical_cols': CATEGORICAL_COLS,
        'categorical_options': {
            'protocol_type': sorted(df['protocol_type'].dropna().unique().tolist()),
            'service': sorted(df['service'].dropna().unique().tolist()),
            'flag': sorted(df['flag'].dropna().unique().tolist())
        },
        'classes': class_labels,
        'test_set_size': len(X_test)
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(payload, output_path)
    print(f"\nTrained models and evaluation payload successfully saved to:\n{output_path}")

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(base_dir, 'data', 'network_traffic.csv')
    model_file = os.path.join(base_dir, 'models', 'ids_model_payload.joblib')
    
    if not os.path.exists(data_file):
        import generate_data
        generate_data.main()
        
    train_and_save_pipeline(data_file, model_file)
