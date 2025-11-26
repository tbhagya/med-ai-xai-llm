#!/usr/bin/env python3
"""
Model Training Script for Stroke Prediction System

This script handles:
- Loading preprocessed SMOTE data
- Hyperparameter tuning with GridSearchCV
- 5-fold cross-validation with optimized parameters
- Final model training on full SMOTE dataset
- Feature importance analysis
- Saving trained model and results
"""

# ================================
# IMPORTS
# ================================
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import joblib

# Model selection
from sklearn.model_selection import StratifiedKFold, GridSearchCV

# ML models
from sklearn.ensemble import RandomForestClassifier

# Metrics
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)

print("="*60)
print("STROKE PREDICTION - MODEL TRAINING")
print("="*60)

# ================================
# CREATE OUTPUT DIRECTORIES
# ================================
print("\nCreating output directories...")
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)
print("✓ Directories created: models/, plots/")

# ================================
# LOAD PREPROCESSED DATA
# ================================
print("\n" + "="*60)
print("LOADING PREPROCESSED DATA")
print("="*60)

# Check if preprocessed data exists
required_files = ["data/X_smote.csv", "data/y_smote.csv", "data/feature_names.pkl"]
missing_files = [f for f in required_files if not os.path.exists(f)]

if missing_files:
    print("ERROR: Missing required files:")
    for f in missing_files:
        print(f"  - {f}")
    print("\nPlease run datapreprocessor.sh first!")
    exit(1)

# Load SMOTE data
X_smote = pd.read_csv("data/X_smote.csv")
y_smote = pd.read_csv("data/y_smote.csv")['stroke']
feature_names = joblib.load("data/feature_names.pkl")

print(f"✓ SMOTE data loaded: {X_smote.shape[0]} samples, {X_smote.shape[1]} features")
print(f"✓ Feature names loaded: {len(feature_names)} features")
print(f"\nClass distribution:")
print(y_smote.value_counts())

# ================================
# HYPERPARAMETER TUNING WITH GRIDSEARCH
# ================================
print("\n" + "="*60)
print("HYPERPARAMETER TUNING")
print("="*60)

rf_model = RandomForestClassifier(random_state=42)

# Parameters for GridSearchCV
param_grid_rf = {
    'n_estimators': [200, 400],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}

print("\nRunning GridSearchCV on SMOTE-balanced dataset...")
print(f"Parameter grid: {param_grid_rf}")

grid_rf = GridSearchCV(
    rf_model,
    param_grid_rf,
    cv=5,
    scoring='recall',  # Focus on recall as per requirements
    n_jobs=-1,
    verbose=2,
    return_train_score=True
)

grid_rf.fit(X_smote, y_smote)

best_params = grid_rf.best_params_
print("\n" + "="*60)
print("BEST HYPERPARAMETERS FOUND:")
print("="*60)
print(best_params)
print(f"\nBest CV Recall Score: {grid_rf.best_score_:.4f}")

print("\n" + "="*60)
print("HYPERPARAMETER TUNING COMPLETED")
print("="*60)

# ================================
# 5-FOLD CROSS-VALIDATION WITH OPTIMIZED PARAMETERS
# ================================
print("\n" + "="*60)
print("5-FOLD CROSS-VALIDATION WITH OPTIMIZED HYPERPARAMETERS")
print("="*60)

# Create model with best parameters
best_rf_model = RandomForestClassifier(**best_params, random_state=42)

# 5-fold stratified CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Metrics storage
cv_metrics = {
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1': [],
    'ROC-AUC': []
}

print("\nPerforming 5-fold cross-validation...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X_smote, y_smote), 1):
    print(f"\nFold {fold}...")
    
    X_train_fold = X_smote.iloc[train_idx] if isinstance(X_smote, pd.DataFrame) else pd.DataFrame(X_smote).iloc[train_idx]
    X_val_fold = X_smote.iloc[val_idx] if isinstance(X_smote, pd.DataFrame) else pd.DataFrame(X_smote).iloc[val_idx]
    y_train_fold = y_smote.iloc[train_idx] if isinstance(y_smote, pd.Series) else pd.Series(y_smote).iloc[train_idx]
    y_val_fold = y_smote.iloc[val_idx] if isinstance(y_smote, pd.Series) else pd.Series(y_smote).iloc[val_idx]
    
    # Train model
    fold_model = RandomForestClassifier(**best_params, random_state=42)
    fold_model.fit(X_train_fold, y_train_fold)
    
    # Predictions
    y_pred = fold_model.predict(X_val_fold)
    y_prob = fold_model.predict_proba(X_val_fold)[:, 1]
    
    # Calculate metrics
    cv_metrics['Accuracy'].append(accuracy_score(y_val_fold, y_pred))
    cv_metrics['Precision'].append(precision_score(y_val_fold, y_pred))
    cv_metrics['Recall'].append(recall_score(y_val_fold, y_pred))
    cv_metrics['F1'].append(f1_score(y_val_fold, y_pred))
    cv_metrics['ROC-AUC'].append(roc_auc_score(y_val_fold, y_prob))

print("\n" + "="*60)
print("CROSS-VALIDATION RESULTS (Mean ± Std)")
print("="*60)

for metric, values in cv_metrics.items():
    mean_val = np.mean(values)
    std_val = np.std(values)
    print(f"{metric:12s}: {mean_val:.4f} ± {std_val:.4f}")

print("\n" + "="*60)
print("CROSS-VALIDATION COMPLETED")
print("="*60)

# ================================
# TRAIN FINAL MODEL ON FULL SMOTE DATASET
# ================================
print("\n" + "="*60)
print("TRAINING FINAL MODEL ON FULL SMOTE DATASET")
print("="*60)

final_model = RandomForestClassifier(**best_params, random_state=42)
final_model.fit(X_smote, y_smote)

print("\n✓ Final model trained on full SMOTE dataset")
print(f"Training samples: {X_smote.shape[0]}")
print(f"Features: {X_smote.shape[1]}")

# Feature importance
importances = final_model.feature_importances_
indices = np.argsort(importances)[::-1]

print("\n--- Feature Importances (Top 10) ---")
for i in range(min(10, len(feature_names))):
    print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(range(len(indices)), importances[indices])
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importance (Final Model)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("plots/feature_importance.png", dpi=150, bbox_inches='tight')
plt.close()
print("\n✓ Feature importance plot saved: plots/feature_importance.png")

# ================================
# SAVE MODEL AND RESULTS
# ================================
print("\n" + "="*60)
print("SAVING MODEL AND RESULTS")
print("="*60)

joblib.dump(final_model, "models/rf_stroke_model.pkl")
print("✓ Model saved: models/rf_stroke_model.pkl")

joblib.dump(feature_names, "models/rf_features.pkl")
print("✓ Feature names saved: models/rf_features.pkl")

joblib.dump(best_params, "models/best_params.pkl")
print("✓ Best parameters saved: models/best_params.pkl")

# Save CV results
cv_results_df = pd.DataFrame(cv_metrics)
cv_results_df.to_csv("models/cv_results.csv", index=False)
print("✓ CV results saved: models/cv_results.csv")

# Save summary statistics
cv_summary = {}
for metric, values in cv_metrics.items():
    cv_summary[f"{metric}_mean"] = np.mean(values)
    cv_summary[f"{metric}_std"] = np.std(values)

cv_summary_df = pd.DataFrame([cv_summary])
cv_summary_df.to_csv("models/cv_summary.csv", index=False)
print("✓ CV summary saved: models/cv_summary.csv")

print("\n" + "="*60)
print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("="*60)
print("\nSummary:")
print(f"- Best hyperparameters: {best_params}")
print(f"- Mean CV Recall: {np.mean(cv_metrics['Recall']):.4f} ± {np.std(cv_metrics['Recall']):.4f}")
print(f"- Mean CV F1: {np.mean(cv_metrics['F1']):.4f} ± {np.std(cv_metrics['F1']):.4f}")
print(f"- Final model trained and saved")
print("\nNext step: Run instanceexplainer.sh <patient_index>")
