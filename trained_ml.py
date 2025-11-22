# ================================
# IMPORTS
# ================================
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import joblib
from collections import Counter

# ML preprocessing
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import KNNImputer

# Model selection
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

# ML models
from sklearn.ensemble import RandomForestClassifier

# SMOTE
from imblearn.over_sampling import SMOTE

# Metrics
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)

import kagglehub

sns.set(style="whitegrid")

# ================================
# LOAD DATASET
# ================================
path = kagglehub.dataset_download("fedesoriano/stroke-prediction-dataset")
print("Dataset downloaded to:", path)

data_df = pd.read_csv(os.path.join(path, "healthcare-dataset-stroke-data.csv"))
print("Dataset Loaded Successfully!")
print(data_df.head())

# ================================
# 1. EXPLORATORY DATA ANALYSIS (EDA)
# ================================
print("\n" + "="*60)
print("EXPLORATORY DATA ANALYSIS")
print("="*60)

# Remove invalid or unwanted gender rows (remove 'Other')
print("\nBefore removing invalid gender rows:", data_df.shape[0])
unique_genders = data_df['gender'].unique()
print("Unique values in 'gender':", unique_genders)

# Remove gender == 'Other'
data_df = data_df[data_df['gender'].notna() & 
                  (data_df['gender'].str.strip() != '') &
                  (data_df['gender'].str.lower() != 'other')]

print("After removing invalid gender rows:", data_df.shape[0])


# Basic info
print("\n--- Dataset Shape ---")
print("Rows:", data_df.shape[0], " Columns:", data_df.shape[1])

print("\n--- Target Distribution ---")
print(data_df['stroke'].value_counts())
print("\nClass proportions:")
print(data_df['stroke'].value_counts(normalize=True))

print("\n--- Missing Values ---")
print(data_df.isnull().sum())

print("\n--- Data Types ---")
print(data_df.dtypes)

print("\n--- Statistical Summary ---")
print(data_df.describe())

print("\n--- Categorical Features Distribution ---")
cat_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
for col in cat_cols:
    print(f"\n{col}:")
    print(data_df[col].value_counts())

# Visualizations
# Class distribution
plt.figure(figsize=(6,4))
sns.countplot(x='stroke', data=data_df)
plt.title("Class Distribution (Original)")
plt.xlabel("Stroke (0=No, 1=Yes)")
plt.ylabel("Count")
plt.show()

# Numeric distributions
numeric_cols = ['age', 'avg_glucose_level', 'bmi']

plt.figure(figsize=(14,5))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(1, 3, i)
    sns.histplot(data_df[col].dropna(), kde=True)
    plt.title(f"{col} Distribution")
plt.tight_layout()
plt.show()

# Boxplots for outliers
plt.figure(figsize=(10,4))
sns.boxplot(data=data_df[numeric_cols])
plt.title("Boxplots — Potential Outliers")
plt.show()

# Stroke vs numeric features
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for idx, col in enumerate(numeric_cols):
    sns.boxplot(x='stroke', y=col, data=data_df, ax=axes[idx])
    axes[idx].set_title(f"{col} by Stroke Status")
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
# Create temporary df with encoded categories for correlation
temp_df = data_df.copy()
encoder_temp = OrdinalEncoder()
temp_df[cat_cols] = encoder_temp.fit_transform(temp_df[cat_cols].astype(str))
corr_matrix = temp_df.corr(numeric_only=True)
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("EDA COMPLETED")
print("="*60)

# ================================
# 2. PREPROCESSING
# ================================
print("\n" + "="*60)
print("PREPROCESSING")
print("="*60)

df = data_df.copy()

# Encode categorical columns
cat_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
encoder = OrdinalEncoder()
df[cat_cols] = encoder.fit_transform(df[cat_cols].astype(str))

# BMI Imputation using correlation-based feature selection
print("\n--- BMI Imputation ---")
corr_bmi = df.corr(numeric_only=True)['bmi'].abs()
print("Pearson correlations with BMI:")
print(corr_bmi)

selected_features = corr_bmi[corr_bmi > 0.20].index.tolist()
if 'bmi' not in selected_features:
    selected_features.append('bmi')

print("\nSelected features for BMI imputation:", selected_features)

imputer = KNNImputer(n_neighbors=5, weights='distance')
df[selected_features] = imputer.fit_transform(df[selected_features])
print("BMI imputation completed.")

# Z-score normalization
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
print("\nZ-score normalization applied to:", numeric_cols)

print("\n" + "="*60)
print("PREPROCESSING COMPLETED")
print("="*60)

# ================================
# 3. REPRESENTATIVE SAMPLE SELECTION
# ================================
print("\n" + "="*60)
print("REPRESENTATIVE SAMPLE SELECTION")
print("="*60)

X = df.drop(columns=['id', 'stroke'])
y = df['stroke']

# We'll select 15 samples: 5 stroke, 10 non-stroke
# Strategy: Use stratified sampling with diversity in top features
# Top features: age, hypertension, avg_glucose_level, bmi, smoking_status

# Denormalize for better interpretation
df_original_scale = df.copy()
df_original_scale[numeric_cols] = scaler.inverse_transform(df[numeric_cols])

# Separate by class
stroke_df = df_original_scale[df_original_scale['stroke'] == 1].reset_index(drop=True)
no_stroke_df = df_original_scale[df_original_scale['stroke'] == 0].reset_index(drop=True)

print(f"\nStroke cases: {len(stroke_df)}, Non-stroke cases: {len(no_stroke_df)}")

# Function to select diverse samples based on quantiles
def select_diverse_samples(df_class, n_samples, top_features):
    """Select diverse samples across feature space using quantile-based sampling"""
    if len(df_class) < n_samples:
        return df_class.sample(n=len(df_class), random_state=42)
    
    # Create quantile bins for numeric features
    selected_indices = []
    
    # Strategy: divide into quantiles and sample from each
    n_quantiles = min(n_samples, len(df_class))
    
    # Use age as primary stratification variable
    df_sorted = df_class.sort_values('age').reset_index(drop=True)
    quantile_size = len(df_sorted) // n_quantiles
    
    for i in range(n_quantiles):
        if i == n_quantiles - 1:
            # Last quantile: take remaining
            subset = df_sorted.iloc[i*quantile_size:]
        else:
            subset = df_sorted.iloc[i*quantile_size:(i+1)*quantile_size]
        
        if len(subset) > 0:
            # Randomly sample one from this quantile
            sampled = subset.sample(n=1, random_state=42+i)
            selected_indices.append(sampled.index[0])
    
    # If we need more samples, randomly sample from remaining
    if len(selected_indices) < n_samples:
        remaining = df_sorted.drop(selected_indices)
        additional = remaining.sample(n=n_samples-len(selected_indices), random_state=42)
        selected_indices.extend(additional.index.tolist())
    
    return df_sorted.loc[selected_indices[:n_samples]]

top_features = ['age', 'hypertension', 'avg_glucose_level', 'bmi', 'smoking_status']

# Select 5 stroke cases
stroke_sample = select_diverse_samples(stroke_df, 5, top_features)
print(f"\nSelected {len(stroke_sample)} stroke cases")

# Select 10 non-stroke cases
no_stroke_sample = select_diverse_samples(no_stroke_df, 10, top_features)
print(f"Selected {len(no_stroke_sample)} non-stroke cases")

# Combine
representative_sample = pd.concat([stroke_sample, no_stroke_sample]).reset_index(drop=True)
print(f"\nTotal representative sample size: {len(representative_sample)}")

# Save representative sample (with original scale for interpretation)
representative_sample.to_csv("representative_sample.csv", index=False)
print("\nRepresentative sample saved to 'representative_sample.csv'")

print("\nRepresentative Sample Distribution:")
print(representative_sample['stroke'].value_counts())
print("\nTop features summary in representative sample:")
print(representative_sample[top_features + ['stroke']].describe())

# Get indices to exclude from SMOTE
representative_indices = representative_sample.index.tolist()

# ================================
# 4. SMOTE ON REMAINING DATA
# ================================
print("\n" + "="*60)
print("SMOTE ON REMAINING DATA")
print("="*60)

# Get original indices from full dataset
original_indices = df.index.tolist()

# We need to map back to original df indices
# Since we've been working with reset indices, we need to track original positions
# Let's use id column or create a mapping

# Recreate the full dataset with original indices
df_with_indices = df.copy()
df_with_indices['original_index'] = range(len(df))

# Get representative sample indices from original df
# Match by denormalized values
representative_original_indices = []
for idx, row in representative_sample.iterrows():
    # Find matching row in df_original_scale
    matching_rows = df_original_scale[
        (df_original_scale['age'] == row['age']) &
        (df_original_scale['stroke'] == row['stroke']) &
        (df_original_scale['avg_glucose_level'] == row['avg_glucose_level'])
    ]
    if len(matching_rows) > 0:
        representative_original_indices.append(matching_rows.index[0])

print(f"\nMatched {len(representative_original_indices)} representative samples to original dataset")

# Remove representative samples from training data
remaining_df = df.drop(index=representative_original_indices)
X_remaining = remaining_df.drop(columns=['id', 'stroke'])
y_remaining = remaining_df['stroke']

print(f"\nRemaining data after excluding representative sample: {X_remaining.shape[0]} rows")
print("Class distribution before SMOTE:")
print(Counter(y_remaining))

# Apply SMOTE
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_remaining, y_remaining)

print("\nAfter SMOTE:")
print(Counter(y_smote))
print(f"SMOTE dataset shape: {X_smote.shape}")

print("\n" + "="*60)
print("SMOTE COMPLETED")
print("="*60)

# ================================
# 5. HYPERPARAMETER TUNING WITH GRIDSEARCH
# ================================
print("\n" + "="*60)
print("HYPERPARAMETER TUNING")
print("="*60)

rf_model = RandomForestClassifier(random_state=42)

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
# 6. 5-FOLD CROSS-VALIDATION WITH OPTIMIZED PARAMETERS
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
# 7. TRAIN FINAL MODEL ON FULL SMOTE DATASET
# ================================
print("\n" + "="*60)
print("TRAINING FINAL MODEL ON FULL SMOTE DATASET")
print("="*60)

final_model = RandomForestClassifier(**best_params, random_state=42)
final_model.fit(X_smote, y_smote)

print("\nFinal model trained on full SMOTE dataset")
print(f"Training samples: {X_smote.shape[0]}")
print(f"Features: {X_smote.shape[1]}")

# Feature importance
feature_names = X_remaining.columns.tolist()
importances = final_model.feature_importances_
indices = np.argsort(importances)[::-1]

print("\n--- Feature Importances ---")
for i in range(len(feature_names)):
    print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(range(len(indices)), importances[indices])
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importance (Final Model)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# ================================
# 8. SAVE MODEL AND ARTIFACTS
# ================================
print("\n" + "="*60)
print("SAVING MODEL AND ARTIFACTS")
print("="*60)

joblib.dump(final_model, "rf_stroke_model.pkl")
print("Model saved: rf_stroke_model.pkl")

joblib.dump(feature_names, "rf_features.pkl")
print("Feature names saved: rf_features.pkl")

joblib.dump(scaler, "zscore_scaler.pkl")
print("Scaler saved: zscore_scaler.pkl")

joblib.dump(encoder, "ordinal_encoder.pkl")
print("Encoder saved: ordinal_encoder.pkl")

joblib.dump(best_params, "best_params.pkl")
print("Best parameters saved: best_params.pkl")

# Save CV results
cv_results_df = pd.DataFrame(cv_metrics)
cv_results_df.to_csv("cv_results.csv", index=False)
print("CV results saved: cv_results.csv")

print("\n" + "="*60)
print("ALL TASKS COMPLETED SUCCESSFULLY!")
print("="*60)
print("\nSummary:")
print(f"- Representative sample: 15 patients (5 stroke, 10 non-stroke)")
print(f"- SMOTE dataset: {X_smote.shape[0]} samples")
print(f"- Best hyperparameters: {best_params}")
print(f"- Cross-validation metrics saved")
print(f"- Final model trained and saved")
