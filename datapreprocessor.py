#!/usr/bin/env python3
"""
Data Preprocessing Script for Ai-XAI-LLM

"""

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

# SMOTE
from imblearn.over_sampling import SMOTE

import kagglehub

sns.set(style="whitegrid")

print("="*60)
print("STROKE PREDICTION - DATA PREPROCESSING")
print("="*60)

# ================================
# CREATE OUTPUT DIRECTORIES
# ================================
print("\nCreating output directories...")
os.makedirs("data", exist_ok=True)
os.makedirs("plots", exist_ok=True)
print("✓ Directories created: data/, plots/")

# ================================
# LOAD DATASET
# ================================
print("\n" + "="*60)
print("LOADING DATASET")
print("="*60)

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

# FIX 5: Add original index column BEFORE any operations
data_df = data_df.reset_index(drop=True)
data_df['original_dataset_index'] = data_df.index

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

# [EDA PLOTS CODE REMAINS THE SAME - OMITTED FOR BREVITY]
# ... (Keep all your existing EDA plotting code here) ...

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
numeric_cols = ['age', 'avg_glucose_level', 'bmi']
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
print("REPRESENTATIVE SAMPLE SELECTION (STRATIFIED SAMPLING)")
print("="*60)

from sklearn.model_selection import train_test_split

X = df.drop(columns=['id', 'stroke', 'original_dataset_index'])
y = df['stroke']

# We'll select 15 samples: 5 stroke, 10 non-stroke
# Strategy: Use stratified sampling with diversity in top features
# Top features: age, hypertension, avg_glucose_level, bmi, smoking_status

# FIX 2: Keep denormalized version for display ONLY
df_original_scale = df.copy()
df_original_scale[numeric_cols] = scaler.inverse_transform(df[numeric_cols])

print(f"\nTotal dataset size: {len(df_original_scale)}")
print(f"Stroke cases: {sum(df_original_scale['stroke'] == 1)}")
print(f"Non-stroke cases: {sum(df_original_scale['stroke'] == 0)}")

# Step 1: Use stratified sampling to get initial sample
n_stroke_needed = 5
n_no_stroke_needed = 10
total_needed = n_stroke_needed + n_no_stroke_needed

# Perform stratified split to get representative sample
_, representative_sample_initial, _, _ = train_test_split(
    df_original_scale, 
    df_original_scale['stroke'],
    test_size=0.05,
    stratify=df_original_scale['stroke'],
    random_state=42
)

print(f"\nInitial stratified sample size: {len(representative_sample_initial)}")

# Step 2: From stratified sample, select diverse samples
top_features = ['age', 'hypertension', 'avg_glucose_level', 'bmi', 'smoking_status']

def select_diverse_stratified_samples(df_class, n_samples, top_features):
    """Select diverse samples using stratified quantile-based sampling."""
    if len(df_class) <= n_samples:
        return df_class.sample(n=len(df_class), random_state=42)
    
    df_work = df_class.copy()
    
    # For numeric features, create quantile bins
    df_work['age_bin'] = pd.qcut(df_work['age'], q=min(5, len(df_work)), labels=False, duplicates='drop')
    df_work['glucose_bin'] = pd.qcut(df_work['avg_glucose_level'], q=min(3, len(df_work)), labels=False, duplicates='drop')
    df_work['bmi_bin'] = pd.qcut(df_work['bmi'], q=min(3, len(df_work)), labels=False, duplicates='drop')
    
    # Create strata combining multiple features
    df_work['strata'] = (df_work['age_bin'].astype(str) + '_' + 
                         df_work['hypertension'].astype(str) + '_' + 
                         df_work['glucose_bin'].astype(str) + '_' +
                         df_work['bmi_bin'].astype(str) + '_' +
                         df_work['smoking_status'].astype(str))
    
    # Sample from each stratum
    strata_counts = df_work['strata'].value_counts()
    samples_per_stratum = max(1, n_samples // len(strata_counts))
    
    selected_samples = []
    remaining_needed = n_samples
    
    # First pass: sample from each stratum
    for stratum in strata_counts.index:
        if remaining_needed <= 0:
            break
        stratum_data = df_work[df_work['strata'] == stratum]
        n_to_sample = min(samples_per_stratum, len(stratum_data), remaining_needed)
        sampled = stratum_data.sample(n=n_to_sample, random_state=42)
        selected_samples.append(sampled)
        remaining_needed -= n_to_sample
    
    # Second pass: if we need more samples, take from remaining strata
    if remaining_needed > 0:
        already_selected = pd.concat(selected_samples).index
        remaining_data = df_work.drop(already_selected)
        if len(remaining_data) > 0:
            additional = remaining_data.sample(n=min(remaining_needed, len(remaining_data)), random_state=42)
            selected_samples.append(additional)
    
    result = pd.concat(selected_samples)
    return df_class.loc[result.index].iloc[:n_samples]

# Separate by class from initial stratified sample
stroke_df = representative_sample_initial[representative_sample_initial['stroke'] == 1].reset_index(drop=True)
no_stroke_df = representative_sample_initial[representative_sample_initial['stroke'] == 0].reset_index(drop=True)

print(f"\n--- Applying Stratified Diverse Sampling ---")

# Select samples (using denormalized for display)
stroke_sample = select_diverse_stratified_samples(stroke_df, n_stroke_needed, top_features)
no_stroke_sample = select_diverse_stratified_samples(no_stroke_df, n_no_stroke_needed, top_features)

# Combine
representative_sample_display = pd.concat([stroke_sample, no_stroke_sample]).reset_index(drop=True)

print(f"\nTotal representative sample size: {len(representative_sample_display)}")

# FIX 2: Get the ORIGINAL indices to extract from normalized df
representative_original_indices = representative_sample_display['original_dataset_index'].tolist()

print(f"\n✓ Representative sample original indices: {representative_original_indices}")

# FIX 4: Extract SCALED versions for model prediction
representative_sample_scaled = df.loc[representative_original_indices].copy()
representative_sample_scaled = representative_sample_scaled.reset_index(drop=True)

# Save SCALED version (for prediction)
representative_sample_scaled.to_csv("data/representative_sample_scaled.csv", index=False)
print("\n✓ Representative sample (SCALED) saved to 'data/representative_sample_scaled.csv'")

# Save UNSCALED version (for display/interpretation)
representative_sample_display.to_csv("data/representative_sample_display.csv", index=False)
print("✓ Representative sample (DISPLAY) saved to 'data/representative_sample_display.csv'")

print("\nRepresentative Sample Distribution:")
print(representative_sample_display['stroke'].value_counts())

# ================================
# 4. SMOTE ON REMAINING DATA
# ================================
print("\n" + "="*60)
print("SMOTE ON REMAINING DATA")
print("="*60)

# FIX 2: Remove representative samples using correct indices
remaining_df = df.drop(index=representative_original_indices)
X_remaining = remaining_df.drop(columns=['id', 'stroke', 'original_dataset_index'])
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

# ================================
# 5. SAVE PREPROCESSED DATA AND ARTIFACTS
# ================================
print("\n" + "="*60)
print("SAVING PREPROCESSED DATA AND ARTIFACTS")
print("="*60)

# Save SMOTE data
X_smote_df = pd.DataFrame(X_smote, columns=X_remaining.columns)
y_smote_df = pd.Series(y_smote, name='stroke')

X_smote_df.to_csv("data/X_smote.csv", index=False)
y_smote_df.to_csv("data/y_smote.csv", index=False)
print("✓ SMOTE data saved: data/X_smote.csv, data/y_smote.csv")

# Save scaler and encoder
joblib.dump(scaler, "data/zscore_scaler.pkl")
print("✓ Scaler saved: data/zscore_scaler.pkl")

joblib.dump(encoder, "data/ordinal_encoder.pkl")
print("✓ Encoder saved: data/ordinal_encoder.pkl")

# Save feature names
feature_names = X_remaining.columns.tolist()
joblib.dump(feature_names, "data/feature_names.pkl")
print("✓ Feature names saved: data/feature_names.pkl")

# Save imputer
joblib.dump(imputer, "data/knn_imputer.pkl")
print("✓ Imputer saved: data/knn_imputer.pkl")

print("\n" + "="*60)
print("DATA PREPROCESSING COMPLETED SUCCESSFULLY!")
print("="*60)
print("\nSummary:")
print(f"- Representative sample: 15 patients (5 stroke, 10 non-stroke)")
print(f"- SMOTE dataset: {X_smote.shape[0]} samples")
print(f"- Features: {X_smote.shape[1]}")
print(f"- All artifacts saved for model training")
print("\nFixes applied:")
print("  ✓ Issue 2: Fixed index tracking with original_dataset_index column")
print("  ✓ Issue 4: Saved SCALED representative sample for predictions")
print("  ✓ Issue 5: Original indices preserved throughout pipeline")
print("\nNext step: Run modeltrainer.sh")