#!/usr/bin/env python3
"""
Data Preprocessing Script for Stroke Prediction System

This script handles:
- Dataset loading from Kaggle
- Exploratory Data Analysis (EDA)
- Data preprocessing (encoding, imputation, normalization)
- Representative sample selection
- SMOTE application on remaining data
- Saving all preprocessed artifacts
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
# Create comprehensive EDA summary in one figure
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

# Row 1: Class distribution (bar + pie) and Gender distribution
ax1 = fig.add_subplot(gs[0, 0])
stroke_counts = data_df['stroke'].value_counts()
sns.countplot(x='stroke', data=data_df, ax=ax1, palette='Set2')
ax1.set_title("Class Distribution (Bar Chart)", fontsize=12, fontweight='bold')
ax1.set_xlabel("Stroke (0=No, 1=Yes)")
ax1.set_ylabel("Count")
for i, v in enumerate(stroke_counts):
    ax1.text(i, v + 50, str(v), ha='center', va='bottom', fontweight='bold')

ax2 = fig.add_subplot(gs[0, 1])
colors = ['#66b3ff', '#ff6666']
ax2.pie(stroke_counts, labels=['No Stroke', 'Stroke'], autopct='%1.1f%%', 
        colors=colors, startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
ax2.set_title("Class Distribution (Pie Chart)", fontsize=12, fontweight='bold')

ax3 = fig.add_subplot(gs[0, 2])
gender_counts = data_df['gender'].value_counts()
ax3.bar(range(len(gender_counts)), gender_counts.values, color=['#99ccff', '#ff99cc'])
ax3.set_xticks(range(len(gender_counts)))
ax3.set_xticklabels(gender_counts.index)
ax3.set_title("Gender Distribution", fontsize=12, fontweight='bold')
ax3.set_ylabel("Count")
for i, v in enumerate(gender_counts.values):
    ax3.text(i, v + 50, str(v), ha='center', va='bottom', fontweight='bold')

ax4 = fig.add_subplot(gs[0, 3])
work_counts = data_df['work_type'].value_counts()
ax4.barh(range(len(work_counts)), work_counts.values, color='skyblue')
ax4.set_yticks(range(len(work_counts)))
ax4.set_yticklabels(work_counts.index)
ax4.set_title("Work Type Distribution", fontsize=12, fontweight='bold')
ax4.set_xlabel("Count")
for i, v in enumerate(work_counts.values):
    ax4.text(v + 50, i, str(v), va='center', fontweight='bold')

# Row 2: Numeric distributions with histograms
numeric_cols = ['age', 'avg_glucose_level', 'bmi']
for i, col in enumerate(numeric_cols):
    ax = fig.add_subplot(gs[1, i])
    sns.histplot(data_df[col].dropna(), kde=True, ax=ax, color='steelblue')
    ax.set_title(f"{col.replace('_', ' ').title()} Distribution", fontsize=12, fontweight='bold')
    ax.set_xlabel(col.replace('_', ' ').title())
    ax.set_ylabel("Frequency")

# Row 2, Column 4: Smoking status pie chart
ax7 = fig.add_subplot(gs[1, 3])
smoking_counts = data_df['smoking_status'].value_counts()
ax7.pie(smoking_counts, labels=smoking_counts.index, autopct='%1.1f%%', 
        startangle=90, textprops={'fontsize': 9})
ax7.set_title("Smoking Status Distribution", fontsize=12, fontweight='bold')

# Row 3: Stroke vs numeric features (boxplots)
for i, col in enumerate(numeric_cols):
    ax = fig.add_subplot(gs[2, i])
    sns.boxplot(x='stroke', y=col, data=data_df, ax=ax, palette='Set2')
    ax.set_title(f"{col.replace('_', ' ').title()} by Stroke Status", fontsize=12, fontweight='bold')
    ax.set_xlabel("Stroke (0=No, 1=Yes)")
    ax.set_ylabel(col.replace('_', ' ').title())

# Row 3, Column 4: Correlation heatmap (smaller)
ax11 = fig.add_subplot(gs[2, 3])
temp_df = data_df.copy()
encoder_temp = OrdinalEncoder()
temp_df[cat_cols] = encoder_temp.fit_transform(temp_df[cat_cols].astype(str))
corr_matrix = temp_df[['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'stroke']].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax11, 
            cbar_kws={'shrink': 0.8}, annot_kws={'fontsize': 8})
ax11.set_title("Correlation Heatmap (Key Features)", fontsize=12, fontweight='bold')

# Add main title
fig.suptitle('Comprehensive Exploratory Data Analysis - Stroke Prediction Dataset', 
             fontsize=16, fontweight='bold', y=0.995)

# Save comprehensive EDA summary
plt.savefig("plots/eda_comprehensive_summary.png", dpi=150, bbox_inches='tight')
plt.close()
print("\n✓ Saved: plots/eda_comprehensive_summary.png (ALL plots in one view)")

# Also save individual plots for reference
# Class distribution
plt.figure(figsize=(6,4))
sns.countplot(x='stroke', data=data_df, palette='Set2')
plt.title("Class Distribution (Original)")
plt.xlabel("Stroke (0=No, 1=Yes)")
plt.ylabel("Count")
plt.savefig("plots/eda_class_distribution.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: plots/eda_class_distribution.png")

# Numeric distributions
plt.figure(figsize=(14,5))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(1, 3, i)
    sns.histplot(data_df[col].dropna(), kde=True)
    plt.title(f"{col} Distribution")
plt.tight_layout()
plt.savefig("plots/eda_numeric_distributions.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: plots/eda_numeric_distributions.png")

# Boxplots for outliers
plt.figure(figsize=(10,4))
sns.boxplot(data=data_df[numeric_cols])
plt.title("Boxplots — Potential Outliers")
plt.savefig("plots/eda_boxplots.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: plots/eda_boxplots.png")

# Stroke vs numeric features
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for idx, col in enumerate(numeric_cols):
    sns.boxplot(x='stroke', y=col, data=data_df, ax=axes[idx], palette='Set2')
    axes[idx].set_title(f"{col} by Stroke Status")
plt.tight_layout()
plt.savefig("plots/eda_stroke_vs_numeric.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: plots/eda_stroke_vs_numeric.png")

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
plt.savefig("plots/eda_correlation_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: plots/eda_correlation_heatmap.png")

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
print("REPRESENTATIVE SAMPLE SELECTION (STRATIFIED SAMPLING)")
print("="*60)

from sklearn.model_selection import train_test_split

X = df.drop(columns=['id', 'stroke'])
y = df['stroke']

# We'll select 15 samples: 5 stroke, 10 non-stroke
# Strategy: Use stratified sampling with diversity in top features
# Top features: age, hypertension, avg_glucose_level, bmi, smoking_status

# Denormalize for better interpretation
df_original_scale = df.copy()
df_original_scale[numeric_cols] = scaler.inverse_transform(df[numeric_cols])

print(f"\nTotal dataset size: {len(df_original_scale)}")
print(f"Stroke cases: {sum(df_original_scale['stroke'] == 1)}")
print(f"Non-stroke cases: {sum(df_original_scale['stroke'] == 0)}")

# Step 1: Use stratified sampling to get initial sample
# We need 15 samples total (5 stroke, 10 non-stroke)
# Calculate sampling fraction to get approximately these numbers
n_stroke_needed = 5
n_no_stroke_needed = 10
total_needed = n_stroke_needed + n_no_stroke_needed

# Perform stratified split to get representative sample
_, representative_sample_initial, _, _ = train_test_split(
    df_original_scale, 
    df_original_scale['stroke'],
    test_size=0.05,  # Get roughly 5% which gives us ~250 samples
    stratify=df_original_scale['stroke'],
    random_state=42
)

print(f"\nInitial stratified sample size: {len(representative_sample_initial)}")
print(f"Stroke cases in initial sample: {sum(representative_sample_initial['stroke'] == 1)}")
print(f"Non-stroke cases in initial sample: {sum(representative_sample_initial['stroke'] == 0)}")

# Step 2: From stratified sample, select diverse samples based on top features
top_features = ['age', 'hypertension', 'avg_glucose_level', 'bmi', 'smoking_status']

def select_diverse_stratified_samples(df_class, n_samples, top_features):
    """
    Select diverse samples using stratified quantile-based sampling.
    Ensures diversity across multiple feature dimensions.
    """
    if len(df_class) <= n_samples:
        return df_class.sample(n=len(df_class), random_state=42)
    
    # Create composite stratification based on quantiles of top features
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
    
    # Drop helper columns and return original data
    return df_class.loc[result.index].iloc[:n_samples]

# Separate by class from initial stratified sample
stroke_df = representative_sample_initial[representative_sample_initial['stroke'] == 1].reset_index(drop=True)
no_stroke_df = representative_sample_initial[representative_sample_initial['stroke'] == 0].reset_index(drop=True)

print(f"\n--- Applying Stratified Diverse Sampling ---")
print(f"Available stroke cases: {len(stroke_df)}")
print(f"Available non-stroke cases: {len(no_stroke_df)}")

# Select 5 stroke cases with diversity
stroke_sample = select_diverse_stratified_samples(stroke_df, n_stroke_needed, top_features)
print(f"\nSelected {len(stroke_sample)} stroke cases using stratified sampling")
print(f"Age range: {stroke_sample['age'].min():.1f} - {stroke_sample['age'].max():.1f}")
print(f"Glucose range: {stroke_sample['avg_glucose_level'].min():.1f} - {stroke_sample['avg_glucose_level'].max():.1f}")

# Select 10 non-stroke cases with diversity
no_stroke_sample = select_diverse_stratified_samples(no_stroke_df, n_no_stroke_needed, top_features)
print(f"\nSelected {len(no_stroke_sample)} non-stroke cases using stratified sampling")
print(f"Age range: {no_stroke_sample['age'].min():.1f} - {no_stroke_sample['age'].max():.1f}")
print(f"Glucose range: {no_stroke_sample['avg_glucose_level'].min():.1f} - {no_stroke_sample['avg_glucose_level'].max():.1f}")

# Combine
representative_sample = pd.concat([stroke_sample, no_stroke_sample]).reset_index(drop=True)
print(f"\nTotal representative sample size: {len(representative_sample)}")
print("Stratification method: Stratified random sampling with feature diversity")

# Save representative sample (with original scale for interpretation)
representative_sample.to_csv("data/representative_sample.csv", index=False)
print("\n✓ Representative sample saved to 'data/representative_sample.csv'")

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
print("\nNext step: Run modeltrainer.sh")
