#!/usr/bin/env python3
"""
Script to Add New Patient to Representative Sample for test runs

This ensures the new patient goes through the EXACT same preprocessing pipeline
as the existing representative samples.
"""

import pandas as pd
import joblib
import numpy as np

print("="*60)
print("ADDING NEW PATIENT TO REPRESENTATIVE SAMPLE")
print("="*60)

# ================================
# LOAD PREPROCESSING ARTIFACTS
# ================================
print("\n[1] Loading preprocessing artifacts...")

scaler = joblib.load("data/zscore_scaler.pkl")
encoder = joblib.load("data/ordinal_encoder.pkl")
imputer = joblib.load("data/knn_imputer.pkl")
feature_names = joblib.load("data/feature_names.pkl")

print("✓ Scaler loaded")
print("✓ Encoder loaded")
print("✓ Imputer loaded")
print("✓ Feature names loaded")

# ================================
# DEFINE NEW PATIENT (RAW VALUES)
# ================================
print("\n[2] Defining new patient (raw values)...")

# Your new patient data
# Format: id, gender, age, hypertension, heart_disease, ever_married, work_type, 
#         Residence_type, avg_glucose_level, bmi, smoking_status, stroke
new_patient_raw = {
    'id': 56669,
    'gender': 1,                # Male
    'age': 81,
    'hypertension': 0,
    'heart_disease': 0,
    'ever_married': 1,
    'work_type': 2,             # Private
    'Residence_type': 0,        # Urban
    'avg_glucose_level': 186.21,
    'bmi': 29.0,
    'smoking_status': 1,        # formerly smoked
    'stroke': 1
}

print("New patient raw data:")
for key, value in new_patient_raw.items():
    print(f"  {key}: {value}")

# ================================
# CREATE DATAFRAME
# ================================
print("\n[3] Creating patient dataframe...")

patient_df = pd.DataFrame([new_patient_raw])
print("✓ Patient dataframe created")

# ================================
# APPLY PREPROCESSING PIPELINE
# ================================
print("\n[4] Applying preprocessing pipeline...")

# Define column groups
cat_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
numeric_cols = ['age', 'avg_glucose_level', 'bmi']

# Step 1: Check if categorical values are already encoded (0, 1, 2, etc.)
# If not, they would need to be encoded using the saved encoder
print("\n  [4.1] Categorical features (already encoded in your input)")
print(f"        Categorical columns: {cat_cols}")

# Step 2: BMI Imputation (if needed - your patient has BMI, so no imputation needed)
print("\n  [4.2] BMI imputation check...")
if pd.isna(patient_df['bmi'].iloc[0]):
    print("        BMI is missing, applying imputation...")
    # Get the features used for imputation
    corr_features = ['age', 'avg_glucose_level', 'bmi', 'hypertension', 'heart_disease']
    patient_df[corr_features] = imputer.transform(patient_df[corr_features])
    print("        ✓ BMI imputed")
else:
    print(f"        BMI present: {patient_df['bmi'].iloc[0]} - no imputation needed")

# Step 3: Z-score normalization (CRITICAL STEP)
print("\n  [4.3] Applying Z-score normalization...")
print(f"        Before scaling: age={patient_df['age'].iloc[0]}, glucose={patient_df['avg_glucose_level'].iloc[0]}, bmi={patient_df['bmi'].iloc[0]}")

patient_df_scaled = patient_df.copy()
patient_df_scaled[numeric_cols] = scaler.transform(patient_df[numeric_cols])

print(f"        After scaling:  age={patient_df_scaled['age'].iloc[0]:.4f}, glucose={patient_df_scaled['avg_glucose_level'].iloc[0]:.4f}, bmi={patient_df_scaled['bmi'].iloc[0]:.4f}")
print("        ✓ Z-score normalization applied")

# ================================
# LOAD EXISTING REPRESENTATIVE SAMPLES
# ================================
print("\n[5] Loading existing representative samples...")

# Load both scaled and display versions
rep_sample_scaled = pd.read_csv("data/representative_sample_scaled.csv")
rep_sample_display = pd.read_csv("data/representative_sample_display.csv")

print(f"✓ Existing scaled samples: {len(rep_sample_scaled)} patients")
print(f"✓ Existing display samples: {len(rep_sample_display)} patients")

# ================================
# APPEND NEW PATIENT
# ================================
print("\n[6] Appending new patient to representative samples...")

# Add to scaled version
rep_sample_scaled_updated = pd.concat([rep_sample_scaled, patient_df_scaled], ignore_index=True)

# Add to display version (unscaled)
rep_sample_display_updated = pd.concat([rep_sample_display, patient_df], ignore_index=True)

print(f"✓ Updated scaled samples: {len(rep_sample_scaled_updated)} patients")
print(f"✓ Updated display samples: {len(rep_sample_display_updated)} patients")

# ================================
# SAVE UPDATED FILES
# ================================
print("\n[7] Saving updated representative samples...")

# Backup original files
rep_sample_scaled.to_csv("data/representative_sample_scaled_backup.csv", index=False)
rep_sample_display.to_csv("data/representative_sample_display_backup.csv", index=False)
print("✓ Backup files created")

# Save updated files
rep_sample_scaled_updated.to_csv("data/representative_sample_scaled.csv", index=False)
rep_sample_display_updated.to_csv("data/representative_sample_display.csv", index=False)
print("✓ Updated files saved")

# ================================
# VERIFICATION
# ================================
print("\n[8] Verification...")

print(f"\nNew patient index: {len(rep_sample_scaled_updated) - 1}")
print("\nNew patient (scaled) features:")
new_patient_scaled = rep_sample_scaled_updated.iloc[-1]
for feature in feature_names:
    if feature in new_patient_scaled.index:
        print(f"  {feature}: {new_patient_scaled[feature]:.4f}")

print("\nNew patient (display) features:")
new_patient_display = rep_sample_display_updated.iloc[-1]
for col in ['id', 'gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 
            'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status', 'stroke']:
    if col in new_patient_display.index:
        print(f"  {col}: {new_patient_display[col]}")

# ================================
# SUMMARY
# ================================
print("\n" + "="*60)
print("NEW PATIENT ADDED SUCCESSFULLY!")
print("="*60)
print(f"\n✓ New patient added at index: {len(rep_sample_scaled_updated) - 1}")
print(f"✓ Total patients now: {len(rep_sample_scaled_updated)}")
print(f"✓ Stroke label: {new_patient_display['stroke']}")
print("\nFiles updated:")
print("  - data/representative_sample_scaled.csv")
print("  - data/representative_sample_display.csv")
print("\nBackup files created:")
print("  - data/representative_sample_scaled_backup.csv")
print("  - data/representative_sample_display_backup.csv")
print(f"\nYou can now use patient index {len(rep_sample_scaled_updated) - 1} in instanceexplainer.sh")