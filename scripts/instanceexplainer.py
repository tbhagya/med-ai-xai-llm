#!/usr/bin/env python3
"""
Stroke risk prediction with AI-XAI-LLM
"""

import os
import joblib
import pandas as pd
import shap
import numpy as np
from openai import OpenAI
from sklearn.preprocessing import StandardScaler
from llm_wrapper import LLMWrapper

# ================================
# CONFIGURATION
# ================================

# Model paths
MODEL_PATH = "models/rf_stroke_model.pkl"
FEATURES_PATH = "models/rf_features.pkl"
SCALER_PATH = "data/zscore_scaler.pkl"
ENCODER_PATH = "data/ordinal_encoder.pkl"

# Data paths
REPRESENTATIVE_SAMPLE_SCALED_PATH = "data/representative_sample_scaled.csv"
REPRESENTATIVE_SAMPLE_DISPLAY_PATH = "data/representative_sample_display.csv"

# LLM configuration (LM Studio)
LMSTUDIO_CONFIG = {
    'base_url': "http://localhost:1234/v1",
    'api_key': "lm-studio",
    'model_name': "lmstudio-ai/gemma-2b-it-GGUF"
}

# ================================
# MEDICAL TERMINOLOGY MAPPING
# ================================

feature_name_map = {
    "gender": "Sex",
    "age": "Age",
    "hypertension": "High Blood Pressure",
    "heart_disease": "Cardiovascular Disease History",
    "ever_married": "Marital History",
    "work_type": "Occupation Type",
    "Residence_type": "Living Environment",
    "avg_glucose_level": "Average Glucose Level",
    "bmi": "Body Mass Index (BMI)",
    "smoking_status": "Smoking History"
}

categorical_value_map = {
    "gender": {0: "Female", 1: "Male"},
    "hypertension": {0: "Absent", 1: "Present"},
    "heart_disease": {0: "Absent", 1: "Present"},
    "ever_married": {0: "No", 1: "Yes"},
    "work_type": {0: "Government Job", 1: "Never Worked", 2: "Private", 3: "Self-employed", 4: "Children"},
    "Residence_type": {0: "Rural", 1: "Urban"},
    "smoking_status": {0: "Unknown", 1: "Formerly Smoked", 2: "Never Smoked", 3: "Smokes"}
}

def format_value(feature, value):
    """Format feature value with appropriate units/labels"""
    if feature in categorical_value_map:
        try:
            val_int = int(value) if not isinstance(value, int) else value
            return categorical_value_map[feature].get(val_int, str(value))
        except (ValueError, TypeError):
            return str(value)
    if feature == "bmi":
        return f"{float(value):.1f} kg/m²"
    if feature == "avg_glucose_level":
        return f"{float(value):.1f} mg/dL"
    if feature == "age":
        return f"{int(float(value))} years"
    return str(value)

# ================================
# LOAD MODEL AND ARTIFACTS
# ================================

print("="*60)
print("LOADING MODEL AND ARTIFACTS")
print("="*60)

# Optimised threshold for dataset
PREDICTION_THRESHOLD = 0.1

rf_model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURES_PATH)
scaler = joblib.load(SCALER_PATH)
encoder = joblib.load(ENCODER_PATH)

print("[OK] Model loaded")
print("[OK] Feature names loaded")
print("[OK] Scaler loaded")
print("[OK] Encoder loaded")

# ================================
# LOAD REPRESENTATIVE SAMPLES
# ================================

print("\n" + "="*60)
print("LOADING REPRESENTATIVE SAMPLES")
print("="*60)

representative_sample_scaled = pd.read_csv(REPRESENTATIVE_SAMPLE_SCALED_PATH)
representative_sample_display = pd.read_csv(REPRESENTATIVE_SAMPLE_DISPLAY_PATH)

print(f"Representative sample loaded: {len(representative_sample_scaled)} patients")
print(f"Valid patient indices: 0 to {len(representative_sample_scaled)-1}")

# ================================
# INTERACTIVE PATIENT SELECTION
# ================================

print("\n" + "="*60)
print("PATIENT SELECTION")
print("="*60)

while True:
    try:
        user_input = input(f"\nEnter patient index (0-{len(representative_sample_scaled)-1}): ").strip()
        PATIENT_INDEX = int(user_input)
        
        if 0 <= PATIENT_INDEX < len(representative_sample_scaled):
            print(f"[OK] Selected patient index: {PATIENT_INDEX}")
            break
        else:
            print(f"[ERROR] Index out of range. Please enter a value between 0 and {len(representative_sample_scaled)-1}")
    except ValueError:
        print("[ERROR] Invalid input. Please enter a valid integer.")
    except KeyboardInterrupt:
        print("\n\n[EXIT] Program terminated by user.")
        exit(0)

# Get patient data from BOTH versions
patient_scaled = representative_sample_scaled.iloc[PATIENT_INDEX].copy()
patient_display = representative_sample_display.iloc[PATIENT_INDEX].copy()

# ================================
# DISPLAY PATIENT INFORMATION
# ================================

print("\n" + "="*60)
print("PATIENT INFORMATION")
print("="*60)
print(f"\nPatient Index: {PATIENT_INDEX}")
print(f"Actual Stroke Status: {'Risk' if patient_display['stroke'] == 1 else 'No Risk'}")
print("\nFeature Values:")

for feature, value in patient_display.items():
    if feature in feature_name_map:
        feature_display = feature_name_map[feature]
        value_display = format_value(feature, value)
        print(f"  {feature_display}: {value_display}")

# ================================
# PREPARE PATIENT FOR PREDICTION
# ================================

print("\n" + "="*60)
print("PREPARING PATIENT DATA")
print("="*60)

X_patient = patient_scaled[feature_names].to_frame().T
print("[OK] Patient data prepared (already scaled)")

# ================================
# AI OUTPUT
# ================================

print("\n" + "="*60)
print("AI OUTPUT")
print("="*60)

# Get probabilities
stroke_proba = rf_model.predict_proba(X_patient)[0]

# Apply optimized threshold
stroke_pred = int(stroke_proba[1] >= PREDICTION_THRESHOLD)

# Determine risk label
risk_label = "Risk" if stroke_pred == 1 else "No Risk"
print(f"\nPredicted Stroke Status: {risk_label}")

# ================================
# XAI OUTPUT
# ================================

print("\n" + "="*60)
print("XAI OUTPUT")
print("="*60)

# Compute SHAP values
explainer = shap.Explainer(rf_model)
shap_exp = explainer(X_patient)
patient_shap_values = shap_exp.values[0, :, 1]

# Create SHAP dataframe
df_shap = pd.DataFrame({
    "feature": feature_names,
    "value_scaled": X_patient.iloc[0].values,
    "shap_value": patient_shap_values
})

# Add original (unscaled) values for display
def get_original_value(feature):
    return patient_display[feature] if feature in patient_display.index else None

df_shap["value_original"] = df_shap["feature"].apply(get_original_value)

# Get top 3 positive risk-contributing features
df_shap["abs_shap"] = df_shap["shap_value"].abs()
top3 = df_shap[df_shap["shap_value"] > 0].nlargest(3, "shap_value").reset_index(drop=True)

# Display ALL SHAP values sorted by absolute value
print("\nFeature Impact (SHAP values):")
df_sorted = df_shap.sort_values("abs_shap", ascending=False).reset_index(drop=True)

for idx, row in df_sorted.iterrows():
    feature_mapped = feature_name_map.get(row['feature'], row['feature'])
    print(f"  {feature_mapped}: {row['shap_value']:+.4f}")

# ================================
# AI-XAI-LLM OUTPUT
# ================================

print("\n" + "="*60)
print("AI-XAI-LLM OUTPUT")
print("="*60)

# Top feature list
top_feature_list = ", ".join([
    feature_name_map.get(f, f)
    for f in top3['feature'].tolist()
])

# Build patient feature block using DISPLAY values
patient_feature_block = "\n".join([
    f"{feature_name_map.get(f, f)}: {format_value(f, v)}"
    for f, v in patient_display.items()
    if f in feature_name_map
])

# Display Stroke Risk and Patient Data
print(f"\nPredicted Stroke Status: {risk_label}\n")
print("Patient Data:")
print(patient_feature_block)
print()

# Initialize LLM wrapper
llm = LLMWrapper(
    base_url=LMSTUDIO_CONFIG['base_url'],
    api_key=LMSTUDIO_CONFIG['api_key'],
    model_name=LMSTUDIO_CONFIG['model_name']
)

# For notebook-style compatibility
client = OpenAI(base_url=LMSTUDIO_CONFIG['base_url'], api_key=LMSTUDIO_CONFIG['api_key'])
model_name = LMSTUDIO_CONFIG['model_name']

# --- System prompt ---
system_prompt = (
    "You are a clinical summarisation assistant specialised in stroke risk assessment. "
    "You MUST follow the exact format shown in the examples. "
    "Generate concise, clinician-friendly, medically accurate explanations using the top 3 features only. "
    "For EACH feature, write 1-2 sentences explaining its medical relevance and impact in natural, clinically accepted terms. "
    "Avoid casual or incorrect labels. For example, when describing BMI, apply standard classification thresholds (e.g., overweight for 25.0–29.9, obesity for ≥30). "
    "End with an Overall Summary sentence aligned with the predicted stroke risk status. "
    "Do not provide recommendations, treatments, management advice, lifestyle changes, monitoring suggestions, or next steps."
)

# --- User prompt with few-shot examples ---
user_prompt = f"""
Use the exact reasoning style and structure demonstrated in the examples below. 

Example 1:

Predicted Stroke Status: Risk
Patient Data:
Sex: Male
Age: 68 years
High Blood Pressure: Present
Cardiovascular Disease History: Absent
Marital History: Yes
Occupation Type: Private
Living Environment: Urban
Average Glucose Level: 120.5 mg/dL
Body Mass Index (BMI): 27.5 kg/m²
Smoking History: Formerly Smoked

Explanation:

The input features that had the greatest impact on the stroke risk prediction were Age, High Blood Pressure, and Smoking History.

Age: Advanced age increases stroke risk due to vascular aging including arterial stiffening and plaque buildup. At 68, this patient's stroke risk is significantly elevated.

High Blood Pressure: High blood pressure causes vascular wall damage and accelerates atherosclerosis, raising stroke probability.

Smoking History: Former smoking contributes to vascular inflammation and endothelial dysfunction, increasing risk.

Overall Summary: The combination of age, high blood pressure, and smoking history contribute to stroke risk.


Example 2:

Predicted Stroke Status: No Risk
Patient Data:
Sex: Female
Age: 42 years
High Blood Pressure: Absent
Cardiovascular Disease History: Absent
Marital History: Yes
Occupation Type: Self-employed
Living Environment: Urban
Average Glucose Level: 88.4 mg/dL
Body Mass Index (BMI): 22.7 kg/m²
Smoking History: Never Smoked

Explanation:

The input features that had the greatest impact on the stroke risk prediction were Age, High Blood Pressure, and Smoking History.

Age: At 42, the patient's vascular age is low, with reduced cumulative risk.

High Blood Pressure: Absence of high blood pressure removes one of the most significant drivers of cerebrovascular disease.

Smoking History: Never smoking avoids vascular toxins and inflammation.

Overall Summary: Youth, normal blood pressure, and no smoking history contribute to no stroke risk.

Now generate the stroke risk report for the following patient. 

CRITICAL: Your response MUST start with this exact sentence:
"The input features that had the greatest impact on the stroke risk prediction were {top_feature_list}."

Then, for EACH of the three features, write its name as a heading followed by 1-2 sentences explaining how it medically impacts this patient's stroke risk.

Finally, end with "Overall Summary:" followed by one sentence summarising the combined effect of these features. 
Do not include any advice or recommendations.

Predicted Stroke Status: {risk_label}

Patient Data:
{patient_feature_block}

Begin your explanation now:

"""

# --- Send to LM Studio ---
print("Explanation:")
response = client.chat.completions.create(
    model=model_name,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
)

# --- Display report ---
print(response.choices[0].message.content)