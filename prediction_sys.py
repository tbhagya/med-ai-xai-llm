#!/usr/bin/env python3
"""
ML-based Stroke Risk Prediction and Explanation System

Uses representative sample for individual predictions and LLM-based explanations.
Interactive patient selection via user input.
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
MODEL_PATH = "rf_stroke_model.pkl"
FEATURES_PATH = "rf_features.pkl"
SCALER_PATH = "zscore_scaler.pkl"
ENCODER_PATH = "ordinal_encoder.pkl"
REPRESENTATIVE_SAMPLE_PATH = "representative_sample.csv"

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
    "hypertension": "Hypertension Status",
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
    "work_type": {0: "Govt_job", 1: "Never_worked", 2: "Private", 3: "Self-employed", 4: "children"},
    "Residence_type": {0: "Rural", 1: "Urban"},
    "smoking_status": {0: "Unknown", 1: "Formerly smoked", 2: "never smoked", 3: "smokes"}
}

# Backward compatibility alias
binary_value_map = {
    "hypertension": categorical_value_map["hypertension"],
    "heart_disease": categorical_value_map["heart_disease"]
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

rf_model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURES_PATH)
scaler = joblib.load(SCALER_PATH)
encoder = joblib.load(ENCODER_PATH)

print("[OK] Model loaded")
print("[OK] Feature names loaded")
print("[OK] Scaler loaded")
print("[OK] Encoder loaded")

# ================================
# LOAD REPRESENTATIVE SAMPLE
# ================================

print("\n" + "="*60)
print("LOADING REPRESENTATIVE SAMPLE")
print("="*60)

representative_sample = pd.read_csv(REPRESENTATIVE_SAMPLE_PATH)
print(f"Representative sample loaded: {len(representative_sample)} patients")
print(f"Valid patient indices: 0 to {len(representative_sample)-1}")

# ================================
# INTERACTIVE PATIENT SELECTION
# ================================

print("\n" + "="*60)
print("PATIENT SELECTION")
print("="*60)

while True:
    try:
        user_input = input(f"\nEnter patient index (0-{len(representative_sample)-1}): ").strip()
        PATIENT_INDEX = int(user_input)
        
        if 0 <= PATIENT_INDEX < len(representative_sample):
            print(f"[OK] Selected patient index: {PATIENT_INDEX}")
            break
        else:
            print(f"[ERROR] Index out of range. Please enter a value between 0 and {len(representative_sample)-1}")
    except ValueError:
        print("[ERROR] Invalid input. Please enter a valid integer.")
    except KeyboardInterrupt:
        print("\n\n[EXIT] Program terminated by user.")
        exit(0)

# Get selected patient (original values)
patient_original = representative_sample.iloc[PATIENT_INDEX].copy()

# ================================
# DISPLAY PATIENT INFORMATION
# ================================

print("\n" + "="*60)
print("PATIENT INFORMATION")
print("="*60)
print(f"\nPatient Index: {PATIENT_INDEX}")
print("\nFeature Values:")

for feature, value in patient_original.items():
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

# Create single-row dataframe (data is already preprocessed in representative_sample.csv)
patient_df = pd.DataFrame([patient_original])

# Extract features for prediction
X_patient = patient_df[feature_names]

print("[OK] Patient data prepared")

# ================================
# AI OUTPUT
# ================================

print("\n" + "="*60)
print("AI OUTPUT")
print("="*60)

stroke_pred = int(rf_model.predict(X_patient)[0])
risk_label = "High" if stroke_pred == 1 else "Low"

print(f"Stroke Risk Prediction: {risk_label}")

# ================================
# XAI OUTPUT
# ================================

print("\n" + "="*60)
print("XAI OUTPUT")
print("="*60)

explainer = shap.Explainer(rf_model)
shap_exp = explainer(X_patient)
patient_shap_values = shap_exp.values[0, :, 1]

# Create SHAP dataframe
df_shap = pd.DataFrame({
    "feature": feature_names,
    "value_encoded": X_patient.iloc[0].values,
    "shap_value": patient_shap_values
})

# Add original values
def get_original_value(feature):
    return patient_original[feature] if feature in patient_original.index else None

df_shap["value_original"] = df_shap["feature"].apply(get_original_value)

# Get top 3 features by SHAP value (for LLM explanation)
df_positive = df_shap[df_shap["shap_value"] > 0].copy()

if len(df_positive) < 3:
    df_shap["abs_shap"] = df_shap["shap_value"].abs()
    top3 = df_shap.nlargest(3, "abs_shap").reset_index(drop=True)
else:
    top3 = df_positive.nlargest(3, "shap_value").reset_index(drop=True)

# Display ALL SHAP values sorted by absolute value
print("\nFeature Impact:")
df_shap["abs_shap"] = df_shap["shap_value"].abs()
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

# Build patient feature block
patient_feature_block = "\n".join([
    f"{feature_name_map.get(f, f)}: {format_value(f, v)}"
    for f, v in patient_original.items()
    if f in feature_name_map
])

# Display Stroke Risk and Patient Data
print(f"\nStroke Risk: {risk_label}\n")
print("Patient Data:")
print(patient_feature_block)
print()

# Initialize LLM wrapper
llm = LLMWrapper(
    base_url=LMSTUDIO_CONFIG['base_url'],
    api_key=LMSTUDIO_CONFIG['api_key'],
    model_name=LMSTUDIO_CONFIG['model_name']
)

# For notebook-style compatibility: create client and model_name variables
client = OpenAI(base_url=LMSTUDIO_CONFIG['base_url'], api_key=LMSTUDIO_CONFIG['api_key'])
model_name = LMSTUDIO_CONFIG['model_name']

# Provide `prediction` variable to match notebook prompts that use {prediction}
prediction = stroke_pred

# --- System prompt ---
system_prompt = (
    "You are a clinical summarisation assistant specialised in stroke risk assessment. "
    "You MUST follow the exact format shown in the examples. "
    "Generate concise, clinician-friendly, medically accurate explanations using the top 3 features only. "
    "For EACH feature, write 2-3 sentences explaining its medical relevance and impact. "
    "Explain only in natural medical terms. "
    "End with an Overall Summary sentence. "
    "Do not provide recommendations, treatments, management advice, lifestyle changes, monitoring suggestions, or next steps."
)

# --- User prompt with few-shot examples included ---
user_prompt = f"""
Use the exact reasoning style amd structure demonstrated in the examples below. 

Example 1:

Stroke Risk: High
Patient Data:
Sex: Male
Age: 68
Hypertension Status: Present
Cardiovascular Disease History: Absent
Marital History: Yes
Occupation Type: Private
Living Environment: Urban
Average Glucose Level: 120.5 mg/dL
Body Mass Index (BMI): 27.5 kg/m²
Smoking History: formerly smoked

Explanation:

The input features that had the greatest impact on the stroke risk prediction were Age, Hypertension Status, and Smoking History.

Age: Advanced age increases stroke risk due to vascular aging including arterial stiffening and plaque buildup. 
At 68, this patient's stroke risk is significantly elevated.

Hypertension Status: High blood pressure causes vascular wall damage and accelerates atherosclerosis, raising stroke probability.

Smoking History: Former smoking contributes to vascular inflammation and endothelial dysfunction, increasing risk.

Overall Summary: The combination of age, hypertension, and smoking history substantially elevates stroke risk.


Example 2:

Stroke Risk: Low
Patient Data:
Sex: Female
Age: 42
Hypertension Status: Absent
Cardiovascular Disease History: Absent
Marital History: Yes
Occupation Type: Self-employed
Living Environment: Urban
Average Glucose Level: 88.4 mg/dL
Body Mass Index (BMI): 22.7 kg/m²
Smoking History: never smoked

Explanation:

The input features that had the greatest impact on the stroke risk prediction were Age, Hypertension Status, and Smoking History.

Age: At 42, the patient's vascular age is low, with reduced cumulative risk.

Hypertension Status: Absence of hypertension removes one of the most significant drivers of cerebrovascular disease.

Smoking History: Never smoking avoids vascular toxins and inflammation.

Overall Summary: Youth, normal blood pressure, and no smoking history contribute to low stroke risk.

Now generate the stroke risk report for the following patient. 

CRITICAL: Your response MUST start with this exact sentence:
"The input features that had the greatest impact on the stroke risk prediction were {top_feature_list}."

Then, for EACH of the three features, write its name as a heading followed by 2-3 sentences explaining how it medically impacts this patient's stroke risk.


Finally, end with "Overall Summary:" followed by one one sentence summarizing the combined effect of these features. 
Do not include any advice or recommendations.


Stroke Risk: {risk_label}

Patient Data:
{patient_feature_block}

Begin your explanation now:

"""

# --- Send to LM Studio (notebook-style call) ---
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