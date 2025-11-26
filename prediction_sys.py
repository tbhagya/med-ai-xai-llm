#!/usr/bin/env python3
"""
ML-based Stroke Risk Prediction and Explanation System

Uses representative sample for individual predictions and LLM-based explanations.
Includes OpenAI-compatible LLM wrapper function.
"""

import os
import joblib
import pandas as pd
import shap
import numpy as np
from openai import OpenAI
from sklearn.preprocessing import StandardScaler

# ================================
# LLM WRAPPER FUNCTION (OpenAI API Compatible)
# ================================

class LLMWrapper:
    """
    OpenAI-compatible LLM wrapper for flexible integration.
    Supports any LLM service that implements OpenAI API format.
    """
    
    def __init__(self, base_url, api_key, model_name):
        """
        Initialize LLM client.
        
        Args:
            base_url (str): Base URL of the LLM service
            api_key (str): API key for authentication
            model_name (str): Model name/identifier
        """
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
    
    def generate_completion(
        self,
        system_prompt,
        user_prompt
    ):
        """
        Generate text completion using chat API.
        
        Args:
            system_prompt (str): System message defining assistant behavior
            user_prompt (str): User query/request
            max_tokens (int): Maximum tokens in response
            temperature (float): Sampling temperature (0.0 = deterministic)
            seed (int): Random seed for reproducibility
            
        Returns:
            str: Generated text response
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating completion: {str(e)}"
    
    def generate_completion_with_metadata(
        self,
        system_prompt,
        user_prompt,
        max_tokens=500,
        temperature=0.0,
        seed=42
    ):
        """
        Generate completion with full metadata.
        
        Returns:
            dict: Contains 'text', 'usage', 'model', 'finish_reason'
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed
            )
            
            return {
                'text': response.choices[0].message.content,
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                },
                'model': response.model,
                'finish_reason': response.choices[0].finish_reason
            }
        except Exception as e:
            return {
                'text': f"Error: {str(e)}",
                'usage': None,
                'model': None,
                'finish_reason': 'error'
            }

# ================================
# CONFIGURATION
# ================================

# Patient selection (0-based index in representative sample)
PATIENT_INDEX = 1  # Select patient from representative sample

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
    "work_type": {0: "Never_worked", 1: "Private", 2: "Self-employed", 3: "Govt_job", 4: "children"},
    "Residence_type": {0: "Rural", 1: "Urban"},
    "smoking_status": {0: "Never smoked", 1: "Formerly smoked", 2: "Smokes", 3: "Unknown"}
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

# Validate patient index
if not (0 <= PATIENT_INDEX < len(representative_sample)):
    raise IndexError(f"PATIENT_INDEX {PATIENT_INDEX} out of range (0-{len(representative_sample)-1})")

# Get selected patient (original values)
patient_original = representative_sample.iloc[PATIENT_INDEX].copy()

# ================================
# PREPROCESS PATIENT FOR PREDICTION
# ================================

print("\n" + "="*60)
print("PREPROCESSING PATIENT DATA")
print("="*60)

# Create single-row dataframe
patient_df = pd.DataFrame([patient_original])

# Encode categorical features (already in encoded form in CSV, but we'll use original logic)
cat_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
numeric_cols = ['age', 'avg_glucose_level', 'bmi']

# Normalize numeric features
patient_df[numeric_cols] = scaler.transform(patient_df[numeric_cols])

# Extract features for prediction
X_patient = patient_df[feature_names]

print("[OK] Patient data preprocessed")

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
    print(f"  {idx+1}. {feature_mapped}: {row['shap_value']:+.4f}")

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

# Build patient feature block
patient_feature_block = "\n".join([
    f"{feature_name_map.get(f, f)}: {format_value(f, v)}"
    for f, v in patient_original.items()
    if f in feature_name_map
])

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
response = client.chat.completions.create(
    model=model_name,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
)

# --- Display report ---
print(response.choices[0].message.content)
