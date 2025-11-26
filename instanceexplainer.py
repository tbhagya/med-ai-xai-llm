#!/usr/bin/env python3
"""
Instance Explanation Script for Stroke Prediction System

This script handles:
- Loading trained model and preprocessed artifacts
- Loading representative sample
- Selecting a specific patient by index (command-line argument)
- Generating SHAP explanations
- Using LLM to produce human-readable clinical explanations

Usage:
    python instanceexplainer.py <patient_index>
    
Example:
    python instanceexplainer.py 12
"""

import os
import sys
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
        user_prompt,
        max_tokens=500,
        temperature=0.0,
        seed=42
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
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed
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

binary_value_map = {
    "hypertension": {1: "Present", 0: "Absent"},
    "heart_disease": {1: "Present", 0: "Absent"}
}

def format_value(feature, value):
    """Format feature value with appropriate units/labels"""
    if feature in binary_value_map:
        return binary_value_map[feature].get(value, value)
    if feature == "bmi":
        return f"{value:.1f} kg/m²"
    if feature == "avg_glucose_level":
        return f"{value:.1f} mg/dL"
    if feature == "age":
        return f"{int(value)} years"
    return value

# ================================
# CONFIGURATION
# ================================

# LLM configuration (LM Studio)
LMSTUDIO_CONFIG = {
    'base_url': "http://localhost:1234/v1",
    'api_key': "lm-studio",
    'model_name': "lmstudio-ai/gemma-2b-it-GGUF"
}

# Model paths
MODEL_PATH = "models/rf_stroke_model.pkl"
FEATURES_PATH = "models/rf_features.pkl"
SCALER_PATH = "data/zscore_scaler.pkl"
ENCODER_PATH = "data/ordinal_encoder.pkl"
REPRESENTATIVE_SAMPLE_PATH = "data/representative_sample.csv"

print("="*80)
print("STROKE PREDICTION - INSTANCE EXPLANATION")
print("="*80)

# ================================
# CREATE OUTPUT DIRECTORIES
# ================================
os.makedirs("reports", exist_ok=True)

# ================================
# PARSE COMMAND-LINE ARGUMENTS
# ================================

if len(sys.argv) < 2:
    print("\nERROR: Patient index not provided!")
    print("\nUsage:")
    print("  python instanceexplainer.py <patient_index>")
    print("\nExample:")
    print("  python instanceexplainer.py 12")
    print("\nNote: Patient index should be between 0 and 14 (15 patients in representative sample)")
    sys.exit(1)

try:
    PATIENT_INDEX = int(sys.argv[1])
except ValueError:
    print(f"\nERROR: Invalid patient index '{sys.argv[1]}'. Must be an integer.")
    sys.exit(1)

# ================================
# CHECK REQUIRED FILES
# ================================

required_files = [
    MODEL_PATH,
    FEATURES_PATH,
    SCALER_PATH,
    ENCODER_PATH,
    REPRESENTATIVE_SAMPLE_PATH
]

missing_files = [f for f in required_files if not os.path.exists(f)]

if missing_files:
    print("\nERROR: Missing required files:")
    for f in missing_files:
        print(f"  - {f}")
    print("\nPlease run datapreprocessor.sh and modeltrainer.sh first!")
    sys.exit(1)

# ================================
# LOAD MODEL AND ARTIFACTS
# ================================

print("\n" + "="*60)
print("LOADING MODEL AND ARTIFACTS")
print("="*60)

rf_model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURES_PATH)
scaler = joblib.load(SCALER_PATH)
encoder = joblib.load(ENCODER_PATH)

print("✓ Model loaded")
print("✓ Feature names loaded")
print("✓ Scaler loaded")
print("✓ Encoder loaded")

# ================================
# LOAD REPRESENTATIVE SAMPLE
# ================================

print("\n" + "="*60)
print("LOADING REPRESENTATIVE SAMPLE")
print("="*60)

representative_sample = pd.read_csv(REPRESENTATIVE_SAMPLE_PATH)
print(f"✓ Representative sample loaded: {len(representative_sample)} patients")
print(f"  Stroke distribution: {representative_sample['stroke'].value_counts().to_dict()}")

# Validate patient index
if not (0 <= PATIENT_INDEX < len(representative_sample)):
    print(f"\nERROR: Patient index {PATIENT_INDEX} out of range!")
    print(f"Valid range: 0 to {len(representative_sample)-1}")
    sys.exit(1)

# Get selected patient (original values)
patient_original = representative_sample.iloc[PATIENT_INDEX].copy()

print(f"\n✓ Selected patient index: {PATIENT_INDEX}")
print("\n--- Original Patient Data ---")
for feature, value in patient_original.items():
    if feature in feature_name_map:
        formatted = format_value(feature, value)
        print(f"  {feature_name_map[feature]}: {formatted}")

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

print("✓ Patient data preprocessed")

# ================================
# MAKE PREDICTION
# ================================

print("\n" + "="*60)
print("PREDICTION")
print("="*60)

stroke_proba = rf_model.predict_proba(X_patient)[0, 1]
stroke_pred = int(rf_model.predict(X_patient)[0])
risk_label = "High" if stroke_pred == 1 else "Low"

print(f"  Prediction: {stroke_pred}")
print(f"  Risk Label: {risk_label}")
print(f"  Probability: {stroke_proba:.3f}")

# ================================
# SHAP ANALYSIS
# ================================

print("\n" + "="*60)
print("SHAP ANALYSIS")
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

# Get top 3 features by SHAP value
df_positive = df_shap[df_shap["shap_value"] > 0].copy()

if len(df_positive) < 3:
    df_shap["abs_shap"] = df_shap["shap_value"].abs()
    top3 = df_shap.nlargest(3, "abs_shap").reset_index(drop=True)
    print(f"\nNote: Only {len(df_positive)} positive SHAP values; using top 3 by absolute value")
else:
    top3 = df_positive.nlargest(3, "shap_value").reset_index(drop=True)

print("\n✓ SHAP analysis completed")
print("\nTop 3 Contributing Features:")
for idx, row in top3.iterrows():
    feature_mapped = feature_name_map.get(row['feature'], row['feature'])
    value_formatted = format_value(row['feature'], row['value_original'])
    print(f"  {idx+1}. {feature_mapped}: {value_formatted} (SHAP: {row['shap_value']:+.4f})")

# ================================
# GENERATE LLM EXPLANATION
# ================================

print("\n" + "="*60)
print("GENERATING CLINICAL EXPLANATION")
print("="*60)

# Initialize LLM wrapper
llm = LLMWrapper(
    base_url=LMSTUDIO_CONFIG['base_url'],
    api_key=LMSTUDIO_CONFIG['api_key'],
    model_name=LMSTUDIO_CONFIG['model_name']
)

# Build patient feature block
patient_feature_block = "\n".join([
    f"{feature_name_map.get(f, f)}: {format_value(f, v)}"
    for f, v in patient_original.items()
    if f in feature_name_map
])

# Top feature list
top_feature_list = ", ".join([
    feature_name_map.get(f, f)
    for f in top3['feature'].tolist()
])

# Individual features
feature_1 = feature_name_map.get(top3.iloc[0]['feature'], top3.iloc[0]['feature'])
feature_2 = feature_name_map.get(top3.iloc[1]['feature'], top3.iloc[1]['feature'])
feature_3 = feature_name_map.get(top3.iloc[2]['feature'], top3.iloc[2]['feature'])

# System prompt
system_prompt = (
    "You are a clinical summarisation assistant specialised in stroke risk assessment. "
    "Generate concise, clinician-friendly, medically accurate explanations using the top 3 features only. "
    "Explain only in natural medical terms. "
    "CRITICAL: You MUST end with an 'Overall Summary:' paragraph that summarizes the combined effect. "
    "Do not provide recommendations, treatments, management advice, lifestyle changes, monitoring suggestions, or next steps."
)

# User prompt
user_prompt = f"""
Use the exact reasoning style and structure demonstrated in the examples below.

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

Clinical Explanation:
The input features that had the greatest impact on the stroke risk prediction were Age, Hypertension Status, and Smoking History.

Age: Advanced age increases stroke risk due to vascular aging including arterial stiffening and plaque buildup. At 68, this patient's stroke risk is significantly elevated.

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

Clinical Explanation:
The input features that had the greatest impact on the stroke risk prediction were Age, Hypertension Status, and Smoking History.

Age: At 42, the patient's vascular age is low, with reduced cumulative risk.

Hypertension Status: Absence of hypertension removes one of the most significant drivers of cerebrovascular disease.

Smoking History: Never smoking avoids vascular toxins and inflammation.

Overall Summary: Youth, normal blood pressure, and no smoking history contribute to low stroke risk.

Now generate the stroke risk report for the following patient:

Stroke Risk: {risk_label}

Patient Data:
{patient_feature_block}

Clinical Explanation:
You must explain exactly these three features: {feature_1}, {feature_2}, and {feature_3}.

REQUIRED FORMAT - You must follow this EXACTLY:

1. First paragraph: "The input features that had the greatest impact on the stroke risk prediction were {top_feature_list}."

2. Second paragraph: Start with "{feature_1}:" then write 2-3 sentences explaining this feature.

3. Third paragraph: Start with "{feature_2}:" then write 2-3 sentences explaining this feature.

4. Fourth paragraph: Start with "{feature_3}:" then write 2-3 sentences explaining this feature.

5. MANDATORY FINAL PARAGRAPH: Start with "Overall Summary:" followed by 1-2 sentences summarizing how these three features combined affect stroke risk.

IMPORTANT RULES:
- Write as plain text paragraphs exactly like the examples above
- Explain each feature in 2-3 clinical sentences describing its medical relevance
- Do not include any advice or recommendations
- You MUST include the "Overall Summary:" paragraph at the end - this is mandatory
- The Overall Summary must synthesize all three features together
"""

# Generate explanation
print("  Connecting to LLM service...")
llm_response = llm.generate_completion_with_metadata(
    system_prompt=system_prompt,
    user_prompt=user_prompt,
    max_tokens=600,
    temperature=0.0,
    seed=42
)

explanation_text = llm_response['text']

# Ensure Overall Summary is included
if "Overall Summary:" not in explanation_text:
    print("\n⚠ Warning: LLM response missing 'Overall Summary' - adding fallback summary")
    # Add a basic summary if LLM didn't provide one
    fallback_summary = f"\n\nOverall Summary: The combination of {feature_1}, {feature_2}, and {feature_3} contributes to a {risk_label.lower()} stroke risk profile for this patient."
    explanation_text += fallback_summary

print("✓ Clinical explanation generated")
if llm_response['usage']:
    print(f"  Tokens used: {llm_response['usage']['total_tokens']}")

# ================================
# FINAL OUTPUT
# ================================

print("\n" + "="*80)
print("CLINICAL STROKE RISK REPORT")
print("="*80)

print(f"\nPatient ID: {PATIENT_INDEX} (from representative sample)")
print(f"Prediction: {stroke_pred} | Risk Level: {risk_label} | Probability: {stroke_proba:.3f}")

print("\n--- Top 3 Contributing Features ---")
for idx, row in top3.iterrows():
    feature_mapped = feature_name_map.get(row['feature'], row['feature'])
    value_formatted = format_value(row['feature'], row['value_original'])
    print(f"{idx+1}. {feature_mapped}: {value_formatted} (SHAP: {row['shap_value']:+.4f})")

print("\n--- Clinical Explanation ---\n")
print(explanation_text)

print("\n" + "="*80)
print("END OF REPORT")
print("="*80)

# ================================
# SAVE REPORT
# ================================

report_filename = f"reports/patient_{PATIENT_INDEX}_report.txt"

with open(report_filename, 'w') as f:
    f.write("="*80 + "\n")
    f.write("CLINICAL STROKE RISK REPORT\n")
    f.write("="*80 + "\n\n")
    f.write(f"Patient ID: {PATIENT_INDEX} (from representative sample)\n")
    f.write(f"Prediction: {stroke_pred} | Risk Level: {risk_label} | Probability: {stroke_proba:.3f}\n\n")
    f.write("--- Top 3 Contributing Features ---\n")
    for idx, row in top3.iterrows():
        feature_mapped = feature_name_map.get(row['feature'], row['feature'])
        value_formatted = format_value(row['feature'], row['value_original'])
        f.write(f"{idx+1}. {feature_mapped}: {value_formatted} (SHAP: {row['shap_value']:+.4f})\n")
    f.write("\n--- Clinical Explanation ---\n\n")
    f.write(explanation_text)
    f.write("\n\n" + "="*80 + "\n")
    f.write("END OF REPORT\n")
    f.write("="*80 + "\n")

print(f"\n✓ Report saved to: {report_filename}")
