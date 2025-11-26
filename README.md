# AI-XAI-LLM: Stroke Prediction with Explainable AI

A machine learning project that predicts stroke risk using Random Forest classification and provides interpretable explanations through SHAP values and LLM-generated natural language summaries.

## Features

- **Stroke Risk Prediction**: Random Forest classifier trained on healthcare stroke dataset
- **Explainable AI**: SHAP (SHapley Additive exPlanations) for model interpretability
- **LLM Integration**: OpenAI-compatible API wrapper for generating natural language explanations
- **Comprehensive ML Pipeline**: Data preprocessing, feature engineering, and model evaluation
- **Representative Sample Analysis**: Focus on diverse patient profiles for explanations

## Project Structure

- `trained_ml.py` - Complete ML pipeline including EDA, preprocessing, model training, and evaluation
- `prediction_sys.py` - Patient explanation system with SHAP analysis and LLM-based explanations
- `requirements.txt` - Python package dependencies
- `README.md` - Project documentation

## Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/tbhagya/med-xai-llm.git
cd med-xai-llm
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

### 3. Activate Virtual Environment

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
.\venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

First Run the complete ML pipeline to train the stroke prediction model:

```bash
python trained_ml.py
```

This script will:
- Download the stroke prediction dataset from Kaggle
- Perform exploratory data analysis (EDA)
- Preprocess data (encoding, imputation, normalization)
- Handle class imbalance with SMOTE
- Train and evaluate Random Forest model
- Generate visualizations
- Save model artifacts

### Generating Predictions and Explanations

Second Run the prediction system to get patient-specific explanations:

```bash
python prediction_sys.py
```

This script will:
- Load the trained model and artifacts
- Analyze representative patient samples
- Generate SHAP explanations
- Create LLM-based natural language explanations (requires LM Studio or compatible API)

## LLM Configuration

The system uses an OpenAI-compatible API for generating explanations. Default configuration uses LM Studio:

```python
LMSTUDIO_CONFIG = {
    'base_url': "http://localhost:1234/v1",
    'api_key': "lm-studio",
    'model_name': "lmstudio-ai/gemma-2b-it-GGUF"
}
```

To use other LLM services (OpenAI, Azure OpenAI, etc.), modify the configuration in `prediction_sys.py`.

## Dataset

The project uses the **Stroke Prediction Dataset** from Kaggle:
- Source: [fedesoriano/stroke-prediction-dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
- Automatically downloaded via `kagglehub`

## Key Dependencies

- **scikit-learn**: Machine learning algorithms
- **pandas & numpy**: Data manipulation
- **SHAP**: Explainable AI
- **imbalanced-learn**: SMOTE for class imbalance
- **matplotlib & seaborn**: Visualization
- **openai**: LLM API integration
- **kagglehub**: Dataset management

## Model Artifacts

After training, the following files are generated:

- `rf_stroke_model.pkl` - Trained Random Forest model
- `rf_features.pkl` - Feature names
- `zscore_scaler.pkl` - StandardScaler for normalization
- `ordinal_encoder.pkl` - Encoder for categorical variables
- `representative_sample.csv` - Selected diverse patient samples
