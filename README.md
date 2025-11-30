# AI-XAI-LLM: Stroke Prediction with Explainable AI

A machine learning project that predicts stroke risk using Random Forest classification and provides interpretable explanations through SHAP values and LLM-generated natural language summaries.

## Features

- **Data Preprocessing**: Handles data loading, EDA, encoding, imputation, normalization, and SMOTE balancing
- **Model Training**: Performs hyperparameter tuning with GridSearchCV and 5-fold cross-validation
- **Instance Explanation**: Generates SHAP-based explanations with LLM-powered clinical interpretations
- **Representative Sample**: Selects diverse patient samples for testing and demonstration

## Project Structure

```
.
├── datapreprocessor.py          # Data loading, preprocessing, and SMOTE
├── datapreprocessor.sh          # Shell script to run preprocessing
├── modeltrainer.py              # Model training and cross-validation
├── modeltrainer.sh              # Shell script to run training
├── instanceexplainer.py         # SHAP explanations and LLM interpretations
├── instanceexplainer.sh         # Shell script to run explanations
├── trained_ml.py                # Legacy: All-in-one ML pipeline (EDA, preprocessing, training)
├── prediction_sys.py            # Legacy: Prediction system with SHAP and LLM explanations
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── data/                        # Preprocessed data and artifacts
│   ├── representative_sample.csv
│   ├── X_smote.csv, y_smote.csv
│   ├── zscore_scaler.pkl
│   ├── ordinal_encoder.pkl
│   ├── knn_imputer.pkl
│   └── feature_names.pkl
├── models/                      # Trained models and results
│   ├── rf_stroke_model.pkl
│   ├── rf_features.pkl
│   ├── best_params.pkl
│   ├── cv_results.csv
│   └── cv_summary.csv
├── plots/                       # Visualizations
│   ├── eda_*.png
│   └── feature_importance.png
└── reports/                     # Patient explanation reports
    └── patient_<N>_report.txt
```

## Requirements

- Python 3.8+
- Virtual environment (venv)
- LM Studio (for LLM-based explanations)
- Internet connection (for Kaggle dataset download)

## Setup Instructions

### 1. Clone or Download the Project

```bash
cd /path/to/stroke_prediction
```

### 2. Create Virtual Environment

**On Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On Windows (Git Bash):**
```bash
python -m venv venv
source venv/Scripts/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- numpy (>=1.24.0)
- pandas (>=2.0.0)
- scikit-learn (>=1.3.0)
- imbalanced-learn (>=0.11.0)
- joblib (>=1.3.0)
- matplotlib (>=3.7.0)
- seaborn (>=0.12.0)
- shap (>=0.42.0)
- openai (>=1.0.0)
- kagglehub (>=0.2.0)

### 4. Verify Installation

**Check Python version:**
```bash
python --version
# Should show Python 3.8 or higher
```

**Verify packages installed:**
```bash
pip list
```

**Test imports:**
```bash
python -c "import sklearn, pandas, numpy, shap, imblearn; print('All imports successful!')"
``` (>=0.2.0)

### 4. Verify Installation

**Check Python version:**
```bash
python --version
# Should show Python 3.8 or higher
```

**Verify packages installed:**
```bash
pip list
```

**Test imports:**
```bash
python -c "import sklearn, pandas, numpy, shap, imblearn; print('All imports successful!')"
```

### 5. Set Up LM Studio (for Instance Explanation)

1. Download and install [LM Studio](https://lmstudio.ai/)
2. Download a compatible model (e.g., `google/gemma-3n-e4b`)
3. Start the local server:
   - Open LM Studio
   - Load the model
   - Click "Start Server" (default: http://localhost:1234)

### 6. Make Shell Scripts Executable (Linux/Mac/Git Bash)

```bash
chmod +x datapreprocessor.sh
chmod +x modeltrainer.sh
chmod +x instanceexplainer.sh
```

## Usage

### Step 1: Data Preprocessing

Run the data preprocessing script to download the dataset, perform EDA, preprocess data, create representative sample, and apply SMOTE:

**On Linux/Mac/Git Bash:**
```bash
./datapreprocessor.sh
```

**On Windows (PowerShell):**
```powershell
.\datapreprocessor.sh
# OR if you have Git Bash installed
bash datapreprocessor.sh
# OR run Python directly
python datapreprocessor.py
```

**Output files:**
- `data/representative_sample.csv` - 15 diverse patient samples (5 stroke, 10 non-stroke)
- `data/X_smote.csv`, `data/y_smote.csv` - SMOTE-balanced training data
- `data/zscore_scaler.pkl` - StandardScaler for normalization
- `data/ordinal_encoder.pkl` - Encoder for categorical features
- `data/knn_imputer.pkl` - KNN imputer for missing values
- `data/feature_names.pkl` - List of feature names
- `plots/*.png` - EDA visualization plots

**Expected runtime:** 2-5 minutes (depending on internet speed for dataset download)

---

### Step 2: Model Training

Run the model training script to perform hyperparameter tuning, cross-validation, and train the final model:

**On Linux/Mac/Git Bash:**
```bash
./modeltrainer.sh
```

**On Windows (PowerShell):**
```powershell
.\modeltrainer.sh
# OR if you have Git Bash installed
bash modeltrainer.sh
# OR run Python directly
python modeltrainer.py
```

**Output files:**
- `models/rf_stroke_model.pkl` - Trained Random Forest model
- `models/rf_features.pkl` - Feature names used by the model
- `models/best_params.pkl` - Optimal hyperparameters from GridSearchCV
- `models/cv_results.csv` - Detailed cross-validation results for each fold
- `models/cv_summary.csv` - Summary statistics (mean ± std) for all metrics
- `plots/feature_importance.png` - Feature importance plot

**Expected runtime:** 10-30 minutes (depending on system performance)

**Cross-validation metrics tracked:**
- Accuracy
- Precision
- Recall (primary optimization metric)
- F1 Score
- ROC-AUC

---

### Step 3: Instance Explanation

Run the instance explanation script to generate SHAP-based explanations and LLM-powered clinical reports for a specific patient:

**On Linux/Mac/Git Bash:**
```bash
./instanceexplainer.sh <patient_index>
```

**On Windows (PowerShell):**
```powershell
.\instanceexplainer.sh <patient_index>
# OR if you have Git Bash installed
bash instanceexplainer.sh <patient_index>
# OR run Python directly
python instanceexplainer.py <patient_index>
```

**Arguments:**
- `<patient_index>`: Integer between 0 and 14 (patient index in representative sample)

**Example:**
```bash
./instanceexplainer.sh 12
```

**Output files:**
- `reports/patient_<index>_report.txt` - Clinical stroke risk report with SHAP explanations and LLM interpretation

**Expected runtime:** 30 seconds - 2 minutes (depending on LLM response time)

**Report contents:**
- Patient demographic and clinical data
- Stroke risk prediction (High/Low) with probability
- Top 3 SHAP-contributing features
- Human-readable clinical explanation generated by LLM

**Generate reports for all patients:**

**Linux/Mac/Git Bash:**
```bash
# Generate reports for all 15 patients
for i in {0..14}; do ./instanceexplainer.sh $i; done
```

**Windows PowerShell:**
```powershell
# Generate reports for all 15 patients
0..14 | ForEach-Object { python instanceexplainer.py $_ }
```

---

## Example Workflow

```bash
# 1. Activate virtual environment
source venv/bin/activate  # Linux/Mac
# OR
.\venv\Scripts\Activate.ps1  # Windows PowerShell

# 2. Preprocess data
./datapreprocessor.sh

# 3. Train model
./modeltrainer.sh

# 4. Generate explanations for different patients
./instanceexplainer.sh 0   # First stroke patient
./instanceexplainer.sh 5   # First non-stroke patient
./instanceexplainer.sh 12  # Another patient
```

## Alternative: Legacy All-in-One Scripts

For quick testing or educational purposes, you can use the legacy scripts that combine multiple steps:

### trained_ml.py - Complete ML Pipeline

Runs the entire ML pipeline (EDA, preprocessing, training, evaluation) in one script:

```bash
python trained_ml.py
```

**What it does:**
- Downloads and loads the dataset
- Performs exploratory data analysis (EDA)
- Preprocesses data (encoding, imputation, normalization)
- Selects representative samples
- Applies SMOTE for class balancing
- Performs hyperparameter tuning with GridSearchCV
- Trains and evaluates the model with 5-fold cross-validation
- Saves all artifacts to the root directory

**Runtime:** ~15-25 minutes

**Output:** Saves model artifacts (`rf_stroke_model.pkl`, `rf_features.pkl`, etc.) and `representative_sample.csv` to the root directory.

### prediction_sys.py - Prediction & Explanation System

Generates predictions and explanations for a predefined patient:

```bash
# Ensure LM Studio is running first!
python prediction_sys.py
```

**What it does:**
- Loads trained model and artifacts
- Loads representative sample
- Makes predictions for a selected patient (configurable in script)
- Generates SHAP explanations
- Uses LLM to create clinical explanations
- Displays results in console

**Prerequisites:** 
- Must run `trained_ml.py` first to generate model artifacts
- LM Studio must be running

**Runtime:** ~30 seconds - 1 minute

**Note:** The legacy scripts save artifacts to the root directory, while the modular workflow uses subdirectories (`data/`, `models/`, `plots/`, `reports/`).

## Important Notes

### Result Variability

**Note: Due to cross-validation and the use of the language model, these scripts will produce slightly different results each time they are executed.**

This variability occurs due to:
1. **Cross-validation shuffling**: Even with fixed random seeds, slight variations may occur
2. **LLM temperature**: While set to 0.0 for deterministic behavior, LLM responses may still vary slightly
3. **SMOTE randomization**: SMOTE uses random sampling (controlled by `random_state=42`)
4. **Random Forest**: Tree-based models have inherent randomness (controlled by `random_state=42`)

The core predictions and SHAP values remain stable, but exact metric values and LLM phrasing may differ slightly between runs.

### Dataset

This project uses the [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) from Kaggle. The dataset is automatically downloaded via `kagglehub` during preprocessing.

**Dataset statistics:**
- ~5,000 patient records
- 11 features (demographic, clinical, lifestyle)
- Binary classification (stroke: 0/1)
- Imbalanced classes (~5% stroke cases)

### Model Configuration

**Hyperparameters tuned:**
- `n_estimators`: [200, 400]
- `max_depth`: [10, 20, None]
- `min_samples_split`: [2, 5]
- `min_samples_leaf`: [1, 2]
- `max_features`: ['sqrt', 'log2']

**Optimization metric:** Recall (to prioritize identifying stroke cases)

### LLM Configuration

Default configuration for LM Studio:
- **Base URL:** http://localhost:1234/v1
- **API Key:** lm-studio (default)
- **Model:** lmstudio-ai/gemma-2b-it-GGUF
- **Temperature:** 0.0 (deterministic)
- **Max Tokens:** 500

To change LLM settings, edit the `LMSTUDIO_CONFIG` dictionary in `instanceexplainer.py`.

## Troubleshooting

### Issue: "Virtual environment not found"
**Solution:** Create virtual environment first:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: "Module not found"
**Solution:** Ensure virtual environment is activated and dependencies are installed:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: "Preprocessed data not found"
**Solution:** Run scripts in order:
1. `datapreprocessor.sh`
2. `modeltrainer.sh`
3. `instanceexplainer.sh`

### Issue: "LLM connection error"
**Solution:** 
1. Ensure LM Studio is running
2. Check server is at http://localhost:1234
3. Verify model is loaded in LM Studio
4. Test connection: `curl http://localhost:1234/v1/models`

### Issue: "Permission denied" (Linux/Mac/Git Bash)
**Solution:** Make scripts executable:
```bash
chmod +x *.sh
```

### Issue: "Git Bash closes immediately"
**Solution:** 
1. Right-click on `datapreprocessor.sh` → Open with → Git Bash
2. Or run from PowerShell: `bash datapreprocessor.sh`
3. The scripts now have pauses to keep the window open
4. Alternatively, open Git Bash first, then run: `./datapreprocessor.sh`

### Issue: "Dataset download fails"
**Solution:**
1. Check internet connection
2. Ensure Kaggle API access is available
3. (Optional) Configure Kaggle API credentials:
   - Create account at [kaggle.com](https://www.kaggle.com)
   - Go to Account → API → Create New API Token
   - Download `kaggle.json` and place in:
     - **Windows:** `C:\Users\<YourUsername>\.kaggle\kaggle.json`
     - **Linux/Mac:** `~/.kaggle/kaggle.json`
   - Set permissions (Linux/Mac): `chmod 600 ~/.kaggle/kaggle.json`
4. Try manual download from [Kaggle dataset page](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) and place `healthcare-dataset-stroke-data.csv` in the working directory

## Output Files Summary

| File | Description | Generated By |
|------|-------------|--------------|
| **data/** | **Preprocessed Data Directory** | |
| `data/representative_sample.csv` | 15 diverse patient samples | datapreprocessor.sh |
| `data/X_smote.csv`, `data/y_smote.csv` | SMOTE-balanced training data | datapreprocessor.sh |
| `data/zscore_scaler.pkl` | StandardScaler for normalization | datapreprocessor.sh |
| `data/ordinal_encoder.pkl` | Categorical encoder | datapreprocessor.sh |
| `data/knn_imputer.pkl` | KNN imputer for BMI | datapreprocessor.sh |
| `data/feature_names.pkl` | List of feature names | datapreprocessor.sh |
| **models/** | **Trained Models Directory** | |
| `models/rf_stroke_model.pkl` | Trained Random Forest model | modeltrainer.sh |
| `models/rf_features.pkl` | Model feature names | modeltrainer.sh |
| `models/best_params.pkl` | Optimal hyperparameters | modeltrainer.sh |
| `models/cv_results.csv` | Cross-validation fold results | modeltrainer.sh |
| `models/cv_summary.csv` | CV summary statistics | modeltrainer.sh |
| **plots/** | **Visualizations Directory** | |
| `plots/eda_*.png` | EDA visualization plots | datapreprocessor.sh |
| `plots/feature_importance.png` | Feature importance plot | modeltrainer.sh |
| **reports/** | **Patient Reports Directory** | |
| `reports/patient_<N>_report.txt` | Clinical explanation report | instanceexplainer.sh |

## Features Used

The system uses the following patient features for prediction:

| Feature | Description | Type |
|---------|-------------|------|
| `gender` | Patient sex (Male/Female) | Categorical |
| `age` | Patient age in years | Numeric |
| `hypertension` | Hypertension status (0/1) | Binary |
| `heart_disease` | Heart disease history (0/1) | Binary |
| `ever_married` | Marital status | Binary |
| `work_type` | Occupation type | Categorical |
| `Residence_type` | Urban/Rural residence | Categorical |
| `avg_glucose_level` | Average blood glucose (mg/dL) | Numeric |
| `bmi` | Body Mass Index (kg/m²) | Numeric |
| `smoking_status` | Smoking history | Categorical |

