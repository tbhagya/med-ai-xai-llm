#!/bin/bash
# modeltrainer.sh
# Shell script to train the stroke prediction model
# This script performs hyperparameter tuning, cross-validation, and model training

echo "========================================"
echo "Model Training for Stroke Prediction"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "ERROR: Virtual environment 'venv' not found!"
    echo "Please create a virtual environment first and install dependencies."
    read -p "Press Enter to exit..."
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
if [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "ERROR: Cannot find virtual environment activation script!"
    read -p "Press Enter to exit..."
    exit 1
fi

# Check if preprocessing was done
if [ ! -f "data/X_smote.csv" ] || [ ! -f "data/y_smote.csv" ]; then
    echo "ERROR: Preprocessed data not found!"
    echo "Please run datapreprocessor.sh first."
    read -p "Press Enter to exit..."
    exit 1
fi

# Check if Python script exists
if [ ! -f "modeltrainer.py" ]; then
    echo "ERROR: modeltrainer.py not found!"
    read -p "Press Enter to exit..."
    exit 1
fi

# Run the training script
echo "Running model training..."
echo "This may take several minutes due to hyperparameter tuning and cross-validation..."
echo ""
python modeltrainer.py

# Check if script executed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Model training completed successfully!"
    echo "========================================"
    echo ""
    echo "Generated files:"
    echo "  models/rf_stroke_model.pkl (trained Random Forest model)"
    echo "  models/rf_features.pkl (feature names)"
    echo "  models/best_params.pkl (optimal hyperparameters)"
    echo "  models/cv_results.csv, models/cv_summary.csv (cross-validation metrics)"
    echo "  plots/feature_importance.png"
    echo ""
    echo "Next step: Run instanceexplainer.sh <patient_index> to explain predictions"
    echo "Example: ./instanceexplainer.sh 12"
    echo ""
    read -p "Press Enter to exit..."
else
    echo ""
    echo "ERROR: Model training failed!"
    read -p "Press Enter to exit..."
    exit 1
fi
