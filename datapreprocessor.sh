#!/bin/bash
# datapreprocessor.sh
# Shell script to run data preprocessing for stroke prediction system
# This script handles data loading, EDA, preprocessing, representative sample selection, and SMOTE

echo "========================================"
echo "Data Preprocessing for AI-XAI-LLM"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "ERROR: Virtual environment 'venv' not found!"
    echo "Please create a virtual environment first:"
    echo "  python -m venv venv"
    echo "  source venv/bin/activate  # On Linux/Mac/Git Bash"
    echo "  .\\venv\\Scripts\\Activate.ps1  # On Windows PowerShell"
    echo "  pip install -r requirements.txt"
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

# Check if Python script exists
if [ ! -f "datapreprocessor.py" ]; then
    echo "ERROR: datapreprocessor.py not found!"
    read -p "Press Enter to exit..."
    exit 1
fi

# Run the preprocessing script
echo "Running data preprocessing..."
echo ""
python datapreprocessor.py

# Check if script executed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Data preprocessing completed successfully!"
    echo "========================================"
    echo ""
    echo "Generated files:"
    echo "  data/representative_sample.csv (15 patients)"
    echo "  data/X_smote.csv, data/y_smote.csv (SMOTE-balanced data)"
    echo "  data/zscore_scaler.pkl, data/ordinal_encoder.pkl, data/knn_imputer.pkl"
    echo "  data/feature_names.pkl"
    echo "  plots/*.png (EDA visualizations)"
    echo ""
    echo "Next step: Run modeltrainer.sh to train the model"
    echo ""
    read -p "Press Enter to exit..."
else
    echo ""
    echo "ERROR: Data preprocessing failed!"
    read -p "Press Enter to exit..."
    exit 1
fi
