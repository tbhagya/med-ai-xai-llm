#!/bin/bash
# instanceexplainer.sh
# Shell script to generate SHAP explanations and LLM-based clinical reports
# for a specific patient from the representative sample

echo "========================================"
echo "Instance Explanation for Stroke Prediction"
echo "========================================"
echo ""

# Check if patient index was provided
if [ $# -eq 0 ]; then
    echo "ERROR: Patient index not provided!"
    echo ""
    echo "Usage:"
    echo "  ./instanceexplainer.sh <patient_index>"
    echo ""
    echo "Example:"
    echo "  ./instanceexplainer.sh 12"
    echo ""
    echo "Note: Patient index should be between 0 and 14 (15 patients in representative sample)"
    read -p "Press Enter to exit..."
    exit 1
fi

PATIENT_INDEX=$1

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

# Check if required files exist
if [ ! -f "models/rf_stroke_model.pkl" ] || [ ! -f "data/representative_sample.csv" ]; then
    echo "ERROR: Required files not found!"
    echo "Please run datapreprocessor.sh and modeltrainer.sh first."
    read -p "Press Enter to exit..."
    exit 1
fi

# Check if Python script exists
if [ ! -f "instanceexplainer.py" ]; then
    echo "ERROR: instanceexplainer.py not found!"
    read -p "Press Enter to exit..."
    exit 1
fi

# Check if LM Studio is running (optional warning)
echo "Note: This script requires LM Studio to be running at http://localhost:1234"
echo "Please ensure LM Studio is running before continuing."
echo ""

# Run the explanation script
echo "Generating explanation for patient $PATIENT_INDEX..."
echo ""
python instanceexplainer.py $PATIENT_INDEX

# Check if script executed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Explanation generated successfully!"
    echo "========================================"
    echo ""
    echo "Generated file:"
    echo "  reports/patient_${PATIENT_INDEX}_report.txt"
    echo ""
    echo "You can run this script again with different patient indices (0-14)"
    echo ""
    read -p "Press Enter to exit..."
else
    echo ""
    echo "ERROR: Explanation generation failed!"
    echo "Please check:"
    echo "  1. LM Studio is running at http://localhost:1234"
    echo "  2. Patient index is valid (0-14)"
    echo "  3. All required model files exist"
    read -p "Press Enter to exit..."
    exit 1
fi
