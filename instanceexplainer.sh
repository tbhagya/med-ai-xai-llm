#!/bin/bash
# instanceexplainer.sh
# Shell script to generate SHAP explanations and LLM-based clinical reports
# for a specific patient from the representative sample

echo "========================================"
echo "Instance Explanation for Stroke Prediction"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "ERROR: Virtual environment 'venv' not found!"
    echo "Please run setup.sh first to create the environment."
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
if [ ! -f "models/rf_stroke_model.pkl" ]; then
    echo "ERROR: Trained model not found!"
    echo "Please run modeltrainer.sh first."
    read -p "Press Enter to exit..."
    exit 1
fi

if [ ! -f "data/representative_sample_scaled.csv" ] || [ ! -f "data/representative_sample_display.csv" ]; then
    echo "ERROR: Representative sample data not found!"
    echo "Please run datapreprocessor.sh first."
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
echo "IMPORTANT: This script requires LM Studio to be running at http://localhost:1234"
echo "Please ensure LM Studio is running with a loaded model before continuing."
echo ""

# Main loop for multiple patient explanations
while true; do
    # Run the explanation script (interactive mode)
    echo "Starting patient explanation system (interactive mode)..."
    echo ""
    python instanceexplainer.py

    # Check if script executed successfully
    if [ $? -eq 0 ]; then
        echo ""
        echo "========================================"
        echo "Explanation generated successfully!"
        echo "========================================"
        echo ""
        
        # Ask if user wants to check another patient
        read -p "Do you want to check another patient? (y/n): " -n 1 -r
        echo ""
        
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Exiting instance explainer. Thank you!"
            echo ""
            break
        fi
        
        echo ""
    else
        echo ""
        echo "ERROR: Explanation generation failed!"
        echo ""
        echo "Please check:"
        echo "  1. LM Studio is running at http://localhost:1234"
        echo "  2. A model is loaded in LM Studio (e.g., gemma-2b-it-GGUF)"
        echo "  3. Patient index entered is valid (0-14)"
        echo "  4. All required model files exist in models/ and data/ folders"
        echo ""
        read -p "Press Enter to exit..."
        exit 1
    fi
done
