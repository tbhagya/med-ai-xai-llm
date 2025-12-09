#!/bin/bash
# setup.sh
# One-time setup script for AI-XAI-LLM project
# Creates virtual environment and installs all dependencies

echo "========================================"
echo "Environment Setup for AI-XAI-LLM"
echo "========================================"
echo ""

# Check if Python is available
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
    echo "ERROR: Python is not installed or not in PATH!"
    echo "Please install Python 3.8 or higher from https://www.python.org/"
    read -p "Press Enter to exit..."
    exit 1
fi

# Use python3 if available, otherwise python
PYTHON_CMD="python"
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
fi

echo "Using Python: $PYTHON_CMD"
$PYTHON_CMD --version
echo ""

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "ERROR: requirements.txt not found!"
    echo "Please ensure requirements.txt is in the current directory."
    read -p "Press Enter to exit..."
    exit 1
fi

# Remove existing virtual environment if it exists
if [ -d "venv" ]; then
    echo "Existing virtual environment found."
    read -p "Do you want to delete and recreate it? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing virtual environment..."
        rm -rf venv
        echo "✓ Old virtual environment removed"
    else
        echo "Keeping existing virtual environment..."
    fi
    echo ""
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv venv
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create virtual environment!"
        echo "Try running: $PYTHON_CMD -m pip install --upgrade pip"
        read -p "Press Enter to exit..."
        exit 1
    fi
    
    echo "✓ Virtual environment created successfully"
    echo ""
fi

# Activate virtual environment
echo "Activating virtual environment..."
if [ -f "venv/Scripts/activate" ]; then
    # Windows Git Bash
    source venv/Scripts/activate
elif [ -f "venv/bin/activate" ]; then
    # Linux/Mac
    source venv/bin/activate
else
    echo "ERROR: Cannot find virtual environment activation script!"
    read -p "Press Enter to exit..."
    exit 1
fi

echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip
echo ""

# Install requirements
echo "Installing required packages from requirements.txt..."
echo "This may take a few minutes..."
echo ""
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Failed to install some packages!"
    echo "Please check the error messages above."
    read -p "Press Enter to exit..."
    exit 1
fi

echo ""
echo "========================================"
echo "Setup Completed Successfully!"
echo "========================================"
echo ""
echo "Installed packages:"
pip list
echo ""
echo "Next steps:"
echo "  1. Run './datapreprocessor.sh' to preprocess the data"
echo "  2. Run './modeltrainer.sh' to train the model"
echo "  3. Run './instanceexplainer.sh' to make predictions and check how predictions were made"
echo ""
echo "Note: On Windows, you may need to run these scripts using Git Bash"
echo "      or use the .bat versions if available."
echo ""