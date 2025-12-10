# AI-XAI-LLM

## Requirements

- Python 3.8+
- LM Studio (for LLM-based explanations)
- Internet connection (for Kaggle dataset download)

## Quick Start

### 1. Clone or Download the Project
```bash
cd /path/to/med-ai-xai-llm
```

### 2. Set Up LM Studio

1. Download and install [LM Studio](https://lmstudio.ai/)
2. Download a compatible model (e.g., `google/gemma-3n-e4b`)
3. Start the local server:
   - Open LM Studio
   - Load the model
   - Click "Start Server" (default: http://localhost:1234)
4. **Keep LM Studio running** when you run the instance explainer

### 3. One-Time Setup

Run the setup script to create virtual environment and install all dependencies:

**On Linux/Mac/Git Bash:**
```bash
chmod +x setup.sh
./setup.sh
```

**On Windows:**
```powershell
bash setup.sh
```

### 4. Run Data Preprocessing
```bash
chmod +x datapreprocessor.sh
./datapreprocessor.sh
```

### 5. Run Model Training
```bash
chmod +x modeltrainer.sh
./modeltrainer.sh
```

### 6. Run Instance Explainer

**Make sure LM Studio is running at http://localhost:1234**
```bash
chmod +x instanceexplainer.sh
./instanceexplainer.sh
```

The script will prompt you to enter a patient index (0-14).

---

## Running Python Scripts Manually

If you prefer to run Python scripts directly instead of using shell scripts, you need to set up the environment first:

### Setup Environment

**On Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Run Scripts

After activating the virtual environment:
```bash
# Step 1: Data preprocessing
python datapreprocessor.py

# Step 2: Model training
python modeltrainer.py

# Step 3: Instance explanation (interactive)
python instanceexplainer.py
```

---

## Important Notes

Due to cross-validation and language model usage, results may vary slightly between runs.
