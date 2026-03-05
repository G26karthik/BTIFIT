# BotTrainer — How to Run
---

## Prerequisites
- Python 3.10+ with pip
- Node.js 18+ with npm
- Gemini API key (set in `BotTrainer/.env`)
- Kaggle API key at `~/.kaggle/kaggle.json` (for dataset download)

---

## Step 1 — Setup Python Environment
```bash
# All commands run from the project root (AIIP/)
cd AIIP

# Create venv inside BotTrainer/
python -m venv BotTrainer/.venv

# Activate (Windows):
.\BotTrainer\.venv\Scripts\Activate.ps1
# Activate (macOS/Linux):
source BotTrainer/.venv/bin/activate

# Install as editable package (makes `python -m BotTrainer` work)
pip install -e .
```

## Step 2 — Download Dataset (Module 3 — Data Loader)
```bash
python -m BotTrainer data_loader
```
Downloads CLINC150 from Kaggle, creates `data/intents.json` and `data/eval_dataset.json`.

## Step 3 — Run ALL Modules in One Command
```bash
python -m BotTrainer
```
Runs every module (1–6) in sequence: utils → llm_client → data_loader → intent_classifier → entity_extractor → evaluator.

## Step 4 — Run Individual Modules (Optional)
```bash
# Module 1 — Utility functions (no API key needed)
python -m BotTrainer utils

# Module 2 — LLM client (tests Gemini connection)
python -m BotTrainer llm_client

# Module 3 — Data loader (downloads CLINC150 dataset)
python -m BotTrainer data_loader

# Module 4 — Intent classifier (classifies 3 sample messages)
python -m BotTrainer intent_classifier

# Module 5 — Entity extractor (extracts entities from 3 samples)
python -m BotTrainer entity_extractor

# Module 6 — Evaluator (runs evaluation on 10 samples, shows metrics)
python -m BotTrainer evaluator
```

## Step 5 — Run Full CLI Pipeline
```bash
python -m BotTrainer pipeline
```
Interactive CLI that classifies intent + extracts entities from your text.

## Step 6 — Run Streamlit App
```bash
python run.py --app
```
Opens the full Streamlit dashboard at `http://localhost:8501`.

---

## Step 7 — Run FastAPI Backend + React Frontend (Production)

### Terminal 1 — Start Backend API
```bash
.\BotTrainer\.venv\Scripts\Activate.ps1
uvicorn BotTrainer.api:app --reload --port 8000
```
API available at `http://localhost:8000/docs` (Swagger UI).

### Terminal 2 — Start React Frontend
```bash
cd frontend
npm install   # first time only
npm run dev
```
Opens the ChatGPT-style UI at `http://localhost:5173`.

---

## Step 8 — Run Tests
```bash
python -m pytest tests/ -v
```

---

## API Endpoints
| Method | Endpoint        | Description                          |
|--------|-----------------|--------------------------------------|
| GET    | /api/health     | Health check + model info            |
| POST   | /api/predict    | Classify intent & extract entities   |
| GET    | /api/intents    | List all 150+ intent names           |
| GET    | /api/history    | View cached prediction history       |