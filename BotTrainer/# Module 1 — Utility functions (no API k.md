# BotTrainer — How to Run
---

## Prerequisites
- Python 3.10+ with pip
- Node.js 18+ with npm
- Gemini API key (set in `BotTrainer/.env`)

---

## Step 1 — Setup Python Environment
```bash
cd BotTrainer
python -m venv .venv
# Windows:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Step 2 — Download Dataset (Module 3 — Data Loader)
```bash
python -m BotTrainer.modules.data_loader
```
Downloads CLINC150 from Kaggle, creates `data/intents.json` and `data/eval_dataset.json`.

## Step 3 — Verify Individual Modules (Optional)
```bash
# Module 1 — Utility functions (no API key needed)
python -m BotTrainer.modules.utils

# Module 2 — LLM client (tests Gemini connection)
python -m BotTrainer.modules.llm_client

# Module 4 — Intent classifier (classifies 3 sample messages)
python -m BotTrainer.modules.intent_classifier

# Module 5 — Entity extractor (extracts entities from 3 samples)
python -m BotTrainer.modules.entity_extractor

# Module 6 — Evaluator (runs evaluation on 10 samples, shows metrics)
python -m BotTrainer.modules.evaluator
```

## Step 4 — Run Full CLI Pipeline
```bash
python -m BotTrainer.pipeline
```
Interactive CLI that classifies intent + extracts entities from your text.

## Step 5 — Run Streamlit App (Legacy UI)
```bash
streamlit run run.py
```
Opens the full Streamlit dashboard at `http://localhost:8501`.

---

## Step 6 — Run FastAPI Backend + React Frontend (Production)

### Terminal 1 — Start Backend API
```bash
cd BotTrainer
.\.venv\Scripts\Activate.ps1
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

## Step 7 — Run Tests
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