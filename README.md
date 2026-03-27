# BotTrainer (AIIP) - LLM-Powered NLU System

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009485.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19.2%2B-61DAFB.svg)](https://react.dev/)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-API-4285F4.svg)](https://ai.google.dev/)

**A production-ready conversational AI system combining TF-IDF retrieval with Google Gemini for accurate intent classification and entity extraction.**

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Configuration](#configuration)
- [Testing](#testing)
- [Deployment](#deployment)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

**BotTrainer** is a full-stack, LLM-powered Natural Language Understanding (NLU) system designed for building intelligent chatbots and conversational AI applications. It leverages:

- **Google Gemini API** for state-of-the-art language understanding
- **TF-IDF-based retrieval** for efficient intent candidate ranking
- **Caching mechanisms** for cost optimization and faster responses
- **CLINC150 dataset** with professional intent classification examples
- **FastAPI backend** for scalable REST API serving
- **React + Vite frontend** for modern chat interface

The system orchestrates end-to-end NLU workflows through a unified pipeline that handles intent classification, entity extraction, confidence scoring, and performance evaluation.

---

## Key Features

✨ **Core Capabilities**
- 🎯 **Intent Classification**: TF-IDF retrieval + LLM-based ranking with confidence scores
- 🏷️ **Entity Extraction**: Structured entity identification from user inputs
- 📊 **Evaluation Framework**: Precision, recall, F1-score, confusion matrices, and error analysis
- ⚡ **Performance Optimized**: Intelligent caching (500-entry LRU), rate limiting, retry logic
- 🔄 **Batch Processing**: ThreadPoolExecutor-based parallel processing

🔧 **Engineering Excellence**
- 🐳 **Docker Ready**: Containerized deployment with health checks
- 🧪 **Comprehensive Testing**: pytest suite covering classifiers, evaluators, extractors, and utilities
- 📝 **Structured Logging**: Debug-friendly logging throughout the pipeline
- 🛡️ **Error Handling**: Graceful failures with fallback mechanisms
- 💾 **Lazy Loading**: Efficient resource management with singleton patterns

🖥️ **User Interfaces**
- 💬 **Interactive CLI**: Direct TUI for testing and iteration
- 🌐 **REST API**: FastAPI with CORS support for web/mobile integration
- 🎨 **Web UI**: React + Vite frontend for modern chat experience

---

## Architecture

### System Design

```
┌────────────────────────────────────────────────────────────────┐
│                        User Interface Layer                     │
├──────────────────────────┬──────────────────────────────────────┤
│    CLI (pipeline.py)     │   React Frontend (frontend/)         │
│                          │        + Vite + TailwindCSS          │
└──────────────┬───────────┴───────────────────┬──────────────────┘
               │                               │
               └───────────────────┬───────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────┐
│                    FastAPI Backend (api.py)                    │
│  /api/health | /api/predict | /api/intents | /api/history    │
└──────────────────────────────────┬──────────────────────────────┘
                                   │
┌──────────────────────────────────▼──────────────────────────────┐
│                    NLU Pipeline (pipeline.py)                   │
│              Core orchestration & prediction logic              │
└──────────┬──────────────────────┬──────────────────┬────────────┘
           │                      │                  │
     ┌─────▼──────┐       ┌──────▼───────┐    ┌────▼─────┐
     │   Intent   │       │    Entity    │    │ Evaluator│
     │Classifier  │       │  Extractor   │    │ & Judge  │
     └─────┬──────┘       └──────┬───────┘    └────┬─────┘
           │                     │                  │
     ┌─────▼──────┐       ┌──────▼───────┐         │
     │  TF-IDF    │       │  Gemini LLM  │────────┘
     │ Retrieval  │       │ API Client   │
     └─────┬──────┘       └──────┬───────┘
           │                     │
           └─────────┬───────────┘
                     │
        ┌────────────▼────────────┐
        │  Data Layer             │
        │  - intents.json         │
        │  - eval_dataset.json    │
        │  - CLINC150 data        │
        │  - LLM prompts          │
        └─────────────────────────┘
```

### Data Flow

```
User Input
    ↓
Intent Classifier (TF-IDF Retrieval)
    ↓
Top-K Intent Candidates
    ↓
Gemini LLM (Intent Selection)
    ↓
Selected Intent + Confidence
    ↓
Entity Extractor (Gemini LLM)
    ↓
Extracted Entities + Reasoning
    ↓
Evaluator & Judge (Scoring)
    ↓
Structured Output
    {
      intent, confidence, entities, reasoning,
      model_used, timestamp, latency
    }
    ↓
API Response / UI Display / Storage
```

### Component Responsibilities

| Component | Purpose | Key Methods |
|-----------|---------|-------------|
| **GeminiClient** | LLM abstraction layer | `generate()`, `generate_batch()`, with retries & rate limiting |
| **IntentClassifier** | Intent ranking | `classify()`, TF-IDF + LLM hybrid approach, caching |
| **EntityExtractor** | Entity identification | `extract()`, structured JSON output, caching |
| **Evaluator** | Performance metrics | `evaluate()`, confusion matrix, F1-score, error analysis |
| **LLMJudge** | Confidence scoring | `judge()`, reasoning extraction |
| **NLUPipeline** | Orchestration | `predict()`, coordinates all components |

---

## Project Structure

```
AIIP/
├── BotTrainer/                      # Main Python package
│   ├── __init__.py
│   ├── __main__.py
│   ├── _runner.py                   # CLI runner
│   ├── api.py                       # FastAPI backend (REST endpoints)
│   ├── bootstrap.py                 # Setup & initialization
│   ├── config.py                    # Centralized configuration
│   ├── pipeline.py                  # Core NLU pipeline & CLI
│   │
│   ├── modules/                     # Reusable ML/NLU components
│   │   ├── __init__.py
│   │   ├── llm_client.py           # Google Gemini API wrapper
│   │   ├── intent_classifier.py    # TF-IDF + LLM classifier
│   │   ├── entity_extractor.py     # LLM-based entity extraction
│   │   ├── evaluator.py            # Evaluation & metrics
│   │   ├── judge.py                # Confidence scoring & reasoning
│   │   ├── data_loader.py          # Kaggle CLINC150 data download
│   │   ├── store.py                # Prediction history storage
│   │   └── utils.py                # Shared utilities & logging
│   │
│   ├── data/                        # Datasets
│   │   ├── intents.json            # 150 intent definitions (CLINC150)
│   │   ├── eval_dataset.json       # Evaluation samples
│   │   └── raw/                    # Raw CLINC150 variants
│   │
│   └── prompts/                     # LLM prompt templates
│       ├── intent_prompt.txt
│       └── entity_prompt.txt
│
├── frontend/                        # React + Vite web UI
│   ├── package.json
│   ├── vite.config.js
│   ├── index.html
│   ├── src/
│   │   ├── main.jsx
│   │   ├── App.jsx
│   │   ├── App.css
│   │   ├── index.css
│   │   └── components/
│   │       ├── ChatArea.jsx        # Main chat interface
│   │       ├── Sidebar.jsx         # Intent/history sidebar
│   │       ├── MessageBubble.jsx   # Message display
│   │       └── EvalDashboard.jsx   # Performance metrics
│   └── public/                     # Static assets
│
├── tests/                          # pytest test suite
│   ├── __init__.py
│   ├── conftest.py                # pytest fixtures
│   ├── test_intent_classifier.py
│   ├── test_entity_extractor.py
│   ├── test_evaluator.py
│   ├── test_judge.py
│   ├── test_llm_client.py
│   └── test_utils.py
│
├── data/                           # Top-level data directory
│   ├── intents.json               # Processed intents
│   ├── eval_dataset.json          # Evaluation set
│   └── raw/                       # Raw data files
│
├── modules/                        # Symlinked/exported modules
├── prompts/                        # Symlinked/exported prompts
│
├── pyproject.toml                 # Project metadata & dependencies
├── requirements.txt               # Python dependencies
├── setup.py                       # Setup script
├── run.py                         # Main entry point dispatcher
├── Dockerfile                     # Container image definition
├── pipeline.py                    # Root-level pipeline wrapper
├── seed_dashboard.py              # Dashboard data seeding
└── README.md                      # This file
```

---

## Tech Stack

### Backend
- **Python 3.10+**: Core language
- **FastAPI 0.110+**: REST API framework
- **Uvicorn 0.29+**: ASGI server
- **Google Gemini SDK 1.0+**: LLM integration
- **Scikit-learn 1.6+**: TF-IDF vectorization & metrics
- **Pandas 2.2+**: Data manipulation
- **Numpy 2.0+**: Numerical operations

### Frontend
- **React 19.2+**: UI library
- **Vite 8.0+**: Build tool & dev server
- **JavaScript (ES2024)**: Frontend logic

### Data & ML
- **Kaggle API 1.7+**: CLINC150 dataset download
- **Matplotlib 3.9+**: Visualization
- **Seaborn 0.13+**: Statistical visualization
- **Plotly 5.24+**: Interactive charts
- **Joblib 1.4+**: Caching & serialization

### DevOps
- **Docker**: Containerization
- **pytest 8.0+**: Testing framework
- **Ruff 0.4+**: Linting & formatting

---

## Quick Start

### Prerequisites
- Python 3.10 or higher
- Node.js 18+ (for frontend development)
- Google Gemini API Key (obtain from [ai.google.dev](https://ai.google.dev))
- Kaggle API credentials (for CLINC150 dataset)

### 1. Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd AIIP

# Create Python virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend
npm install
cd ..
```

### 2. Configure API Keys

Create a `.env` file in the `BotTrainer/` directory:

```env
GEMINI_API_KEY=your_api_key_here
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

### 3. Bootstrap Data

```bash
python run.py --setup
```

### 4. Run the System

**Option A: Interactive CLI**
```bash
python run.py
```

**Option B: FastAPI Backend**
```bash
python run.py --api
# API available at http://localhost:8000
```

**Option C: Frontend Development**
```bash
cd frontend
npm run dev
# Frontend at http://localhost:5173
# Connects to API at http://localhost:8000
```

---

## Installation

### Detailed Setup Guide

```bash
# 1. Clone and navigate to project
git clone <repository-url>
cd AIIP

# 2. Create virtual environment
python -m venv venv

# 3. Activate environment (Windows)
.\venv\Scripts\activate
# or (macOS/Linux)
source venv/bin/activate

# 4. Upgrade pip
pip install --upgrade pip

# 5. Install dependencies
pip install -e .  # Installs using pyproject.toml

# 6. Install development dependencies
pip install -e ".[dev]"

# 7. Setup data
python run.py --setup

# 8. Verify installation
python -m pytest tests/ -v
```

---

## Usage

### CLI Mode

**Interactive Pipeline:**
```bash
python run.py
```

Example interaction:
```
BotTrainer NLU Pipeline (interactive mode)
============================================
Enter a user query (or 'quit' to exit):
> I want to cancel my flight

Processing...

Intent: flight_cancel
Confidence: 0.95
Entities: {
  "flight_number": "AA123",
  "date": "tomorrow"
}
Reasoning: User explicitly mentions cancellation and provides flight context.
Model: gemini-2.5-flash
Latency: 1.23s
```

### API Mode

**Start the API Server:**
```bash
python run.py --api
```

Server runs on `http://localhost:8000`

**Health Check:**
```bash
curl http://localhost:8000/api/health
```

**Make Predictions:**
```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I want to cancel my flight"}'
```

Response:
```json
{
  "intent": "flight_cancel",
  "confidence": 0.95,
  "entities": {
    "flight_number": "AA123",
    "date": "tomorrow"
  },
  "reasoning": "User explicitly mentions cancellation...",
  "model": "gemini-2.5-flash",
  "timestamp": "2026-03-27T10:30:45Z",
  "latency_ms": 1230
}
```

### Frontend UI

**Start Development Server:**
```bash
cd frontend
npm run dev
```

Access at `http://localhost:5173`

**Build for Production:**
```bash
npm run build
# Output: frontend/dist/
```

---

## API Documentation

### Endpoints Reference

#### `GET /api/health`
Health check endpoint.

**Response (200 OK):**
```json
{
  "status": "healthy",
  "timestamp": "2026-03-27T10:30:45Z"
}
```

#### `POST /api/predict`
Main prediction endpoint for intent & entity extraction.

**Request Body:**
```json
{
  "text": "I want to cancel my flight",
  "context": {}
}
```

**Response (200 OK):**
```json
{
  "intent": "flight_cancel",
  "confidence": 0.95,
  "entities": {
    "flight_number": "AA123",
    "date": "tomorrow"
  },
  "reasoning": "...",
  "model": "gemini-2.5-flash",
  "timestamp": "2026-03-27T10:30:45Z",
  "latency_ms": 1230
}
```

#### `GET /api/intents`
List all available intents.

**Response (200 OK):**
```json
{
  "intents": [
    "flight_cancel",
    "flight_book",
    "weather_query",
    ...
  ],
  "total": 150
}
```

#### `GET /api/history`
Retrieve prediction history.

**Query Parameters:**
- `limit`: Maximum results (default: 100)
- `offset`: Pagination offset (default: 0)

**Response (200 OK):**
```json
{
  "predictions": [
    {
      "text": "I want to cancel...",
      "intent": "flight_cancel",
      "confidence": 0.95,
      "timestamp": "2026-03-27T10:30:45Z"
    },
    ...
  ],
  "total": 250,
  "limit": 100,
  "offset": 0
}
```

---

## Configuration

### Environment Variables

Create `.env` file in `BotTrainer/` directory:

```env
# Google Gemini API
GEMINI_API_KEY=your_api_key_here

# Kaggle (for CLINC150 dataset)
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key

# Optional: Override defaults
PRIMARY_MODEL=gemini-2.5-flash
FAST_MODEL=gemini-2.5-flash-lite
CACHE_MAX_SIZE=500
TFIDF_MAX_FEATURES=15000
TFIDF_NGRAM_RANGE=1,3
N_FEW_SHOT=5
```

### Configuration Class

Modify `BotTrainer/config.py`:

```python
@dataclass(frozen=True)
class Config:
    gemini_api_key: str = getenv("GEMINI_API_KEY", "")
    primary_model: str = "gemini-2.5-flash"
    fast_model: str = "gemini-2.5-flash-lite"
    cache_max_size: int = 500
    tfidf_max_features: int = 15000
    tfidf_ngram_range: tuple[int, int] = (1, 3)
    n_few_shot: int = 5
```

---

## Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Test Suite

```bash
# Test intent classifier
pytest tests/test_intent_classifier.py -v

# Test entity extractor
pytest tests/test_entity_extractor.py -v

# Test evaluator
pytest tests/test_evaluator.py -v

# Test with coverage
pytest tests/ --cov=BotTrainer --cov-report=html
```

### Test Coverage

```
BotTrainer/modules/
├── intent_classifier.py      ✓ Unit tests
├── entity_extractor.py       ✓ Unit tests
├── evaluator.py              ✓ Unit tests
├── judge.py                  ✓ Unit tests
├── llm_client.py             ✓ Unit tests
└── utils.py                  ✓ Unit tests
```

---

## Deployment

### Docker Deployment

**Build Image:**
```bash
docker build -t bottrainer:latest .
```

**Run Container:**
```bash
docker run -p 8000:8000 \
  -e GEMINI_API_KEY=your_key_here \
  bottrainer:latest
```

**Docker Compose (Optional):**
```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      GEMINI_API_KEY: ${GEMINI_API_KEY}
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Health Check

```bash
curl http://localhost:8000/api/health
```

---

## Screenshots

### Chat Interface
The main chat UI for interacting with the NLU system:

![Chat interface](Chat%20interface.jpeg)

### Real Chat Conversation
Example of a real conversation with intent and entity extraction:

![Real Chat](Real%20Chat.jpeg)

### Evaluation Dashboard
Performance metrics and system evaluation dashboard:

![Evaluation Dashboard](Evaluation%20Dashboard.jpeg)

### Evaluation Per Intent Performance
Detailed performance metrics broken down by individual intents:

![Evaluation Per Intent Performance](Evaluation%20Per%20Intent%20Performance.jpeg)

---

## Contributing

We welcome contributions! Follow these steps:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** changes: `git commit -m 'Add amazing feature'`
4. **Push** to branch: `git push origin feature/amazing-feature`
5. **Create** a Pull Request

### Code Standards

- Follow PEP 8 conventions
- Use type hints for all functions
- Write docstrings for all classes/functions
- Maintain test coverage above 80%
- Run `ruff check` before committing

```bash
# Format code
ruff check --fix .

# Run linter
ruff check .
```

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## Support & Contact

- 📧 **Email**: support@example.com
- 🐛 **Issues**: [GitHub Issues](https://github.com/yourusername/AIIP/issues)
- 📖 **Documentation**: Full docs at `/docs`

---

## Acknowledgments

- **CLINC150 Dataset**: CLINC - A Cross-Domain Benchmark for Natural Language Understanding
- **Google Gemini**: Advanced LLM capabilities
- **Scikit-learn**: TF-IDF vectorization
- **FastAPI**: Modern async web framework
- **React & Vite**: Frontend excellence

---

**Last Updated**: March 2026 | Version 1.0.0

## Requirements

- Python 3.10+
- Node.js 18+ (for frontend)
- Gemini API key (`GEMINI_API_KEY` in `BotTrainer/.env`)
- Kaggle credentials (`~/.kaggle/kaggle.json`) if you need to regenerate datasets

## Setup

From repository root:

```bash
pip install -e .
python run.py --setup
```

Then set your Gemini key in `BotTrainer/.env`.

## Run

### 1. CLI

```bash
python run.py
```

### 2. FastAPI backend

```bash
python run.py --api
```

Or directly:

```bash
python -m uvicorn BotTrainer.api:app --host 0.0.0.0 --port 8000 --reload
```

### 3. React frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend default URL: `http://localhost:5173`

## API Endpoints

- `GET /api/health`
- `POST /api/predict`
- `GET /api/intents`
- `GET /api/history`

Example:

```bash
curl -X POST http://localhost:8000/api/predict \
	-H "Content-Type: application/json" \
	-d '{"message": "Transfer $200 to savings"}'
```

## Evaluation

Evaluation flow is in `BotTrainer/modules/evaluator.py` and can be used programmatically with:

- `Evaluator.run_evaluation(...)`
- `Evaluator.compute_metrics(...)`
- `Evaluator.generate_confusion_matrix(...)`
- `Evaluator.get_error_analysis(...)`

## Project Layout

```text
AIIP/
├── BotTrainer/
│   ├── api.py
│   ├── bootstrap.py
│   ├── config.py
│   ├── pipeline.py
│   ├── data/
│   ├── modules/
│   └── prompts/
├── frontend/
├── tests/
├── run.py
├── setup.py
├── requirements.txt
├── pyproject.toml
└── Dockerfile
```

## Docker

Container runs FastAPI:

```bash
docker build -t bottrainer .
docker run -p 8000:8000 bottrainer
```

Health check target: `GET /api/health`
