# BotTrainer

```
 ____        _  _____          _
| __ )  ___ | ||_   _| __ __ _(_)_ __   ___ _ __
|  _ \ / _ \| __|| || '__/ _` | | '_ \ / _ \ '__|
| |_) | (_) | |_ | || | | (_| | | | | |  __/ |
|____/ \___/ \__||_||_|  \__,_|_|_| |_|\___|_|

  LLM-Based NLU Model Trainer & Evaluator for Chatbots
```

---

## Overview

**BotTrainer** is a production-grade Natural Language Understanding (NLU) pipeline that leverages **Google Gemini 2.5 Flash** to classify intents and extract entities from chatbot user messages. It ships with the **CLINC150** dataset (**150 intents**, **15,000 training utterances**), a smart TF-IDF + LLM classification approach with built-in caching and partial-response recovery, full evaluation metrics, and a polished **Streamlit** web interface with real-time latency tracking.

### Key Features

- 🎯 **150-intent classification** — Semantic matching via TF-IDF retrieval + Gemini few-shot prompting
- 🔍 **Entity extraction** — Automatic extraction of locations, amounts, dates, products, and more
- ⚡ **Performance optimized** — In-memory LRU caching (500 entries), blended TF-IDF scoring, trigram features
- 🛡️ **Robust JSON recovery** — Truncated LLM responses are automatically salvaged via progressive fixups
- 📊 **Full evaluation suite** — Accuracy, Precision, Recall, F1, confusion matrix, and error analysis
- 🖥️ **4-page Streamlit app** — Live Demo, Dataset Explorer, Evaluation Dashboard, and Settings
- ⏱️ **Latency tracking** — Response time displayed per prediction with color-coded badges

---

## Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                        BotTrainer Pipeline                        │
│                                                                   │
│  User Message                                                     │
│       │                                                           │
│       ▼                                                           │
│  ┌──────────────┐  TF-IDF (1-3gram)  ┌──────────────────┐       │
│  │ Intent       │───blended score────▶│  Top-30 Intents  │       │
│  │ Classifier   │   (max+mean)        │  + 5 Few-Shot Ex │       │
│  └──────┬───────┘                     └────────┬─────────┘       │
│         │  ▲ cache hit?                        │                  │
│         │  └── return cached ──┐               │                  │
│         │                      │               │                  │
│         │         ┌────────────┘───────────────┘                  │
│         │         ▼                                               │
│         │  ┌─────────────┐                                        │
│         │  │ Gemini 2.5  │◄── intent_prompt.txt                   │
│         │  │   Flash     │    (1024 max tokens)                   │
│         │  └──────┬──────┘                                        │
│         │         │  JSON: {intent, confidence, reasoning}        │
│         │         ▼                                               │
│         │  ┌─────────────┐                                        │
│         └─▶│   Entity    │◄── entity_prompt.txt                   │
│            │  Extractor  │    (512 max tokens)                    │
│            └──────┬──────┘                                        │
│                   │  JSON: {entity_type: value, ...}              │
│                   ▼                                               │
│         ┌─────────────────┐                                       │
│         │ Pipeline Output │──▶ intent + entities + latency_ms     │
│         └─────────────────┘                                       │
│                                                                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐    │
│  │  Streamlit   │    │  Evaluator   │    │  Confusion       │    │
│  │  Web App     │    │  F1/Acc/P/R  │    │  Matrix + Errors │    │
│  └──────────────┘    └──────────────┘    └──────────────────┘    │
└───────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Component         | Technology                  | Version      |
|-------------------|-----------------------------|--------------|
| LLM Engine        | Google Gemini 2.5 Flash     | latest       |
| LLM Fallback      | Google Gemini 2.5 Flash Lite| latest       |
| LLM SDK           | `google-genai`              | >= 1.0.0     |
| Web Framework     | Streamlit                   | >= 1.42.0    |
| ML / NLP          | scikit-learn (TF-IDF)       | >= 1.6.0     |
| Data              | pandas, numpy               | >= 2.2 / 2.0 |
| Visualisation     | Plotly, Matplotlib, Seaborn | >= 5.24 / 3.9 / 0.13 |
| Dataset Source    | Kaggle (CLINC150)           | —            |
| Config            | python-dotenv               | >= 1.0.0     |
| Progress Bars     | tqdm                        | >= 4.66.0    |

---

## Prerequisites

- **Python 3.10+**
- **Kaggle API key** — download from [kaggle.com/settings](https://www.kaggle.com/settings) → API → Create New Token → place at `~/.kaggle/kaggle.json`
- **Gemini API key** — get from [Google AI Studio](https://aistudio.google.com/apikey)

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/G26karthik/BTIFIT.git
cd BTIFIT

# 2. Run the setup script (creates .venv, installs deps, downloads dataset)
python setup.py

# 3. Add your Gemini API key to .env
#    Open .env and replace "your_gemini_api_key_here" with your actual key

# 4. Launch the app
streamlit run app.py
```

> **Note:** `setup.py` automatically creates a `.venv` virtual environment, installs all dependencies, locates your `kaggle.json`, downloads CLINC150, and generates `intents.json` + `eval_dataset.json`.

---

## Usage

### CLI Mode

```bash
python pipeline.py
```

Starts an interactive loop where you type messages and receive intent + entity predictions:

```
💬 You: Book a Train to Jammu from Hyderabad
🤖 Analysing …
──────────────────────────────────────────────────
  Intent     : book_flight
  Confidence : 90.00%
  Entities   : {'product': 'Train', 'destination': 'Jammu', 'origin': 'Hyderabad'}
  Reasoning  : The user's goal is to book long-distance travel, closest to book_flight.
  Model      : gemini-2.5-flash
──────────────────────────────────────────────────
```

### Streamlit Web App

```bash
streamlit run app.py
```

| Page | Description |
|------|-------------|
| 🏠 **Live Demo** | Type messages, get real-time intent + entity analysis with latency tracking |
| 📊 **Dataset Explorer** | Browse all 150 intents, view example distributions, search by intent |
| 🧪 **Evaluation** | Run batch evaluation (20–750 samples), view F1/accuracy/confusion matrix |
| ⚙️ **Settings** | API key status, model config, few-shot tuning (1–5 examples per intent) |

---

## Project Structure

```
BotTrainer/
│
├── data/
│   ├── raw/                        # Raw Kaggle download (CLINC150 CSVs)
│   ├── intents.json                # Processed dataset (150 intents, 15,000 examples)
│   └── eval_dataset.json           # Balanced evaluation set (750 samples, 5 per intent)
│
├── prompts/
│   ├── intent_prompt.txt           # Few-shot intent classification template
│   └── entity_prompt.txt           # Entity extraction template
│
├── modules/
│   ├── __init__.py                 # Package init
│   ├── data_loader.py              # Kaggle download + CLINC150 preprocessing
│   ├── llm_client.py               # Gemini client (retry, rate-limit, dual model)
│   ├── intent_classifier.py        # TF-IDF retrieval + LLM classification + caching
│   ├── entity_extractor.py         # Entity extraction via LLM + caching
│   ├── evaluator.py                # Metrics, confusion matrix, error analysis
│   └── utils.py                    # JSON parsing, partial recovery, validation
│
├── pipeline.py                     # Full NLU pipeline + interactive CLI
├── app.py                          # Streamlit 4-page web app (557 lines)
├── setup.py                        # One-command project setup (venv + deps + data)
├── requirements.txt                # 11 Python dependencies
├── .env.example                    # Environment variable template
├── .gitignore                      # Git ignore rules
├── 00.md                           # 100 ready-made test messages
└── README.md                       # This file
```

---

## Dataset — CLINC150

**CLINC150** is a widely used intent classification benchmark:

| Metric | Value |
|--------|-------|
| Total intents | **150** (10 domains: banking, travel, kitchen, auto, etc.) |
| Training utterances | **15,000** (100 per intent) |
| Evaluation samples | **750** (5 per intent, balanced) |
| Source | [Kaggle — CLINC150 Dataset](https://www.kaggle.com/datasets/hongtrung/clinc150-dataset) |

The dataset is automatically downloaded and preprocessed by `setup.py`. The `intents.json` file contains all 150 intents with their example utterances, while `eval_dataset.json` holds a balanced test split.

---

## Performance Optimizations

| Optimization | Description |
|---|---|
| **In-memory LRU cache** | 500-entry cache on both classifier & extractor — repeated messages return in ~0ms |
| **Blended TF-IDF scoring** | `0.7 × max + 0.3 × mean` per-intent score for more robust retrieval |
| **Trigram features** | `ngram_range=(1,3)` with 15K features and `sublinear_tf` for better text matching |
| **Top-30 candidate retrieval** | Broader candidate pool reduces misclassification as out_of_scope |
| **5 few-shot examples** | More examples per intent → better LLM context (configurable 1–5 in Settings) |
| **Partial JSON recovery** | Truncated LLM responses are auto-repaired: bracket closing → regex extraction |
| **1024 output tokens** | Increased from 512 to prevent response truncation on complex prompts |
| **Rate limiting** | Built-in 60 req/min throttle with exponential backoff on API errors |

---

## How Evaluation Works

1. **Balanced sampling** — Select *N* samples from `eval_dataset.json` (default: 100, max: 750)
2. **Batch classification** — Each sample is classified via TF-IDF retrieval → Gemini LLM with tqdm progress
3. **Metrics computed** — Overall Accuracy, Macro Precision, Macro Recall, Macro F1 (via scikit-learn)
4. **Per-intent breakdown** — Individual precision/recall/F1 per intent, sortable and color-coded
5. **Confusion matrix** — Top-30 intents by frequency rendered as a Seaborn heatmap
6. **Error analysis** — All misclassified samples listed with true vs predicted intent, filterable
7. **Export** — Download full results as JSON for offline analysis

---

## Module Reference

| Module | Class / Function | Description |
|--------|-----------------|-------------|
| `llm_client.py` | `GeminiClient` | Gemini API wrapper with retry (3 attempts), rate limiting, dual model support |
| `intent_classifier.py` | `IntentClassifier` | TF-IDF index + few-shot prompt + LLM + cache. Methods: `classify()`, `batch_classify()` |
| `entity_extractor.py` | `EntityExtractor` | Prompt-based entity extraction + cache. Method: `extract()` |
| `evaluator.py` | `Evaluator` | `run_evaluation()`, `compute_metrics()`, `generate_confusion_matrix()`, `get_error_analysis()` |
| `data_loader.py` | `DataLoader` | `download_dataset()`, `build_intents_json()`, `build_eval_dataset()`, `run_full_pipeline()` |
| `utils.py` | — | `safe_json_loads()` (with partial recovery), `validate_intent_result()`, `load_json_file()`, `save_json_file()` |
| `pipeline.py` | `NLUPipeline` | Orchestrates classifier + extractor. Method: `predict()` returns full result with `latency_ms` |
| `app.py` | — | Streamlit app with 4 pages, cached resource loading, custom CSS |

---

## Example Predictions

| Input Message | Intent | Confidence | Entities |
|---|---|---|---|
| "Book a Train to Jammu from Hyderabad" | `book_flight` | 90% | destination: Jammu, origin: Hyderabad, product: Train |
| "What's my account balance?" | `balance` | 95% | — |
| "Transfer $200 to savings" | `transfer` | 92% | amount: $200, destination: savings |
| "Tell me a funny joke" | `tell_joke` | 97% | — |
| "Set an alarm for 7 AM" | `alarm` | 96% | time: 7 AM |
| "How's the weather in Mumbai?" | `weather` | 94% | location: Mumbai |

See [`00.md`](00.md) for **100 ready-made test messages** covering all intent domains.

---

## Screenshots

> *Add screenshots of the Streamlit app here after first launch.*
>
> Recommended captures:
> 1. Live Demo page with a classified message + latency badge
> 2. Dataset Explorer bar chart
> 3. Evaluation metrics dashboard with per-intent breakdown
> 4. Confusion matrix heatmap

---

## License

This project is released under the **MIT License**.

```
MIT License

Copyright (c) 2026 BotTrainer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
