# Supply Chain Attack Detector

A complete cybersecurity project that detects **typosquatting** and
**dependency confusion** attacks across npm and PyPI registries.

Combines five detection layers — name similarity, metadata heuristics, Python AST
analysis, JavaScript AST analysis, and a trained ML classifier — into a single
0–100 risk score. Ships with a live web dashboard, GitHub Actions CI gate,
4-channel alert engine, and a full evaluation pipeline against the OSSF
malicious-packages dataset.

**Built with Python 3.13 / Python 3.11+ · Pure Python ML algorithms**

---

## Key results

| Metric | Heuristic baseline | Logistic regression |
|---|---|---|
| Precision | 51.3% | **100%** |
| Recall | 100% | **100%** |
| F1 score | 0.678 | **1.000** |
| False positive rate | **95%** | **0%** |

**Top finding:** Publisher account age alone accounts for 58.6% of model weight —
far outweighing name similarity, which hand-tuned heuristics over-rely on.

---

## Project structure

```
supply-chain-detector/
│
├── db/
│   └── models.py                    # SQLAlchemy ORM — Package, ScanResult,
│                                    # DetectorSignal, Allowlist tables
│
├── monitor/
│   ├── registry_poller.py           # Polls npm (/-/all) and PyPI (RSS) for new packages
│   ├── package_fetcher.py           # Fetches full metadata via registry APIs → DB
│   └── continuous_monitor.py        # Full pipeline loop: poll → fetch → score → alert
│
├── detectors/
│   ├── name_similarity.py           # RapidFuzz · Levenshtein · Soundex · homoglyph · separator swap
│   ├── metadata_heuristics.py       # 7 signals: publisher age, version jump, no README…
│   ├── static_analysis.py           # Regex scanner: shell exec, env access, obfuscation…
│   ├── ast_analyzer.py              # Python AST visitor with import alias + taint tracking
│   └── js_ast_analyzer.py           # JS postinstall hook + package.json scripts scanner
│
├── scoring/
│   └── risk_scorer.py               # Weighted combiner → 0–100 score + LOW/MEDIUM/HIGH
│
├── integrations/
│   ├── ci_gate.py                   # CLI tool for GitHub Actions (exits 3 on HIGH risk)
│   ├── alert_engine.py              # Slack · email · webhook · console alert channels
│   └── rate_limiter.py              # Token bucket per-IP + global limit + TTL scan cache
│
├── dashboard/
│   └── main.py                      # FastAPI REST API + built-in HTML dashboard UI
│
├── ml/
│   ├── feature_extractor.py         # 17-feature numeric vector (name · metadata · AST)
│   ├── classifier.py                # Logistic regression · StandardScaler · 5-fold CV
│   ├── ensemble.py                  # Soft voting + stacking meta-classifier
│   ├── saved_model.json             # Serialized LogReg weights (generated)
│   └── saved_ensemble.json          # Serialized stacking ensemble (generated)
│
├── evaluation/
│   ├── ossf_evaluator.py            # Precision · recall · F1 vs OSSF labelled dataset
│   ├── compare_detectors.py         # Ablation study — 4 detector configurations
│   └── ossf_pipeline.py             # Auto-download OSSF repo + parse + retrain model
│
├── tests/
│   ├── test_detectors.py            # Name similarity · metadata · static (20 tests)
│   ├── test_alert_engine.py         # Channel dispatch logic (12 tests)
│   ├── test_ast_analyzer.py         # Python AST patterns (13 tests)
│   └── test_ml.py                   # Feature extractor · LR · CV · pipeline (40 tests)
│
├── .env.example                     # Environment variable template
├── requirements.txt                 # pip dependencies
└── README.md
```

---

## Setup

```bash
# 1. Create and activate a virtual environment
python3.13 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy and configure environment variables
cp .env.example .env

# 4. Initialise the database (creates supply_chain.db)
python -c "from db.models import init_db; init_db()"
```

## Usage

### Scan a package (CLI)

```bash
# Pretty output
python -m integrations.ci_gate check --name reqests --registry pypi

# JSON output (for scripting / CI pipelines)
python -m integrations.ci_gate check --name reqests --registry pypi --output json

# Fail on MEDIUM or higher (default: HIGH only)
python -m integrations.ci_gate check --name reqests --registry pypi --fail-on medium
```

### Web dashboard

```bash
uvicorn dashboard.main:app --reload --port 8000
# Open http://localhost:8000
```

### Continuous monitoring

```bash
python -m monitor.continuous_monitor --interval 60 --limit 20 --medium --once
```

### Run tests

```bash
pytest tests/ -v
```

## Tech stack

- **Language:** Python 3.13 / 3.11+
- **Database:** SQLite (via SQLAlchemy 2.0)
- **API:** FastAPI + Uvicorn
- **Similarity Engine:** RapidFuzz
- **Testing:** pytest
