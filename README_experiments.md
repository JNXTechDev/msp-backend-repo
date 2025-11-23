# Experiments (MSP)

This directory contains a script to reproduce the main experiments used during development.

Scripts
- `scripts/experiments.py`: runs the following steps and writes outputs to `backend/experiments_out/`:
  - 5-fold stratified CV with in-fold class balancing (resampling of minority class)
  - Coefficient inspection for the enhanced model (top tokens + domain_flag coefficient)
  - Evaluate saved `model.pkl` + `vectorizer.pkl` (if present) on `datasets/spam_assassin.csv`
  - Deduplicate `enron_spam_data.csv`, retrain, and report delta in metrics

How to run (PowerShell)
```
.\\venv\\Scripts\\Activate.ps1
.\venv\Scripts\python.exe scripts\experiments.py
```

Outputs
- `backend/experiments_out/kfold_metrics.json`
- `backend/experiments_out/coefficients.json`
- `backend/experiments_out/dedupe_result.json`
