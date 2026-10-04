# StreamBox Churn Prediction Challenge

Welcome to the repo! This codebase was built to tackle the StreamBox Churn Prediction Kaggle competition. The goal is to predict customer churn probability within 30 days after a snapshot date, using historical customer profiles and event logs.

## 📁 Repository Structure

```text
├── docs/
│   ├── PROBLEM_STATEMENT.md     # Full rules, data descriptions, and goals
│   └── previous_run_logs.md     # Logs of the various feature pipeline phases
├── data/                        # NOT COMMITTED: Place competition CSVs here
├── outputs*/                    # NOT COMMITTED: Artifacts, predictions, and model logs
├── cache/                       # NOT COMMITTED: Feature cache files
├── streambox_v1_baseline.py     # Initial baseline (Customer profile + basic event features)
├── streambox_v2.py              # Advanced feature engineering engine (Decay, gaps, relativity)
├── streambox_v3.py              # Hyperparameter tuning, feature selection (top 75), and ensemble creation
├── streambox_v4.py              # Experimental generalization script (behavioral states, adversarial shift)
├── submission_notebook.py       # Self-contained Kaggle submission script 
├── convert_to_notebook.py       # Helper to convert submission_notebook.py into a Kaggle .ipynb
├── submission_notebook.ipynb    # The final Notebook ready for Kaggle upload
```

## 🚀 Current Status

*   **Metric:** ROC-AUC
*   **Public Leaderboard Score:** `0.8653` (Rank #11 as of Submission #1)
*   **Best Single Model:** CatBoost (Top 75 features)
*   **Final Ensemble Pipeline (Submission #1):**
    *   **CatBoost:** 70%
    *   **LightGBM:** 25%
    *   **XGBoost:** 5%
    *   Validation uses a strict temporal split (Train on Jan-Mar, test on Apr) to avoid target leakage.

## 🧠 Approach

1.  **Strict Leakage Prevention:** We explicitly cut off any events occurring after a customer's `snapshot_date`. If they churned in reality, we don't accidentally peek at what they did on the day they churned unless it was before the snapshot cutoff.
2.  **Extensive Feature Engineering (`streambox_v2.py`):** 211 features generated representing:
    *   *Event windows:* 7d, 14d, 30d, 60d, 90d, 120d (Sessions, tickets, failures).
    *   *Recency:* Days since last event.
    *   *Gaps:* Time between sessions (median, max, mean).
    *   *Activity Decay:* Recent activity vs older activity (e.g., sessions in last 30d vs 31-60d).
3.  **Generalization Focus (`streambox_v4.py`):** We noticed a drop from our local April validation (`~0.871`) to the May/June Public Leaderboard (`~0.865`). V4 investigates the train/test distribution shift using adversarial validation and creates new compact behavioral and event sequence states. 

## 🛠️ How to use

1.  **Data:** Make sure you have the competition data (`train.csv`, `test.csv`, `events.csv`, `sample_submission.csv`) inside the `data/` directory.
2.  **Run V3/V4 Pipeline:** Run `python streambox_v3.py` or `python streambox_v4.py` to recreate local validation results.
3.  **Generate Submission:** Run `python submission_notebook.py` to train on all the data and produce the final Kaggle predictions (`outputs_submission/submission.csv`).
4.  **Notebook creation:** Run `python convert_to_notebook.py` to create the `.ipynb` file that you actually upload to Kaggle.
