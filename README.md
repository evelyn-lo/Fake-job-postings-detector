# Fake Job Detector

This project trains a logistic-regression model on Kaggle’s fake job postings dataset, and provides a small web app where users can paste a job ad and instantly get a fraud score, an explanation, and basic model stats. 

## Repository layout

- `data/fake_job_postings.csv` — original dataset (no changes made).
- `src/train_model.py` — training script that builds the TF-IDF + categorical pipeline, evaluates it, and writes artifacts.
- `src/analyze_dataset.py` — exploratory analysis + phrase mining (outputs in `reports/`).
- `src/compare_models.py` — benchmark suite for alternative classifiers.
- `src/contextual_similarity.py` — uses sentence-transformer embeddings to find nearest neighbors
- `models/fake_job_classifier.joblib` — serialized scikit-learn pipeline used by the web app.
- `models/model_report.json` — metrics, training config, and top feature weights for transparency.
- `webapp/` — Flask site (`app.py` + templates/static assets) that provides the UI and reasoning.
- `requirements.txt` — minimal dependency lockfile for both training and inference.

## Quick start

```bash
pip3 install -r requirements.txt        # add --user if needed
python3 src/train_model.py              # regenerates the model + report
FLASK_APP=webapp.app flask run          # serve the website at http://127.0.0.1:5000
```

> The training command prints the evaluation summary and fully refreshes `models/` on each run, so you can tweak hyperparameters by editing `src/train_model.py` and re-running.

## Open-source code and modifications

- **External code imported:** none. The training pipeline, benchmarking utilities, EDA scripts, contextual embedding workflow, and Flask interface were written from scratch for this project.
- **Third-party libraries:** standard Python packages listed in `requirements.txt` (Flask, scikit-learn, pandas, sentence-transformers, etc.). These are used via their public APIs without copying source code.
- **New code implemented:** end-to-end training script (`src/train_model.py`), comparison suite (`src/compare_models.py`), dataset analysis generator (`src/analyze_dataset.py`), contextual similarity coder (`src/contextual_similarity.py`), shared feature definitions (`src/feature_defs.py`), Flask app + templates/static assets under `webapp/`, and the new reporting artifacts inside `reports/`.
- **Nontrivial changes:** repeated iterations added missing-text features, UI transparency elements (confidence meter, suspicious phrases, counterfactual guidance), model benchmarking, and documentation. No template or starter repo was used.

## Model details

- **Features:** Concatenated free text (`title`, `location`, `department`, `salary_range`, `company_profile`, `description`, `requirements`, `benefits`) vectorized with TF-IDF (1-2 grams, top 5k terms) plus numeric meta-features (`text_length`, `telecommuting`, `has_company_logo`, `has_questions`) mirroring the workflow in Sohil Sharma's tutorial.
- **Classifier:** `LogisticRegression(solver='liblinear', class_weight='balanced', max_iter=2000, C=1.0)`.
- **Split:** 80/20 stratified train/test (`random_state=42`) on 17,880 rows (4.84% fraudulent).
- **Held-out metrics:** accuracy 96.4%, precision 57.8%, recall 92.5%, F1 0.711, ROC AUC 0.987.
- **Transparency:** `models/model_report.json` stores the training config, metrics, and the top 15 coefficients supporting both the "fake" and "legit" classes. These weights are also surfaced in the UI.

## Model comparison

Benchmark the full preprocessing stack against multiple classifiers with:

```bash
python3 src/compare_models.py
```

# Fake Job Detector

This project trains a logistic-regression model on Kaggle’s fake job postings dataset, and provides a small web app where users can paste a job ad and instantly get a fraud score, an explanation, and basic model stats. 

## Repository layout

- `data/fake_job_postings.csv` &mdash; original dataset (no changes made).
- `src/train_model.py` &mdash; training script that builds the TF-IDF + categorical pipeline, evaluates it, and writes artifacts.
- `src/analyze_dataset.py` &mdash; exploratory analysis + phrase mining (outputs in `reports/`).
- `src/compare_models.py` &mdash; benchmark suite for alternative classifiers.
- `src/contextual_similarity.py` &mdash; uses sentence-transformer embeddings to find nearest neighbors
- `models/fake_job_classifier.joblib` &mdash; serialized scikit-learn pipeline used by the web app.
- `models/model_report.json` &mdash; metrics, training config, and top feature weights for transparency.
- `webapp/` &mdash; Flask site (`app.py` + templates/static assets) that provides the UI and reasoning.
- `requirements.txt` &mdash; minimal dependency lockfile for both training and inference.

## Quick start

```bash
pip3 install -r requirements.txt        # add --user if needed
python3 src/train_model.py              # regenerates the model + report
FLASK_APP=webapp.app flask run          # serve the website at http://127.0.0.1:5000
```

> The training command prints the evaluation summary and fully refreshes `models/` on each run, so you can tweak hyperparameters by editing `src/train_model.py` and re-running.

## Open-source code and modifications

- **External code imported:** none. The training pipeline, benchmarking utilities, EDA scripts, contextual embedding workflow, and Flask interface were written from scratch for this project.
- **Third-party libraries:** standard Python packages listed in `requirements.txt` (Flask, scikit-learn, pandas, sentence-transformers, etc.). These are used via their public APIs without copying source code.
- **New code implemented:** end-to-end training script (`src/train_model.py`), comparison suite (`src/compare_models.py`), dataset analysis generator (`src/analyze_dataset.py`), contextual similarity coder (`src/contextual_similarity.py`), shared feature definitions (`src/feature_defs.py`), Flask app + templates/static assets under `webapp/`, and the new reporting artifacts inside `reports/`.
- **Nontrivial changes:** repeated iterations added missing-text features, UI transparency elements (confidence meter, suspicious phrases, counterfactual guidance), model benchmarking, and documentation. No template or starter repo was used.

## Model details

- **Features:** Concatenated free text (`title`, `location`, `department`, `salary_range`, `company_profile`, `description`, `requirements`, `benefits`) vectorized with TF-IDF (1-2 grams, top 5k terms) plus numeric meta-features (`text_length`, `telecommuting`, `has_company_logo`, `has_questions`) mirroring the workflow in Sohil Sharma's tutorial.
- **Classifier:** `LogisticRegression(solver='liblinear', class_weight='balanced', max_iter=2000, C=1.0)`.
- **Split:** 80/20 stratified train/test (`random_state=42`) on 17,880 rows (4.84% fraudulent).
- **Held-out metrics:** accuracy 96.4%, precision 57.8%, recall 92.5%, F1 0.711, ROC AUC 0.987.
- **Transparency:** `models/model_report.json` stores the training config, metrics, and the top 15 coefficients supporting both the "fake" and "legit" classes. These weights are also surfaced in the UI.

## Model comparison

Benchmark the full preprocessing stack against multiple classifiers with:

```bash
python3 src/compare_models.py
```





