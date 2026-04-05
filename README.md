# Credit Risk Model

End-to-end machine learning project for **credit scoring** based on a client's credit history.

The project covers the full ML workflow: data loading, aggregation of event-level credit history, feature engineering, model training, evaluation, and saving the final inference pipeline.

## Project Goal

The objective is to predict the binary target `flag` for each client using historical credit-related events.

This repository demonstrates:
- working with large tabular data split into parquet parts,
- building custom aggregation logic at the `id` level,
- creating behavioral features from credit history,
- comparing baseline and boosted models,
- packaging the final solution into a reusable serialized pipeline.

## Business Task

In a credit scoring setting, raw data often comes in a **long event format**: one client can have multiple credit records over time.  
To train a classification model, this history must be transformed into one row per client with informative aggregated features.

This project solves that problem by:
1. aggregating credit history per client,
2. engineering risk-related features,
3. training classification models,
4. selecting the best approach by ROC-AUC,
5. saving the trained pipeline for future inference.

## Solution Overview

### 1. Data processing
The raw training data is loaded from multiple parquet files.  
The target is loaded separately and merged after feature aggregation.

### 2. Credit history aggregation
The core part of the project is a custom aggregation module that converts credit history into client-level features.

Implemented in:
- `src/aggregator.py`

The aggregator computes features related to:
- number of credit events,
- delinquency history,
- utilization dynamics,
- outstanding debt behavior,
- payment-related patterns,
- categorical credit attributes.

### 3. Feature engineering
Additional behavioral features are created on top of aggregated data.

Implemented in:
- `src/feature_engineering.py`

Examples of generated features:
- ratio of serious to mild overdue events,
- fraction of serious overdue cases,
- change in utilization over time,
- trends in debt and credit limit.

### 4. Modeling
Two models were compared:
- `LogisticRegression` — interpretable baseline,
- `LightGBM` — main boosted model.

Training and pipeline logic are implemented in:
- `src/pipeline.py`

### 5. Final pipeline
The final trained pipeline includes:
- aggregation,
- feature engineering,
- preprocessing,
- trained model.

The serialized pipeline is stored in:
- `models/credit_pipeline.pkl`

## Results

### Baseline models
- **LogisticRegression**: ROC-AUC ≈ **0.7157**
- **LightGBM**: ROC-AUC ≈ **0.7390**

### After tuning
- **LogisticRegression**: ROC-AUC ≈ **0.7226**
- **LightGBM**: ROC-AUC ≈ **0.7361**

### Final best result
- **LightGBM validation ROC-AUC: 0.7547**

## Tech Stack

- Python
- pandas
- numpy
- scikit-learn
- LightGBM
- scipy
- matplotlib
- pyarrow
- dill / pickle

## Project Structure

```text
credit-risk-model/
│
├── data/                      # Prediction outputs for the test dataset
├── models/                    # Saved trained artifacts
│   └── credit_pipeline.pkl
├── notebooks/                 # Jupyter notebooks with experiments and analysis
├── src/                       # Source code
│   ├── aggregator.py          # Credit history aggregation logic
│   ├── feature_engineering.py # Feature engineering
│   └── pipeline.py            # Training and inference pipeline
├── .gitignore
├── README.md
└── requirements.txt
```

## Notes About Repository Contents

- The `data/` directory contains prediction results for the test dataset.
- The `models/` directory contains the trained serialized pipeline.
- The model artifact is around 20 MB, which is acceptable for a regular GitHub repository.
- The exploratory work and experiments are stored in `notebooks/`.

## How to Run

### 1. Clone the repository

```bash
git clone <your_repo_url>
cd credit-risk-model
```

### 2. Create environment and install dependencies

```bash
pip install -r requirements.txt
```

### 3. Open notebooks

```bash
jupyter notebook
```

Then open the notebook from the `notebooks/` directory and run the cells in order.

## Model Usage

Example of loading the trained pipeline:

```python
import dill

with open("models/credit_pipeline.pkl", "rb") as f:
    pipeline = dill.load(f)
```

Example of prediction:

```python
pred = pipeline.predict(raw_df)
```

Expected output:
- client identifier
- predicted probability / score for the positive class

## What This Project Shows

This repository is a solid ML portfolio project because it demonstrates:
- data preparation from raw event-level records,
- custom feature aggregation logic,
- feature engineering for tabular data,
- work with class imbalance,
- model comparison and tuning,
- saving the final model for reuse.

## Possible Improvements

Further improvements that can make the project stronger:
- add a separate `predict.py` script for inference without notebooks,
- move configuration values to a dedicated config file,
- add experiment tracking,
- add unit tests for aggregation and feature engineering modules,
- add a small sample input/output example for faster onboarding.

## Conclusion

This project is not just a notebook with model training.  
It is a compact **production-style ML prototype** for a credit scoring task with:
- custom preprocessing logic,
- clear modular source code,
- measurable model quality,
- reusable trained pipeline.

For a **Junior / Junior+ / Middle ML** portfolio, this is a strong project because it shows both modeling skills and understanding of the full ML workflow.
