# Customer Segmentation and Churn Prediction

This project uses the IBM Telco Customer Churn dataset (7,043 customers) to answer two questions:

1. **Who are our customers?** Unsupervised k-means clustering groups customers by tenure, contract type, and spend.
2. **Who is likely to leave, and why?** Supervised classifiers predict churn, and feature importance explains the main drivers.

## Project layout

```
data/raw/telco_churn.csv   source dataset (unchanged)
src/churn/                 reusable modules (loading, features, clustering, modelling)
tests/                     pytest suite, one file per module
reports/                   generated metrics and figures
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

## Status

Work in progress. Sections below are filled in as each stage is completed.
