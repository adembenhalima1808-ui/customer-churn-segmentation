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

Run the full pipeline (segmentation + churn models + figures) and write `reports/metrics.json`:

```bash
PYTHONPATH=src python -m churn.report
```

## Results

**Segmentation.** k=4 was chosen by silhouette score (0.36) over the k=2..8 sweep in
`reports/figures/elbow_silhouette.png`. The four segments:

| Segment | Share | Avg. tenure (months) | Avg. monthly charges |
|---|---|---|---|
| Legacy low-spend | 15% | 46 | $26 |
| Growing high-value | 30% | 26 | $82 |
| New / at-risk | 30% | 8 | $46 |
| Loyal premium | 25% | 62 | $91 |

**Churn prediction.** Logistic regression and a random forest both clear the
majority-class baseline by a wide margin on recall and F1:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Baseline (majority class) | 0.73 | 0.00 | 0.00 | 0.00 | 0.50 |
| Logistic regression | 0.74 | 0.51 | 0.78 | 0.62 | 0.85 |
| Random forest | 0.76 | 0.54 | 0.77 | 0.63 | 0.84 |

Top churn drivers by random-forest feature importance: contract length, tenure,
monthly charges, total charges, and the ratio of monthly to total spend — all
consistent with the usual "new, high-bill, short-contract" churn profile.

## Testing

61 tests across data loading, feature engineering, segmentation, models, figures,
and the end-to-end report, with 99% line coverage on `src/churn`:

```bash
PYTHONPATH=src python -m pytest --cov=churn --cov-report=term-missing
```
