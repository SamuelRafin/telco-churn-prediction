# Notes

## Gotchas hit along the way

**TotalCharges isn't numeric.** Reads as object because 11 rows contain `" "`
instead of a number. All 11 have tenure 0, so they're new customers who haven't
been billed. Filling with 0.0 rather than dropping or imputing a mean.

**SHAP direction.** First pass used the signed mean SHAP value to decide whether
a feature pushed toward churn or away from it, and every single feature came
back as "reduces churn". Obvious in hindsight: 73% of customers don't churn so
almost everything has a negative mean contribution. Switched to correlating the
feature's value against its SHAP value.

**Untuned, logistic regression wins.** Default-parameter run had the baseline at
0.8460 ROC-AUC vs XGBoost at 0.8409. Only after the randomised search did
XGBoost come out ahead, and only by 0.004. Keeping both runs in mind, the
practical read is that this dataset doesn't have much non-linear structure.

**sklearn 1.8 deprecated `penalty`.** Had it in the logistic search space, which
spammed FutureWarnings across the whole tuning run. Removed; L2 is the default
anyway.

**`SeniorCitizen` is already 0/1** in the raw data, so it goes in NUMERIC_COLUMNS
rather than getting one-hot encoded into two redundant columns.

**Streamlit binds 0.0.0.0 by default.** Noticed it advertising an external URL
on my home IP. Use `--server.address 127.0.0.1`.

## Threshold

Precision floor is set at 0.50 in `evaluate.py`. It's arbitrary but defensible:
below that, more than half your retention budget goes to people who were
staying. The right version of this is CLV vs offer cost, which I haven't done.

Re-run `python -m churn.evaluate --write-threshold` after retraining or the
threshold in config.yaml belongs to the old model.

## If I pick this up again

- Calibration. Haven't checked whether 0.88 means 88%.
- `TotalCharges / tenure` as an average-spend feature.
- The fiber optic churn rate (41.9%) is worth a proper look. Premium product,
  worst retention, feels like a pricing or quality story rather than a
  modelling one.
