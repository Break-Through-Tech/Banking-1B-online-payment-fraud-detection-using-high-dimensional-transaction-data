# Model interface contract

Owner: P4 (model). Agreed Day 1 by P4, P5, P6.

Status: DRAFT.

## Purpose

How the scoring job (P5) and the Streamlit app (P6, P7) call the model without knowing its internals, and the pinned transform so training and scoring compute features identically (no train/serve skew).

## Artifacts P4 publishes

Saved to `model/artifacts/`, versioned by folder or filename:

- `model.pkl`: trained XGBoost model.
- `transform.pkl`: raw transaction record to model feature vector. Callers use this, never reimplement it.
- `feature_list.json`: ordered feature names the model expects.
- `score_mapping.json`: probability to 1..999 band, percentile mapping fit on validation and frozen.
- `model_version`: string, e.g. `v3-2026-11-20`.

## Scoring functions

P4 provides these in `model/score.py`. Everyone else imports them; nobody loads the raw model and scores on their own.

```python
score_one(raw_record: dict) -> dict      # one ScoreResult
score_batch(df: pd.DataFrame) -> pd.DataFrame  # one row per ScoreResult
```

ScoreResult:

```json
{
  "transaction_id": 2987000,
  "score": 0.0347,
  "score_band": 912,
  "decision": "review",
  "reasons": [
    {"feature": "g_cluster_size", "value": 14, "direction": "up", "shap": 0.42,
     "text": "shares a device with 14 cards"},
    {"feature": "tab_amt_vs_card_median", "value": 6.1, "direction": "up", "shap": 0.31,
     "text": "about 6x this card's usual amount"}
  ],
  "model_version": "v3-2026-11-20"
}
```

`decision` is one of "approve", "review", "block", from the chosen band cutoffs. `text` is the plain-language reason shown to cardholders and analysts.

## How the app uses it

The Streamlit app loads the artifacts once through `app/model_loader.py` (cached with `st.cache_resource`) and calls `score_one` for single lookups on the Analyst page. Bulk scoring of the test split is done ahead of time by P5's scoring job, which writes results to the app database (see `app_data.md`), so pages load fast.

When a new model version is published (by hand or by the Airflow retrain DAG), the app picks it up on restart or through a "Reload model" button that calls `load_model.clear()`.

## Rules

- If the model is not ready, `score.py` returns a dummy result with random scores in this exact shape, so P5, P6, and P7 are never blocked.
- `reasons` are the top N SHAP contributors (default 3, confirm N).
- `score_band` uses the frozen validation mapping; never recompute on live data.
- Transform and mapping are fit on train/val only.

## TODO (confirm Day 1)

N reasons; field names (`shap` vs `shap_value`); direction as "up"/"down" or signed number; decision band cutoffs; who writes the `text` templates for each feature.
