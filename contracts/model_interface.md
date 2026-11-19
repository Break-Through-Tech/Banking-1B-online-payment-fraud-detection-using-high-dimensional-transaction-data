\# Model interface contract



Owner: P4 (model). Agreed Day 1 by P4, P5.

Status: DRAFT.



\## Purpose

How serving (P5) calls the model without knowing its internals, and the pinned transform so training and serving compute features identically (no train/serve skew).



\## Artifacts P4 publishes (shared drive, versioned)

\- `model.pkl`: trained XGBoost model.

\- `transform.pkl`: raw transaction record to model feature vector. Serving calls this, never reimplements it.

\- `feature\_list.json`: ordered feature names the model expects.

\- `score\_mapping.json`: probability to 1..999 band, percentile mapping fit on validation and frozen.

\- `model\_version`: string, e.g. `v3-2026-11-20`.



\## Scoring function (conceptual)

```

score\_one(raw\_record: dict) -> ScoreResult

score\_batch(raw\_records: list\[dict]) -> list\[ScoreResult]

```

ScoreResult:

```

{

&#x20; "transaction\_id": 2987000,

&#x20; "score": 0.0347,

&#x20; "score\_band": 912,

&#x20; "reasons": \[

&#x20;   {"feature": "g\_cluster\_size", "value": 14, "direction": "up", "shap": 0.42},

&#x20;   {"feature": "tab\_amt\_vs\_card\_median", "value": 6.1, "direction": "up", "shap": 0.31}

&#x20; ],

&#x20; "model\_version": "v3-2026-11-20"

}

```



\## Rules

\- Serving loads the four artifacts and calls them. If the model is not ready, serving uses a dummy returning random scores in this exact shape, so P5 is never blocked.

\- `reasons` are the top N SHAP contributors (default 3, confirm N).

\- `score\_band` uses the frozen validation mapping; never recompute on live data.

\- Transform and mapping are fit on train/val only.



\## TODO (confirm Day 1)

N reasons; field names (`shap` vs `shap\_value`); direction as "up"/"down" or signed number; dummy model behavior.

