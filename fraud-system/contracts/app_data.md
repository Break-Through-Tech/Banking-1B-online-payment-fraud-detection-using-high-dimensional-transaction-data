# App data contract

Owners: P5 (scoring job, owns the database) and P6 (Streamlit pages). P7 reads it for metrics and retraining. Agreed Day 1.

Status: DRAFT. Pages build against `contracts/sample_data/sample_app.db` until the real scoring job runs.

## Purpose

Replaces the old API contract. There is no HTTP API: the scoring job writes to a SQLite database, and the Streamlit app and the Airflow retrain DAG read from it. These tables are the shared interface.

## Location

`data/app.db` (SQLite, not committed). A small seeded copy lives at `contracts/sample_data/sample_app.db` so every page works offline from Day 1.

## Table: alerts

One row per scored transaction that crossed the review cutoff. Written by P5's scoring job.

| Column | Type | Notes |
|---|---|---|
| transaction_id | INTEGER PRIMARY KEY | IEEE-CIS `TransactionID` |
| amount | REAL | `TransactionAmt` |
| product_cd | TEXT | `ProductCD` |
| card_type | TEXT | `card6`, e.g. debit or credit |
| score | REAL | model probability |
| score_band | INTEGER | 1..999 |
| decision | TEXT | "review" or "block" |
| reasons_json | TEXT | JSON list of reasons from the ScoreResult |
| model_version | TEXT | which model scored it |
| status | TEXT | "pending", "confirmed_fraud", "confirmed_legit" |
| scored_at | TEXT | ISO timestamp of scoring |

## Table: labels

One row per cardholder response. Written by the Cardholder page; read by the Analyst page, the Metrics page, and the Airflow retrain DAG.

| Column | Type | Notes |
|---|---|---|
| transaction_id | INTEGER | references alerts |
| response | TEXT | "fraud" or "legit" |
| responded_at | TEXT | ISO timestamp |

When a label is written, the Cardholder page also updates `alerts.status` to match.

## Table: metrics

Written by P4's evaluation and by the Airflow drift check; read by the Metrics page.

| Column | Type | Notes |
|---|---|---|
| model_version | TEXT | |
| metric | TEXT | e.g. "pr_auc", "ks", "detection_at_budget", "psi_drift" |
| value | REAL | |
| computed_at | TEXT | ISO timestamp |

## Rules

- Only P5's scoring job writes `alerts` rows; only the Cardholder page writes `labels`.
- Pages open the database read-only except for the label write.
- Schema changes go through this file by agreement.

## TODO (confirm Day 1)

Review and block cutoffs; whether alerts include all scored rows or only flagged ones; drift metric choice.
