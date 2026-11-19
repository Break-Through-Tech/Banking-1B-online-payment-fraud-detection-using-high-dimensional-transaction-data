# Fraud Detection System (BTT AI Studio, Banking 1B)

A real-time card fraud detection system built on the IEEE-CIS dataset, sponsored by Mastercard through Break Through Tech. This repo is standalone for now and merges into the BTT team repo once the app is complete.

## What this is

A model that scores a card transaction for fraud, wrapped in a working system: real-time scoring, explanations for each flag, a cardholder confirmation app whose answers retrain the model, and an analyst dashboard. The model is the assignment; the system around it is what makes it more than a notebook.

## Our "why" (read before interviews)

We justify every tool by the shape of the problem, not the size of the data. The dataset fits in memory, so we did not need distributed compute for volume. We chose streaming and orchestration deliberately to build and reason about the production architecture:

- Fraud is a real-time event problem (decisions in milliseconds during authorization), so the scoring path is a stream.
- Production ML is recurring, scheduled, dependency-managed work, so retraining and drift checks are orchestrated.
- The data fits in memory, so the core runs in plain Python where every step is legible. We can explain how each heavier tool would earn its place at real volume.

## Architecture

Two layers. A transparent core (the backbone, must work) and a production layer (the hireable tools, wraps the core and never blocks it).

Core pipeline: data and time-based split -> features (tabular + graph) -> model (XGBoost + SHAP) + honest evaluation -> FastAPI scoring -> product (apps + dashboards).

Production layer: Kafka + Spark Structured Streaming on the real-time scoring path; Airflow orchestrating feature rebuilds, retraining, and drift checks. Diagrams in `docs/`.

Rule: the core must run without the production layer. If streaming or orchestration slips, the demo still works.

## Tech stack and the why

| Tool | Where | Why (interview answer) |
|---|---|---|
| pandas / Polars | features, core data | Data fits in memory; keep every step legible |
| NetworkX | graph features | Fraud rings are shared-identifier clusters |
| XGBoost + SHAP | model, reasons | Strong tabular model; SHAP gives per-flag reasons |
| FastAPI + Docker | scoring service | Low-latency scoring endpoint we fully control |
| Kafka | event stream | Fraud is a real-time event stream |
| Spark Structured Streaming | stream processor | Score events on arrival; builds on our CS 4371 Spark work |
| Airflow | orchestration | Retraining and drift are recurring scheduled jobs |
| React | front end | Full control over the display |
| Postgres | app state | Labels and card status |

Data note: this is the public IEEE-CIS (Vesta) dataset, a Mastercard-sponsored challenge, not Mastercard's data. No real geolocation, item categories, or calendar dates; time is a relative offset.

## Repo structure

```
contracts/      the 3 day-1 interfaces + tiny sample data (the decoupler)
data/           ingestion + time-based split            (P1)
features/
  tabular/      aggregation + velocity features          (P2)
  graph/        NetworkX ring/cluster features           (P3)
model/          training, SHAP, evaluation               (P4)
serving/
  api/          FastAPI scoring service                  (P5)
  streaming/    Kafka + Spark Structured Streaming        (P5)
frontend/       cardholder + analyst apps                (P6)
dashboards/     metrics dashboards                        (P7)
  airflow/      retrain + drift DAGs                      (P7)
docs/           diagrams and notes
```

## Team: 7 roles, no overlap

Each person owns one folder. Cross-person needs go through `contracts/`, never by editing someone else's folder.

- P1 Data lead: `data/`, owns the time-based split and the `data_schema` contract.
- P2 Tabular features: `features/tabular/`.
- P3 Graph features: `features/graph/` (NetworkX).
- P4 Model and evaluation: `model/`, owns the model-interface contract.
- P5 Serving and streaming: `serving/` (FastAPI, Kafka, Spark). Heaviest role; gets help at integration.
- P6 Front end: `frontend/`, co-owns the `api_contract` with P5.
- P7 Dashboards and orchestration: `dashboards/` and `airflow/`. Second heaviest; gets help at integration.

## The 3 contracts (agree on Day 1, before building)

Everyone builds against these, so nobody waits on anyone until integration.

1. Data schema: columns and types of the train/val/test tables (owner P1).
2. Model interface: input row format in, {score, top SHAP reasons} out (owner P4).
3. API contract: the JSON for `/score`, `/alerts`, `/metrics` (owners P5 and P6).

Commit a tiny sample dataset and hardcoded sample JSON on Day 1 so the front end and dashboards can mock everything immediately.

## 7-day plan

Hard dependencies are only: Day 1 contracts, P1's split handoff (others use the sample until then), and the two integration days. Everything else runs in parallel on mocks.

- Day 1 (all): repo setup, lock the 3 contracts, commit sample data. P1 starts the split. Everyone branches: `p1-data`, `p2-tabular`, ... and opens PRs into `main`.
- Day 2: P1 publishes the real split. P2/P3 build features against the schema. P4 scaffolds the model and eval harness on sample features. P5 stands up FastAPI on a dummy model and gets Kafka running locally. P6 builds UI on the mock API. P7 builds dashboards on mock metrics and drafts the Airflow skeleton.
- Day 3: P2/P3 finish a first feature pass. P4 trains the first real model. P5 scores via the dummy and gets a Kafka replay working. P6 UI functional on mocks. P7 dashboards functional on mocks; first retrain DAG drafted.
- Day 4: P4 combines tabular + graph features, produces real model + SHAP + honest evaluation numbers. P5 wires Spark Structured Streaming onto Kafka. P6 polishes. P7 runs the retrain DAG end to end on a schedule.
- Day 5 (integration 1, batch path): swap mocks for real. P5 loads P4's real model; P6 and P7 point to the real API. End-to-end on the batch path.
- Day 6 (integration 2, streaming path): Kafka -> Spark -> scoring -> dashboard live. Fix interface mismatches. Full dry run.
- Day 7: polish, rehearse the demo, write results into this README, buffer for slippage.

## Scope discipline

Must finish (the grade): core model, honest time-split evaluation, FastAPI scoring, one front end. Stretch, only if the core is done: cardholder feedback loop, streaming path, Airflow. Future work (slide, not build): federated learning, extra dashboards, Databricks. A working core with a partial production layer demos well; a half-built everything does not.

## How to run

(Fill in as pieces land: how to start the API, run the pipeline, launch the front end, bring up Kafka and Airflow.)