# Fraud Detection System (BTT AI Studio, Banking 1B)

A real-time card fraud detection system built on the IEEE-CIS dataset, sponsored by Mastercard through Break Through Tech. This repo is standalone for now and merges into the BTT team repo once the app is complete.

## What this is

A model that scores a card transaction for fraud, wrapped in a working system: real-time scoring, an explanation for each flag, a cardholder confirmation app whose answers retrain the model, and an analyst dashboard. The model is the assignment; the system around it is what makes it more than a notebook.

## Our "why" (read before interviews)

We justify every tool by the shape of the problem, not the size of the data. The dataset fits in memory, so we did not need distributed compute for volume. We chose streaming and orchestration deliberately, to build and reason about the production architecture:

- Fraud is a real-time event problem (decisions in milliseconds during authorization), so the scoring path is a stream.
- Production ML is recurring, scheduled, dependency-managed work, so retraining and drift checks are orchestrated.
- The data fits in memory, so the core runs in plain Python where every step is legible, and we can explain how each heavier tool would earn its place at real volume.

## Architecture

Two layers. A transparent core (the backbone, must work) and a production layer (the hireable tools, which wrap the core and never block it).

Core pipeline: data and time-based split, then features (tabular + graph), then model (XGBoost + SHAP) with honest evaluation, then FastAPI scoring, then product (apps + dashboards).

Production layer: Kafka and Spark Structured Streaming on the real-time scoring path; Airflow orchestrating feature rebuilds, retraining, and drift checks. Diagrams in docs/.

Rule: the core must run without the production layer. If streaming or orchestration slips, the demo still works.

## Tech stack and the why

| Tool | Where | Why (interview answer) |
|---|---|---|
| pandas / Polars | features, core data | Data fits in memory; keep every step legible |
| NetworkX | graph features | Fraud rings are shared-identifier clusters |
| XGBoost + SHAP | model, reasons | Strong tabular model; SHAP gives per-flag reasons |
| FastAPI + Docker | scoring service | Low-latency scoring endpoint we fully control |
| Kafka | event stream | Fraud is a real-time event stream |
| Spark Structured Streaming | stream processor | Score events on arrival; fraud is a real-time stream |
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

## The 3 contracts (agree and freeze on Day 1)

Everyone builds against these, so nobody waits on anyone until integration.

1. contracts/data_schema.md: columns and types of the train/val/test tables, and the split rule (owner P1).
2. contracts/model_interface.md: input row in, {score, band, reasons} out (owner P4).
3. contracts/api_contract.md: the JSON for /score, /alerts, /metrics (owners P5 and P6).

Commit a tiny sample dataset and hardcoded sample JSON on Day 1, so the front end and dashboards can mock everything immediately.

## Per-person plan (7 people, 7 days)

Each person owns one folder. Cross-person needs go through contracts/, never by editing someone else's folder. The only real handoffs are: P1's split (Day 2), features to model (Days 3 to 4), model to serving (Days 4 to 5), and API to front end and dashboards (Days 5 to 6). Everything else runs in parallel against the frozen contracts and mocks.

## Per-person plan (7 people, 7 days)

### Roles and primary tools

| Person | Owns (folder) | Primary tools |
| --- | --- | --- |
| **P1 — Data + split** | data/ | Python, pandas or Polars, Parquet |
| **P2 — Tabular features** | features/tabular/ | pandas, scikit-learn (encoders) |
| **P3 — Graph features** | features/graph/ | NetworkX, pandas |
| **P4 — Model + evaluation** | model/ | XGBoost, SHAP, scikit-learn (metrics), MLflow |
| **P5 — Serving + streaming** | serving/ | FastAPI, Docker, Kafka, Spark Structured Streaming, Postgres |
| **P6 — Front end** | frontend/ | React / Next.js |
| **P7 — Dashboards + orchestration** | dashboards/, airflow/ | React + Recharts, Airflow, WebSocket |

Only four handoffs cross people: P1's split (Day 2), features to model (Days 3 to 4), model to serving (Days 4 to 5), API to front end and dashboards (Days 5 to 6). Everything else runs in parallel against the frozen contracts and mocks.

### Day by day

| Person | Day 1 | Day 2 | Day 3 | Day 4 | Day 5 | Day 6 | Day 7 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **P1** | agree data_schema; commit sample.parquet | publish real split (Parquet) | leakage check; document columns | fix schema issues | integration support | integration support | buffer / demo support |
| **P2** | read schema; start on sample (pandas) | build tab_* on train (encodings, aggregates) | finish first pass; key-check with P1 | finalize; hand to P4 | fix eval issues | integration | polish |
| **P3** | design nodes/edges on sample | build identifier graph (NetworkX) | compute g_* (past-only neighbor stats) | finalize; hand to P4 | integration | integration | polish |
| **P4** | eval harness on mock; write model_interface | baseline on tab_* (XGBoost) | combine tab_* + g_*; model + SHAP | tune; publish artifacts; report metrics | hand model to P5 | with vs without graph compare | results into README |
| **P5** | FastAPI on dummy model; Docker | /score, /alerts, /respond; Postgres | Kafka local + replay script | Spark Streaming; load real model | integrate real model behind API | streaming end to end | polish; p50/p95 latency |
| **P6** | React scaffold; mock API (static JSON) | cardholder confirm screen (mock) | analyst view (mock) | polish | wire to real API | fixes | demo polish |
| **P7** | dashboard scaffold; Airflow skeleton | KPIs + charts (mock /metrics) | dashboards done; draft retrain DAG | DAG runs retrain + drift | wire real /metrics | live replay (WebSocket) | polish; demo |

### Shared milestones
| Day | Milestone |
| --- | --- |
| **1** | All seven agree and freeze the 3 contracts, then branch |
| **5** | Integration 1: batch path, real model and API |
| **6** | Integration 2: streaming path (Kafka to Spark to score) |
| **7** | Rehearse the demo, write results, buffer |

## Scope discipline

Must finish (the grade): core model, honest time-split evaluation, FastAPI scoring, one front end. Stretch, only if the core is done: cardholder feedback loop, streaming path, Airflow. Future work (slide, not build): federated learning, extra dashboards, Databricks. A working core with a partial production layer demos well; a half-built everything does not.

Note: Kafka, Spark, and Airflow may be new to the team. They are the heaviest tools here, so treat the production layer as a deliberate stretch and protect the core first.

## How to run

(Fill in as pieces land: how to start the API, run the pipeline, launch the front end, bring up Kafka and Airflow.)

## Branch workflow

Each person works on their own branch and opens a pull request into main. Edit only your own folder; contracts/ and this README change by agreement.

```
git checkout -b p3-graph
# work in your folder, commit, then:
git push -u origin p3-graph
```