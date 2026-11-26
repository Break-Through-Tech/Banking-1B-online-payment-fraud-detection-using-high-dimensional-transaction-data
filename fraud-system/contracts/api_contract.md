\# API contract



Owners: P5 (serving) and P6 (front end). Agreed Day 1.

Status: DRAFT. Front end and dashboards build against these shapes with mock JSON until the real API is ready.



\## Conventions

\- JSON over HTTP, base path `/api`.

\- All ids are the IEEE-CIS `TransactionID`.

\- Errors: `{"error": {"code": "string", "message": "string"}}` with proper HTTP status.

\- Auth: TODO (none for the demo, or a shared token).

\- Mock JSON lives in `contracts/sample\_data/` so P6 and P7 develop offline.



\## POST /api/score

Request:

```

{ "transaction": { "TransactionID": 2987000, "TransactionAmt": 340.0, "ProductCD": "W",

&#x20;   "card4": "visa", "card6": "debit", "DeviceType": "mobile" } }

```

Response:

```

{ "transaction\_id": 2987000, "score": 0.0347, "score\_band": 912,

&#x20; "decision": "review",

&#x20; "reasons": \[ {"feature": "g\_cluster\_size", "text": "shares a device with 14 cards", "shap": 0.42} ],

&#x20; "model\_version": "v3-2026-11-20" }

```

`decision` is one of "approve", "review", "block", from the chosen cutoff.



\## GET /api/alerts

Query: `?status=pending\&min\_band=850\&limit=50`

Response:

```

{ "alerts": \[ { "transaction\_id": 2987000, "amount": 340.0, "merchant": "BestBuy.com",

&#x20;     "card\_last4": "4417", "score\_band": 912, "status": "pending",

&#x20;     "reasons": \[ {"text": "about 6x your usual amount"} ] } ],

&#x20; "total": 73 }

```

`status` is one of "pending", "confirmed\_fraud", "confirmed\_legit".



\## POST /api/alerts/{transaction\_id}/respond

The cardholder confirmation (feedback loop).

Request:

```

{ "response": "fraud" }

```

Response:

```

{ "transaction\_id": 2987000, "new\_status": "confirmed\_fraud",

&#x20; "card\_status": "frozen", "label\_recorded": true }

```

`response` is "fraud" or "legit".



\## GET /api/metrics

Response:

```

{ "kpis": { "transactions\_today": 48210, "fraud\_caught": 612,

&#x20;   "fraud\_dollars\_prevented": 184000, "alerts\_per\_fraud": 9.2, "review\_queue": 73 },

&#x20; "fraud\_rate\_series": \[ {"t": 1, "rate": 0.031}, {"t": 2, "rate": 0.028} ],

&#x20; "score\_distribution": { "legit": \[], "fraud": \[], "cutoff\_band": 850 },

&#x20; "by\_segment": \[ {"segment": "credit", "fraud\_rate": 0.041},

&#x20;                 {"segment": "debit", "fraud\_rate": 0.023} ] }

```



\## TODO (confirm Day 1)

Exact field names and shapes; decision band thresholds; whether /metrics is polled or pushed over WebSocket for the live replay; pagination on /alerts.

