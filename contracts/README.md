\# Data schema contract



Owner: P1 (data lead). Agreed Day 1 by P1, P2, P3, P4.

Status: DRAFT — fill the TODOs and freeze before anyone builds features.



\## Purpose

The canonical tables every downstream task reads. Feature and model code depend on these names, types, and the split rule. Change this file only by agreement, because a change ripples to everyone.



\## Source

IEEE-CIS Fraud Detection (Vesta). Two raw tables joined on `TransactionID`:

\- transaction: TransactionID, isFraud, TransactionDT, TransactionAmt, ProductCD, card1..card6, addr1, addr2, dist1, dist2, P\_emaildomain, R\_emaildomain, C1..C14, D1..D15, M1..M9, V1..V339

\- identity: TransactionID, id\_01..id\_38, DeviceType, DeviceInfo



Join: left join transaction with identity on TransactionID. Identity is missing for many rows; keep nulls, missingness is signal.



\## Keys and special columns

\- `TransactionID` (int): primary key, one row per transaction.

\- `TransactionDT` (int): seconds offset from a fixed reference. The time axis. Not a real date.

\- `isFraud` (int 0/1): label. Present in train and validation, ignored for test at scoring time.



\## Split rule (time-based, no shuffle)

Sort by `TransactionDT` ascending, then cut by time:

\- train: earliest \~70 percent

\- validation: next \~15 percent (tuning, threshold, score mapping)

\- test: most recent \~15 percent (touched once, final number)



TODO (confirm Day 1): exact cut points. Earlier plan used an 80/10/10-style split with a held-out validation; confirm the percentages.



Leakage rule: anything learned from data (encodings, aggregates, the score mapping, graph neighbor stats) is fit on train, or train+val where stated, and applied unchanged to later splits. Never fit on validation or test.



\## Modeling table (gold) that P2, P3, P4 read

Every feature table keeps these so they join cleanly:

\- `TransactionID` (int, key)

\- `TransactionDT` (int, time)

\- `isFraud` (int, label)

\- plus feature columns, open set. Naming convention: `tab\_\*` for tabular (P2), `g\_\*` for graph (P3), to avoid collisions.



\## Files and location

\- Format: Parquet.

\- Location: shared drive or bucket, NOT git. Path: TODO.

\- Names: `train.parquet`, `val.parquet`, `test.parquet`, and `sample.parquet` (small, committed to `contracts/sample\_data/` for mocking).



\## Fixed vs TODO

Fixed: keys, time column, label, feature prefixes, leakage rule. TODO: split percentages, storage path, how feature tables merge (join on TransactionID).

