# IEEE-CIS Fraud EDA: Team Reference Notes

A companion to `BTT_Mastercard_EDA_Combined.ipynb`. Use it to find a piece of code in the notebook, understand what it does, and see what it found, without reading all 120+ cells.

**How to find code:** every analysis cell in the notebook starts with a comment such as `# DO (B4-H2): ...`. This guide quotes those comments, so you can search the notebook (Ctrl+F) for them. `B3-H2` means Branch 3, Hypothesis 2.

**All numbers** come from `train_df` (the first 80% of the training data by time) unless stated otherwise. Validation data (`val_df`) was set aside before any analysis and was never used to look for patterns.

---

## Contents

1. [The one-paragraph summary](#1-the-one-paragraph-summary)
2. [Setup: loading, memory, and the time split](#2-setup-loading-memory-and-the-time-split)
3. [Engineered features: what we built and why](#3-engineered-features-what-we-built-and-why)
4. [Helper functions](#4-helper-functions)
5. [Results, branch by branch](#5-results-branch-by-branch)
6. [Decisions carried into modeling](#6-decisions-carried-into-modeling)
7. [Known weaknesses and open questions](#7-known-weaknesses-and-open-questions)
8. [Glossary of the statistics used](#8-glossary-of-the-statistics-used)

---

## 1. The one-paragraph summary

Fraud in this data depends mostly on **which account is transacting, which product it is, and which data is present or missing**. Fraud clusters heavily by account: 89% of fraud in repeat accounts sits in accounts that are at least half fraud. But repeat accounts hold only about half of all fraud, and transactions with a missing address, just 11% of the data, hold another 37%. Product `C` has the highest fraud rate (11.3%) but product `W`, with the lowest rate (2.1%), holds the most fraud cases (43%). Many signals that look strong overall turn out to be **product mix**, so every finding was checked within product type. The signals that survive that check are hour of day, card type (credit vs debit), device type (mobile vs desktop), and transaction amount.

---

## 2. Setup: loading, memory, and the time split

### Loading and joining (cell after "Loading, Joining, and Splitting")

The data comes in two tables joined on `TransactionID`:

- **Transaction table:** every transaction (amount, product, card, email, the `C`, `D`, `M`, `V` columns).
- **Identity table:** device and browser details, but only for about a quarter of transactions.

```python
train = train_transaction.merge(train_identity, on="TransactionID", how="left")
test_identity.columns = test_identity.columns.str.replace("-", "_")
```

`how="left"` keeps every transaction, even ones without identity data (their identity columns are just empty). The second line fixes a naming mismatch: the test identity file uses `id-01` while training uses `id_01`.

### Memory downcast (from Eric's EDA)

```python
def downcast(df, skip=("TransactionAmt",)):
    floats = [c for c in df.select_dtypes("float64").columns if c not in skip]
    df[floats] = df[floats].astype("float32")
```

Stores decimal columns with half the memory. `TransactionAmt` is skipped so its exact cents are preserved for any future amount features.

### The time split

```python
train = train.sort_values("TransactionDT").reset_index(drop=True)
cutoff = int(len(train) * 0.8)
train_df = train.iloc[:cutoff].copy()
val_df = train.iloc[cutoff:].copy()
```

Sorts all transactions by time and takes the last 20% as validation. This mimics reality: the model learns from the past and is judged on the future. A random split would let the model learn from transactions that happen *after* the ones it's tested on.

| Set | Days | Transactions | Fraud rate |
|---|---|---|---|
| `train_df` | 1 to 141 | 472,432 | 3.51% |
| `val_df` | 141 to 182 | 118,108 | 3.44% |
| Kaggle test (no labels) | 213 to 395 | 506,691 | unknown |

### Time-gap check (from Eric's EDA)

Search: `# (from Eric's EDA) Where does each period fall in time?`

Prints the day range of each set. **The test set starts 30 days after the labeled data ends**, but our validation set starts the very next day after training. So validation scores may be somewhat better than what the model will get on test. During modeling, consider leaving a gap of a few weeks between the training and validation windows.

---

## 3. Engineered features: what we built and why

This section covers every new column or object the notebook creates, in the order it appears. For each one: where it's made, what it does, and whether it's meant to be a model feature or was only used for analysis.

> **Rule for all engineered features:** anything computed from data (groupings, medians, clusters, lists of columns to keep) must be computed on `train_df` only, then applied unchanged to `val_df` and test. Otherwise information from the future leaks into training.

### 3.1 Time features: `day`, `hour`, `dow`

**Where:** Branch 1, the cell starting `decision_log = []`.

```python
train_df["day"]  = (train_df["TransactionDT"] // 86400).astype(int)
train_df["hour"] = (train_df["TransactionDT"] // 3600 % 24).astype(int)
train_df["dow"]  = train_df["day"] % 7
```

`TransactionDT` is seconds from an unknown starting point, not a real date.

- **`day`:** divides by 86,400 (seconds in a day) to get the day number.
- **`hour`:** divides by 3,600 (seconds in an hour), then `% 24` wraps it into 0 to 23.
- **`dow`:** "day of week," the day number `% 7`, so 0 to 6. We don't know which number is Monday, since the start date is unknown.

**Model feature?** `hour` yes (strong signal, see B3-H2). `dow` low priority (little signal). `day` **no**: the test set covers days 213 to 395, which the model never saw, so raw day values can't generalize.

### 3.2 `has_identity`

**Where:** `# DO (H3): identity coverage and fraud rate with vs without identity`

```python
id_cols = [c for c in train_df.columns if c.startswith("id_")] + ["DeviceType", "DeviceInfo"]
train_df["has_identity"] = train_df[id_cols].notna().any(axis=1).astype(int)
```

Marks a transaction `1` if **any** identity or device column has a value, else `0`. In practice it matches exactly whether `id_01` is present, since every identity record includes `id_01`.

**Model feature?** Yes, but it overlaps heavily with `ProductCD` (product `W` never has identity data).

### 3.3 Missingness blocks: `null_counts` and `blocks`

**Where:** `# DO (H2): group columns by identical null count to find missingness blocks`

```python
null_counts = train_df[original_cols].isna().sum()
blocks = null_counts[null_counts > 0].groupby(null_counts[null_counts > 0]).apply(lambda s: list(s.index))
```

Counts the missing values in each column, then groups columns that have **exactly the same** number of missing values. The next two cells confirm that columns in the same group are missing on **exactly the same rows**, not just the same number of rows.

A **block** is a group of columns that are always missing together, most likely because they come from the same data source: if the source didn't report, all its columns are empty at once.

**Model feature?** Not directly. It's the foundation for 3.4, 3.5 and 3.6, and it means one "is missing" flag per block is enough.

### 3.4 `v_keep`: the reduced set of V columns (adapted from Eric's EDA)

**Where:** `# DO (H2 follow-up): near-duplicate V columns within each missingness block, train_df only`

The 339 `V` columns are anonymous features made by the data provider, and many are near-copies of each other. This cell removes the copies.

```python
v_sample = train_df[v_all].sample(100_000, random_state=0)
...
corr = sub.corr(method="spearman").abs().fillna(0)
for c in sub.nunique().sort_values(ascending=False).index:
    if all(corr.loc[c, k] <= V_THRESHOLD for k in kept):
        kept.append(c)
```

Step by step:

1. Take a 100,000-row sample of `train_df` (faster, and never touches validation).
2. Work **within each missingness block**, so every correlation is measured on the same rows.
3. Measure **Spearman** correlation between every pair of V columns in the block. Spearman catches any "both rise together" relationship, not just straight-line ones.
4. Sort the block's columns by how many distinct values they have (more distinct values = more detail).
5. Walk down that list and keep a column only if it is **not** more than 0.95 correlated with any column already kept.

**Result:** `v_keep` holds **210 of 339** V columns. 129 were dropped as near-duplicates, and all 14 V blocks had at least one.

**Model feature?** Yes, as the V set for logistic regression or neural nets. Optional for tree models (they cope with redundancy but train faster without it). The 0.95 threshold is a starting point: confirm on validation that the reduced set scores about as well as the full one.

### 3.5 `X_prof` and `profile_key`: missingness profiles

**Where:** `# One indicator per missingness pattern (1 = missing), from the H4 table`

```python
X_prof = pd.DataFrame(
    {new: train_df[old].isna().astype(np.int8) for old, new in pattern_rep.items()},
    index=train_df.index)
profile_key = X_prof.astype(str).agg("".join, axis=1)
```

- **`X_prof`:** one row per transaction and one column per missingness pattern (67 of them). Each value is `1` if the transaction is missing that pattern, `0` if present. One representative column stands in for each block (for example, `na_V169` represents all 19 columns of its block).
- **`profile_key`:** glues each row's 67 zeros and ones into one string like `1111...0000`. Transactions with the same string have exactly the same missing-data profile.

**Model feature?** Not as-is: there are 11,341 distinct profiles and new ones will appear in later data. It feeds 3.6 and 3.7.

### 3.6 `is_common` / `tail_label`: rare vs common profile

**Where:** the cells under "Rare vs common profiles", starting `top_profiles = profiles.index[: cross + 1]`.

```python
top_profiles = profiles.index[: cross + 1]      # the 118 most common profiles
is_common = profile_key.isin(top_profiles)
```

Labels each transaction as belonging to one of the **118 most common profiles** (which together cover 80% of transactions) or to the **rare tail** (everything else).

**Model feature?** Possibly, mainly useful within product `C` (see results). The list of common profiles must be defined on `train_df` only.

### 3.7 `na_cluster`: k-means clusters of missingness profiles

**Where:** the cell starting `km = MiniBatchKMeans(n_clusters=best_k, ...)`, after the cell that chooses `k`.

```python
km = MiniBatchKMeans(n_clusters=best_k, random_state=42, n_init=5, batch_size=4096).fit(X_prof)
train_df["na_cluster"] = km.labels_
```

Groups transactions into 3 clusters based only on their missingness profile (`X_prof`). `k = 3` was chosen with the silhouette score (see glossary). The clustering was **not** given product type or identity, yet it recovered groups that line up with both.

**Model feature?** A candidate. It would need the fitted `km` model applied to validation and test, not refitted.

### 3.8 PCA coordinates (`coords`)

**Where:** `# ... PCA ...` cell starting `pca = PCA(n_components=2, random_state=42)`.

Squeezes the 67 missingness indicators into 2 numbers per transaction so they can be plotted. The 2 components keep 67.2% of the variation. **Analysis only:** used for the scatter and hexbin charts, not a model feature.

### 3.9 `log_amt` and amount bands

**Where:** `# DO (Branch 3, H1): transaction amount by fraud status, overall and within product` and `# DO (B3-H1 follow-up): fraud rate by amount band, overall and within product`.

```python
amt["log_amt"] = np.log1p(amt["TransactionAmt"])
amt["band"] = pd.cut(amt["TransactionAmt"], [0, 10, 30, 100, 500, np.inf], ...)
```

- **`log_amt`:** `log(1 + amount)`. Amounts are very skewed (most are small, a few are huge), and the log spreads them out so charts are readable.
- **`band`:** buckets amounts into up to $10, $10 to 30, $30 to 100, $100 to 500, and over $500 (Eric's bands).

**Model feature?** Analysis only; they live in a temporary `amt` table. Tree models can use raw `TransactionAmt`. A suggested future feature is **amount divided by the product's median legitimate amount** (computed on train only).

### 3.10 `P_provider` and `R_provider`: grouped email domains

**Where:** `# DO (B3-H4a): fraud rate by purchaser email domain and provider`

```python
train_df["P_provider"] = train_df["P_emaildomain"].str.split(".").str[0]
```

Keeps only the part of the email domain before the first dot, so `yahoo.com`, `yahoo.fr` and `yahoo.co.uk` all become `yahoo`. This turns 59 purchaser domains into 44 providers, and absorbs new domain variants that may show up in later data.

**Model feature?** `P_provider` yes. `R_provider` was created but not analyzed further.

### 3.11 `email_match`

**Where:** `# DO (B3-H4b): does the recipient domain match the purchaser domain?`

```python
train_df["email_match"] = np.select(
    [R missing, P missing, P == R],
    ["no recipient", "no purchaser", "same domain"],
    default="different domain")
```

Puts each transaction into one of four categories based on the purchaser and recipient email domains. The conditions are checked in order, so a transaction with no recipient is labeled "no recipient" even if the purchaser is also missing.

**Model feature?** Low priority. It mostly repeats product type (see results).

### 3.12 `device_family`

**Where:** `# DO (B3-H5): device information among transactions with identity data`, then refined in the cell starting `def device_family(s):` (the second definition, which adds more rules).

```python
rules = [("windows", "windows"), ("ios", "ios"), ("sm-", "samsung"), ..., ("build/", "android_other")]
for key, fam in rules:
    if key in s:
        return fam
return "other"
```

`DeviceInfo` has 1,639 different raw strings like `SM-G930V Build/NRD90M`. This function lowercases each one and checks it against a list of keywords **in order**, returning the first family that matches (so `sm-` becomes `samsung`). Any leftover Android build string becomes `android_other`; anything unrecognized becomes `other`. Result: 17 families.

Order matters: `windows` is checked before `linux`, and the catch-all `build/` rule is last so named brands are matched first.

**Model feature?** Yes, instead of raw `DeviceInfo`. Only exists for transactions with identity data.

### 3.13 `D1n`, `card_addr`, `uid`: the approximate account key

**Where:** `# DO (B4-H1): build entity keys and test whether day - D1 is stable within card1 + addr1`

```python
train_df["D1n"] = train_df["day"] - train_df["D1"]
no_addr = train_df["addr1"].isna()
addr_key = train_df["addr1"].astype(str).where(~no_addr, "noaddr" + train_df["TransactionID"].astype(str))
train_df["card_addr"] = train_df["card1"].astype(str) + "_" + addr_key
train_df["uid"] = train_df["card_addr"] + "_" + train_df["D1n"].astype(str)
```

The data has no customer or account ID, so we approximate one.

- **`D1n`:** the card's estimated **start day**. We believe `D1` counts days since the card was first used, so `day − D1` should give the same number for every transaction on the same card, whatever day it happens. For example, a card that started on day 150 and is used on days 160, 175 and 190 has `D1` values of 10, 25 and 40, and `D1n` is 150 every time.
- **`card_addr`:** card identifier (`card1`) plus billing region (`addr1`). `card1` alone is coarse and shared by many cards, so the address narrows it down. If the address is missing, the transaction gets its own unique marker (`noaddr` + its `TransactionID`) instead.
- **`uid`:** `card_addr` plus `D1n`. This splits a card-and-address group into its separate cards, and is our best guess at an account.

The same keys are built for `val_df` in `# DO (B4-H3)`, using validation **features only, never its labels**.

> **Fixed issue:** an earlier version turned a missing `addr1` into the text `"nan"`, so all address-missing transactions for a `card1` were merged into one fake account (the largest was 8,088 transactions). Now each of the 53,761 address-missing transactions (11.4%) gets its own marker, as shown above. The same rule is applied to `val_df`.

**Model feature?** The raw `uid` should **not** be a feature (the model would memorize specific accounts). Instead, compute per-account summaries without labels: number of transactions, typical amount, how unusual the current transaction is for that account.

---

## 4. Helper functions

These small functions are reused across the notebook.

### `log(branch, finding, decision)`

Adds one row to `decision_log`, a running list of what each analysis found and what we decided because of it. The final cell prints the full log as a table.

### `category_rr(data, col, min_n=1000)`

**Where:** `# DO (B3-H3): fraud rate by card network and card type`

For each value of a column (for example each card network), compares the fraud rate of that value against **all other transactions**, and returns:

| Column | Meaning |
|---|---|
| `n`, `share` | how many transactions have this value, and what share of the data |
| `fraud_rate` | fraud rate for this value |
| `RR_vs_rest` | relative risk: this value's fraud rate divided by everyone else's |
| `ci_low`, `ci_high` | 95% confidence interval for the relative risk |
| `reliable` | `True` if at least 1,000 transactions, so the rate is stable |

Used for card type, card network, email domains, email match, and devices.

### `classify(r)` (inside the H4 statistical test)

Labels each missingness pattern as "meaningful", "significant but small", or "not significant" using the adjusted p-value and the confidence interval. See B2-H4 below.

### `entity_purity(data, key, min_size=2, seed=None)`

**Where:** `# DO (B4-H2): does fraud cluster within entities?`

For a given key (`card1`, `card_addr` or `uid`), groups transactions into entities, keeps entities with at least 2 transactions, and reports:

| Output | Meaning |
|---|---|
| `all_legit` | share of entities with no fraud at all |
| `all_fraud` | share of entities where every transaction is fraud |
| `mixed` | share of entities with both fraud and legitimate transactions |
| `fraud_in_mostly_fraud_entities` | share of all fraud that sits in entities at least 50% fraud |

With `seed` set, it **shuffles the fraud labels randomly first**. Comparing real vs shuffled shows how much clustering is real rather than chance, because the shuffle keeps every entity the same size and only scrambles which transactions are fraud.

---

## 5. Results, branch by branch

The notebook is organized as four branches, each with hypotheses (H1, H2, ...). Each hypothesis follows **Plan** (state it), **Do** (run the code), **Check** (what it showed), **Act** (the decision, logged).

### Branch 1: Target and time

**H1: Fraud is rare.** Confirmed. 3.51% of training transactions are fraud (16,599 of 472,432), 3.44% in validation. A model that always says "legitimate" would be 96.5% accurate while catching nothing, so we use **ROC-AUC and PR-AUC**, not accuracy.

**H2: The fraud rate changes over time.** Confirmed.

| Period | Fraud rate | Transactions |
|---|---|---|
| Days 1 to 25 | 2.39% | 116,291 |
| After day 25 | 3.88% | 356,141 |

Relative risk 0.62 (95% CI 0.59 to 0.64): the early period has about 40% less fraud. Days 17 to 25 are also the busiest (up to about 6,800 transactions a day), possibly holiday shopping diluting the fraud rate, though the real dates are unknown. The dip on day 141 is only because the train/validation split cut that day in half.

**H2 follow-up: product mix or real change?** Search: `# DO (B1 follow-up)`. Both changed, but mostly fraud *within* products:

| Product | Share, days 1-25 | Share, after day 25 | Fraud, days 1-25 | Fraud, after day 25 |
|---|---|---|---|---|
| W | 53.7% | 79.6% | 1.84% | 2.12% |
| C | 11.2% | 12.2% | 8.32% | 12.23% |
| H | 16.8% | 2.9% | 1.67% | 10.32% |
| R | 16.0% | 3.8% | 0.89% | 7.38% |
| S | 2.4% | 1.5% | 2.27% | 8.23% |

Early on, `H` and `R` were a third of transactions with almost no fraud; later they shrank to under 7% while their fraud rates rose six- to eightfold. Product-level rates are therefore time-dependent: the overall `H` (4.6%) and `R` (3.6%) rates blend two very different periods.

**Decision:** keep the time-based split; don't use raw `TransactionDT` or `day` as features; treat product-level rates as time-dependent.

### Branch 2: Data quality

**H1: Many columns are mostly empty.** Confirmed.

| Missing | Columns |
|---|---|
| 0 to 10% | 92 |
| 10 to 50% | 93 |
| 50 to 90% | 217 |
| 90 to 100% | 12 |

414 of 434 columns have missing values; only 20 are complete. Missing rates pile up at a few exact levels, an early hint of blocks.

**H2: Columns go missing in blocks.** Confirmed. The 414 incomplete columns fall into 67 groups by missing count. **24 groups have two or more columns** (these are the blocks, covering 371 columns), and all 24 were verified to be missing on identical rows. The other 43 columns each have their own unique pattern.

**H2 follow-up: redundant V columns.** 210 of 339 V columns kept, 129 dropped as near-duplicates (see 3.4). For comparison, Eric's global Pearson check on this sample finds 85 V columns with a near-duplicate somewhere; the within-block method finds more because it compares columns on their shared rows and catches non-linear matches.

**H3: Few transactions have identity data, and those differ.** Confirmed, but the effect is smaller than it first looks.

| | Transactions | Fraud rate |
|---|---|---|
| With identity | 120,464 (25.5%) | 7.55% |
| Without identity | 351,968 | 2.13% |

Overall relative risk is **3.54**. But product `W` never has identity data and has the lowest fraud rate, so this partly compares product `W` to everything else. Within product `C`, the only product with enough transactions on both sides, the relative risk is **2.04** (95% CI 1.84 to 2.27). **Use 2.04 as the identity effect.**

**Contribution view** (search: `# Identity: how much of the overall fraud rate`): a group's contribution is its share of transactions times its fraud rate, and contributions add up to the overall 3.51%. With identity: 25.5% × 7.55% = 1.93 points (55% of all fraud). Without: 74.5% × 2.13% = 1.59 points (45%).

**H4: Missingness itself predicts fraud.** Confirmed. For each of the 67 patterns, the notebook compares fraud when the columns are missing vs present, with a chi-square test and Benjamini-Hochberg correction (see glossary).

| Verdict | Patterns |
|---|---|
| Meaningful: presence raises fraud | 33 |
| Meaningful: missing raises fraud | 9 |
| Significant but small | 19 |
| Not significant | 6 |

Strongest examples:

| Pattern | Fraud when present | Fraud when missing | Ratio |
|---|---|---|---|
| `D7` (94% missing) | 14.97% | 2.74% | 5.47 |
| `D12` | 11.37% | 2.53% | 4.50 |
| `R_emaildomain` | 7.93% | 2.11% | 3.75 |
| Identity columns (about 75% missing) | about 7.6% | about 2.1% | about 3.5 |
| `addr1`/`addr2` | 2.5% | 11.4% | 0.22 |

The identity columns all show about 3.5, which is one signal (whether an identity record exists), not many. All 6 non-significant patterns have tiny missing groups (12 to 936 rows).

Per transaction: the median fraud transaction is missing **139** of 434 fields, compared with **212** for legitimate ones. Fraud transactions tend to be *more* complete.

**Decision:** keep missing values (tree models handle them natively), never drop a column just because it's mostly empty (`D7` is 94% empty and one of the strongest signals).

**H5: Transactions form a few missingness profiles.** Confirmed.

- **H5a:** 11,341 distinct profiles, but the top 5 cover 25% of transactions, the top 118 cover 80%. 5,959 profiles appear only once.
- **Rare vs common:** rare-tail transactions have 7.11% fraud vs 2.62% for common profiles. But the rare tail is 33% product `C`, compared with 7% of common profiles. Within products:

| Product | Rare tail fraud | Top-118 fraud | Relative risk |
|---|---|---|---|
| C | 14.58% | 7.24% | 2.01 |
| H | 4.48% | 4.73% | 0.95 |
| R | 3.21% | 3.91% | 0.82 |
| W | 2.32% | 2.04% | 1.14 |

So rarity matters mainly within product `C`.

- **H5b and H5c, clustering:** k-means with k = 3 (silhouette about 0.55) found:

| Cluster | Share | Contents | Fraud rate |
|---|---|---|---|
| 0 | 75.0% | No identity data, 98% product `W` | 2.1% |
| 1 | 14.5% | Identity data, products `H`, `R`, `S` | 4.4% |
| 2 | 10.6% | Identity data, product `C` | 12.0% |

Contributions (search: `# Cluster contribution`): 1.60, 0.64 and 1.27 points respectively.

The clustering never saw product or identity, yet recovered both. **Product type is the backbone of the data's structure.**

**Fraud rate vs fraud count by product:**

| Product | Transactions | Fraud rate | Share of all fraud |
|---|---|---|---|
| C | 56,410 | 11.33% | 38.5% |
| S | 8,127 | 6.19% | 3.0% |
| H | 29,691 | 4.64% | 8.3% |
| R | 32,203 | 3.63% | 7.0% |
| W | 346,001 | 2.07% | 43.1% |

A model must do well on both `C` (highest rate) and `W` (most fraud cases).

**Contribution view** (search: `# Product contribution`):

| Product | Share of transactions | Fraud rate | Contribution |
|---|---|---|---|
| W | 73.2% | 2.07% | 1.52 points |
| C | 11.9% | 11.33% | 1.35 points |
| H | 6.3% | 4.64% | 0.29 points |
| R | 6.8% | 3.63% | 0.25 points |
| S | 1.7% | 6.19% | 0.11 points |
| Total | 100% | | 3.51% |

`W` is almost three quarters of the data, so it pulls the overall rate toward its own 2.07%. The cell also plots share of transactions vs share of fraud per product.

### Branch 3: Feature signal

Every relationship here is checked **within product type**, because product type distorted several Branch 2 results.

**H1: Transaction amount.** Confirmed. The median fraud amount is higher than the median legitimate amount in **every** product:

| Product | Legit median | Fraud median | Ratio |
|---|---|---|---|
| H | $50.00 | $150.00 | 3.0× |
| R | $125.00 | $200.00 | 1.6× |
| W | $77.95 | $117.00 | 1.5× |
| S | $30.00 | $40.00 | 1.3× |
| C | $31.25 | $34.56 | 1.1× |

**H1 follow-up, amount bands (resolves Eric's small-payment finding).** Overall, Eric's pattern reproduces: payments up to $10 are 7.16% fraud vs 2.94% for $30 to 100. But within products, fraud **rises** with amount:

| Band | C | W | S |
|---|---|---|---|
| Up to $10 | 8.0% | 1.3% | 4.9% |
| $10 to 30 | 10.7% | 1.2% | 5.2% |
| $30 to 100 | 12.0% | 1.6% | 6.3% |
| $100 to 500 | 14.6% | 2.5% | 13.9% |
| Over $500 | (2 txns) | 4.4% | (34 txns) |

4,663 of the 5,953 payments up to $10 (78%) are product `C`, the riskiest product. **The small-payment effect is product mix, not card testing.**

**H2: Hour of day.** Confirmed, and the most consistent signal. Fraud peaks during hours 5 to 10, the lowest-volume hours, in **every** product:

| Product | Fraud, hours 5 to 10 | Fraud, other hours | Relative risk (95% CI) |
|---|---|---|---|
| W | 3.31% | 2.01% | 1.64 (1.50 to 1.80) |
| C | 20.59% | 10.45% | 1.97 (1.86 to 2.09) |
| H | 10.67% | 4.26% | 2.51 (2.17 to 2.90) |
| R | 7.44% | 3.48% | 2.13 (1.73 to 2.63) |
| S | 13.02% | 5.85% | 2.23 (1.69 to 2.93) |

Caveat: the 5 to 10 window was picked after looking at the chart, so these ratios are probably a bit optimistic. Day of week shows little signal (3.2% to 3.8% across days).

**How much fraud happens at night?** (search: `# How much fraud falls in the night window overall`): hours 5 to 10 are 4.9% of transactions at 7.92% fraud (vs 3.29% otherwise) and hold **11.0% of all fraud**. Strong but narrow: about 89% of fraud happens at ordinary hours.

**H3: Card attributes.** Partly confirmed.

- **Card type (`card6`):** credit 6.73% vs debit 2.40% overall. Holds within `W` (4.44% vs 1.61%, 2.76×) and `C` (16.51% vs 7.76%, 2.13×), which hold about 82% of fraud. Near 1× in `H`, `R`, `S`.
- **Card network (`card4`):** Visa and Mastercard are nearly identical (3.47% vs 3.49%). Discover's overall 2.14× comes mainly from product `W` (8.15% vs about 2% for Visa and Mastercard). American Express appears mostly in product `R`, where it's lower.

**H4: Email domains.** Partly confirmed; much of the overall pattern is product mix.

- **Overall:** Outlook (9.49%), Hotmail (5.16%) and Gmail (4.39%) look risky; Yahoo, AOL and MSN look safe.
- **Within product `W`:** provider barely matters (most between 1.3% and 3.4%, near W's 2.07%).
- **Within product `C`** (baseline 11.33%): Gmail 16.56%, Outlook 15.73%, iCloud 13.02% above; AOL 9.84%, Hotmail 7.59%, anonymous 6.51%, Yahoo 4.62% below. Similar in `H` and `R`.
- **Hotmail:** 20,477 of its 37,736 transactions are product `C`, which inflates its overall rate. Within `C` it's *below* average.
- **Internet-provider domains** (att.net, sbcglobal.net, verizon.net): very low overall (0.46% to 0.67%) **and** low within product `W` (0.5% to 0.8% vs 2.07%). This is a real signal, not product mix.
- **Email match:** "same domain" looks 4.2× riskier overall, but email structure is set by product: `W` never has a recipient, `C` almost always matches, `S` almost never has a purchaser. Matching raises fraud only in `H` (8.80% vs 3.18%) and `R` (4.27% vs 2.59%), and reverses in `C`.

**H5: Device information** (identity transactions only, so no product `W`). Partly confirmed.

- **Device type:** mobile 9.90% vs desktop 6.24%. Mobile is riskier in every product: `R` 3.41×, `S` 2.92×, `H` 1.32×, `C` 1.15×.
- **Device family:** Apple devices look safe overall (macOS 1.85%, iOS 6.13%) but never appear in product `C`, so that's partly product mix. Within `C`, every Android family is above Windows (9.45%): Huawei 17.8%, Motorola 16.3%, other Android 16.3%, Samsung 14.4%. `android_other` is the riskiest sizable family overall: 18.5% (relative risk 2.53, CI 2.32 to 2.75).
- **Missing `DeviceInfo`** looks risky overall (relative risk 1.51), but 81% of those transactions are product `C`, and within `C` its rate matches the product average.

### Branch 3 scorecard

| Signal | Overall | Within product | Verdict |
|---|---|---|---|
| Amount | Fraud median higher | Higher in every product (1.1× to 3.0×) | Holds |
| Payments up to $10 | 7.2% vs 2.9% at $30-100 | Fraud rises with amount in every product | Product mix |
| Hours 5 to 10 | About 10% fraud | 1.6× to 2.5× in every product | Holds |
| Credit vs debit | 6.73% vs 2.40% | 2.8× in W, 2.1× in C | Holds where most fraud is |
| Discover | 2.1× | Mainly product W | Partly product mix |
| Hotmail | 5.2% | Below average in C | Product mix |
| Gmail / Outlook | Above average | Above average in C, H, R | Holds outside W |
| ISP domains | 0.5% to 0.7% | Low within W too | Holds |
| Email match | 4.2× | Real only in H and R | Mostly product mix |
| Mobile vs desktop | 9.9% vs 6.2% | 1.2× to 3.4× in every product | Holds |
| Android families | Riskiest devices | All above Windows within C | Holds |

### Branch 4: Entity structure

**Why:** the dataset's labeling logic marks later transactions linked to a flagged account as fraud too, so fraud should cluster by account. Uses the `uid` from section 3.13.

**H1: `D1n` works as a card start date.** Partly supported.

We grouped transactions by `card1` + `addr1`, kept groups with at least 5 transactions (10,855 groups), and counted how many different start days (`D1n`) each group contains.

- A typical group has **5** different start days. With the start days shuffled randomly, a typical group has **11**. So `D1n` clearly tracks real card structure.
- But only **10%** of groups have a single start day, and some have hundreds (max 451). Several cards sharing an identifier and address explains some of that, but not hundreds. So `D1n` is noisy for some transactions, probably because `D1` doesn't always mean exactly what we inferred (its meaning is masked).

This is a descriptive comparison with one shuffle, not a formal statistical test. It's why we call `uid` an **approximate** account.

| Key | Entities | Median size | Singletons | Largest |
|---|---|---|---|---|
| `card1` | 12,730 | 4 | 26.6% | 12,205 |
| `card1` + `addr1` | 88,160 | 1 | 76.8% | 4,721 |
| `uid` | 221,455 | 1 | 68.8% | 361 |

The `card1` + `addr1` and `uid` counts include the 53,761 address-missing transactions, each its own entity, which is why singletons are so common.

**H2: Fraud clusters within accounts.** Strongly confirmed. Entities with at least 2 transactions, real vs shuffled labels:

| Key | Mixed: real | Mixed: shuffled | Fraud in mostly-fraud entities: real | Shuffled |
|---|---|---|---|---|
| `card1` | 14.4% | 33.6% | 5.3% | 0.8% |
| `card1` + `addr1` | 9.9% | 26.9% | 22.6% | 2.9% |
| `uid` | 1.5% | 14.0% | **89.4%** | 19.7% |

**How the 89.4% is calculated** (the line `frauds[frauds / sizes >= 0.5].sum() / frauds.sum()`):

1. Keep accounts with at least 2 transactions.
2. Find accounts where at least half the transactions are fraud.
3. Divide the fraud in those accounts by all fraud in the kept accounts.

Example: 4 accounts with 5/5, 3/4, 1/10 and 1/6 fraud. Total fraud = 10; fraud in the two mostly-fraud accounts = 8; result = 80%.

The finer the key, the stronger the clustering, which suggests `uid` is close to real accounts.

**Caveat:** only entities with 2+ transactions count, and each address-missing transaction is now its own one-transaction entity. So the 89.4% describes repeat accounts with a known address, not the address-missing group (which has 11.4% fraud). Before the missing-address fix this figure was 77.2%: the merged fake accounts mixed unrelated fraud and legit transactions and diluted the clustering.

**H3: Accounts carry over into later periods.** Confirmed. **41.7%** of validation transactions come from a `uid` already seen in training (32.0% of distinct validation uids). This slightly undercounts: an address-missing validation transaction gets a unique marker, so it can never match. Test starts a month later, so its overlap is probably lower.

**Follow-up: where does the fraud live?** Search: `# DO (B4 follow-up): fraud by account type`. Each training transaction is labeled by its `uid`: "address missing" if `addr1` is empty (checked first), "repeat account" if its `uid` appears 2+ times in training, otherwise "one-transaction account".

| Account type | Share of transactions | Fraud rate | Share of all fraud |
|---|---|---|---|
| Repeat account | 67.7% | 2.71% | 52.3% |
| Address missing | 11.4% | 11.40% | **36.9%** |
| One-transaction account | 20.9% | 1.81% | 10.8% |

The 89.4% describes only the repeat-account row, so roughly **47% of all fraud** sits in mostly-fraud repeat accounts. **Address-missing transactions are the biggest blind spot:** 11.4% of transactions but 37% of fraud, and the account key can't link them. Understanding this group, and whether email or device could link its transactions, is a key next step.

Seen vs new accounts in validation are not compared, because that would use validation labels; do it during model evaluation.

---

## 6. Decisions carried into modeling

| Area | Decision |
|---|---|
| **Metrics** | ROC-AUC and PR-AUC, never accuracy |
| **Validation** | Keep the time split; consider a gap of a few weeks to mimic the 30-day gap before test |
| **Reporting** | Report separately for product `W` vs `C`, and for seen vs new accounts |
| **Model** | Tree models (LightGBM or XGBoost): handle missing values natively and learn product interactions |
| **Missing data** | Keep it; never drop columns by emptiness alone; one missing flag per block if needed |
| **V columns** | Use `v_keep` (210 columns) for LR/MLP; optional for trees |
| **Features to include** | `ProductCD`, `TransactionAmt`, `hour` (encoded so 23 sits next to 0, e.g. sine/cosine), `card6`, `card4`, `P_provider`, `has_identity`, `DeviceType`, `device_family`, label-free account summaries per `uid` |
| **Features to consider** | Amount relative to product median; amount frequency; rare-profile flag; `na_cluster` |
| **Features to avoid** | Raw `TransactionDT` or `day`; raw `uid`; account fraud history unless those labels would really be known at prediction time |
| **Leakage rule** | Anything learned from data is computed on `train_df` and applied unchanged to validation and test |

---

## 7. Known weaknesses and open questions

1. **`uid` is approximate.** `D1n` is noisy for some groups, and address-missing transactions (37% of all fraud) can't be linked to accounts, so the account story covers only about half of fraud.
2. **Product rates drift over time.** `H` and `R` fraud rose six- to eightfold after day 25, for reasons we don't know yet.
3. **The hour window was chosen after looking.** Confirm the hours 5 to 10 effect on validation.
4. **Validation may be optimistic.** It starts right after training; test starts 30 days later.
5. **Masked features.** We can show that `D7` or `V169` matters, but not always why.
6. **Label-based account features are a leakage trap.** Chargebacks arrive days or weeks after a transaction, so an account's recent fraud labels wouldn't be known when scoring a new transaction.
7. **B4-H1 is descriptive.** A permutation test with many shuffles and a threshold stated in advance would make it a formal test.

---

## 8. Glossary of the statistics used

**Fraud rate.** Share of transactions that are fraud. Since `isFraud` is 0 or 1, it's just the mean of `isFraud`.

**Relative risk (RR).** One group's fraud rate divided by another's. RR = 2 means fraud is twice as common in the first group; RR = 0.5 means half as common; RR = 1 means no difference. It measures association, not cause.

**95% confidence interval (CI).** The range of relative risks consistent with the data. If the whole interval is above 1 (or below 1), the difference is reliable. Computed on the log scale:

$$\text{CI} = \exp\left(\ln(\text{RR}) \pm 1.96 \sqrt{\tfrac{1}{a} - \tfrac{1}{n_1} + \tfrac{1}{c} - \tfrac{1}{n_2}}\right)$$

where $a$, $c$ are fraud counts and $n_1$, $n_2$ are group sizes.

**"Meaningful" cutoff.** RR above 1.5 or below 0.67, i.e. fraud at least 50% more likely on one side.

**Chi-square test.** Tests whether fraud rates differ between two groups more than chance would explain. Gives a p-value.

**Benjamini-Hochberg correction.** When you run 67 tests at once, some will look significant by luck. This adjusts the p-values to keep the share of false discoveries at 5%.

**Why significance isn't enough.** With 470,000 rows, even tiny differences are statistically significant. So B2-H4 requires both a small adjusted p-value **and** a confidence interval entirely beyond the 1.5 or 0.67 cutoff.

**Product mix (confounding).** When a feature looks linked to fraud only because it's more common in a high-fraud product. Checking the relationship within each product removes this.

**Spearman correlation.** Like ordinary (Pearson) correlation, but based on ranks, so it catches any relationship where both columns rise together, not only straight lines.

**k-means and silhouette score.** k-means groups rows into k clusters of similar rows. The silhouette score (from -1 to 1) measures how well separated the clusters are; higher is better. It peaked at k = 3 (about 0.55).

**PCA.** Compresses many columns into a few summary numbers that keep as much of the variation as possible. Used here only to plot the missingness profiles in 2D.

**Shuffled baseline.** Randomly scrambling one column (labels or `D1n`) while keeping everything else fixed. It shows what the numbers would look like with no real relationship, so the gap between real and shuffled is the real effect.
