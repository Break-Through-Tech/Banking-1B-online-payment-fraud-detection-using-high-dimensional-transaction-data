# Feature Engineering, Explained


Imagine a reviewer sitting at a payment-review desk. Each payment arrives with a receipt: its amount, product, card information, and sometimes device or identity information.

Reading a receipt tells the reviewer what happened. Adding context helps the reviewer ask better questions:

- Is this amount large compared with other payments for the same product?
- Is important information missing?
- Have we observed earlier payments from the same approximate account?
- Is this payment much larger than those earlier payments?

**Feature engineering means turning the information we have into useful descriptions a model can learn from.** A feature is one input column. For example, `TransactionAmt` describes the amount itself, while `amount_to_product_median` describes the amount in context.

The notebook prepares those descriptions. It does not train a fraud model or prove that any new feature improves fraud detection. That comes in the modeling stage.

The examples in this guide are illustrative unless identified as numbers from the saved full run.

## 1. Set up the workspace

The first step finds the project and its data files, then sets the options for the run.

Think of this as organizing the review desk before opening the receipts. Everyone needs to know where the incoming records live, where finished work goes, and which instructions to follow.

| Setting | Simple meaning |
|---|---|
| `DATA_DIR` | The folder containing the downloaded CSV files |
| `DATA_DIR_OVERRIDE` | An optional different location for those files |
| `OUTPUT_DIR` | Where prepared tables and preparation rules are saved |
| `TRAIN_SHARE = 0.80` | Approximately the earliest 80% of labeled payments is used for training |
| `VALIDATION_SHARE = 0.10` | Approximately the next 10% is used to compare model choices |
| `READ_NROWS = None` | Read all transaction rows |
| `SAVE_ARTIFACTS = True` | Write the prepared results to disk |

Setting `READ_NROWS` to a smaller number creates a quick practice run, called a smoke run. It checks that the workflow works; it is not a dataset for reporting final results. Those outputs go into a separate smoke-run subfolder.

## 2. Attach identity information while keeping every payment

The data has two kinds of records: transaction records and identity records. `TransactionID` connects them.

Imagine receipts and optional visitor-information cards. We attach a visitor card to the matching receipt when one exists. A receipt without a visitor card still belongs in the pile.

That is what a **left join** does: every transaction stays, and matching identity information is added.

| TransactionID | Payment amount | Matching identity record? | `has_identity_record` |
|---|---:|---|---:|
| 101 | 20 | Yes | 1 |
| 102 | 75 | No | 0 |

The second payment remains in the dataset. Its identity fields are missing rather than causing the payment to be deleted. A matched identity record can itself contain missing fields, so `has_identity_record = 1` does not mean every identity field is filled.

The notebook also standardizes test identity names such as `id-01` to `id_01`, checks for duplicate IDs, and checks that joins do not multiply rows. It stores many decimal columns using less memory, while keeping the original amount precision until amount features have been calculated.

## 3. Separate learning, practice, and final evaluation by time

Imagine studying earlier cases, using later cases for practice, and saving the latest cases for a final exam.

The notebook sorts labeled payments by `TransactionDT` and divides them into three periods. In the saved full run:

| Period | Payments | Purpose |
|---|---:|---|
| Training | 472,432 | Learn preparation rules and, later, train the model |
| Validation | 59,054 | Compare features and model settings |
| Final holdout | 59,054 | Evaluate the finished choice once |
| Kaggle test | 506,691 | Produce predictions; fraud labels are unavailable |

These correspond to an 80% / 10% / 10% split of the labeled data. A boundary can move slightly so payments with the same timestamp stay in the same period.

The EDA used an 80% / 20% split. This notebook retains its training period and divides the later 20% into validation and final holdout.

`TransactionDT` is elapsed time in seconds from an unknown reference point. It tells us the order of payments, but does not give a verified calendar date or local timezone. The Kaggle test period also starts later than the labeled periods, with a gap; our adjacent validation period does not reproduce that gap.

The fraud answer, `isFraud`, is separated from the features. Think of keeping the answer sheet out of the reviewer’s input notes. **Data leakage** means allowing information into an experiment that should not have been available—for example, using future payments to describe an earlier payment.

## 4. Describe payment size and recurring time patterns

### Amounts on different scales

`TransactionAmt` keeps the payment’s original amount. `amount_log1p` adds another view using `log(1 + amount)`.

Imagine a map that must show both a small town and a large country. A smaller scale makes the large area fit on the page. A logarithm similarly compresses large amounts while preserving their order.

| Amount | Approximate `log(1 + amount)` |
|---:|---:|
| 9 | 2.30 |
| 99 | 4.61 |
| 999 | 6.91 |

The amount grows dramatically, but the transformed values are closer together. The model receives both representations; validation will establish whether the extra one helps.

`amount_fractional_unit` records the part after the decimal point. For an amount of `25.375`, it is `0.375`.

`amount_has_subcent_precision` records whether the amount contains precision beyond whole cents. `25.37` has a flag of 0; `25.375` has a flag of 1. This describes formatting, not a confirmed reason for the payment or proof of fraud.

### Time as a clock face

On a number line, hour 23 and hour 0 look far apart. On a clock, they are neighbors.

The notebook adds `hour_sin` and `hour_cos`, which represent two coordinates on a circle. Using both describes a position around the clock and makes the end and beginning of a cycle close together. `weekday_sin` and `weekday_cos` do the same for a seven-day cycle.

It also keeps `relative_hour` and `relative_weekday` as categorical cycle positions. Because the starting point is unknown, these are not verified local hours or named weekdays. We cannot call weekday 0 “Monday.”

Raw `TransactionDT` and raw day are excluded from model inputs. They remain useful for ordering records, but future periods have time values beyond those seen during training.

## 5. Describe missing information

Imagine two forms: one has nearly every box filled, while the other contains many blanks. The pattern of blanks tells us something about how information was collected, even before reading the filled boxes.

The notebook counts missing values across the original input fields and within several groups:

| Feature | What it counts or records |
|---|---|
| `raw_missing_count` | Missing original inputs, excluding the target, transaction ID, and raw time |
| `identity_missing_count` | Missing identity and device inputs |
| `card_address_missing_count` | Missing card and address inputs |
| `v_missing_count` | Missing anonymous V inputs |
| `addr1_missing`, `D1_missing`, and similar flags | Whether a particular original field is missing: 1 means missing, 0 means present |

The individual flags also cover `addr2`, purchaser email, recipient email, and device information.

These counts are calculated before categorical missing values are converted into explicit category names. They use fixed lists of original columns, so adding new features does not change what gets counted.

A blank does not automatically mean fraud. Different products may collect different information. The model must learn whether missingness is useful alongside the other inputs.

The notebook keeps anonymous `C`, `D`, and `V` inputs, including sparse ones. Their names reveal little about their meaning, but that alone is not a reason to discard them. This version does not reduce the V columns or cluster missingness patterns.

## 6. Simplify email and device descriptions

Imagine organizing books by author or genre instead of giving every edition its own shelf. Broader groups make similar items easier to recognize.

The notebook trims spaces and lowercases email domains. It keeps those normalized domains and adds a provider-like group using the first part of the domain:

| Domain | Provider-like group |
|---|---|
| `yahoo.com` | `yahoo` |
| `yahoo.fr` | `yahoo` |
| `gmail.com` | `gmail` |

The resulting features are `P_provider` for the purchaser and `R_provider` for the recipient. This grouping is a simple rule, not a verified identification of the company behind every domain.

`email_domain_relationship` has five possible descriptions: same domain, different domains, purchaser missing, recipient missing, or both missing. Two matching domains do not mean two matching email addresses: different people can both use `gmail.com`.

For devices, keyword rules turn raw `DeviceInfo` strings into `device_family` values. For example, `SM-G930V Build/NRD90M` matches the `sm-` rule and becomes `samsung`. The first matching rule wins. A present but unrecognized string becomes `other`; missing or blank device text gets the missing category.

Raw `DeviceInfo` is removed from model inputs and replaced with this family. The simpler groups may help with unfamiliar device versions, but their usefulness still needs model validation.

## 7. Build a fixed reference guide from training

Imagine writing a reference guide from the training receipts: which categories were common, which rounded amounts appeared frequently, and what a typical amount looked like for each product. Once written, that guide is applied unchanged to later payments.

### Category and amount frequency

Suppose a particular card code appears in 200 of 1,000 training payments. Its training frequency is:

```text
200 / 1,000 = 0.20, or 20%
```

`card1_train_frequency` attaches that reference value to payments with the same code. Similar frequency features cover `card2`, `addr1`, both full email domains, `P_provider`, and `device_family`.

Missing values have their own training frequency. A value absent from training gets frequency 0, meaning “not observed in our training reference,” not “impossible” or “fraudulent.”

`amount_train_frequency` uses amounts rounded to cents. For this particular lookup, an amount such as `10.004` is grouped with `10.00`. The separate sub-cent feature still describes the original amount’s finer precision.

### Amount compared with its product

The **median** is the middle amount when amounts are sorted. It is less affected by a few extremely large payments than an average.

Suppose product W has a training median amount of 20. A payment of 100 gets:

```text
amount_to_product_median = 100 / 20 = 5
```

This means five times the product’s training median. A ratio near 1 means near that reference amount. An unfamiliar product uses the overall training median. The denominator is never allowed below 1, which avoids division by zero and very small denominators.

The medians use all training payments, without looking at fraud labels.

**These reference features describe the training population. They are not a payment’s historical activity.** Even an early training row uses the complete training-period reference, just as it would use a scaler fitted on the training set. Later datasets never update this guide.

## 8. Read the activity log without reading ahead

The reference guide from Step 7 stays fixed. This step uses a different idea: an activity log that grows as payments arrive.

Imagine the reviewer opening that log at the moment a payment arrives. Earlier entries are available; tomorrow’s entries are not.

### An approximate account key

The dataset does not provide a verified customer ID for this purpose. The notebook creates an approximate grouping key from `card1`, `addr1`, and a time-based value:

```text
start_day_proxy = elapsed_day - D1
approximate key = card1 + addr1 + start_day_proxy
```

For example, elapsed day 160 minus `D1 = 10` gives 150. At elapsed day 175, `D1 = 25` also gives 150. If `D1` behaves as hypothesized, the shared result can help link related payments.

This is like matching paper records using several clues when a reliable membership number is unavailable. It remains an approximation: shared card/address codes can combine different people, and `D1` has a masked meaning. We cannot claim the key identifies a real customer or a confirmed account-opening date.

If any component is missing or non-finite, the key stays missing. Incomplete records are not pooled into one invented account. The key itself is saved as metadata and excluded from model inputs.

### Features from earlier payments

| Feature | Simple meaning |
|---|---|
| `account_key_missing` | We cannot form the grouping key |
| `account_has_prior_history` | At least one strictly earlier payment exists for this key |
| `account_prior_count` | Number of strictly earlier payments |
| `account_prior_mean_amount` | Average amount across those earlier payments |
| `amount_to_prior_account_mean` | Current amount divided by that earlier average, with the denominator floored at 1 |
| `account_seconds_since_previous` | Seconds since the latest strictly earlier payment timestamp |

When there is no earlier history, the prior count is 0 and the mean, amount ratio, and time gap are missing. Zero history means “we have no observed past,” not “this is a safe payment.”

### Two simultaneous payments cannot see each other

The notebook checks this exact example for one key, A:

| Time in seconds | Current amount | Prior count | Prior mean amount |
|---:|---:|---:|---:|
| 10 | 10 | 0 | Missing |
| 20 | 20 | 1 | 10 |
| 20 | 40 | 1 | 10 |
| 30 | 80 | 3 | 23.33 |

Both payments at time 20 see only the amount-10 payment at time 10. Neither sees the other time-20 payment. At time 30, all three earlier payments are available, so their mean is `(10 + 20 + 40) / 3 = 23.33`.

The notebook groups payments by account and timestamp, calculates running totals, and subtracts the entire current timestamp’s group. This excludes the current payment and all simultaneous payments.

The workflow simulates scoring payments one after another. Earlier validation, holdout, and Kaggle payments can enter the activity log for later payments, without using their fraud labels. They do not update the Step 7 reference guide or train the model. A system unable to update its activity log during prediction would need a different evaluation setup.

Although the event log combines all periods, each payment’s history uses only smaller timestamps. Adding a future payment must leave earlier history features unchanged.

## 9. Give every table the same input format

Imagine handing different reviewers forms with identical boxes in identical positions. “Payment amount” must mean the same thing on every form.

The notebook gives training, validation, holdout, and Kaggle tables the same columns and category definitions. This consistent layout is called a **schema**.

A category is a named group rather than a measurable quantity. Think of bus route numbers: route 40 is not twice as much bus as route 20. Card and address codes are treated similarly. The notebook also treats `id_12`–`id_38`, text fields, and relative hour/weekday positions as categories.

Allowed category values are learned from training, with two special values:

| Category | Meaning |
|---|---|
| `__MISSING__` | No value was supplied |
| `__UNSEEN__` | A value was supplied, but it was absent from the training vocabulary |

Numerical columns use `float32`, a smaller-memory decimal representation. Their missing values remain `NaN`, meaning unknown; they are not automatically changed to zero. Infinite numerical values are converted to missing values.

These tables suit a tree model that understands pandas categories and numerical missing values. A logistic regression or neural network needs additional preparation: filling missing numerical values, encoding categories, and usually scaling numbers. Those rules must also be fitted on training only. Category codes should not be passed to such a model as ordinary continuous measurements.

## 10. Check the tables and the time logic

Imagine checking that every exam paper has the right student name and that nobody received an answer sheet by mistake.

The notebook checks that feature rows, labels, and metadata match by `TransactionID`; columns and category vocabularies match across tables; and forbidden inputs such as `isFraud`, raw time, and the approximate account key are absent from the feature columns.

It also checks the behavior of small examples:

- Simultaneous payments see the same earlier history.
- Missing keys do not create shared history.
- Adding a future payment does not change past features.
- Shuffling the event rows does not change the results when matched back by ID.
- Unfamiliar categories and rounded amounts receive frequency 0.
- An unfamiliar product uses the overall training median.

These checks establish that the preparation follows its intended rules. They do not establish that the eventual fraud model will perform well.

## 11. Inspect the prepared features

Imagine inspecting a few packed orders before shipping them. The notebook displays feature-family counts, example training rows, and a table of missingness and distinct values.

Its three charts show:

| Chart | What to notice |
|---|---|
| Raw amounts up to the training 99th percentile | The shape of ordinary amounts without the largest amounts dominating the chart |
| Log-transformed amounts | How the transformed scale represents both smaller and larger payments |
| Missing-input counts | How information availability varies between payments |

The 99th-percentile cutoff changes the first chart only. It does not remove those large payments from training.

All these exploratory views use training data. Also, a categorical missing value has already become `__MISSING__`, so it does not appear as `NaN` in the missingness summary. The earlier missingness features still preserve that information.

The saved full run contains **469 input columns: 54 categorical and 415 numerical**. These are not 469 newly invented features; many are retained original inputs.

## 12. Save a handoff that another teammate can reuse

Imagine handing over the prepared forms, a separate answer sheet, and the instructions used to prepare them. The next teammate should be able to train a model without guessing how the data was processed.

Artifacts are saved under `outputs/feature_engineering/`:

| File or pattern | Contents |
|---|---|
| `X_<split>.parquet` | Model inputs, indexed by transaction ID |
| `y_<split>.parquet` | Fraud labels for training, validation, and holdout; no Kaggle label file |
| `metadata_<split>.parquet` | Time, approximate account key, and whether that key appeared in training |
| `preprocessing_rules.json` | Training frequencies, amount medians, and category vocabularies |
| `run_manifest.json` | Run settings, split details, feature lists, comparison sets, and processing policies |
| `feature_dictionary.csv` | Feature families, types, missing-value policies, and source rules |
| `history_state_after_kaggle.parquet` | Activity totals and latest timestamps after the complete Kaggle period |

Parquet stores tables compactly and preserves their types. JSON stores readable settings and mappings. The notebook reloads the saved training feature table and checks that it matches the original table.

The history snapshot is the activity log **after the entire stream**. It cannot be used to initialize an earlier validation experiment: that would give the earlier experiment future information. The saved files and notebook describe how to rebuild this workflow; they are not yet a complete production prediction service.

## How the next model experiment uses this work

The manifest defines three comparison sets:

| Set | Included inputs |
|---|---|
| `raw_baseline` | Retained original inputs, with normalized email domains and raw `DeviceInfo` omitted |
| `engineered_without_history` | Those inputs plus the engineered features outside the past-activity family |
| `engineered_with_history` | All prepared inputs, including past activity |

Think of testing a recipe: first try the base recipe, then add one group of ingredients, then another. Keep the data split and model setup consistent so the comparison tells us whether the added features helped.

Validation guides those comparisons. The final holdout is evaluated once after model and feature choices are fixed. Kaggle predictions are separate, and should be matched to the sample submission by transaction ID because this notebook sorted payments by time.

Useful questions for the modeling stage are how many alerts are correct (**precision**), how much fraud is caught (**recall**), and how many legitimate payments are incorrectly flagged. PR-AUC summarizes the precision/recall trade-off across alert thresholds; its exact scoring definition should be recorded. Reviewing only the top-risk 1% of payments is another practical way to compare models at a fixed review budget.

The notebook has prepared clearer descriptions of payments. Whether those descriptions improve future fraud detection is the question the next experiment must answer.
