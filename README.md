# Online Payment Fraud Detection Using High-Dimensional Transaction Data

Banking 1B · Break Through Tech AI Studio · Fall 2026

---

### 👥 **Team Members**

Team names follow the [Team Alliance](https://docs.google.com/document/d/17AW2hXTLYdMQHgcYF3R-NMvCK_1ujHQXuc5blRLUXW0/edit?tab=t.0). Contributions are documented in the [combined EDA](notebooks/BTT_Mastercard_EDA_Combined.ipynb) and the final feature-engineering notebook's [team attribution table](notebooks/BTT_Mastercard_Feature_Engineering_Final.ipynb). Assigned modeling work follows the [October task plan](https://docs.google.com/document/d/1JGJLuNtPFucyJG5O20psLaR05h6cs0px0gejBr7NE5A/edit?tab=t.0).

| Name | GitHub Handle | Contribution |
|---|---|---|
| Kyaw Thi (Eric) Ha | [@KTH-Sys](https://github.com/KTH-Sys) | EDA contributions on memory use, time gaps, redundant V features, and amount bands; chronological splitting, email/device features, frequency encoding, account history, missingness counts, leakage checks, and saved outputs. Assigned to deep-learning model development. |
| Salamot Fakoya (Sal) | [@Sal-Fakoya](https://github.com/Sal-Fakoya) | Hypothesis-driven EDA and notebook consolidation; second account key, distinct-value history, missingness-block flags, reduced-V features, neural-net inputs, signal checks, KS tables, and per-run configuration. Assigned to feature engineering, baseline models, deep learning, and output comparison. |
| Jean Cho | — | Device-family features, frequency encoding, and missingness-profile clustering. Assigned to the logistic-regression baseline and output comparison. |
| Jennifer Perez-Juarez | — | Out-of-time target encoding and stability checks; new-device/email/address flags, transaction velocity, account amount statistics, and repeated-amount features. Assigned to deep-learning model development. |
| Salvador Frias | — | Kaggle-discussion research; categorical treatment of the `id_14` timezone, sub-cent amount flags, and time, frequency, device, and V-feature reduction ideas incorporated into the combined notebook. Assigned to the XGBoost baseline. |
| Susie Marrero | — | Feature-engineering work on amount, time, missingness, email, device, frequency, and account history; proposed 7-day and 30-day velocity windows incorporated into the final notebook. |
| Bianca Blevins | — | Assigned to the logistic-regression baseline and comparison of model outputs in the October task plan. |

A dash indicates an unverified GitHub handle. Modeling assignments remain planned work until their results are recorded; the October plan currently marks these tasks as “To-do.”

---

## 🎯 **Project Highlights**

- Combined team EDA and feature-engineering work into one canonical notebook for each stage.
- Built features describing transaction amount, time, email, device, missingness, and prior account behavior.
- Prepared separate input tables for tree models and neural networks, with chronological validation and a labeled final holdout.
- Documented strong class imbalance, product-specific fraud patterns, and the importance of missingness and account history.
- Added single-feature signal checks and reusable KS tables to support the upcoming model comparison.

---

## 👩🏽‍💻 **Setup and Installation**

Clone the repository and install dependencies from the repository root:

```bash
git clone https://github.com/Break-Through-Tech/Banking-1B-online-payment-fraud-detection-using-high-dimensional-transaction-data.git
cd Banking-1B-online-payment-fraud-detection-using-high-dimensional-transaction-data
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Accept the IEEE-CIS Fraud Detection competition rules in Kaggle before downloading the data. Choose a download workflow:

- **Notebook / Colab:** run [import_kaggle_data.ipynb](notebooks/import_kaggle_data.ipynb). In a local checkout, it supports credentials from environment variables, `.env`, or `~/.kaggle/kaggle.json`, followed by Kaggle login. To use `.env`, copy [.env.example](.env.example) and fill in your credentials. Standalone Colab sessions can use the login prompt after installing `kagglehub` and `pandas`.
- **Local CLI:** authenticate with `.venv/bin/kaggle auth login`, then run `bash scripts/download_data.sh` to extract the competition files into `data/`.

Run the [combined EDA](notebooks/BTT_Mastercard_EDA_Combined.ipynb), then the [final feature-engineering notebook](notebooks/BTT_Mastercard_Feature_Engineering_Final.ipynb), from top to bottom. In feature engineering, set `CONFIG["data_path"]` to the local `data/` folder to use the CLI download, or leave it as `None` to download with KaggleHub. Change `CONFIG["run_name"]` when changing experiment settings. Generated tables and configuration go under `outputs/feature_engineering/<run_name>/`.

The full notebooks require several GB of RAM. Large datasets, generated features, models, and local credentials are ignored by Git; the two CSVs already tracked in `data/` are retained. Never commit credentials or their notebook output. Reports and presentations live in `markdowns/`, and executable notebooks live in `notebooks/`.

---

## 🏗️ **Project Overview**

This Break Through Tech AI Studio project investigates whether transaction history and behavioral context improve online payment fraud detection. Using the public IEEE-CIS dataset, the team will compare conventional machine-learning baselines with neural networks and, where reliable account sequences can be constructed, temporal models.

The intended outcome is a reproducible pipeline that improves fraud detection while measuring false alerts and review workload. The planned final deliverables include model evaluation, explanations of fraud-risk scores, and a lightweight prototype for scoring sample transactions. Scope and milestones are documented in the [Project Brief and Workplan](https://docs.google.com/document/d/1ODtH1VlzTyfoYWEl5Saj2oqifH6FETS7MzNWqx31ay4/edit?tab=t.0) and [Challenge Overview](Challenge-Project-Overview.md).

---

## 📊 **Data Exploration**

The IEEE-CIS Fraud Detection dataset contains 590,540 labeled training transactions and 506,691 unlabeled Kaggle test transactions. Transaction and identity CSV tables contain hundreds of numerical and categorical variables, many anonymized. The target is `isFraud`; identity records are available for only a subset of transactions.

The [combined EDA notebook](notebooks/BTT_Mastercard_EDA_Combined.ipynb) joins the tables on `TransactionID`, aligns identity-column names, reduces memory use, and examines the earliest 80% of labeled transactions. Its analysis covers fraud prevalence, time shifts, missingness blocks and profiles, product mix, transaction amounts, card/email/device attributes, and approximate account structure. Findings are checked within product type to distinguish product-mix effects from signals that persist within products.

Challenges include heavy missingness, class imbalance, anonymized feature definitions, approximate account keys, and a roughly 30-day gap before the Kaggle test period. Plots and detailed findings are available in the [EDA summary notes](markdowns/EDA_Summary_Notes_1.md) and [EDA presentation](markdowns/BTT-Presentation-EDA.pptx).

---

## 🧠 **Model Development**

The current implementation prepares model inputs and evaluation tools. The [final feature-engineering notebook](notebooks/BTT_Mastercard_Feature_Engineering_Final.ipynb) combines categorical and frequency encoding, out-of-time target encoding, account-history and velocity features, missingness indicators and clusters, and a reduced V-column feature set. Fitted preparation rules use training data, and behavioral histories use strictly earlier timestamps.

The modeling split is chronological: 80% training (472,432 rows), 10% validation (59,054), and 10% final holdout (59,054), with boundaries kept between timestamps. This divides the EDA's later 20% into validation and final holdout. Kaggle test labels are unavailable.

The [October task plan](https://docs.google.com/document/d/1JGJLuNtPFucyJG5O20psLaR05h6cs0px0gejBr7NE5A/edit?tab=t.0) assigns the next experiments:

| Planned experiment | Assigned members |
|---|---|
| Logistic regression on raw, engineered, and history-enriched inputs | Jean, Bianca, Salamot |
| XGBoost on the same feature variants | Salamot, Salvador |
| Small neural-network / deep-learning baseline | Eric, Jennifer, Salamot |
| Compare model outputs | Bianca, Jean, Salamot |

The project brief also proposes LightGBM and optional sequence models. Feature comparisons can use the saved `raw`, `all_no_history`, `all`, and `all_reduced_v` sets. Validation will guide feature selection and hyperparameter tuning, emphasizing PR-AUC, precision, recall, F1, false-positive rate, and recall at a fixed review rate; ROC-AUC, fraud-dollar capture, and KS tables provide additional views. Final model comparison and holdout evaluation are pending.

---

## 📈 **Results & Key Findings**

Current results cover EDA and saved single-feature diagnostics:

| Finding | Evidence |
|---|---|
| Fraud is rare | About 3.51% of transactions in the EDA training period are fraudulent. Accuracy alone would obscure poor fraud detection. |
| Product mix matters | Product `C` has an 11.33% fraud rate, while high-volume product `W` contains about 43% of fraud cases. |
| Missingness carries information | 414 of 434 EDA columns contain missing values, with recurring blocks and profiles that differ in fraud rate. |
| Account coverage is incomplete | Address-missing transactions represent about 11.4% of training data and 36.9% of its fraud, motivating a second account key. |
| Target encoding has individual signal | Saved validation ROC-AUC is 0.728 for `te_card2` and 0.683 for `te_R_emaildomain`, each scored alone. |

Sources: [EDA summary notes](markdowns/EDA_Summary_Notes_1.md) and saved outputs in the [feature-engineering notebook](notebooks/BTT_Mastercard_Feature_Engineering_Final.ipynb). The single-feature scores are preliminary diagnostics. Trained-model performance, fairness analysis, and final holdout results remain to be established.

---

## 🚀 **Next Steps**

- Train and compare the assigned baselines using consistent chronological splits and feature variants.
- Measure whether account history and engineered features improve validation performance over raw inputs.
- Evaluate seen versus new accounts, product segments, time drift, and fraud-dollar capture.
- Confirm the advisor's detection-rate and false-positive-ratio targets and report the corresponding KS tables.
- Tune the selected models on validation, then evaluate the final choice once on the labeled holdout.
- Add SHAP explanations, document limitations, and build the planned fraud-risk scoring prototype.

Track tasks on the team's [GitHub Project board](https://github.com/orgs/Break-Through-Tech/projects/180).

---

## 📝 **License**

A project license has not yet been selected. The team will confirm the license with the Challenge Advisor. Dataset access and use are governed separately by the Kaggle competition terms.

---

## 📄 **References**

- [Team Alliance — Banking 1B](https://docs.google.com/document/d/17AW2hXTLYdMQHgcYF3R-NMvCK_1ujHQXuc5blRLUXW0/edit?tab=t.0): team roster and collaboration agreement.
- [Milestone #2 Task Planning — October](https://docs.google.com/document/d/1JGJLuNtPFucyJG5O20psLaR05h6cs0px0gejBr7NE5A/edit?tab=t.0): assigned feature-engineering, modeling, and comparison tasks.
- [Project Brief and Workplan](https://docs.google.com/document/d/1ODtH1VlzTyfoYWEl5Saj2oqifH6FETS7MzNWqx31ay4/edit?tab=t.0): objectives, success metrics, milestones, and tools.
- [Challenge Overview](Challenge-Project-Overview.md) and [project plan](markdowns/plan.md).
- [IEEE-CIS Fraud Detection dataset and competition](https://www.kaggle.com/competitions/ieee-fraud-detection).
- [EDA summary notes](markdowns/EDA_Summary_Notes_1.md) and [EDA presentation](markdowns/BTT-Presentation-EDA.pptx).
- [Feature engineering explanation](markdowns/Feature_Engineering_Explained.md), [summary report](markdowns/BTT_Mastercard_Feature_Engineering_Summary.pdf), and [presentation](<markdowns/Feature Engineering-Better-Explanation.pptx>).
- [Final feature-engineering notebook](notebooks/BTT_Mastercard_Feature_Engineering_Final.ipynb): detailed team attribution, feature definitions, checks, and saved diagnostics.

---

## 🙏 **Acknowledgements**

Thank you to Challenge Advisor **Debasmita Das**, AI Studio Coach **Harshini Donepudi**, and **Break Through Tech AI Studio** for guidance and support. We also acknowledge IEEE-CIS, Vesta, and Kaggle for making the competition data available, and every Banking 1B teammate for the analyses and feature-engineering work combined here.
