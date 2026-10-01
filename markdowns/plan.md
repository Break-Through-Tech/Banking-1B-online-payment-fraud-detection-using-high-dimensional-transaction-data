# Online Payment Fraud Detection Plan

## Goal

Build a model that spots fraudulent online payments while keeping false alerts low. Use the IEEE-CIS dataset and compare a strong basic model with a neural network. Check whether past transaction patterns help when the data supports it.

## 1. Set up and understand the data — September

- [ ] Set up a Python environment and install `requirements.txt`.
- [ ] Download the dataset using `scripts/download_data.sh` or load it with `notebooks/import_kaggle_data.ipynb`.
- [ ] Keep large data files and Kaggle credentials out of Git.
- [ ] Join the labeled transaction data with identity data using `TransactionID`, keeping transactions that have no identity record.
- [ ] Check missing values, duplicate IDs, data types, and the number of fraud cases.
- [ ] Make simple charts of payment amounts, fraud rates, and changes over time.

**Output:** A notebook explaining what is in the data and what needs cleaning.

## 2. Prepare the data and build a first model — September

- [ ] Use `isFraud` as the answer to predict: `1` means fraud and `0` means legitimate.
- [ ] Sort by `TransactionDT` and split the labeled data into earlier training data, later validation data, and the latest data for final testing.
- [ ] Keep the final test set untouched until model choices are finished. The official Kaggle test data has no fraud labels, so use it only for predictions.
- [ ] Handle missing values and convert categories into a form each model can use.
- [ ] Learn cleaning rules, scaling, and feature selection from training data only to avoid using future information.
- [ ] Train a simple model, such as logistic regression, then a stronger tree model, such as LightGBM.
- [ ] Try class weights so the model pays more attention to the rare fraud cases.
- [ ] Save the data split, settings, and validation results so the work can be repeated.

**Output:** A working first model and a results table for comparison.

## 3. Improve and compare models — October

- [ ] Compare promising tree models, such as LightGBM, XGBoost, or CatBoost.
- [ ] Train a basic neural network called a multilayer perceptron (MLP).
- [ ] Try useful feature changes and a small number of model settings, using the same validation data for fair comparisons.
- [ ] If reliable transaction groups can be formed, add past activity features, such as recent payment counts or time since the previous payment.
- [ ] Build those features using only information available before each transaction. Do not assume shared card or device fields identify one customer.
- [ ] Compare results with and without past activity features to see whether they help.
- [ ] Optionally try a sequence model, such as a GRU or LSTM, if meaningful transaction histories can be built and time allows.

**Output:** A comparison showing which model works best and whether past activity helps.

## 4. Evaluate and explain the results — November

- [ ] Choose the model and its fraud-alert cutoff using validation data, then evaluate once on the held-out final test set.
- [ ] Use precision-recall AUC (AUPRC) as the main score for comparing models when fraud is rare.
- [ ] Report precision (how many alerts are correct), recall (how much fraud is caught), and F1 (a balance of both).
- [ ] Report the false-positive rate: the share of legitimate payments incorrectly flagged.
- [ ] Measure how much fraud is caught at a fixed review limit, such as reviewing the highest-risk 1% of payments, and report the fraud amount captured too.
- [ ] Review missed fraud, false alerts, and performance across time periods and transaction groups.
- [ ] Use feature importance or SHAP to explain which inputs influence predictions, noting that many columns have anonymous meanings.

**Output:** Final results, clear charts, example explanations, and known limitations.

## 5. Build the demo and finish documentation — November, ready by December

- [ ] Build a small demo that accepts a sample transaction and returns a fraud-risk score with supporting explanations.
- [ ] Save the model and the preprocessing steps it needs.
- [ ] Update `README.md` with setup instructions, how to run the work, and the main findings.
- [ ] Prepare a final presentation covering the problem, approach, results, and next steps.
- [ ] Explain that the demo is a research prototype and describe what would need checking before real use.

**Output:** A working demo, repeatable workflow, completed README, and presentation.

## Team routine

- [ ] Track weekly tasks on a GitHub Projects board with an owner for each task.
- [ ] Share progress and blockers during team check-ins.
- [ ] Record each experiment's model, features, settings, and results in one shared table.
