---

> ## Challenge Advisor: Update & Finalize Your Project Overview
>
> > 💡 **These grey text instructions are just for you, the team's Challenge Advisor; please delete them once you have completed the steps below.**
>
> We've pre-populated this Challenge Project Overview page — which is what will be shared with your Break Through Tech student team in August — using the details from your submission form. You should have received an email inviting you to join this repo as a Collaborator, enabling you to add files and make edits.
> 
> In order for your project to be finalized and assigned to a team, please:
> 1. **Review all sections below** and update or expand any content as needed, making sure to address the SME Feedback in the section immediately below. Look for square brackets to find the places below that require additional inputs from you (e.g., "About [Company / Org Name]").
> 2. **Add your dataset** to the [data folder](data) in this repo.
> 3. **Close the Issue assigned to you in this repo** to let us know that you have made your edits and the overview page is ready for final review. You can do this by going to the _Issues_ tab in the top left section of the menu above, add a comment that says "CA review complete", and click the button to Close the Issue. 
>
> If you're unfamiliar with how to edit a page like this in GitHub, check out [this tutorial](https://ubc-lib-geo.github.io/gis-workshop-waml-template/content/handson/edit-readme.html) for a quick overview (start with step 2 and only edit this page), and [this guide](https://ubc-lib-geo.github.io/gis-workshop-waml-template/content/markdown.html) on how to use Markdown to compose text.
>
>
> ❌ Remember that this is a public repo. Do NOT include: Proprietary data, PII, API keys, credentials, or anything confidential.

---

## 📋 BTT Internal Evaluation Notes
*(This section is for BTT staff and CAs only — remove before sharing with students)*

### Technical Vetting
| Check | Status | Notes |
| :--- | :--- | :--- |
| Python Compatibility | 🟢 | The tech stack is centered on Python, using Python-compatible libraries for machine learning and data processing. |
| Data Readiness | 🟢 | The IEEE-CIS Fraud Detection dataset is publicly available and appears ready for use, containing 590,000 transactions without significant preprocessing requirements indicated. |
| Resource Check | 🟢 | Users can utilize Google Colab, which offers sufficient computing resources in its free tier without specialized hardware requirements. |

### Internal Scores
- **Student Fit Score:** 7/10
- **Technical Depth Score:** 8/10
- **Overall Recommendation:** REVISE

### Advisor Feedback Draft
The project has a clear application and relevant industry context, encouraging student engagement. However, the diversity of techniques may introduce confusion. Simplifying modeling options or focusing on a few key methods could improve clarity and success. Additionally, ensure that students are equipped with the skills to interpret model results effectively, as this is essential for real-world applications. Recommend providing workshops or resources on the specific ML libraries to be used.

---

# Online Payment Fraud Detection Using High-Dimensional Transaction Data

**Industry:** Banking / Payment Industry  
**Challenge Advisor:** Debasmita Das, debasmita.das@iiml.org  
**AI Studio Coach:** Harshini Donepudi, harshini.donepudi@breakthroughtech.org  
**Program:** Break Through Tech AI Studio – Fall 2026


## 🏢 About Payment Industry
The payment industry enables money to move between consumers, merchants, banks, and digital platforms. It includes card networks, payment processors, digital wallets, merchant acquiring, fraud prevention, cross-border payments, and other financial technology services. Visa is one of the largest players in the industry. It operates a global payment network that connects cardholders, merchants, issuing banks, and acquiring banks. Its services include transaction processing, debit and credit payments, cross-border payments, fraud prevention, tokenization, authentication, and data services. Mastercard is Visa’s main global competitor and offers similar payment network services, along with cybersecurity, identity verification, open banking, analytics, and consulting. Both companies benefit from very large networks of banks and merchants. Stripe and PayPal operate somewhat differently. Stripe focuses on helping businesses accept and manage digital payments through services such as online checkout, subscriptions, fraud prevention, and marketplace payments. PayPal combines merchant payment services with consumer wallets such as PayPal and Venmo. Overall, the payment industry is moving beyond basic transaction processing toward areas such as security, data, embedded finance, and faster cross-border payments.

---

## 🎯 The Challenge
### Project Summary
In this project, you will use online payment transaction data and machine-learning and deep-learning techniques, including gradient-boosted trees, recurrent neural networks, and Transformer-based sequence models, to build a binary classification model that predicts whether an online payment transaction is fraudulent or legitimate. This will help payment companies address payment fraud losses, manual-review workload, customer friction, and the need to detect suspicious transactions accurately and quickly.

The project will use the publicly available IEEE-CIS Fraud Detection dataset, which contains approximately 590,000 transactions and more than 400 transaction, card, identity, device, and behavioral variables. The team will investigate whether information from a transaction’s temporal and behavioral context improves fraud detection compared with models that treat each transaction independently.

### Success Criteria

The primary evaluation metrics will be:

- Area under the precision-recall curve, or AUPRC
- Fraud-class recall
- Fraud-class precision
- Fraud-class F1-score
- False-positive rate
- Recall at a fixed review rate or fixed false-positive rate

ROC-AUC may be reported as a secondary metric, but AUPRC will be emphasized because it is more informative for highly imbalanced classification.

A successful outcome by December would include:

1. A reproducible data-processing and modelling pipeline.
2. A strong conventional baseline model.
3. At least one temporal or deep-learning model evaluated using a chronological split.
4. Evidence showing whether temporal context improves fraud detection over static transaction features.
5. An interpretable fraud-risk output that identifies the factors contributing to high-risk predictions.
6. A demonstration prototype capable of scoring sample transactions.
7. Clear documentation of model limitations, data leakage risks, and possible deployment considerations.
   
### Project Milestones
Use these milestones to guide your work. Your team will create a GitHub Projects board to track tasks within each milestone.

| Month | Milestone | Key Activities |
|---|---|---|
| September | Data understanding, preparation, and baseline development | Review the IEEE-CIS dataset, relevant research papers, and existing fraud-detection approaches. Define the binary target, business objective, assumptions, and evaluation framework. Clean and merge transaction and identity data. Analyze missing values, categorical variables, feature distributions, temporal patterns, and class imbalance. Create a chronological training, validation, and testing strategy.<br><br>Develop initial baseline models, such as logistic regression, random forest, LightGBM, or XGBoost. Establish baseline results using fraud-focused evaluation metrics. |
| October | Building Deep Neural Networks | Create leakage-safe behavioral features using only information available before each transaction. Temporal and sequential features are optional and should be explored only if reliable entity groupings and transaction sequences can be constructed from the data. Develop a deep neural-network baseline using an MLP, with appropriate preprocessing and categorical embeddings where useful. Address class imbalance using methods such as class weighting and focal loss. Perform feature-selection and hyperparameter experiments, and compare tree-based models, static neural networks, and, where feasible, temporal deep-learning models. |
| November | Evaluation, explainability, and prototype delivery | Finalize model selection using the chronological validation and test sets. Evaluate model performance across fraud recall, precision, false-positive rate, and operational review volume. Analyze errors and identify fraud cases missed by the model. Apply explainability techniques such as SHAP to identify important transaction and behavioral factors. Test model robustness across different time periods and transaction segments. Build a lightweight dashboard or API prototype that returns a fraud-risk score and supporting explanations. Prepare final technical documentation, presentation, and recommendations for future development. |

> **Note for the team:** Please create a GitHub Projects board in this repository to break these milestones into weekly tasks. Go to the **Projects** tab → **New project** → Choose **Board** → Add columns for each month.

---

## 📊 Dataset
**Name and Source:** IEEE-CIS Fraud Detection Dataset (Kaggle)  
**Format:** CSV/TSV  
**Size:** 5gb to 10gb  
**Location:** https://www.kaggle.com/c/ieee-fraud-detection/data

### Key Details
- **Brief description of what's in the data:** The IEEE-CIS Fraud Detection dataset contains real-world e-commerce transaction data provided for the purpose of developing models that can distinguish fraudulent transactions from legitimate ones. The dataset is divided mainly into two groups of files: transaction data and identity data. The target variable is 'isFraud'. The transaction data contains information directly related to each purchase. Important variables include TransactionID, transaction amount, transaction time, product category, card-related information, billing or address-related variables, email domains, and several groups of anonymized features. For example, variables beginning with card describe characteristics associated with the payment card, while variables such as P_emaildomain and R_emaildomain relate to purchaser and recipient email domains. The identity data provides additional information about some transactions and can be connected to the transaction table using TransactionID. These features include device and browser-related information, such as device type, device details, operating-system or browser characteristics, as well as several anonymized identity variables. Not every transaction has a corresponding identity record, so this part of the dataset contains a substantial amount of missing data. The main outcome variable is isFraud. A value of 1 indicates that the transaction was identified as fraudulent, while a value of 0 represents a legitimate transaction. Fraudulent transactions make up a relatively small portion of the observations, so the dataset is highly imbalanced. This is an important consideration when building predictive models because simply predicting most transactions as legitimate could result in high overall accuracy while still performing poorly at detecting actual fraud.

- **Data Size**: The IEEE-CIS Fraud Detection training dataset contains 590,540 transactions collected over roughly a six-month period. Of these, 20,663 transactions are labeled as fraudulent, while approximately 569,877 are legitimate. This means fraud represents only about 3.5% of the transactions, making the dataset highly imbalanced.

- **Any known limitations or preprocessing needed:** One important limitation of the IEEE-CIS Fraud Detection dataset is that the official test data does not contain the isFraud target variable. Because the fraud labels for the Kaggle test set are not available, we cannot use that dataset to independently evaluate model performance. Therefore, for this analysis, we will rely on the labeled training data and divide it into separate subsets for model training, validation, and final blind testing. The blind test set will be kept completely separate during model development and used only for the final evaluation. Another important consideration is the time variable, TransactionDT. This variable represents the number of seconds from an undisclosed reference point rather than an actual calendar date. Although we cannot identify the exact date of each transaction, TransactionDT still provides the chronological ordering of transactions. This is especially important in fraud detection because fraud patterns and customer behavior can change over time. For this reason, the blind test set should be constructed as an out-of-time sample rather than by randomly selecting transactions from the full dataset. Earlier transactions can be used for training and validation, while the most recent portion of the labeled data is held out as the blind test set. This better reflects a real-world deployment scenario, where a fraud model is trained using historical transactions and then applied to transactions that occur in the future. It also provides a more realistic assessment of how well the model generalizes when fraud patterns evolve over time.

- **Data Description** The IEEE-CIS Fraud Detection dataset contains a very large number of features describing the transaction itself, the payment instrument, customer or address information, historical behavior, and device or identity characteristics. The transaction and identity tables can be joined using TransactionID, although identity information is available only for a subset of transactions. Many of the variables have been anonymized for privacy and commercial reasons, so their exact business definitions are not always available.
   - TransactionID is the unique identifier for each transaction. It is mainly used to identify records and to join the transaction table with the identity table. It should generally not be interpreted as a meaningful behavioral feature on its own, although its ordering may indirectly reflect how observations were recorded.
   - TransactionDT represents the time of the transaction as the number of seconds from an undisclosed reference point. It therefore preserves the chronological ordering of transactions but does not provide an actual calendar date. This feature is especially important for our analysis because it allows us to create time-based training, validation, and blind test samples. Rather than randomly splitting transactions, we can reserve the most recent transactions as an out-of-time blind set, which better reflects how a fraud model would be used in production.
   - TransactionAmt records the monetary value of the transaction. Transaction amount can be highly informative in fraud detection because fraudulent activity may have different spending patterns from legitimate transactions. For example, unusually large amounts, very small test transactions, or amounts that are unusual relative to other transactions associated with the same card or customer may provide useful fraud signals.
   - ProductCD is a categorical variable representing the type or category of product associated with the transaction. The values are anonymized rather than being descriptive product names. Nevertheless, fraud rates can differ substantially across products or transaction channels, so this variable can help the model distinguish higher-risk transaction types from lower-risk ones.
   - The card1 through card6 variables contain anonymized information about the payment card. According to the competition documentation, these features represent card-related characteristics such as card type, card category, issuer or bank information, and related attributes, although the precise meaning of every field is not disclosed. These variables can be particularly useful because fraud risk may differ across card issuers, card types, or groups of cards.
   - The addr1 and addr2 variables contain address-related information. addr1 represents the purchaser's billing region, while addr2 represents the billing country. These fields do not provide the actual address and are anonymized. Address information can still be useful because geographic patterns may help identify unusual transactions, particularly when combined with card, email, or device information.
   - The dist1 and dist2 variables represent distance-related measures. Their exact calculation is intentionally not fully disclosed, but they capture distances associated with transaction-related information, including relationships involving billing or other transaction attributes. Large or unusual distances may potentially indicate that the transaction is occurring under circumstances that differ from the customer's normal behavior.
   - P_emaildomain and R_emaildomain contain the purchaser and recipient email domains. Examples may correspond to common email providers, although the data is intended to identify domains rather than individual email addresses. These variables can provide useful information about the relationship between the purchaser and recipient and may also help detect unusual or higher-risk combinations of transaction characteristics.
   - The C1 through C14 variables are anonymized counting features. They represent different types of counts associated with a payment card or other transaction-related entities. Although the exact definitions are not publicly disclosed, they can capture aspects of transaction history or activity frequency. This makes them useful behavioral variables because a transaction occurring in the context of unusually high or low historical activity may have a different fraud risk.
   - The D1 through D15 variables are anonymized time-difference or timedelta features. They measure time intervals associated with previous transactions or other events. For example, these features may help capture how recently a card, customer, address, or related entity was previously observed. Time-since-event variables are particularly useful for fraud detection because unusual transaction frequency or sudden changes in activity can indicate suspicious behavior.
   - The M1 through M9 variables are categorical match features. They indicate whether certain pieces of transaction information match one another. The precise comparison behind each variable is anonymized, but conceptually they describe consistency between different pieces of transaction information. Mismatches can potentially be important fraud indicators because legitimate customers often exhibit relatively consistent combinations of billing, shipping, card, and account information
   - The V1 through V339 variables make up the largest feature group in the transaction dataset. These are anonymized engineered variables created by Vesta and are believed to describe various relationships, counts, rankings, and transaction characteristics. Their individual business definitions are not disclosed. Even though they are difficult to interpret directly, they can contain substantial predictive information and are therefore commonly included in machine-learning models. The large number of V variables also contributes considerably to the dimensionality of the dataset.
   - The identity table adds another set of variables called id_01 through id_38. These describe characteristics associated with the identity or digital environment involved in the transaction. Some are numerical while others are categorical. Their exact definitions are mostly anonymized, but they can capture information such as account or device characteristics, network-related information, and other indicators available during authentication. Identity information is not present for every transaction, so missingness itself can also become potentially informative.
   - DeviceType provides a broader classification of the device used for the transaction, such as desktop or mobile. Device type can be useful because transaction behavior and fraud patterns may differ across channels. For example, fraud characteristics observed on mobile devices may not necessarily be the same as those observed on desktop transactions.
   - DeviceInfo contains more detailed information about the device environment. Depending on the observation, it may include information related to the device, operating system, browser, or platform. This variable has many different values and can therefore require preprocessing or grouping before being used effectively in a model. Device information can be valuable for detecting cases where the device used for a transaction differs from typical legitimate behavior.
  
---
## 🛠️ Suggested Approach

**ML Problem Type:** Classification, Deep Learning / Neural Networks, Try to Explore the possibility of Time Series Approaches if time permits (not a necessary) 

**Recommended Libraries:**
- For exploring tree-based boosting algorithms in Python, the most useful libraries are XGBoost, LightGBM, and CatBoost. XGBoost is widely used for structured and tabular datasets and provides strong performance, regularization options, feature importance measures, and support for class imbalance. LightGBM is especially suitable for large datasets because it is designed for fast and memory-efficient gradient boosting. CatBoost is another strong option, particularly when the data contains many categorical variables, because it can handle categorical features more directly and reduces the amount of manual encoding required.
- Scikit-learn should also be included as a core library because it provides the broader machine-learning workflow around these models. It can be used for preprocessing, train-validation splitting, feature transformations, pipelines, baseline tree models, hyperparameter tuning, and evaluation metrics such as precision, recall, F1-score, ROC-AUC, and precision-recall AUC. It also integrates well with XGBoost, LightGBM, and CatBoost.
- For deep learning, PyTorch or TensorFlow/Keras are the main choices. PyTorch provides a flexible framework for building custom neural networks and is especially useful if we want to experiment with embeddings for categorical variables, multilayer neural networks, or more advanced architectures for tabular data. TensorFlow with Keras provides a higher-level interface that can make it easier to build and train feed-forward neural networks quickly. For this project, a standard multilayer perceptron could be used as an initial deep-learning benchmark, with numerical features normalized and categorical variables represented using embeddings or encoded features.
- For more specialized deep learning on tabular data, libraries such as PyTorch Tabular or pytorch-tabnet can also be explored. These provide implementations of models designed specifically for structured data, including architectures such as TabNet and models that combine categorical embeddings with dense numerical features. They can be useful for comparing traditional boosting methods with more modern neural-network approaches.
- Supporting libraries such as pandas and NumPy will be needed for data manipulation and numerical processing, while Matplotlib can be used for exploratory analysis and model-performance visualization. For hyperparameter optimization, Optuna is particularly useful because it works well with XGBoost, LightGBM, CatBoost, and deep-learning models and can efficiently search large parameter spaces.

**Evaluation Metrics:**
- Accuracy measures the percentage of all transactions that are classified correctly. Since fraud is rare, accuracy can be misleading because a model may achieve a high value simply by predicting most transactions as legitimate.
- Precision measures the percentage of transactions flagged as fraud that are actually fraudulent. High precision reduces unnecessary investigations and minimizes the number of legitimate customers incorrectly flagged.
- Recall, or True Positive Rate, measures the percentage of actual fraudulent transactions detected by the model. High recall is important because missed fraud directly represents financial and operational risk.
- F1-score combines precision and recall into a single metric. It is useful when we want a balanced measure that considers both missed fraud and false fraud alerts.
- False Positive Rate, or FPR, measures the percentage of legitimate transactions incorrectly classified as fraud. This is particularly important in payments because even a small FPR can create a large number of false alerts when transaction volumes are high.
- TPR at a fixed FPR measures how much fraud can be detected while keeping false positives within an acceptable level. For example, we can compare models based on the percentage of fraud detected at FPR levels such as 0.1%, 0.5%, or 1%.
- ROC-AUC measures how well the model ranks fraudulent transactions above legitimate transactions across all possible thresholds. It provides a broad measure of discrimination, although it can look relatively strong in highly imbalanced datasets.
- PR-AUC measures the relationship between precision and recall across thresholds. Because it focuses more directly on the minority fraud class, it is particularly useful for evaluating fraud detection models.
- KS statistic measures the maximum separation between the fraud and non-fraud populations. It can be expressed as the maximum difference between TPR and FPR across thresholds. A higher KS indicates stronger separation between fraudulent and legitimate transactions.
- Confusion matrix shows the numbers of true positives, false positives, true negatives, and false negatives. It gives a straightforward operational view of how many fraud cases are detected, missed, or incorrectly flagged.
- In addition to measuring fraud detection by transaction count, we should also evaluate performance based on the fraud dollar amount captured. A model may detect many fraudulent transactions but still miss a small number of very high-value fraud cases, which could be much more costly to the business.
- A useful approach is to rank transactions by predicted fraud risk and divide them into score buckets or percentiles, such as the top 1%, 5%, 10%, and 20% of transactions. For each bucket, we can calculate both the percentage of fraudulent transactions captured, and the percentage of total fraudulent dollar amount captured. This shows whether the highest-risk scores are successfully concentrating not only fraud cases, but also the most financially significant fraud. For example, if the top 5% highest-risk transactions contain 60% of fraud cases but 75% of the total fraud amount, that would indicate that the model is particularly effective at identifying higher-value fraud. This type of fraud amount capture curve is important from a business perspective because it connects model performance directly to potential financial loss prevention.

---

## 📚 Resources to Get Started

The following resources will help your team understand the problem space and potential technical approaches for this project:

**Background Reading:**
- [e.g., Link to an article or blog post about the problem domain]
- [e.g., Link to an industry report or case study]

**Technical Tutorials:**
- For **behavioral and temporal feature engineering**, Pandas is the main library to use. Useful functions include `groupby`, `shift`, and `rolling` for creating transaction velocity, time since previous transaction, rolling amount statistics, and similar leakage-safe features.
   - https://pandas.pydata.org/docs/reference/groupby.html
   - https://pandas.pydata.org/docs/reference/api/pandas.api.typing.SeriesGroupBy.shift.html
   - https://pandas.pydata.org/docs/user_guide/window.html
- For a **multilayer perceptron (MLP) / standard neural network**, PyTorch provides the core neural-network modules needed to build feed-forward models.
   - https://docs.pytorch.org/docs/stable/nn.html
- For **categorical embeddings**, PyTorch provides `torch.nn.Embedding`, which can be used to represent features such as card identifiers, device information, email domains, or product categories.
   - https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html
- For **GRU models**, PyTorch provides a built-in recurrent GRU layer that can be used if reliable transaction sequences can be created.
   - https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html
- For **LSTM models**, PyTorch provides `torch.nn.LSTM`, which can capture longer-term dependencies across transaction sequences.
   - https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html
- For a **CNN-GRU model**, a one-dimensional convolution can first extract local patterns in a sequence and then feed those representations into a GRU.
   - https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
   - https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html
- For **Transformer-based models**, PyTorch provides Transformer encoder and multi-head attention components. These would be optional and should be explored only if meaningful sequential data can be constructed.
   - https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html
   - [https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
- For **class weighting in neural networks**, PyTorch's `BCEWithLogitsLoss` supports the `pos_weight` parameter, which can give greater weight to fraudulent transactions.
   - https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
- For **focal loss**, TensorFlow/Keras provides an implementation of binary focal cross-entropy, which can help focus training on harder examples in an imbalanced dataset.
   - https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryFocalCrossentropy
- For **feature selection**, Scikit-learn provides univariate selection, recursive feature elimination, and model-based selection methods.
   - https://scikit-learn.org/stable/modules/feature_selection.html
- For **hyperparameter tuning**, Scikit-learn provides both grid search and randomized search.
   - https://scikit-learn.org/stable/modules/grid_search.html
   - https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
- For more advanced **hyperparameter optimization**, Optuna can be used with neural networks, XGBoost, LightGBM, and other models.
   - https://optuna.readthedocs.io/
- For **XGBoost**, the official documentation covers model parameters, tuning, categorical features, and handling class imbalance.
   - https://xgboost.readthedocs.io/en/stable/
   - https://xgboost.readthedocs.io/en/stable/parameter.html
   - https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html
   - https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html
- For **LightGBM**, the official documentation covers `LGBMClassifier`, model parameters, class imbalance options, and parameter tuning.
   -  https://lightgbm.readthedocs.io/
   -  https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html
   -  https://lightgbm.readthedocs.io/en/latest/Parameters.html
   -  https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
- For **CatBoost**, the official documentation covers the classifier and native handling of categorical variables.
   - https://catboost.ai/docs/
   - https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier
   - https://catboost.ai/docs/en/features/categorical-features
- For **evaluation metrics** such as ROC-AUC, precision, recall, F1-score, confusion matrix, and precision-recall curves, Scikit-learn provides the main reference documentation.
   - https://scikit-learn.org/stable/modules/model_evaluation.html
   - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
   - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
   - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html
   - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
For this project, the main modeling stack would be **LightGBM, XGBoost, CatBoost, and a MLP**, with GRU, LSTM, CNN-GRU, and Transformer models treated as optional extensions if the data supports reliable temporal sequences.

**Code Examples:**
- [e.g., Link to a relevant GitHub repo]
- [e.g., Link to a sample implementation or starter code]

**Other:**
- [Links to any additional resources — e.g., papers, videos, podcasts, etc.]

*Feel free to explore beyond these, and share anything interesting you find with me!*

---

## 🤝 How We'll Work Together

**Official check-ins:** During our biweekly 45-minute AI Studio Lab Section meeting block (2nd and 4th week of every month)

 **Other ways to reach out to me with questions:** 
* [e.g., Your team's channel within Break Through Tech’s Discord space]
* [e.g., Email; please copy your teammates and AI Studio Coach]
* [e.g., Request a team check-in on Zoom]
* [Note: I will aim to respond within 48 hours. Please reach out to your AI Studio Coach with urgent questions.]

> 💡 **Challenge Advisor: Please update the above based on your availability and preference. If you are not able to answer questions or meet with fellows outside of the biweekly Lab Section check-ins, simply write in "N/A (only available during the official check-in times)"**

**Recommended free coding / collaboration tools**
* […]
* […]

---

## 🚀 Getting Started

1. **Review this overview document** and note any questions for our first meeting
2. **Begin reviewing the dataset** using the link above
3. **Read the GitHub Projects documentation** [here](https://docs.github.com/en/issues/planning-and-tracking-with-projects/learning-about-projects/about-projects)

I’m excited to work with you!

---

## ❓ Questions?

Please bring any questions to our first meeting during the week of August 24th (Break Through Tech’s Bridge to Studio - Session C). 
