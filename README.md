# Online Payment Fraud Detection

Banking 1B's Break Through Tech AI Studio project with Mastercard explores fraud detection using the high-dimensional IEEE-CIS transaction and identity data.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Accept the IEEE-CIS Fraud Detection competition rules in Kaggle before downloading the data. Choose either download workflow:

- **Notebook / Colab:** run [import_kaggle_data.ipynb](notebooks/import_kaggle_data.ipynb). It supports Kaggle login and, in a local checkout, credentials from environment variables, `.env`, or `~/.kaggle/kaggle.json`. To use `.env`, copy [.env.example](.env.example) and fill in your credentials. Never commit credentials or their notebook output.
- **Local CLI:** authenticate with `.venv/bin/kaggle auth login`, then run `bash scripts/download_data.sh` to extract the competition files into `data/`.

Large datasets, generated features, models, and local credentials are ignored by Git. The two CSVs already tracked in `data/` are retained.

## Analysis workflow

1. [Import Kaggle data](notebooks/import_kaggle_data.ipynb): authenticate, download, list the files, and read the raw tables.
2. [Combined EDA](notebooks/BTT_Mastercard_EDA_Combined.ipynb): the team's consolidated exploratory analysis, including time shifts, missingness, feature signals, and account structure. This supersedes `BTT-Mastercard-EDA_Done.ipynb`.
3. [Final feature engineering](notebooks/BTT_Mastercard_Feature_Engineering_Final.ipynb): combined team features, chronological train/validation/holdout splits, leakage checks, and tree and neural-network input tables. Set `CONFIG["data_path"]` to the local `data/` folder to use the CLI download, or leave it as `None` to download with KaggleHub. Generated tables go under `outputs/feature_engineering/<run_name>/`.

Run each analysis notebook from top to bottom. Full analysis requires the complete competition dataset; downloading data alone does not run the analysis or establish model performance.

## Project documentation

- [Challenge overview](Challenge-Project-Overview.md)
- [Getting started for fellows](Getting-Started-for-Fellows.md)
- [Project plan](markdowns/plan.md)
- [EDA summary notes](markdowns/EDA_Summary_Notes_1.md)
- [Feature engineering explanation](markdowns/Feature_Engineering_Explained.md)
- [Feature engineering summary](markdowns/BTT_Mastercard_Feature_Engineering_Summary.pdf)

Reports and presentations live in `markdowns/`; executable notebooks live in `notebooks/`. Duplicate notebook and report copies have been consolidated. Earlier notebook versions remain available in Git history.
