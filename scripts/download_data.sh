#!/usr/bin/env bash
# Download the IEEE-CIS Fraud Detection dataset into data/.
#
# Prerequisites:
#   1. Accept the competition rules (required, or the download 403s):
#        https://www.kaggle.com/competitions/ieee-fraud-detection/rules
#   2. Authenticate the CLI once, either:
#        .venv/bin/kaggle auth login          # OAuth browser flow (recommended)
#      or generate a token at https://www.kaggle.com/settings/api and:
#        export KAGGLE_API_TOKEN=xxxxxxxx     # or save it to ~/.kaggle/access_token
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KAGGLE="$REPO_ROOT/.venv/bin/kaggle"
DATA_DIR="$REPO_ROOT/data"

"$KAGGLE" competitions download -c ieee-fraud-detection -p "$DATA_DIR"
unzip -o "$DATA_DIR/ieee-fraud-detection.zip" -d "$DATA_DIR"
rm "$DATA_DIR/ieee-fraud-detection.zip"

echo "Done. Files in data/:"
ls -lh "$DATA_DIR"
