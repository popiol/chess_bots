#!/usr/bin/env bash
set -euo pipefail

python -m venv .venv
source .venv/bin/activate

set -a
source .env
set +a

python -m pip install --upgrade pip
pip install -r requirements.txt

python -m playwright install

./scripts/setup_db.sh
python -m src.db.setup_db
