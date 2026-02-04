#!/usr/bin/env bash
set -euo pipefail

apt-get update
apt-get install -y postgresql postgresql-contrib

set -a
source .env
set +a

python -m pip install --upgrade pip
pip install -r requirements.txt

python -m playwright install-deps
python -m playwright install

./scripts/setup_db.sh
python -m src.db.setup_db
