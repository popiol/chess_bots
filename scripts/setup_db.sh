#!/usr/bin/env bash
set -euo pipefail

required=(
  CHESS_BOTS_DB_NAME
  CHESS_BOTS_DB_USER
  CHESS_BOTS_DB_PASSWORD
)

missing=()
for key in "${required[@]}"; do
  if [[ -z "${!key:-}" ]]; then
    missing+=("$key")
  fi
done

if [[ ${#missing[@]} -gt 0 ]]; then
  echo "Missing required env vars: ${missing[*]}" >&2
  exit 1
fi

db_name="$CHESS_BOTS_DB_NAME"
db_user="$CHESS_BOTS_DB_USER"
db_pass="$CHESS_BOTS_DB_PASSWORD"

sudo -u postgres psql -v ON_ERROR_STOP=1 <<SQL
DO \$\$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '${db_user}') THEN
    CREATE ROLE ${db_user} WITH LOGIN PASSWORD '${db_pass}';
  END IF;
END
\$\$;
SQL

# Create the database outside of any transaction/function (CREATE DATABASE
# is not allowed inside a transaction block). Use createdb if the database
# does not already exist.
if ! sudo -u postgres psql -tAc "SELECT 1 FROM pg_database WHERE datname='${db_name}'" | grep -q 1; then
  sudo -u postgres createdb -O "${db_user}" "${db_name}"
fi

# Ensure privileges
sudo -u postgres psql -v ON_ERROR_STOP=1 -c "GRANT ALL PRIVILEGES ON DATABASE ${db_name} TO ${db_user};"
