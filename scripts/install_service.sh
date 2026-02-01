#!/usr/bin/env bash
set -euo pipefail

# Usage: sudo ./scripts/install_service.sh /path/to/contrib/systemd/chess_bots_runner.service

UNIT_SOURCE=${1:-contrib/systemd/chessbots.service}
UNIT_NAME=$(basename "$UNIT_SOURCE")

if [[ $EUID -ne 0 ]]; then
  echo "This script installs the systemd unit and must be run with sudo."
  echo "Example: sudo $0 $UNIT_SOURCE"
  exit 1
fi

if [[ ! -f "$UNIT_SOURCE" ]]; then
  echo "Unit file not found: $UNIT_SOURCE" >&2
  exit 1
fi

cp "$UNIT_SOURCE" /etc/systemd/system/"$UNIT_NAME"
systemctl daemon-reload
systemctl enable --now "$UNIT_NAME"

echo "Installed and started $UNIT_NAME"
