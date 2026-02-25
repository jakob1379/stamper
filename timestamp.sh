#!/usr/bin/env bash
# space-timestamps.sh  (using tee)

set -euo pipefail
FILE=${1:-timestamps.csv}

[[ -f "$FILE" ]] || echo "timestamp" > "$FILE"

echo "Recording timestamps to $FILE (SPACE = record, q = quit)"

while true; do
  read -rsn1
  date -u +"%Y-%m-%dT%H:%M:%SZ" | tee -a "$FILE"
done
