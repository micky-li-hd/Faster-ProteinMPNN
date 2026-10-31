#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 3 ]; then
  echo "Usage: bash scripts/run_six_epoch.sh DATA_ROOT SHARED_LOCK GPU_INDEX" >&2
  exit 2
fi
repo="$(cd "$(dirname "$0")/.." && pwd)"
data="$1"
lock="$2"
gpu="$3"
py="${PYTHON:-python3.12}"
series="$repo/runs-six-epoch"
mkdir "$series"
cd "$repo/scripts"
for arm in reference1 candidate reference2; do
  variant=reference
  if [ "$arm" = candidate ]; then variant=candidate; fi
  "$py" train.py --variant "$variant" --recipe "$repo/six_epoch_recipe.json" --data "$data" --out "$series/$arm" --gpu "$gpu" --timeout 7200 --lock "$lock"
  "$py" audit_run.py "$series/$arm"
done
