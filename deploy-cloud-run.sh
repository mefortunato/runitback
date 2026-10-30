#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  source "$ROOT_DIR/.env"
  set +a
fi

SERVICE="${CLOUD_RUN_SERVICE:-runitback}"
REGION="${CLOUD_RUN_REGION:-us-central1}"
PROJECT="${GOOGLE_CLOUD_PROJECT:-$(gcloud config get-value project 2>/dev/null || true)}"
AUTH_FLAGS=()

if [[ "${CLOUD_RUN_ALLOW_UNAUTHENTICATED:-false}" == "true" ]]; then
  AUTH_FLAGS+=(--allow-unauthenticated)
fi

if [[ -z "$PROJECT" || "$PROJECT" == "(unset)" ]]; then
  echo "Set GOOGLE_CLOUD_PROJECT or configure a default with 'gcloud config set project PROJECT_ID'." >&2
  exit 1
fi

cd "$ROOT_DIR"
gcloud run deploy "$SERVICE" \
  --source . \
  --project "$PROJECT" \
  --region "$REGION" \
  --max 1 \
  "${AUTH_FLAGS[@]}"