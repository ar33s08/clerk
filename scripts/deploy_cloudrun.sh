#!/usr/bin/env bash
# Cloud Run deploy (us-central1). Requires: gcloud auth + a corpus snapshot in ./data
set -euo pipefail
docker build -t clerk .
gcloud run deploy clerk --source . --region us-central1 \
  --allow-unauthenticated --memory 2Gi --cpu 2 \
  --min-instances 0 --max-instances 2 --timeout 300s \
  --set-env-vars "CLERK_LLM_BASE_URL=${CLERK_LLM_BASE_URL},CLERK_LLM_MODEL=${CLERK_LLM_MODEL:-qwen3.8-flash-next},CLERK_API_TOKEN=${CLERK_API_TOKEN:?}"
echo "note: serverless cold start rebuilds the index from /data; for >10k chunks move to pgvector first (docs/ARCHITECTURE.md)"
