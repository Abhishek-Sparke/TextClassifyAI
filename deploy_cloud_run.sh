#!/usr/bin/env bash
# =============================================================================
# Automated Google Cloud Run Deployment Script for TextClassify AI API
# =============================================================================

set -e

# Configuration variables
PROJECT_ID=${GCP_PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}
REGION=${GCP_REGION:-"us-central1"}
SERVICE_NAME="textclassify-api"
IMAGE_TAG="gcr.io/${PROJECT_ID}/${SERVICE_NAME}:v2.0.0"

if [ -z "$PROJECT_ID" ]; then
    echo "Error: GCP_PROJECT_ID is not set and could not be detected from gcloud."
    echo "Usage: GCP_PROJECT_ID=your-project-id ./deploy_cloud_run.sh"
    exit 1
fi

echo "================================================================="
echo " Deploying TextClassify AI API to Google Cloud Run"
echo " Project:  $PROJECT_ID"
echo " Region:   $REGION"
echo " Service:  $SERVICE_NAME"
echo " Image:    $IMAGE_TAG"
echo "================================================================="

# 1. Enable required Google Cloud APIs
echo "[Step 1/4] Enabling Cloud Run and Cloud Build APIs..."
gcloud services enable run.googleapis.com cloudbuild.googleapis.com containerregistry.googleapis.com --project="$PROJECT_ID"

# 2. Build container image using Google Cloud Build
echo "[Step 2/4] Building container image in Cloud Build..."
gcloud builds submit --project="$PROJECT_ID" --tag="$IMAGE_TAG" -f Dockerfile.api .

# 3. Deploy container to Google Cloud Run
echo "[Step 3/4] Deploying container to Cloud Run..."
gcloud run deploy "$SERVICE_NAME" \
    --project="$PROJECT_ID" \
    --image="$IMAGE_TAG" \
    --region="$REGION" \
    --platform="managed" \
    --allow-unauthenticated \
    --memory="1Gi" \
    --cpu="1" \
    --min-instances="0" \
    --max-instances="10" \
    --port="8000"

# 4. Extract deployed Cloud Run URL
SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" --project="$PROJECT_ID" --region="$REGION" --format="value(status.url)")

echo "================================================================="
echo " DEPLOYMENT SUCCESSFUL!"
echo " Service URL: $SERVICE_URL"
echo " Swagger Docs: ${SERVICE_URL}/docs"
echo " Health Check: ${SERVICE_URL}/health"
echo "================================================================="
echo ""
echo "To connect your Vercel frontend, update your Vercel environment variable:"
echo "  vercel env add VITE_API_URL $SERVICE_URL"
echo "Or in the Vercel Dashboard -> Settings -> Environment Variables:"
echo "  Key:   VITE_API_URL"
echo "  Value: $SERVICE_URL"
echo "================================================================="
