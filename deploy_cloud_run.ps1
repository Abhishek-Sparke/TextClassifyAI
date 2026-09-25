# =============================================================================
# Automated Google Cloud Run Deployment Script (PowerShell)
# =============================================================================

param (
    [string]$ProjectId = $env:GCP_PROJECT_ID,
    [string]$Region = "us-central1"
)

if (-not $ProjectId) {
    $ProjectId = (gcloud config get-value project 2>$null)
}

if (-not $ProjectId) {
    Write-Error "GCP Project ID not found. Set environment variable GCP_PROJECT_ID or pass -ProjectId <id>"
    exit 1
}

$ServiceName = "textclassify-api"
$ImageTag = "gcr.io/$ProjectId/${ServiceName}:v2.0.0"

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host " Deploying TextClassify AI API to Google Cloud Run" -ForegroundColor Green
Write-Host " Project: $ProjectId"
Write-Host " Region:  $Region"
Write-Host " Service: $ServiceName"
Write-Host " Image:   $ImageTag"
Write-Host "=================================================================" -ForegroundColor Cyan

# 1. Enable services
Write-Host "[Step 1/4] Enabling Cloud Run & Build APIs..." -ForegroundColor Yellow
gcloud services enable run.googleapis.com cloudbuild.googleapis.com containerregistry.googleapis.com --project=$ProjectId

# 2. Build image
Write-Host "[Step 2/4] Submitting build to Cloud Build..." -ForegroundColor Yellow
gcloud builds submit --project=$ProjectId --tag=$ImageTag -f Dockerfile.api .

# 3. Deploy
Write-Host "[Step 3/4] Deploying to Cloud Run..." -ForegroundColor Yellow
gcloud run deploy $ServiceName `
    --project=$ProjectId `
    --image=$ImageTag `
    --region=$Region `
    --platform="managed" `
    --allow-unauthenticated `
    --memory="1Gi" `
    --cpu="1" `
    --port="8000"

# 4. URL
$ServiceUrl = (gcloud run services describe $ServiceName --project=$ProjectId --region=$Region --format="value(status.url)")

Write-Host "=================================================================" -ForegroundColor Green
Write-Host " DEPLOYMENT SUCCESSFUL!" -ForegroundColor Green
Write-Host " API URL:      $ServiceUrl" -ForegroundColor Cyan
Write-Host " Swagger Docs: $ServiceUrl/docs" -ForegroundColor Cyan
Write-Host " Health Check: $ServiceUrl/health" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next, connect your Vercel frontend by setting the environment variable in Vercel:"
Write-Host "  Key:   VITE_API_URL" -ForegroundColor Yellow
Write-Host "  Value: $ServiceUrl" -ForegroundColor Yellow
