# =============================================================================
# Automated AWS ECS (Fargate) Deployment Script (PowerShell)
# =============================================================================

param (
    [string]$AwsRegion = $env:AWS_REGION,
    [string]$AwsAccountId = $env:AWS_ACCOUNT_ID
)

if (-not $AwsRegion) {
    $AwsRegion = "us-east-1"
}

if (-not $AwsAccountId) {
    try {
        $AwsAccountId = (aws sts get-caller-identity --query Account --output text 2>$null)
    } catch {
        $AwsAccountId = $null
    }
}

if (-not $AwsAccountId) {
    Write-Error "AWS Account ID could not be determined. Please configure your AWS credentials or pass -AwsAccountId <id>."
    exit 1
}

$RepoName = "textclassify-api"
$ClusterName = "textclassify-cluster"
$ServiceName = "textclassify-service"
$Tag = "v2.0.0"
$EcrUri = "$AwsAccountId.dkr.ecr.$AwsRegion.amazonaws.com/$RepoName"

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host " Deploying TextClassify AI API to AWS ECS Fargate" -ForegroundColor Green
Write-Host " Account: $AwsAccountId"
Write-Host " Region:  $AwsRegion"
Write-Host " ECR URI: $EcrUri:$Tag"
Write-Host "=================================================================" -ForegroundColor Cyan

# 1. Login to AWS ECR
Write-Host "[Step 1/5] Authenticating Docker with Amazon ECR..." -ForegroundColor Yellow
$loginPwd = (aws ecr get-login-password --region $AwsRegion)
$loginPwd | docker login --username AWS --password-stdin "$AwsAccountId.dkr.ecr.$AwsRegion.amazonaws.com"

# 2. Create ECR repository if it doesn't exist
Write-Host "[Step 2/5] Verifying ECR repository..." -ForegroundColor Yellow
$repoCheck = aws ecr describe-repositories --repository-names $RepoName --region $AwsRegion 2>$null
if (-not $repoCheck) {
    aws ecr create-repository --repository-name $RepoName --region $AwsRegion
}

# 3. Build & tag Docker image
Write-Host "[Step 3/5] Building Docker image..." -ForegroundColor Yellow
docker build -t "$RepoName:$Tag" -f Dockerfile.api .
docker tag "$RepoName:$Tag" "$EcrUri:$Tag"
docker tag "$RepoName:$Tag" "$EcrUri:latest"

# 4. Push to ECR
Write-Host "[Step 4/5] Pushing image to Amazon ECR..." -ForegroundColor Yellow
docker push "$EcrUri:$Tag"
docker push "$EcrUri:latest"

# 5. Update ECS Service
Write-Host "[Step 5/5] Updating ECS task definition and service..." -ForegroundColor Yellow
$taskDefContent = Get-Content -Raw "ecs-task-definition.json"
$taskDefContent = $taskDefContent -replace "<AWS_ACCOUNT_ID>", $AwsAccountId -replace "<AWS_REGION>", $AwsRegion
$tempTaskDef = "ecs-task-definition-resolved.json"
Set-Content -Path $tempTaskDef -Value $taskDefContent

aws ecs register-task-definition --cli-input-json "file://$tempTaskDef" --region $AwsRegion
Remove-Item -Path $tempTaskDef -Force

aws ecs update-service --cluster $ClusterName --service $ServiceName --force-new-deployment --region $AwsRegion

Write-Host "=================================================================" -ForegroundColor Green
Write-Host " AWS ECS DEPLOYMENT INITIATED!" -ForegroundColor Green
Write-Host " Monitor deployment in AWS Console -> ECS -> Clusters -> $ClusterName" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Green
