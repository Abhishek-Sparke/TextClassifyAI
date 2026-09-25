#!/usr/bin/env bash
# =============================================================================
# Automated AWS ECS (Fargate) Deployment Script for TextClassify AI API
# =============================================================================

set -e

AWS_REGION=${AWS_REGION:-"us-east-1"}
AWS_ACCOUNT_ID=${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text 2>/dev/null)}
REPO_NAME="textclassify-api"
CLUSTER_NAME="textclassify-cluster"
SERVICE_NAME="textclassify-service"
TAG="v2.0.0"

if [ -z "$AWS_ACCOUNT_ID" ]; then
    echo "Error: AWS_ACCOUNT_ID could not be determined. Please configure your AWS credentials."
    exit 1
fi

ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${REPO_NAME}"

echo "================================================================="
echo " Deploying TextClassify AI API to AWS ECS Fargate"
echo " Account: $AWS_ACCOUNT_ID"
echo " Region:  $AWS_REGION"
echo " ECR URI: $ECR_URI:$TAG"
echo "================================================================="

# 1. Login to AWS ECR
echo "[Step 1/5] Authenticating Docker with Amazon ECR..."
aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

# 2. Create ECR repository if it doesn't exist
echo "[Step 2/5] Verifying ECR repository..."
aws ecr describe-repositories --repository-names "$REPO_NAME" --region "$AWS_REGION" >/dev/null 2>&1 || \
    aws ecr create-repository --repository-name "$REPO_NAME" --region "$AWS_REGION"

# 3. Build & tag Docker image
echo "[Step 3/5] Building Docker image..."
docker build -t "$REPO_NAME:$TAG" -f Dockerfile.api .
docker tag "$REPO_NAME:$TAG" "$ECR_URI:$TAG"
docker tag "$REPO_NAME:$TAG" "$ECR_URI:latest"

# 4. Push to ECR
echo "[Step 4/5] Pushing image to Amazon ECR..."
docker push "$ECR_URI:$TAG"
docker push "$ECR_URI:latest"

# 5. Update ECS Service
echo "[Step 5/5] Updating ECS service to deploy new task..."
sed -i "s|<AWS_ACCOUNT_ID>|${AWS_ACCOUNT_ID}|g" ecs-task-definition.json
sed -i "s|<AWS_REGION>|${AWS_REGION}|g" ecs-task-definition.json

aws ecs register-task-definition --cli-input-json file://ecs-task-definition.json --region "$AWS_REGION"
aws ecs update-service --cluster "$CLUSTER_NAME" --service "$SERVICE_NAME" --force-new-deployment --region "$AWS_REGION"

echo "================================================================="
echo " AWS ECS DEPLOYMENT INITIATED!"
echo " Monitor deployment in AWS Console -> ECS -> Clusters -> $CLUSTER_NAME"
echo "================================================================="
