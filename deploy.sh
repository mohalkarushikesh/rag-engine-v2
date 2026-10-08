#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-command deploy: build the image, push to ECR, apply Terraform.
#
# Prereqs on the machine running this (NOT the offline corp box):
#   - Docker, AWS CLI v2, Terraform >= 1.5
#   - AWS credentials with rights to create ECR/IAM/App Runner
#   - Internet access (docker build downloads the HF models)
#   - Bedrock model access enabled for the embed + chat models in your region
#
# Usage:
#   AWS_REGION=us-east-1 IMAGE_TAG=$(git rev-parse --short HEAD) ./deploy.sh
#
# Use a UNIQUE IMAGE_TAG per deploy (e.g. the git sha) so App Runner picks up
# the new image -- reusing "latest" won't trigger a redeploy.
# ---------------------------------------------------------------------------
set -euo pipefail

AWS_REGION="${AWS_REGION:-us-east-1}"
SERVICE_NAME="${SERVICE_NAME:-rag-app}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

cd "$(dirname "$0")"
TF=(terraform -chdir=infra)
VARS=(-var="aws_region=${AWS_REGION}" -var="service_name=${SERVICE_NAME}" -var="image_tag=${IMAGE_TAG}")

echo ">> [1/4] terraform init + create ECR repository"
"${TF[@]}" init -input=false
"${TF[@]}" apply -input=false -auto-approve -target=aws_ecr_repository.app "${VARS[@]}"

ECR_URL="$("${TF[@]}" output -raw ecr_repository_url)"
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"

echo ">> [2/4] docker login to ECR"
aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

echo ">> [3/4] build + push image (${ECR_URL}:${IMAGE_TAG})"
docker build -t "${SERVICE_NAME}:${IMAGE_TAG}" .
docker tag "${SERVICE_NAME}:${IMAGE_TAG}" "${ECR_URL}:${IMAGE_TAG}"
docker push "${ECR_URL}:${IMAGE_TAG}"

echo ">> [4/4] terraform apply (create/update App Runner service)"
"${TF[@]}" apply -input=false -auto-approve "${VARS[@]}"

echo
echo ">> Done. Service URL:"
"${TF[@]}" output -raw service_url
echo