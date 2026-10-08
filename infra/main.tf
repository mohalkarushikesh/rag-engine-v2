data "aws_caller_identity" "current" {}

locals {
  account_id = data.aws_caller_identity.current.account_id
  image_uri  = "${aws_ecr_repository.app.repository_url}:${var.image_tag}"
}

# ---------------------------------------------------------------------------
# Container registry -- App Runner pulls the app image from here.
# ---------------------------------------------------------------------------
resource "aws_ecr_repository" "app" {
  name                 = var.service_name
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

# Keep only the most recent images to control storage cost.
resource "aws_ecr_lifecycle_policy" "app" {
  repository = aws_ecr_repository.app.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 10 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 10
      }
      action = { type = "expire" }
    }]
  })
}

# ---------------------------------------------------------------------------
# IAM: ACCESS role -- lets the App Runner service pull the image from ECR.
# ---------------------------------------------------------------------------
resource "aws_iam_role" "apprunner_access" {
  name = "${var.service_name}-apprunner-access"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "build.apprunner.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "apprunner_ecr" {
  role       = aws_iam_role.apprunner_access.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess"
}

# ---------------------------------------------------------------------------
# IAM: INSTANCE role -- assumed by the running app. This is how the Bedrock
# backend authenticates: no access keys are stored in the image or config.
# ---------------------------------------------------------------------------
resource "aws_iam_role" "apprunner_instance" {
  name = "${var.service_name}-apprunner-instance"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "tasks.apprunner.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

# Least-privilege: only the Bedrock invoke actions the RAG app needs.
resource "aws_iam_role_policy" "bedrock_invoke" {
  name = "${var.service_name}-bedrock-invoke"
  role = aws_iam_role.apprunner_instance.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream",
      ]
      Resource = "arn:aws:bedrock:*::foundation-model/*"
    }]
  })
}

# ---------------------------------------------------------------------------
# Autoscaling -- scale out under load, keep >=1 warm so the index stays built.
# ---------------------------------------------------------------------------
resource "aws_apprunner_auto_scaling_configuration_version" "app" {
  auto_scaling_configuration_name = var.service_name

  max_concurrency = var.max_concurrency
  min_size        = var.min_size
  max_size        = var.max_size
}

# ---------------------------------------------------------------------------
# The App Runner service itself.
# ---------------------------------------------------------------------------
resource "aws_apprunner_service" "app" {
  service_name = var.service_name

  source_configuration {
    authentication_configuration {
      access_role_arn = aws_iam_role.apprunner_access.arn
    }
    auto_deployments_enabled = false

    image_repository {
      image_identifier      = local.image_uri
      image_repository_type = "ECR"

      image_configuration {
        port = var.container_port
        runtime_environment_variables = {
          RAG_BACKEND             = var.rag_backend
          AWS_REGION              = var.aws_region
          RAG_BEDROCK_EMBED_MODEL = var.bedrock_embed_model
          RAG_BEDROCK_CHAT_MODEL  = var.bedrock_chat_model
          RAG_PDF_PATHS           = var.rag_pdf_paths
          PORT                    = var.container_port
        }
      }
    }
  }

  instance_configuration {
    cpu               = var.cpu
    memory            = var.memory
    instance_role_arn = aws_iam_role.apprunner_instance.arn
  }

  # Liveness is decoupled from the (slow) index build: /healthz returns 200 as
  # soon as the web process is up, so a long first-time build never fails here.
  health_check_configuration {
    protocol            = "HTTP"
    path                = "/healthz"
    interval            = 10
    timeout             = 5
    healthy_threshold   = 1
    unhealthy_threshold = 5
  }

  auto_scaling_configuration_arn = aws_apprunner_auto_scaling_configuration_version.app.arn

  depends_on = [aws_iam_role_policy_attachment.apprunner_ecr]
}
