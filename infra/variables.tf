variable "aws_region" {
  description = "AWS region to deploy into (must have Bedrock + App Runner available)."
  type        = string
  default     = "us-east-1"
}

variable "service_name" {
  description = "Name used for the ECR repo, App Runner service, and IAM roles."
  type        = string
  default     = "rag-app"
}

variable "image_tag" {
  description = "Image tag in ECR to deploy (the deploy script builds/pushes this tag first)."
  type        = string
  default     = "latest"
}

variable "cpu" {
  description = "App Runner vCPU (e.g. 1024 = 1 vCPU). Valid: 256/512/1024/2048/4096."
  type        = string
  default     = "1024"
}

variable "memory" {
  description = "App Runner memory in MB. Must pair with cpu per App Runner's allowed combos."
  type        = string
  default     = "3072"
}

variable "container_port" {
  description = "Port the container listens on (matches the Dockerfile EXPOSE / gunicorn bind)."
  type        = string
  default     = "8080"
}

variable "rag_backend" {
  description = "Which answer backend the deployed app uses: 'bedrock' (AWS) or 'local' (offline)."
  type        = string
  default     = "bedrock"
}

variable "bedrock_embed_model" {
  description = "Bedrock embeddings model id (enable it under Bedrock > Model access)."
  type        = string
  default     = "amazon.titan-embed-text-v2:0"
}

variable "bedrock_chat_model" {
  description = "Bedrock generative model id for answers (enable it under Bedrock > Model access)."
  type        = string
  default     = "anthropic.claude-3-haiku-20240307-v1:0"
}

variable "rag_pdf_paths" {
  description = "Comma-separated PDF paths (inside the image) to index."
  type        = string
  default     = "gk_ques_ans.pdf"
}

variable "min_size" {
  description = "App Runner autoscaling: minimum number of instances kept warm."
  type        = number
  default     = 1
}

variable "max_size" {
  description = "App Runner autoscaling: maximum number of instances."
  type        = number
  default     = 3
}

variable "max_concurrency" {
  description = "Requests per instance before App Runner scales out."
  type        = number
  default     = 50
}