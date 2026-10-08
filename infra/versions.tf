terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Optional: uncomment and configure to store state remotely (recommended for
  # teams). Left local by default so `terraform apply` works with no setup.
  # backend "s3" {
  #   bucket = "my-tf-state-bucket"
  #   key    = "rag-app/terraform.tfstate"
  #   region = "us-east-1"
  # }
}

provider "aws" {
  region = var.aws_region
}
