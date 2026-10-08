output "ecr_repository_url" {
  description = "Push the app image here before applying the App Runner service."
  value       = aws_ecr_repository.app.repository_url
}

output "service_url" {
  description = "Public HTTPS URL of the deployed RAG app."
  value       = "https://${aws_apprunner_service.app.service_url}"
}

output "service_arn" {
  description = "ARN of the App Runner service."
  value       = aws_apprunner_service.app.arn
}

output "instance_role_arn" {
  description = "IAM role the running app assumes (used to call Bedrock)."
  value       = aws_iam_role.apprunner_instance.arn
}