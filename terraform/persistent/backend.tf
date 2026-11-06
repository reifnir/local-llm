terraform {
  # Bucket, region and key are passed to `terraform init`. See .envrc.example.
  backend "s3" {
    encrypt      = true
    use_lockfile = true
  }
  required_providers {
    aws = { source = "hashicorp/aws" }
  }
  # Need 1.11+ for ephemeral variables
  required_version = ">= 1.11"
}
