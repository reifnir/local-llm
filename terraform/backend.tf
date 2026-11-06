terraform {
  # Bucket, region and key are passed to `terraform init`. See .envrc.example.
  backend "s3" {
    encrypt      = true
    use_lockfile = true
  }
  required_providers {
    aws = { source = "hashicorp/aws" }
  }
}
