# Everything that outlives the GPU instance. The EC2 project one directory up reads these outputs
# from this project's state, so apply this one first:
#
#   terraform init \
#     -backend-config="region=$TERRAFORM_STATE_REGION" \
#     -backend-config="bucket=$TERRAFORM_STATE_BUCKET" \
#     -backend-config="key=$TERRAFORM_STATE_KEY_PREFIX/persistent/terraform.tfstate"
#   TF_VAR_hf_token="$HF_TOKEN" terraform apply

provider "aws" {
  region = "us-east-1"
}

locals {
  # g7e is only offered in us-east-1b/c/d. The instance has to launch in the volume's AZ.
  availability_zone = "us-east-1c"
  # Write-only values aren't stored in state, so Terraform can't see a change. Bump this to push a new token.
  hf_token_version = 1
}

variable "hf_token" {
  description = "Hugging Face token to store in Secrets Manager. Ephemeral, so it stays out of state and plan files."
  type        = string
  sensitive   = true
  ephemeral   = true
}

resource "aws_ebs_volume" "data" {
  availability_zone = local.availability_zone
  type              = "gp3"
  size              = 500
  # The 125 MB/s default is slow for 50+ GB files.
  throughput = 500
  encrypted  = true

  tags = {
    Name = "local-llm-data"
  }
}

resource "aws_secretsmanager_secret" "hf_token" {
  name = "local-llm/hf-token"
  # The token can be reissued, and a recovery window would block recreating this name.
  recovery_window_in_days = 0
}

resource "aws_secretsmanager_secret_version" "hf_token" {
  secret_id                = aws_secretsmanager_secret.hf_token.id
  secret_string_wo         = var.hf_token
  secret_string_wo_version = local.hf_token_version
}

output "availability_zone" {
  value = aws_ebs_volume.data.availability_zone
}

output "data_volume_id" {
  value = aws_ebs_volume.data.id
}

output "hf_token_secret_arn" {
  value = aws_secretsmanager_secret.hf_token.arn
}
