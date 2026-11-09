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
  # Where the EC2 project runs. The provider region above only holds the original volume and the
  # secret's primary.
  data_region = "eu-north-1"
  # g7e is only offered in eu-north-1a/c. The instance has to launch in the volume's AZ.
  availability_zone = "eu-north-1a"
  # Cross-region copy of a snapshot of the us-east-1 volume. Only read when the volume is created.
  seed_snapshot_id = "snap-0f2eaf5b584d04bcd"
  # Write-only values aren't stored in state, so Terraform can't see a change. Bump this to push a new token.
  hf_token_version = 1
}

variable "hf_token" {
  description = "Hugging Face token to store in Secrets Manager. Ephemeral, so it stays out of state and plan files."
  type        = string
  sensitive   = true
  ephemeral   = true
}

# The original volume. Remove once the eu-north-1 copy is verified.
resource "aws_ebs_volume" "data" {
  availability_zone = "us-east-1d"
  type              = "gp3"
  size              = 500
  # The 125 MB/s default is slow for 50+ GB files.
  throughput = 500
  encrypted  = true

  tags = {
    Name = "local-llm-data"
  }
}

resource "aws_ebs_volume" "data_eu_north_1" {
  region            = local.data_region
  availability_zone = local.availability_zone
  snapshot_id       = local.seed_snapshot_id
  type              = "gp3"
  size              = 500
  # The 125 MB/s default is slow for 50+ GB files.
  throughput = 500
  encrypted  = true

  tags = {
    Name = "local-llm-data"
  }

  lifecycle {
    # snapshot_id forces replacement. This lets the seed snapshot be deleted later.
    ignore_changes = [snapshot_id]
  }
}

resource "aws_secretsmanager_secret" "hf_token" {
  name = "local-llm/hf-token"
  # The token can be reissued, and a recovery window would block recreating this name.
  recovery_window_in_days = 0

  # The instance reads the token from its own region.
  replica {
    region = local.data_region
  }
}

resource "aws_secretsmanager_secret_version" "hf_token" {
  secret_id                = aws_secretsmanager_secret.hf_token.id
  secret_string_wo         = var.hf_token
  secret_string_wo_version = local.hf_token_version
}

output "region" {
  value = local.data_region
}

output "availability_zone" {
  value = aws_ebs_volume.data_eu_north_1.availability_zone
}

output "data_volume_id" {
  value = aws_ebs_volume.data_eu_north_1.id
}

# A replica's ARN is the primary's with the region swapped.
output "hf_token_secret_arn" {
  value = replace(aws_secretsmanager_secret.hf_token.arn, ":us-east-1:", ":${local.data_region}:")
}
