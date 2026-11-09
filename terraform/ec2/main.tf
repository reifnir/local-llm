provider "aws" {
  # Wherever the persistent project put the data volume.
  region = data.terraform_remote_state.persistent.outputs.region
}

data "aws_caller_identity" "current" {}

output "debug" {
  value = data.aws_caller_identity.current
}

# G and VT quotas are measured in vCPUs; one g7e.12xlarge needs 48.
locals {
  g_and_vt_vcpus = 48
}
