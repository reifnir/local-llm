locals {
  instance_type = "g7e.8xlarge"
  persistent    = data.terraform_remote_state.persistent.outputs
}

# The data volume and HF token live in their own project so this one can be applied and destroyed freely.
data "terraform_remote_state" "persistent" {
  backend = "s3"

  config = {
    region = "us-east-1"
    bucket = "reifnir-root-terraform-state-primary"
    key    = "reifnir-public-projects/local-llm/persistent/terraform.tfstate"
  }
}

data "aws_region" "current" {}

data "aws_vpc" "default" {
  default = true
}

# Amazon Linux 2023 with NVIDIA open drivers, CUDA, and the SSM agent preinstalled.
data "aws_ssm_parameter" "dlami_al2023" {
  name = "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-amazon-linux-2023/latest/ami-id"
}

data "aws_iam_policy_document" "ec2_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "llm" {
  name               = "local-llm-instance"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume_role.json
}

resource "aws_iam_role_policy_attachment" "llm_ssm" {
  role       = aws_iam_role.llm.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

data "aws_iam_policy_document" "read_hf_token" {
  statement {
    actions   = ["secretsmanager:GetSecretValue"]
    resources = [local.persistent.hf_token_secret_arn]
  }
}

resource "aws_iam_role_policy" "llm_read_hf_token" {
  name   = "read-hf-token"
  role   = aws_iam_role.llm.id
  policy = data.aws_iam_policy_document.read_hf_token.json
}

resource "aws_iam_instance_profile" "llm" {
  name = "local-llm-instance"
  role = aws_iam_role.llm.name
}

# Session Manager is agent-initiated over outbound 443, so no ingress rules are needed.
resource "aws_security_group" "llm" {
  name        = "local-llm-instance"
  description = "local-llm instance: egress only, accessed via SSM Session Manager"
  vpc_id      = data.aws_vpc.default.id
}

resource "aws_vpc_security_group_egress_rule" "llm_all" {
  security_group_id = aws_security_group.llm.id
  ip_protocol       = "-1"
  cidr_ipv4         = "0.0.0.0/0"
}

resource "aws_instance" "llm" {
  ami                    = data.aws_ssm_parameter.dlami_al2023.value
  instance_type          = local.instance_type
  availability_zone      = local.persistent.availability_zone
  iam_instance_profile   = aws_iam_instance_profile.llm.name
  vpc_security_group_ids = [aws_security_group.llm.id]
  user_data = templatefile("${path.module}/cloud-init.yaml", {
    volume_id           = local.persistent.data_volume_id
    hf_token_secret_arn = local.persistent.hf_token_secret_arn
    region              = data.aws_region.current.region
  })
  # One-time spot instances can't be stopped, which an in-place user_data update would need.
  user_data_replace_on_change = true

  # Needed to reach the SSM endpoints from the default VPC (no NAT or VPC endpoints).
  associate_public_ip_address = true

  # On interruption the instance is terminated, which also deletes the root volume.
  instance_market_options {
    market_type = "spot"

    spot_options {
      max_price                      = "5.00"
      spot_instance_type             = "one-time"
      instance_interruption_behavior = "terminate"
    }
  }

  metadata_options {
    http_tokens = "required"
  }

  root_block_device {
    volume_type = "gp3"
    volume_size = 200
    encrypted   = true
    # Instance store NVMe is always wiped and released on termination; this covers the EBS root.
    delete_on_termination = true
  }

  tags = {
    Name = "local-llm"
  }

  lifecycle {
    # Don't replace the instance every time a new DLAMI is published.
    ignore_changes = [ami]
  }
}

resource "aws_volume_attachment" "data" {
  device_name = "/dev/sdf"
  volume_id   = local.persistent.data_volume_id
  instance_id = aws_instance.llm.id
  # The volume is mounted, so a detach would hang on destroy. Terminating the instance releases it.
  skip_destroy = true
}

output "instance_id" {
  value = aws_instance.llm.id
}
