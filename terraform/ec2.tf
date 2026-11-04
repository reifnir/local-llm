locals {
  instance_type = "g7.24xlarge"
  # g7 isn't offered in us-east-1e/f; pinning the AZ picks that AZ's default subnet.
  availability_zone = "us-east-1a"
}

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
  availability_zone      = local.availability_zone
  iam_instance_profile   = aws_iam_instance_profile.llm.name
  vpc_security_group_ids = [aws_security_group.llm.id]
  user_data              = file("${path.module}/cloud-init.yaml")

  # Needed to reach the SSM endpoints from the default VPC (no NAT or VPC endpoints).
  associate_public_ip_address = true

  metadata_options {
    http_tokens = "required"
  }

  root_block_device {
    volume_type = "gp3"
    volume_size = 200
    encrypted   = true
  }

  tags = {
    Name = "local-llm"
  }

  lifecycle {
    # Don't replace the instance every time a new DLAMI is published.
    ignore_changes = [ami]
  }
}

output "instance_id" {
  value = aws_instance.llm.id
}
