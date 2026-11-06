# resource "aws_servicequotas_service_quota" "g_and_vt_on_demand" {
#   service_code = "ec2"
#   quota_code   = "L-DB2E81BA" # Running On-Demand G and VT instances
#   value        = local.g_and_vt_vcpus
# }

# resource "aws_servicequotas_service_quota" "g_and_vt_spot" {
#   service_code = "ec2"
#   quota_code   = "L-3819A6DF" # All G and VT Spot Instance Requests
#   value        = local.g_and_vt_vcpus
# }


# removed {
#   from = aws_servicequotas_service_quota.g_and_vt_spot

#   lifecycle {
#     destroy = false
#   }
# }

# removed {
#   from = aws_servicequotas_service_quota.g_and_vt_on_demand

#   lifecycle {
#     destroy = false
#   }
# }
