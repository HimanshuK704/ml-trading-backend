terraform {
  required_providers {
    local = {
      source  = "hashicorp/local"
      version = "~> 2.5"
    }
    time = {
      source  = "hashicorp/time"
      version = "~> 0.11"
    }
  }
}

provider "local" {}
provider "time" {}

# New timestamp each apply
resource "time_static" "now" {}

# Append a line to output.txt on each apply
resource "null_resource" "append_log" {
  # forces re-run every apply
  triggers = {
    always = timestamp()
  }

  provisioner "local-exec" {
    command = "echo ${time_static.now.rfc3339} - Terraform apply ran successfully >> output.txt"
  }
}