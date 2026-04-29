provider "null" {}

resource "null_resource" "run_backend" {
  provisioner "local-exec" {
    command = "cd .. && uvicorn main:app"
  }
}