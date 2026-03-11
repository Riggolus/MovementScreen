terraform {
  required_version = ">= 1.5"

  required_providers {
    render = {
      source  = "render-oss/render"
      version = "~> 1.3"
    }
  }
}

provider "render" {
  api_key  = var.render_api_key
  owner_id = var.render_owner_id
}

# ── Web service ───────────────────────────────────────────────────────────────
#
# Free plan spins down after 15 min of inactivity (30-60 s cold start on
# next request). Upgrade to "starter" ($7/mo) for always-on.

resource "render_web_service" "web" {
  name   = var.app_name
  plan   = var.web_plan
  region = var.region

  runtime_source = {
    docker = {
      repo_url        = var.github_repo_url
      branch          = var.github_branch
      auto_deploy     = true
      dockerfile_path = "./Dockerfile"
      context         = "."
    }
  }

  health_check_path = "/"
}
