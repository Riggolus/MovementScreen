terraform {
  required_version = ">= 1.5"

  required_providers {
    render = {
      source  = "render-oss/render"
      version = "~> 1.3"
    }
  }

  # Uncomment to store state remotely (recommended for teams / CI).
  # backend "s3" {
  #   bucket = "your-tf-state-bucket"
  #   key    = "movementscreen/terraform.tfstate"
  #   region = "us-east-1"
  # }
}

provider "render" {
  api_key  = var.render_api_key
  owner_id = var.render_owner_id
}

# ── PostgreSQL ────────────────────────────────────────────────────────────────
#
# Free plan: 256 MB RAM, 1 GB storage, deleted after 90 days.
# Change db_plan to "basic_256mb" or higher for a persistent production database.

resource "render_postgres" "db" {
  name    = "${var.app_name}-db"
  plan    = var.db_plan
  region  = var.region
  version = var.db_version
}

# ── Web service ───────────────────────────────────────────────────────────────
#
# Render builds the Docker image from the Dockerfile in your GitHub repo.
# On each push to the configured branch Render will rebuild and redeploy
# automatically (auto_deploy = true).
#
# Plan notes:
#   - starter  ($7/mo)  512 MB RAM — minimum viable for MediaPipe + FastAPI
#   - standard ($25/mo) 2 GB RAM   — recommended for production workloads

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

  # Health check — FastAPI returns 200 on GET /
  health_check_path = "/"

  env_vars = {
    # The internal connection string routes traffic within Render's private network
    # (faster, no egress charges) — do not expose this URL publicly.
    "DATABASE_URL" = {
      value = render_postgres.db.connection_info.internal_connection_string
    }

    "JWT_SECRET" = {
      value = var.jwt_secret
    }

    # Render injects $PORT automatically; this makes it explicit.
    "PORT" = {
      value = "10000"
    }
  }
}
