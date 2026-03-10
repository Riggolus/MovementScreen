variable "render_api_key" {
  description = "Render API key — Dashboard → Account Settings → API Keys"
  type        = string
  sensitive   = true
}

variable "render_owner_id" {
  description = <<-EOT
    Render owner ID for the account or team that will own the resources.
    Find it at Dashboard → Account Settings, or via:
      curl -s -H "Authorization: Bearer <your-api-key>" \
        https://api.render.com/v1/owners?limit=1 | jq '.[0].owner.id'
  EOT
  type        = string
}

variable "app_name" {
  description = "Base name for all Render resources (web service, database)."
  type        = string
  default     = "movementscreen"
}

variable "region" {
  description = "Render region for all resources."
  type        = string
  default     = "oregon"
  validation {
    condition     = contains(["oregon", "ohio", "virginia", "frankfurt", "singapore"], var.region)
    error_message = "region must be one of: oregon, ohio, virginia, frankfurt, singapore."
  }
}

variable "web_plan" {
  description = <<-EOT
    Render plan for the web service.
    MediaPipe requires at least 512 MB RAM — "starter" ($7/mo, 512 MB) is the minimum
    recommended plan. The free plan spins down after 15 min of inactivity which adds
    30-60 s cold-start latency and may not have enough RAM for MediaPipe.
    Options: starter | standard | pro | pro_plus | pro_max | pro_ultra
  EOT
  type        = string
  default     = "starter"
}

variable "db_plan" {
  description = <<-EOT
    Render PostgreSQL plan.
    "free" (256 MB, 1 GB storage) expires after 90 days — suitable for evaluation only.
    Use "basic_256mb" ($7/mo) or higher for a persistent production database.
    Options: free | basic_256mb | basic_1gb | standard_4gb | pro_4gb | ...
  EOT
  type        = string
  default     = "free"
}

variable "db_version" {
  description = "PostgreSQL major version."
  type        = string
  default     = "16"
}

variable "github_repo_url" {
  description = <<-EOT
    Full HTTPS URL of the GitHub (or GitLab) repository that Render will build from.
    Example: "https://github.com/your-username/MovementScreen"
    The repository must already be connected to your Render account under
    Dashboard → Account Settings → Git Connections.
  EOT
  type        = string
}

variable "github_branch" {
  description = "Git branch to deploy."
  type        = string
  default     = "main"
}

variable "jwt_secret" {
  description = <<-EOT
    Secret used to sign JWT access and refresh tokens.
    Must be a strong, random string of at least 32 characters.
    Generate one with: python -c "import secrets; print(secrets.token_hex(32))"
  EOT
  type      = string
  sensitive = true
}
