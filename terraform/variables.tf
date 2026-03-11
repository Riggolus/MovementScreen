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
  description = "Base name for the Render web service."
  type        = string
  default     = "movementscreen"
}

variable "region" {
  description = "Render region."
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
    "free" spins down after 15 min of inactivity (cold start ~30-60 s).
    "starter" ($7/mo) stays always-on.
    Options: free | starter | standard | pro
  EOT
  type        = string
  default     = "free"
}

variable "github_repo_url" {
  description = <<-EOT
    Full HTTPS URL of the GitHub repository that Render will build from.
    Example: "https://github.com/your-username/MovementScreen"
    The repository must be connected to your Render account under
    Dashboard → Account Settings → Git Connections.
  EOT
  type        = string
}

variable "github_branch" {
  description = "Git branch to deploy."
  type        = string
  default     = "main"
}
