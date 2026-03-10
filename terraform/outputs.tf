output "app_url" {
  description = "Public URL of the deployed MovementScreen application."
  value       = "https://${render_web_service.web.url}"
}

output "service_id" {
  description = "Render service ID — use this to trigger manual deploys via the Render API."
  value       = render_web_service.web.id
}

output "database_id" {
  description = "Render Postgres instance ID."
  value       = render_postgres.db.id
}

output "database_external_url" {
  description = <<-EOT
    External PostgreSQL connection string for database administration tools
    (pgAdmin, DBeaver, psql from your local machine, etc.).
    Marked sensitive — retrieve with: terraform output -raw database_external_url
  EOT
  value     = render_postgres.db.connection_info.external_connection_string
  sensitive = true
}

output "database_psql_command" {
  description = "Ready-to-run psql command for direct database access."
  value       = render_postgres.db.connection_info.psql_command
  sensitive   = true
}
