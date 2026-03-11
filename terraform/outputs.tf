output "app_url" {
  description = "Public URL of the deployed MovementScreen application."
  value       = render_web_service.web.url
}

output "service_id" {
  description = "Render service ID — use this to trigger manual deploys via the Render API."
  value       = render_web_service.web.id
}
