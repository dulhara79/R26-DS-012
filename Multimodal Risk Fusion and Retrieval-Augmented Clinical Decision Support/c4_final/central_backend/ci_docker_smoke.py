"""Smoke only: used by the Docker packaging GitHub Actions job."""
from fastapi.testclient import TestClient
from main import app

response = TestClient(app).get("/ready")
assert response.status_code == 200, response.text
assert response.json()["schema_revision"] == "0003_push_registry_outbox"
print("Central Backend Docker image readiness confirmed")
