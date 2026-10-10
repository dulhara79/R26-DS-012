"""Container must include reproducible in-process fusion dependencies."""

from pathlib import Path


def test_docker_includes_frozen_fusion_and_requires_db():
    root = Path(__file__).resolve().parents[2]
    docker = (root / "central_backend" / "Dockerfile").read_text()
    assert "fusion_service/requirements.txt" in docker
    assert "fusion_service/fusion.py" in docker
    assert "fusion_service/harmonise.py" in docker
    assert "fusion_service/reference/*.json" in docker
    assert "DATABASE_URL is required" in docker
    assert "migrate_p0.py &&" not in docker
    assert (root / ".dockerignore").exists()
