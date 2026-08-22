from __future__ import annotations

from importlib import resources
from pathlib import Path


def scaffold_project(project_root: Path | str, overwrite: bool = False) -> dict[str, str]:
    """Create the minimal operator-owned project layout from packaged safe defaults."""
    root = Path(project_root).expanduser().resolve()
    directories = [
        root / "config",
        root / "data" / "local",
        root / "data" / "benchmark",
        root / "var",
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    created: dict[str, str] = {}
    resource_root = resources.files("care_anxrag").joinpath("resources")
    templates = {
        root / "config" / "sources.yaml": resource_root.joinpath("sources.yaml"),
        root / ".env.example": resource_root.joinpath("env.example"),
    }
    for destination, resource in templates.items():
        if destination.exists() and not overwrite:
            created[str(destination)] = "preserved"
            continue
        destination.write_text(resource.read_text(encoding="utf-8"), encoding="utf-8")
        created[str(destination)] = "written"
    return created
