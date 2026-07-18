from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Find the project root from a path inside the repository."""
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "configs" / "project_config.yaml").exists():
            return candidate
    raise FileNotFoundError("No se encontro configs/project_config.yaml desde la ruta actual.")


PROJECT_ROOT = find_project_root()
CONFIG_PATH = PROJECT_ROOT / "configs" / "project_config.yaml"


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)

