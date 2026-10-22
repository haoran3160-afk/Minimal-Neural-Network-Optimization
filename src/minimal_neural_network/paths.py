"""Paths for an editable checkout, independent of the caller's working directory."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
REPORTS_DIR = PROJECT_ROOT / "reports"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"


def result_path(filename, directory=RESULTS_DIR):
    filename = str(filename)
    if Path(filename).name != filename:
        raise ValueError("filename must be a single result filename")
    groups = {".csv": "tables", ".png": "figures", ".json": "artifacts", ".npz": "artifacts"}
    try:
        group = groups[Path(filename).suffix.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported result type: {filename}") from exc
    return Path(directory) / group / filename


def prepare_results(directory=RESULTS_DIR):
    for group in ("tables", "figures", "artifacts"):
        (Path(directory) / group).mkdir(parents=True, exist_ok=True)
