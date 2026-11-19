from pathlib import Path

import typer


def resolve_target(target: str) -> list[Path]:
    if target.startswith("hf://"):
        raise typer.BadParameter("hf:// support coming in feat/ingest-hub")
    path = Path(target)
    if not path.exists():
        raise typer.BadParameter(f"Path not found: {target}")
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.is_file())
