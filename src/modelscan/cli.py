from pathlib import Path

import typer

from modelscan import __version__
from modelscan.ingest import resolve_target
from modelscan.models import Severity
from modelscan.static import scan as static_scan

app = typer.Typer(help="modelscan: a pre-production checkpoint for AI models.")


@app.callback()
def main() -> None:
    """Keeps 'scan' as a subcommand (Typer collapses single-command apps otherwise)."""


@app.command()
def version() -> None:
    """Print the version."""
    typer.echo(__version__)


@app.command()
def scan(
    target: str = typer.Argument(..., help="File, folder, or hf://user/model"),
    json_out: bool = typer.Option(False, "--json", help="Print JSON output"),
    out: Path | None = typer.Option(None, "--out", help="Write JSON report to file"),
) -> None:
    """Statically scan a model artifact."""
    files = resolve_target(target)
    result = static_scan(files)

    payload = result.model_dump_json(indent=2)
    if out:
        out.write_text(payload)
    if json_out:
        typer.echo(payload)
    else:
        typer.echo(f"Scanned {len(files)} file(s) - status: {result.status.value}")
        for f in result.findings:
            typer.echo(f"[{f.severity.value}] {f.rule_id} {f.file}: {f.message}")

    critical = any(f.severity == Severity.CRITICAL for f in result.findings)
    raise typer.Exit(code=2 if critical else 0)
