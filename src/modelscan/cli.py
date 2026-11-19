import typer

app = typer.Typer(help="Inspect an AI model artifact and return a verdict.")


@app.callback()
def main() -> None:
    """Pre-production checkpoint for AI models."""


@app.command()
def scan(path: str) -> None:
    """Scan a model artifact (not implemented yet)."""
    typer.echo(f"scan not implemented: {path}")
    raise typer.Exit(code=2)
