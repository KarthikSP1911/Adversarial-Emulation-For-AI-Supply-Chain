from typer.testing import CliRunner

from modelscan.cli import app

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.stdout


def test_scan_file(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"hello")
    result = runner.invoke(app, ["scan", str(f)])
    assert result.exit_code == 0
    assert "Scanned 1 file" in result.stdout


def test_scan_missing_path():
    result = runner.invoke(app, ["scan", "does/not/exist"])
    assert result.exit_code != 0
