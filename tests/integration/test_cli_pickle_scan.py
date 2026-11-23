import json
import pickle
import zipfile

from typer.testing import CliRunner

from modelscan.cli import app

runner = CliRunner()


class _Marker:
    """Pickling this does not run it; unpickling would create the marker."""

    def __init__(self, path):
        self.path = path

    def __reduce__(self):
        import os

        return (os.system, (f"echo pwned > {self.path}",))


def test_malicious_pickle_is_blocked_and_never_executed(tmp_path):
    marker = tmp_path / "marker.txt"
    f = tmp_path / "evil.pkl"
    f.write_bytes(pickle.dumps(_Marker(marker)))
    result = runner.invoke(app, ["scan", str(f), "--json"])
    assert result.exit_code == 2
    data = json.loads(result.stdout)
    assert any(x["rule_id"] == "PKL-001" for x in data["findings"])
    assert not marker.exists()


def test_benign_pickle_passes(tmp_path):
    f = tmp_path / "ok.pkl"
    f.write_bytes(pickle.dumps({"w": [0.1, 0.2]}))
    result = runner.invoke(app, ["scan", str(f)])
    assert result.exit_code == 0


def test_pytorch_style_zip_is_scanned(tmp_path):
    marker = tmp_path / "marker.txt"
    f = tmp_path / "model.pt"
    with zipfile.ZipFile(f, "w") as zf:
        zf.writestr("archive/data.pkl", pickle.dumps(_Marker(marker)))
    result = runner.invoke(app, ["scan", str(f)])
    assert result.exit_code == 2
    assert "PKL-001" in result.stdout
    assert not marker.exists()


def test_corrupt_pickle_reports_parse_error(tmp_path):
    f = tmp_path / "bad.pkl"
    f.write_bytes(pickle.dumps({"a": 1})[:-3])
    result = runner.invoke(app, ["scan", str(f)])
    assert "PKL-090" in result.stdout


def test_unrelated_bin_is_skipped(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"hello")
    result = runner.invoke(app, ["scan", str(f)])
    assert result.exit_code == 0
