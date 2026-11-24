"""Static scanner orchestrator.

Runs every format-specific scanner and the pickle engine over a list of files
and aggregates the results into a single :class:`~modelscan.models.InspectorResult`.

Scanners invoked
----------------
- :mod:`modelscan.static.pickle_scan`         – pickle / PyTorch containers
- :mod:`modelscan.static.formats.torch`       – PyTorch ZIP structure
- :mod:`modelscan.static.formats.keras`       – Keras / TF SavedModel
- :mod:`modelscan.static.formats.onnx`        – ONNX protobuf graphs
- :mod:`modelscan.static.formats.gguf`        – GGUF LLM checkpoints
- :mod:`modelscan.static.formats.numpy`       – NumPy .npy / .npz arrays
- :mod:`modelscan.static.templates`           – chat-template / Jinja2 abuse
- :mod:`modelscan.static.remote_code`         – shipped Python files
"""

from __future__ import annotations

from pathlib import Path

from modelscan.models import Finding, InspectorResult, Status
from modelscan.static import pickle_scan as pickle_engine
from modelscan.static.formats import gguf, keras, numpy, onnx, torch
from modelscan.static.remote_code import scan as _scan_remote_code
from modelscan.static.templates import scan as _scan_template

# ---------------------------------------------------------------------------
# Format scanners: each returns list[Finding] for one file.
# They are called in order; a file may match multiple scanners.
# ---------------------------------------------------------------------------
_FORMAT_SCANNERS = [
    torch.scan,
    keras.scan,
    onnx.scan,
    gguf.scan,
    numpy.scan,
    _scan_template,
    _scan_remote_code,
]


def scan(files: list[Path]) -> InspectorResult:
    """Run all static scanners over *files* and return a combined result.

    Parameters
    ----------
    files:
        Flat list of :class:`~pathlib.Path` objects to inspect.

    Returns
    -------
    InspectorResult
        Aggregated findings, coverage notes, and overall status.
    """
    findings: list[Finding] = []
    coverage: list[str] = []
    errors: list[str] = []
    pickle_files = skipped = 0

    for path in files:
        # --- pickle / PyTorch engine ---
        outcome = pickle_engine.scan_file(path)
        if not outcome.scanned:
            skipped += 1
        else:
            pickle_files += 1
            findings += outcome.findings
            coverage += outcome.notes
            if outcome.error:
                errors.append(outcome.error)

        # --- format-specific scanners ---
        for scanner in _FORMAT_SCANNERS:
            try:
                findings += scanner(path)
            except Exception as exc:  # noqa: BLE001 – fail open per scanner
                errors.append(
                    f"{scanner.__module__}.{scanner.__name__}({path}): "
                    f"{type(exc).__name__}: {exc}"
                )

    coverage.insert(
        0,
        f"pickle engine: {pickle_files} scanned, {skipped} not a pickle candidate "
        f"(of {len(files)} file(s))",
    )
    return InspectorResult(
        inspector="static",
        status=Status.ERROR if errors else Status.OK,
        findings=findings,
        coverage=coverage,
        error="; ".join(errors) or None,
    )
