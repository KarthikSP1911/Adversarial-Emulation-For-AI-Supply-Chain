from pathlib import Path

from modelscan.models import Finding, InspectorResult, Status
from modelscan.static import pickle as pickle_engine


def scan(files: list[Path]) -> InspectorResult:
    findings: list[Finding] = []
    coverage: list[str] = []
    errors: list[str] = []
    pickle_files = skipped = 0

    for path in files:
        outcome = pickle_engine.scan_file(path)
        if not outcome.scanned:
            skipped += 1
            continue
        pickle_files += 1
        findings += outcome.findings
        coverage += outcome.notes
        if outcome.error:
            errors.append(outcome.error)

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
