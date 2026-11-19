from pathlib import Path

from modelscan.models import InspectorResult, Status


def scan(files: list[Path]) -> InspectorResult:
    # Placeholder: real checkers get plugged in on later branches
    return InspectorResult(
        inspector="static",
        status=Status.OK,
        findings=[],
        coverage=[f"listed {len(files)} file(s)"],
    )
