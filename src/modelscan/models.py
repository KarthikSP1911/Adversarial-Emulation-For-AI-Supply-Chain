from enum import Enum

from pydantic import BaseModel, Field


class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class Status(str, Enum):
    OK = "ok"
    ERROR = "error"
    NOT_RUN = "not_run"


class Finding(BaseModel):
    rule_id: str
    severity: Severity
    inspector: str
    file: str
    message: str
    offset: int | None = None
    evidence: dict = Field(default_factory=dict)


class InspectorResult(BaseModel):
    inspector: str
    status: Status
    findings: list[Finding] = Field(default_factory=list)
    coverage: list[str] = Field(default_factory=list)
    error: str | None = None
