from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
from pydantic import BaseModel


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class JobPayload(BaseModel):
    job_id: str
    source_path: Path
    output_dir: Path
    engine: str
    voice: str
    speed: float
    use_gpu: bool


@dataclass(slots=True)
class Job:
    id: str
    source_path: Path
    output_dir: Path
    voice: str = "default"
    speed: float = 1.0
    use_gpu: bool = True
    status: JobStatus = JobStatus.PENDING
    error: str | None = None
    progress: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)
