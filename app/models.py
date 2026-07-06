from __future__ import annotations

from enum import Enum
from pathlib import Path

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
