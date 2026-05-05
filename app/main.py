from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from loguru import logger

from app.config import settings
from app.job_runner import JobRunner
from app.storage import get_job, init_db, list_jobs


app = FastAPI(title="Russian Book to Audio")
runner = JobRunner()
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
INDEX_HTML_PATH = TEMPLATES_DIR / "index.html"


@app.on_event("startup")
def startup() -> None:
    logger.info("Initializing database and directories...")
    init_db()
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    (settings.output_dir / "uploads").mkdir(parents=True, exist_ok=True)
    logger.info("Application startup complete.")


@app.get("/", response_class=FileResponse)
def index() -> FileResponse:
    return FileResponse(path=INDEX_HTML_PATH, media_type="text/html")


@app.post("/jobs")
async def create_job(
    file: UploadFile = File(...),
    speaker_id: int = Form(default=0),
    speed: float = Form(default=1.0),
    use_gpu: bool = Form(default=True),
) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required.")

    ext = Path(file.filename).suffix.lower()
    if ext not in {".txt"}:
        raise HTTPException(status_code=400, detail="Only .txt is supported in MVP.")

    upload_path = settings.output_dir / "uploads" / file.filename
    raw = await file.read()
    upload_path.write_bytes(raw)

    logger.info(f"Received file {file.filename}, submitting job...")
    job_id = runner.submit(
        source_path=upload_path, voice=str(speaker_id), speed=speed, use_gpu=use_gpu
    )
    return {"job_id": job_id}


@app.get("/jobs")
def get_jobs() -> list[dict]:
    return list_jobs()


@app.get("/jobs/{job_id}")
def get_job_status(job_id: str) -> dict:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job
