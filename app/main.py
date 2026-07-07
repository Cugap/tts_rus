from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from loguru import logger

from app.config import settings
from app.job_runner import JobRunner
from app.storage import get_job, init_db, list_jobs
from app.text_processing import SUPPORTED_BOOK_EXTENSIONS, Fb2ValidationResult, validate_fb2

MAX_UPLOAD_SIZE_MB = 200
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024


def _fb2_validation_http_detail(result: Fb2ValidationResult) -> dict:
    return {
        "message": "FB2 не подходит для озвучки",
        "issues": [{"code": issue.code, "message": issue.message} for issue in result.issues],
        "stats": result.stats,
    }


@asynccontextmanager
async def lifespan(application: FastAPI):
    logger.info("Initializing database and directories...")
    init_db()
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    (settings.output_dir / "uploads").mkdir(parents=True, exist_ok=True)
    logger.info("Application startup complete.")
    yield


app = FastAPI(title="Russian Book to Audio", lifespan=lifespan)
runner = JobRunner()
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
INDEX_HTML_PATH = TEMPLATES_DIR / "index.html"


@app.get("/", response_class=FileResponse)
def index() -> FileResponse:
    return FileResponse(path=INDEX_HTML_PATH, media_type="text/html")


@app.get("/engines")
def get_engines() -> dict:
    return {
        "xtts": {
            "name": "XTTS v2 (Клонирование голоса)",
            "speakers": [
                {"id": "speakers/my_voice.wav", "name": "Мой голос"},
                {"id": "default", "name": "Встроенный (speakers_xtts)"},
            ],
        },
        "hf_vits_local": {
            "name": "VITS (Локальная модель)",
            "speakers": [
                {"id": "0", "name": "Женский"},
                {"id": "1", "name": "Мужской"},
            ],
        },
        "sapi": {
            "name": "Windows SAPI",
            "speakers": [{"id": "default", "name": "Системный голос"}],
        },
    }


async def _check_file_size(file: UploadFile) -> bytes:
    ONE_MB = 1024 * 1024
    MAX_READ = MAX_UPLOAD_SIZE_BYTES + ONE_MB
    raw = await file.read(MAX_READ)
    if len(raw) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Файл слишком большой: {len(raw) / ONE_MB:.1f} МБ (максимум {MAX_UPLOAD_SIZE_MB} МБ).",
        )
    return raw


@app.post("/books/validate")
async def validate_book(file: UploadFile = File(...)) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required.")

    ext = Path(file.filename).suffix.lower()
    if ext != ".fb2":
        raise HTTPException(status_code=400, detail="Validation is supported only for .fb2 files.")

    raw = await _check_file_size(file)
    result = validate_fb2(raw)
    return {
        "ok": result.ok,
        "issues": [{"code": issue.code, "message": issue.message} for issue in result.issues],
        "stats": result.stats,
    }


@app.post("/jobs")
async def create_job(
    file: UploadFile = File(...),
    engine: str = Form(default="auto"),
    speaker: str = Form(default="0"),
    speed: float = Form(default=1.0),
    use_gpu: bool = Form(default=True),
) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required.")

    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_BOOK_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_BOOK_EXTENSIONS))
        raise HTTPException(status_code=400, detail=f"Supported formats: {supported}.")

    upload_path = settings.output_dir / "uploads" / file.filename
    raw = await _check_file_size(file)

    if ext == ".fb2":
        validation = validate_fb2(raw)
        if not validation.ok:
            raise HTTPException(
                status_code=400,
                detail=_fb2_validation_http_detail(validation),
            )

    upload_path.write_bytes(raw)

    logger.info(f"Received file {file.filename}, submitting job...")
    job_id = runner.submit(
        source_path=upload_path, engine=engine, voice=speaker, speed=speed, use_gpu=use_gpu
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
