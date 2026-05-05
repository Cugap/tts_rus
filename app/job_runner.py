from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from queue import Queue

from loguru import logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from app.config import settings
from app.models import JobStatus, JobPayload
from app.storage import insert_job, update_job
from app.text_processing import (
    load_text_file,
    split_into_chapters,
    split_text_safely,
    normalize_text,
)
from app.tts_engine import TTSEngine


class JobRunner:
    def __init__(self) -> None:
        self._queue: Queue[JobPayload] = Queue()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def submit(self, source_path: Path, voice: str, speed: float, use_gpu: bool) -> str:
        job_id = str(uuid.uuid4())
        output_dir = self._allocate_output_dir(source_path.stem)

        insert_job(
            job_id=job_id,
            source_path=str(source_path),
            output_dir=str(output_dir),
            voice=voice,
            speed=speed,
            use_gpu=use_gpu,
        )

        payload = JobPayload(
            job_id=job_id,
            source_path=source_path,
            output_dir=output_dir,
            voice=voice,
            speed=speed,
            use_gpu=use_gpu,
        )
        self._queue.put(payload)
        logger.info(f"Job {job_id} submitted and added to queue.")
        return job_id

    @staticmethod
    def _allocate_output_dir(base_name: str) -> Path:
        INITIAL_DIR_SUFFIX = 2
        safe_name = base_name.strip() or "book"
        candidate = settings.output_dir / safe_name
        suffix = INITIAL_DIR_SUFFIX
        while candidate.exists():
            candidate = settings.output_dir / f"{safe_name}_{suffix}"
            suffix += 1
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate

    def _run_loop(self) -> None:
        while True:
            payload = self._queue.get()
            try:
                self._process_job(payload)
            except Exception as err:  # pragma: no cover
                logger.exception(f"Job {payload.job_id} failed with error: {err}")
                update_job(payload.job_id, status=JobStatus.FAILED, error=str(err))
            finally:
                self._queue.task_done()

    def _process_job(self, payload: JobPayload) -> None:
        job_id = payload.job_id
        source_path = payload.source_path
        output_dir = payload.output_dir
        voice = payload.voice
        speed = payload.speed
        use_gpu = payload.use_gpu

        logger.info(f"Starting job {job_id} for {source_path}")
        update_job(job_id, status=JobStatus.RUNNING, progress=0.0, error="")

        text = load_text_file(source_path)
        chapters = split_into_chapters(text)
        if not chapters:
            raise ValueError("Book text is empty after parsing.")

        engine = TTSEngine(voice=voice, speed=speed, use_gpu=use_gpu)
        plan: list[tuple[int, int, str]] = []
        chapter_chunks: dict[int, list[str]] = {}

        for chapter in chapters:
            chunks = split_text_safely(
                chapter.text, max_chars=settings.max_chars_per_chunk
            )
            chapter_chunks[chapter.number] = chunks
            for part_idx, chunk in enumerate(chunks, start=1):
                plan.append((chapter.number, part_idx, chunk))

        total = len(plan)
        done = 0
        manifest = {
            "engine": getattr(engine, "engine_mode", "unknown"),
            "device": getattr(engine, "device", "cpu"),
            "speaker": voice,
            "chapters": [],
            "files": [],
        }

        for chapter in chapters:
            manifest["chapters"].append(
                {"number": chapter.number, "title": chapter.title}
            )

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )
        def synthesize_with_retry(chunk_text: str, out_path: Path):
            current_max = settings.max_chars_per_chunk
            while True:
                try:
                    engine.synthesize_to_file(chunk_text[:current_max], out_path)
                    break
                except MemoryError:
                    current_max = max(
                        settings.min_chars_per_chunk,
                        int(current_max * settings.memory_reduction_factor),
                    )
                    if current_max <= settings.min_chars_per_chunk:
                        raise

        for chapter_num, part_num, chunk in plan:
            filename = f"chapter_{chapter_num:03d}_part_{part_num:03d}.mp3"
            out_file = output_dir / filename

            # Normalize text (numbers to words, accents)
            normalized_chunk = normalize_text(chunk)

            logger.debug(f"Synthesizing {filename}...")
            synthesize_with_retry(normalized_chunk, out_file)

            done += 1
            progress = done / total
            manifest["files"].append(
                {
                    "chapter": chapter_num,
                    "part": part_num,
                    "file": filename,
                }
            )
            (output_dir / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            update_job(job_id, progress=progress, meta=manifest)

        logger.info(f"Job {job_id} completed successfully.")
        update_job(job_id, status=JobStatus.DONE, progress=1.0, meta=manifest, error="")
