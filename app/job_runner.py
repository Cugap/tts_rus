from __future__ import annotations

import json
import re
import shutil
import subprocess
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
    load_chapters,
    split_text_safely,
    normalize_text,
    normalize_text_no_accents,
    normalize_text_for_xtts,
)
from app.tts_engine import TTSEngine


class JobRunner:
    def __init__(self) -> None:
        self._queue: Queue[JobPayload] = Queue()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def submit(self, source_path: Path, engine: str, voice: str, speed: float, use_gpu: bool, concat: bool = True) -> str:
        job_id = str(uuid.uuid4())
        output_dir = self._allocate_output_dir(source_path.stem)

        insert_job(
            job_id=job_id,
            source_path=str(source_path),
            output_dir=str(output_dir),
            voice=voice,
            speed=speed,
            use_gpu=use_gpu,
            concat=concat,
        )

        payload = JobPayload(
            job_id=job_id,
            source_path=source_path,
            output_dir=output_dir,
            engine=engine,
            voice=voice,
            speed=speed,
            use_gpu=use_gpu,
            concat=concat,
        )
        self._queue.put(payload)
        logger.info(f"Job {job_id} submitted and added to queue.")
        return job_id

    @staticmethod
    def _allocate_output_dir(base_name: str) -> Path:
        INITIAL_DIR_SUFFIX = 1
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
                # Clean up partial output on failure
                if payload.output_dir.exists():
                    import shutil
                    shutil.rmtree(payload.output_dir, ignore_errors=True)
            finally:
                self._queue.task_done()

    @staticmethod
    def _concat_mp3_files(output_dir: Path, book_name: str, manifest: dict) -> str | None:
        """Склеить все MP3-файлы в output_dir в один через FFmpeg concat demuxer."""
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            logger.warning("FFmpeg не найден в PATH. Склеивание пропущено.")
            return None

        # Собираем все файлы из manifest в правильном порядке
        mp3_files: list[Path] = []
        for entry in manifest.get("files", []):
            fname = entry.get("file", "")
            mp3_files.append(output_dir / fname)

        # Проверяем, что файлы существуют
        existing = [f for f in mp3_files if f.exists()]
        if len(existing) < 2:
            logger.info("Меньше 2 файлов — склеивание не требуется.")
            return None

        # Создаём временный файл списка для concat demuxer
        concat_list = output_dir / "_concat_list.txt"
        try:
            concat_list.write_text(
                "\n".join(
                    f"file '{f.name}'" for f in existing
                ),
                encoding="utf-8",
            )

            concat_name = settings.concat_filename.format(book_name=book_name)
            concat_path = output_dir / concat_name

            logger.info(f"Склеивание {len(existing)} файлов в {concat_name}...")
            result = subprocess.run(
                [
                    ffmpeg, "-y",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", str(concat_list),
                    "-c", "copy",
                    str(concat_path),
                ],
                capture_output=True, text=True, timeout=3600,
            )
            if result.returncode != 0:
                logger.error(f"FFmpeg concat не удался: {result.stderr[:500]}")
                return None

            logger.info(f"Склеивание завершено: {concat_name}")
            return concat_name

        except Exception as exc:
            logger.error(f"Ошибка при склеивании: {exc}")
            return None
        finally:
            concat_list.unlink(missing_ok=True)

    def _process_job(self, payload: JobPayload) -> None:
        job_id = payload.job_id
        source_path = payload.source_path
        output_dir = payload.output_dir
        engine_name = payload.engine
        voice = payload.voice
        speed = payload.speed
        use_gpu = payload.use_gpu

        logger.info(f"Starting job {job_id} for {source_path}")
        update_job(job_id, status=JobStatus.RUNNING, progress=0.0, error="")

        chapters = load_chapters(source_path)
        if not chapters:
            raise ValueError("Book text is empty after parsing.")

        engine = TTSEngine(engine=engine_name, voice=voice, speed=speed, use_gpu=use_gpu)
        is_xtts = engine.engine_mode == "xtts"

        # Plan: list of (chapter_num, part_num, sub_part_num, raw_text)
        plan: list[tuple[int, int, int, str]] = []

        for chapter in chapters:
            chunks = split_text_safely(
                chapter.text, max_chars=settings.max_chars_per_chunk
            )
            for part_idx, chunk in enumerate(chunks, start=1):
                plan.append((chapter.number, part_idx, 0, chunk))

        # Normalize, and sub-split entries that exceed XTTS char limit
        # sub_part_num == 0 means no sub-splitting was needed
        # For XTTS we skip accent (`+`) marks — they hurt quality
        _SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")

        def _split_xtts_chunk(text: str, max_chars: int) -> list[str]:
            """Split text into chunks ≤ max_chars, first by sentence, then by comma, then by words."""
            if len(text) <= max_chars:
                return [text]

            # 1) Split by sentences
            sentences = _SENTENCE_SPLIT_RE.split(text)
            result: list[str] = []
            buf: list[str] = []
            buf_len = 0

            for sent in sentences:
                if len(sent) > max_chars:
                    # Single sentence too long — flush buffer first, then split sentence
                    if buf:
                        result.append(" ".join(buf))
                        buf = []
                        buf_len = 0
                    # Split long sentence by commas
                    parts = re.split(r"(?<=,)\s+", sent)
                    for part in parts:
                        if len(part) > max_chars:
                            # Force-split by character count at word boundaries
                            words = part.split()
                            wbuf: list[str] = []
                            wlen = 0
                            for w in words:
                                if wlen + len(w) + 1 > max_chars and wbuf:
                                    result.append(" ".join(wbuf))
                                    wbuf = []
                                    wlen = 0
                                wbuf.append(w)
                                wlen += len(w) + 1
                            if wbuf:
                                result.append(" ".join(wbuf))
                        else:
                            if buf_len + len(part) + 1 > max_chars and buf:
                                result.append(" ".join(buf))
                                buf = []
                                buf_len = 0
                            buf.append(part)
                            buf_len += len(part) + 1
                else:
                    if buf_len + len(sent) + 1 > max_chars and buf:
                        result.append(" ".join(buf))
                        buf = []
                        buf_len = 0
                    buf.append(sent)
                    buf_len += len(sent) + 1

            if buf:
                result.append(" ".join(buf))
            return result

        # Константа для пауз между предложениями:
        # Двойной пробел после точки/вопроса/восклицания заставляет XTTS/VITS
        # делать более длинную естественную паузу.
        _PAUSE_MARKER_RE = re.compile(r"(?<=[.!?])\s+(?=[А-ЯA-Z])")

        sub_plan: list[tuple[int, int, int, str]] = []
        for chapter_num, part_num, _, chunk in plan:
            if is_xtts:
                normalized = normalize_text_for_xtts(chunk)
                # Двойные пробелы после знаков препинания → более длинные паузы
                normalized = _PAUSE_MARKER_RE.sub("  ", normalized)
                sub_chunks = _split_xtts_chunk(normalized, settings.xtts_max_chars)
                if len(sub_chunks) == 1:
                    # sub_part_num=0 — дробление не потребовалось
                    sub_plan.append((chapter_num, part_num, 0, sub_chunks[0]))
                else:
                    for sub_idx, sub_text in enumerate(sub_chunks, start=1):
                        sub_plan.append((chapter_num, part_num, sub_idx, sub_text))
            else:
                normalized = normalize_text(chunk)
                # Для VITS/SAPI: двойные пробелы после знаков препинания
                normalized = _PAUSE_MARKER_RE.sub("  ", normalized)
                sub_plan.append((chapter_num, part_num, 0, normalized))

        total = len(sub_plan)
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

        for chapter_num, part_num, sub_part_num, text in sub_plan:
            if sub_part_num:
                filename = f"chapter_{chapter_num:03d}_part_{part_num:03d}_{sub_part_num:03d}.mp3"
            else:
                filename = f"chapter_{chapter_num:03d}_part_{part_num:03d}.mp3"
            out_file = output_dir / filename

            logger.debug(f"Synthesizing {filename}...")
            synthesize_with_retry(text, out_file)

            done += 1
            progress = done / total
            manifest_entry: dict = {
                "chapter": chapter_num,
                "part": part_num,
                "file": filename,
            }
            if sub_part_num:
                manifest_entry["sub_part"] = sub_part_num
            manifest["files"].append(manifest_entry)
            update_job(job_id, progress=progress, meta=manifest)

        # ── Склеивание всех частей в один файл ────────────────────────────
        concat_file: str | None = None
        if payload.concat and settings.concat_enabled:
            concat_file = self._concat_mp3_files(output_dir, source_path.stem, manifest)

        # Write manifest once at the end
        manifest_data = {**manifest}
        if concat_file:
            manifest_data["concat_file"] = concat_file
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"Job {job_id} completed successfully.")
        update_job(job_id, status=JobStatus.DONE, progress=1.0, meta=manifest_data, error="")
