from __future__ import annotations

import json
from pathlib import Path

from app import job_runner, text_processing
from app.models import JobPayload, JobStatus
from app.storage import get_job, init_db, insert_job


class FakeTTSEngine:
    def __init__(
        self, engine: str = "auto", voice: str = "default", speed: float = 1.0, use_gpu: bool = True
    ):
        self.requested_engine = engine
        self.voice = voice
        self.speed = speed
        self.use_gpu = use_gpu
        self.engine_mode = "fake_tts"
        self.device = "cpu"

    def synthesize_to_file(self, text: str, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Minimal payload that allows us to assert output exists.
        out_path.write_bytes(f"FAKE_WAV:{len(text)}".encode("utf-8"))


def test_book_example_pipeline_creates_chapter_parts(tmp_path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    source_path = (
        repo_root
        / "book_example"
        / "Kaym_Warhammer-40000-Eres-Horusa_47_Staraya-Zemlya_RuLit_Me.txt"
    )
    assert source_path.exists(), "Expected example book file in book_example/"

    db_path = tmp_path / "jobs_test.db"
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(job_runner, "TTSEngine", FakeTTSEngine)
    monkeypatch.setattr(text_processing, "get_accentizer", lambda: False)

    # Override settings before init_db
    import app.config as app_config
    app_config.settings.db_path = db_path
    app_config.settings.output_dir = output_dir

    init_db()
    runner = job_runner.JobRunner()
    job_id = "test-book-example"
    insert_job(
        job_id=job_id,
        source_path=str(source_path),
        output_dir=str(output_dir),
        voice="default",
        speed=1.0,
        use_gpu=False,
    )

    payload = JobPayload(
        job_id=job_id,
        source_path=source_path,
        output_dir=output_dir,
        engine="auto",
        voice="default",
        speed=1.0,
        use_gpu=False,
    )
    runner._process_job(payload)

    job = get_job(job_id)
    assert job is not None
    assert job["status"] == JobStatus.DONE.value
    assert job["progress"] == 1.0
    assert job["meta"]["engine"] == "fake_tts"

    manifest_path = output_dir / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["chapters"], "Expected at least one chapter in manifest"
    assert manifest["files"], "Expected at least one output file in manifest"

    for file_item in manifest["files"]:
        chapter = file_item["chapter"]
        part = file_item["part"]
        sub_part = file_item.get("sub_part", 0)
        file_name = file_item["file"]
        if sub_part:
            assert file_name == f"chapter_{chapter:03d}_part_{part:03d}_{sub_part:03d}.mp3"
        else:
            assert file_name == f"chapter_{chapter:03d}_part_{part:03d}.mp3"
        generated = output_dir / file_name
        assert generated.exists()
        assert generated.read_bytes().startswith(b"FAKE_WAV:")
