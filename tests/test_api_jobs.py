from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from app import job_runner, main, text_processing
from app.models import JobStatus
from app.storage import init_db


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
        out_path.write_bytes(f"FAKE_WAV:{len(text)}".encode("utf-8"))


def test_api_create_job_and_get_status(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "jobs_test.db"
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "uploads").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(main.settings, "db_path", db_path)
    monkeypatch.setattr(main.settings, "output_dir", output_dir)
    monkeypatch.setattr(job_runner, "TTSEngine", FakeTTSEngine)
    monkeypatch.setattr(text_processing, "get_accentizer", lambda: False)

    init_db()
    main.runner = job_runner.JobRunner()

    with TestClient(main.app) as client:
        book_text = (
            "Глава 1\n"
            "Это первая глава. Тут немного текста для синтеза.\n\n"
            "Глава 2\n"
            "Это вторая глава. Еще немного текста.\n"
        )
        response = client.post(
            "/jobs",
            files={"file": ("book.txt", book_text.encode("utf-8"), "text/plain")},
            data={"speaker_id": "1", "speed": "1.0", "use_gpu": "false"},
        )
        assert response.status_code == 200
        job_id = response.json()["job_id"]

        # Wait for background queue completion in test environment.
        main.runner._queue.join()

        status_resp = client.get(f"/jobs/{job_id}")
        assert status_resp.status_code == 200
        job_data = status_resp.json()
        assert job_data["status"] == JobStatus.DONE.value
        assert job_data["progress"] == 1.0
        assert job_data["meta"]["engine"] == "fake_tts"
        assert Path(job_data["output_dir"]).name == "book"

        ui_resp = client.get("/")
        assert ui_resp.status_code == 200
        assert "Конвертер книги в аудио" in ui_resp.text

    out_job_dir = output_dir / "book"
    manifest_path = out_job_dir / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["chapters"]) == 2
    assert len(manifest["files"]) >= 2

    for item in manifest["files"]:
        chapter = item["chapter"]
        part = item["part"]
        filename = item["file"]
        assert filename == f"chapter_{chapter:03d}_part_{part:03d}.mp3"
        generated = out_job_dir / filename
        assert generated.exists()
