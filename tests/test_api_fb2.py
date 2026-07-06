from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app import main


def test_validate_book_endpoint_accepts_sample() -> None:
    sample = Path(__file__).resolve().parent.parent / "book_example" / "sample.fb2"
    with TestClient(main.app) as client:
        response = client.post(
            "/books/validate",
            files={"file": ("sample.fb2", sample.read_bytes(), "application/xml")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["issues"] == []


def test_validate_book_endpoint_returns_stats(tmp_path: Path) -> None:
    sample = Path(__file__).resolve().parent.parent / "book_example" / "sample.fb2"
    with TestClient(main.app) as client:
        response = client.post(
            "/books/validate",
            files={"file": ("sample.fb2", sample.read_bytes(), "application/xml")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert "stats" in payload
    assert payload["stats"]["total_chars"] > 0
    assert payload["stats"]["paragraphs"] > 0
    assert "max_paragraph_len" in payload["stats"]
    assert "letter_chars" in payload["stats"]


def test_validate_book_endpoint_rejects_broken_fb2(tmp_path: Path) -> None:
    broken = tmp_path / "broken.fb2"
    broken.write_text("<FictionBook>", encoding="utf-8")

    with TestClient(main.app) as client:
        response = client.post(
            "/books/validate",
            files={"file": ("broken.fb2", broken.read_bytes(), "application/xml")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is False
    assert payload["issues"]


def test_create_job_rejects_invalid_fb2(tmp_path: Path, monkeypatch) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "uploads").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(main.settings, "output_dir", output_dir)

    broken = b"<FictionBook>"
    with TestClient(main.app) as client:
        response = client.post(
            "/jobs",
            files={"file": ("broken.fb2", broken, "application/xml")},
            data={"engine": "auto", "speaker": "0", "speed": "1.0", "use_gpu": "false"},
        )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["message"] == "FB2 не подходит для озвучки"
    assert detail["issues"]
