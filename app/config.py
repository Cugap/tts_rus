from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    base_dir: Path = Path(__file__).resolve().parent.parent
    output_dir: Path = base_dir / "output"
    db_path: Path = base_dir / "jobs.db"

    # TTS Settings
    max_chars_per_chunk: int = 600
    min_chars_per_chunk: int = 400
    memory_reduction_factor: float = 0.7

    # MP3 Settings
    mp3_bitrate_kbps: int = 128
    mp3_quality_normal: int = 2

    # XTTS Docker API Settings
    xtts_api_url: str = "http://localhost:8020/tts_to_audio/"
    xtts_speaker_wav: str = "default_speaker.wav"
    xtts_language: str = "ru"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
settings.output_dir.mkdir(parents=True, exist_ok=True)

# Backward compatibility aliases
BASE_DIR = settings.base_dir
OUTPUT_DIR = settings.output_dir
DB_PATH = settings.db_path
