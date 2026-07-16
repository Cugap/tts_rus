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

    # XTTS v2 character limit per utterance (~182 для русского языка)
    xtts_max_chars: int = 180

    # MP3 Settings
    mp3_bitrate_kbps: int = 192       # Повышен с 128 для лучшего качества
    mp3_quality_normal: int = 2
    mp3_use_vbr: bool = True          # Использовать VBR вместо CBR
    mp3_vbr_quality: float = 2.0      # VBR quality (0=лучшее, 9=худшее)

    # XTTS v2 generation parameters (для стабильности и качества)
    xtts_temperature: float = 0.65      # Чуть ниже дефолта — стабильнее
    xtts_top_k: int = 50                # Top-K sampling
    xtts_top_p: float = 0.85            # Nucleus sampling
    xtts_repetition_penalty: float = 5.0  # Защита от зацикливания

    # Audio post-processing
    audio_normalize_peak: bool = True   # Нормализация пиковой громкости
    audio_trim_silence: bool = True     # Обрезка тишины в начале/конце
    audio_trim_threshold_db: float = -40.0  # Порог тишины в dB

    # Concat Settings (склеивание всех частей в один файл)
    concat_enabled: bool = True
    concat_filename: str = "{book_name}_full.mp3"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
settings.output_dir.mkdir(parents=True, exist_ok=True)
