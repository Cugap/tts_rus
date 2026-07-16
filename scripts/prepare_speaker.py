"""
Скрипт подготовки референсного аудио для клонирования голоса (XTTS v2).

Качество клонированного голоса напрямую зависит от качества референсной записи.
Этот скрипт автоматически:
1. Обрезает тишину в начале и конце
2. Нормализует пиковую громкость (до -1 dB)
3. Передискретизирует до 22050 Гц (оптимально для XTTS)
4. Конвертирует в WAV (16-bit PCM, моно)
5. Обрезает до нужной длины (рекомендуется 6-30 секунд)

Использование:
    python scripts/prepare_speaker.py input.mp3 -o speakers/my_voice.wav
    python scripts/prepare_speaker.py input.wav -o speakers/my_voice.wav --max-duration 15
"""

from __future__ import annotations

import argparse
import wave
from pathlib import Path

import numpy as np


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    """Read any audio file via soundfile (WAV, MP3, FLAC, OGG etc.)."""
    import soundfile as sf

    data, sr = sf.read(str(path), always_2d=True, dtype="float32")
    # soundfile returns (samples, channels)
    if data.shape[1] > 1:
        # Convert to mono by averaging channels
        data = data.mean(axis=1)
    else:
        data = data.flatten()
    return data, sr


def resample(data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to target sample rate using linear interpolation."""
    if orig_sr == target_sr:
        return data
    duration = len(data) / orig_sr
    target_len = int(duration * target_sr)
    indices = np.linspace(0, len(data) - 1, target_len)
    return np.interp(indices, np.arange(len(data)), data)


def trim_silence(
    data: np.ndarray, sr: int, threshold_db: float = -40.0, min_duration: float = 0.1
) -> np.ndarray:
    """Trim leading and trailing silence."""
    threshold = 10.0 ** (threshold_db / 20.0)
    mask = np.abs(data) > threshold
    if not mask.any():
        return data
    safety = int(sr * min_duration)
    start = max(0, int(np.argmax(mask)) - safety)
    end = min(len(data), len(data) - int(np.argmax(mask[::-1])) + safety)
    return data[start:end]


def normalize_peak(data: np.ndarray, target_db: float = -1.0) -> np.ndarray:
    """Normalize peak amplitude to target dBFS."""
    peak = np.max(np.abs(data))
    if peak <= 0:
        return data
    target_amp = 10.0 ** (target_db / 20.0)
    gain = target_amp / peak
    return np.clip(data * gain, -1.0, 1.0)


def trim_to_duration(data: np.ndarray, sr: int, max_seconds: float) -> np.ndarray:
    """Trim/crop audio to at most max_seconds."""
    max_samples = int(max_seconds * sr)
    if len(data) <= max_samples:
        return data
    # Take from the middle — usually the most stable part
    start = (len(data) - max_samples) // 2
    return data[start : start + max_samples]


def write_wav(data: np.ndarray, sr: int, path: Path) -> None:
    """Write float32 mono audio as 16-bit PCM WAV."""
    pcm16 = (data * 32767.0).astype(np.int16).tobytes()
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Подготовка референсного аудио для XTTS v2"
    )
    parser.add_argument("input", type=Path, help="Входной аудиофайл (WAV, MP3, FLAC, OGG...)")
    parser.add_argument("-o", "--output", type=Path, default=Path("speakers/my_voice.wav"),
                        help="Выходной WAV (по умолчанию: speakers/my_voice.wav)")
    parser.add_argument("--target-sr", type=int, default=22050,
                        help="Целевая частота дискретизации (по умолчанию: 22050)")
    parser.add_argument("--max-duration", type=float, default=20.0,
                        help="Максимальная длительность в секундах (по умолчанию: 20)")
    parser.add_argument("--trim-threshold", type=float, default=-40.0,
                        help="Порог тишины в dB (по умолчанию: -40)")
    args = parser.parse_args()

    if not args.input.exists():
        parser.error(f"Файл не найден: {args.input}")

    print(f"📂 Читаем: {args.input}")
    data, orig_sr = read_audio(args.input)
    duration = len(data) / orig_sr
    print(f"   Формат: {orig_sr} Гц, {duration:.1f} сек")

    # 1. Resample
    if orig_sr != args.target_sr:
        print(f"🔄 Передискретизация: {orig_sr} → {args.target_sr} Гц")
        data = resample(data, orig_sr, args.target_sr)

    # 2. Trim silence
    print(f"✂️  Обрезка тишины (порог: {args.trim_threshold} dB)...")
    data = trim_silence(data, args.target_sr, args.trim_threshold)
    trimmed_dur = len(data) / args.target_sr
    print(f"   После обрезки: {trimmed_dur:.1f} сек")

    # 3. Normalize peak
    print("🔊 Нормализация громкости...")
    data = normalize_peak(data)

    # 4. Trim to max duration
    if trimmed_dur > args.max_duration:
        print(f"✂️  Обрезка до {args.max_duration} сек (берём середину)...")
        data = trim_to_duration(data, args.target_sr, args.max_duration)
        final_dur = len(data) / args.target_sr
        print(f"   После обрезки: {final_dur:.1f} сек")

    # 5. Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"💾 Сохраняем: {args.output}")
    write_wav(data, args.target_sr, args.output)
    print("✅ Готово! Референсный голос подготовлен.")

    # Recommendation
    final_dur = len(data) / args.target_sr
    if final_dur < 3:
        print("⚠️  Внимание: длительность менее 3 секунд — качество клонирования может быть низким.")
    elif final_dur > 30:
        print("💡 Совет: для клонирования достаточно 6-20 секунд чистой речи.")


if __name__ == "__main__":
    main()
