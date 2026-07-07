from __future__ import annotations

import base64
import os
import subprocess
import sys
import wave
from pathlib import Path

import psutil

from loguru import logger

from app.config import settings

# ── FFmpeg DLL search path (needed by torchcodec on Windows) ────────────────
_ffmpeg_dll_dir = os.path.join(os.environ.get("USERPROFILE", ""), ".ffmpeg-dlls")
if os.path.isdir(_ffmpeg_dll_dir):
    os.environ["PATH"] = _ffmpeg_dll_dir + os.pathsep + os.environ.get("PATH", "")
    try:
        os.add_dll_directory(_ffmpeg_dll_dir)
    except Exception:
        pass

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


XTTS_MODEL_DIR = Path(__file__).resolve().parent.parent / "xtts_models" / "v2.0.2"
XTTS_DEFAULT_SPEAKER = Path(__file__).resolve().parent.parent / "speakers" / "my_voice.wav"
XTTS_HELPER_SCRIPT = Path(__file__).resolve().parent / "_xtts_synthesize.py"
XTTS_PYTHON = Path(os.environ.get("XTTS_PYTHON", sys.executable))


def _xtts_python() -> str:
    return str(XTTS_PYTHON)


_XTTS_MODEL_CACHE: dict[str, object] = {}
_XTTS_MARKER = object()  # Sentinel: use subprocess instead of in-process


def _apply_tts_shims() -> None:
    """Apply compatibility shims for coqui-tts on newer torch/transformers."""
    try:
        import transformers.pytorch_utils as _tpu
        if not hasattr(_tpu, "isin_mps_friendly") and torch is not None:
            _tpu.isin_mps_friendly = lambda elements, test_elements, **kw: torch.isin(elements, test_elements)
    except Exception:
        pass
    try:
        import librosa
        import numpy as _np
        if not hasattr(librosa, "magphase"):
            librosa.magphase = lambda D, power=1.0: (_np.abs(D) ** power, _np.exp(1j * _np.angle(D)))
        if not hasattr(librosa, "pyin"):
            librosa.pyin = lambda *a, **kw: (None, None, None)
    except Exception:
        pass
    try:
        import soundfile as _sf
        import torchaudio as _ta

        def _patched_load(audiopath: str, **kwargs) -> tuple:
            out_frames = kwargs.pop("out_frames", None)
            # always_2d=True → returns (channels, samples) for mono → 2D tensor
            data, sr = _sf.read(audiopath, always_2d=True, dtype="float32")
            # soundfile returns (samples, channels), transpose to (channels, samples)
            import numpy as _np2
            if data.ndim == 2:
                data = data.T
            else:
                data = data.reshape(1, -1)
            tensor = torch.from_numpy(data)
            if out_frames is not None:
                tensor = tensor[:, :out_frames]
            return tensor, sr

        _ta.load = _patched_load
    except Exception:
        pass


def _load_xtts_model(device: str = "cpu") -> object:
    """Load XTTS model in-process and return the model object."""
    _apply_tts_shims()
    from TTS.api import TTS  # type: ignore
    tts = TTS(
        model_path=str(XTTS_MODEL_DIR),
        config_path=str(XTTS_MODEL_DIR / "config.json"),
    )
    tts = tts.to(device)
    return tts


def _check_xtts_subprocess() -> None:
    """Verify that the helper script can run (fallback path)."""
    if not XTTS_HELPER_SCRIPT.exists():
        raise RuntimeError(
            "XTTS v2 requires coqui-tts (pip install coqui-tts). "
            "See _xtts_synthesize.py for details."
        )
    result = subprocess.run(
        [_xtts_python(), str(XTTS_HELPER_SCRIPT), "--check"],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"XTTS не доступен: {result.stderr.strip() or result.stdout.strip()}"
        )


class TTSEngine:
    def __init__(
        self, engine: str = "auto", voice: str = "default", speed: float = 1.0, use_gpu: bool = True
    ):
        self.requested_engine = engine
        self.voice = voice
        self.speed = speed
        self.use_gpu = use_gpu
        self.device = "cpu"
        self._engine_mode = "sapi"
        self._hf_model = None
        self._hf_tokenizer = None
        self._xtts_model = None
        self._xtts_speaker_wav: str | None = None

        self._init_engine()

    @property
    def engine_mode(self) -> str:
        return self._engine_mode

    def _init_engine(self) -> None:
        can_cuda = bool(
            self.use_gpu
            and torch is not None
            and hasattr(torch, "cuda")
            and torch.cuda.is_available()
        )
        self.device = "cuda" if can_cuda else "cpu"

        # ── XTTS v2 ────────────────────────────────────────────────────────
        if self.requested_engine in ("auto", "xtts"):
            xtts_config = XTTS_MODEL_DIR / "config.json"
            if XTTS_MODEL_DIR.exists() and xtts_config.exists():
                try:
                    # Try in-process loading; fall back to subprocess marker
                    _cache_key = str(XTTS_MODEL_DIR)
                    if _cache_key not in _XTTS_MODEL_CACHE:
                        _XTTS_MODEL_CACHE[_cache_key] = _load_xtts_model(self.device)
                    self._xtts_model = _XTTS_MODEL_CACHE[_cache_key]
                    self._engine_mode = "xtts"
                    self._xtts_speaker_wav = self._resolve_speaker_wav(self.voice)
                    logger.info("XTTS v2 загружен in-process")
                    return
                except ImportError:
                    # TTS library not importable — use subprocess path
                    logger.warning("TTS import failed, используем subprocess для XTTS")
                    self._xtts_model = _XTTS_MARKER
                    _check_xtts_subprocess()
                    self._engine_mode = "xtts"
                    self._xtts_speaker_wav = self._resolve_speaker_wav(self.voice)
                    return
                except Exception as _xtts_err:
                    logger.error(f"XTTS v2 загрузка не удалась: {_xtts_err}")
                    self._xtts_model = None
                    if self.requested_engine == "xtts":
                        raise RuntimeError(
                            f"XTTS v2 requested but failed to load: {_xtts_err}"
                        ) from _xtts_err
            elif self.requested_engine == "xtts":
                raise RuntimeError("XTTS v2 requested but model not found in xtts_models/v2.0.2/")

        # ── VITS (HF) ──────────────────────────────────────────────────────
        if self.requested_engine in ("auto", "hf_vits_local"):
            hf_model_dir = (
                Path(__file__).resolve().parent.parent
                / "tts_ru_free_hf_vits_high_multispeaker"
            )
            if hf_model_dir.exists():
                try:
                    from transformers import AutoTokenizer, VitsModel  # type: ignore

                    self._hf_model = VitsModel.from_pretrained(str(hf_model_dir)).to(
                        self.device
                    )
                    self._hf_model.eval()
                    self._hf_tokenizer = AutoTokenizer.from_pretrained(str(hf_model_dir))
                    self._engine_mode = "hf_vits_local"
                    return
                except Exception:
                    self._hf_model = None
                    self._hf_tokenizer = None
                    if self.requested_engine == "hf_vits_local":
                        raise RuntimeError("HF VITS requested but failed to load")
            elif self.requested_engine == "hf_vits_local":
                raise RuntimeError("HF VITS requested but model not found")

        if self.requested_engine not in ("auto", "sapi"):
            raise RuntimeError(f"Requested engine '{self.requested_engine}' is not available.")

        self._engine_mode = "sapi"

    def _resolve_speaker_wav(self, voice: str) -> str:
        """Resolve a speaker reference audio path from voice setting."""
        custom = Path(voice)
        if custom.exists():
            return str(custom)
        if XTTS_DEFAULT_SPEAKER.exists():
            return str(XTTS_DEFAULT_SPEAKER)
        # Fallback: use the built-in speaker embedding
        return str(XTTS_MODEL_DIR / "speakers_xtts.pth")

    @staticmethod
    def _memory_guard(min_free_mb: int = 300) -> None:
        BYTES_IN_MB = 1024 * 1024
        free_mb = psutil.virtual_memory().available / BYTES_IN_MB
        if free_mb < min_free_mb:
            raise MemoryError(f"Low RAM: available {free_mb:.1f}MB")

    def synthesize_to_file(self, text: str, out_path: Path) -> None:
        self._memory_guard()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        is_mp3 = out_path.suffix.lower() == ".mp3"
        temp_wav = out_path.with_suffix(".wav") if is_mp3 else out_path

        if self._engine_mode == "xtts" and self._xtts_model is not None:
            self._xtts_to_file(text=text, out_path=temp_wav)
        elif (
            self._engine_mode == "hf_vits_local"
            and self._hf_model is not None
            and self._hf_tokenizer is not None
            and torch is not None
        ):
            self._hf_vits_to_file(text=text, out_path=temp_wav)
        elif self._engine_mode == "sapi":
            self._sapi_to_file(text=text, out_path=temp_wav)
        else:
            raise RuntimeError("No available TTS engine")

        if is_mp3:
            self._convert_wav_to_mp3(temp_wav, out_path)

    def _convert_wav_to_mp3(self, wav_path: Path, mp3_path: Path) -> None:
        import lameenc

        with wave.open(str(wav_path), "rb") as wav:
            sample_rate = wav.getframerate()
            channels = wav.getnchannels()
            pcm_data = wav.readframes(wav.getnframes())

        encoder = lameenc.Encoder()
        encoder.set_bit_rate(settings.mp3_bitrate_kbps)
        encoder.set_in_sample_rate(sample_rate)
        encoder.set_channels(channels)
        encoder.set_quality(settings.mp3_quality_normal)

        mp3_data = encoder.encode(pcm_data)
        mp3_data += encoder.flush()

        mp3_path.write_bytes(mp3_data)
        wav_path.unlink(missing_ok=True)

    def _sapi_to_file(self, text: str, out_path: Path) -> None:
        SAPI_MIN_RATE = -10
        SAPI_MAX_RATE = 10
        SAPI_RATE_MULTIPLIER = 10

        rate = max(
            SAPI_MIN_RATE,
            min(SAPI_MAX_RATE, int((self.speed - 1.0) * SAPI_RATE_MULTIPLIER)),
        )
        # Base64-encode the text to avoid PowerShell escaping issues
        text_b64 = base64.b64encode(text.encode("utf-8")).decode("ascii")
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$bytes = [Convert]::FromBase64String('{text_b64}'); "
            "$text = [System.Text.Encoding]::UTF8.GetString($bytes); "
            f"$s.Rate = {rate}; "
            f"$s.SetOutputToWaveFile('{out_path}'); "
            "$s.Speak($text); "
            "$s.Dispose();"
        )
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            message = (result.stderr or result.stdout or "Unknown SAPI error").strip()
            raise RuntimeError(f"SAPI synthesis failed: {message}")

    def _hf_vits_to_file(self, text: str, out_path: Path) -> None:
        PCM_16BIT_MAX = 32767.0
        WAV_CHANNELS = 1
        WAV_SAMPLE_WIDTH_BYTES = 2

        assert self._hf_model is not None
        assert self._hf_tokenizer is not None
        assert torch is not None

        speaker = self._parse_speaker_id(self.voice)
        inputs = self._hf_tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = self._hf_model(**inputs, speaker_id=speaker).waveform
            waveform = output.squeeze(0).detach().cpu().clamp(-1.0, 1.0)
            pcm16 = (waveform * PCM_16BIT_MAX).to(torch.int16).numpy().tobytes()

        with wave.open(str(out_path), "wb") as wav_file:
            wav_file.setnchannels(WAV_CHANNELS)
            wav_file.setsampwidth(WAV_SAMPLE_WIDTH_BYTES)
            wav_file.setframerate(int(self._hf_model.config.sampling_rate))
            wav_file.writeframes(pcm16)

    def _xtts_to_file(self, text: str, out_path: Path) -> None:
        assert self._xtts_model is not None

        _err_output: str = ""

        # In-process path: model object has tts_to_file
        if self._xtts_model is not _XTTS_MARKER:
            try:
                logger.debug(f"XTTS in-process синтез: {len(text)} символов")
                self._xtts_model.tts_to_file(
                    text=text,
                    speaker_wav=self._xtts_speaker_wav,
                    language="ru",
                    file_path=str(out_path),
                )
                logger.debug(f"XTTS in-process синтез завершён: {out_path}")
                return
            except Exception as _inproc_err:
                _err_output = str(_inproc_err)
                logger.error(f"XTTS in-process синтез не удался: {_inproc_err}")

        # Subprocess fallback
        logger.debug(f"XTTS subprocess синтез: {len(text)} символов")
        result = subprocess.run(
            [
                _xtts_python(),
                str(XTTS_HELPER_SCRIPT),
                str(XTTS_MODEL_DIR),
                self._xtts_speaker_wav or str(XTTS_DEFAULT_SPEAKER),
                str(out_path),
                text,
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            _msg = result.stderr.strip() or result.stdout.strip() or _err_output
            logger.error(f"XTTS subprocess ошибка (rc={result.returncode}): {_msg[:500]}")
            if "libtorchcodec" in _msg or "FFmpeg" in _msg:
                raise RuntimeError(
                    "XTTS требует FFmpeg на Windows. Установите FFmpeg (full-shared) "
                    "или используйте отдельный Python 3.11/3.12 venv.\n"
                    "Скачать FFmpeg: https://ffmpeg.org/download.html"
                )
            raise RuntimeError(f"XTTS subprocess failed: {_msg}")

    @staticmethod
    def _parse_speaker_id(value: str) -> int:
        normalized = (value or "").strip().lower()
        if normalized in {"1", "m", "male", "man", "м", "муж", "мужской"}:
            return 1
        return 0
