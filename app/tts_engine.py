from __future__ import annotations

import gc
import subprocess
import tempfile
import wave
from pathlib import Path

import psutil

import httpx
from loguru import logger
from app.config import settings

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


class TTSEngine:
    def __init__(
        self, voice: str = "default", speed: float = 1.0, use_gpu: bool = True
    ):
        self.voice = voice
        self.speed = speed
        self.use_gpu = use_gpu
        self.device = "cpu"
        self._engine_mode = "sapi"
        self._coqui = None
        self._hf_model = None
        self._hf_tokenizer = None

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

        # Check if XTTS Docker API is available
        try:
            response = httpx.get(
                settings.xtts_api_url.replace("/tts_to_audio/", "/languages"),
                timeout=2.0,
            )
            if response.status_code == 200:
                logger.info("Connected to XTTS Docker API")
                self._engine_mode = "xtts_api"
                return
        except Exception:
            logger.debug("XTTS Docker API not available, falling back to local models")

        # Preferred local option: HF VITS model from project directory.
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

        # Primary option: Coqui TTS with CUDA/CPU.
        try:
            from TTS.api import TTS as CoquiTTS  # type: ignore

            model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
            self._coqui = CoquiTTS(model_name=model_name, progress_bar=False).to(
                self.device
            )
            self._engine_mode = "coqui"
            return
        except Exception:
            self._coqui = None

        # Fallback option: Windows SAPI via PowerShell.
        self._engine_mode = "sapi"

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

        if self._engine_mode == "coqui" and self._coqui is not None:
            try:
                self._coqui.tts_to_file(text=text, file_path=str(temp_wav))
            except RuntimeError as err:
                if "out of memory" in str(err).lower():
                    self._switch_to_cpu()
                    self._coqui.tts_to_file(text=text, file_path=str(temp_wav))
                else:
                    raise

        elif (
            self._engine_mode == "hf_vits_local"
            and self._hf_model is not None
            and self._hf_tokenizer is not None
            and torch is not None
        ):
            self._hf_vits_to_file(text=text, out_path=temp_wav)

        elif self._engine_mode == "xtts_api":
            self._xtts_api_to_file(text=text, out_path=temp_wav)

        elif self._engine_mode == "sapi":
            self._sapi_to_file(text=text, out_path=temp_wav)

        else:
            raise RuntimeError("No available TTS engine")

        if is_mp3:
            self._convert_wav_to_mp3(temp_wav, out_path)

    def _convert_wav_to_mp3(self, wav_path: Path, mp3_path: Path) -> None:
        import lameenc

        MP3_BITRATE_KBPS = 128
        MP3_QUALITY_NORMAL = 2

        with wave.open(str(wav_path), "rb") as wav:
            sample_rate = wav.getframerate()
            channels = wav.getnchannels()
            pcm_data = wav.readframes(wav.getnframes())

        encoder = lameenc.Encoder()
        encoder.set_bit_rate(MP3_BITRATE_KBPS)
        encoder.set_in_sample_rate(sample_rate)
        encoder.set_channels(channels)
        encoder.set_quality(MP3_QUALITY_NORMAL)

        mp3_data = encoder.encode(pcm_data)
        mp3_data += encoder.flush()

        mp3_path.write_bytes(mp3_data)
        wav_path.unlink(missing_ok=True)

    def _xtts_api_to_file(self, text: str, out_path: Path) -> None:
        payload = {
            "text": text,
            "speaker_wav": settings.xtts_speaker_wav,
            "language": settings.xtts_language,
        }
        try:
            response = httpx.post(settings.xtts_api_url, json=payload, timeout=120.0)
            response.raise_for_status()
            out_path.write_bytes(response.content)
        except Exception as e:
            raise RuntimeError(f"XTTS API synthesis failed: {e}")

    def _sapi_to_file(self, text: str, out_path: Path) -> None:
        SAPI_MIN_RATE = -10
        SAPI_MAX_RATE = 10
        SAPI_RATE_MULTIPLIER = 10

        path_ps = str(out_path).replace("'", "''")
        with tempfile.NamedTemporaryFile(
            "w", suffix=".txt", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(text)
            text_path = tmp.name
        text_path_ps = text_path.replace("'", "''")

        # SAPI speed range is from -10 to 10.
        rate = max(
            SAPI_MIN_RATE,
            min(SAPI_MAX_RATE, int((self.speed - 1.0) * SAPI_RATE_MULTIPLIER)),
        )
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$text = Get-Content -LiteralPath '{text_path_ps}' -Raw -Encoding UTF8; "
            f"$s.Rate = {rate}; "
            f"$s.SetOutputToWaveFile('{path_ps}'); "
            "$s.Speak($text); "
            "$s.Dispose(); "
            f"Remove-Item -LiteralPath '{text_path_ps}' -Force;"
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
        normalized_text = text.lower()
        inputs = self._hf_tokenizer(normalized_text, return_tensors="pt")
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

    @staticmethod
    def _parse_speaker_id(value: str) -> int:
        normalized = (value or "").strip().lower()
        if normalized in {"1", "m", "male", "man", "м", "муж", "мужской"}:
            return 1
        return 0

    def _switch_to_cpu(self) -> None:
        if self._coqui is None:
            return
        self.device = "cpu"
        self._coqui.to("cpu")
        if torch is not None and hasattr(torch, "cuda"):
            torch.cuda.empty_cache()
        gc.collect()
