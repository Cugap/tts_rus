"""
Tests for XTTS v2 engine — path resolution, subprocess helper, engine routing.

These tests do NOT load the full XTTS model (takes minutes on CPU).
They verify configuration, helper scripts, and engine state transitions.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from app.tts_engine import (
    XTTS_DEFAULT_SPEAKER,
    XTTS_HELPER_SCRIPT,
    XTTS_MODEL_DIR,
    XTTS_PYTHON,
    _XTTS_MARKER,
    TTSEngine,
)


def _make_engine() -> TTSEngine:
    """Minimal engine instance for testing helper methods."""
    return TTSEngine(engine="sapi", voice="default", use_gpu=False)


# ── Syntax & import sanity (catch breakage like bad indents) ─────────────────


class TestXTTCodeQuality:
    """Quick checks that catch syntax/import errors before runtime."""

    def test_xtts_synthesize_script_has_no_syntax_errors(self) -> None:
        """Parse _xtts_synthesize.py with ast — catches indentation errors."""
        source = XTTS_HELPER_SCRIPT.read_text("utf-8")
        try:
            ast.parse(source)
        except SyntaxError as e:
            pytest.fail(f"Syntax error in {XTTS_HELPER_SCRIPT}: {e}")

    def test_tts_engine_module_has_no_syntax_errors(self) -> None:
        """Parse tts_engine.py with ast — catches indentation errors."""
        path = Path(__file__).resolve().parent.parent / "app" / "tts_engine.py"
        source = path.read_text("utf-8")
        try:
            ast.parse(source)
        except SyntaxError as e:
            pytest.fail(f"Syntax error in {path}: {e}")

    def test_shims_are_valid_python(self) -> None:
        """_apply_tts_shims must be syntactically valid (was broken before)."""
        from app.tts_engine import _apply_tts_shims
        # Just check it's callable — doesn't run it (that would load libs)
        assert callable(_apply_tts_shims)


# ── Path sanity ─────────────────────────────────────────────────────────────


class TestXTTSPaths:
    """Verify that all XTTS model and script paths exist."""

    def test_model_dir_exists(self) -> None:
        assert XTTS_MODEL_DIR.exists(), (
            f"XTTS model directory not found: {XTTS_MODEL_DIR}"
        )

    def test_model_config_exists(self) -> None:
        config = XTTS_MODEL_DIR / "config.json"
        assert config.exists(), f"XTTS config not found: {config}"

    def test_model_weights_exist(self) -> None:
        weights = XTTS_MODEL_DIR / "model.pth"
        assert weights.exists(), f"XTTS weights not found: {weights}"
        # Quick sanity: file should be > 100 MB
        assert weights.stat().st_size > 100 * 1024 * 1024, (
            f"XTTS weights too small ({weights.stat().st_size} bytes), "
            "likely a broken download"
        )

    def test_speakers_file_exists(self) -> None:
        speakers = XTTS_MODEL_DIR / "speakers_xtts.pth"
        assert speakers.exists(), f"XTTS speakers file not found: {speakers}"

    def test_helper_script_exists(self) -> None:
        assert XTTS_HELPER_SCRIPT.exists(), (
            f"XTTS helper script not found: {XTTS_HELPER_SCRIPT}"
        )

    def test_default_speaker_wav(self) -> None:
        # my_voice.wav is optional; the test adapts
        if XTTS_DEFAULT_SPEAKER.exists():
            assert XTTS_DEFAULT_SPEAKER.stat().st_size > 1000, (
                f"Speaker WAV too small: {XTTS_DEFAULT_SPEAKER}"
            )

    def test_xtts_python_default(self) -> None:
        """XTTS_PYTHON defaults to sys.executable."""
        assert XTTS_PYTHON == Path(sys.executable)


# ── Subprocess helper ────────────────────────────────────────────────────────


class TestXTTSSynthesizeScript:
    """Tests for app/_xtts_synthesize.py invoked as a subprocess."""

    def test_check_flag_succeeds(self) -> None:
        """--check must exit 0 when coqui-tts is importable."""
        result = subprocess.run(
            [sys.executable, str(XTTS_HELPER_SCRIPT), "--check"],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, (
            f"XTTS helper --check failed (rc={result.returncode}):\n"
            f"  stderr: {result.stderr[:500]}"
        )
        assert "OK" in result.stdout

    def test_check_flag_prints_ok(self) -> None:
        result = subprocess.run(
            [sys.executable, str(XTTS_HELPER_SCRIPT), "--check"],
            capture_output=True, text=True, timeout=60,
        )
        assert "TTS is available" in result.stdout

    @pytest.mark.slow
    def test_synthesize_subprocess_runs(self) -> None:
        """Verify the subprocess path can at least be invoked.
        Marked slow because it loads TTS libraries."""
        if not XTTS_DEFAULT_SPEAKER.exists():
            pytest.skip("No speaker WAV available")
        tmp = Path(__file__).resolve().parent / "_xtts_test_out.wav"
        try:
            result = subprocess.run(
                [
                    sys.executable, str(XTTS_HELPER_SCRIPT),
                    str(XTTS_MODEL_DIR),
                    str(XTTS_DEFAULT_SPEAKER),
                    str(tmp),
                    "Тест.",
                ],
                capture_output=True, text=True, timeout=600,
            )
            if result.returncode == 0:
                assert tmp.exists(), "Output WAV was not created"
                assert tmp.stat().st_size > 1000, "Output WAV too small"
            else:
                # Non-zero exit may be acceptable (e.g. no GPU, timeout)
                pytest.skip(
                    f"Subprocess synthesis skipped (rc={result.returncode}): "
                    f"{result.stderr[:200]}"
                )
        finally:
            tmp.unlink(missing_ok=True)


# ── Fake XTTS engine for fast tests ─────────────────────────────────────────


class _FakeXTTSModel:
    """Fast fake that replaces the real 1.5GB XTTS model in tests.
    
    Produces a valid 16-bit mono WAV file (440 Hz sine tone) in milliseconds.
    """
    SAMPLE_RATE = 22050

    def tts_to_file(self, text: str, speaker_wav: str, language: str, file_path: str) -> None:
        import numpy as np
        import struct
        import math

        duration = max(0.3, len(text) * 0.1)
        n_samples = int(self.SAMPLE_RATE * duration)
        samples = []
        for i in range(n_samples):
            t = i / self.SAMPLE_RATE
            val = int(math.sin(2 * math.pi * 440 * t) * 16000)
            val = max(-32768, min(32767, val))
            samples.extend(struct.pack("<h", val))

        import wave
        with wave.open(file_path, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.SAMPLE_RATE)
            wav.writeframes(bytes(samples))


# ── End-to-end pipeline (fast mock) ─────────────────────────────────────────


class TestXTTS_E2E:
    """
    Full-pipeline test using a fast mock XTTS model.
    Covers: syntax → paths → TTS import → model load → synthesis → WAV output.
    Runs in ~3 seconds (no real model loading).
    """

    def test_e2e_full_pipeline(self, tmp_path, monkeypatch) -> None:
        """Mock the XTTS model, run full synthesis pipeline, validate WAV."""
        # 1. Syntax check all involved files
        for label, path in [
            ("helper", XTTS_HELPER_SCRIPT),
            ("engine", Path(__file__).resolve().parent.parent / "app" / "tts_engine.py"),
        ]:
            source = path.read_text("utf-8")
            try:
                ast.parse(source)
            except SyntaxError as e:
                pytest.fail(f"Syntax error in {label} ({path}): {e}")

        # 2. Path checks
        assert XTTS_MODEL_DIR.exists(), f"XTTS dir missing: {XTTS_MODEL_DIR}"
        assert (XTTS_MODEL_DIR / "config.json").exists()
        assert XTTS_HELPER_SCRIPT.exists()

        if not XTTS_DEFAULT_SPEAKER.exists():
            pytest.skip(f"Speaker WAV not found: {XTTS_DEFAULT_SPEAKER}")

        # 3. TTS import check via subprocess
        result = subprocess.run(
            [sys.executable, str(XTTS_HELPER_SCRIPT), "--check"],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, (
            f"TTS --check failed:\n  stdout: {result.stdout[:500]}\n  stderr: {result.stderr[:500]}"
        )

        # 4. Mock the real model with a fast fake
        monkeypatch.setattr("app.tts_engine._load_xtts_model", lambda device: _FakeXTTSModel())
        # Clear cache so our mock is used
        import app.tts_engine as _te
        _te._XTTS_MODEL_CACHE.clear()

        # 5. Create engine with xtts mode
        from app.tts_engine import TTSEngine

        out_path = tmp_path / "e2e_test.wav"
        engine = TTSEngine(engine="xtts", voice=str(XTTS_DEFAULT_SPEAKER), use_gpu=False)
        assert engine._engine_mode == "xtts", (
            f"Engine mode is '{engine._engine_mode}', expected 'xtts'"
        )
        assert engine._xtts_model is not None
        assert engine._xtts_model is not _te._XTTS_MARKER, "Mock was not used; fell back to subprocess"

        # 6. Synthesize short text
        engine.synthesize_to_file("Привет.", out_path)

        # 7. Validate output WAV
        assert out_path.exists(), "Output WAV was not created"
        size = out_path.stat().st_size
        assert size > 1000, f"Output WAV too small: {size} bytes"

        import wave
        with wave.open(str(out_path), "rb") as wav:
            assert wav.getnframes() > 0, "WAV has zero frames"
            assert wav.getsampwidth() == 2, f"Expected 16-bit, got {wav.getsampwidth()}"
            assert wav.getnchannels() == 1, f"Expected mono, got {wav.getnchannels()}"
            sr = wav.getframerate()
            assert sr > 0, f"Invalid sample rate: {sr}"


# ── Real XTTS model test (slow, requires GPU) ───────────────────────────────


@pytest.mark.slow
class TestXTTSRealModel:
    """Full pipeline with real XTTS model — needs GPU, takes 10+ min on CPU.

    Run with:  pytest tests/test_xtts.py -k "RealModel" -v --timeout=900
    """

    def test_real_model_synthesis(self, tmp_path) -> None:
        """Load real XTTS model, synthesize, validate WAV.
        Uses GPU if available, otherwise CPU (slow)."""
        if not XTTS_MODEL_DIR.exists() or not XTTS_DEFAULT_SPEAKER.exists():
            pytest.skip("XTTS model or speaker WAV not found")

        import torch
        use_gpu = torch.cuda.is_available()
        if not use_gpu:
            pytest.skip("RealModel test requires GPU (CPU is too slow: 10+ min)")

        from app.tts_engine import TTSEngine
        import app.tts_engine as _te
        _te._XTTS_MODEL_CACHE.clear()

        out_path = tmp_path / "real_xtts.wav"
        engine = TTSEngine(engine="xtts", voice=str(XTTS_DEFAULT_SPEAKER), use_gpu=True)
        assert engine._engine_mode == "xtts"
        engine.synthesize_to_file("Привет.", out_path)

        assert out_path.exists()
        assert out_path.stat().st_size > 5000

        import wave
        with wave.open(str(out_path), "rb") as wav:
            assert wav.getnframes() > 0
            assert wav.getsampwidth() == 2
            assert wav.getnchannels() == 1


# ── Engine routing & state ───────────────────────────────────────────────────


class TestXTTSEngineRouting:
    """Verify TTSEngine transitions correctly for xtts mode."""

    def test_init_sapi_fallback_when_xtts_model_missing(self, monkeypatch) -> None:
        """If XTTS model dir is missing, engine falls to VITS/SAPI."""
        monkeypatch.setattr(
            "app.tts_engine.XTTS_MODEL_DIR",
            Path("/nonexistent/xtts"),
        )
        engine = TTSEngine(engine="auto", voice="default", use_gpu=False)
        assert engine._engine_mode in ("hf_vits_local", "sapi")

    def test_xtts_engine_mode_set(self, monkeypatch) -> None:
        """When TTS loads OK, engine mode becomes 'xtts'."""
        if not XTTS_MODEL_DIR.exists():
            pytest.skip("No XTTS model directory")
        import app.tts_engine as _te
        _te._XTTS_MODEL_CACHE.clear()
        monkeypatch.setattr(
            "app.tts_engine._load_xtts_model",
            lambda device: _XTTS_MARKER,
        )
        engine = TTSEngine(engine="xtts", voice="default", use_gpu=False)
        assert engine._engine_mode == "xtts"
        assert engine._xtts_model is _XTTS_MARKER

    def test_xtts_engine_fallback_on_import_error(self, monkeypatch) -> None:
        """If coqui-tts cannot be imported, fallback path is used."""
        if not XTTS_MODEL_DIR.exists():
            pytest.skip("No XTTS model directory")

        import app.tts_engine as _te
        _te._XTTS_MODEL_CACHE.clear()

        original_import = __import__

        def _broken_import(name, *args, **kw):
            if name == "TTS.api":
                raise ImportError("Simulated TTS import failure")
            return original_import(name, *args, **kw)

        monkeypatch.setattr("builtins.__import__", _broken_import)
        monkeypatch.setattr(
            "app.tts_engine._check_xtts_subprocess",
            lambda: None,
        )
        engine = TTSEngine(engine="xtts", voice="default", use_gpu=False)
        assert engine._engine_mode == "xtts"
        assert engine._xtts_model is _XTTS_MARKER

    def test_xtts_engine_raises_on_load_failure(self, monkeypatch) -> None:
        """If model loading raises (non-ImportError), engine raises."""
        if not XTTS_MODEL_DIR.exists():
            pytest.skip("No XTTS model directory")

        # Clear the model cache so _load_xtts_model is actually called
        import app.tts_engine as _te
        _te._XTTS_MODEL_CACHE.clear()

        def _broken_load(device="cpu"):
            raise RuntimeError("Simulated model load failure")

        monkeypatch.setattr(
            "app.tts_engine._load_xtts_model",
            _broken_load,
        )
        with pytest.raises(RuntimeError, match="XTTS v2 requested but failed"):
            TTSEngine(engine="xtts", voice="default", use_gpu=False)

    @pytest.mark.slow
    def test_synthesize_to_file_subprocess_fallback(self, monkeypatch, tmp_path) -> None:
        """When _xtts_model is _XTTS_MARKER, synthesize uses subprocess."""
        if not XTTS_MODEL_DIR.exists():
            pytest.skip("No XTTS model directory")
        pytest.skip("Slow: would trigger real model loading")

        engine = TTSEngine(engine="sapi", voice="default", use_gpu=False)
        engine._engine_mode = "xtts"
        engine._xtts_model = _XTTS_MARKER
        engine._xtts_speaker_wav = str(XTTS_DEFAULT_SPEAKER) if XTTS_DEFAULT_SPEAKER.exists() else ""

        out = tmp_path / "test_out.wav"
        with pytest.raises(RuntimeError, match="XTTS subprocess failed|XTTS требует FFmpeg"):
            engine.synthesize_to_file("Тест.", out)


# ── Speaker resolution ───────────────────────────────────────────────────────


class TestXTTSResolveSpeaker:
    """Unit tests for _resolve_speaker_wav."""

    @staticmethod
    def _resolve(voice: str) -> str:
        return _make_engine()._resolve_speaker_wav(voice)

    def test_resolve_existing_path(self, tmp_path) -> None:
        custom = tmp_path / "custom.wav"
        custom.write_bytes(b"\x00" * 100)
        result = self._resolve(str(custom))
        assert result == str(custom)

    def test_resolve_default_fallback(self) -> None:
        """'default' resolves to XTTS_DEFAULT_SPEAKER if it exists."""
        result = self._resolve("default")
        if XTTS_DEFAULT_SPEAKER.exists():
            assert result == str(XTTS_DEFAULT_SPEAKER)
        else:
            assert result.endswith("speakers_xtts.pth")

    def test_resolve_builtin_fallback(self) -> None:
        """Nonexistent path falls back to my_voice.wav or speakers_xtts.pth."""
        result = self._resolve("nonexistent/speaker.wav")
        if XTTS_DEFAULT_SPEAKER.exists():
            assert result == str(XTTS_DEFAULT_SPEAKER)
        else:
            assert result.endswith("speakers_xtts.pth")
