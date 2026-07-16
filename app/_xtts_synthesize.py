"""
XTTS v2 synthesis helper — called via subprocess from tts_engine.py.

Usage:
    python _xtts_synthesize.py --check
    python _xtts_synthesize.py <model_dir> <speaker_wav> <output_path> <text>

Requires:
    pip install coqui-tts torch torchaudio

Note: coqui-tts has compatibility issues with Python 3.14+ and torch 2.11+.
If you encounter import errors, try creating a dedicated virtual environment
with Python 3.11 or 3.12 and installing dependencies there:
    python3.11 -m venv xtts_venv
    xtts_venv\\Scripts\\pip install coqui-tts torch torchaudio
Then set XTTS_PYTHON env var to xtts_venv\\Scripts\\python.exe
"""
from __future__ import annotations

import sys
from pathlib import Path

# ── FFmpeg DLL search path (needed by torchcodec on Windows) ────────────────
import os as _os
_ffmpeg_dll_dir = _os.path.join(_os.environ.get("USERPROFILE", ""), ".ffmpeg-dlls")
if _os.path.isdir(_ffmpeg_dll_dir):
    _os.environ["PATH"] = _ffmpeg_dll_dir + _os.pathsep + _os.environ.get("PATH", "")
    try:
        _os.add_dll_directory(_ffmpeg_dll_dir)
    except Exception:
        pass

# ── Compatibility shims ─────────────────────────────────────────────────────
try:
    import torch
except Exception:
    torch = None

# transformers 5.x removed isin_mps_friendly
try:
    import transformers.pytorch_utils as _tpu
    if not hasattr(_tpu, "isin_mps_friendly") and torch is not None:
        _tpu.isin_mps_friendly = lambda elements, test_elements, **kw: torch.isin(elements, test_elements)
except Exception:
    pass

# librosa ≥ 0.11 removed magphase, pyin
try:
    import librosa
    import numpy as _np
    if not hasattr(librosa, "magphase"):
        librosa.magphase = lambda D, power=1.0: (_np.abs(D) ** power, _np.exp(1j * _np.angle(D)))
    if not hasattr(librosa, "pyin"):
        librosa.pyin = lambda *a, **kw: (None, None, None)
except Exception:
    pass

# torchcodec may not be available on Windows — patch torchaudio to use soundfile
try:
    import soundfile as _sf
    import torchaudio as _ta

    def _patched_load(audiopath: str, **kwargs) -> tuple:
        """Bypass torchcodec — use soundfile for audio loading."""
        out_frames = kwargs.pop("out_frames", None)
        # always_2d=True → (samples, channels), transpose to (channels, samples)
        data, sr = _sf.read(audiopath, always_2d=True, dtype="float32")
        import numpy as _np2
        if data.ndim == 2:
            data = data.T
        else:
            data = data.reshape(1, -1)
        import torch
        tensor = torch.from_numpy(data)
        if out_frames is not None:
            tensor = tensor[:, :out_frames]
        return tensor, sr

    _ta.load = _patched_load
except Exception:
    pass

# torchcodec may not be available on Windows without FFmpeg
try:
    from transformers.utils.import_utils import is_torchcodec_available
    if not is_torchcodec_available():
        from transformers.utils.import_utils import is_torch_greater_or_equal as _orig_ver
        def _patched_ver(v, **kw):
            return False if v.startswith("2.9") else _orig_ver(v, **kw)
        from transformers.utils import import_utils as _tui
        _tui.is_torch_greater_or_equal = _patched_ver
except Exception:
    pass
# ─────────────────────────────────────────────────────────────────────────────


def check() -> None:
    """Verify that TTS can be imported."""
    try:
        from TTS.api import TTS  # noqa: F401
        print("OK: TTS is available")
    except ImportError as e:
        print(f"ERROR: TTS import failed: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


def synthesize(
    model_dir: str,
    speaker_wav: str,
    output_path: str,
    text: str,
    *,
    temperature: float = 0.65,
    top_k: int = 30,
    top_p: float = 0.80,
    repetition_penalty: float = 20.0,
    speed: float = 1.0,
) -> None:
    """Run XTTS synthesis with configurable generation parameters."""
    from TTS.api import TTS  # type: ignore

    tts = TTS(model_path=model_dir, config_path=str(Path(model_dir) / "config.json"))
    tts.tts_to_file(
        text=text,
        speaker_wav=speaker_wav,
        language="ru",
        file_path=output_path,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        speed=speed,
    )


def main() -> None:
    if len(sys.argv) == 2 and sys.argv[1] == "--check":
        check()
        return

    if len(sys.argv) < 5:
        print(
            "Usage: python _xtts_synthesize.py <model_dir> <speaker_wav> <output_path> <text> "
            "[temperature] [top_k] [top_p] [repetition_penalty] [speed]",
            file=sys.stderr,
        )
        sys.exit(1)

    model_dir = sys.argv[1]
    speaker_wav = sys.argv[2]
    output_path = sys.argv[3]
    text = sys.argv[4]

    # Optional generation parameters (passed from tts_engine.py for subprocess path)
    kwargs: dict = {}
    if len(sys.argv) > 5:
        kwargs["temperature"] = float(sys.argv[5])
    if len(sys.argv) > 6:
        kwargs["top_k"] = int(sys.argv[6])
    if len(sys.argv) > 7:
        kwargs["top_p"] = float(sys.argv[7])
    if len(sys.argv) > 8:
        kwargs["repetition_penalty"] = float(sys.argv[8])
    if len(sys.argv) > 9:
        kwargs["speed"] = float(sys.argv[9])

    # Validate args
    if not Path(model_dir).exists():
        print(f"ERROR: model_dir not found: {model_dir}", file=sys.stderr)
        sys.exit(1)
    if not Path(speaker_wav).exists():
        print(f"ERROR: speaker_wav not found: {speaker_wav}", file=sys.stderr)
        sys.exit(1)

    synthesize(model_dir, speaker_wav, output_path, text, **kwargs)


if __name__ == "__main__":
    main()
