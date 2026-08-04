"""
Transcription runs fully local via faster-whisper.
"""
import numpy as np
from faster_whisper import WhisperModel

import config

_model = None


def get_model():
    global _model
    if _model is None:
        _model = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")
    return _model


def transcribe(pcm_bytes: bytes) -> str:
    if not pcm_bytes:
        return ""
    model = get_model()
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    segments, _ = model.transcribe(
        audio,
        language="en",
        beam_size=2,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=300),
    )
    return " ".join(seg.text.strip() for seg in segments).strip()