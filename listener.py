"""
Records audio and uses VAD to detect when the person has actually finished
speaking. Also discards short noise blips that aren't real speech.
"""
import collections
import webrtcvad
import pyaudio

import config


class UtteranceListener:
    def __init__(self):
        self.vad = webrtcvad.Vad(config.VAD_AGGRESSIVENESS)
        self.pa = pyaudio.PyAudio()

    def record_utterance(self, max_seconds: float = None) -> bytes:
        """
        Records from the mic until the person pauses. Returns raw PCM16
        audio, or b"" if what was heard didn't amount to real speech (too
        short / just noise, or nothing said within max_seconds) — callers
        should treat empty bytes as "nothing was said."

        max_seconds overrides config.MAX_UTTERANCE_SECONDS — use a shorter
        value for follow-up turns so silence times out on the window you
        actually configured instead of the (longer) default cap.
        """
        stream = self.pa.open(
            rate=config.SAMPLE_RATE,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=int(config.SAMPLE_RATE * config.FRAME_DURATION_MS / 1000),
        )

        silence_frames_needed = config.SILENCE_DURATION_MS // config.FRAME_DURATION_MS
        effective_max_seconds = max_seconds if max_seconds is not None else config.MAX_UTTERANCE_SECONDS
        max_frames = int(effective_max_seconds * 1000 / config.FRAME_DURATION_MS)

        ring = collections.deque(maxlen=silence_frames_needed)
        audio_frames = []
        speech_frame_count = 0
        heard_speech = False
        frame_count = 0

        try:
            while frame_count < max_frames:
                frame = stream.read(
                    int(config.SAMPLE_RATE * config.FRAME_DURATION_MS / 1000),
                    exception_on_overflow=False,
                )
                frame_count += 1
                is_speech = self.vad.is_speech(frame, config.SAMPLE_RATE)
                audio_frames.append(frame)
                ring.append(is_speech)

                if is_speech:
                    heard_speech = True
                    speech_frame_count += 1

                if heard_speech and len(ring) == ring.maxlen and not any(ring):
                    break
        finally:
            stream.close()

        # Not enough actual speech detected — likely just background noise.
        # Discard rather than send a near-empty/garbage clip to Whisper.
        if speech_frame_count < config.MIN_SPEECH_FRAMES:
            return b""

        return b"".join(audio_frames)