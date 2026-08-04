"""
True streaming wake-word detection using openWakeWord's pretrained
"hey jarvis" model, layered with speaker verification.

This replaces the old approach (record a full utterance -> wait for you to
go silent -> transcribe the whole clip with Whisper -> fuzzy-match the
text). That round-trip is where most of the "lag before it wakes up" was
coming from. openWakeWord instead scores small (80ms) audio frames
continuously as they stream in from the mic, so detection fires the moment
the phrase is heard.

Flow per listen_once() call:
  1. Open the mic and read 80ms frames in a loop.
  2. Feed each frame to openWakeWord. It returns a 0-1 score for how
     confident it is that "hey jarvis" was just heard.
  3. The moment the score crosses the threshold, the wake word has fired.
  4. openWakeWord only tells us THAT the phrase was heard, not whose voice
     it was — so we keep a rolling ~1.5s buffer of raw audio, and the
     moment we detect a wake, we run that buffered clip through the
     Resemblyzer voice verifier to confirm it's actually you.
  5. If no voiceprint has been enrolled yet, voice verification is skipped
     (with a warning) instead of crashing the whole program.
"""
import collections

import numpy as np
import pyaudio

import config
from voice_id.verify import is_my_voice, voiceprint_available

FRAME_SAMPLES = 1280  # 80ms @ 16kHz — openWakeWord's native frame size


class WakeWordListener:
    def __init__(self):
        self.pa = pyaudio.PyAudio()
        self._model = None
        self._voice_id_enabled = True

    def start(self):
        import openwakeword
        from openwakeword.model import Model

        print("[wake_word] loading openWakeWord model (downloads once on first run)...")
        openwakeword.utils.download_models([config.WAKE_WORD_MODEL])
        # inference_framework="onnx" explicitly — tflite wheels aren't
        # reliably available on Apple Silicon, onnx is.
        self._model = Model(
            wakeword_models=[config.WAKE_WORD_MODEL],
            inference_framework="onnx",
        )

        self._voice_id_enabled = voiceprint_available()
        if not self._voice_id_enabled:
            print(
                "\n[wake_word] WARNING: no enrolled voiceprint found — anyone's "
                "voice will trigger the wake word right now.\n"
                "  Run: python3 voice_id/record.py   then   python3 voice_id/enroll.py\n"
                "  Restart main.py afterward to enable voice-only wake.\n"
            )

        suffix = " (your voice only)" if self._voice_id_enabled else " (voice check disabled)"
        print(f'[wake_word] listening for "hey jarvis"{suffix}')

    def listen_once(self) -> str:
        """Blocks until the wake word is heard (and, if enrolled, verified
        as your voice). Opens its own mic stream and closes it before
        returning, so it doesn't fight with the conversation listener for
        the input device."""
        buffer_frames = int(config.SAMPLE_RATE * config.VOICE_ID_CLIP_SECONDS / FRAME_SAMPLES) + 1
        ring = collections.deque(maxlen=buffer_frames)

        stream = self.pa.open(
            rate=config.SAMPLE_RATE,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=FRAME_SAMPLES,
        )
        try:
            while True:
                raw = stream.read(FRAME_SAMPLES, exception_on_overflow=False)
                ring.append(raw)
                frame = np.frombuffer(raw, dtype=np.int16)

                prediction = self._model.predict(frame)
                score = prediction.get(config.WAKE_WORD_MODEL, 0.0)

                if score < config.WAKE_WORD_THRESHOLD:
                    continue

                if self._voice_id_enabled:
                    clip = b"".join(ring)
                    if not is_my_voice(clip):
                        print('[wake_word] heard "hey jarvis" but voice didn\'t match — ignoring')
                        continue

                return "hey jarvis"
        finally:
            stream.close()

    def close(self):
        self.pa.terminate()
